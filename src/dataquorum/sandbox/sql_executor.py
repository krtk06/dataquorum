"""Run one LLM-written SQL query safely against a SQLite or DuckDB file (spec S5, roadmap wk 9).

Two layers, either of which alone should stop a write:

1. `check_query` (sqlglot): exactly one statement; it must be a query (SELECT / WITH / set
   operation); no INSERT, UPDATE, DELETE, MERGE, DDL, PRAGMA, ATTACH, COPY, SET or unparsed
   commands anywhere in the tree; every table must be a real table of this database or a CTE
   defined in the query (so table functions like DuckDB's `read_csv('/etc/passwd')` are
   rejected); no file, extension or system functions.
2. The connection itself is read-only: SQLite opened with `mode=ro` plus `query_only`;
   DuckDB opened `read_only` with external access (files, network) disabled.

Queries also get a time limit and a row cap. Rejections and SQL errors come back as
`ExecutionResult(ok=False, ...)`, so the engine can ask the LLM to fix the query.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from dataquorum.sandbox.executor import OUTPUT_FILES, ExecutionResult, InputKind, OutputKind

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.Pragma, exp.Attach, exp.Detach, exp.Copy, exp.Set, exp.Command, exp.Use,
    exp.TruncateTable, exp.Into,
)  # fmt: skip
FORBIDDEN_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto", "read_text",
    "read_blob", "read_ndjson", "glob", "parquet_scan", "csv_scan", "load_extension",
    "readfile", "writefile", "edit", "fts5", "sqlite_compileoption_get", "getenv",
    "current_setting", "duckdb_settings", "duckdb_secrets",
}  # fmt: skip


def dialect_for(path: Path) -> str:
    return "duckdb" if path.suffix.lower() == ".duckdb" else "sqlite"


def table_names(path: Path) -> set[str]:
    if dialect_for(path) == "duckdb":
        con = duckdb.connect(str(path), read_only=True)
        try:
            return {str(r[0]).lower() for r in con.execute("SHOW TABLES").fetchall()}
        finally:
            con.close()
    lite = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = lite.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        return {str(r[0]).lower() for r in rows.fetchall()}
    finally:
        lite.close()


def check_query(sql: str, dialect: str, allowed_tables: set[str]) -> str | None:
    """Return why the query is not allowed, or None if it may run."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError as exc:
        return f"SQL does not parse: {str(exc).splitlines()[0]}"
    if len(statements) != 1:
        return f"Exactly one statement is allowed, got {len(statements)}."
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        return f"Only read-only queries are allowed, got {type(tree).__name__.upper()}."
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return f"{type(node).__name__.upper()} is not allowed in a read-only query."
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS:
                return f"Function {name}() is not allowed."
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return f"Table functions are not allowed: {table.sql(dialect=dialect)}"
        name = table.name.lower()
        if table.db or name not in allowed_tables | ctes:
            return f"Unknown table {table.sql(dialect=dialect)}; allowed: {sorted(allowed_tables)}."
    return None


class SqlExecutor:
    def __init__(self, timeout_seconds: float = 30, max_rows: int = 200_000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    def run(
        self,
        code: str,
        input_path: Path,
        workdir: Path,
        kind: OutputKind = "dataframe",
        input_kind: InputKind = "database",
    ) -> ExecutionResult:
        start = time.perf_counter()
        if input_kind != "database" or kind != "dataframe":
            raise ValueError("SqlExecutor runs queries on databases and returns tables only")
        dialect = dialect_for(input_path)
        problem = check_query(code, dialect, table_names(input_path))
        if problem:
            return ExecutionResult(
                False, f"Query rejected: {problem}", None, time.perf_counter() - start
            )
        try:
            frame = (
                self._run_duckdb(code, input_path)
                if dialect == "duckdb"
                else self._run_sqlite(code, input_path)
            )
        except Exception as exc:  # SQL errors are the query's fault, not the sandbox's
            return ExecutionResult(
                False, f"{type(exc).__name__}: {exc}"[:1000], None, time.perf_counter() - start
            )
        if len(frame) > self.max_rows:
            return ExecutionResult(
                False,
                f"Query returned more than {self.max_rows} rows.",
                None,
                time.perf_counter() - start,
            )
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / OUTPUT_FILES["dataframe"]
        frame.to_parquet(out, index=False)
        return ExecutionResult(True, None, out, time.perf_counter() - start)

    def _run_sqlite(self, sql: str, path: Path) -> pd.DataFrame:
        lite = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        deadline = time.monotonic() + self.timeout_seconds
        lite.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            lite.execute("PRAGMA query_only = ON")
            cursor = lite.execute(sql)
            rows = cursor.fetchmany(self.max_rows + 1)
            columns = [d[0] for d in cursor.description or []]
            return pd.DataFrame.from_records(rows, columns=columns)
        finally:
            lite.close()

    def _run_duckdb(self, sql: str, path: Path) -> pd.DataFrame:
        con = duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})
        timer = threading.Timer(self.timeout_seconds, con.interrupt)
        timer.start()
        try:
            return con.execute(sql).fetch_df().head(self.max_rows + 1)
        finally:
            timer.cancel()
            con.close()
