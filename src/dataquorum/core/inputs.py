"""What an agent reads: a DataFrame, a folder of files, or a database file.

- "dataframe": the caller's DataFrame, saved as Parquet; generated code gets `df`.
- "files":     a folder the caller points at; mounted read-only in the sandbox at /io/data;
               generated code defines `load(data_dir: str) -> pd.DataFrame`.
- "database":  a SQLite or DuckDB file; the SQL agent's query runs on a read-only connection.

Each kind knows how to describe itself for the prompt (`describe`) and what "before" means
for checks that compare input and output (`before_frame`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pandas as pd

from dataquorum.core.prompts import summarize_frame
from dataquorum.sandbox.executor import InputKind

DATABASE_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".duckdb")
TEXT_SUFFIXES = (".csv", ".tsv", ".txt", ".json", ".jsonl")


def prepare(data: pd.DataFrame | str | Path, workdir: Path) -> tuple[InputKind, Path]:
    """Store or locate the input for a run."""
    if isinstance(data, pd.DataFrame):
        path = workdir / "input.parquet"
        data.to_parquet(path, index=False)
        return "dataframe", path
    path = Path(data).resolve()
    if path.is_dir():
        return "files", path
    if path.is_file() and path.suffix.lower() in DATABASE_SUFFIXES:
        return "database", path
    raise ValueError(f"Input must be a DataFrame, a folder, or a database file: {path}")


def describe(kind: InputKind, path: Path) -> str:
    if kind == "dataframe":
        return summarize_frame(pd.read_parquet(path))
    if kind == "files":
        return describe_folder(path)
    return describe_database(path)


def before_frame(kind: InputKind, path: Path) -> pd.DataFrame:
    """Input as a frame for checks that compare before/after; empty for files/databases."""
    return pd.read_parquet(path) if kind == "dataframe" else pd.DataFrame()


def describe_folder(folder: Path, max_files: int = 30, preview_lines: int = 5) -> str:
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    lines = [f"Folder with {len(files)} files (paths relative to data_dir):"]
    for p in files[:max_files]:
        rel = p.relative_to(folder).as_posix()
        lines.append(f"- {rel} ({p.stat().st_size:,} bytes)")
        if p.suffix.lower() in TEXT_SUFFIXES:
            with p.open(encoding="utf-8", errors="replace") as f:
                head = [next(f, "").rstrip("\n")[:200] for _ in range(preview_lines)]
            lines += [f"    | {h}" for h in head if h]
    if len(files) > max_files:
        lines.append(f"... and {len(files) - max_files} more")
    return "\n".join(lines)


def describe_database(path: Path, sample_rows: int = 3) -> str:
    """Tables, columns with types, row counts and a few rows, via a read-only connection."""
    lines = [f"Database {path.name}:"]
    if path.suffix.lower() == ".duckdb":
        con = duckdb.connect(str(path), read_only=True)
        try:
            for (table,) in con.execute("SHOW TABLES").fetchall():
                cols = con.execute(f'DESCRIBE "{table}"').fetchall()
                count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
                sample = con.execute(f'SELECT * FROM "{table}" LIMIT {sample_rows}').fetchall()
                lines += _table_lines(table, [(c[0], c[1]) for c in cols], count, sample)
        finally:
            con.close()
        return "\n".join(lines)
    lite = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        tables = lite.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (table,) in tables:
            info = lite.execute(f'PRAGMA table_info("{table}")').fetchall()
            total = lite.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
            rows = lite.execute(f'SELECT * FROM "{table}" LIMIT {sample_rows}').fetchall()
            lines += _table_lines(table, [(c[1], c[2]) for c in info], total, rows)
    finally:
        lite.close()  # `with` on sqlite3 only commits; it does not close
    return "\n".join(lines)


def _table_lines(
    table: str,
    columns: list[tuple[str, str]],
    count: tuple[int, ...] | None,
    sample: list[tuple[object, ...]],
) -> list[str]:
    out = [f"- table {table} ({count[0] if count else '?'} rows)"]
    out += [f"    {name}: {kind}" for name, kind in columns]
    out += [f"    sample: {row}" for row in sample]
    return out
