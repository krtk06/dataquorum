"""SQL executor: the sqlglot guard, read-only connections, limits, and the SQL agent."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from engine_fakes import ScriptedLLM

from dataquorum.agents import SQL
from dataquorum.core.engine import Engine, GeneratedSQL, RecommendedSteps
from dataquorum.core.state import DataFrameOutput
from dataquorum.core.statuses import Status
from dataquorum.sandbox.sql_executor import SqlExecutor, check_query, table_names

CUSTOMERS = pd.DataFrame({"id": [1, 2, 3], "region": ["north", "south", "north"]})
ORDERS = pd.DataFrame({"customer_id": [1, 1, 2, 3], "amount": [10.0, 5.0, 7.5, 2.5]})


@pytest.fixture(params=["sqlite", "duckdb"])
def db(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    if request.param == "sqlite":
        path = tmp_path / "shop.db"
        with sqlite3.connect(path) as lite:
            CUSTOMERS.to_sql("customers", lite, index=False)
            ORDERS.to_sql("orders", lite, index=False)
        return path
    path = tmp_path / "shop.duckdb"
    con = duckdb.connect(str(path))
    con.register("c", CUSTOMERS)
    con.register("o", ORDERS)
    con.execute("CREATE TABLE customers AS SELECT * FROM c")
    con.execute("CREATE TABLE orders AS SELECT * FROM o")
    con.close()
    return path


def dialect(db: Path) -> str:
    return "duckdb" if db.suffix == ".duckdb" else "sqlite"


GOOD = [
    "SELECT * FROM customers",
    "SELECT region, SUM(amount) AS total FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY region",
    "WITH t AS (SELECT customer_id, SUM(amount) s FROM orders GROUP BY 1) SELECT * FROM t ORDER BY s DESC",
    "SELECT id FROM customers UNION SELECT customer_id FROM orders",
]  # fmt: skip

ATTACKS = [
    "DELETE FROM orders",
    "UPDATE orders SET amount = 0",
    "INSERT INTO orders VALUES (9, 1.0)",
    "DROP TABLE orders",
    "CREATE TABLE x AS SELECT 1",
    "SELECT 1; DROP TABLE orders",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'other.db' AS other",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('secrets.parquet')",
    "SELECT load_extension('evil')",
    "SELECT * FROM other.orders",
    "SELECT * FROM no_such_table",
    "SELECT * INTO backup FROM orders",
    "COPY orders TO 'out.csv'",
    "SET threads = 1",
]  # fmt: skip


@pytest.mark.parametrize("sql", GOOD)
def test_read_only_queries_are_allowed(sql: str, db: Path) -> None:
    assert check_query(sql, dialect(db), table_names(db)) is None


@pytest.mark.parametrize("sql", ATTACKS)
def test_writes_files_and_unknown_tables_are_rejected(sql: str, db: Path) -> None:
    assert check_query(sql, dialect(db), table_names(db)) is not None


def test_executor_runs_a_query(db: Path, tmp_path: Path) -> None:
    result = SqlExecutor().run(GOOD[1], db, tmp_path / "w")
    assert result.ok, result.error
    assert result.output_path is not None
    out = pd.read_parquet(result.output_path).sort_values("region")
    assert out["total"].tolist() == [17.5, 7.5]


def test_rejected_query_is_an_error_result_and_data_is_untouched(db: Path, tmp_path: Path) -> None:
    result = SqlExecutor().run("DELETE FROM orders", db, tmp_path / "w")
    assert not result.ok and result.error is not None and "rejected" in result.error
    assert SqlExecutor().run("SELECT COUNT(*) AS n FROM orders", db, tmp_path / "w2").ok


def test_connection_is_read_only_even_without_the_guard(db: Path) -> None:
    executor = SqlExecutor()
    run = executor._run_duckdb if db.suffix == ".duckdb" else executor._run_sqlite
    with pytest.raises(Exception, match=r"(?i)read.?only|readonly|query_only"):
        run("DELETE FROM orders", db)


def test_runaway_query_times_out(db: Path, tmp_path: Path) -> None:
    forever = (
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n"
    )
    result = SqlExecutor(timeout_seconds=1).run(forever, db, tmp_path / "w")
    assert not result.ok and result.error is not None


def test_sql_agent_end_to_end(db: Path, tmp_path: Path) -> None:
    llm = ScriptedLLM(
        RecommendedSteps(steps=["join orders to customers", "sum amount per region"]),
        GeneratedSQL(sql="DELETE FROM orders"),  # rejected, fed back, then fixed
        GeneratedSQL(sql=GOOD[1]),
    )
    engine = Engine(SQL, llm, SqlExecutor(), runs_dir=tmp_path / "runs")
    result = engine.start(db, "Total order amount per region").result
    assert result is not None
    assert result.status is Status.SUCCESS, result.error
    assert result.usage.retries == 1
    assert "Query rejected" in llm.prompts_for("GeneratedSQL")[1]
    summary = llm.prompts_for("RecommendedSteps")[0]
    assert "table customers" in summary and "table orders" in summary
    assert isinstance(result.output, DataFrameOutput)
    assert sorted(result.output.load()["total"]) == [7.5, 17.5]


def test_sql_agent_rejects_dataframe_input(tmp_path: Path) -> None:
    engine = Engine(SQL, ScriptedLLM(), SqlExecutor(), runs_dir=tmp_path)
    with pytest.raises(ValueError, match="reads database input"):
        engine.start(CUSTOMERS, "anything")
