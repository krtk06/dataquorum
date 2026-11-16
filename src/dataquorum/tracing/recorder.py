"""Run recorder: one DuckDB row per run, plus a spend ledger for the budget guard (spec S8).

Tables (in `runs/dataquorum.duckdb` by default, git-ignored):

- `runs`: one row per finished `AgentResult` (status, tokens, cost, retries, checks, error,
  code, git commit, optional eval task id). Re-recording a run id replaces its row.
- `llm_spend`: one row per LLM call, written by `BudgetGuard` through `DuckDBLedger`.
  This replaces the week-2 JSONL ledger.

The Studio's Runs page (weeks 15-16) and the eval stats (week 7) read these tables.
DuckDB allows one writer process per file; within a process, calls are serialised by a lock.
"""

from __future__ import annotations

import subprocess
import threading
from datetime import datetime
from functools import cache
from pathlib import Path

import duckdb
import pandas as pd

from dataquorum.core.budget import SpendEntry
from dataquorum.core.state import AgentResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    parent_run_id VARCHAR,
    created_at TIMESTAMPTZ,
    agent VARCHAR,
    status VARCHAR,
    output_kind VARCHAR,
    output_ref VARCHAR,
    models VARCHAR[],
    calls INTEGER,
    retries INTEGER,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    reasoning_tokens BIGINT,
    cached_tokens BIGINT,
    list_cost_usd DOUBLE,
    allowance_usd DOUBLE,
    seconds DOUBLE,
    failed_checks VARCHAR[],
    checks JSON,
    error VARCHAR,
    message VARCHAR,
    code VARCHAR,
    git_commit VARCHAR,
    eval_task_id VARCHAR
);
CREATE TABLE IF NOT EXISTS llm_spend (
    spent_at TIMESTAMPTZ,
    model VARCHAR,
    allowance_usd DOUBLE
);
"""


@cache
def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _output_ref(result: AgentResult) -> tuple[str | None, str | None]:
    out = result.output
    if out is None:
        return None, None
    if out.kind == "dataframe":
        return out.kind, str(out.path)
    if out.kind == "model":
        return out.kind, out.uri
    return out.kind, None  # figures and text live in the result itself


class RunRecorder:
    def __init__(self, path: Path = Path("runs/dataquorum.duckdb")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = duckdb.connect(str(path))
        self.lock = threading.Lock()
        with self.lock:
            self.con.execute(_SCHEMA)

    def record(self, result: AgentResult, *, eval_task_id: str | None = None) -> None:
        u = result.usage
        kind, ref = _output_ref(result)
        failed = [c.name for c in result.validation if not c.passed]
        checks = "[" + ",".join(c.model_dump_json() for c in result.validation) + "]"
        row = [
            result.run_id, result.parent_run_id, result.created_at, result.agent,
            str(result.status), kind, ref, u.models, u.calls, u.retries, u.prompt_tokens,
            u.completion_tokens, u.reasoning_tokens, u.cached_tokens, u.list_cost_usd,
            u.allowance_usd, u.seconds, failed, checks, result.error, result.message,
            result.code, git_commit(), eval_task_id,
        ]  # fmt: skip
        placeholders = ", ".join("?" * len(row))
        with self.lock:
            self.con.execute(f"INSERT OR REPLACE INTO runs VALUES ({placeholders})", row)

    def runs(self, where: str = "TRUE") -> pd.DataFrame:
        """Recorded runs as a DataFrame, newest first. `where` is trusted SQL (dev use)."""
        with self.lock:
            return self.con.execute(
                f"SELECT * FROM runs WHERE {where} ORDER BY created_at DESC"
            ).df()

    def close(self) -> None:
        with self.lock:
            self.con.close()


class DuckDBLedger:
    """`Ledger` for `BudgetGuard`, stored in the recorder's database."""

    def __init__(self, recorder: RunRecorder) -> None:
        self.recorder = recorder

    def append(self, entry: SpendEntry) -> None:
        with self.recorder.lock:
            self.recorder.con.execute(
                "INSERT INTO llm_spend VALUES (?, ?, ?)",
                [entry.at, entry.model, entry.allowance_usd],
            )

    def total_since(self, since: datetime) -> float:
        with self.recorder.lock:
            row = self.recorder.con.execute(
                "SELECT COALESCE(SUM(allowance_usd), 0) FROM llm_spend WHERE spent_at >= ?", [since]
            ).fetchone()
        return float(row[0]) if row else 0.0
