from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from engine_fakes import InProcessExecutor, ScriptedLLM

from dataquorum.agents import CLEANING
from dataquorum.core.budget import BudgetExceededError, BudgetGuard, BudgetLimits, SpendEntry
from dataquorum.core.cost import Usage
from dataquorum.core.engine import Engine, GeneratedCode, RecommendedSteps
from dataquorum.core.state import AgentResult, DataFrameOutput, FigureOutput
from dataquorum.core.statuses import CheckResult, Severity
from dataquorum.tracing.recorder import DuckDBLedger, RunRecorder


@pytest.fixture
def recorder(tmp_path: Path) -> RunRecorder:
    return RunRecorder(tmp_path / "trace.duckdb")


def test_records_success_and_failure(recorder: RunRecorder, tmp_path: Path) -> None:
    table = DataFrameOutput.save(pd.DataFrame({"a": [1]}), tmp_path / "t.parquet")
    warning = CheckResult(name="row_loss", passed=False, severity=Severity.WARNING, detail="x")
    usage = Usage(models=["m"], calls=2, retries=1, completion_tokens=50, allowance_usd=0.002)
    ok = AgentResult.from_run(agent="cleaning", output=table, checks=[warning], usage=usage)
    bad = AgentResult.from_run(agent="wrangling", output=None, error="KeyError: 'x'")
    recorder.record(ok)
    recorder.record(bad, eval_task_id="task-7")

    runs = recorder.runs().set_index("run_id")
    assert len(runs) == 2
    row = runs.loc[ok.run_id]
    assert row["status"] == "needs_review"
    assert row["output_kind"] == "dataframe" and row["output_ref"].endswith("t.parquet")
    assert list(row["failed_checks"]) == ["row_loss"]
    assert (row["calls"], row["retries"]) == (2, 1)
    assert row["allowance_usd"] == pytest.approx(0.002)
    failed = runs.loc[bad.run_id]
    assert failed["status"] == "failed" and failed["eval_task_id"] == "task-7"
    assert failed["error"] == "KeyError: 'x'"


def test_recording_same_run_twice_replaces_row(recorder: RunRecorder) -> None:
    fig = FigureOutput(figure={"data": [{"type": "bar", "x": [1], "y": [2]}]})
    result = AgentResult.from_run(agent="viz", output=fig)
    recorder.record(result)
    recorder.record(result.model_copy(update={"message": "updated"}))
    runs = recorder.runs()
    assert len(runs) == 1 and runs.loc[0, "message"] == "updated"
    assert runs.loc[0, "output_kind"] == "figure"


def test_duckdb_ledger_drives_budget_guard(recorder: RunRecorder) -> None:
    ledger = DuckDBLedger(recorder)
    now = datetime.now(UTC)
    ledger.append(SpendEntry(now - timedelta(days=8), 5.0, "m"))  # outside both windows
    ledger.append(SpendEntry(now - timedelta(hours=6), 0.4, "m"))  # week only
    ledger.append(SpendEntry(now - timedelta(minutes=5), 0.7, "m"))
    guard = BudgetGuard(ledger, mode="eval", limits=BudgetLimits(1.0, 2.0))
    five_h, week = guard.spent()
    assert five_h == pytest.approx(0.7) and week == pytest.approx(1.1)
    guard.check()
    ledger.append(SpendEntry(now, 0.3, "m"))
    with pytest.raises(BudgetExceededError):
        guard.check()


def test_engine_records_finished_runs(recorder: RunRecorder, tmp_path: Path) -> None:
    llm = ScriptedLLM(
        RecommendedSteps(steps=["strip city"]),
        GeneratedCode(
            python_code="def transform(df):\n    return df.assign(city=df['city'].str.strip())\n"
        ),
    )
    engine = Engine(CLEANING, llm, InProcessExecutor(), runs_dir=tmp_path, recorder=recorder)
    run = engine.start(pd.DataFrame({"city": [" a", "b "]}), "strip city")
    runs = recorder.runs()
    assert runs["run_id"].tolist() == [run.run_id]
    assert runs.loc[0, "agent"] == "data_cleaning" and runs.loc[0, "status"] == "success"


def test_old_database_is_migrated(tmp_path: Path) -> None:
    import duckdb

    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))  # week-4 schema: no execution_seconds column
    con.execute("CREATE TABLE runs (run_id VARCHAR PRIMARY KEY, parent_run_id VARCHAR, "
                "created_at TIMESTAMPTZ, agent VARCHAR, status VARCHAR, output_kind VARCHAR, "
                "output_ref VARCHAR, models VARCHAR[], calls INTEGER, retries INTEGER, "
                "prompt_tokens BIGINT, completion_tokens BIGINT, reasoning_tokens BIGINT, "
                "cached_tokens BIGINT, list_cost_usd DOUBLE, allowance_usd DOUBLE, seconds DOUBLE, "
                "failed_checks VARCHAR[], checks JSON, error VARCHAR, message VARCHAR, "
                "code VARCHAR, git_commit VARCHAR, eval_task_id VARCHAR)")  # fmt: skip
    con.close()
    recorder = RunRecorder(path)
    usage = Usage(models=["m"], calls=1, seconds=2.0, execution_seconds=1.5)
    result = AgentResult.from_run(agent="a", output=None, error="x", usage=usage)
    recorder.record(result)
    row = recorder.runs().iloc[0]
    assert (row["seconds"], row["execution_seconds"]) == (2.0, 1.5)
