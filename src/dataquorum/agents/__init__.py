"""Data-science agents. Each agent is an `AgentSpec`; the shared engine runs it.

from dataquorum.agents import CLEANING, make_engine

engine = make_engine(CLEANING)
run = engine.start(df, "Clean this data")
print(run.result.status, run.result.message)
"""

from __future__ import annotations

from pathlib import Path

from dataquorum.agents.cleaning import CLEANING
from dataquorum.agents.sql import SQL
from dataquorum.agents.visualization import VISUALIZATION
from dataquorum.agents.wrangling import WRANGLING
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.budget import BudgetGuard, Mode
from dataquorum.core.engine import Engine, ResultSink, StructuredLLM
from dataquorum.sandbox.executor import Executor

AGENTS: dict[str, AgentSpec] = {
    spec.name: spec for spec in (CLEANING, WRANGLING, VISUALIZATION, SQL)
}


def make_engine(
    spec: AgentSpec,
    *,
    llm: StructuredLLM | None = None,
    executor: Executor | None = None,
    recorder: ResultSink | None = None,
    runs_dir: Path = Path("runs/engine"),
    budget_mode: Mode = "interactive",
    max_retries: int = 3,
) -> Engine:
    """Engine with real defaults: Go via `.env` (with the budget guard), the Docker sandbox,
    and the DuckDB run recorder in `runs/dataquorum.duckdb`."""
    if recorder is None:
        from dataquorum.tracing.recorder import RunRecorder

        recorder = RunRecorder()
    if llm is None:
        from dataquorum.core.llm import LLMClient, LLMConfig
        from dataquorum.tracing.recorder import DuckDBLedger, RunRecorder

        budget = None
        if isinstance(recorder, RunRecorder):
            budget = BudgetGuard(DuckDBLedger(recorder), mode=budget_mode)
        llm = LLMClient(LLMConfig.from_env(), budget=budget)
    if executor is None and spec.language == "sql":
        from dataquorum.sandbox.sql_executor import SqlExecutor

        executor = SqlExecutor()
    elif executor is None:
        from dataquorum.sandbox.docker_executor import DockerExecutor

        executor = DockerExecutor()
    return Engine(
        spec, llm, executor, runs_dir=runs_dir, max_retries=max_retries, recorder=recorder
    )


__all__ = ["AGENTS", "CLEANING", "SQL", "VISUALIZATION", "WRANGLING", "make_engine"]
