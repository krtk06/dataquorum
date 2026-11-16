"""Replay real runs of the week-4 agents (DeepSeek V4.1 Flash via Go), recorded once.

Recording (DQ_RECORD=1) uses the real LLM and the Docker sandbox image
`dataquorum/sandbox:0.1`. Replay feeds back the recorded LLM replies and runs the recorded,
reviewed code in-process; it never runs freshly generated code.

    DQ_RECORD=1 uv run pytest tests/test_agents_replay.py     # needs .env key + Docker
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx2
import pandas as pd
from engine_fakes import InProcessExecutor
from pydantic import SecretStr
from replay import RecordingTransport, ReplayTransport

from dataquorum.agents import AGENTS
from dataquorum.core.engine import Engine
from dataquorum.core.llm import LLMClient, LLMConfig
from dataquorum.core.state import AgentResult, DataFrameOutput, FigureOutput
from dataquorum.core.statuses import Status
from dataquorum.sandbox.docker_executor import DockerExecutor
from dataquorum.sandbox.executor import Executor

CASSETTES = Path(__file__).parent / "cassettes"
SALES = pd.DataFrame(
    {
        "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "region": ["North", "South", "North", "East", "South", "East", "North", "West"],
        "revenue": [120.0, 80.5, 99.9, 45.0, 60.0, 30.25, 75.0, 52.0],
        "units": [3, 2, 2, 1, 2, 1, 3, 1],
    }
)


def run_agent(name: str, request: str, cassette: str, tmp_path: Path) -> AgentResult:
    transport: RecordingTransport | ReplayTransport
    executor: Executor
    if os.getenv("DQ_RECORD"):
        transport, config = RecordingTransport(CASSETTES / cassette), LLMConfig.from_env()
        executor = DockerExecutor()
    else:
        transport = ReplayTransport(CASSETTES / cassette)
        config = LLMConfig(api_key=SecretStr("replay"))
        executor = InProcessExecutor()
    llm = LLMClient(config, http_client=httpx2.Client(transport=transport))
    engine = Engine(AGENTS[name], llm, executor, runs_dir=tmp_path, max_retries=1)
    try:
        result = engine.start(SALES, request).result
    finally:
        if isinstance(transport, RecordingTransport):
            transport.save()
    assert result is not None
    return result


def test_wrangling_aggregates(tmp_path: Path) -> None:
    result = run_agent(
        "data_wrangling",
        "Total revenue and total units per region, sorted by revenue, highest first.",
        "wrangling_totals.json",
        tmp_path,
    )
    assert result.status is Status.SUCCESS, result.error
    assert isinstance(result.output, DataFrameOutput)
    out = result.output.load()
    assert out.iloc[0, 0] == "North" and len(out) == 4
    assert out.select_dtypes("number").sum().round(2).tolist() == [562.65, 15.0]


def test_wrangling_does_not_invent_columns(tmp_path: Path) -> None:
    """Same request that made the original return an empty table with invented columns.

    Recorded 2026-11-16: recommend flagged the missing column, both code attempts raised
    ValueError naming it, and the run ended `failed` (3 LLM calls).
    """
    result = run_agent(
        "data_wrangling",
        "Show total revenue by sales region and product category as a table.",
        "wrangling_missing_columns.json",
        tmp_path,
    )
    assert result.status is Status.FAILED
    assert result.output is None
    assert result.error is not None and "product_category" in result.error
    assert "no product category column" in result.steps[0]


def test_visualization_bar_chart(tmp_path: Path) -> None:
    result = run_agent(
        "data_visualization",
        "Bar chart of total revenue by region.",
        "visualization_bar.json",
        tmp_path,
    )
    assert result.status is Status.SUCCESS, result.error
    assert isinstance(result.output, FigureOutput)
    traces = result.output.figure["data"]
    assert traces and traces[0]["type"] == "bar"
