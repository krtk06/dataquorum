"""Replay a real engine run (DeepSeek V4.1 Flash via Go) recorded once.

Recording (DQ_RECORD=1) uses the real LLM and the Docker sandbox. Replay feeds the recorded
LLM replies back and runs the recorded code in-process. That code is fixed, committed and
was reviewed when recorded; replay never runs freshly generated code.

    DQ_RECORD=1 uv run pytest tests/test_engine_replay.py     # needs .env key + Docker
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx2
import pandas as pd
from engine_fakes import InProcessExecutor
from pydantic import SecretStr
from replay import RecordingTransport, ReplayTransport

from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.engine import Engine
from dataquorum.core.llm import LLMClient, LLMConfig
from dataquorum.core.state import DataFrameOutput
from dataquorum.core.statuses import Status
from dataquorum.sandbox.docker_executor import DockerExecutor
from dataquorum.sandbox.executor import Executor

CASSETTE = Path(__file__).parent / "cassettes" / "engine_cleaning_run.json"

SPEC = AgentSpec(
    name="data_cleaning",
    role="You are a careful data-cleaning agent for pandas DataFrames.",
    recommend_guidance=(
        "Recommend only the cleaning the user asked for, plus fixes needed to make the "
        "requested columns usable. Never drop rows or columns unless asked."
    ),
    code_guidance="Prefer vectorised pandas string methods over Python loops.",
)
REQUEST = (
    "Convert income to float dollars ('$3.4k' means 3400; 'n/a', 'N/A', '' and missing "
    "become NaN). Strip city, collapse repeated spaces, and title-case it. Keep every row."
)
DATA = pd.DataFrame(
    {
        "age": [34.0, None, 51.0, 29.0, None, 42.0],
        "income": ["$1,200", "$3.4k", "n/a", " $2,050 ", "", None],
        "city": ["new york", "NEW YORK", "boston ", " Chicago", "san  francisco", "seattle"],
    }
)


def test_cleaning_run_against_recorded_go_replies(tmp_path: Path) -> None:
    transport: RecordingTransport | ReplayTransport
    executor: Executor
    if os.getenv("DQ_RECORD"):
        transport, config, executor = (
            RecordingTransport(CASSETTE),
            LLMConfig.from_env(),
            DockerExecutor(),
        )
    else:
        transport = ReplayTransport(CASSETTE)
        config = LLMConfig(api_key=SecretStr("replay"))
        executor = InProcessExecutor()

    llm = LLMClient(config, http_client=httpx2.Client(transport=transport))
    run = Engine(SPEC, llm, executor, runs_dir=tmp_path).start(DATA, REQUEST)
    if isinstance(transport, RecordingTransport):
        transport.save()

    result = run.result
    assert result is not None
    assert result.status is Status.SUCCESS, result.error
    assert result.steps and result.code and "def transform" in result.code
    assert result.usage.calls >= 2 and result.usage.allowance_usd > 0
    assert isinstance(result.output, DataFrameOutput)
    out = result.output.load()
    income = out["income"]
    assert income[[0, 1, 3]].tolist() == [1200.0, 3400.0, 2050.0]
    assert income[[2, 4, 5]].isna().all()  # "n/a", "", None
    assert out["city"].tolist() == [
        "New York", "New York", "Boston", "Chicago", "San Francisco", "Seattle",
    ]  # fmt: skip
