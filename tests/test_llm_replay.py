"""Replay a real OpenCode Go call recorded once from DeepSeek V4.1 Flash.

CI replays the cassette (no network, no key). To re-record against the live API:

    DQ_RECORD=1 uv run pytest tests/test_llm_replay.py

Recording costs well under a cent and needs OPENCODE_GO_API_KEY in .env.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import httpx2
from pydantic import BaseModel, SecretStr
from replay import RecordingTransport, ReplayTransport

from dataquorum.core.llm import LLMClient, LLMConfig

CASSETTE = Path(__file__).parent / "cassettes" / "structured_cleaning_plan.json"


class CleaningPlan(BaseModel):
    steps: list[str]
    python_code: str


MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a data-cleaning agent. Return cleaning steps and one Python function "
            "`clean(df: pd.DataFrame) -> pd.DataFrame` that implements them."
        ),
    },
    {
        "role": "user",
        "content": (
            "Columns: age (float, some NaN), income (strings like '$1,200' and 'n/a'), "
            "city (mixed case, stray spaces)."
        ),
    },
]


def test_structured_call_against_recorded_go_response() -> None:
    recording = bool(os.getenv("DQ_RECORD"))
    transport: RecordingTransport | ReplayTransport
    if recording:
        transport = RecordingTransport(CASSETTE)
        config = LLMConfig.from_env()
    else:
        transport = ReplayTransport(CASSETTE)
        config = LLMConfig(api_key=SecretStr("replay"))

    client = LLMClient(config, http_client=httpx2.Client(transport=transport))
    result = client.structured(MESSAGES, CleaningPlan, reasoning_effort="low")
    if isinstance(transport, RecordingTransport):
        transport.save()

    plan = result.value
    assert plan.steps
    tree = ast.parse(plan.python_code)  # the code field is real Python
    assert any(isinstance(n, ast.FunctionDef) and n.name == "clean" for n in ast.walk(tree))
    assert result.model == "deepseek-v4.1-flash"
    assert result.usage.calls == 1
    assert result.usage.completion_tokens > 0
    assert result.usage.allowance_usd > 0
