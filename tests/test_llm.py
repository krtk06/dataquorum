"""LLMClient behaviour against a fake HTTP server (no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel, SecretStr

from dataquorum.core.budget import BudgetExceededError, BudgetGuard, InMemoryLedger, SpendEntry
from dataquorum.core.llm import (
    LLMClient,
    LLMConfig,
    LLMError,
    StructuredOutputError,
    strict_json_schema,
)


class Step(BaseModel):
    name: str
    reason: str


class Plan(BaseModel):
    steps: list[Step]
    python_code: str


VALID = json.dumps({"steps": [{"name": "fix income", "reason": "strings"}], "python_code": "x = 1"})
GO_USAGE = {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "completion_tokens_details": {"reasoning_tokens": 60},
    "opencode": {"list_cost": 0.001, "allowance": 0.0006},
}


def completion(content: str, model: str = "deepseek-v4.1-flash", **message: Any) -> dict[str, Any]:
    return {
        "id": "cmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, **message},
                "finish_reason": "stop",
            }
        ],
        "usage": GO_USAGE,
    }


class FakeServer:
    """Returns queued (status, body) pairs and keeps every request for assertions."""

    def __init__(self, *replies: tuple[int, dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        status, body = self.replies.pop(0)
        return httpx2.Response(status, json=body)

    def body(self, i: int) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.requests[i].content)
        return result


def make_client(server: FakeServer, budget: BudgetGuard | None = None, **cfg: Any) -> LLMClient:
    config = LLMConfig(api_key=SecretStr("test-key"), max_retries=0, **cfg)
    http = httpx2.Client(transport=httpx2.MockTransport(server))
    return LLMClient(config, budget=budget, session_id="dq-test", http_client=http)


MESSAGES = [{"role": "user", "content": "clean this"}]
ERROR_400 = {"error": {"message": "response_format json_schema not supported"}}
ERROR_500 = {"error": {"message": "upstream down"}}


def test_happy_path_returns_validated_model_and_usage() -> None:
    server = FakeServer((200, completion(VALID, reasoning_content="thinking...")))
    result = make_client(server).structured(MESSAGES, Plan)

    assert result.value.steps[0].name == "fix income"
    assert result.reasoning == "thinking..."
    assert result.model == "deepseek-v4.1-flash"
    assert result.usage.calls == 1
    assert result.usage.reasoning_tokens == 60
    assert result.usage.allowance_usd == pytest.approx(0.0006)

    body = server.body(0)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "reasoning_effort" not in body  # provider default unless configured


def test_sends_identity_headers_and_bearer_key() -> None:
    server = FakeServer((200, completion(VALID)))
    make_client(server).structured(MESSAGES, Plan)
    headers = server.requests[0].headers
    assert headers["user-agent"].startswith("dataquorum/")
    assert headers["x-opencode-session"] == "dq-test"
    assert headers["authorization"] == "Bearer test-key"


def test_reasoning_effort_from_call_overrides_config() -> None:
    server = FakeServer((200, completion(VALID)), (200, completion(VALID)))
    client = make_client(server, reasoning_effort="medium")
    client.structured(MESSAGES, Plan)
    client.structured(MESSAGES, Plan, reasoning_effort="low")
    assert server.body(0)["reasoning_effort"] == "medium"
    assert server.body(1)["reasoning_effort"] == "low"


def test_falls_back_to_json_object_when_json_schema_rejected() -> None:
    server = FakeServer((400, ERROR_400), (200, completion(VALID)))
    result = make_client(server).structured(MESSAGES, Plan)

    assert result.value.python_code == "x = 1"
    second = server.body(1)
    assert second["response_format"] == {"type": "json_object"}
    assert second["messages"][0]["role"] == "system"
    assert "JSON Schema" in second["messages"][0]["content"]
    assert result.usage.retries == 1
    assert result.usage.calls == 1  # the rejected request had no usage


def test_one_repair_attempt_then_success() -> None:
    server = FakeServer((200, completion('{"steps": "oops"}')), (200, completion(VALID)))
    result = make_client(server).structured(MESSAGES, Plan)

    assert result.value.python_code == "x = 1"
    assert (result.usage.calls, result.usage.retries) == (2, 1)
    repair = server.body(1)["messages"]
    assert repair[-2] == {"role": "assistant", "content": '{"steps": "oops"}'}
    assert "did not match" in repair[-1]["content"]


def test_invalid_twice_raises_with_usage_and_raw_reply() -> None:
    server = FakeServer((200, completion("not json")), (200, completion("still not json")))
    with pytest.raises(StructuredOutputError) as info:
        make_client(server).structured(MESSAGES, Plan)
    assert info.value.raw == "still not json"
    assert info.value.usage.calls == 2  # spend is reported even on failure


def test_server_error_falls_back_to_other_model() -> None:
    server = FakeServer((500, ERROR_500), (200, completion(VALID, model="glm-5.3-flash")))
    result = make_client(server).structured(MESSAGES, Plan)
    assert server.body(0)["model"] == "deepseek-v4.1-flash"
    assert server.body(1)["model"] == "glm-5.3-flash"
    assert result.model == "glm-5.3-flash"


def test_server_error_without_fallback_raises() -> None:
    server = FakeServer((500, ERROR_500))
    with pytest.raises(LLMError):
        make_client(server, fallback_model=None).structured(MESSAGES, Plan)


def test_bad_request_in_json_object_mode_is_not_retried_forever() -> None:
    server = FakeServer((400, ERROR_400), (400, ERROR_400))
    with pytest.raises(LLMError, match="rejected"):
        make_client(server).structured(MESSAGES, Plan)
    assert len(server.requests) == 2


def test_budget_guard_blocks_before_any_request_and_records_spend() -> None:
    ledger = InMemoryLedger()
    ledger.append(SpendEntry(datetime.now(UTC), 100.0, "m"))
    server = FakeServer()
    with pytest.raises(BudgetExceededError):
        make_client(server, BudgetGuard(ledger, mode="eval")).structured(MESSAGES, Plan)
    assert server.requests == []

    fresh = InMemoryLedger()
    server = FakeServer((200, completion(VALID)))
    make_client(server, BudgetGuard(fresh, mode="eval")).structured(MESSAGES, Plan)
    assert fresh.entries[0].allowance_usd == pytest.approx(0.0006)


def test_strict_schema_closes_nested_objects() -> None:
    schema = strict_json_schema(Plan)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["steps", "python_code"]
    step = schema["$defs"]["Step"]
    assert step["additionalProperties"] is False
    assert step["required"] == ["name", "reason"]
