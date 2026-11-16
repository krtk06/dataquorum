"""LLM client: structured output only, with cost metering and a budget guard (spec S3).

Every LLM step in DataQuorum asks for a Pydantic model and gets back a validated instance.
Code is a field of that model, never text scraped from a reply.

How one `structured()` call works:
1. Check the budget guard.
2. Ask with `response_format=json_schema` (strict). If the endpoint rejects that with a
   400, switch to `json_object` mode and put the schema in a system message.
3. Validate the reply against the schema. If it does not match, send the error back once
   and ask again (one repair attempt). If it still does not match, raise
   `StructuredOutputError`. There is no fallback to regex extraction.
4. If the primary model keeps failing for server-side reasons (after the SDK's own
   retries), try the fallback model once.
5. Record usage after every HTTP call, so failed attempts are still counted.

We use the `openai` SDK directly rather than LangChain's chat models because Go's
extra usage fields (`usage.opencode.allowance`) and DeepSeek's `reasoning_content`
are only visible on the raw response.
"""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, cast

import httpx2
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, ValidationError

import dataquorum
from dataquorum.core.budget import BudgetGuard
from dataquorum.core.cost import Usage, usage_from_response

T = TypeVar("T", bound=BaseModel)
ReasoningEffort = Literal["minimal", "low", "medium", "high"]
Message = dict[str, str]

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_FALLBACK_MODEL = "glm-5.3-flash"

# Errors that are worth retrying on another model (server-side or capacity problems).
_FALLBACK_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.RateLimitError,
)


class LLMConfig(BaseModel):
    api_key: SecretStr
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL
    # None = provider default. Week 2 experiment picks the project default.
    reasoning_effort: ReasoningEffort | None = None
    temperature: float | None = None
    timeout_seconds: float = 180.0
    max_retries: int = 3  # SDK-level retries for connection errors, 429 and 5xx

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Read OPENCODE_GO_API_KEY and optional DQ_* overrides from the environment/.env."""
        load_dotenv()
        key = os.getenv("OPENCODE_GO_API_KEY")
        if not key:
            raise RuntimeError("OPENCODE_GO_API_KEY is not set (add it to .env).")
        effort = os.getenv("DQ_REASONING_EFFORT") or None
        return cls(
            api_key=SecretStr(key),
            base_url=os.getenv("DQ_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("DQ_MODEL", DEFAULT_MODEL),
            fallback_model=os.getenv("DQ_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL) or None,
            reasoning_effort=cast(ReasoningEffort | None, effort),
        )


class LLMError(RuntimeError):
    """An LLM step failed. `usage` holds what was spent before the failure."""

    def __init__(self, message: str, usage: Usage) -> None:
        super().__init__(message)
        self.usage = usage


class StructuredOutputError(LLMError):
    """The model did not return valid output for the schema, even after one repair."""

    def __init__(self, message: str, usage: Usage, raw: str | None) -> None:
        super().__init__(message, usage)
        self.raw = raw


@dataclass
class StructuredResult[T: BaseModel]:
    value: T
    usage: Usage
    model: str
    reasoning: str | None = None  # stored for tracing, never executed


def strict_json_schema(schema_model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema adjusted for strict mode: every property required, no extras."""
    schema = copy.deepcopy(schema_model.model_json_schema())

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return schema


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        *,
        budget: BudgetGuard | None = None,
        session_id: str | None = None,
        http_client: httpx2.Client | None = None,
    ) -> None:
        self.config = config
        self.budget = budget
        self.session_id = session_id or f"dq-{uuid.uuid4()}"
        self._client = openai.OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            default_headers={
                "User-Agent": f"dataquorum/{dataquorum.__version__}",
                "x-opencode-session": self.session_id,
            },
            http_client=http_client,
        )

    def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        reasoning_effort: ReasoningEffort | None = None,
        model: str | None = None,
    ) -> StructuredResult[T]:
        """Ask for `schema` and return a validated instance (see module docstring)."""
        effort = reasoning_effort or self.config.reasoning_effort
        primary = model or self.config.model
        try:
            return self._structured_on(primary, messages, schema, effort)
        except LLMError as exc:
            fallback = self.config.fallback_model
            if fallback and fallback != primary and isinstance(exc.__cause__, _FALLBACK_ERRORS):
                result = self._structured_on(fallback, messages, schema, effort)
                result.usage = exc.usage + result.usage
                return result
            raise

    def _structured_on(
        self, model: str, messages: list[Message], schema: type[T], effort: ReasoningEffort | None
    ) -> StructuredResult[T]:
        usage = Usage()
        convo = list(messages)
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": strict_json_schema(schema),
                "strict": True,
            },
        }
        repaired = False
        while True:
            try:
                content, reasoning, call_usage = self._call(model, convo, response_format, effort)
            except openai.BadRequestError as exc:
                if response_format["type"] != "json_schema":
                    raise LLMError(f"{model} rejected the request: {exc}", usage) from exc
                response_format, convo = _json_object_mode(schema, convo)
                usage.retries += 1
                continue
            except openai.OpenAIError as exc:
                raise LLMError(f"{model} call failed: {exc}", usage) from exc
            usage = usage + call_usage

            try:
                value = schema.model_validate_json(content or "")
                return StructuredResult(value=value, usage=usage, model=model, reasoning=reasoning)
            except ValidationError as exc:
                if repaired:
                    raise StructuredOutputError(
                        f"{model} returned invalid {schema.__name__} after one repair: {exc}",
                        usage,
                        content,
                    ) from exc
                repaired = True
                usage.retries += 1
                convo = [*convo, *_repair_messages(content, exc)]

    def _call(
        self,
        model: str,
        messages: list[Message],
        response_format: dict[str, Any],
        effort: ReasoningEffort | None,
    ) -> tuple[str | None, str | None, Usage]:
        if self.budget:
            self.budget.check()
        extra: dict[str, Any] = {}
        if effort:
            extra["reasoning_effort"] = effort
        if self.config.temperature is not None:
            extra["temperature"] = self.config.temperature

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            response_format=cast(Any, response_format),
            **extra,
        )
        seconds = time.perf_counter() - start

        raw_usage = response.usage.model_dump() if response.usage else {}
        usage = usage_from_response(model, raw_usage, seconds=seconds, when=datetime.now(UTC))
        if self.budget:
            self.budget.record(usage)

        message = response.choices[0].message
        reasoning = (message.model_extra or {}).get("reasoning_content")
        return message.content, reasoning, usage


def _json_object_mode(
    schema: type[BaseModel], messages: list[Message]
) -> tuple[dict[str, Any], list[Message]]:
    """Fallback when json_schema is not supported: schema goes into a system message."""
    instruction = (
        "Reply with a single JSON object, and nothing else, that matches this JSON Schema:\n"
        + json.dumps(strict_json_schema(schema))
    )
    return {"type": "json_object"}, [{"role": "system", "content": instruction}, *messages]


def _repair_messages(content: str | None, error: ValidationError) -> list[Message]:
    return [
        {"role": "assistant", "content": content or ""},
        {
            "role": "user",
            "content": (
                "Your reply did not match the required JSON schema. Errors:\n"
                f"{error.errors(include_url=False)}\n"
                "Reply again with only the corrected JSON object."
            ),
        },
    ]
