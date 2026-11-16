"""Token usage and cost accounting for LLM calls.

OpenCode Go reports two costs per call in `usage.opencode`:
- `list_cost`: the public price of the call.
- `allowance`: what the call counts against the subscription limit. For DeepSeek V4.1 Flash
  this was 0.6 x list cost on 2026-11-16; for GLM-5.3-Flash it equalled list cost.

We always prefer Go's own numbers and only fall back to the price table below when a
response does not include them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class Price:
    """Off-peak USD per 1M tokens, plus how Go applies peak pricing and allowance."""

    input: float
    output: float
    cached_input: float
    peak_multiplier: float = 1.0
    allowance_ratio: float = 1.0


PRICES: dict[str, Price] = {
    # From the Go pricing page (CLAUDE.md); allowance ratio observed in the smoke test.
    "deepseek-v4.1-flash": Price(0.15, 0.60, 0.003, peak_multiplier=2.0, allowance_ratio=0.6),
    # Derived from two observed list costs on 2026-11-16 (0.15 in / 0.50 out). Cached-input
    # price and peak behaviour are unverified, so we assume no discount and no peak.
    "glm-5.3-flash": Price(0.15, 0.50, 0.15),
}

# Go peak hours: 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday.
_PEAK_HOURS_UTC = frozenset({1, 2, 3, 6, 7, 8, 9})


class UnknownModelPriceError(KeyError):
    """No price is known for a model and the response did not include Go's costs."""


def is_peak(when: datetime) -> bool:
    utc = when.astimezone(UTC)
    return utc.weekday() < 5 and utc.hour in _PEAK_HOURS_UTC


class Usage(BaseModel):
    """Accumulated usage for one or more LLM calls (spec S2)."""

    models: list[str] = []
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    list_cost_usd: float = 0.0
    allowance_usd: float = 0.0
    calls: int = 0
    retries: int = 0
    seconds: float = 0.0  # LLM wall time
    execution_seconds: float = 0.0  # sandbox wall time (set by the engine)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            models=self.models + [m for m in other.models if m not in self.models],
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            list_cost_usd=self.list_cost_usd + other.list_cost_usd,
            allowance_usd=self.allowance_usd + other.allowance_usd,
            calls=self.calls + other.calls,
            retries=self.retries + other.retries,
            seconds=self.seconds + other.seconds,
            execution_seconds=self.execution_seconds + other.execution_seconds,
        )


def _int(value: Any) -> int:
    return int(value) if value else 0


def usage_from_response(
    model: str, raw: dict[str, Any], *, seconds: float, when: datetime
) -> Usage:
    """Build a Usage from one response's `usage` dict (OpenAI shape plus Go's extras)."""
    prompt = _int(raw.get("prompt_tokens"))
    completion = _int(raw.get("completion_tokens"))
    reasoning = _int((raw.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    cached = _int((raw.get("prompt_tokens_details") or {}).get("cached_tokens"))

    go = raw.get("opencode") or {}
    if go.get("list_cost") is not None and go.get("allowance") is not None:
        list_cost, allowance = float(go["list_cost"]), float(go["allowance"])
    else:
        list_cost = estimate_list_cost(model, prompt, completion, cached, when)
        allowance = list_cost * PRICES[model].allowance_ratio

    return Usage(
        models=[model],
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cached_tokens=cached,
        list_cost_usd=list_cost,
        allowance_usd=allowance,
        calls=1,
        seconds=seconds,
    )


def estimate_list_cost(
    model: str, prompt: int, completion: int, cached: int, when: datetime
) -> float:
    """List cost from the price table. Reasoning tokens are part of `completion`."""
    if model not in PRICES:
        raise UnknownModelPriceError(model)
    p = PRICES[model]
    cost = ((prompt - cached) * p.input + cached * p.cached_input + completion * p.output) / 1e6
    return cost * (p.peak_multiplier if is_peak(when) else 1.0)
