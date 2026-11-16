"""Week-1 smoke test for the OpenCode Go endpoint.

Sends one plain call and one structured-output call, then reports real token usage
(including reasoning and cached tokens, if the API returns them), latency and an
estimated cost. Run with:

    uv run python scripts/go_smoke_test.py
    uv run python scripts/go_smoke_test.py --model glm-5.3-flash

Needs OPENCODE_GO_API_KEY in the environment or in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import dataquorum

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"

# Off-peak USD per 1M tokens for deepseek-v4.1-flash (from CLAUDE.md). Peak is 2x.
PRICE_IN, PRICE_OUT, PRICE_CACHED_IN = 0.15, 0.60, 0.003
PEAK_HOURS_UTC = {1, 2, 3, 6, 7, 8, 9}  # 01:00-04:00 and 06:00-10:00, Mon-Fri


class CleaningPlan(BaseModel):
    """Small schema to test structured output."""

    steps: list[str]
    python_code: str


def is_peak(now: datetime) -> bool:
    return now.weekday() < 5 and now.hour in PEAK_HOURS_UTC


def estimate_cost(usage: dict[str, Any], peak: bool) -> float:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    cost = ((prompt - cached) * PRICE_IN + cached * PRICE_CACHED_IN + completion * PRICE_OUT) / 1e6
    return cost * (2 if peak else 1)


def run_call(client: OpenAI, model: str, label: str, **kwargs: Any) -> dict[str, Any]:
    start = time.perf_counter()
    resp = client.chat.completions.create(model=model, **kwargs)
    elapsed = time.perf_counter() - start
    message = resp.choices[0].message
    usage = resp.usage.model_dump() if resp.usage else {}
    reasoning_text = getattr(message, "reasoning_content", None) or (
        (message.model_extra or {}).get("reasoning_content")
    )
    return {
        "label": label,
        "model": resp.model,
        "seconds": round(elapsed, 2),
        "finish_reason": resp.choices[0].finish_reason,
        "usage": usage,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "has_reasoning_content": bool(reasoning_text),
        "content": message.content,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.getenv("DQ_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("DQ_BASE_URL", DEFAULT_BASE_URL))
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENCODE_GO_API_KEY")
    if not api_key:
        print("OPENCODE_GO_API_KEY is not set (add it to .env).", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        default_headers={
            "User-Agent": f"dataquorum/{dataquorum.__version__}",
            "x-opencode-session": f"dq-smoke-{uuid.uuid4()}",
        },
        timeout=120,
    )

    task = (
        "A pandas DataFrame `df` has columns age (some NaN), income (strings like '$1,200') "
        "and city (mixed case). Propose cleaning steps and the Python code."
    )
    results = [
        run_call(
            client,
            args.model,
            "plain",
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        ),
        run_call(
            client,
            args.model,
            "structured",
            messages=[
                {"role": "system", "content": "You are a data-cleaning agent."},
                {"role": "user", "content": task},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "CleaningPlan",
                    "schema": CleaningPlan.model_json_schema(),
                    "strict": True,
                },
            },
        ),
    ]

    now = datetime.now(UTC)
    peak = is_peak(now)
    for r in results:
        r["est_cost_usd"] = round(estimate_cost(r["usage"], peak), 6)
    structured = results[1]
    try:
        CleaningPlan.model_validate_json(structured["content"] or "")
        structured["schema_valid"] = True
    except ValueError as exc:
        structured["schema_valid"] = False
        structured["schema_error"] = str(exc)[:300]

    print(json.dumps({"utc": now.isoformat(), "peak": peak, "calls": results}, indent=2))
    return 0 if structured["schema_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
