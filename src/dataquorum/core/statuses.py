"""Run statuses and the one function that decides them (spec S2, S4).

The status of a run is computed here from facts: is there an output, is it empty, which
checks failed, did something raise. The LLM never chooses a status. This is the fix for
the original project's false successes (bug log #5, #6).

Decision order:
1. An error was raised, or there is no output             -> failed
2. The output is empty (0 rows, no traces, blank text)     -> failed
3. Any blocking check failed                               -> failed
4. Any warning check failed, or a human must decide        -> needs_review
5. Otherwise                                               -> success
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel


class Status(StrEnum):
    SUCCESS = "success"  # output exists, is not empty, and every blocking check passed
    FAILED = "failed"  # no usable output: errors, retries exhausted, empty output, refusal
    NEEDS_REVIEW = "needs_review"  # usable output, but a warning fired or a human must decide


class Severity(StrEnum):
    BLOCKING = "blocking"  # a failure means the output is not usable
    WARNING = "warning"  # a failure means the output needs a human look


class CheckResult(BaseModel):
    name: str
    passed: bool
    severity: Severity
    detail: str = ""


@dataclass(frozen=True)
class Decision:
    status: Status
    error: str | None  # set whenever status is not success


def decide_status(
    *,
    has_output: bool,
    output_empty: bool,
    checks: list[CheckResult],
    error: str | None = None,
    review_reason: str | None = None,
) -> Decision:
    if error:
        return Decision(Status.FAILED, error)
    if not has_output:
        return Decision(Status.FAILED, "No output was produced.")
    if output_empty:
        return Decision(Status.FAILED, "The output is empty.")

    blocking = [c for c in checks if not c.passed and c.severity is Severity.BLOCKING]
    if blocking:
        return Decision(Status.FAILED, "Blocking checks failed: " + _describe(blocking))

    warnings = [c for c in checks if not c.passed and c.severity is Severity.WARNING]
    reasons = []
    if warnings:
        reasons.append("Warning checks failed: " + _describe(warnings))
    if review_reason:
        reasons.append(review_reason)
    if reasons:
        return Decision(Status.NEEDS_REVIEW, " ".join(reasons))
    return Decision(Status.SUCCESS, None)


def _describe(checks: list[CheckResult]) -> str:
    return "; ".join(f"{c.name} ({c.detail})" if c.detail else c.name for c in checks)
