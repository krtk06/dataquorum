"""Turn (agent status, checker results) into one outcome per task, and summarise a run.

Outcomes, from best to worst for a project whose agents "must prove their work":

| outcome          | meaning                                                              |
|------------------|----------------------------------------------------------------------|
| correct          | right answer (or, for impossible tasks, a correct refusal)          |
| flagged          | wrong answer, but the agent said `needs_review`                      |
| honest_failure   | no answer, and the agent said `failed`                               |
| false_failure    | right answer, but the agent said `failed`                            |
| false_success    | wrong answer (or an invented one) reported as `success`              |

`false_success` is the headline metric: it is exactly what the original project did.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from dataquorum.evals.checkers import CheckOutcome

Outcome = Literal["correct", "flagged", "honest_failure", "false_failure", "false_success"]


@dataclass
class TaskResult:
    task_id: str
    agent: str
    tags: list[str]
    expect_status: str
    status: str
    outcome: Outcome
    checks: list[CheckOutcome]
    error: str | None = None
    run_id: str | None = None
    calls: int = 0
    retries: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    allowance_usd: float = 0.0
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["checks"] = [asdict(c) for c in self.checks]
        return row


def classify(status: str, expect_status: str, checks: list[CheckOutcome]) -> Outcome:
    checks_pass = all(c.passed for c in checks)
    if expect_status == "failed":  # impossible task: refusing is the right answer
        if status == "failed":
            return "correct"
        return "flagged" if status == "needs_review" else "false_success"
    if status == "failed":
        return "false_failure" if checks_pass else "honest_failure"
    if checks_pass:
        return "correct"
    return "flagged" if status == "needs_review" else "false_success"


def summarise(results: list[TaskResult]) -> dict[str, Any]:
    def block(rs: list[TaskResult]) -> dict[str, Any]:
        n = len(rs)
        counts = Counter(r.outcome for r in rs)
        return {
            "n": n,
            "correct": counts["correct"],
            "accuracy": counts["correct"] / n if n else 0.0,
            "false_success": counts["false_success"],
            "false_success_rate": counts["false_success"] / n if n else 0.0,
            "outcomes": dict(counts),
            "allowance_usd": round(sum(r.allowance_usd for r in rs), 5),
            "median_seconds": round(statistics.median(r.seconds for r in rs), 1) if rs else 0.0,
            "mean_calls": round(sum(r.calls for r in rs) / n, 2) if n else 0.0,
        }

    by_agent = {
        a: block([r for r in results if r.agent == a]) for a in sorted({r.agent for r in results})
    }
    return {"overall": block(results), "by_agent": by_agent}


def format_summary(summary: dict[str, Any]) -> str:
    lines = ["| Scope | n | Correct | False success | Other outcomes | Allowance $ | Median s |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    rows = [("overall", summary["overall"]), *summary["by_agent"].items()]
    for name, b in rows:
        other = {k: v for k, v in b["outcomes"].items() if k not in ("correct", "false_success")}
        lines.append(
            f"| {name} | {b['n']} | {b['correct']} ({b['accuracy']:.0%}) | "
            f"{b['false_success']} ({b['false_success_rate']:.0%}) | {other or '-'} | "
            f"{b['allowance_usd']:.4f} | {b['median_seconds']} |"
        )
    return "\n".join(lines)
