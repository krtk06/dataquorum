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
Rates come with 95% Wilson intervals; repeated runs add unbiased pass@k.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from dataquorum.evals.checkers import CheckOutcome
from dataquorum.evals.stats import pass_at_k, wilson_interval

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
    repeat: int = 0  # 0..repeats-1 when each task is run several times
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


def _block(rs: list[TaskResult]) -> dict[str, Any]:
    n = len(rs)
    counts = Counter(r.outcome for r in rs)
    return {
        "n": n,
        "correct": counts["correct"],
        "accuracy": counts["correct"] / n if n else 0.0,
        "accuracy_ci95": wilson_interval(counts["correct"], n),
        "false_success": counts["false_success"],
        "false_success_rate": counts["false_success"] / n if n else 0.0,
        "false_success_ci95": wilson_interval(counts["false_success"], n),
        "outcomes": dict(counts),
        "allowance_usd": round(sum(r.allowance_usd for r in rs), 5),
        "median_seconds": round(statistics.median(r.seconds for r in rs), 1) if rs else 0.0,
        "mean_calls": round(sum(r.calls for r in rs) / n, 2) if n else 0.0,
    }


def summarise(results: list[TaskResult]) -> dict[str, Any]:
    agents = sorted({r.agent for r in results})
    summary: dict[str, Any] = {
        "overall": _block(results),
        "by_agent": {a: _block([r for r in results if r.agent == a]) for a in agents},
    }
    repeats = max((r.repeat for r in results), default=0) + 1
    if repeats > 1:
        summary["pass_at_k"] = pass_at_k_table(results, repeats)
    return summary


def pass_at_k_table(results: list[TaskResult], repeats: int) -> dict[str, float]:
    """Mean unbiased pass@k over the tasks that have all `repeats` attempts."""
    per_task: dict[str, list[bool]] = {}
    for r in results:
        per_task.setdefault(r.task_id, []).append(r.outcome == "correct")
    complete = [v for v in per_task.values() if len(v) == repeats]
    if not complete:
        return {}
    return {
        f"pass@{k}": round(sum(pass_at_k(repeats, sum(v), k) for v in complete) / len(complete), 4)
        for k in range(1, repeats + 1)
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "| Scope | n | Correct (95% CI) | False success (95% CI) | Other | Allowance $ | Median s |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, b in [("overall", summary["overall"]), *summary["by_agent"].items()]:
        other = {k: v for k, v in b["outcomes"].items() if k not in ("correct", "false_success")}
        alo, ahi = b["accuracy_ci95"]
        flo, fhi = b["false_success_ci95"]
        lines.append(
            f"| {name} | {b['n']} | {b['correct']} ({b['accuracy']:.0%}, {alo:.0%}-{ahi:.0%}) | "
            f"{b['false_success']} ({b['false_success_rate']:.0%}, {flo:.0%}-{fhi:.0%}) | "
            f"{other or '-'} | {b['allowance_usd']:.4f} | {b['median_seconds']} |"
        )
    if summary.get("pass_at_k"):
        lines.append("\n" + ", ".join(f"{k} = {v:.1%}" for k, v in summary["pass_at_k"].items()))
    return "\n".join(lines)
