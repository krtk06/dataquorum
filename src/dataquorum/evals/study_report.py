"""Matched comparisons and failure evidence for the Week 14 study."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import asdict, replace
from typing import Any, Literal

from dataquorum.core.findings import grounding_check
from dataquorum.core.state import AgentResult, DataFrameOutput
from dataquorum.evals.checkers import CheckOutcome
from dataquorum.evals.scoring import TaskResult, classify
from dataquorum.evals.stats import bootstrap_ci, mcnemar, wilson_interval

Failure = Literal[
    "correct",
    "declared_blocker",
    "upstream_llm",
    "structured_output",
    "agent_validation",
    "execution",
    "oracle_mismatch",
    "status_mismatch",
    "grader_review",
]


def failure_stage(scored: TaskResult, result: AgentResult) -> Failure:
    """Classify evidence from the recorded result. A grader exception needs review."""
    if any(c.detail.startswith(("checker raised ", "unknown checker ")) for c in scored.checks):
        return "grader_review"
    error = result.error or ""
    if error.startswith(("Recommending steps failed:", "Code generation failed:")):
        return "structured_output" if " after one repair:" in error else "upstream_llm"
    if error.startswith("Cannot do this with the given data:"):
        return "declared_blocker"
    if scored.outcome == "correct" and scored.expect_status != "failed":
        return "correct"
    if any(not c.passed for c in result.validation):
        return "agent_validation"
    if result.output is None or result.output.is_empty:
        return "execution"
    if any(not c.passed for c in scored.checks):
        return "oracle_mismatch"
    return "status_mismatch"


def record_trial(
    arm: str, scored: TaskResult, result: AgentResult, seconds: float
) -> dict[str, Any]:
    table_passes = all(c.passed for c in scored.checks)
    legacy_outcome = scored.outcome
    if (
        scored.agent == "eda"
        and scored.status != "failed"
        and isinstance(result.output, DataFrameOutput)
    ):
        if result.message.startswith("- "):
            figures = [line.removeprefix("- ") for line in result.message.splitlines()]
            check = grounding_check(figures, result.output.load())
            findings = CheckOutcome("findings_grounded", check.passed, check.detail)
        else:
            findings = CheckOutcome("findings_grounded", False, "No findings were returned.")
        checks = [*scored.checks, findings]
        scored = replace(
            scored, checks=checks, outcome=classify(scored.status, scored.expect_status, checks)
        )
    stage = failure_stage(scored, result)
    verified = scored.outcome == "correct"
    if scored.status == "failed" and scored.expect_status == "failed":
        verified = stage == "declared_blocker" and all(c.passed for c in scored.checks)
    return {
        **scored.as_dict(),
        "arm": arm,
        "failure_stage": stage,
        "verified_correct": verified,
        "legacy_outcome": legacy_outcome,
        "oracle_output_passed": table_passes,
        "wall_seconds": seconds,
        "agent_checks": [c.model_dump(mode="json") for c in result.validation],
    }


def matched_comparison(
    rows: list[dict[str, Any]], arms: list[str], planned: list[str]
) -> dict[str, Any]:
    """All arms use the same task denominator. Tasks with grader errors remain visible."""
    indexed = {(row["arm"], row["task_id"]): row for row in rows}
    shared = [
        task
        for task in planned
        if all(
            (arm, task) in indexed and indexed[arm, task]["failure_stage"] != "grader_review"
            for arm in arms
        )
    ]
    result: dict[str, Any] = {
        "matched_task_ids": shared,
        "planned_per_arm": len(planned),
        "completed_per_arm": {arm: sum((arm, task) in indexed for task in planned) for arm in arms},
        "arms": {},
        "pairs_vs_full": {},
    }
    if not shared:
        return result
    size = len(shared)
    for arm in arms:
        selected = [indexed[arm, task] for task in shared]
        correct = sum(row["verified_correct"] for row in selected)
        false_success = sum(row["outcome"] == "false_success" for row in selected)
        cost = sum(row["allowance_usd"] for row in selected)
        result["arms"][arm] = {
            "correct": correct,
            "correct_ci95": wilson_interval(correct, size),
            "false_success": false_success,
            "false_success_ci95": wilson_interval(false_success, size),
            "oracle_output_passes": sum(row["oracle_output_passed"] for row in selected),
            "blocked": sum(row["failure_stage"] == "declared_blocker" for row in selected),
            "legacy_correct": sum(row["legacy_outcome"] == "correct" for row in selected),
            "allowance_usd": cost,
            "cost_per_verified_task": cost / correct if correct else None,
            "median_wall_seconds": statistics.median(row["wall_seconds"] for row in selected),
        }
        if arm == arms[0]:
            continue
        control = [indexed[arms[0], task] for task in shared]
        costs = [
            b["allowance_usd"] - a["allowance_usd"] for a, b in zip(control, selected, strict=True)
        ]
        times = [
            b["wall_seconds"] - a["wall_seconds"] for a, b in zip(control, selected, strict=True)
        ]
        result["pairs_vs_full"][arm] = {
            "accuracy": asdict(
                mcnemar(
                    {a["task_id"]: a["verified_correct"] for a in control},
                    {b["task_id"]: b["verified_correct"] for b in selected},
                )
            ),
            "false_success": asdict(
                mcnemar(
                    {a["task_id"]: a["outcome"] == "false_success" for a in control},
                    {b["task_id"]: b["outcome"] == "false_success" for b in selected},
                )
            ),
            "mean_allowance_delta": statistics.mean(costs),
            "allowance_delta_ci95": bootstrap_ci(costs, resamples=2000),
            "mean_wall_seconds_delta": statistics.mean(times),
            "wall_seconds_delta_ci95": bootstrap_ci(times, resamples=2000),
        }
    return result


def summarize_study(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    ablations = [a["name"] for a in plan["arms"] if a["model"] == plan["arms"][0]["model"]]
    models = [a["name"] for a in plan["arms"] if a["variant"] == "full"]
    task_ids = [t["id"] for t in plan["tasks"]]
    return {
        "completed_trials": len(rows),
        "planned_trials": len(task_ids) * len(plan["arms"]),
        "graded_trial_allowance_usd": sum(r["allowance_usd"] for r in rows),
        "ablations": matched_comparison(rows, ablations, task_ids),
        "models": matched_comparison(rows, models, task_ids),
        "failure_stages": dict(Counter(r["failure_stage"] for r in rows)),
        "review_queue": [
            {
                key: r[key]
                for key in ("arm", "task_id", "outcome", "failure_stage", "error", "checks")
            }
            for r in rows
            if not r["verified_correct"]
        ],
    }
