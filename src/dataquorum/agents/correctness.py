"""Pre-fit audit agent. The host independently recomputes every row from the input."""

from __future__ import annotations

import re

import pandas as pd

from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.statuses import CheckResult, Severity
from dataquorum.core.validation import Check, CheckContext
from dataquorum.ml.correctness import Problem, audit


def _column(request: str, role: str) -> str | None:
    name = r"""(?:`([^`]+)`|'([^']+)'|"([^"]+)"|([\w.-]+))"""
    pattern = rf"\b{role}(?:\s+column)?\s*(?:is\s+|=\s*|:\s*){name}"
    match = re.search(pattern, request, flags=re.I)
    return next(v for v in match.groups() if v is not None).rstrip(".") if match else None


def requested_audit(frame: pd.DataFrame, request: str) -> pd.DataFrame:
    target = _column(request, "target")
    classification = bool(re.search(r"\bclassification\b", request, flags=re.I))
    regression = bool(re.search(r"\bregression\b", request, flags=re.I))
    if target is None or classification == regression:
        raise ValueError(
            "Name the target ('target is ...') and specify classification or regression"
        )
    problem: Problem = "classification" if classification else "regression"
    return audit(
        frame,
        target,
        problem=problem,
        group=_column(request, "group"),
        time=_column(request, "time"),
        split=_column(request, "split"),
    )


def _matches(before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext) -> CheckResult:
    expected = requested_audit(before, ctx.request)
    cols = list(expected.columns)
    if list(after.columns) != cols:
        ok, detail = False, f"Expected columns {cols}"
    else:
        want = sorted(expected.itertuples(index=False, name=None))
        got = sorted(after.itertuples(index=False, name=None))
        ok = got == want
        detail = "All audit rows verified" if ok else f"Expected {want}, got {got[:12]}"
    return CheckResult(name="", severity=Severity.BLOCKING, passed=ok, detail=detail)


def _risks(before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext) -> CheckResult:
    flagged = after.loc[after["kind"].eq("risk"), ["subject", "value"]]
    return CheckResult(
        name="",
        severity=Severity.WARNING,
        passed=flagged.empty,
        detail="Review these risks: " + str(flagged.to_dict("records"))
        if len(flagged)
        else "No risks flagged",
    )


CORRECTNESS = AgentSpec(
    name="ml_correctness",
    role="You audit supervised machine-learning datasets before fitting a model.",
    recommend_guidance="""Identify the target, problem type, and optional group, time or split
columns. Ask for a blocker if the target or problem type is absent. Plan to check for leakage,
train/test contamination and a suitable validation strategy and metric.""",
    code_guidance="""Use the installed function from dataquorum.ml.correctness import audit.
Return audit(df, target, problem='classification' or 'regression', group=..., time=..., split=...).
Use the specified column names and omit optional roles that were not explicitly supplied.
The helper inspects the full frame and returns the required kind, subject, value table.
Do not change, add or remove report rows.""",
    generic_checks=(),
    extra_checks=(
        Check("audit_matches_input", Severity.BLOCKING, _matches),
        Check("review_flagged_risks", Severity.WARNING, _risks),
    ),
)
