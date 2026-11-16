"""Validation checks for DataFrame agents (spec S4).

A check is a function `(before, after, context) -> CheckResult`. The engine runs the
generic checks below plus whatever the agent adds, after every successful execution.

- A failed **blocking** check sends the run to `fix` (it counts as a retry).
- A failed **warning** check never triggers a fix; it makes the final status
  `needs_review`.

A check that itself raises is reported as a failed check of its own severity, so a bug in
a check can never turn into a silent pass.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dataquorum.core.statuses import CheckResult, Severity


@dataclass(frozen=True)
class CheckContext:
    request: str  # the user's request, unchanged
    row_loss_threshold: float = 0.5

    def mentions(self, *words: str) -> bool:
        text = self.request.lower()
        return any(w.lower() in text for w in words)


# `after` is a DataFrame for table agents and a Plotly figure dict for figure agents.
CheckFn = Callable[[pd.DataFrame, Any, CheckContext], CheckResult]


@dataclass(frozen=True)
class Check:
    name: str
    severity: Severity
    fn: CheckFn


def run_checks(
    checks: Sequence[Check], before: pd.DataFrame, after: Any, context: CheckContext
) -> list[CheckResult]:
    results = []
    for check in checks:
        try:
            result = check.fn(before, after, context)
            results.append(
                result.model_copy(update={"name": check.name, "severity": check.severity})
            )
        except Exception as exc:  # a broken check must not pass silently
            results.append(
                CheckResult(
                    name=check.name,
                    passed=False,
                    severity=check.severity,
                    detail=f"check raised {type(exc).__name__}: {exc}",
                )
            )
    return results


def _result(passed: bool, detail: str) -> CheckResult:
    # name and severity are filled in by run_checks from the Check definition
    return CheckResult(name="", passed=passed, severity=Severity.WARNING, detail=detail)


def has_rows_and_columns(
    before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext
) -> CheckResult:
    return _result(
        len(after) > 0 and len(after.columns) > 0,
        f"{len(after)} rows x {len(after.columns)} columns",
    )


def row_loss(before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext) -> CheckResult:
    if ctx.mentions(
        "filter", "remove rows", "drop rows", "drop duplicates", "deduplicate", "keep only"
    ):
        return _result(True, "row removal was requested")
    lost = 1 - len(after) / len(before) if len(before) else 0.0
    return _result(
        lost <= ctx.row_loss_threshold, f"{len(before)} -> {len(after)} rows ({lost:.0%} lost)"
    )


def columns_dropped(before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext) -> CheckResult:
    missing = [str(c) for c in before.columns if c not in after.columns]
    # Naming a column is not permission to drop it ("title-case city" must keep city).
    asked_to_remove = ctx.mentions("drop", "remove", "delete", "exclude", "without")
    unrequested = [c for c in missing if not (asked_to_remove and ctx.mentions(c))]
    if not unrequested:
        return _result(True, "no unrequested columns dropped")
    return _result(False, f"dropped without being asked: {unrequested}")


MISSING_TOKENS = frozenset({"", "n/a", "na", "nan", "null", "none", "-", "?", "missing"})


def missing_like(series: pd.Series) -> int:
    """NaN/None plus strings that mean "missing" ("n/a", "", "null", ...).

    Cleaning "n/a" into NaN is not a loss of data, so both sides are counted this way.
    """
    tokens = series.map(lambda v: isinstance(v, str) and v.strip().lower() in MISSING_TOKENS)
    return int(series.isna().sum() + tokens.sum())


def missing_values_increased(
    before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext
) -> CheckResult:
    if len(after) != len(before):
        return _result(True, "row count changed; compared by row_loss instead")
    grew = {
        str(c): (missing_like(before[c]), missing_like(after[c]))
        for c in before.columns
        if c in after.columns and missing_like(after[c]) > missing_like(before[c])
    }
    if not grew:
        return _result(True, "no column gained missing values")
    return _result(False, "missing values grew (before, after): " + str(grew))


def numeric_became_text(
    before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext
) -> CheckResult:
    changed = [
        str(c)
        for c in before.columns
        if c in after.columns
        and pd.api.types.is_numeric_dtype(before[c])
        and not pd.api.types.is_numeric_dtype(after[c])
    ]
    if not changed:
        return _result(True, "numeric columns stayed numeric")
    return _result(False, f"numeric columns became non-numeric: {changed}")


GENERIC_DATAFRAME_CHECKS: tuple[Check, ...] = (
    Check("has_rows_and_columns", Severity.BLOCKING, has_rows_and_columns),
    Check("row_loss", Severity.WARNING, row_loss),
    Check("columns_dropped", Severity.WARNING, columns_dropped),
    Check("missing_values_increased", Severity.WARNING, missing_values_increased),
    Check("numeric_became_text", Severity.WARNING, numeric_became_text),
)


# ------------------------------------------------------------------ figure checks
_POINT_FIELDS = ("x", "y", "z", "values", "labels", "r", "theta", "lat", "lon", "locations")


def point_count(values: Any) -> int:
    """Length of a Plotly data array: a list, or Plotly 6+'s base64 typed array."""
    if isinstance(values, list | tuple):
        return len(values)
    if isinstance(values, dict) and "bdata" in values and "dtype" in values:
        raw = base64.b64decode(values["bdata"])
        return len(raw) // np.dtype(values["dtype"]).itemsize
    return 0


def figure_has_traces(before: pd.DataFrame, fig: dict[str, Any], ctx: CheckContext) -> CheckResult:
    traces = fig.get("data") or []
    return _result(len(traces) > 0, f"{len(traces)} traces")


def traces_have_points(before: pd.DataFrame, fig: dict[str, Any], ctx: CheckContext) -> CheckResult:
    counts = [
        max((point_count(t.get(k)) for k in _POINT_FIELDS), default=0)
        for t in fig.get("data") or []
    ]
    empty = [i for i, n in enumerate(counts) if n == 0]
    if not counts or empty:
        return _result(False, f"traces with no data points: {empty or 'all'}")
    return _result(True, f"points per trace: {counts}")


def figure_has_title(before: pd.DataFrame, fig: dict[str, Any], ctx: CheckContext) -> CheckResult:
    title = (fig.get("layout") or {}).get("title")
    text = title.get("text") if isinstance(title, dict) else title
    return _result(bool(text and str(text).strip()), f"title: {text!r}")


GENERIC_FIGURE_CHECKS: tuple[Check, ...] = (
    Check("figure_has_traces", Severity.BLOCKING, figure_has_traces),
    Check("traces_have_points", Severity.BLOCKING, traces_have_points),
    Check("figure_has_title", Severity.WARNING, figure_has_title),
)
