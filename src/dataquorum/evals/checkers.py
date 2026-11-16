"""Checkers: compare an agent's output with the known answer (never with the LLM's claims).

Each checker gets the built task (clean sample, messy input, expected frame) and the agent's
output (a DataFrame, a Plotly figure dict, or None) and returns a `CheckOutcome`.

Column names in wrangling and chart outputs are the agent's choice, so aggregate checkers
match by **values**: the group columns must exist by name, and some numeric column must hold
the right number for every group.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dataquorum.evals.tasks import BuiltTask

Output = pd.DataFrame | dict[str, Any] | None


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    passed: bool
    detail: str


def _same(a: Any, b: Any, atol: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-6, atol=atol))
    except (TypeError, ValueError):
        return str(a) == str(b)


def _frame(output: Output, name: str) -> pd.DataFrame | CheckOutcome:
    if isinstance(output, pd.DataFrame):
        return output
    return CheckOutcome(name, False, "no table output")


# ------------------------------------------------------------------ cleaning
def matches_expected(
    b: BuiltTask, output: Output, columns: list[str], atol: float = 1e-6
) -> CheckOutcome:
    name = f"matches_expected{columns}"
    out = _frame(output, name)
    if isinstance(out, CheckOutcome):
        return out
    if len(out) != len(b.expected):
        return CheckOutcome(name, False, f"{len(out)} rows, expected {len(b.expected)}")
    problems = []
    for col in columns:
        if col not in out.columns:
            problems.append(f"{col}: missing")
            continue
        expected = b.expected[col].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(expected) and not pd.api.types.is_numeric_dtype(out[col]):
            problems.append(f"{col}: not numeric ({out[col].dtype})")
            continue
        bad = [
            i
            for i, (x, y) in enumerate(zip(out[col], expected, strict=True))
            if not _same(x, y, atol)
        ]
        if bad:
            i = bad[0]
            problems.append(
                f"{col}: {len(bad)} wrong, e.g. row {i}: {out[col].iloc[i]!r} != {expected.iloc[i]!r}"
            )
    return CheckOutcome(name, not problems, "; ".join(problems) or "all values match")


def row_count(b: BuiltTask, output: Output, n: int | None = None) -> CheckOutcome:
    out = _frame(output, "row_count")
    if isinstance(out, CheckOutcome):
        return out
    want = len(b.expected) if n is None else n
    return CheckOutcome("row_count", len(out) == want, f"{len(out)} rows, expected {want}")


def columns_kept(b: BuiltTask, output: Output, columns: list[str] | None = None) -> CheckOutcome:
    out = _frame(output, "columns_kept")
    if isinstance(out, CheckOutcome):
        return out
    want = columns or [str(c) for c in b.expected.columns]
    missing = [c for c in want if c not in out.columns]
    return CheckOutcome(
        "columns_kept", not missing, f"missing {missing}" if missing else "all kept"
    )


def no_missing(b: BuiltTask, output: Output, columns: list[str]) -> CheckOutcome:
    out = _frame(output, "no_missing")
    if isinstance(out, CheckOutcome):
        return out
    counts = {c: int(out[c].isna().sum()) if c in out else -1 for c in columns}
    bad = {c: n for c, n in counts.items() if n != 0}
    return CheckOutcome("no_missing", not bad, f"missing counts {bad}" if bad else "none missing")


# ------------------------------------------------------------------ wrangling
def _expected_aggregate(
    b: BuiltTask, group_by: list[str], value: str, agg: str, query: str | None
) -> pd.Series:
    source = b.clean.query(query, engine="python") if query else b.clean
    grouped = source.groupby(group_by, dropna=False)[value]
    return grouped.size() if agg == "count" else grouped.agg(agg)


def aggregate(
    b: BuiltTask,
    output: Output,
    group_by: list[str] | str,
    value: str,
    agg: str = "sum",
    query: str | None = None,
    order: str | None = None,
    top: int | None = None,
    atol: float = 0.01,
) -> CheckOutcome:
    keys = [group_by] if isinstance(group_by, str) else group_by
    name = f"aggregate({agg} {value} by {keys})"
    out = _frame(output, name)
    if isinstance(out, CheckOutcome):
        return out
    # Work on a copy: keys are cast to text below and other checkers share this output.
    out = out.reset_index() if not set(keys) <= set(out.columns) else out.copy()
    missing = [k for k in keys if k not in out.columns]
    if missing:
        return CheckOutcome(name, False, f"group columns missing: {missing}")
    expected = _expected_aggregate(b, keys, value, agg, query)
    if top is not None:
        expected = expected.sort_values(ascending=False).head(top)
    exp_frame = expected.rename("__expected__").reset_index()
    for k in keys:
        exp_frame[k], out[k] = exp_frame[k].astype(str), out[k].astype(str)
    if len(out) != len(exp_frame) or set(out[keys].itertuples(index=False)) != set(
        exp_frame[keys].itertuples(index=False)
    ):
        return CheckOutcome(
            name, False, f"groups differ: {len(out)} rows, expected {len(exp_frame)}"
        )
    merged = out.merge(exp_frame, on=keys, how="left")
    for col in [c for c in out.columns if c not in keys and pd.api.types.is_numeric_dtype(out[c])]:
        if all(_same(x, y, atol) for x, y in zip(merged[col], merged["__expected__"], strict=True)):
            if order:
                values = out[col].to_numpy()
                ok = bool(
                    np.all(np.diff(values) <= 0)
                    if order == "desc"
                    else np.all(np.diff(values) >= 0)
                )
                if not ok:
                    return CheckOutcome(name, False, f"values right but not sorted {order} ({col})")
            return CheckOutcome(name, True, f"column {col!r} matches")
    return CheckOutcome(name, False, "no numeric column matches the expected values")


def filter_rows(b: BuiltTask, output: Output, query: str, key: str) -> CheckOutcome:
    name = f"filter_rows({query})"
    out = _frame(output, name)
    if isinstance(out, CheckOutcome):
        return out
    expected = b.clean.query(query, engine="python")
    if key not in out.columns:
        return CheckOutcome(name, False, f"key column {key!r} missing")
    got, want = sorted(map(str, out[key])), sorted(map(str, expected[key]))
    return CheckOutcome(name, got == want, f"{len(got)} rows, expected {len(want)}")


def derived_column(
    b: BuiltTask, output: Output, expression: str, atol: float = 1e-6
) -> CheckOutcome:
    name = f"derived_column({expression})"
    out = _frame(output, name)
    if isinstance(out, CheckOutcome):
        return out
    if len(out) != len(b.clean):
        return CheckOutcome(name, False, f"{len(out)} rows, expected {len(b.clean)}")
    expected = pd.Series(b.clean.eval(expression)).reset_index(drop=True)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) and all(
            _same(x, y, atol) for x, y in zip(out[col], expected, strict=True)
        ):
            return CheckOutcome(name, True, f"column {col!r} matches")
    return CheckOutcome(name, False, "no column matches the expression")


# ------------------------------------------------------------------ figures
def plotly_values(values: Any) -> list[Any]:
    """Decode a Plotly data array: a plain list or Plotly 6+'s base64 typed array."""
    if isinstance(values, dict) and "bdata" in values:
        raw = base64.b64decode(values["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(values["dtype"])).tolist()
    return list(values) if isinstance(values, list | tuple) else []


def _figure(output: Output, name: str) -> dict[str, Any] | CheckOutcome:
    if isinstance(output, dict) and output.get("data"):
        return output
    return CheckOutcome(name, False, "no figure with data")


def figure_type(b: BuiltTask, output: Output, types: list[str]) -> CheckOutcome:
    fig = _figure(output, "figure_type")
    if isinstance(fig, CheckOutcome):
        return fig
    found = sorted({str(t.get("type", "scatter")) for t in fig["data"]})
    return CheckOutcome("figure_type", bool(set(found) & set(types)), f"trace types {found}")


def figure_aggregate(
    b: BuiltTask, output: Output, group_by: str, value: str, agg: str = "sum",
    query: str | None = None, atol: float = 0.01,
) -> CheckOutcome:  # fmt: skip
    name = f"figure_aggregate({agg} {value} by {group_by})"
    fig = _figure(output, name)
    if isinstance(fig, CheckOutcome):
        return fig
    expected = _expected_aggregate(b, [group_by], value, agg, query)
    want = {str(k): v for k, v in expected.items()}
    got: dict[str, Any] = {}
    for trace in fig["data"]:
        cats, nums = plotly_values(trace.get("x")), plotly_values(trace.get("y"))
        if trace.get("orientation") == "h":
            cats, nums = nums, cats
        if trace.get("type") == "histogram" and not nums:  # px.histogram counts raw values
            counts = pd.Series(cats).astype(str).value_counts()
            got.update({str(k): int(v) for k, v in counts.items()})
            continue
        got.update({str(c): v for c, v in zip(cats, nums, strict=False)})
    if set(got) != set(want):
        return CheckOutcome(name, False, f"categories {sorted(got)[:8]} != {sorted(want)[:8]}")
    wrong = [k for k in want if not _same(got[k], want[k], atol)]
    return CheckOutcome(name, not wrong, f"wrong values for {wrong}" if wrong else "all bars match")


def figure_point_count(b: BuiltTask, output: Output, n: int | None = None) -> CheckOutcome:
    fig = _figure(output, "figure_point_count")
    if isinstance(fig, CheckOutcome):
        return fig
    total = sum(len(plotly_values(t.get("x")) or plotly_values(t.get("y"))) for t in fig["data"])
    want = len(b.clean) if n is None else n
    return CheckOutcome("figure_point_count", total == want, f"{total} points, expected {want}")


def no_output(b: BuiltTask, output: Output) -> CheckOutcome:
    """For impossible requests: the right answer is no answer."""
    empty = output is None
    return CheckOutcome("no_output", empty, "no output" if empty else "produced an output")


CHECKERS: dict[str, Callable[..., CheckOutcome]] = {
    "matches_expected": matches_expected,
    "row_count": row_count,
    "columns_kept": columns_kept,
    "no_missing": no_missing,
    "aggregate": aggregate,
    "filter_rows": filter_rows,
    "derived_column": derived_column,
    "figure_type": figure_type,
    "figure_aggregate": figure_aggregate,
    "figure_point_count": figure_point_count,
    "no_output": no_output,
}


def run_checker(
    kind: str, built: BuiltTask, output: Output, params: dict[str, Any]
) -> CheckOutcome:
    if kind not in CHECKERS:
        return CheckOutcome(kind, False, f"unknown checker {kind!r}")
    try:
        return CHECKERS[kind](built, output, **params)
    except Exception as exc:  # a broken checker must not pass
        return CheckOutcome(kind, False, f"checker raised {type(exc).__name__}: {exc}")
