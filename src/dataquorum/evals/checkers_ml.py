"""Checkers for the week-9 agents: EDA summaries and feature engineering.

EDA tables are free-form, so these checkers look for values, not column names: a column that
lists the dataset's column names next to a numeric column with the right statistic, or a cell
holding the right correlation. Feature checkers test the transformed table directly: target
untouched, everything else numeric and complete, categories one-hot encoded with correct 0/1
values, numeric columns standardised (mean 0, std 1) or min-max scaled to [0, 1].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pandas as pd

from dataquorum.evals.checkers import CheckOutcome, Output, _frame, _same, group_key
from dataquorum.evals.tasks import BuiltTask

STATS: dict[str, Callable[[pd.Series], float]] = {
    "missing": lambda s: float(s.isna().sum()),
    "missing_pct": lambda s: float(s.isna().mean() * 100),
    "mean": lambda s: float(s.mean()),
    "median": lambda s: float(s.median()),
    "std": lambda s: float(s.std()),
    "nunique": lambda s: float(s.nunique(dropna=True)),
}


def per_column_stat(
    b: BuiltTask, output: Output, stat: str, columns: list[str], atol: float = 0.01
) -> CheckOutcome:
    name = f"per_column_stat({stat})"
    out = _frame(output, name)
    if isinstance(out, CheckOutcome):
        return out
    source = b.messy if isinstance(b.messy, pd.DataFrame) else b.clean
    want = {c: STATS[stat](source[c]) for c in columns}
    out = out.reset_index()
    for label_col in out.columns:
        labels = [group_key(v) for v in out[label_col]]
        if not set(columns) <= set(labels):
            continue
        rows = {lab: i for i, lab in enumerate(labels)}
        for col in [
            c for c in out.columns if c != label_col and pd.api.types.is_numeric_dtype(out[c])
        ]:
            if all(_same(out[col].iloc[rows[c]], want[c], atol) for c in columns):
                return CheckOutcome(name, True, f"{stat} in {col!r}, columns in {label_col!r}")
        return CheckOutcome(
            name,
            False,
            f"columns listed in {label_col!r} but no column has the right {stat}: {want}",
        )
    return CheckOutcome(name, False, f"no column lists all of {columns}")


def correlation(
    b: BuiltTask, output: Output, x: str, y: str, method: str = "pearson", atol: float = 0.005
) -> CheckOutcome:
    name = f"correlation({x}, {y})"
    out = _frame(output, name)
    if isinstance(out, CheckOutcome):
        return out
    source = b.messy if isinstance(b.messy, pd.DataFrame) else b.clean
    want = float(source[x].corr(source[y], method=cast(Any, method)))
    cells = out.select_dtypes("number").to_numpy().ravel()
    hit = bool(np.any(np.abs(cells - want) <= atol))
    return CheckOutcome(
        name,
        hit,
        f"expected {want:.4f}"
        + ("" if hit else f"; table numbers {np.round(cells[:8], 4).tolist()}"),
    )


def column_unchanged(b: BuiltTask, output: Output, column: str) -> CheckOutcome:
    name = f"column_unchanged({column})"
    out = _frame(output, name)
    if isinstance(out, CheckOutcome):
        return out
    source = b.messy if isinstance(b.messy, pd.DataFrame) else b.clean
    if column not in out.columns:
        return CheckOutcome(name, False, "column missing")
    same = len(out) == len(source) and all(
        _same(x, y, 0.0) for x, y in zip(out[column], source[column], strict=True)
    )
    return CheckOutcome(name, same, "unchanged" if same else "values changed")


def all_numeric(
    b: BuiltTask, output: Output, except_columns: list[str] | None = None
) -> CheckOutcome:
    out = _frame(output, "all_numeric")
    if isinstance(out, CheckOutcome):
        return out
    skip = set(except_columns or [])
    text = [
        str(c) for c in out.columns if c not in skip and not pd.api.types.is_numeric_dtype(out[c])
    ]
    gaps = [str(c) for c in out.columns if c not in skip and out[c].isna().any()]
    ok = not text and not gaps
    return CheckOutcome(
        "all_numeric",
        ok,
        "all numeric and complete" if ok else f"non-numeric {text}, missing {gaps}",
    )


def onehot(b: BuiltTask, output: Output, column: str) -> CheckOutcome:
    """One 0/1 column per category of `column` (name contains the category), matching rows."""
    name = f"onehot({column})"
    out = _frame(output, name)
    if isinstance(out, CheckOutcome):
        return out
    source = b.messy if isinstance(b.messy, pd.DataFrame) else b.clean
    if len(out) != len(source):
        return CheckOutcome(name, False, f"{len(out)} rows, expected {len(source)}")
    problems = []
    for category in sorted(str(v) for v in source[column].dropna().unique()):
        matches = [
            c
            for c in out.columns
            if str(c) != column and str(c).endswith(category) and str(c).startswith(column)
        ]
        if not matches:
            problems.append(f"no column for {category!r}")
            continue
        want = (source[column].astype(str) == category).astype(float).to_numpy()
        if not np.allclose(out[matches[0]].astype(float).to_numpy(), want):
            problems.append(f"{matches[0]!r} has wrong 0/1 values")
    return CheckOutcome(
        name, not problems, "; ".join(problems[:4]) or "one 0/1 column per category"
    )


def _scaled(b: BuiltTask, output: Output, columns: list[str], kind: str) -> CheckOutcome:
    name = f"{kind}({columns})"
    out = _frame(output, name)
    if isinstance(out, CheckOutcome):
        return out
    problems = []
    for col in columns:
        if col not in out.columns:
            problems.append(f"{col} missing")
            continue
        values = out[col].astype(float)
        if kind == "standardized":
            ok = abs(values.mean()) < 1e-6 and abs(values.std(ddof=0) - 1) < 0.01
            stat = f"mean {values.mean():.3g}, std {values.std(ddof=0):.3g}"
        else:
            ok = abs(values.min()) < 1e-9 and abs(values.max() - 1) < 1e-9
            stat = f"min {values.min():.3g}, max {values.max():.3g}"
        if not ok:
            problems.append(f"{col}: {stat}")
    return CheckOutcome(name, not problems, "; ".join(problems) or "scaled as asked")


def standardized(b: BuiltTask, output: Output, columns: list[str]) -> CheckOutcome:
    return _scaled(b, output, columns, "standardized")


def minmax(b: BuiltTask, output: Output, columns: list[str]) -> CheckOutcome:
    return _scaled(b, output, columns, "minmax")


ML_CHECKERS: dict[str, Callable[..., CheckOutcome]] = {
    "per_column_stat": per_column_stat,
    "correlation": correlation,
    "column_unchanged": column_unchanged,
    "all_numeric": all_numeric,
    "onehot": onehot,
    "standardized": standardized,
    "minmax": minmax,
}


def all_checkers() -> dict[str, Callable[..., Any]]:
    from dataquorum.evals.checkers import CHECKERS

    return {**CHECKERS, **ML_CHECKERS}
