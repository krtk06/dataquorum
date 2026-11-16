"""Seeded fault injection: turn a clean sample into (messy input, expected output).

Each fault damages `messy` and updates `expected` to what a correct cleaner should return
for a request that asks for that repair. Faults run in order with their own seeded RNG,
so a task always builds the same frames.

Fault kinds (column-level unless noted):
- placeholders      numeric -> text with "n/a", "N/A", "", "-" mixed in; expected NaN
                    (or the median of the remaining values with `expect: median`)
- missing           real NaN; expected median / mode (only meaningful with imputation)
- currency_text     numbers -> "$1,234.50"; expected the numbers (rounded to cents)
- thousands_suffix  numbers >= 1000 -> "12.5k"; expected the numbers
- category_variants some labels -> "PRIVATE", "private", " Private "; expected the original
- outliers          some numbers x factor; expected NaN. Exposes `{<column>_threshold}` for
                    the request text (10x the largest real value).
- duplicates        (row-level) exact copies inserted right after originals; expected the
                    sample without them
- parse_numeric_text  no damage; source text numbers (blanks) -> expected floats / NaN
- zero_means_missing  no damage; source zeros that mean "unknown" -> expected NaN
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

Expect = Literal["nan", "median", "mode"]


class FaultSpec(BaseModel):
    kind: str
    column: str | None = None
    frac: float = 0.15
    expect: Expect = "nan"
    factor: float = 1000.0


@dataclass
class Faulted:
    messy: pd.DataFrame
    expected: pd.DataFrame
    facts: dict[str, Any] = field(default_factory=dict)  # values a request may mention


def _pick(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    k = max(1, round(n * frac))
    return np.sort(rng.choice(n, size=min(k, n), replace=False))


def _fmt_number(v: Any) -> Any:
    if pd.isna(v):
        return None
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


def _numeric(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col, errors="coerce").astype(float)


def _fill(expected: pd.DataFrame, column: str, rows: np.ndarray, spec: FaultSpec) -> None:
    """Expected value at damaged rows. Imputation fills every gap in the column, including
    values that were already missing in the source data."""
    keep = np.setdiff1d(np.arange(len(expected)), rows)
    observed = expected[column].iloc[keep].dropna()
    if spec.expect == "median":
        fill: Any = float(observed.median())
    elif spec.expect == "mode":
        fill = observed.mode().sort_values().iloc[0]  # lowest label wins ties, deterministically
    else:
        expected.loc[rows, column] = np.nan
        return
    expected.loc[rows, column] = fill
    expected[column] = expected[column].fillna(fill)


def placeholders(f: Faulted, spec: FaultSpec, rng: np.random.Generator) -> None:
    col = _require(spec)
    rows = _pick(len(f.messy), spec.frac, rng)
    f.expected[col] = _numeric(f.expected[col])
    text = f.messy[col].map(_fmt_number).astype(object)
    tokens = np.array(["n/a", "N/A", "", "-"], dtype=object)
    text.iloc[rows] = rng.choice(tokens, size=len(rows))
    f.messy[col] = text
    _fill(f.expected, col, rows, spec)


def missing(f: Faulted, spec: FaultSpec, rng: np.random.Generator) -> None:
    col = _require(spec)
    if spec.expect == "nan":
        raise ValueError("missing faults need expect: median or mode, else nothing to fix")
    rows = _pick(len(f.messy), spec.frac, rng)
    if pd.api.types.is_numeric_dtype(f.messy[col]):
        f.messy[col] = f.messy[col].astype(float)
        f.expected[col] = f.expected[col].astype(float)
    else:
        f.messy[col] = f.messy[col].astype(object)
    f.messy.loc[rows, col] = None if f.messy[col].dtype == object else np.nan
    _fill(f.expected, col, rows, spec)


def currency_text(f: Faulted, spec: FaultSpec, rng: np.random.Generator) -> None:
    col = _require(spec)
    values = _numeric(f.messy[col]).round(2)
    f.messy[col] = values.map(lambda v: None if pd.isna(v) else f"${v:,.2f}").astype(object)
    f.expected[col] = values


def thousands_suffix(f: Faulted, spec: FaultSpec, rng: np.random.Generator) -> None:
    col = _require(spec)
    values = _numeric(f.messy[col])

    def fmt(v: float) -> str | None:
        if pd.isna(v):
            return None
        return f"{v / 1000:g}k" if abs(v) >= 1000 else f"{v:g}"

    text = values.map(fmt)
    f.messy[col] = text.astype(object)
    # Expected = exactly what the text says, so rounding in "%g" is not counted as an error.
    f.expected[col] = text.map(_parse_suffix).astype(float)


def _parse_suffix(s: Any) -> float:
    if s is None:
        return np.nan
    return float(s[:-1]) * 1000 if s.endswith("k") else float(s)


def category_variants(f: Faulted, spec: FaultSpec, rng: np.random.Generator) -> None:
    col = _require(spec)
    rows = _pick(len(f.messy), spec.frac, rng)
    styles: list[Callable[[str], str]] = [
        str.upper,
        str.lower,
        lambda s: f" {s} ",
        lambda s: f"{s}  ",
    ]
    messy = f.messy[col].astype(object)
    for row in rows:
        value = messy.iloc[row]
        if isinstance(value, str):
            messy.iloc[row] = styles[int(rng.integers(len(styles)))](value)
    f.messy[col] = messy


def outliers(f: Faulted, spec: FaultSpec, rng: np.random.Generator) -> None:
    col = _require(spec)
    values = _numeric(f.messy[col])
    candidates = np.flatnonzero(values.notna().to_numpy() & (values.to_numpy() > 0))
    k = max(1, round(len(values) * spec.frac))
    rows = np.sort(rng.choice(candidates, size=min(k, len(candidates)), replace=False))
    f.facts[f"{col}_threshold"] = f"{float(values.max()) * 10:g}"
    values.iloc[rows] = values.iloc[rows] * spec.factor
    f.messy[col] = values
    f.expected[col] = _numeric(f.expected[col])
    f.expected.loc[rows, col] = np.nan


def duplicates(f: Faulted, spec: FaultSpec, rng: np.random.Generator) -> None:
    rows = set(_pick(len(f.messy), spec.frac, rng).tolist())
    order = [i for i in range(len(f.messy)) for _ in range(2 if i in rows else 1)]
    f.messy = f.messy.iloc[order].reset_index(drop=True)


def parse_numeric_text(f: Faulted, spec: FaultSpec, rng: np.random.Generator) -> None:
    """No damage: the source column is already numbers stored as text (e.g. telco
    TotalCharges with blank strings). Expected: numbers, blanks as NaN."""
    col = _require(spec)
    text = f.messy[col].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
    f.expected[col] = pd.to_numeric(text.replace("", np.nan), errors="coerce").astype(float)


def zero_means_missing(f: Faulted, spec: FaultSpec, rng: np.random.Generator) -> None:
    """No damage: in the source, 0 encodes "not measured" (e.g. diabetes blood pressure)."""
    col = _require(spec)
    f.expected[col] = _numeric(f.expected[col]).replace(0.0, np.nan)


def _require(spec: FaultSpec) -> str:
    if not spec.column:
        raise ValueError(f"fault {spec.kind!r} needs a column")
    return spec.column


FAULTS: dict[str, Callable[[Faulted, FaultSpec, np.random.Generator], None]] = {
    "placeholders": placeholders,
    "missing": missing,
    "currency_text": currency_text,
    "thousands_suffix": thousands_suffix,
    "category_variants": category_variants,
    "outliers": outliers,
    "duplicates": duplicates,
    "parse_numeric_text": parse_numeric_text,
    "zero_means_missing": zero_means_missing,
}


def apply_faults(clean: pd.DataFrame, faults: list[FaultSpec], seed: int) -> Faulted:
    """Apply faults in order. Duplicates are applied last so row-level faults line up."""
    result = Faulted(messy=clean.copy().reset_index(drop=True), expected=clean.copy())
    result.expected = result.expected.reset_index(drop=True)
    ordered = sorted(enumerate(faults), key=lambda p: p[1].kind == "duplicates")
    for index, spec in ordered:
        if spec.kind not in FAULTS:
            raise ValueError(f"unknown fault kind {spec.kind!r}; known: {sorted(FAULTS)}")
        FAULTS[spec.kind](result, spec, np.random.default_rng(seed * 1000 + index))
    return result
