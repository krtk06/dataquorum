"""Reliability curve, Brier score and expected calibration error for binary forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Calibration:
    curve: pd.DataFrame
    brier: float
    ece: float


def calibration(
    y: pd.Series, probability: pd.Series, *, positive: Any, bins: int = 10
) -> Calibration:
    """Equal-width bins, including 1 in the final bin. ECE weights by bin population."""
    if y.empty or not y.index.equals(probability.index) or y.isna().any() or bins < 1:
        raise ValueError("Provide aligned, nonempty observations and positive bin count")
    if len(set(y) | {positive}) > 2:
        raise ValueError("Observed labels and the positive label must form a binary outcome")
    p = probability.to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    observed = y.eq(positive).to_numpy(dtype=float)
    rows = pd.DataFrame(
        {
            "bin": np.minimum((p * bins).astype(int), bins - 1),
            "probability": p,
            "observed": observed,
        }
    )
    curve = rows.groupby("bin", as_index=False).agg(
        count=("observed", "size"),
        mean_probability=("probability", "mean"),
        fraction_positive=("observed", "mean"),
    )
    error = (curve["mean_probability"] - curve["fraction_positive"]).abs()
    return Calibration(
        curve, float(np.mean((p - observed) ** 2)), float(np.average(error, weights=curve["count"]))
    )
