"""Global tree SHAP importance for a bounded, seeded sample of evaluation rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def explain_tree(
    model: Any, features: pd.DataFrame, *, max_rows: int = 128, seed: int = 7
) -> pd.DataFrame:
    import shap

    if features.empty or max_rows < 1:
        raise ValueError("Provide evaluation rows and a positive sample limit")
    x = features.sample(min(len(features), max_rows), random_state=seed)
    values = np.asarray(shap.TreeExplainer(model)(x).values)
    if values.ndim not in (2, 3) or values.shape[:2] != x.shape or not np.isfinite(values).all():
        raise ValueError(f"Unexpected SHAP values with shape {values.shape}")
    # Multi-output classifiers have a final class axis. Average magnitudes across classes.
    importance = np.abs(values).mean(axis=(0, 2) if values.ndim == 3 else 0)
    report = pd.DataFrame({"feature": list(x.columns), "mean_abs_shap": importance, "rows": len(x)})
    return report.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
