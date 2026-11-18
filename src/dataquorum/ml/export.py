"""Train a forest in Docker and export a verified model artifact."""

from __future__ import annotations

import json
import tempfile
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from dataquorum.ml.correctness import Problem
from dataquorum.ml.report import _fit, _run_forest_code

_PARITY_ROWS = 32


def _export(
    frame: pd.DataFrame,
    target: str,
    problem: Problem,
    split: str,
    group: str | None,
    time: str | None,
    trials: int,
    budget_s: float,
    seed: int,
) -> pd.DataFrame:
    from skops.io import dumps

    tuned, train, test = _fit(frame, target, problem, split, group, time, trials, budget_s, seed)
    model = dumps(tuned.model)
    manifest = {
        "format_version": 1,
        "problem": problem,
        "target": target,
        "features": tuned.features,
        "training_dtypes": [str(train[c].dtype) for c in tuned.features],
        "sklearn_version": version("scikit-learn"),
        "model_sha256": sha256(model).hexdigest(),
        "metric": tuned.metric,
        "cv_score": tuned.cv_score,
        "trials": tuned.trials,
        "params": tuned.params,
        "seed": seed,
    }
    predictions = tuned.model.predict(test[tuned.features].head(_PARITY_ROWS)).tolist()
    return pd.DataFrame(
        [{"manifest": json.dumps(manifest), "model": model, "predictions": json.dumps(predictions)}]
    )


def export_forest(
    frame: pd.DataFrame,
    target: str,
    destination: Path,
    *,
    problem: Problem,
    split: str = "split",
    group: str | None = None,
    time: str | None = None,
    trials: int = 12,
    budget_s: float = 30,
    seed: int = 7,
) -> Path:
    """Fit on train rows and publish the file after a host-side prediction parity check."""
    from dataquorum.ml.artifact import load_forest, pack_forest

    if not np.isfinite(budget_s) or budget_s <= 0:
        raise ValueError("Provide a positive, finite time budget")
    args = (target, problem, split, group, time, trials, budget_s, seed)
    code = (
        "from dataquorum.ml.export import _export\n"
        f"def transform(df):\n    return _export(df, *{args!r})"
    )
    row = _run_forest_code(frame, code, budget_s=budget_s).iloc[0]
    archive = pack_forest(row["manifest"], row["model"])
    expected = json.loads(row["predictions"])

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dq-export-", dir=destination.parent) as temp:
        staged = Path(temp) / "forest.zip"
        staged.write_bytes(archive)
        loaded = load_forest(staged)
        features = list(loaded.metadata.features)
        probe = frame.loc[frame[split].eq("test"), features].head(_PARITY_ROWS)
        actual = loaded.predict(probe).tolist()
        matches = (
            np.allclose(actual, expected, rtol=1e-10, atol=1e-12)
            if problem == "regression"
            else actual == expected
        )
        if not matches:
            raise RuntimeError("Reloaded model predictions differ from sandbox predictions")
        staged.replace(destination)
    return destination
