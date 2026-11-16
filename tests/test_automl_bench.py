"""AutoML benchmark helpers: split, scoring (vs scikit-learn) and the report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from dataquorum.ml.automl import positive_label
from dataquorum.ml.automl.benchmark import report, score, split


def test_split_is_seeded_and_stratified() -> None:
    frame = pd.DataFrame({"x": range(100), "y": ["a"] * 80 + ["b"] * 20})
    train, test = split(frame, "y", "binary", seed=3)
    again, _ = split(frame, "y", "binary", seed=3)
    assert len(test) == 25 and train.equals(again)
    assert (test["y"] == "b").sum() == 5  # 25% of each class
    assert set(train["x"]).isdisjoint(set(test["x"]))


def test_scores_match_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.choice(["bad", "good"], 200))
    proba = rng.random(200).round(2)  # rounding makes ties
    assert score("binary", y, pd.DataFrame({"proba": proba}), "good") == pytest.approx(
        roc_auc_score(y == "good", proba)
    )
    labels = pd.Series(rng.choice(["x", "y", "z"], 50))
    guess = pd.DataFrame({"pred": rng.choice(["x", "y", "z"], 50)})
    assert score("multiclass", labels, guess) == pytest.approx(
        accuracy_score(labels, guess["pred"])
    )
    target, pred = pd.Series(rng.random(30)), pd.DataFrame({"pred": rng.random(30)})
    assert score("regression", target, pred) == pytest.approx(
        mean_squared_error(target, pred["pred"]) ** 0.5
    )


def test_positive_label_is_second_sorted() -> None:
    assert positive_label(pd.Series(["good", "bad", "good"])) == "good"
    with pytest.raises(ValueError):
        positive_label(pd.Series(["a", "b", "c"]))


def test_report_ranks_backends(tmp_path: Path) -> None:
    rows = []
    for task, kind, metric, vals in [("t1", "binary", "roc_auc", (0.9, 0.8)), ("t2", "regression", "rmse", (1.0, 2.0)),
                                     ("t3", "binary", "roc_auc", (0.7, 0.6))]:  # fmt: skip
        for backend, v in zip(["good", "bad"], vals, strict=True):
            rows.append({"task": task, "task_type": kind, "metric": metric, "backend": backend, "seed": 0,
                         "score": v, "ok": True, "fit_seconds": 5.0, "wall_seconds": 9.0})  # fmt: skip
    rows.append({"task": "t1", "task_type": "binary", "metric": "roc_auc", "backend": "bad", "seed": 1,
                 "score": None, "ok": False, "error": "oom"})  # fmt: skip
    (tmp_path / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    text = report(tmp_path)
    assert "Average rank over 3 tasks (1 = best): good 1.00, bad 2.00" in text
    assert "Failed runs: {'bad': 1}" in text
