"""Smoke test for the AutoGluon image: fit a small synthetic classification task offline.

Run inside the container with this file mounted at /io/in and a writable /io/out.
Prints one JSON line with timing, score and the models AutoGluon trained.
"""

import json
import resource
import time

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=3000, n_features=20, n_informative=8, n_classes=2, random_state=0
)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y
train, test = train_test_split(df, test_size=0.25, random_state=0, stratify=df["target"])
train.to_parquet("/io/out/train.parquet")

start = time.perf_counter()
predictor = TabularPredictor(label="target", path="/io/out/ag_models", verbosity=1).fit(
    pd.read_parquet("/io/out/train.parquet"),
    time_limit=90,
    presets="medium_quality",
)
seconds = time.perf_counter() - start
board = predictor.leaderboard(test, silent=True)

print(
    json.dumps(
        {
            "fit_seconds": round(seconds, 1),
            "test_accuracy": round(float(predictor.evaluate(test, silent=True)["accuracy"]), 4),
            "best_model": predictor.model_best,
            "models_trained": board["model"].tolist(),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
            "peak_children_rss_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024,
        }
    )
)
