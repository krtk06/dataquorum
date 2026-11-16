"""Runs INSIDE an AutoML container (no dataquorum install there): fit one backend, predict.

Input  /io/in/job.json   {backend, target, task_type, budget_s, seed, metric, cpus, mem_gb}
       /io/in/train.parquet (with target), /io/in/test.parquet (without target)
Output /io/out/predictions.parquet  column `pred`; for binary tasks also `proba` (probability
                                    of `positive`, the second label in sorted order)
       /io/out/meta.json            backend version, best model, model count, fit seconds

Scoring is done on the host from the predictions, so no backend grades itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

IN, OUT = Path("/io/in"), Path("/io/out")


def run_flaml(job: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    import flaml
    from flaml import AutoML

    target, task = job["target"], job["task_type"]
    automl = AutoML()
    metric = {"binary": "roc_auc", "multiclass": "accuracy", "regression": "rmse"}[task]
    automl.fit(
        X_train=train.drop(columns=[target]),
        y_train=train[target],
        task="regression" if task == "regression" else "classification",
        time_budget=job["budget_s"],
        seed=job["seed"],
        metric=metric,
        n_jobs=job["cpus"],
        verbose=0,
    )
    out = {"pred": automl.predict(test)}
    if task == "binary":
        classes = list(automl.classes_)
        out["proba"] = automl.predict_proba(test)[:, classes.index(job["positive"])]
    return {
        "frame": pd.DataFrame(out),
        "version": flaml.__version__,
        "best_model": automl.best_estimator,
        "n_models": len(automl.config_history),
    }


def run_h2o(job: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    import h2o
    from h2o.automl import H2OAutoML

    target, task = job["target"], job["task_type"]
    Path("/tmp/h2o").mkdir(parents=True, exist_ok=True)
    h2o.init(
        nthreads=job["cpus"],
        max_mem_size=f"{job['mem_gb']}G",
        ice_root="/tmp/h2o",
        verbose=False,
        port=54321,
    )
    h2o.no_progress()
    try:
        frame = h2o.H2OFrame(train)
        if task != "regression":
            frame[target] = frame[target].asfactor()
        sort = {"binary": "AUC", "multiclass": "mean_per_class_error", "regression": "RMSE"}[task]
        aml = H2OAutoML(
            max_runtime_secs=job["budget_s"], seed=job["seed"], sort_metric=sort, verbosity=None
        )
        aml.train(y=target, training_frame=frame)
        preds = aml.leader.predict(h2o.H2OFrame(test)).as_data_frame(use_multi_thread=False)
        out = {"pred": preds["predict"].to_numpy()}
        if task == "binary":
            out["proba"] = preds[str(job["positive"])].to_numpy()
        n_models = int(aml.leaderboard.nrows)
        return {
            "frame": pd.DataFrame(out),
            "version": h2o.__version__,
            "best_model": aml.leader.model_id,
            "n_models": n_models,
        }
    finally:
        h2o.cluster().shutdown()


def run_autogluon(job: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    import autogluon.tabular as agt
    from autogluon.tabular import TabularPredictor

    target, task = job["target"], job["task_type"]
    metric = {
        "binary": "roc_auc",
        "multiclass": "accuracy",
        "regression": "root_mean_squared_error",
    }[task]
    predictor = TabularPredictor(
        label=target, eval_metric=metric, path="/tmp/ag", verbosity=0, problem_type=task
    )
    # AutoGluon 1.6 has no fit-level seed: every model uses its default seed 0, so for this
    # backend the benchmark seed only changes the train/test split (recorded in meta).
    predictor.fit(
        train,
        time_limit=job["budget_s"],
        presets="medium_quality",
        ag_args_fit={"num_cpus": job["cpus"]},
    )
    out = {"pred": predictor.predict(test).to_numpy()}
    if task == "binary":
        out["proba"] = predictor.predict_proba(test)[job["positive"]].to_numpy()
    return {
        "frame": pd.DataFrame(out),
        "version": agt.__version__,
        "best_model": predictor.model_best,
        "n_models": len(predictor.model_names()),
        "seed_applied": "split only (model seeds fixed at 0)",
    }


BACKENDS = {"flaml": run_flaml, "h2o": run_h2o, "autogluon": run_autogluon}


def main() -> None:
    job = json.loads((IN / "job.json").read_text(encoding="utf-8"))
    train, test = pd.read_parquet(IN / "train.parquet"), pd.read_parquet(IN / "test.parquet")
    start = time.perf_counter()
    result = BACKENDS[job["backend"]](job, train, test)
    seconds = time.perf_counter() - start
    result["frame"].to_parquet(OUT / "predictions.parquet", index=False)
    meta = {k: v for k, v in result.items() if k != "frame"} | {"fit_seconds": round(seconds, 1)}
    (OUT / "meta.json").write_text(json.dumps(meta, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
