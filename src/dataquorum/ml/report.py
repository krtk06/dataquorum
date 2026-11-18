"""Run a tuned forest, SHAP, or binary calibration report in the week 11 sandbox."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from dataquorum.ml.correctness import Problem, audit

if TYPE_CHECKING:
    from dataquorum.ml.tuning import TunedForest

Report = Literal["tuning", "shap", "calibration"]


def _fit(
    frame: pd.DataFrame,
    target: str,
    problem: Problem,
    split: str,
    group: str | None,
    time: str | None,
    trials: int,
    budget_s: float,
    seed: int,
) -> tuple[TunedForest, pd.DataFrame, pd.DataFrame]:
    from dataquorum.ml.tuning import tune_forest

    plan = audit(frame, target, problem=problem, split=split, group=group, time=time)
    risks = plan[plan["kind"].eq("risk")]
    if not risks.empty:
        raise ValueError(f"Resolve risks before fitting: {risks.to_dict('records')}")
    train = frame.loc[frame[split].eq("train")].drop(columns=split)
    test = frame.loc[frame[split].eq("test")]
    tuned = tune_forest(
        train,
        target,
        problem=problem,
        group=group,
        time=time,
        seed=seed,
        trials=trials,
        budget_s=budget_s,
    )
    return tuned, train, test


def _report(
    frame: pd.DataFrame,
    target: str,
    problem: Problem,
    report: Report,
    split: str,
    group: str | None,
    time: str | None,
    trials: int,
    budget_s: float,
    seed: int,
) -> pd.DataFrame:
    from dataquorum.ml.calibration import calibration
    from dataquorum.ml.explain import explain_tree

    tuned, _, test = _fit(frame, target, problem, split, group, time, trials, budget_s, seed)
    if report == "tuning":
        return pd.DataFrame(
            [
                {
                    "metric": tuned.metric,
                    "cv_score": tuned.cv_score,
                    "trials": tuned.trials,
                    **tuned.params,
                }
            ]
        )
    x = test[tuned.features]
    if report == "shap":
        return explain_tree(tuned.model, x, seed=seed)
    if report != "calibration" or problem != "classification" or len(tuned.model.classes_) != 2:
        raise ValueError("Calibration needs binary classification")
    positive = tuned.model.classes_[1]
    p = pd.Series(tuned.model.predict_proba(x)[:, 1], index=test.index)
    if not set(test[target].dropna()).issubset(tuned.model.classes_):
        raise ValueError("Holdout labels must belong to the training classes")
    result = calibration(test[target], p, positive=positive)
    return result.curve.assign(brier=result.brier, ece=result.ece, positive_label=positive)


def forest_report(
    frame: pd.DataFrame,
    target: str,
    *,
    problem: Problem,
    report: Report,
    split: str = "split",
    group: str | None = None,
    time: str | None = None,
    trials: int = 12,
    budget_s: float = 30,
    seed: int = 7,
) -> pd.DataFrame:
    """Frame contains a train/test marker. Fits and model objects stay in the container."""
    args = (target, problem, report, split, group, time, trials, budget_s, seed)
    code = (
        "from dataquorum.ml.report import _report\n"
        f"def transform(df):\n    return _report(df, *{args!r})"
    )
    return _run_forest_code(frame, code, budget_s=budget_s)


def _run_forest_code(frame: pd.DataFrame, code: str, *, budget_s: float) -> pd.DataFrame:
    from dataquorum.sandbox.docker_executor import DockerExecutor

    with tempfile.TemporaryDirectory(prefix="dq-forest-") as temp:
        root = Path(temp)
        path = root / "samples.parquet"
        frame.to_parquet(path, index=False)
        result = DockerExecutor(timeout_seconds=budget_s + 60).run(code, path, root)
        if not result.ok or result.output_path is None:
            raise RuntimeError(result.error or "Forest report produced no output")
        return pd.read_parquet(result.output_path)
