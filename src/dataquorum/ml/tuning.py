"""Seeded, single-worker Optuna search for a small random forest on numeric training data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)

from dataquorum.ml.correctness import Problem, audit


@dataclass
class TunedForest:
    model: Any
    features: list[str]
    metric: str
    cv_score: float
    trials: int
    params: dict[str, Any]


def validation_folds(
    train: pd.DataFrame,
    target: str,
    *,
    problem: Problem,
    group: str | None,
    time: str | None,
    seed: int,
) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    """Position indices. Equal timestamps stay together and test groups stay unseen."""
    y = train[target]
    if time is not None:
        dates = pd.to_datetime(train[time], utc=True, errors="raise")
        ticks = np.sort(dates.unique())
        splitter = TimeSeriesSplit(n_splits=min(3, len(ticks) - 1))
        folds = []
        for earlier, later in splitter.split(ticks):
            val = np.flatnonzero(dates.isin(ticks[later]))
            mask = dates.isin(ticks[earlier])
            if group:
                mask &= ~train[group].isin(train[group].iloc[val])
            folds.append((np.flatnonzero(mask), val))
    elif group is not None:
        grouped = GroupKFold(n_splits=min(3, train[group].nunique()))
        folds = list(grouped.split(train, y, train[group]))
    elif problem == "classification":
        stratified = StratifiedKFold(
            n_splits=min(3, int(y.value_counts().min())), shuffle=True, random_state=seed
        )
        folds = list(stratified.split(train, y))
    else:
        folds = list(
            KFold(n_splits=min(3, len(train)), shuffle=True, random_state=seed).split(train)
        )
    for fit, val in folds:
        if not len(fit) or not len(val):
            raise ValueError(
                "A validation fold is empty after enforcing group and time constraints"
            )
        if problem == "classification" and (
            y.iloc[fit].nunique() != y.nunique() or y.iloc[val].nunique() != y.nunique()
        ):
            raise ValueError("Each training and validation fold must cover all classes")
    return folds


def tune_forest(
    train: pd.DataFrame,
    target: str,
    *,
    problem: Problem,
    group: str | None = None,
    time: str | None = None,
    seed: int = 7,
    trials: int = 12,
    budget_s: float = 30,
) -> TunedForest:
    """Fit using training rows only. The budget stops new trials; Docker caps total time."""
    import optuna

    if trials < 1 or not np.isfinite(budget_s) or budget_s <= 0:
        raise ValueError("Provide a positive trial count and time budget")
    if "split" in train and target != "split":
        raise ValueError("Pass only training rows and remove the split marker")
    plan = audit(train, target, problem=problem, group=group, time=time)
    risks = plan[plan["kind"].eq("risk")]
    if not risks.empty:
        raise ValueError(f"Resolve these risks before tuning: {risks.to_dict('records')}")
    # Resolve inferred roles exactly as in the audit, including their column names.
    for kind, subject, _value in plan.itertuples(index=False, name=None):
        if kind == "group":
            group = subject
        elif kind == "time":
            time = subject
    columns = [c for c in train.columns if c not in (target, group, time)]
    x, y = train[columns], train[target]
    if (
        not columns
        or not all(pd.api.types.is_numeric_dtype(x[c]) for c in columns)
        or not np.isfinite(x.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Supply complete, finite numeric predictors")
    metric = str(plan.loc[plan["kind"].eq("metric"), "value"].iloc[0])
    scoring = "neg_root_mean_squared_error" if metric == "rmse" else metric
    folds = validation_folds(train, target, problem=problem, group=group, time=time, seed=seed)
    forest = RandomForestRegressor if problem == "regression" else RandomForestClassifier

    def make_model(params: dict[str, Any]) -> Any:
        return forest(n_estimators=32, n_jobs=1, random_state=seed, **params)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
        scores = cross_val_score(
            make_model(params), x, y, cv=folds, scoring=scoring, error_score="raise"
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, timeout=budget_s, n_jobs=1)
    model = make_model(study.best_params).fit(x, y)
    score = -study.best_value if metric == "rmse" else study.best_value
    return TunedForest(model, columns, metric, score, len(study.trials), study.best_params)
