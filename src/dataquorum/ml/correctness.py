"""Deterministic pre-fit audit. Each row describes a recommendation or a risk to review."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

Problem = Literal["classification", "regression"]
COLUMNS = ["kind", "subject", "value"]
_ID = re.compile(r"(?:^id$|_id$|^uuid$|^identifier$)", re.I)
_POST = re.compile(r"(?:^|_)(?:post_outcome|after_outcome|final_outcome|post_event)(?:_|$)", re.I)


def _target_proxy(x: pd.Series, y: pd.Series, problem: Problem) -> bool:
    mask = x.notna() & y.notna()
    x, y = x[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    if x.nunique() < 2 or y.nunique() < 2:
        return False
    if x.astype(object).eq(y.astype(object)).all():
        return True
    if len(x) < 20:
        return False
    if problem == "classification":
        # Balanced recall prevents a majority-only predictor passing on an imbalanced target.
        # Exclude high-cardinality lookup tables, which can memorize individual rows.
        if x.nunique() <= min(50, len(x) // 5):
            counts = pd.crosstab(x, y)
            predictions = x.map(counts.idxmax(axis=1))
            recall = predictions.eq(y).groupby(y, observed=True).mean()
            if recall.mean() >= 0.99:
                return True
        if y.nunique() != 2:
            return False
        y = pd.Series(pd.factorize(y)[0], index=y.index)
        if pd.api.types.is_numeric_dtype(x):
            # Check a single numeric threshold in either direction. Evaluate only between
            # distinct values so equal measurements cannot be split using their labels.
            counts = pd.crosstab(x, y)
            cumulative = counts.cumsum() / counts.sum()
            gap = (cumulative[0] - cumulative[1]).abs().max()
            return bool((1 + gap) / 2 >= 0.99)
    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        return bool(abs(x.astype(float).corr(y.astype(float))) >= 0.995)
    return False


def _role(df: pd.DataFrame, explicit: str | None, candidates: list[str], role: str) -> str | None:
    if explicit:
        if explicit not in df:
            raise ValueError(f"Unknown {role} column {explicit!r}")
        return explicit
    if len(candidates) > 1:
        raise ValueError(f"Specify a {role} column among {candidates}")
    return next(iter(candidates), None)


def audit(
    frame: pd.DataFrame,
    target: str,
    *,
    problem: Problem,
    group: str | None = None,
    time: str | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """Use training labels only. A split column, when present, must mark train/test rows.

    Repeated *_id columns and datetime columns supply default group and time roles.
    Explicit roles also support other names and ISO timestamp strings.
    """
    if target not in frame or not frame.columns.is_unique or frame.empty:
        raise ValueError("Provide a nonempty table with unique columns and the named target")
    if problem not in ("classification", "regression"):
        raise ValueError("Specify classification or regression")
    if split is None and "split" in frame and target != "split":
        split = "split"
    train = frame
    test = frame.iloc[:0]
    if split is not None:
        if (
            split not in frame
            or set(frame[split].dropna().unique()) != {"train", "test"}
            or frame[split].isna().any()
        ):
            raise ValueError("The split column must contain both train and test labels only")
        train, test = frame.loc[frame[split].eq("train")], frame.loc[frame[split].eq("test")]
    if not all(isinstance(c, str) for c in train.columns):
        raise ValueError("Use string column names for the audit")
    cols = [str(c) for c in train.columns if c not in (target, split)]
    group = _role(
        train,
        group,
        [c for c in cols if c != time and _ID.search(c) and train[c].nunique() < 0.95 * len(train)],
        "group",
    )
    time = _role(
        train,
        time,
        [c for c in cols if c != group and pd.api.types.is_datetime64_any_dtype(train[c])],
        "time",
    )
    roles = [c for c in (target, group, time, split) if c is not None]
    if len(set(roles)) != len(roles):
        raise ValueError("Target, group, time and split must use distinct columns")
    y = train[target]
    if y.isna().any() or y.nunique() < 2:
        raise ValueError("Training targets must be complete and contain at least two values")
    if problem == "regression" and (
        not pd.api.types.is_numeric_dtype(y) or not np.isfinite(y).all()
    ):
        raise ValueError("Regression needs finite numeric training targets")
    risks: list[tuple[str, str, str]] = []

    def flag(subject: str, reason: str) -> None:
        risks.append(("risk", subject, reason))

    features = [c for c in cols if c not in (group, time)]
    if not features:
        raise ValueError("Provide at least one predictor in addition to the target and split roles")
    for col in features:
        x = train[col]
        if _ID.search(col) and x.nunique() >= 0.95 * len(train):
            flag(col, "identifier")
        if _POST.search(col):
            flag(col, "post_outcome")
        if _target_proxy(x, y, problem):
            flag(col, "target_proxy")
    if not test.empty and features:
        # Exclude administrative split metadata and labels, so relabelled copies still match.
        train_keys = pd.MultiIndex.from_frame(train[cols])
        if pd.MultiIndex.from_frame(test[cols]).isin(train_keys).any():
            flag(str(split), "row_overlap")
    if group is not None:
        if frame[group].isna().any() or train[group].nunique() < 2:
            raise ValueError(
                "Group validation needs complete groups and at least two training groups"
            )
        if test[group].isin(train[group]).any():
            flag(group, "group_overlap")
    if time is not None:
        dates = pd.to_datetime(frame[time], utc=True, errors="raise")
        if dates.isna().any():
            raise ValueError("Time validation needs complete timestamps")
        training_dates = dates if split is None else dates[frame[split].eq("train")]
        if training_dates.nunique() < 3:
            raise ValueError("Time validation needs at least three distinct training timestamps")
        if split and training_dates.max() >= dates[frame[split].eq("test")].min():
            flag(time, "time_overlap")
    if group and time:
        strategy = "grouped_forward_split"
    elif time:
        strategy = "TimeSeriesSplit"
    elif group:
        strategy = "GroupKFold"
    elif problem == "classification":
        strategy = "StratifiedKFold"
    else:
        strategy = "KFold"
    if problem == "classification" and strategy == "StratifiedKFold" and y.value_counts().min() < 2:
        raise ValueError("Stratification requires at least two examples of each class")
    metric = "rmse"
    if problem == "classification":
        metric = "balanced_accuracy" if y.value_counts(normalize=True).min() < 0.1 else "accuracy"
    rows = [("validation", target, strategy), ("metric", target, metric)]
    if group:
        rows.append(("group", group, "hold_out_groups"))
    if time:
        rows.append(("time", time, "forward_only"))
    return pd.DataFrame([*rows, *risks], columns=COLUMNS)
