"""Feature Engineering agent: turns a table into model-ready features with scikit-learn.

Generated code builds a scikit-learn pipeline (ColumnTransformer with imputers, encoders and
scalers) and returns the transformed table. The checks are about ML hygiene, all blocking:

- every row is kept (feature engineering must not silently drop samples);
- the target column named in the request is present and unchanged;
- the target is not encoded into features (no `class_>50K` one-hot columns: leakage);
- every feature column is numeric and has no missing values.
"""

from __future__ import annotations

import re

import pandas as pd

from dataquorum.agents.checks import UNIQUE_COLUMN_NAMES
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.statuses import CheckResult, Severity
from dataquorum.core.validation import Check, CheckContext

_TARGET = re.compile(
    r"target(?:\s+column)?\s*(?:is|=|:)\s*[`'\"]?([\w\-.>]+?)[`'\"]?(?=[\s,.;)]|$)", re.I
)


def target_column(request: str, columns: list[str]) -> str | None:
    """The target named in the request ("The target column is class."), if it exists."""
    match = _TARGET.search(request)
    if not match:
        return None
    name = match.group(1).rstrip(".")
    return name if name in columns else None


def _result(passed: bool, detail: str) -> CheckResult:
    return CheckResult(name="", passed=passed, severity=Severity.BLOCKING, detail=detail)


def rows_kept(before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext) -> CheckResult:
    return _result(len(after) == len(before), f"{len(before)} -> {len(after)} rows")


def target_unchanged(before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext) -> CheckResult:
    target = target_column(ctx.request, [str(c) for c in before.columns])
    if target is None:
        return _result(True, "no target named in the request")
    if target not in after.columns:
        return _result(False, f"target column {target!r} is missing from the output")
    same = len(after) == len(before) and before[target].reset_index(drop=True).equals(
        after[target].reset_index(drop=True)
    )
    return _result(same, f"target {target!r} " + ("unchanged" if same else "was modified"))


def target_not_encoded(before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext) -> CheckResult:
    target = target_column(ctx.request, [str(c) for c in before.columns])
    if target is None:
        return _result(True, "no target named in the request")
    names = [str(c).split("__", 1)[-1] for c in after.columns]  # sklearn `cat__class_yes`
    leaked = [n for n in names if n != target and n.startswith(f"{target}_")]
    return _result(
        not leaked, f"target encoded into features: {leaked}" if leaked else "no target leakage"
    )


def features_ready(before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext) -> CheckResult:
    target = target_column(ctx.request, [str(c) for c in before.columns])
    features = [c for c in after.columns if str(c) != target]
    text = [str(c) for c in features if not pd.api.types.is_numeric_dtype(after[c])]
    gaps = [str(c) for c in features if after[c].isna().any()]
    problems = ([f"non-numeric: {text}"] if text else []) + (
        [f"missing values: {gaps}"] if gaps else []
    )
    return _result(
        not problems, "; ".join(problems) or f"{len(features)} numeric, complete features"
    )


def missing_not_encoded(
    before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext
) -> CheckResult:
    """SimpleImputer ignores Python None in text columns, so OneHotEncoder makes `city_None`.
    Numeric and complete, so `features_ready` passes it; flag it for review instead."""
    suspect = [str(c) for c in after.columns if str(c).endswith(("_None", "_nan", "_NaN", "_<NA>"))]
    return CheckResult(
        name="", passed=not suspect, severity=Severity.WARNING,
        detail=f"missing values one-hot encoded as a category: {suspect}" if suspect else "none",
    )  # fmt: skip


FEATURE_CHECKS: tuple[Check, ...] = (
    Check("missing_not_encoded", Severity.WARNING, missing_not_encoded),
    Check("rows_kept", Severity.BLOCKING, rows_kept),
    Check("target_unchanged", Severity.BLOCKING, target_unchanged),
    Check("target_not_encoded", Severity.BLOCKING, target_not_encoded),
    Check("features_ready", Severity.BLOCKING, features_ready),
)

FEATURES = AgentSpec(
    name="feature_engineering",
    role="You are a feature-engineering agent that prepares tables for machine learning.",
    recommend_guidance="""Plan model-ready features: which columns are numeric, categorical,
identifiers or the target; how to impute missing values; how to encode categories (one-hot
for low cardinality, grouping rare levels or dropping for very high cardinality); whether to
scale. Leave the target column exactly as it is and never use it to build features. Follow
the request when it names encoders, scalers or columns to drop.""",
    code_guidance="""Use scikit-learn: a ColumnTransformer with Pipelines (SimpleImputer,
OneHotEncoder(handle_unknown="ignore", sparse_output=False), StandardScaler as needed), call
.set_output(transform="pandas") so feature names are kept, fit_transform the feature columns
only, then add the untouched target column back (same name, same order of rows). Every
returned feature must be numeric with no missing values. Keep every row. You may import
sklearn modules.""",
    generic_checks=(),
    extra_checks=(*FEATURE_CHECKS, UNIQUE_COLUMN_NAMES),
)
