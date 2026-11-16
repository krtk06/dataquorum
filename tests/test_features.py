"""Feature engineering agent: target parsing and ML-hygiene checks, and a sklearn run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from engine_fakes import InProcessExecutor, ScriptedLLM

from dataquorum.agents import FEATURES
from dataquorum.agents.features import target_column
from dataquorum.core.engine import Engine, GeneratedCode, RecommendedSteps
from dataquorum.core.statuses import Status
from dataquorum.core.validation import CheckContext, run_checks

RAW = pd.DataFrame(
    {
        "age": [30.0, np.nan, 50.0, 41.0],
        "city": ["a", "b", "a", None],
        "class": ["yes", "no", "no", "yes"],
    }
)
REQUEST = "Prepare features for a model. The target column is class."


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("The target column is class.", "class"),
        ("target: class", "class"),
        ("Target is `class`, predict it", "class"),
        ("Predict churn", None),
        ("The target column is missing_col.", None),
    ],
)
def test_target_column(request_text: str, expected: str | None) -> None:
    assert target_column(request_text, list(RAW.columns)) == expected


def failed(after: pd.DataFrame) -> list[str]:
    results = run_checks(FEATURES.checks, RAW, after, CheckContext(request=REQUEST))
    return [r.name for r in results if not r.passed]


def test_good_features_pass() -> None:
    after = pd.DataFrame(
        {"age": [0.1, 0.0, 1.2, -1.3], "city_a": [1.0, 0, 1, 0], "class": RAW["class"]}
    )
    assert failed(after) == []


@pytest.mark.parametrize(
    ("after", "expect"),
    [
        (
            pd.DataFrame({"age": [1.0, 2.0], "class": ["yes", "no"]}),
            ["rows_kept", "target_unchanged"],
        ),
        (
            pd.DataFrame({"age": [1.0, 2, 3, 4], "class": ["no", "no", "no", "yes"]}),
            ["target_unchanged"],
        ),
        (
            pd.DataFrame(
                {"age": [1.0, 2, 3, 4], "class_yes": [1.0, 0, 0, 1], "class": RAW["class"]}
            ),
            ["target_not_encoded"],
        ),
        (
            pd.DataFrame(
                {"age": [1.0, np.nan, 3, 4], "city": ["a", "b", "a", "c"], "class": RAW["class"]}
            ),
            ["features_ready"],
        ),
        (pd.DataFrame({"age": [1.0, 2, 3, 4]}), ["target_unchanged"]),
    ],
)
def test_hygiene_failures(after: pd.DataFrame, expect: list[str]) -> None:
    assert failed(after) == expect


SKLEARN_CODE = """
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def transform(df):
    # SimpleImputer ignores Python None in text columns; make it a real NaN first.
    X = df.drop(columns=["class"]).replace({None: np.nan})
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["age"]),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["city"]),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    features = pre.fit_transform(X)
    features["class"] = df["class"].values
    return features
"""


def test_feature_engineering_run_with_sklearn(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        RecommendedSteps(steps=["impute", "scale age", "one-hot city"]),
        GeneratedCode(python_code=SKLEARN_CODE),
    )
    result = (
        Engine(FEATURES, llm, InProcessExecutor(), runs_dir=tmp_path).start(RAW, REQUEST).result
    )
    assert result is not None
    assert result.status is Status.SUCCESS, result.error
    out = result.output.load()  # type: ignore[union-attr]
    assert list(out.columns) == ["age", "city_a", "city_b", "class"]
    assert abs(out["age"].mean()) < 1e-9 and out.isna().sum().sum() == 0


def test_missing_values_encoded_as_a_category_is_flagged() -> None:
    after = pd.DataFrame(
        {
            "age": [1.0, 2, 3, 4],
            "city_a": [1.0, 0, 1, 0],
            "city_None": [0.0, 0, 0, 1],
            "class": RAW["class"],
        }
    )
    assert failed(after) == ["missing_not_encoded"]
