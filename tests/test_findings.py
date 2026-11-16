"""Grounded findings for the EDA agent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from engine_fakes import InProcessExecutor, ScriptedLLM

from dataquorum.agents import EDA
from dataquorum.core.engine import Engine, GeneratedCode, RecommendedSteps
from dataquorum.core.findings import Findings, ungrounded_numbers
from dataquorum.core.statuses import Status

TABLE = pd.DataFrame(
    {
        "column": ["age", "income", "city"],
        "missing_count": [12, 0, 3],
        "missing_pct": [0.2531, 0.0, 0.0625],
    }
)


@pytest.mark.parametrize(
    ("text", "missing"),
    [
        ("age has 12 missing values.", []),
        ("About 25% of age is missing.", []),  # 0.2531 written as a percent
        ("25.3% of age is missing.", []),
        ("6.25% of city values are missing.", []),
        ("Only 3 of 3 columns have gaps.", []),  # small integers are skipped
        ("age is missing 30% of the time.", ["30%"]),  # not in the table
        ("income has 1,500 missing values.", ["1,500"]),
        ("The mean age is 38.4.", ["38.4"]),
    ],
)
def test_numbers_must_come_from_the_table(text: str, missing: list[str]) -> None:
    assert ungrounded_numbers([text], TABLE) == missing


DATA = pd.DataFrame({"age": [30.0, None, 40.0, None], "city": ["a", "b", None, "a"]})
CODE = (
    "import pandas as pd\n"
    "def transform(df):\n"
    "    n = df.isna().sum()\n"
    "    return pd.DataFrame({'column': n.index, 'missing_count': n.values,\n"
    "                         'missing_pct': (n.values / len(df) * 100)})\n"
)


def run_eda(findings: list[str], tmp_path: Path) -> object:
    llm = ScriptedLLM(RecommendedSteps(steps=["count missing per column"]),
                      GeneratedCode(python_code=CODE), Findings(findings=findings))  # fmt: skip
    return (
        Engine(EDA, llm, InProcessExecutor(), runs_dir=tmp_path)
        .start(DATA, "Missing values per column?")
        .result
    )


def test_grounded_findings_become_the_message(tmp_path: Path) -> None:
    result = run_eda(
        ["age is missing in 50% of rows (2 values).", "city has 1 missing value."], tmp_path
    )
    assert result.status is Status.SUCCESS  # type: ignore[attr-defined]
    assert result.message.startswith("- age is missing in 50% of rows")  # type: ignore[attr-defined]


def test_invented_numbers_make_the_run_need_review(tmp_path: Path) -> None:
    result = run_eda(["age is missing in 62.5% of rows."], tmp_path)
    assert result.status is Status.NEEDS_REVIEW  # type: ignore[attr-defined]
    assert "62.5%" in result.error  # type: ignore[attr-defined]
