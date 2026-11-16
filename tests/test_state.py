from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from dataquorum.core.cost import Usage
from dataquorum.core.state import (
    AgentResult,
    DataFrameOutput,
    FigureOutput,
    ModelOutput,
    TextOutput,
)
from dataquorum.core.statuses import CheckResult, Severity, Status

ROW_LOSS = CheckResult(name="row_loss", passed=False, severity=Severity.WARNING, detail="60%")
NO_ROWS = CheckResult(name="has_rows", passed=False, severity=Severity.BLOCKING)


@pytest.fixture
def table(tmp_path: Path) -> DataFrameOutput:
    df = pd.DataFrame({"city": ["Boston", "Chicago"], "income": [1200.0, None]})
    return DataFrameOutput.save(df, tmp_path / "out.parquet")


@pytest.fixture
def empty_table(tmp_path: Path) -> DataFrameOutput:
    # What the original returned for "revenue by region" on data with no such columns.
    df = pd.DataFrame({"sales_region": [], "product_category": [], "revenue": []})
    return DataFrameOutput.save(df, tmp_path / "empty.parquet")


def test_dataframe_output_round_trip(table: DataFrameOutput) -> None:
    assert (table.rows, table.columns) == (2, ["city", "income"])
    assert table.dtypes["income"] == "float64"
    loaded = table.load()
    assert loaded["city"].tolist() == ["Boston", "Chicago"]
    assert pd.isna(loaded["income"][1])


def test_clean_run_is_success(table: DataFrameOutput) -> None:
    result = AgentResult.from_run(agent="cleaning", output=table, code="def clean(df): ...")
    assert result.status is Status.SUCCESS
    assert result.error is None
    assert "all checks passed" in result.message


def test_empty_table_is_failed_not_success(empty_table: DataFrameOutput) -> None:
    """Regression test for the original's silent false success (bug log #6)."""
    result = AgentResult.from_run(agent="wrangling", output=empty_table)
    assert result.status is Status.FAILED
    assert result.error == "The output is empty."


def test_warning_gives_needs_review(table: DataFrameOutput) -> None:
    result = AgentResult.from_run(agent="cleaning", output=table, checks=[ROW_LOSS])
    assert result.status is Status.NEEDS_REVIEW
    assert result.error is not None and "row_loss (60%)" in result.error


def test_retries_exhausted_is_failed_and_keeps_usage() -> None:
    """Regression test for bug log #5: exhausted retries must not look finished."""
    usage = Usage(models=["deepseek-v4.1-flash"], calls=4, retries=3, allowance_usd=0.004)
    result = AgentResult.from_run(
        agent="cleaning", output=None, error="KeyError: 'income' after 3 retries", usage=usage
    )
    assert result.status is Status.FAILED
    assert result.usage.retries == 3
    assert "KeyError" in result.message


@pytest.mark.parametrize(
    "output",
    [
        FigureOutput(figure={"data": [], "layout": {}}),
        TextOutput(text="   "),
        ModelOutput(uri="", framework="sklearn"),
    ],
)
def test_empty_outputs_of_every_kind_fail(output: FigureOutput | TextOutput | ModelOutput) -> None:
    assert AgentResult.from_run(agent="x", output=output).status is Status.FAILED


def test_cannot_construct_success_with_empty_output(empty_table: DataFrameOutput) -> None:
    with pytest.raises(ValidationError, match="non-empty output"):
        AgentResult(agent="x", status=Status.SUCCESS, output=empty_table)


def test_cannot_construct_success_with_failed_blocking_check(table: DataFrameOutput) -> None:
    with pytest.raises(ValidationError, match="blocking"):
        AgentResult(agent="x", status=Status.SUCCESS, output=table, validation=[NO_ROWS])


def test_cannot_construct_success_with_error(table: DataFrameOutput) -> None:
    with pytest.raises(ValidationError, match="cannot carry an error"):
        AgentResult(agent="x", status=Status.SUCCESS, output=table, error="boom")


def test_failed_needs_an_error() -> None:
    with pytest.raises(ValidationError, match="needs an error"):
        AgentResult(agent="x", status=Status.FAILED)


def test_needs_review_needs_an_output() -> None:
    with pytest.raises(ValidationError, match="output to review"):
        AgentResult(agent="x", status=Status.NEEDS_REVIEW, error="check this")


def test_json_round_trip_keeps_output_kind(table: DataFrameOutput) -> None:
    original = AgentResult.from_run(
        agent="cleaning",
        output=table,
        checks=[ROW_LOSS],
        steps=["parse income"],
        usage=Usage(models=["m"], calls=1),
        parent_run_id="parent",
    )
    restored = AgentResult.model_validate_json(original.model_dump_json())
    assert restored == original
    assert isinstance(restored.output, DataFrameOutput)
