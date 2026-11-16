import numpy as np
import pandas as pd
import pytest

from dataquorum.core.statuses import CheckResult, Severity
from dataquorum.core.validation import (
    GENERIC_DATAFRAME_CHECKS,
    Check,
    CheckContext,
    missing_like,
    run_checks,
)

BEFORE = pd.DataFrame(
    {"age": [30.0, np.nan, 40.0, 50.0], "income": ["$1", "n/a", "", "$4"], "city": list("abcd")}
)
CTX = CheckContext(request="clean the data")


def results(after: pd.DataFrame, ctx: CheckContext = CTX) -> dict[str, CheckResult]:
    return {r.name: r for r in run_checks(GENERIC_DATAFRAME_CHECKS, BEFORE, after, ctx)}


def test_good_cleaning_passes_everything() -> None:
    after = BEFORE.assign(income=[1.0, np.nan, np.nan, 4.0])  # "n/a" and "" -> NaN is not a loss
    assert all(r.passed for r in results(after).values())


def test_empty_output_fails_blocking_check() -> None:
    r = results(BEFORE.iloc[0:0])["has_rows_and_columns"]
    assert not r.passed and r.severity is Severity.BLOCKING


def test_row_loss_warns_unless_requested() -> None:
    after = BEFORE.iloc[:1]
    assert not results(after)["row_loss"].passed
    asked = CheckContext(request="filter to rows where age > 35")
    assert results(after, asked)["row_loss"].passed


def test_dropped_column_warns_unless_named_in_request() -> None:
    after = BEFORE.drop(columns=["city"])
    assert not results(after)["columns_dropped"].passed
    asked = CheckContext(request="drop the city column")
    assert results(after, asked)["columns_dropped"].passed
    only_named = CheckContext(request="title-case the city column")
    assert not results(after, only_named)["columns_dropped"].passed


def test_real_missing_value_increase_warns() -> None:
    after = BEFORE.assign(age=[np.nan, np.nan, 40.0, 50.0])
    r = results(after)["missing_values_increased"]
    assert not r.passed and "age" in r.detail


def test_numeric_turned_to_text_warns() -> None:
    after = BEFORE.assign(age=pd.Series(["30", None, "40", "50"], dtype=object))
    assert not results(after)["numeric_became_text"].passed


def test_a_check_that_raises_is_a_failed_check() -> None:
    def broken(before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext) -> CheckResult:
        raise RuntimeError("bug in check")

    [r] = run_checks([Check("broken", Severity.BLOCKING, broken)], BEFORE, BEFORE, CTX)
    assert not r.passed and r.severity is Severity.BLOCKING
    assert "RuntimeError" in r.detail


@pytest.mark.parametrize(
    ("values", "expected"),
    [(["a", None, "n/a", " NULL ", ""], 4), ([1.0, None, 3.0], 1), (["x", "y"], 0)],
)
def test_missing_like(values: list[object], expected: int) -> None:
    assert missing_like(pd.Series(values)) == expected
