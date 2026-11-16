"""Checks shared by several agents (on top of the generic ones in core/validation.py)."""

from __future__ import annotations

import pandas as pd

from dataquorum.core.statuses import CheckResult, Severity
from dataquorum.core.validation import Check, CheckContext


def _result(passed: bool, detail: str) -> CheckResult:
    return CheckResult(name="", passed=passed, severity=Severity.WARNING, detail=detail)


def unique_column_names(
    before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext
) -> CheckResult:
    dupes = sorted({str(c) for c in after.columns if list(after.columns).count(c) > 1})
    return _result(not dupes, f"duplicate column names: {dupes}" if dupes else "all unique")


def no_all_null_columns(
    before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext
) -> CheckResult:
    empty = [str(c) for c in after.columns if len(after) and after[c].isna().all()]
    return _result(not empty, f"columns with only missing values: {empty}" if empty else "none")


def duplicate_rows_not_increased(
    before: pd.DataFrame, after: pd.DataFrame, ctx: CheckContext
) -> CheckResult:
    b, a = int(before.duplicated().sum()), int(after.duplicated().sum())
    return _result(a <= b, f"duplicate rows {b} -> {a}")


UNIQUE_COLUMN_NAMES = Check("unique_column_names", Severity.BLOCKING, unique_column_names)
NO_ALL_NULL_COLUMNS = Check("no_all_null_columns", Severity.WARNING, no_all_null_columns)
DUPLICATES_NOT_INCREASED = Check(
    "duplicate_rows_not_increased", Severity.WARNING, duplicate_rows_not_increased
)
