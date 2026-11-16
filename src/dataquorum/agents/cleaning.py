"""Data Cleaning agent.

Matches the original's behaviour: when the user only says "clean", it follows a standard
checklist (missing values, types, duplicates, extreme outliers, text tidying); explicit user
instructions always win. Unlike the original, dropping rows or columns that were not asked
for is flagged by checks, so the result becomes `needs_review` instead of passing silently.
"""

from dataquorum.agents.checks import (
    DUPLICATES_NOT_INCREASED,
    NO_ALL_NULL_COLUMNS,
    UNIQUE_COLUMN_NAMES,
)
from dataquorum.core.agent_spec import AgentSpec

CLEANING = AgentSpec(
    name="data_cleaning",
    role="You are a careful data-cleaning agent working on pandas DataFrames.",
    recommend_guidance="""Recommend the cleaning steps this dataset needs.
If the user gives specific instructions, follow them exactly and add only fixes that are
needed to carry them out. If the user only asks to "clean" the data, consider, in order:
1. Columns with more than 40% missing values: recommend dropping them only if they look
   unusable, and say so explicitly.
2. Convert columns to their correct types (numbers stored as text, dates, booleans,
   currency or percentage strings).
3. Treat placeholders such as "n/a", "", "null", "-" as missing.
4. Impute remaining missing values: median for numeric, most frequent value for categorical.
5. Remove exact duplicate rows.
6. Remove only extreme outliers (beyond 3x the interquartile range) and only for numeric
   columns where they are clearly errors.
7. Tidy text: strip whitespace and make categories consistent.
Never drop rows or columns silently; name every column you would drop.""",
    code_guidance="""Prefer vectorised pandas operations (str accessor, to_numeric with
errors="coerce", to_datetime) over Python loops. Keep column names unless asked to rename.
Keep the original row order.""",
    extra_checks=(UNIQUE_COLUMN_NAMES, NO_ALL_NULL_COLUMNS, DUPLICATES_NOT_INCREASED),
)
