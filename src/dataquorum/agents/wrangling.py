"""Data Wrangling agent: filter, join-free reshaping, aggregate, pivot, derive columns.

Wrangling changes the shape of data on purpose, so the generic "row loss" and "columns
dropped" warnings are switched off. What stays: the output must have rows and columns, and
column names must be unique. A request that names columns the data does not have should
fail, not invent them (the original returned an empty table with made-up columns, bug #6).
"""

from dataquorum.agents.checks import NO_ALL_NULL_COLUMNS, UNIQUE_COLUMN_NAMES
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.statuses import Severity
from dataquorum.core.validation import Check, has_rows_and_columns

WRANGLING = AgentSpec(
    name="data_wrangling",
    role="You are a data-wrangling agent that reshapes pandas DataFrames.",
    recommend_guidance="""Recommend the steps that turn the data into what the user asked
for: filtering, grouping and aggregating, pivoting or melting, sorting, deriving new
columns, renaming. Use only columns that exist in the data summary. If the request needs a
column that does not exist, say so in the first step instead of inventing one.""",
    code_guidance="""Use only columns that exist in `df`. If a required column is missing,
raise a ValueError that names it; never create placeholder columns or return an empty frame
to hide the problem. After groupby, call reset_index() so groups are columns. Give derived
columns clear snake_case names.""",
    generic_checks=(Check("has_rows_and_columns", Severity.BLOCKING, has_rows_and_columns),),
    extra_checks=(UNIQUE_COLUMN_NAMES, NO_ALL_NULL_COLUMNS),
)
