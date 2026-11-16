"""EDA agent: answers exploratory questions with a summary table plus grounded findings.

The code computes the numbers (missing values per column, distributions, group statistics,
correlations, value counts) in the sandbox; the LLM then writes 2-5 findings that may only
quote numbers from that table (core/findings.py). Findings with numbers that are not in the
table make the run `needs_review`.
"""

from dataquorum.agents.checks import UNIQUE_COLUMN_NAMES
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.statuses import Severity
from dataquorum.core.validation import Check, has_rows_and_columns

EDA = AgentSpec(
    name="eda",
    role="You are an exploratory data analysis agent.",
    recommend_guidance="""Plan the statistics that answer the question: which columns, which
summary (counts, missing values, mean/median/std, quantiles, value counts, group statistics,
correlations), and how to lay out the result table so each answer is one clearly labelled
row or cell. Use only columns in the data summary.""",
    code_guidance="""Return one tidy summary table, not the raw data: one row per column,
group or category, with descriptive column names (for example column, missing_count,
missing_pct, mean, median). Keep it under 60 rows. Compute percentages from 0 to 100 and say
so in the column name. Do not round unless the request asks.""",
    generic_checks=(Check("has_rows_and_columns", Severity.BLOCKING, has_rows_and_columns),),
    extra_checks=(UNIQUE_COLUMN_NAMES,),
    explain_findings=True,
)
