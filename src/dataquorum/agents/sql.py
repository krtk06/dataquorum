"""SQL Database agent: answers a question with one read-only query on a SQLite/DuckDB file.

The query is checked by sqlglot and run on a read-only connection (sandbox/sql_executor.py),
so a request like "delete the old orders" can only fail. An empty result is reported as
`failed` ("the output is empty") rather than `success`; the error says so, which is how a
user tells "no rows match" from a broken query.
"""

from dataquorum.agents.checks import NO_ALL_NULL_COLUMNS, UNIQUE_COLUMN_NAMES
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.statuses import Severity
from dataquorum.core.validation import Check, has_rows_and_columns

SQL = AgentSpec(
    name="sql_database",
    role="You are a SQL analyst who answers questions with one read-only query.",
    recommend_guidance="""Plan the query: which tables and columns answer the question, how
tables join (name the join keys), filters, grouping, ordering and limits. Use only tables and
columns in the schema. If the question needs data the database does not have, or asks to
change data (insert, update, delete, create), set the blocker and explain.""",
    code_guidance="""Write the query in the database's dialect (shown in the schema). Give
computed columns short snake_case aliases. Use explicit JOIN ... ON. Do not add LIMIT unless
the question asks for a number of rows.""",
    input_kind="database",
    language="sql",
    generic_checks=(Check("has_rows_and_columns", Severity.BLOCKING, has_rows_and_columns),),
    extra_checks=(UNIQUE_COLUMN_NAMES, NO_ALL_NULL_COLUMNS),
)
