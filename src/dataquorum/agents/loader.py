"""Data Loader agent: turns a folder of files into one DataFrame.

The folder is mounted read-only in the sandbox at /io/data; generated code defines
`load(data_dir)`. It handles CSV/TSV (separators, encodings, junk lines above the header),
Excel sheets, JSON and Parquet, and combining several files. It loads data; it does not clean
it beyond what reading correctly needs (types, header, separator). Pickle files are never
read (unpickling runs arbitrary code).
"""

from dataquorum.agents.checks import NO_ALL_NULL_COLUMNS, UNIQUE_COLUMN_NAMES
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.statuses import Severity
from dataquorum.core.validation import Check, has_rows_and_columns

LOADER = AgentSpec(
    name="data_loader",
    role="You are a data-loading agent that reads files into a pandas DataFrame.",
    recommend_guidance="""Decide which file(s) answer the request, how to read each one
(format, separator, encoding, sheet name, lines to skip above the header, which row is the
header) and, for several files, how to combine them (stack rows with the same columns, and
add a source column only if asked). Use the file previews in the summary to spot separators
and junk lines. If the requested file does not exist, set the blocker and name what is
missing.""",
    code_guidance="""Build paths with os.path.join(data_dir, <relative path from the
summary>). Read every value faithfully: do not drop rows, fill values or rename columns unless
the request asks. Stack multiple files with pd.concat(..., ignore_index=True) in sorted file
order. Never use pickle or eval.""",
    input_kind="files",
    generic_checks=(Check("has_rows_and_columns", Severity.BLOCKING, has_rows_and_columns),),
    extra_checks=(UNIQUE_COLUMN_NAMES, NO_ALL_NULL_COLUMNS),
)
