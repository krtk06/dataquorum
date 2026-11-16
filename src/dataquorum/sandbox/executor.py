"""The interface the engine uses to run generated code (spec S5).

What the generated code must define depends on what the agent reads and produces:

| input      | output    | language | generated code                                     |
|------------|-----------|----------|----------------------------------------------------|
| dataframe  | dataframe | python   | `transform(df) -> pd.DataFrame` -> output.parquet  |
| dataframe  | figure    | python   | `make_figure(df) -> plotly Figure` -> figure.json  |
| files      | dataframe | python   | `load(data_dir: str) -> pd.DataFrame`              |
| database   | dataframe | sql      | one read-only SELECT (see sandbox/sql_executor.py) |

An executor never raises for bad generated code: failures come back as
`ExecutionResult(ok=False, error=...)`. Python runs only in `DockerExecutor`; SQL runs in
`SqlExecutor` on a read-only connection after a sqlglot check. Tests use in-process fakes
that only run code written by the test author.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

OutputKind = Literal["dataframe", "figure"]
InputKind = Literal["dataframe", "files", "database"]
Language = Literal["python", "sql"]

FUNCTION_NAMES: dict[OutputKind, str] = {"dataframe": "transform", "figure": "make_figure"}
LOAD_FUNCTION = "load"
OUTPUT_FILES: dict[OutputKind, str] = {"dataframe": "output.parquet", "figure": "figure.json"}


def function_name(input_kind: InputKind, output_kind: OutputKind) -> str:
    return LOAD_FUNCTION if input_kind == "files" else FUNCTION_NAMES[output_kind]


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    error: str | None
    output_path: Path | None  # Parquet for dataframes, JSON for figures
    seconds: float


class Executor(Protocol):
    def run(
        self,
        code: str,
        input_path: Path,
        workdir: Path,
        kind: OutputKind = "dataframe",
        input_kind: InputKind = "dataframe",
    ) -> ExecutionResult: ...
