"""The interface the engine uses to run generated code (spec S5).

Generated code defines one function, chosen by the kind of output the agent produces:

- "dataframe": `transform(df: pd.DataFrame) -> pd.DataFrame`, written to `output.parquet`
- "figure":    `make_figure(df: pd.DataFrame) -> plotly Figure`, written to `figure.json`

An executor never raises for bad generated code: failures come back as
`ExecutionResult(ok=False, error=...)`. `DockerExecutor` is the only executor for real use;
tests use an in-process fake that only runs code written by the test author.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

OutputKind = Literal["dataframe", "figure"]

FUNCTION_NAMES: dict[OutputKind, str] = {"dataframe": "transform", "figure": "make_figure"}
OUTPUT_FILES: dict[OutputKind, str] = {"dataframe": "output.parquet", "figure": "figure.json"}


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    error: str | None
    output_path: Path | None  # Parquet for dataframes, JSON for figures
    seconds: float


class Executor(Protocol):
    def run(
        self, code: str, input_path: Path, workdir: Path, kind: OutputKind = "dataframe"
    ) -> ExecutionResult: ...
