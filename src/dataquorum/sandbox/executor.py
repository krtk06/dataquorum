"""The interface the engine uses to run generated code (spec S5).

Generated code defines `transform(df: pd.DataFrame) -> pd.DataFrame`. An executor runs it
on a Parquet input and writes a Parquet output. It never raises for bad generated code:
failures come back as `ExecutionResult(ok=False, error=...)`.

`DockerExecutor` (docker_executor.py) is the only executor for real use. Tests use an
in-process fake that only ever runs code written by the test author.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

FUNCTION_NAME = "transform"


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    error: str | None
    output_path: Path | None
    seconds: float


class Executor(Protocol):
    def run(self, code: str, input_path: Path, workdir: Path) -> ExecutionResult: ...
