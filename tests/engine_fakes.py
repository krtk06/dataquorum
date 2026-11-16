"""Test doubles for the engine: a scripted LLM and an in-process executor.

`InProcessExecutor` runs code with exec() in the test process. It is only ever given code
written in these tests, never LLM output (real runs use DockerExecutor).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from dataquorum.core.cost import Usage
from dataquorum.core.llm import LLMError, Message, StructuredResult
from dataquorum.sandbox.executor import (
    OUTPUT_FILES,
    ExecutionResult,
    InputKind,
    OutputKind,
    function_name,
)

CALL_USAGE = Usage(models=["fake-model"], calls=1, completion_tokens=100, allowance_usd=0.001)


class ScriptedLLM:
    """Returns queued replies in order; records the messages of every call."""

    def __init__(self, *replies: BaseModel | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, list[Message]]] = []

    def structured(
        self, messages: list[Message], schema: type[Any], **_: Any
    ) -> StructuredResult[Any]:
        self.calls.append((schema.__name__, messages))
        if not self.replies:
            raise AssertionError(f"Unexpected LLM call for {schema.__name__}")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        assert isinstance(reply, schema), f"queued {type(reply).__name__}, asked {schema.__name__}"
        return StructuredResult(value=reply, usage=CALL_USAGE.model_copy(), model="fake-model")

    def prompts_for(self, schema_name: str) -> list[str]:
        return [m[-1]["content"] for name, m in self.calls if name == schema_name]


def llm_failure() -> LLMError:
    return LLMError("model unavailable", CALL_USAGE.model_copy())


class InProcessExecutor:
    def run(
        self,
        code: str,
        input_path: Path,
        workdir: Path,
        kind: OutputKind = "dataframe",
        input_kind: InputKind = "dataframe",
    ) -> ExecutionResult:
        start = time.perf_counter()
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / OUTPUT_FILES[kind]
        try:
            namespace: dict[str, Any] = {}
            exec(code, namespace)
            func = namespace[function_name(input_kind, kind)]
            out = (
                func(str(input_path))
                if input_kind == "files"
                else func(pd.read_parquet(input_path))
            )
            if kind == "dataframe":
                if not isinstance(out, pd.DataFrame):
                    raise TypeError(f"returned {type(out).__name__}")
                out.to_parquet(path, index=False)
            else:
                path.write_text(out.to_json(), encoding="utf-8")
        except Exception as exc:
            return ExecutionResult(False, f"{type(exc).__name__}: {exc}", None, 0.0)
        return ExecutionResult(True, None, path, time.perf_counter() - start)
