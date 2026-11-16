"""Throwaway mini agent: write code -> run it in the sandbox -> check -> retry (week 2).

A learning step before the week-3 engine. It uses the real building blocks
(`core/llm.py`, `core/state.py`, the Docker sandbox) wired together with LangGraph:

    START -> generate -> execute -> validate --(fix)--> generate ...
                                       |
                                       +--(done)--> finalize -> END

Run it:
    uv run python research/mini_agent/agent.py
    uv run python research/mini_agent/agent.py --sabotage-first   # watch a retry happen

LangGraph in four ideas (see README.md for more):
1. State: one object every node reads. Here a Pydantic model, `MiniState`.
2. Nodes: plain functions `state -> dict`. The dict holds only the fields that change;
   LangGraph merges it into the state.
3. Edges: fixed arrows (`add_edge`) or a function that picks the next node
   (`add_conditional_edges`). The retry loop is just an arrow pointing backwards.
4. Compile, then `invoke(initial_state)`: LangGraph runs nodes until it reaches END.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from sandbox import run_transform

from dataquorum.core.cost import Usage
from dataquorum.core.llm import LLMClient, LLMConfig, LLMError
from dataquorum.core.state import AgentResult, DataFrameOutput
from dataquorum.core.statuses import CheckResult, Severity

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- 1. State
class MiniState(BaseModel):
    task: str
    input_path: Path
    workdir: Path
    max_attempts: int = 3
    sabotage_first: bool = False  # demo only: replace attempt 1 with broken code

    attempt: int = 0
    steps: list[str] = []
    code: str | None = None
    error: str | None = None  # last execution or LLM error
    output_path: Path | None = None
    checks: list[CheckResult] = []
    usage: Usage = Usage()
    result: AgentResult | None = None


class CodeAttempt(BaseModel):
    """What the LLM must return (structured output, no code scraping)."""

    steps: list[str]
    python_code: str


SYSTEM = (
    "You write one Python function `transform(df: pd.DataFrame) -> pd.DataFrame`. "
    "Import only pandas and numpy, inside or above the function. Do not read or write "
    "files or use the network. Return a new DataFrame."
)
BROKEN_CODE = "def transform(df):\n    return df['no_such_column'].sum()\n"


def describe(df: pd.DataFrame) -> str:
    sample = df.head(6).to_dict(orient="records")
    return f"Columns and dtypes: {df.dtypes.astype(str).to_dict()}\nSample rows: {sample}"


# ---------------------------------------------------------------- 2. Nodes
def feedback(state: MiniState) -> str | None:
    """Why the previous attempt needs fixing: a crash, or failed blocking checks."""
    if state.error:
        return state.error
    failed = [c for c in state.checks if not c.passed and c.severity is Severity.BLOCKING]
    return "; ".join(f"check '{c.name}' failed ({c.detail})" for c in failed) or None


class Generate:
    """A node can be any callable taking the state. A class lets it hold the LLM client."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def __call__(self, state: MiniState) -> dict[str, Any]:
        attempt = state.attempt + 1
        user = f"Task: {state.task}\n\n{describe(pd.read_parquet(state.input_path))}"
        problem = feedback(state)
        if problem and state.code:  # this is a fix attempt: show what went wrong
            user += (
                f"\n\nYour previous code failed: {problem}\n\n"
                f"Previous code:\n{state.code}\n\nReturn corrected code."
            )
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            result = self.client.structured(messages, CodeAttempt)
        except LLMError as exc:
            print(f"  [generate] attempt {attempt}: LLM failed: {exc}")
            # Clear the old code so execute/validate do not re-check the previous attempt.
            return {
                "attempt": attempt,
                "code": None,
                "error": str(exc),
                "usage": state.usage + exc.usage,
            }

        code = result.value.python_code
        if state.sabotage_first and attempt == 1:
            code = BROKEN_CODE
            print("  [generate] attempt 1: replaced with broken code (--sabotage-first)")
        else:
            print(f"  [generate] attempt {attempt}: {len(code.splitlines())} lines of code")
        return {
            "attempt": attempt,
            "steps": result.value.steps,
            "code": code,
            "error": None,
            "usage": state.usage + result.usage,
        }


def execute(state: MiniState) -> dict[str, Any]:
    if state.code is None:  # generate failed; nothing to run, and no stale output
        return {"output_path": None}
    run = run_transform(state.code, state.input_path, state.workdir / f"attempt_{state.attempt}")
    print(
        f"  [execute]  ok={run.ok} in {run.seconds:.1f}s" + (f": {run.error}" if run.error else "")
    )
    return {"error": run.error, "output_path": run.output_path}


def validate(state: MiniState) -> dict[str, Any]:
    if state.output_path is None:
        return {"checks": []}
    before = pd.read_parquet(state.input_path)
    after = pd.read_parquet(state.output_path)
    lost = 1 - len(after) / max(len(before), 1)
    missing = sorted(set(before.columns) - set(after.columns))
    checks = [
        CheckResult(
            name="has_rows",
            passed=len(after) > 0,
            severity=Severity.BLOCKING,
            detail=f"{len(after)} rows",
        ),
        CheckResult(
            name="income_is_numeric",
            passed="income" in after and pd.api.types.is_numeric_dtype(after["income"]),
            severity=Severity.BLOCKING,
            detail="task-specific check",
        ),
        CheckResult(
            name="row_loss",
            passed=lost <= 0.5,
            severity=Severity.WARNING,
            detail=f"lost {lost:.0%} of rows",
        ),
        CheckResult(
            name="columns_kept",
            passed=not missing,
            severity=Severity.WARNING,
            detail=f"missing {missing}" if missing else "all kept",
        ),
    ]
    failed = [c.name for c in checks if not c.passed]
    print(f"  [validate] failed checks: {failed or 'none'}")
    return {"checks": checks}


def finalize(state: MiniState) -> dict[str, Any]:
    output = None
    if state.output_path:
        frame = pd.read_parquet(state.output_path)
        output = DataFrameOutput.save(frame, state.workdir / "result.parquet")
    # Retries = schema repairs inside core/llm.py + extra attempts made by this loop.
    usage = state.usage.model_copy(update={"retries": state.usage.retries + state.attempt - 1})
    result = AgentResult.from_run(
        agent="mini_cleaner",
        output=output,
        checks=state.checks,
        error=state.error,
        code=state.code,
        steps=state.steps,
        usage=usage,
    )
    return {"result": result}


# ---------------------------------------------------------------- 3. Edges
def route_after_validate(state: MiniState) -> Literal["fix", "done"]:
    """The retry decision. Fix if something blocking went wrong and attempts remain."""
    blocking_failed = any(not c.passed and c.severity is Severity.BLOCKING for c in state.checks)
    if (state.error or blocking_failed) and state.attempt < state.max_attempts:
        print("  [route]    -> fix")
        return "fix"
    print("  [route]    -> done")
    return "done"


def build_graph(client: LLMClient) -> Any:
    graph = StateGraph(MiniState)
    graph.add_node("generate", Generate(client))
    graph.add_node("execute", execute)
    graph.add_node("validate", validate)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "execute")
    graph.add_edge("execute", "validate")
    graph.add_conditional_edges(
        "validate", route_after_validate, {"fix": "generate", "done": "finalize"}
    )
    graph.add_edge("finalize", END)
    return graph.compile()  # 4. Compile


# ---------------------------------------------------------------- Demo
def messy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [34, None, 51, 29, None, 42, 38, 61, None, 25, 47, 33],
            "income": [
                "$1,200",
                "$3.4k",
                "n/a",
                " $2,050 ",
                "980",
                "$4,100",
                "1,750.50",
                None,
                "N/A",
                "$1,100",
                "",
                "$2,250",
            ],
            "city": [
                "new york",
                "NEW YORK",
                "Boston",
                "boston ",
                " Chicago",
                "chicago",
                "Seattle",
                "seattle",
                "san  francisco",
                "Boston",
                "Chicago",
                "new york",
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sabotage-first", action="store_true")
    parser.add_argument("--show-graph", action="store_true", help="print Mermaid and exit")
    args = parser.parse_args()

    client = LLMClient(LLMConfig.from_env())
    app = build_graph(client)
    if args.show_graph:
        print(app.get_graph().draw_mermaid())
        return

    workdir = ROOT / "runs" / "mini_agent" / uuid.uuid4().hex[:8]
    workdir.mkdir(parents=True)
    input_path = workdir / "input.parquet"
    messy_frame().to_parquet(input_path, index=False)

    task = (
        "Clean the data: convert income to float dollars ('$3.4k' = 3400, missing or "
        "unparseable -> NaN), strip and title-case city, keep every row."
    )
    print(f"Run dir: {workdir}")
    final = app.invoke(
        MiniState(
            task=task, input_path=input_path, workdir=workdir, sabotage_first=args.sabotage_first
        )
    )
    result: AgentResult = final["result"]
    print(f"\nStatus: {result.status}  |  attempts: {final['attempt']}  |  {result.message}")
    u = result.usage
    print(
        f"Usage: {u.calls} LLM calls, {u.completion_tokens} output tokens, "
        f"${u.allowance_usd:.5f} allowance, {u.seconds:.1f}s LLM time"
    )
    if isinstance(result.output, DataFrameOutput):
        print(result.output.load().to_string())


if __name__ == "__main__":
    main()
