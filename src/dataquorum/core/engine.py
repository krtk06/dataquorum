"""The shared agent engine (spec S6), built on LangGraph.

    summarize -> recommend -> [human_review] -> generate -> execute -> validate -> finalize
                    ^               |               ^                      |
                    +--- revise ----+               +-------- fix ---------+

- summarize: deterministic data summary (no LLM).
- recommend: LLM returns ordered steps (structured output).
- human_review (optional): pauses with `interrupt()`; resumes with Approve / Revise / Reject.
  Revise sends the notes back to recommend; the user's request is never edited.
- generate: LLM returns `transform(df)` code (structured output). On a fix attempt it also
  sees the previous code and what went wrong.
- execute: runs the code through an `Executor` (Docker in real use).
- validate: generic + agent checks. Errors or failed blocking checks go back to generate
  while retries remain.
- finalize: `AgentResult.from_run` decides the status. Never a false success.

Agents supply only an `AgentSpec`. A `BudgetExceededError` from the LLM layer is not
caught: an eval run should stop, not record a failure per task.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import pandas as pd
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from dataquorum.core import prompts
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.cost import Usage
from dataquorum.core.llm import LLMError, Message, ReasoningEffort, StructuredResult
from dataquorum.core.review import Approve, Reject, Revise, parse_review
from dataquorum.core.state import AgentResult, DataFrameOutput, FigureOutput
from dataquorum.core.statuses import CheckResult, Severity
from dataquorum.core.validation import CheckContext, run_checks
from dataquorum.sandbox.executor import Executor

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    """What the engine needs from an LLM client (LLMClient satisfies it)."""

    def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        reasoning_effort: ReasoningEffort | None = None,
        model: str | None = None,
    ) -> StructuredResult[T]: ...


class RecommendedSteps(BaseModel):
    steps: list[str]
    blocker: str | None = Field(
        default=None,
        description=(
            "Set only when the request cannot be done with this data, for example it needs "
            "a column that does not exist. Say what is missing. Otherwise null."
        ),
    )


class GeneratedCode(BaseModel):
    python_code: str


class EngineState(BaseModel):
    run_id: str
    request: str  # verbatim, never edited
    input_path: str
    workdir: str
    max_retries: int = 3
    human_review: bool = False

    data_summary: str = ""
    steps: list[str] = []
    review_notes: list[str] = []
    review_kind: Literal["approve", "revise", "reject"] | None = None
    rejected_reason: str | None = None
    blocker: str | None = None  # recommend found the request impossible with this data
    code: str | None = None
    error: str | None = None
    output_path: str | None = None
    checks: list[CheckResult] = []
    attempts: int = 0  # code generations: 1 + fixes
    usage: Usage = Usage()
    result: AgentResult | None = None


def blocking_failed(checks: list[CheckResult]) -> list[CheckResult]:
    return [c for c in checks if not c.passed and c.severity is Severity.BLOCKING]


class Nodes:
    def __init__(self, spec: AgentSpec, llm: StructuredLLM, executor: Executor) -> None:
        self.spec, self.llm, self.executor = spec, llm, executor

    def summarize(self, s: EngineState) -> dict[str, Any]:
        return {"data_summary": prompts.summarize_frame(pd.read_parquet(s.input_path))}

    def recommend(self, s: EngineState) -> dict[str, Any]:
        messages = prompts.recommend_messages(self.spec, s.request, s.data_summary, s.review_notes)
        try:
            out = self.llm.structured(
                messages, RecommendedSteps, reasoning_effort=self.spec.reasoning_effort
            )
        except LLMError as exc:
            return {"error": f"Recommending steps failed: {exc}", "usage": s.usage + exc.usage}
        blocker = (out.value.blocker or "").strip() or None
        return {
            "steps": out.value.steps,
            "blocker": blocker,
            "error": None,
            "usage": s.usage + out.usage,
        }

    def human_review(self, s: EngineState) -> dict[str, Any]:
        answer = interrupt(
            {
                "run_id": s.run_id,
                "agent": self.spec.name,
                "request": s.request,
                "steps": s.steps,
                "review_notes": s.review_notes,
            }
        )
        decision = parse_review(answer)
        if isinstance(decision, Revise):
            return {
                "review_kind": "revise",
                "review_notes": [*s.review_notes, decision.instructions],
            }
        if isinstance(decision, Reject):
            return {"review_kind": "reject", "rejected_reason": decision.reason}
        return {"review_kind": "approve"}

    def generate(self, s: EngineState) -> dict[str, Any]:
        problem = s.error or "; ".join(
            f"check '{c.name}' failed: {c.detail}" for c in blocking_failed(s.checks)
        )
        messages = prompts.code_messages(
            self.spec,
            s.request,
            s.data_summary,
            s.review_notes,
            s.steps,
            previous_code=s.code if problem else None,
            problem=problem or None,
        )
        attempts = s.attempts + 1
        try:
            out = self.llm.structured(
                messages, GeneratedCode, reasoning_effort=self.spec.reasoning_effort
            )
        except LLMError as exc:
            return {
                "attempts": attempts,
                "code": None,
                "output_path": None,
                "checks": [],
                "error": f"Code generation failed: {exc}",
                "usage": s.usage + exc.usage,
            }
        return {
            "attempts": attempts,
            "code": out.value.python_code,
            "error": None,
            "checks": [],
            "output_path": None,
            "usage": s.usage + out.usage,
        }

    def execute(self, s: EngineState) -> dict[str, Any]:
        if s.code is None:
            return {}
        workdir = Path(s.workdir) / f"attempt_{s.attempts}"
        kind = self.spec.output_kind
        run = self.executor.run(s.code, Path(s.input_path), workdir, kind)
        usage = s.usage.model_copy(update={"seconds": s.usage.seconds + run.seconds})
        output = str(run.output_path) if run.output_path else None
        return {"error": run.error, "output_path": output, "usage": usage}

    def _load_output(self, path: str) -> Any:
        if self.spec.output_kind == "figure":
            return json.loads(Path(path).read_text(encoding="utf-8"))
        return pd.read_parquet(path)

    def validate(self, s: EngineState) -> dict[str, Any]:
        if s.output_path is None:
            return {"checks": []}
        before, after = pd.read_parquet(s.input_path), self._load_output(s.output_path)
        checks = run_checks(self.spec.checks, before, after, CheckContext(request=s.request))
        return {"checks": checks}

    def finalize(self, s: EngineState) -> dict[str, Any]:
        rejected = s.review_kind == "reject"
        output: DataFrameOutput | FigureOutput | None = None
        if s.output_path and not rejected:
            loaded = self._load_output(s.output_path)
            if self.spec.output_kind == "figure":
                output = FigureOutput(figure=loaded)
            else:
                output = DataFrameOutput.save(loaded, Path(s.workdir) / "result.parquet")
        error: str | None
        if rejected:
            error = f"Rejected by reviewer: {s.rejected_reason}"
        elif s.blocker:
            error = f"Cannot do this with the given data: {s.blocker}"
        else:
            error = s.error
        retries = s.usage.retries + max(s.attempts - 1, 0)
        result = AgentResult.from_run(
            run_id=s.run_id,
            agent=self.spec.name,
            output=output,
            checks=s.checks,
            error=error,
            code=s.code,
            steps=s.steps,
            usage=s.usage.model_copy(update={"retries": retries}),
        )
        return {"result": result}


def after_recommend(s: EngineState) -> str:
    if s.error or s.blocker:  # nothing worth generating code for
        return "finalize"
    return "human_review" if s.human_review else "generate"


def after_review(s: EngineState) -> str:
    return {"revise": "recommend", "reject": "finalize"}.get(s.review_kind or "", "generate")


def after_validate(s: EngineState) -> str:
    failed = s.error is not None or bool(blocking_failed(s.checks))
    return "generate" if failed and s.attempts <= s.max_retries else "finalize"


def build_graph(spec: AgentSpec, llm: StructuredLLM, executor: Executor) -> StateGraph[Any]:
    n = Nodes(spec, llm, executor)
    g: StateGraph[Any] = StateGraph(EngineState)
    for name in (
        "summarize",
        "recommend",
        "human_review",
        "generate",
        "execute",
        "validate",
        "finalize",
    ):
        g.add_node(name, getattr(n, name))
    g.add_edge(START, "summarize")
    g.add_edge("summarize", "recommend")
    g.add_conditional_edges("recommend", after_recommend, ["human_review", "generate", "finalize"])
    g.add_conditional_edges("human_review", after_review, ["recommend", "generate", "finalize"])
    g.add_edge("generate", "execute")
    g.add_edge("execute", "validate")
    g.add_conditional_edges("validate", after_validate, ["generate", "finalize"])
    g.add_edge("finalize", END)
    return g


# Types stored in checkpoints. LangGraph will refuse unregistered types in a future version.
_CHECKPOINT_TYPES = [
    ("dataquorum.core.cost", "Usage"),
    ("dataquorum.core.statuses", "CheckResult"),
    ("dataquorum.core.statuses", "Severity"),
    ("dataquorum.core.statuses", "Status"),
    ("dataquorum.core.state", "AgentResult"),
    ("dataquorum.core.state", "DataFrameOutput"),
    ("dataquorum.core.state", "FigureOutput"),
    ("dataquorum.core.state", "ModelOutput"),
    ("dataquorum.core.state", "TextOutput"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
]


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)


@dataclass(frozen=True)
class EngineRun:
    run_id: str
    result: AgentResult | None = None  # set when the run finished
    review: dict[str, Any] | None = None  # set when the run is paused for review

    @property
    def paused(self) -> bool:
        return self.review is not None


class ResultSink(Protocol):
    """Where finished results go (tracing.recorder.RunRecorder in real use)."""

    def record(self, result: AgentResult) -> None: ...


class Engine:
    def __init__(
        self,
        spec: AgentSpec,
        llm: StructuredLLM,
        executor: Executor,
        *,
        runs_dir: Path = Path("runs/engine"),
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        max_retries: int = 3,
        recorder: ResultSink | None = None,
    ) -> None:
        self.spec = spec
        self.runs_dir = runs_dir
        self.max_retries = max_retries
        self.recorder = recorder
        saver = checkpointer or InMemorySaver(serde=checkpoint_serializer())
        self.app = build_graph(spec, llm, executor).compile(checkpointer=saver)

    def start(
        self,
        data: pd.DataFrame,
        request: str,
        *,
        human_review: bool = False,
        run_id: str | None = None,
    ) -> EngineRun:
        run_id = run_id or uuid.uuid4().hex
        workdir = self.runs_dir / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        input_path = workdir / "input.parquet"
        data.to_parquet(input_path, index=False)
        state = EngineState(
            run_id=run_id,
            request=request,
            input_path=str(input_path),
            workdir=str(workdir),
            max_retries=self.max_retries,
            human_review=human_review,
        )
        return self._run(run_id, state)

    def resume(
        self, run_id: str, decision: Approve | Revise | Reject | dict[str, Any] | str
    ) -> EngineRun:
        """Continue a paused run. Invalid replies raise here, before the graph resumes."""
        parsed = parse_review(decision)
        return self._run(run_id, Command(resume=parsed.model_dump()))

    def _run(self, run_id: str, graph_input: Any) -> EngineRun:
        config: Any = {"configurable": {"thread_id": run_id}, "recursion_limit": 100}
        out = self.app.invoke(graph_input, config)
        if out.get("__interrupt__"):
            return EngineRun(run_id=run_id, review=out["__interrupt__"][0].value)
        result: AgentResult = out["result"]
        if self.recorder is not None:
            self.recorder.record(result)
        return EngineRun(run_id=run_id, result=result)
