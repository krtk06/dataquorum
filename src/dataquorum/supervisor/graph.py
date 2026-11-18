"""Bounded LangGraph supervisor for DataQuorum's validated specialists."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from dataquorum.agents import AGENTS
from dataquorum.core import inputs
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.budget import BudgetExceededError
from dataquorum.core.cost import Usage
from dataquorum.core.engine import Engine, ResultSink
from dataquorum.core.llm import LLMError
from dataquorum.core.state import AgentResult, DataFrameOutput
from dataquorum.core.statuses import Status
from dataquorum.sandbox.docker_executor import SandboxUnavailableError
from dataquorum.sandbox.executor import InputKind
from dataquorum.supervisor.router_llm import LLMRouter, RouteDecision


class _State(TypedDict):
    run_id: str
    request: str
    input_kind: InputKind
    input_path: Path
    seen: frozenset[tuple[str, str, str]]
    decisions: tuple[RouteDecision, ...]
    children: tuple[AgentResult, ...]
    usage: Usage
    error: str | None
    result: AgentResult | None


@dataclass(frozen=True)
class SupervisorRun:
    result: AgentResult
    decisions: tuple[RouteDecision, ...]
    children: tuple[AgentResult, ...]


def _fingerprint(kind: InputKind, path: Path) -> str:
    if kind == "files":
        return str(path.resolve())
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


class Supervisor:
    def __init__(
        self,
        router: LLMRouter,
        engine_factory: Callable[[AgentSpec], Engine],
        *,
        runs_dir: Path = Path("runs/supervisor"),
        max_steps: int = 8,
        recorder: ResultSink | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.router = router
        self.engine_factory = engine_factory
        self.runs_dir = runs_dir
        self.max_steps = max_steps
        self.recorder = recorder

        graph: StateGraph[_State] = StateGraph(_State)
        graph.add_node("route", self._route)
        graph.add_node("dispatch", self._dispatch)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", self._after_route, ["dispatch", "finalize"])
        graph.add_conditional_edges("dispatch", self._after_dispatch, ["route", "finalize"])
        graph.add_edge("finalize", END)
        self._graph = graph.compile()

    def start(self, data: pd.DataFrame | str | Path, request: str) -> SupervisorRun:
        run_id = uuid.uuid4().hex
        workdir = self.runs_dir / run_id
        workdir.mkdir(parents=True, exist_ok=False)
        kind, path = inputs.prepare(data, workdir)
        state = _State(
            run_id=run_id,
            request=request,
            input_kind=kind,
            input_path=path,
            seen=frozenset(),
            decisions=(),
            children=(),
            usage=Usage(),
            error=None,
            result=None,
        )
        final = self._graph.invoke(state, {"recursion_limit": 2 * self.max_steps + 5})
        result = AgentResult.model_validate(final["result"])
        if self.recorder is not None:
            self.recorder.record(result)
        return SupervisorRun(result, final["decisions"], final["children"])

    def _route(self, state: _State) -> dict[str, Any]:
        try:
            summary = inputs.describe(state["input_kind"], state["input_path"])
            out = self.router.route(
                state["request"], state["input_kind"], summary, state["children"]
            )
        except LLMError as exc:
            return {"error": f"Routing failed: {exc}", "usage": state["usage"] + exc.usage}
        except BudgetExceededError as exc:
            return {"error": str(exc)}
        decision = out.value
        error = decision.refusal
        if error is None:
            if decision.next == "finish":
                if not state["children"]:
                    error = "The supervisor finished before an agent produced a result."
            elif len(state["children"]) >= self.max_steps:
                error = f"Supervisor reached its {self.max_steps}-step limit before completion."
        return {
            "error": error,
            "usage": state["usage"] + out.usage,
            "decisions": (*state["decisions"], decision),
        }

    def _after_route(self, state: _State) -> str:
        if state["error"] is not None or state["decisions"][-1].next == "finish":
            return "finalize"
        return "dispatch"

    def _dispatch(self, state: _State) -> dict[str, Any]:
        decision = state["decisions"][-1]
        kind, path = state["input_kind"], state["input_path"]
        spec = AGENTS[decision.next]
        if spec.input_kind != kind:
            return {"error": f"{spec.name} requires {spec.input_kind} input, but received {kind}."}
        key = (spec.name, kind, _fingerprint(kind, path))
        if key in state["seen"]:
            return {"error": f"{spec.name} was already run on identical input."}
        engine = self.engine_factory(spec)
        data = pd.read_parquet(path) if kind == "dataframe" else path
        try:
            run = engine.start(
                data,
                state["request"],
                parent_run_id=state["run_id"],
                routing_instructions=decision.instructions,
            )
        except (BudgetExceededError, SandboxUnavailableError) as exc:
            return {"error": f"{spec.name} could not run: {exc}"}
        result = run.result
        if result is None:
            return {"error": f"{spec.name} did not return a completed result."}
        update = {
            "seen": state["seen"] | {key},
            "children": (*state["children"], result),
            "usage": state["usage"] + result.usage,
        }
        if result.status is Status.FAILED:
            update["error"] = result.error
        elif isinstance(result.output, DataFrameOutput):
            update.update(input_kind="dataframe", input_path=result.output.path)
        return update

    def _after_dispatch(self, state: _State) -> str:
        return "finalize" if state["error"] is not None else "route"

    def _finalize(self, state: _State) -> dict[str, AgentResult]:
        children = state["children"]
        review = [f"{r.agent}: {r.error}" for r in children if r.status is Status.NEEDS_REVIEW]
        error = state["error"]
        return {
            "result": AgentResult.from_run(
                run_id=state["run_id"],
                agent="supervisor",
                output=children[-1].output if children else None,
                error=error,
                review_reason=" ".join(review) or None,
                message=error,
                usage=state["usage"],
                steps=[r.agent for r in children],
            )
        }
