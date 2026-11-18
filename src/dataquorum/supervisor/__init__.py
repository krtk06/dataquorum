"""Supervisor graph and request routers."""

from pathlib import Path

from dataquorum.agents import make_engine
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.budget import BudgetGuard, Mode
from dataquorum.core.engine import Engine, StructuredLLM
from dataquorum.core.llm import LLMClient, LLMConfig
from dataquorum.sandbox.docker_executor import DockerExecutor
from dataquorum.supervisor.graph import Supervisor, SupervisorRun
from dataquorum.supervisor.router_llm import LLMRouter, RouteDecision
from dataquorum.tracing.recorder import DuckDBLedger, RunRecorder


def make_supervisor(
    *,
    llm: StructuredLLM | None = None,
    runs_dir: Path = Path("runs/supervisor"),
    recorder: RunRecorder | None = None,
    budget_mode: Mode = "interactive",
    max_steps: int = 8,
) -> Supervisor:
    """Share the LLM client, budget and recorder across the parent and child engines."""
    recorder = recorder if recorder is not None else RunRecorder()
    client = (
        llm
        if llm is not None
        else LLMClient(
            LLMConfig.from_env(), budget=BudgetGuard(DuckDBLedger(recorder), mode=budget_mode)
        )
    )
    docker = DockerExecutor()

    def build(spec: AgentSpec) -> Engine:
        return make_engine(
            spec,
            llm=client,
            recorder=recorder,
            runs_dir=runs_dir / "agents",
            executor=docker if spec.language == "python" else None,
        )

    return Supervisor(
        LLMRouter(client), build, runs_dir=runs_dir, max_steps=max_steps, recorder=recorder
    )


__all__ = ["LLMRouter", "RouteDecision", "Supervisor", "SupervisorRun", "make_supervisor"]
