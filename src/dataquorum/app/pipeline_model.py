"""Versioned Studio recipes and an executor shared with exported Python scripts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from dataquorum.agents import AGENTS, make_engine
from dataquorum.core.state import AgentResult, DataFrameOutput
from dataquorum.core.statuses import Status
from dataquorum.sandbox.executor import InputKind
from dataquorum.tracing.recorder import RunRecorder

SOURCE = "source"
Data = pd.DataFrame | Path
StepId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,31}$")]
MAX_STEPS = 12


class Step(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=False)

    id: StepId
    label: str = Field(min_length=1, max_length=60)
    agent: str
    request: str = Field(min_length=1, max_length=4000)
    input: StepId = SOURCE

    @model_validator(mode="after")
    def check_fields(self) -> Step:
        if self.id == SOURCE or self.agent not in AGENTS:
            raise ValueError("Use a unique step ID and an available agent.")
        if not self.request.strip() or not self.label.strip():
            raise ValueError("Give each step a name and a request.")
        return self


class Pipeline(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=80)
    steps: tuple[Step, ...] = Field(max_length=MAX_STEPS)

    @model_validator(mode="after")
    def check_graph(self) -> Pipeline:
        if not self.name.strip():
            raise ValueError("Give your pipeline a name.")
        prior: dict[str, Step] = {}
        for step in self.steps:
            if step.id in prior:
                raise ValueError(f"Duplicate step ID: {step.id}")
            if step.input != SOURCE:
                parent = prior.get(step.input)
                if parent is None:
                    raise ValueError(f"{step.label} must use the source or an earlier step.")
                if AGENTS[parent.agent].output_kind != "dataframe":
                    raise ValueError(f"{step.label} requires a table from {parent.label}.")
                if AGENTS[step.agent].input_kind != "dataframe":
                    raise ValueError(f"{step.label} needs a source file or folder.")
            prior[step.id] = step
        return self

    def validate_input(self, kind: InputKind) -> None:
        if not self.steps:
            raise ValueError("Add a step before running the pipeline.")
        for step in self.steps:
            expected = AGENTS[step.agent].input_kind
            if step.input == SOURCE and kind != expected:
                raise ValueError(f"{step.label} needs {expected} input. The source is {kind}.")


@dataclass(frozen=True)
class Source:
    name: str
    kind: InputKind
    data: Data


@dataclass(frozen=True)
class StepRun:
    step: Step
    result: AgentResult | None = None
    blocked: str | None = None

    @property
    def status(self) -> str:
        return self.result.status.value if self.result else "blocked"


@dataclass(frozen=True)
class PipelineRun:
    steps: tuple[StepRun, ...]

    @property
    def status(self) -> Status:
        statuses = {item.status for item in self.steps}
        if Status.FAILED in statuses:
            return Status.FAILED
        if Status.NEEDS_REVIEW in statuses or "blocked" in statuses:
            return Status.NEEDS_REVIEW
        return Status.SUCCESS


Runner = Callable[[Step, Data], AgentResult]


def execute_pipeline(pipeline: Pipeline, source: Source, run_step: Runner) -> PipelineRun:
    """Feed verified tables forward and continue independent branches after a failure."""
    pipeline.validate_input(source.kind)
    runs: dict[str, StepRun] = {}
    for step in pipeline.steps:
        if step.input == SOURCE:
            data = source.data
        else:
            parent = runs[step.input]
            result = parent.result
            if (
                result is None
                or result.status != Status.SUCCESS
                or not isinstance(result.output, DataFrameOutput)
            ):
                runs[step.id] = StepRun(
                    step, blocked=f"Waiting for a verified table from {parent.step.label}."
                )
                continue
            data = result.output.load()
        runs[step.id] = StepRun(step, result=run_step(step, data))
    return PipelineRun(tuple(runs.values()))


def run_live_pipeline(pipeline: Pipeline, source: Source) -> PipelineRun:
    from dataquorum.core.budget import BudgetExceededError
    from dataquorum.sandbox.docker_executor import SandboxUnavailableError

    pipeline.validate_input(source.kind)
    recorder = RunRecorder()
    try:

        def run(step: Step, data: Data) -> AgentResult:
            try:
                engine = make_engine(
                    AGENTS[step.agent],
                    recorder=recorder,
                    runs_dir=Path("runs/studio/engine"),
                    budget_mode="eval",
                )
                result = engine.start(data, step.request).result
                if result is not None:
                    return result
                error = "The step paused before it produced a result."
            except (BudgetExceededError, SandboxUnavailableError) as exc:
                error = str(exc)
            failure = AgentResult.from_run(agent=step.agent, output=None, error=error)
            recorder.record(failure)
            return failure

        return execute_pipeline(pipeline, source, run)
    finally:
        recorder.close()
