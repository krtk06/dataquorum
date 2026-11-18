"""Exercise supervisor routing through the existing engines with scripted responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import pandas as pd
import pytest
from engine_fakes import InProcessExecutor, ScriptedLLM
from test_llm import FakeServer, completion, make_client

from dataquorum.agents import AGENTS
from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.engine import Engine, GeneratedCode, RecommendedSteps, StructuredLLM
from dataquorum.core.state import DataFrameOutput, FigureOutput
from dataquorum.core.statuses import Status
from dataquorum.supervisor import LLMRouter, RouteDecision, Supervisor
from dataquorum.supervisor.router_learned import LearnedRouter
from dataquorum.supervisor.router_llm import Route
from dataquorum.tracing.recorder import RunRecorder

FRAME = pd.DataFrame({"x": [1, 2, 3]})
PLAN = RecommendedSteps(steps=["Perform this step"])
IDENTITY = GeneratedCode(python_code="def transform(df): return df")


def pick(route: Route, instructions: str = "", refusal: str | None = None) -> RouteDecision:
    return RouteDecision(next=route, instructions=instructions, reason="scripted", refusal=refusal)


def supervisor(
    llm: StructuredLLM,
    root: Path,
    *,
    max_steps: int = 8,
    recorder: RunRecorder | None = None,
) -> Supervisor:
    def factory(spec: AgentSpec) -> Engine:
        return Engine(
            spec,
            llm,
            InProcessExecutor(),
            runs_dir=root / "children",
            recorder=recorder,
            max_retries=0,
        )

    return Supervisor(
        LLMRouter(llm), factory, runs_dir=root, max_steps=max_steps, recorder=recorder
    )


def test_routing_schema_tracks_the_registered_agents() -> None:
    assert set(get_args(Route)) == {*AGENTS, "finish"}


def test_child_prompts_preserve_verbatim_request_and_route_instructions(tmp_path: Path) -> None:
    request = "  Adjust x before charting it.\nKeep every row.  "
    shift = GeneratedCode(python_code="def transform(df): return df.assign(x=df.x + 2)")
    chart = GeneratedCode(
        python_code="import plotly.express as px\ndef make_figure(df):\n"
        "    return px.line(df, y='x', title='Adjusted')"
    )
    llm = ScriptedLLM(
        pick("data_wrangling", "Shift x by 2."),
        PLAN,
        shift,
        pick("data_visualization", "Chart the adjusted table."),
        PLAN,
        chart,
        pick("finish"),
    )
    recorder = RunRecorder(tmp_path / "trace.duckdb")
    try:
        workflow = supervisor(llm, tmp_path, recorder=recorder).start(FRAME, request)
        assert workflow.result.status is Status.SUCCESS
        assert isinstance(workflow.result.output, FigureOutput)
        assert workflow.result.usage.calls == 7
        assert all(r.parent_run_id == workflow.result.run_id for r in workflow.children)
        assert set(recorder.runs("parent_run_id IS NOT NULL")["run_id"]) == {
            r.run_id for r in workflow.children
        }
    finally:
        recorder.close()

    for schema in ("RecommendedSteps", "GeneratedCode"):
        first, second = llm.prompts_for(schema)
        for prompt, hint in ((first, "Shift x by 2."), (second, "Chart the adjusted table.")):
            assert prompt.split("\n\nData summary:", 1)[0] == "User request (verbatim):\n" + request
            assert "Supervisor instructions for this step" in prompt and hint in prompt
        assert "'x': 5" in second


def test_folder_route_changes_input_kind_after_loading(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    FRAME.to_csv(source / "input.csv", index=False)
    loader = GeneratedCode(
        python_code="from pathlib import Path\nimport pandas as pd\n"
        "def load(data_dir): return pd.read_csv(Path(data_dir) / 'input.csv')"
    )
    llm = ScriptedLLM(
        pick("data_loader"), PLAN, loader, pick("data_wrangling"), PLAN, IDENTITY, pick("finish")
    )
    run = supervisor(llm, tmp_path).start(source, "Load the CSV and return its rows.")
    assert isinstance(run.result.output, DataFrameOutput)
    pd.testing.assert_frame_equal(run.result.output.load(), FRAME)
    route_prompts = llm.prompts_for("RouteDecision")
    assert "Eligible agents: data_loader" in route_prompts[0]
    assert "Current input kind: dataframe" in route_prompts[1]


@pytest.mark.parametrize("route", ["finish", "data_wrangling"])
def test_refusal_has_priority_over_the_route_field(tmp_path: Path, route: Route) -> None:
    llm = ScriptedLLM(pick(route, refusal="The requested operation is unavailable."))
    run = supervisor(llm, tmp_path).start(FRAME, "Make an unsupported report.")
    assert run.result.status is Status.FAILED and not run.children
    assert run.result.message == "The requested operation is unavailable."


def test_failed_child_stops_the_workflow(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        pick("data_wrangling"),
        RecommendedSteps(steps=[], blocker="Missing column b"),
    )
    run = supervisor(llm, tmp_path).start(FRAME, "Aggregate b.")
    assert run.result.status is Status.FAILED
    assert "Missing column b" in run.result.message
    assert len(run.children) == 1 and len(llm.calls) == 2


def test_later_success_preserves_an_earlier_warning(tmp_path: Path) -> None:
    frame = FRAME.assign(extra=1)
    drop = GeneratedCode(python_code="def transform(df): return df[['x']]")
    llm = ScriptedLLM(
        pick("data_cleaning"),
        PLAN,
        drop,
        pick("data_wrangling"),
        PLAN,
        IDENTITY,
        pick("finish"),
    )
    run = supervisor(llm, tmp_path).start(frame, "Keep all columns.")
    assert [r.status for r in run.children] == [Status.NEEDS_REVIEW, Status.SUCCESS]
    assert run.result.status is Status.NEEDS_REVIEW
    assert "data_cleaning" in run.result.message


@pytest.mark.parametrize(
    ("route", "message"),
    [("finish", "before an agent"), ("data_loader", "requires files input")],
)
def test_unusable_route_fails_without_a_child(tmp_path: Path, route: Route, message: str) -> None:
    run = supervisor(ScriptedLLM(pick(route)), tmp_path).start(FRAME, "Test.")
    assert run.result.status is Status.FAILED and not run.children
    assert message in run.result.message


def test_refuses_identical_repeated_input_even_if_instructions_change(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        pick("data_wrangling"),
        PLAN,
        IDENTITY,
        pick("data_wrangling", "Try again with different wording."),
    )
    run = supervisor(llm, tmp_path).start(FRAME, "Return the table.")
    assert run.result.status is Status.FAILED
    assert "already run on identical input" in run.result.message
    assert len(run.children) == 1


def test_repeated_agent_can_use_a_changed_table(tmp_path: Path) -> None:
    shift = GeneratedCode(python_code="def transform(df): return df.assign(x=df.x + 2)")
    llm = ScriptedLLM(
        pick("data_wrangling"), PLAN, shift, pick("data_wrangling"), PLAN, shift, pick("finish")
    )
    run = supervisor(llm, tmp_path).start(FRAME, "Apply two shifts in sequence.")
    assert run.result.status is Status.SUCCESS
    assert isinstance(run.result.output, DataFrameOutput)
    assert run.result.output.load()["x"].tolist() == [5, 6, 7]


def test_late_refusal_keeps_exact_reason_and_partial_trace(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        pick("data_wrangling"),
        PLAN,
        IDENTITY,
        pick("finish", refusal="The remaining report requires an absent field."),
    )
    run = supervisor(llm, tmp_path).start(FRAME, "Return the table and analyze b.")
    assert run.result.status is Status.FAILED and len(run.children) == 1
    assert run.result.message == "The remaining report requires an absent field."


def test_step_budget_caps_execution_but_allows_finishing(tmp_path: Path) -> None:
    for subsequent in ("data_cleaning", "finish"):
        llm = ScriptedLLM(pick("data_wrangling"), PLAN, IDENTITY, pick(subsequent))
        run = supervisor(llm, tmp_path, max_steps=1).start(FRAME, "Return the table.")
        assert len(run.children) == 1
        if subsequent == "finish":
            assert run.result.status is Status.SUCCESS
        else:
            assert run.result.status is Status.FAILED and "1-step limit" in run.result.message


def test_invalid_router_output_gets_one_repair_then_fails(tmp_path: Path) -> None:
    ambiguous = (
        '{"next":"before we finish try data_cleaning","instructions":"","reason":"","refusal":null}'
    )
    server = FakeServer((200, completion(ambiguous)), (200, completion("")))
    run = supervisor(make_client(server), tmp_path).start(FRAME, "Clean x then finish.")
    assert run.result.status is Status.FAILED and not run.children
    assert "after one repair" in run.result.message
    assert run.result.usage.calls == 2 and run.result.usage.retries == 1


def test_provider_refusal_is_preserved_without_repair(tmp_path: Path) -> None:
    reason = "I cannot provide that operation."
    server = FakeServer((200, completion("", refusal=reason)))
    run = supervisor(make_client(server), tmp_path).start(FRAME, "Run an unavailable operation.")
    assert run.result.status is Status.FAILED and not run.children
    assert run.result.message == reason
    assert run.result.usage.calls == 1 and run.result.usage.retries == 0


def test_learned_router_excludes_incompatible_agents() -> None:
    class Classifier:
        classes_ = ("data_cleaning", "data_loader", "refuse")

        def predict_proba(self, features: Any) -> list[list[float]]:
            return [[0.9, 0.07, 0.03]]

    class Encoder:
        def transform(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    router = LearnedRouter(Classifier(), Encoder())
    assert router.predict("Clean strings.", "files") == "data_loader"
    assert router.predict("Clean strings.", "database") == "refuse"
