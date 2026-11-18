"""Edit recipes, run validated steps, and export Studio pipelines."""

from __future__ import annotations

import uuid
from html import escape
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from dataquorum.agents import AGENTS
from dataquorum.app.demo import DEMO, demo_source, run_demo
from dataquorum.app.export import export_script
from dataquorum.app.pipeline_canvas import inspector_key, render_canvas
from dataquorum.app.pipeline_model import MAX_STEPS, SOURCE, Pipeline, Step, run_live_pipeline
from dataquorum.app.studio_state import AGENT_LABELS, STATUS_LABELS, Workspace

RUN_ROOT = Path("runs/studio")


def _save(state: Workspace, steps: tuple[Step, ...]) -> None:
    state.set_pipeline(Pipeline(name=state.pipeline.name, steps=steps))
    st.rerun()


def _input_options(state: Workspace, before: tuple[Step, ...], agent: str | None) -> dict[str, str]:
    options = {SOURCE: state.source.name + " (source)"}
    if agent is None or AGENTS[agent].input_kind == "dataframe":
        options.update(
            (step.id, step.label)
            for step in before
            if AGENTS[step.agent].output_kind == "dataframe"
        )
    return options


def _add_step(state: Workspace) -> None:
    with st.form(f"new-step-{state.revision}"):
        agent = st.selectbox("Agent", AGENT_LABELS, format_func=AGENT_LABELS.__getitem__)
        options = _input_options(state, state.pipeline.steps, None)
        input_id = st.selectbox("Input", list(options), format_func=options.__getitem__)
        label = st.text_input("Step name", max_chars=60, placeholder="Defaults to the agent name")
        request = st.text_area(
            "Request", max_chars=4000, height=100, key=f"new-request-{state.revision}"
        )
        save, cancel = st.columns(2)
        if cancel.form_submit_button("Cancel", width="stretch"):
            state.adding_step = False
            st.rerun()
        if save.form_submit_button("Create step", type="primary", width="stretch"):
            try:
                step = Step(
                    id="step-" + uuid.uuid4().hex[:8],
                    label=label or AGENT_LABELS[agent],
                    agent=agent,
                    request=request,
                    input=input_id,
                )
                _save(state, (*state.pipeline.steps, step))
            except ValidationError as exc:
                st.error(exc.errors()[0]["msg"])


def _files(state: Workspace) -> None:
    with st.popover("Files & export", icon=":material/folder_open:", width="stretch"):
        title = st.text_input(
            "Pipeline name",
            value=state.pipeline.name,
            max_chars=80,
            key=f"pipeline-name-{state.revision}",
        )
        col1, col2 = st.columns(2)
        if col1.button("Rename", width="stretch"):
            try:
                state.pipeline = Pipeline(name=title, steps=state.pipeline.steps)
                state.revision += 1
                st.rerun()
            except ValidationError as exc:
                st.error(exc.errors()[0]["msg"])
        if col2.button("Reset to demo", width="stretch"):
            state.set_pipeline(DEMO)
            state.set_source(demo_source(), sample_active=True)
            st.rerun()
        if st.button("New blank pipeline", width="stretch"):
            state.set_pipeline(Pipeline(name="Untitled pipeline", steps=()))
            st.rerun()
        st.download_button(
            "Save pipeline JSON",
            state.pipeline.model_dump_json(indent=2),
            "pipeline.dq.json",
            "application/json",
            key="save_recipe",
            on_click="ignore",
            width="stretch",
        )
        st.download_button(
            "Export runnable Python",
            export_script(state.pipeline),
            "run_pipeline.py",
            "text/x-python",
            key="export_python",
            on_click="ignore",
            width="stretch",
        )
        upload = st.file_uploader(
            "Open pipeline JSON", type=["json"], key=f"pipeline-upload-{state.revision}"
        )
        if (
            st.button("Open pipeline", disabled=upload is None, width="stretch")
            and upload is not None
        ):
            try:
                if upload.size > 1024**2:
                    raise ValueError("Pipeline recipes must be at most 1 MB.")
                state.set_pipeline(Pipeline.model_validate_json(upload.getvalue()))
                st.rerun()
            except (ValidationError, ValueError) as exc:
                st.error(f"Could not open the recipe. {exc}")


def _step_inspector(state: Workspace) -> None:
    with st.container(key="dq_inspector"):
        st.header("Step settings")
        if not state.pipeline.steps:
            st.info("Add a step to set its request and input.")
            return
        ids = {s.id: s.label for s in state.pipeline.steps}
        key = inspector_key(state)
        step_id = st.selectbox("Step", ids, format_func=ids.__getitem__, key=key)
        index = next(i for i, step in enumerate(state.pipeline.steps) if step.id == step_id)
        step = state.pipeline.steps[index]
        st.caption(f"{AGENT_LABELS[step.agent]} agent")
        with st.form(f"settings-{state.revision}-{step.id}", border=False):
            label = st.text_input("Step name", step.label, max_chars=60)
            choices = _input_options(state, state.pipeline.steps[:index], step.agent)
            source = st.selectbox(
                "Input",
                choices,
                index=list(choices).index(step.input),
                format_func=choices.__getitem__,
            )
            request = st.text_area("Request", step.request, height=114, max_chars=4000)
            if st.form_submit_button("Save changes", width="stretch"):
                try:
                    updated = Step(
                        id=step.id, label=label, agent=step.agent, request=request, input=source
                    )
                    steps = list(state.pipeline.steps)
                    steps[index] = updated
                    _save(state, tuple(steps))
                except ValidationError as exc:
                    st.error(exc.errors()[0]["msg"])
        used = any(s.input == step.id for s in state.pipeline.steps)
        if st.button(
            "Remove step",
            icon=":material/delete_outline:",
            key=f"remove-{step.id}",
            disabled=used,
            help="Change dependent steps' inputs before removing this step.",
            width="stretch",
        ):
            _save(state, tuple(s for s in state.pipeline.steps if s.id != step.id))


def _run(state: Workspace, live: bool) -> None:
    selected = st.session_state.get(inspector_key(state))
    state.run = None
    state.revision += 1
    if selected is not None:
        st.session_state[inspector_key(state)] = selected
    with st.spinner("Running pipeline…", show_time=True):
        try:
            state.run = (
                run_live_pipeline(state.pipeline, state.source)
                if live
                else run_demo(state.pipeline, RUN_ROOT)
            )
        except (ValueError, OSError, RuntimeError, duckdb.Error) as exc:
            st.error(str(exc))


def _recent_steps(state: Workspace) -> None:
    if state.run is None:
        return
    st.header("Step results")
    for item in state.run.steps:
        with st.expander(f"{item.step.label} · {STATUS_LABELS[item.status]}"):
            if item.result is None:
                st.warning(item.blocked or "Step blocked")
                continue
            result = item.result
            if result.error:
                st.warning(result.error)
            st.write(result.message)
            if result.validation:
                st.dataframe(
                    pd.DataFrame([check.model_dump() for check in result.validation]),
                    hide_index=True,
                    width="stretch",
                )
            if result.code:
                st.code(
                    result.code, language="sql" if item.step.agent == "sql_database" else "python"
                )


def _source_and_outputs(state: Workspace) -> None:
    with st.container(key="dq_source_card"):
        source, results = st.columns(2)
        with source:
            st.markdown("**Source data**")
            st.write(state.source.name)
            if isinstance(state.source.data, pd.DataFrame):
                st.caption(
                    f"{len(state.source.data):,} rows · {len(state.source.data.columns)} columns"
                )
        with results:
            st.markdown("**Last run**")
            if state.run is None:
                st.write("No run yet")
            else:
                verified = sum(step.status == "success" for step in state.run.steps)
                st.write(f"{verified} of {len(state.run.steps)} steps verified")


def pipeline_page(state: Workspace, *, live: bool) -> None:
    with st.container(key="dq_pipeline_header"):
        heading, file_actions, run_actions = st.columns(3, vertical_alignment="center")
    with heading:
        st.html(f'<h1 class="dq-recipe-title">{escape(state.pipeline.name)}</h1>')
    with file_actions:
        _files(state)
    demo_ready = state.sample_active and state.pipeline.steps == DEMO.steps
    with run_actions:
        if st.button(
            "Run pipeline",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
            disabled=not state.pipeline.steps or (not live and not demo_ready),
        ):
            _run(state, live)
    if not live and not demo_ready:
        st.info("Switch to live agents to run a custom pipeline, or reset to the demo.")
    with st.container(key="dq_pipeline_layout"):
        columns = st.columns([2.25, 1], gap="medium")
    with columns[0], st.container(key="dq_canvas"):
        with st.container(key="dq_workflow_head"):
            header, action = st.columns(2, vertical_alignment="center")
        with header:
            st.header("Workflow", anchor=False)
            size = len(state.pipeline.steps)
            status = state.run.status.value if state.run else "ready"
            st.html(
                f'<div class="dq-run-state" role="status">'
                f'<span class="dq-state" data-state="{status}">{STATUS_LABELS[status]}</span>'
                f'<span class="dq-step-count">{size} {"step" if size == 1 else "steps"}</span>'
                "</div>"
            )
        with action:
            if st.button(
                "Add step",
                icon=":material/add:",
                width="stretch",
                disabled=size >= MAX_STEPS or state.adding_step,
            ):
                state.adding_step = True
        if state.adding_step:
            _add_step(state)
        render_canvas(state)
        if state.pipeline.steps:
            st.html('<a class="dq-inspector-link" href="#step-settings">Go to step settings</a>')
        else:
            st.info("Add a step to start building this pipeline.")
        _source_and_outputs(state)
    with columns[1]:
        _step_inspector(state)
    if state.run is not None:
        with st.container(key="dq_results"):
            _recent_steps(state)
