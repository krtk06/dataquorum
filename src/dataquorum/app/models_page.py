"""Audit model inputs and run the existing sandbox model tools."""

from __future__ import annotations

import uuid
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
import streamlit as st

from dataquorum.app.demo import model_sample
from dataquorum.app.studio_state import Workspace
from dataquorum.app.views import pick_table
from dataquorum.ml.correctness import Problem, audit
from dataquorum.ml.export import export_forest
from dataquorum.ml.report import Report, forest_report
from dataquorum.sandbox.docker_executor import SandboxUnavailableError

REPORTS: dict[str, Report] = {
    "Tuning summary": "tuning",
    "Feature importance": "shap",
    "Calibration": "calibration",
}


def _selected_frame(state: Workspace) -> pd.DataFrame | None:
    if state.sample_active and st.toggle("Model demo data", value=True, key="model-demo"):
        return model_sample()
    return pick_table(state, "model-source")


def _artifact(state: Workspace) -> None:
    if state.model_artifact is None:
        return
    path = Path(state.model_artifact)
    try:
        artifact = path.read_bytes()
    except OSError:
        st.error("The exported artifact is unavailable. Train and export a new forest.")
        return
    st.download_button(
        "Download forest artifact",
        artifact,
        "forest.zip",
        "application/zip",
        width="stretch",
    )
    with st.expander("Register a version"):
        name = st.text_input("Registered model name", "studio-forest")
        if st.button("Register locally", disabled=not name.strip()):
            from dataquorum.ml.registry import ForestRegistry

            try:
                registered = ForestRegistry(Path("runs/registry")).register(name.strip(), path)
                st.success(f"Registered {registered.name} version {registered.version}.")
            except Exception as exc:
                st.error(f"Could not register the model. {exc}")


def model_page(state: Workspace) -> None:
    with st.container(key="dq_page_intro"):
        st.title("Model")
        st.caption("Audit model inputs and use Docker to generate a report or train a forest.")
    frame = _selected_frame(state)
    if frame is None or frame.empty:
        st.info("Load or produce a table to audit it for modeling.")
        return
    columns = [str(col) for col in frame.columns]
    split_index = columns.index("split") + 1 if "split" in columns else 0
    default_target = columns.index("conversion") if "conversion" in columns else 0
    registry_available = all(find_spec(name) is not None for name in ("skops", "mlflow"))
    st.header("Model inputs", anchor=False)
    with st.form(f"model-form-{state.revision}", border=False):
        left, right = st.columns(2)
        with left:
            target = st.selectbox("Target", columns, index=default_target)
            label = st.selectbox("Problem", ["Classification", "Regression"])
            problem: Problem = "classification" if label == "Classification" else "regression"
            report = st.selectbox("Report", list(REPORTS))
        with right:
            split = st.selectbox(
                "Split column",
                [None, *columns],
                index=split_index,
                format_func=lambda v: v or "Auto / none",
            )
            group = st.selectbox(
                "Group column", [None, *columns], format_func=lambda v: v or "Auto"
            )
            date = st.selectbox("Time column", [None, *columns], format_func=lambda v: v or "Auto")
        a, b, c = st.columns(3)
        with a:
            auditing = st.form_submit_button("Audit inputs", type="primary", width="stretch")
        with b:
            reporting = st.form_submit_button("Generate report", width="stretch")
        with c:
            training = st.form_submit_button(
                "Train and export", width="stretch", disabled=not registry_available
            )
    if not registry_available:
        st.caption("Install the serving extra to export and register a trained forest.")
    if auditing or reporting or training:
        state.model_result, state.model_artifact = None, None
        state.model_result_name = None
        try:
            with st.spinner("Checking model inputs…" if auditing else "Running in Docker…"):
                if auditing:
                    state.model_result = audit(
                        frame, target, problem=problem, split=split, group=group, time=date
                    )
                    state.model_result_name = "Input audit"
                else:
                    actual_split = split or "split"
                    if actual_split not in frame:
                        raise ValueError("Choose a split column with train and test labels.")
                    if training:
                        artifact = Path("runs/studio/models") / f"{uuid.uuid4().hex}.zip"
                        state.model_artifact = str(
                            export_forest(
                                frame,
                                target,
                                artifact,
                                problem=problem,
                                split=actual_split,
                                group=group,
                                time=date,
                                trials=4,
                                budget_s=15,
                            )
                        )
                    else:
                        state.model_result = forest_report(
                            frame,
                            target,
                            problem=problem,
                            report=REPORTS[report],
                            split=actual_split,
                            group=group,
                            time=date,
                            trials=4,
                            budget_s=15,
                        )
                        state.model_result_name = report
        except (ValueError, RuntimeError, OSError, SandboxUnavailableError) as exc:
            st.error(str(exc))
    if state.model_result is not None:
        st.header(state.model_result_name or "Result")
        st.dataframe(state.model_result, hide_index=True, width="stretch")
    _artifact(state)
