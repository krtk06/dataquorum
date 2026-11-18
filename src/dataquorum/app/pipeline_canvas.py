"""Connect Studio state to the accessible workflow diagram."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dataquorum.app.pipeline_model import SOURCE
from dataquorum.app.studio_state import AGENT_LABELS, STATUS_LABELS, Workspace

_ASSETS = Path(__file__).parent
_WORKFLOW_KEY = "dq_workflow"
_workflow = st.components.v2.component(
    "dataquorum_workflow",
    html='<div class="workflow"></div>',
    js=(_ASSETS / "workflow.js").read_text(encoding="utf-8"),
    css=(_ASSETS / "workflow.css").read_text(encoding="utf-8"),
)


def inspector_key(state: Workspace) -> str:
    return f"inspector-{state.revision}"


def render_canvas(state: Workspace) -> None:
    steps = state.pipeline.steps
    if not steps:
        return
    statuses = {r.step.id: r.status for r in state.run.steps} if state.run else {}
    nodes = [{"id": SOURCE, "label": state.source.name, "agent": "Source"}]
    for step in steps:
        status = statuses.get(step.id, "ready")
        nodes.append(
            {
                "id": step.id,
                "label": step.label,
                "agent": AGENT_LABELS[step.agent],
                "parent": step.input,
                "status": status,
                "statusLabel": STATUS_LABELS[status],
            }
        )
    # Apply the current browser event before providing the next selection to
    # the component, so the diagram and inspector agree after consecutive clicks.
    key = inspector_key(state)
    chosen = st.session_state.get(_WORKFLOW_KEY, {}).get("chosen")
    if any(step.id == chosen for step in steps):
        st.session_state[key] = chosen
    _workflow(
        key=_WORKFLOW_KEY,
        data={
            "nodes": nodes,
            "selected": st.session_state.get(key, steps[0].id),
        },
        on_chosen_change=lambda: None,
    )
