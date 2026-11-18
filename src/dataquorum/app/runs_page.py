"""Inspect current steps and read historical DuckDB traces."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from dataquorum.app.studio_state import AGENT_LABELS, STATUS_LABELS, Workspace


def read_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    with duckdb.connect(str(path), read_only=True) as con:
        return con.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 200").df()


def _current(state: Workspace) -> None:
    if state.run is None:
        st.info("Run a pipeline to see the status and cost of each step.")
        return
    rows = []
    for step in state.run.steps:
        results = step.result
        rows.append(
            {
                "step": step.step.label,
                "status": STATUS_LABELS[step.status],
                "allowance": results.usage.allowance_usd if results else 0.0,
                "seconds": (results.usage.seconds + results.usage.execution_seconds)
                if results
                else 0.0,
                "detail": (results.error or results.message) if results else step.blocked,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _history() -> None:
    try:
        rows = read_history(Path("runs/dataquorum.duckdb"))
    except duckdb.Error as exc:
        st.warning(f"The local run history is unavailable. {exc}")
        return
    if rows.empty:
        st.info("Live agent runs will appear here.")
        return
    with st.container(key="dq_overview"):
        total, verified, allowance = st.columns(3)
        total.metric("Recent runs", len(rows))
        verified.metric("Verified", int(rows["status"].eq("success").sum()))
        allowance.metric("Go allowance", f"${rows['allowance_usd'].sum():.4f}")
    a, b = st.columns(2)
    with a:
        status = st.selectbox(
            "Status",
            ["All", *sorted(set(rows["status"]))],
            format_func=lambda v: STATUS_LABELS.get(v, v),
        )
    with b:
        agent = st.selectbox(
            "Agent",
            ["All", *sorted(set(rows["agent"]))],
            format_func=lambda v: AGENT_LABELS.get(v, v),
        )
    if status != "All":
        rows = rows.loc[rows["status"].eq(status)]
    if agent != "All":
        rows = rows.loc[rows["agent"].eq(agent)]
    if rows.empty:
        st.info("No recorded runs match these filters.")
        return
    shown = rows.loc[
        :, ["created_at", "agent", "status", "allowance_usd", "retries", "run_id", "parent_run_id"]
    ].copy()
    shown["agent"] = shown["agent"].map(lambda v: AGENT_LABELS.get(v, v))
    shown["status"] = shown["status"].map(lambda v: STATUS_LABELS.get(v, v))
    st.dataframe(shown, hide_index=True, width="stretch")
    lookup = {str(row.run_id): str(row.agent) for row in rows.itertuples()}
    selected = st.selectbox(
        "Inspect run", list(lookup), index=None, format_func=lambda v: f"{lookup[v]} · {v[:20]}"
    )
    if selected:
        run = rows.loc[rows["run_id"].eq(selected)].iloc[0]
        if pd.notna(run["error"]):
            st.warning(run["error"])
        if pd.notna(run["checks"]):
            st.json(run["checks"], expanded=False)
        if pd.notna(run["code"]) and run["code"]:
            st.code(run["code"], language="sql" if run["agent"] == "sql_database" else "python")


def runs_page(state: Workspace) -> None:
    with st.container(key="dq_page_intro"):
        st.title("Runs")
        st.caption("Review verification, execution time, and allowance for pipeline steps.")
    current, history = st.tabs(["Current pipeline", "Recorded history"])
    with current:
        _current(state)
    with history:
        _history()
