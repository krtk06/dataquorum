"""Data inspection and Plotly output views."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from dataquorum.app.demo import demo_source
from dataquorum.app.sources import FILE_SUFFIXES, TABLE_SUFFIXES, uploaded_source
from dataquorum.app.studio_state import STATUS_LABELS, Workspace
from dataquorum.app.style import CHART
from dataquorum.core.inputs import DATABASE_SUFFIXES, describe
from dataquorum.core.state import DataFrameOutput, FigureOutput
from dataquorum.core.statuses import Status


def _load_data(state: Workspace) -> None:
    with st.expander("Load source", expanded=False, icon=":material/upload_file:"):
        kind = st.selectbox("Source format", ["Table", "Database", "Folder"])
        choices = {"Table": TABLE_SUFFIXES, "Database": DATABASE_SUFFIXES, "Folder": FILE_SUFFIXES}
        uploads = st.file_uploader(
            "Source files",
            type=[s[1:] for s in choices[kind]],
            accept_multiple_files=True,
            key=f"source-upload-{state.revision}-{kind}",
        )
        actions = st.columns(2)
        with actions[0]:
            if st.button("Use uploaded source", disabled=not uploads, width="stretch"):
                try:
                    source = uploaded_source(
                        kind, [(f.name, f.getvalue()) for f in uploads], Path("runs/studio/sources")
                    )
                except (ValueError, OSError) as exc:
                    st.error(f"Could not load the source. {exc}")
                else:
                    state.set_source(source)
                    st.rerun()
        with actions[1]:
            if st.button("Use demo source", width="stretch", disabled=state.sample_active):
                state.set_source(demo_source(), sample_active=True)
                st.rerun()


def pick_table(state: Workspace, key: str) -> pd.DataFrame | None:
    tables = state.tables()
    if not tables:
        return None
    choice = st.selectbox("Table", list(tables), key=f"{key}-{state.revision}")
    source = tables[choice]
    if isinstance(source, DataFrameOutput):
        try:
            return source.load()
        except (OSError, ValueError):
            st.error("This table could not be read. Run the pipeline again to recreate it.")
            return None
    return source


def data_page(state: Workspace) -> None:
    with st.container(key="dq_page_intro"):
        st.title("Data")
        st.caption("Inspect the source and verified tables from your workflow.")
    _load_data(state)
    frame = pick_table(state, "data-source")
    if frame is None:
        if isinstance(state.source.data, Path):
            st.header(state.source.name)
            try:
                st.code(describe(state.source.kind, state.source.data), language="text")
            except Exception as exc:
                st.error(f"Could not inspect the source. {exc}")
            st.info("Use a File loader or SQL step to create a table for inspection.")
        return
    with st.container(key="dq_overview"):
        rows, cols, missing = st.columns(3, gap="small")
        rows.metric("Rows", f"{len(frame):,}")
        cols.metric("Columns", len(frame.columns))
        missing.metric("Missing values", f"{int(frame.isna().sum().sum()):,}")
    st.header("Preview")
    st.dataframe(frame.head(250), hide_index=True, width="stretch")
    st.caption(f"Showing {min(250, len(frame)):,} of {len(frame):,} rows.")
    with st.expander("Column profile"):
        profile = pd.DataFrame(
            {
                "column": frame.columns,
                "type": frame.dtypes.astype(str).tolist(),
                "missing": frame.isna().sum().tolist(),
            }
        )
        st.dataframe(profile, hide_index=True, width="stretch")
    st.download_button(
        "Download table as CSV",
        frame.to_csv(index=False),
        "table.csv",
        "text/csv",
        icon=":material/download:",
    )


def _show_plot(figure: go.Figure) -> None:
    st.plotly_chart(figure, width="stretch", theme="streamlit")
    st.download_button(
        "Download chart JSON",
        pio.to_json(figure),
        "figure.json",
        "application/json",
        icon=":material/download:",
    )


def charts_page(state: Workspace) -> None:
    with st.container(key="dq_page_intro"):
        st.title("Charts")
        st.caption("Review a chart from the pipeline or plot a table.")
    figures = {}
    for index, item in enumerate(state.run.steps if state.run else (), start=1):
        result = item.result
        if result and result.status != Status.FAILED and isinstance(result.output, FigureOutput):
            figures[f"{index} · {item.step.label}"] = (result.output, item.status)
    if figures:
        with st.container(key="dq_chart_toolbar"):
            choice = st.selectbox("Pipeline output", list(figures), key=f"figure-{state.revision}")
            output, status = figures[choice]
            st.caption(STATUS_LABELS[status])
        _show_plot(go.Figure(output.figure))
        return
    st.header("Chart a table")
    frame = pick_table(state, "chart-source")
    if frame is None or frame.empty:
        st.info("Load or produce a table to create a chart.")
        return
    numeric = list(frame.select_dtypes(include="number").columns)
    if not numeric:
        st.info("This table needs a numeric column to chart.")
        return
    a, b, c = st.columns(3)
    with a:
        kind = st.selectbox("Chart type", ["Bar", "Line", "Scatter"])
    with b:
        x = st.selectbox("X axis", list(frame.columns), index=min(1, len(frame.columns) - 1))
    with c:
        y = st.selectbox("Y axis", numeric, index=len(numeric) - 1)
    chart = {"Bar": px.bar, "Line": px.line, "Scatter": px.scatter}[kind]
    figure = chart(frame.head(1000), x=x, y=y, color_discrete_sequence=[CHART])
    _show_plot(figure)
    if len(frame) > 1000:
        st.caption("The preview uses the first 1,000 rows.")
