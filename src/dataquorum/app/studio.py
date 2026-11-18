"""Streamlit entry point for DataQuorum Studio."""

import streamlit as st

from dataquorum.app.models_page import model_page
from dataquorum.app.pipeline_page import pipeline_page
from dataquorum.app.runs_page import runs_page
from dataquorum.app.studio_state import workspace
from dataquorum.app.style import BRAND, STYLE
from dataquorum.app.views import charts_page, data_page

NAVIGATION = ["Pipeline", "Data", "Charts", "Model", "Runs"]


def main() -> None:
    st.set_page_config(
        page_title="DataQuorum Studio", layout="wide", initial_sidebar_state="collapsed"
    )
    st.html(STYLE)
    state = workspace()
    with st.container(key="dq_topbar"):
        brand, navigation, options = st.columns(3, vertical_alignment="center")
    with brand:
        st.html(BRAND)
    with navigation, st.container(key="dq_nav"):
        page = st.radio(
            "Workspace",
            NAVIGATION,
            horizontal=True,
            label_visibility="collapsed",
            key="navigation",
            width="stretch",
        )
    with options, st.popover("Live mode" if st.session_state.get("live_agents") else "Demo mode"):
        live = st.toggle("Live agents", key="live_agents")
        st.caption("Live runs use Docker, send a data preview to OpenCode Go, and use allowance.")
    if page == "Pipeline":
        pipeline_page(state, live=live)
    elif page == "Data":
        data_page(state)
    elif page == "Charts":
        charts_page(state)
    elif page == "Model":
        model_page(state)
    else:
        runs_page(state)


if __name__ == "__main__":
    main()
