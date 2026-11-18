"""Session state and shared output selectors for the Streamlit Studio."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from dataquorum.app.demo import DEMO, demo_source
from dataquorum.app.pipeline_model import Pipeline, PipelineRun, Source
from dataquorum.core.state import DataFrameOutput
from dataquorum.core.statuses import Status

AGENT_LABELS = {
    "data_cleaning": "Cleaning",
    "data_wrangling": "Wrangling",
    "data_visualization": "Visualization",
    "data_loader": "File loader",
    "sql_database": "SQL",
    "eda": "Exploratory analysis",
    "feature_engineering": "Feature engineering",
    "ml_correctness": "ML correctness",
}
STATUS_LABELS = {
    "success": "Verified",
    "needs_review": "Needs review",
    "failed": "Failed",
    "blocked": "Blocked",
    "ready": "Ready",
}


@dataclass
class Workspace:
    pipeline: Pipeline = DEMO
    source: Source = field(default_factory=demo_source)
    sample_active: bool = True
    run: PipelineRun | None = None
    revision: int = 0
    adding_step: bool = False
    model_result: pd.DataFrame | None = None
    model_result_name: str | None = None
    model_artifact: str | None = None

    def reset_results(self) -> None:
        self.run = None
        self.model_result = None
        self.model_result_name = None
        self.model_artifact = None
        self.revision += 1

    def set_pipeline(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.adding_step = False
        self.reset_results()

    def set_source(self, source: Source, *, sample_active: bool = False) -> None:
        self.source, self.sample_active = source, sample_active
        self.reset_results()

    def tables(self) -> dict[str, pd.DataFrame | DataFrameOutput]:
        choices: dict[str, pd.DataFrame | DataFrameOutput] = {}
        if isinstance(self.source.data, pd.DataFrame):
            choices["Source · " + self.source.name] = self.source.data
        for i, entry in enumerate(self.run.steps if self.run else (), start=1):
            result = entry.result
            if (
                result is not None
                and result.status == Status.SUCCESS
                and isinstance(result.output, DataFrameOutput)
            ):
                choices[f"{i} · {entry.step.label}"] = result.output
        return choices


def workspace() -> Workspace:
    if "workspace" not in st.session_state:
        st.session_state["workspace"] = Workspace()
    result: Workspace = st.session_state["workspace"]
    return result
