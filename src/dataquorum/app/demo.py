"""Checked-in demonstration data and deterministic steps for a free local walkthrough."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from dataquorum.agents import AGENTS
from dataquorum.app.pipeline_model import (
    Data,
    Pipeline,
    PipelineRun,
    Source,
    Step,
    execute_pipeline,
)
from dataquorum.app.style import CHART
from dataquorum.core.state import AgentResult, DataFrameOutput, FigureOutput
from dataquorum.core.validation import CheckContext, run_checks

DEMO = Pipeline(
    name="Retail sales analysis",
    steps=(
        Step(
            id="clean",
            label="Clean orders",
            agent="data_cleaning",
            request="Strip whitespace and title-case the region names. Keep every row.",
        ),
        Step(
            id="revenue",
            label="Summarize revenue",
            agent="data_wrangling",
            input="clean",
            request="Sum revenue and count orders per region. Sort by revenue descending.",
        ),
        Step(
            id="chart",
            label="Chart by region",
            agent="data_visualization",
            input="revenue",
            request="Create a horizontal bar chart of total revenue by region.",
        ),
    ),
)
DEMO_SOURCE = "Retail orders"


def demo_source() -> Source:
    rng = np.random.default_rng(17)
    size = 192
    units = rng.integers(1, 12, size)
    return Source(
        DEMO_SOURCE,
        "dataframe",
        pd.DataFrame(
            {
                "order_id": [f"R-{101 + i}" for i in range(size)],
                "date": pd.date_range("2026-10-01", periods=size, freq="3h"),
                "region": rng.choice([" North", "south", "East ", "west"], size),
                "channel": rng.choice(["Online", "Store", "Partner"], size),
                "units": units,
                "revenue": np.round(units * rng.uniform(12, 50, size), 2),
            }
        ),
    )


def model_sample() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    signal, noise = rng.normal(size=(2, 160))
    return pd.DataFrame(
        {
            "visits": signal,
            "basket": noise,
            "conversion": (signal + rng.normal(scale=0.6, size=160) > 0).astype(int),
            "split": ["train"] * 120 + ["test"] * 40,
        }
    )


def run_demo(pipeline: Pipeline, root: Path) -> PipelineRun:
    """Execute the bundled recipe with fixed Python and the same agent validations."""
    if pipeline.steps != DEMO.steps:
        raise ValueError("Load the demo recipe to run it offline.")
    root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="demo-", dir=root))

    def run(step: Step, data: Data) -> AgentResult:
        assert isinstance(data, pd.DataFrame)
        spec = AGENTS[step.agent]
        actual: pd.DataFrame | dict[str, Any]
        if step.id == "chart":
            ordered = data.sort_values("revenue")
            figure = px.bar(
                ordered,
                x="revenue",
                y="region",
                orientation="h",
                title="Revenue by region",
                text_auto="$,.0f",
            )
            figure.update_traces(marker_color=CHART)
            figure.update_layout(
                xaxis_title="Revenue (USD)",
                yaxis_title=None,
                margin={"t": 52, "b": 24, "l": 20, "r": 50},
            )
            chart = FigureOutput(figure=figure.to_dict())
            output: FigureOutput | DataFrameOutput = chart
            actual = chart.figure
        else:
            if step.id == "clean":
                table = data.assign(region=data["region"].str.strip().str.title())
            else:
                table = (
                    data.groupby("region", as_index=False)
                    .agg(revenue=("revenue", "sum"), orders=("order_id", "size"))
                    .sort_values("revenue", ascending=False, ignore_index=True)
                )
            output = DataFrameOutput.save(table, workdir / f"{step.id}.parquet")
            actual = table
        checks = run_checks(spec.checks, data, actual, CheckContext(request=step.request))
        return AgentResult.from_run(agent=step.agent, output=output, checks=checks)

    return execute_pipeline(pipeline, demo_source(), run)
