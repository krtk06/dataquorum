"""Agent specs and figure support, with a scripted LLM and an in-process executor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from engine_fakes import InProcessExecutor, ScriptedLLM

from dataquorum.agents import AGENTS, CLEANING, VISUALIZATION, WRANGLING
from dataquorum.core import prompts
from dataquorum.core.engine import Engine, GeneratedCode, RecommendedSteps
from dataquorum.core.state import AgentResult, FigureOutput
from dataquorum.core.statuses import Status
from dataquorum.core.validation import CheckContext, point_count, run_checks

SALES = pd.DataFrame(
    {"region": ["north", "south", "north", "east"], "revenue": [100.0, 80.0, 50.0, 30.0]}
)
STEPS = RecommendedSteps(steps=["group by region", "sum revenue"])


def run(
    spec_name: str, code: str, tmp_path: Path, request: str = "show revenue by region"
) -> AgentResult:
    """One engine run where every code attempt returns `code` (max one retry)."""
    llm = ScriptedLLM(STEPS, GeneratedCode(python_code=code), GeneratedCode(python_code=code))
    engine = Engine(AGENTS[spec_name], llm, InProcessExecutor(), runs_dir=tmp_path, max_retries=1)
    result = engine.start(SALES, request).result
    assert result is not None
    return result


def test_registry_and_kinds() -> None:
    assert set(AGENTS) == {"data_cleaning", "data_wrangling", "data_visualization"}
    assert CLEANING.function_name == "transform"
    assert VISUALIZATION.output_kind == "figure" and VISUALIZATION.function_name == "make_figure"


def test_code_rules_follow_output_kind() -> None:
    fig_prompt = prompts.code_messages(VISUALIZATION, "r", "s", [], ["x"])[0]["content"]
    table_prompt = prompts.code_messages(WRANGLING, "r", "s", [], ["x"])[0]["content"]
    assert "make_figure(df: pd.DataFrame)" in fig_prompt and "plotly" in fig_prompt
    assert "transform(df: pd.DataFrame) -> pd.DataFrame" in table_prompt


def test_wrangling_aggregation_is_success_not_warning(tmp_path: Path) -> None:
    code = "def transform(df):\n    return df.groupby('region', as_index=False)['revenue'].sum()\n"
    result = run("data_wrangling", code, tmp_path, "total revenue by region")
    assert result.status is Status.SUCCESS  # rows and columns shrank on purpose


def test_wrangling_refuses_to_invent_columns(tmp_path: Path) -> None:
    """The original returned an empty table with made-up columns (bug #6)."""
    code = (
        "def transform(df):\n"
        "    missing = {'product_category'} - set(df.columns)\n"
        "    if missing:\n"
        "        raise ValueError(f'missing columns: {sorted(missing)}')\n"
        "    return df\n"
    )
    result = run("data_wrangling", code, tmp_path, "revenue by product category")
    assert result.status is Status.FAILED
    assert result.error is not None and "product_category" in result.error


def test_visualization_returns_a_checked_figure(tmp_path: Path) -> None:
    code = (
        "import plotly.express as px\n"
        "def make_figure(df):\n"
        "    totals = df.groupby('region', as_index=False)['revenue'].sum()\n"
        "    return px.bar(totals, x='region', y='revenue', title='Revenue by region')\n"
    )
    result = run("data_visualization", code, tmp_path)
    assert result.status is Status.SUCCESS
    assert isinstance(result.output, FigureOutput)
    assert result.output.figure["data"][0]["type"] == "bar"


def test_figure_without_title_needs_review(tmp_path: Path) -> None:
    code = (
        "import plotly.express as px\n"
        "def make_figure(df):\n"
        "    return px.bar(df, x='region', y='revenue')\n"
    )
    result = run("data_visualization", code, tmp_path)
    assert result.status is Status.NEEDS_REVIEW
    assert result.error is not None and "figure_has_title" in result.error


def test_empty_figure_fails(tmp_path: Path) -> None:
    code = "import plotly.graph_objects as go\ndef make_figure(df):\n    return go.Figure()\n"
    assert run("data_visualization", code, tmp_path).status is Status.FAILED


def test_point_count_reads_plotly_typed_arrays() -> None:
    fig = px.scatter(x=np.arange(5, dtype=float), y=np.arange(5) * 2)
    encoded = json.loads(fig.to_json())["data"][0]  # Plotly 6+ writes base64 typed arrays
    assert point_count(encoded["x"]) == 5 and point_count(encoded["y"]) == 5
    assert point_count([1, 2, 3]) == 3 and point_count(None) == 0


@pytest.mark.parametrize(
    ("spec_name", "after", "expect_failed"),
    [
        ("data_cleaning", SALES.assign(extra=np.nan), ["no_all_null_columns"]),
        ("data_cleaning", pd.concat([SALES, SALES.iloc[:1]]), ["duplicate_rows_not_increased"]),
        # imputation made two rows identical but added none: not a warning
        ("data_cleaning", SALES.assign(revenue=100.0), []),
        ("data_wrangling", SALES.iloc[:1], []),
    ],
)
def test_agent_specific_checks(
    spec_name: str, after: pd.DataFrame, expect_failed: list[str]
) -> None:
    results = run_checks(AGENTS[spec_name].checks, SALES, after, CheckContext(request="clean"))
    extra_names = {c.name for c in AGENTS[spec_name].extra_checks}
    failed = [r.name for r in results if not r.passed and r.name in extra_names]
    assert failed == expect_failed
