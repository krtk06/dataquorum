"""Verify Studio recipes and output gating without paid model calls."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from dataquorum.app.demo import DEMO, run_demo
from dataquorum.app.export import export_script
from dataquorum.app.pipeline_model import Data, Pipeline, Source, Step, execute_pipeline
from dataquorum.app.sources import uploaded_source
from dataquorum.core.state import AgentResult, DataFrameOutput, FigureOutput
from dataquorum.core.statuses import Status


def test_demo_round_trips_recipe_and_runs(tmp_path: Path) -> None:
    recipe = Pipeline.model_validate_json(DEMO.model_dump_json(indent=2))
    assert recipe == DEMO
    exported = ast.parse(export_script(recipe))
    assert any(isinstance(node, ast.ImportFrom) for node in exported.body)
    result = run_demo(recipe, tmp_path)
    assert result.status == Status.SUCCESS
    clean, summary, chart = (step.result for step in result.steps)
    assert clean and isinstance(clean.output, DataFrameOutput)
    assert summary and isinstance(summary.output, DataFrameOutput)
    assert chart and isinstance(chart.output, FigureOutput)
    assert clean.output.rows == 192
    assert summary.output.load()["orders"].sum() == 192
    assert len(chart.output.figure["data"]) == 1
    assert all(step.result and step.result.usage.calls == 0 for step in result.steps)


@pytest.mark.parametrize("input_id", ["missing", "same"])
def test_recipe_rejects_unavailable_or_cyclic_input(input_id: str) -> None:
    step = Step(id="same", label="Test", agent="data_wrangling", request="Group", input=input_id)
    with pytest.raises(ValidationError, match="earlier step"):
        Pipeline(name="Broken", steps=(step,))


def test_recipe_rejects_a_chart_as_upstream_input() -> None:
    chart = Step(id="chart", label="Chart", agent="data_visualization", request="Make a chart")
    summarize = Step(id="summary", label="Summary", agent="eda", request="Summarize", input="chart")
    with pytest.raises(ValidationError, match="requires a table"):
        Pipeline(name="Invalid", steps=(chart, summarize))


def test_review_blocks_only_the_dependent_branch(tmp_path: Path) -> None:
    clean, summarize, _ = DEMO.steps
    independent = Step(id="independent", label="Other", agent="eda", request="Summarize source")
    pipeline = Pipeline(name="Branches", steps=(clean, summarize, independent))
    source = Source("Rows", "dataframe", pd.DataFrame({"value": [1, 2]}))
    called = []

    def run(step: Step, data: Data) -> AgentResult:
        called.append(step.id)
        assert isinstance(data, pd.DataFrame)
        return AgentResult.from_run(
            agent=step.agent,
            output=DataFrameOutput.save(data, tmp_path / f"{step.id}.parquet"),
            review_reason="Inspect the clean-up" if step.id == "clean" else None,
        )

    results = execute_pipeline(pipeline, source, run)
    assert results.status == Status.NEEDS_REVIEW
    assert called == ["clean", "independent"]
    assert results.steps[1].status == "blocked"
    assert results.steps[2].status == "success"


@pytest.mark.parametrize("name", ["../escape.csv", "nested\\file.csv", "NUL.csv", "bad:alt.csv"])
def test_upload_rejects_paths_and_reserved_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="plain filenames"):
        uploaded_source("Folder", [(name, b"amount\n1\n")], tmp_path)
    assert not list(tmp_path.iterdir())


def test_upload_reads_table_and_stages_folder(tmp_path: Path) -> None:
    csv = b"amount\n8\n"
    source = uploaded_source("Table", [("rows.csv", csv)], tmp_path)
    assert source.kind == "dataframe" and isinstance(source.data, pd.DataFrame)
    assert source.data["amount"].tolist() == [8]
    folder = uploaded_source("Folder", [("rows.csv", csv), ("more.csv", csv)], tmp_path)
    assert folder.kind == "files" and isinstance(folder.data, Path)
    assert (folder.data / "rows.csv").read_bytes() == csv


def test_studio_demo_in_streamlit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(repo / "src/dataquorum/app/studio.py"), default_timeout=20).run()
    assert not app.exception
    run = next(button for button in app.button if button.label == "Run pipeline")
    run.click().run()
    assert not app.exception
    assert any(node.label == "Chart by region · Verified" for node in app.expander)
    for page in ("Data", "Charts", "Model", "Runs"):
        app.radio(key="navigation").set_value(page).run()
        assert not app.exception
        assert app.title[0].value == page
    app.radio(key="navigation").set_value("Pipeline").run()
    next(button for button in app.button if button.label == "Add step").click().run()
    request = next(field for field in app.text_area if (field.key or "").startswith("new-request-"))
    request.set_value("Inspect the source values.")
    next(button for button in app.button if button.label == "Create step").click().run()
    assert not app.exception
    assert app.session_state["workspace"].pipeline.steps[-1].request == "Inspect the source values."
