"""Loader agent: folder description and an engine run on a folder input."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from engine_fakes import InProcessExecutor, ScriptedLLM

from dataquorum.agents import LOADER
from dataquorum.core.engine import Engine, GeneratedCode, RecommendedSteps
from dataquorum.core.inputs import describe_folder, prepare
from dataquorum.core.state import DataFrameOutput
from dataquorum.core.statuses import Status


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    (d / "2025").mkdir(parents=True)
    (d / "2025" / "jan.csv").write_text(
        "# exported by tool\nid;amount\n1;10,5\n2;3,0\n", encoding="utf-8"
    )
    (d / "2025" / "feb.csv").write_text("# exported by tool\nid;amount\n3;7,25\n", encoding="utf-8")
    (d / "notes.txt").write_text("ignore me\n", encoding="utf-8")
    return d


def test_folder_description_lists_files_with_previews(folder: Path) -> None:
    text = describe_folder(folder)
    assert "3 files" in text
    assert "- 2025/jan.csv" in text and "| id;amount" in text
    assert "| # exported by tool" in text


def test_prepare_detects_input_kinds(folder: Path, tmp_path: Path) -> None:
    assert prepare(folder, tmp_path)[0] == "files"
    db = tmp_path / "x.db"
    db.write_bytes(b"")
    assert prepare(db, tmp_path)[0] == "database"
    assert prepare(pd.DataFrame({"a": [1]}), tmp_path)[0] == "dataframe"
    with pytest.raises(ValueError):
        prepare(tmp_path / "missing.csv", tmp_path)


def test_loader_engine_run(folder: Path, tmp_path: Path) -> None:
    code = (
        "import os\nimport pandas as pd\n"
        "def load(data_dir):\n"
        "    parts = [pd.read_csv(os.path.join(data_dir, '2025', f), sep=';', decimal=',', skiprows=1)\n"
        "             for f in ['feb.csv', 'jan.csv']]\n"
        "    return pd.concat(parts, ignore_index=True).sort_values('id', ignore_index=True)\n"
    )
    llm = ScriptedLLM(
        RecommendedSteps(steps=["read both CSVs", "stack"]), GeneratedCode(python_code=code)
    )
    run = Engine(LOADER, llm, InProcessExecutor(), runs_dir=tmp_path / "runs").start(
        folder, "Load all 2025 CSVs"
    )
    assert run.result is not None and run.result.status is Status.SUCCESS, run.result
    assert isinstance(run.result.output, DataFrameOutput)
    assert run.result.output.load()["amount"].tolist() == [10.5, 3.0, 7.25]
    prompt = llm.prompts_for("RecommendedSteps")[0]
    assert "2025/feb.csv" in prompt and "notes.txt" in prompt
