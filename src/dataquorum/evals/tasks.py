"""Eval task schema, YAML loading, and building a task's input frames.

A task file holds shared `defaults` and a list of `tasks`:

    defaults: {agent: data_cleaning, dataset: adult, rows: 300}
    tasks:
      - id: clean-adult-capital-gain-currency
        columns: [age, workclass, capital-gain]
        faults: [{kind: currency_text, column: capital-gain}]
        request: Convert capital-gain from dollar text to a number. Keep every row.
        checks:
          - {kind: matches_expected, columns: [capital-gain]}
          - {kind: row_count}

Requests may use `{name}` placeholders filled from fault facts (e.g. `{fnlwgt_threshold}`).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from dataquorum.evals import datasets
from dataquorum.evals.faults import FaultSpec, apply_faults
from dataquorum.evals.inputs import DatabaseSpec, FileSpec, write_database, write_files

TASKS_DIR = Path(__file__).parent / "tasks"
ExpectStatus = Literal["ok", "failed", "success", "needs_review"]  # ok = success or needs_review


class CheckSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class Task(BaseModel):
    id: str
    agent: str
    dataset: str
    request: str
    checks: list[CheckSpec] = Field(min_length=1)
    rows: int = 300
    columns: list[str] | None = None
    seed: int = 7
    faults: list[FaultSpec] = []
    expect_status: ExpectStatus = "ok"
    tags: list[str] = []
    files: list[FileSpec] = []  # loader tasks: the input is a folder of these files
    database: DatabaseSpec | None = None  # SQL tasks: the input is a SQLite file


class _TaskFile(BaseModel):
    defaults: dict[str, Any] = {}
    tasks: list[dict[str, Any]]


def load_tasks(paths: Iterable[Path] | None = None) -> list[Task]:
    files = sorted(paths) if paths is not None else sorted(TASKS_DIR.glob("*.yaml"))
    tasks: list[Task] = []
    for path in files:
        raw = _TaskFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for item in raw.tasks:
            merged = {**raw.defaults, **item}
            merged.setdefault("tags", [])
            merged["tags"] = [*raw.defaults.get("tags", []), *item.get("tags", [])]
            tasks.append(Task.model_validate(merged))
    ids = [t.id for t in tasks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate task ids: {dupes}")
    return tasks


@dataclass(frozen=True)
class BuiltTask:
    task: Task
    clean: pd.DataFrame  # sampled, before faults
    messy: pd.DataFrame | Path  # what the agent gets: a frame, a folder or a database
    expected: pd.DataFrame  # what a correct cleaning would return
    request: str  # with fault facts filled in


def build(
    task: Task,
    loader: Callable[[str], pd.DataFrame] | None = None,
    workdir: Path | None = None,
) -> BuiltTask:
    frame = (loader or datasets.load)(task.dataset)  # looked up at call time (tests patch it)
    if task.columns:
        missing = [c for c in task.columns if c not in frame.columns]
        if missing:
            raise ValueError(f"{task.id}: dataset {task.dataset} has no columns {missing}")
        frame = frame[task.columns]
    frame = frame.drop_duplicates()  # so injected duplicates are the only ones
    n = min(task.rows, len(frame))
    clean = frame.sample(n=n, random_state=task.seed).reset_index(drop=True)
    faulted = apply_faults(clean, task.faults, task.seed)
    request = task.request.format(**faulted.facts) if faulted.facts else task.request
    if task.files or task.database:
        root = workdir or Path(tempfile.mkdtemp(prefix=f"dq-{task.id}-"))
        if task.files:
            folder = root / "input"
            expected = write_files(task.files, faulted.messy, folder)
            return BuiltTask(task, clean, folder, expected, request)
        assert task.database is not None
        db = root / "input.db"
        write_database(task.database, faulted.messy, db)
        return BuiltTask(task, clean, db, faulted.expected, request)
    return BuiltTask(task, clean, faulted.messy, faulted.expected, request)
