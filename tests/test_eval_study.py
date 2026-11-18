"""Offline evidence that the Week 14 runner measures the intended changes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from engine_fakes import InProcessExecutor, ScriptedLLM, llm_failure

from dataquorum.core.budget import BudgetExceededError
from dataquorum.core.cost import Usage
from dataquorum.core.engine import GeneratedCode, RecommendedSteps
from dataquorum.core.findings import Findings
from dataquorum.core.llm import DEFAULT_BASE_URL, LLMConfig, Message, StructuredResult
from dataquorum.evals import study
from dataquorum.evals.runner import RunOptions, run_task
from dataquorum.evals.study import ARMS, StudyBudget, make_plan, ordered_trials, study_tasks
from dataquorum.evals.study_report import matched_comparison, record_trial
from dataquorum.evals.tasks import Task
from dataquorum.tracing.recorder import RunRecorder

PLAN = RecommendedSteps(steps=["Produce the requested answer"])


def synthetic_task(agent: str, request: str, check: dict[str, object], expect: str = "ok") -> Task:
    return Task.model_validate(
        {
            "id": "scripted",
            "agent": agent,
            "dataset": "audit-fixture",
            "columns": ["signal", "label"],
            "request": request,
            "checks": [check],
            "expect_status": expect,
        }
    )


def test_plan_pairs_each_of_eight_agents_with_every_arm() -> None:
    tasks = study_tasks()
    assert len(tasks) == 16
    assert len(Counter(t.agent for t in tasks)) == 8
    assert set(Counter(t.agent for t in tasks).values()) == {2}
    assert len({a.model for a in ARMS}) == 5
    plan = make_plan(tasks)
    assert len({tuple(key) for key in plan["order"]}) == 112
    assert ordered_trials(tasks) == ordered_trials(tasks)
    assert not ARMS[0].options().no_checks and ARMS[1].options().no_checks
    assert ARMS[2].options().no_recommend


def test_no_checks_can_turn_omitted_leakage_into_false_success(tmp_path: Path) -> None:
    task = next(t for t in study_tasks() if t.id == "audit-categorical-proxy")
    omitted = GeneratedCode(
        python_code=(
            "from dataquorum.ml.correctness import audit\n"
            "def transform(df):\n"
            "    return audit(df.drop(columns='code'), 'label', problem='classification')"
        )
    )
    fixed = GeneratedCode(
        python_code=(
            "from dataquorum.ml.correctness import audit\n"
            "def transform(df): return audit(df, 'label', problem='classification')"
        )
    )
    for options in (ARMS[0].options(), ARMS[1].options()):
        replies = (PLAN, omitted) if options.no_checks else (PLAN, omitted, fixed)
        scored, result = run_task(
            task,
            ScriptedLLM(*replies),
            InProcessExecutor(),
            tmp_path,
            f"ablation-{options.no_checks}",
            options,
        )
        assert scored.outcome == ("false_success" if options.no_checks else "correct")
        assert scored.retries == (0 if options.no_checks else 1)
        if options.no_checks:
            row = record_trial("ablated", scored, result, 1.0)
            assert row["failure_stage"] == "oracle_mismatch"


def test_no_recommend_can_turn_missing_column_into_false_success(tmp_path: Path) -> None:
    task = synthetic_task(
        "data_wrangling", "Return the mean of absent.", {"kind": "no_output"}, "failed"
    )
    plan = RecommendedSteps(steps=[], blocker="Missing absent.")
    invented = GeneratedCode(python_code="def transform(df): return df.assign(absent=12)")
    rows = []
    for arm, reply in ((ARMS[0], plan), (ARMS[2], invented)):
        scored, result = run_task(
            task, ScriptedLLM(reply), InProcessExecutor(), tmp_path, arm.name, arm.options()
        )
        rows.append(record_trial(arm.name, scored, result, 1.0))
    assert [(r["outcome"], r["failure_stage"]) for r in rows] == [
        ("correct", "declared_blocker"),
        ("false_success", "oracle_mismatch"),
    ]
    report = matched_comparison(rows, [ARMS[0].name, ARMS[2].name], [task.id, "unfinished"])
    assert report["matched_task_ids"] == [task.id]
    assert report["arms"][ARMS[0].name]["correct"] == 1
    assert report["arms"][ARMS[2].name]["false_success"] == 1


def test_no_checks_disables_grounding_and_research_measures_it(tmp_path: Path) -> None:
    task = synthetic_task(
        "eda",
        "Report the mean of signal.",
        {
            "kind": "per_column_stat",
            "columns": ["signal"],
            "stat": "mean",
        },
    )
    code = GeneratedCode(
        python_code=(
            "import pandas as pd\n"
            "def transform(df):\n"
            "    return pd.DataFrame({'column':['signal'], 'mean':[df.signal.mean()]})"
        )
    )
    for arm, status, outcome in (
        (ARMS[0], "needs_review", "flagged"),
        (ARMS[1], "success", "false_success"),
    ):
        llm = ScriptedLLM(PLAN, code, Findings(findings=["The mean is 4139."]))
        scored, result = run_task(task, llm, InProcessExecutor(), tmp_path, arm.name, arm.options())
        assert scored.status == status and scored.outcome == "correct"
        assert scored.calls == 3  # Both arms still ask for findings.
        row = record_trial(arm.name, scored, result, 1.0)
        assert row["oracle_output_passed"] and row["outcome"] == outcome
        assert not row["verified_correct"]


def test_provider_failure_is_not_a_verified_blocker(tmp_path: Path) -> None:
    task = synthetic_task("data_wrangling", "Return absent.", {"kind": "no_output"}, "failed")
    scored, result = run_task(
        task,
        ScriptedLLM(llm_failure()),
        InProcessExecutor(),
        tmp_path,
        "provider",
        RunOptions(max_retries=0),
    )
    row = record_trial("full", scored, result, 1.0)
    assert row["legacy_outcome"] == "correct"
    assert row["failure_stage"] == "upstream_llm"
    assert not row["verified_correct"]


def test_study_stop_uses_its_persisted_spend_across_resumes(tmp_path: Path) -> None:
    recorder = RunRecorder(tmp_path / "spend.duckdb")
    try:
        guard = StudyBudget(recorder, tmp_path, 0.60)
        guard.check()
        guard.record(Usage(models=["m"], allowance_usd=0.47))
        guard.check()
        guard.record(Usage(models=["m"], allowance_usd=0.14))
        resumed = StudyBudget(recorder, tmp_path, 0.60)
        assert resumed.total() == pytest.approx(0.61)
        assert resumed.spent()[0] == pytest.approx(0.61)
        with pytest.raises(BudgetExceededError, match="Study allowance"):
            resumed.check()
    finally:
        recorder.close()


def test_study_resume_preserves_distinct_arm_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = next(t for t in study_tasks() if t.id == "audit-clean")
    fixed = GeneratedCode(
        python_code="from dataquorum.ml.correctness import audit\n"
        "def transform(df): return audit(df, 'label', problem='classification')"
    )
    calls: list[str] = []

    def fake_client(config: LLMConfig, *, budget: StudyBudget, session_id: str) -> ScriptedLLM:
        class Client(ScriptedLLM):
            def structured(
                self, messages: list[Message], schema: type[Any], **kwargs: Any
            ) -> StructuredResult[Any]:
                calls.append(session_id)
                budget.check()
                budget.record(Usage(models=[config.model], allowance_usd=0.001))
                return super().structured(messages, schema, **kwargs)

        return Client(fixed) if session_id.endswith("no-recommend") else Client(PLAN, fixed)

    class LocalExecutor(InProcessExecutor):
        image = "test-image"

        @property
        def client(self) -> SimpleNamespace:
            return SimpleNamespace(
                ping=lambda: True, images=SimpleNamespace(get=lambda _: SimpleNamespace(id="test"))
            )

    monkeypatch.setattr(study, "study_tasks", lambda: [task])
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "unused-test-placeholder")
    monkeypatch.setenv("DQ_BASE_URL", DEFAULT_BASE_URL)
    monkeypatch.setattr(study, "DockerExecutor", LocalExecutor)
    monkeypatch.setattr(study, "LLMClient", fake_client)
    monkeypatch.setattr(study, "RunRecorder", lambda: RunRecorder(tmp_path / "runs.duckdb"))
    args = ["run", "--out", str(tmp_path), "--max-allowance", "0.60"]
    assert study.main(args) == 0
    assert study.main(args) == 0
    assert len(calls) == 13
    with (tmp_path / "trials.jsonl").open(encoding="utf-8") as log:
        rows = [json.loads(line) for line in log]
    assert len(rows) == len({r["run_id"] for r in rows}) == len(ARMS)
    summary = study.report(tmp_path)
    assert summary["completed_trials"] == len(ARMS)
    assert summary["metered_allowance_usd"] == pytest.approx(0.013)
