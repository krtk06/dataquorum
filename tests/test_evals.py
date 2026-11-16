"""Eval suite: faults, checkers, scoring, task files and the runner (no network, no data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from engine_fakes import InProcessExecutor, ScriptedLLM

from dataquorum.agents import AGENTS
from dataquorum.core.engine import GeneratedCode, RecommendedSteps
from dataquorum.evals import datasets
from dataquorum.evals.checkers import CHECKERS, CheckOutcome, run_checker
from dataquorum.evals.faults import FAULTS, FaultSpec, apply_faults
from dataquorum.evals.oracle import check_task
from dataquorum.evals.runner import run_eval
from dataquorum.evals.scoring import classify, summarise
from dataquorum.evals.tasks import Task, build, load_tasks

CLEAN = pd.DataFrame(
    {
        "region": ["north", "south", "north", "east", "south", "east", "north", "west"] * 5,
        "amount": [120.0, 80.5, 99.9, 4500.0, 60.0, 30.25, 1750.0, 52.0] * 5,
        "units": [3, 2, 2, 1, 2, 1, 3, 1] * 5,
    }
)
CLEAN["id"] = range(len(CLEAN))


@pytest.fixture(autouse=True)
def synthetic_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets, "load", lambda name: CLEAN.copy())


def task(**kw: object) -> Task:
    base: dict[str, object] = {"id": "t", "agent": "data_cleaning", "dataset": "x", "rows": 40,
                               "request": "r", "checks": [{"kind": "row_count"}]}  # fmt: skip
    return Task.model_validate({**base, **kw})


# ------------------------------------------------------------------ faults
def test_faults_are_seeded_and_expected_is_the_repair() -> None:
    specs = [
        FaultSpec(kind="currency_text", column="amount"),
        FaultSpec(kind="duplicates", frac=0.1),
    ]
    a, b = apply_faults(CLEAN, specs, seed=1), apply_faults(CLEAN, specs, seed=1)
    pd.testing.assert_frame_equal(a.messy, b.messy)
    assert a.messy["amount"].iloc[0] == "$120.00" and len(a.messy) == 44
    assert a.expected["amount"].tolist() == CLEAN["amount"].tolist()


def test_placeholders_median_fills_with_median_of_known_values() -> None:
    f = apply_faults(CLEAN, [FaultSpec(kind="placeholders", column="amount", expect="median")], 3)
    damaged = f.messy["amount"].isin(["n/a", "N/A", "", "-"])
    assert damaged.sum() == 6
    median = CLEAN.loc[~damaged, "amount"].median()
    assert (f.expected.loc[damaged, "amount"] == median).all()


def test_thousands_suffix_and_outliers() -> None:
    f = apply_faults(CLEAN, [FaultSpec(kind="thousands_suffix", column="amount")], 0)
    assert f.messy["amount"].iloc[3] == "4.5k" and f.expected["amount"].iloc[3] == 4500.0
    g = apply_faults(CLEAN, [FaultSpec(kind="outliers", column="units", frac=0.1)], 0)
    assert g.facts["units_threshold"] == "30"
    hit = g.messy["units"] > 30
    assert hit.sum() == 4 and g.expected.loc[hit, "units"].isna().all()


def test_category_variants_keep_meaning() -> None:
    f = apply_faults(CLEAN, [FaultSpec(kind="category_variants", column="region", frac=0.5)], 2)
    assert (f.messy["region"].str.strip().str.lower() == CLEAN["region"]).all()
    assert (f.messy["region"] != CLEAN["region"]).sum() > 5
    assert f.expected["region"].tolist() == CLEAN["region"].tolist()


def test_unknown_fault_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fault kind"):
        apply_faults(CLEAN, [FaultSpec(kind="nope", column="amount")], 0)


# ------------------------------------------------------------------ checkers
def test_matches_expected_reports_first_wrong_value() -> None:
    b = build(task(faults=[{"kind": "currency_text", "column": "amount"}]))
    good = run_checker("matches_expected", b, b.expected, {"columns": ["amount"]})
    assert good.passed
    wrong = b.expected.assign(amount=b.expected["amount"] + 1)
    bad = run_checker("matches_expected", b, wrong, {"columns": ["amount"]})
    assert not bad.passed and "40 wrong" in bad.detail
    text = run_checker("matches_expected", b, b.messy, {"columns": ["amount"]})
    assert not text.passed and "not numeric" in text.detail


def test_aggregate_matches_by_value_not_column_name() -> None:
    b = build(task(agent="data_wrangling"))
    totals = CLEAN.groupby("region")["amount"].sum().reset_index()
    renamed = totals.rename(columns={"amount": "total_amount_usd"}).sort_values("total_amount_usd")
    params = {"group_by": "region", "value": "amount", "agg": "sum"}
    assert run_checker("aggregate", b, renamed, params).passed
    assert not run_checker("aggregate", b, renamed, {**params, "order": "desc"}).passed
    assert not run_checker("aggregate", b, renamed.iloc[:2], params).passed
    assert not run_checker("aggregate", b, CLEAN, params).passed


def test_figure_aggregate_reads_typed_arrays_and_histograms() -> None:
    b = build(task(agent="data_visualization"))
    totals = CLEAN.groupby("region", as_index=False)["amount"].sum()
    fig = json.loads(px.bar(totals, x="region", y="amount").to_json())
    assert run_checker("figure_aggregate", b, fig, {"group_by": "region", "value": "amount"}).passed
    hist = json.loads(px.histogram(CLEAN, x="region").to_json())
    params = {"group_by": "region", "value": "amount", "agg": "count"}
    assert run_checker("figure_aggregate", b, hist, params).passed
    assert not run_checker("figure_aggregate", b, fig, params).passed


def test_broken_or_unknown_checker_fails() -> None:
    b = build(task())
    assert not run_checker("nope", b, CLEAN, {}).passed
    assert not run_checker("matches_expected", b, CLEAN, {"bad_param": 1}).passed


# ------------------------------------------------------------------ scoring
@pytest.mark.parametrize(
    ("status", "expect", "passed", "outcome"),
    [
        ("success", "ok", True, "correct"),
        ("needs_review", "ok", True, "correct"),
        ("success", "ok", False, "false_success"),
        ("needs_review", "ok", False, "flagged"),
        ("failed", "ok", False, "honest_failure"),
        ("failed", "ok", True, "false_failure"),
        ("failed", "failed", True, "correct"),
        ("success", "failed", False, "false_success"),
    ],
)
def test_classify(status: str, expect: str, passed: bool, outcome: str) -> None:
    assert classify(status, expect, [CheckOutcome("c", passed, "")]) == outcome


# ------------------------------------------------------------------ task files
def test_task_files_are_consistent() -> None:
    all_tasks = load_tasks()
    assert len(all_tasks) >= 50
    for t in all_tasks:
        assert t.agent in AGENTS, t.id
        assert t.dataset in datasets.DATASETS, t.id
        assert all(f.kind in FAULTS for f in t.faults), t.id
        assert all(c.kind in CHECKERS for c in t.checks), t.id
    assert sum(t.expect_status == "failed" for t in all_tasks) >= 3


@pytest.mark.skipif(
    not (datasets.data_dir() / "adult.parquet").exists(), reason="datasets not cached"
)
def test_every_real_task_passes_the_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # use the real cached datasets
    problems = {t.id: p for t in load_tasks() if (p := check_task(build(t)))}
    assert problems == {}


# ------------------------------------------------------------------ runner
def test_runner_scores_and_records_outcomes(tmp_path: Path) -> None:
    fix = "import pandas as pd\ndef transform(df):\n    return df.assign(amount=pd.to_numeric(df['amount'].str.replace('$', '').str.replace(',', '')))\n"
    wrong = "def transform(df):\n    return df\n"
    good_task = task(id="good", faults=[{"kind": "currency_text", "column": "amount"}],
                     checks=[{"kind": "matches_expected", "columns": ["amount"]}])  # fmt: skip
    bad_task = good_task.model_copy(update={"id": "bad"})
    steps = RecommendedSteps(steps=["parse amount"])
    llm = ScriptedLLM(
        steps, GeneratedCode(python_code=fix), steps, GeneratedCode(python_code=wrong)
    )
    out = tmp_path / "eval"
    out.mkdir()
    results, stopped = run_eval(
        [good_task, bad_task], llm_factory=lambda: llm, executor=InProcessExecutor(),
        recorder=None, out_dir=out, workers=1, max_retries=0,
    )  # fmt: skip
    assert stopped is None
    by_id = {r.task_id: r for r in results}
    assert by_id["good"].outcome == "correct"
    # Returning the text unchanged: the engine's own checks pass, but the answer is wrong.
    assert by_id["bad"].outcome == "false_success"
    assert summarise(results)["overall"]["false_success"] == 1
    assert len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert np.isclose(sum(r.allowance_usd for r in results), 0.004)
