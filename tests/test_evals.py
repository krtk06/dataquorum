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
from dataquorum.evals.checkers import CheckOutcome, run_checker
from dataquorum.evals.checkers_ml import all_checkers
from dataquorum.evals.faults import FAULTS, FaultSpec, apply_faults
from dataquorum.evals.oracle import check_task
from dataquorum.evals.runner import RunOptions, run_eval
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
    assert isinstance(b.messy, pd.DataFrame)
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
        assert all(c.kind in all_checkers() for c in t.checks), t.id
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
        recorder=None, out_dir=out, workers=1, options=RunOptions(max_retries=0),
    )  # fmt: skip
    assert stopped is None
    by_id = {r.task_id: r for r in results}
    assert by_id["good"].outcome == "correct"
    # Returning the text unchanged: the engine's own checks pass, but the answer is wrong.
    assert by_id["bad"].outcome == "false_success"
    assert summarise(results)["overall"]["false_success"] == 1
    assert len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert np.isclose(sum(r.allowance_usd for r in results), 0.004)


def test_repeats_and_ablations(tmp_path: Path) -> None:
    fix = (
        "import pandas as pd\ndef transform(df):\n    return df.assign("
        "amount=pd.to_numeric(df['amount'].str.replace('$', '').str.replace(',', '')))\n"
    )
    noop = "def transform(df):\n    return df\n"
    t = task(id="rep", faults=[{"kind": "currency_text", "column": "amount"}],
             checks=[{"kind": "matches_expected", "columns": ["amount"]}])  # fmt: skip
    # --no-recommend: no RecommendedSteps calls; --repeats 3: three attempts, one wrong.
    llm = ScriptedLLM(GeneratedCode(python_code=fix), GeneratedCode(python_code=fix),
                      GeneratedCode(python_code=noop))  # fmt: skip
    out = tmp_path / "eval"
    out.mkdir()
    options = RunOptions(max_retries=0, no_checks=True, no_recommend=True, repeats=3)
    results, _ = run_eval([t], llm_factory=lambda: llm, executor=InProcessExecutor(),
                          recorder=None, out_dir=out, workers=1, options=options)  # fmt: skip
    assert sorted(r.repeat for r in results) == [0, 1, 2]
    assert all(r.calls == 1 for r in results)  # one GeneratedCode call each, no recommend
    assert llm.prompts_for("RecommendedSteps") == []
    assert "No step plan was made" in llm.prompts_for("GeneratedCode")[0]
    summary = summarise(results)
    assert summary["overall"]["correct"] == 2
    assert summary["pass_at_k"]["pass@1"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["pass_at_k"]["pass@2"] == 1.0


def test_compare_two_runs(tmp_path: Path) -> None:
    from dataquorum.evals.compare import compare

    def write_run(name: str, outcomes: dict[str, str], cost: float) -> Path:
        d = tmp_path / name
        d.mkdir()
        meta = {"eval_id": name, "label": name, "model": "m", "reasoning_effort": "low",
                "options": {"max_retries": 2, "no_checks": name == "b"}}  # fmt: skip
        (d / "summary.json").write_text(json.dumps(meta), encoding="utf-8")
        rows = [{"task_id": t, "outcome": o, "allowance_usd": cost, "seconds": 1.0}
                for t, o in outcomes.items()]  # fmt: skip
        lines = [json.dumps(r) for r in rows]
        (d / "results.jsonl").write_text(chr(10).join(lines), encoding="utf-8")
        return d

    a = write_run("a", {f"t{i}": "correct" for i in range(10)}, 0.001)
    b = write_run("b", {**{f"t{i}": "correct" for i in range(6)},
                        **{f"t{i}": "false_success" for i in range(6, 10)}}, 0.002)  # fmt: skip
    report = compare(a, b)
    assert "10/10 (100%" in report and "6/10 (60%" in report
    assert "only A correct = 4, only B correct = 0, p = 0.125" in report
    assert "B - A cost per task: +0.001" in report
    assert "- t6: A ['correct'] -> B ['false_success']" in report
    assert "no_checks" in report


def test_category_variants_keep_the_original_spelling_a_strict_majority() -> None:
    frame = pd.DataFrame({"label": ["rare"] + ["two"] * 2 + ["five"] * 5 + ["common"] * 32})
    f = apply_faults(frame, [FaultSpec(kind="category_variants", column="label", frac=0.9)], 0)
    for label in frame["label"].unique():
        spellings = f.messy["label"][f.messy["label"].str.strip().str.lower() == label]
        exact = int((spellings == label).sum())
        assert exact > len(spellings) - exact, (label, spellings.tolist())
    assert (f.messy["label"] == "rare").sum() == 1  # a single label is never varied


def test_oracle_flags_a_label_that_only_appears_as_a_variant() -> None:
    from dataquorum.evals.oracle import label_problems
    from dataquorum.evals.tasks import BuiltTask

    t = task(checks=[{"kind": "matches_expected", "columns": ["region"]}])
    clean = pd.DataFrame({"region": ["Rare_Place", "north", "north"]})
    messy = pd.DataFrame({"region": ["rare_place", "north", "north"]})
    problems = label_problems(BuiltTask(t, clean, messy, clean, "r"))
    assert problems and "Rare_Place" in problems[0]
    assert label_problems(BuiltTask(t, clean, clean, clean, "r")) == []


def test_write_files_builds_parts_decoys_and_expected(tmp_path: Path) -> None:
    from dataquorum.evals.inputs import FileSpec, write_files

    specs = [
        FileSpec(path="b/2.csv", rows=(20, 40), sep=";", preamble=["note"]),
        FileSpec(path="b/1.csv", rows=(0, 20)),
        FileSpec(path="old.csv", decoy=True),
        FileSpec(path="README.txt", format="text", text="hello"),
    ]
    expected = write_files(specs, CLEAN, tmp_path)
    assert expected["id"].tolist() == list(range(40))  # file-name order: b/1 then b/2
    assert (tmp_path / "b" / "2.csv").read_text(encoding="utf-8").startswith("note\nregion;")
    assert (tmp_path / "README.txt").read_text(encoding="utf-8") == "hello"
    decoy = pd.read_csv(tmp_path / "old.csv")
    assert decoy["amount"].min() > 1000


def test_write_database_with_lookup_needs_a_join(tmp_path: Path) -> None:
    import sqlite3

    from dataquorum.evals.inputs import DatabaseSpec, LookupSpec, write_database

    db = tmp_path / "x.db"
    write_database(
        DatabaseSpec(table="sales", lookups=[LookupSpec(column="region", table="regions")]),
        CLEAN,
        db,
    )
    con = sqlite3.connect(db)
    try:
        columns = [r[1] for r in con.execute("PRAGMA table_info(sales)")]
        joined = con.execute(
            "SELECT r.name, SUM(s.amount) FROM sales s JOIN regions r ON r.id = s.region_id GROUP BY r.name ORDER BY r.name"
        ).fetchall()
    finally:
        con.close()
    assert "region" not in columns and "region_id" in columns
    want = CLEAN.groupby("region")["amount"].sum().sort_index()
    assert [name for name, _ in joined] == list(want.index)
    assert np.allclose([total for _, total in joined], want.to_numpy())
