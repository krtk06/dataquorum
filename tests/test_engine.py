"""Engine behaviour with a scripted LLM and an in-process executor (no network, no Docker)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest
from engine_fakes import InProcessExecutor, ScriptedLLM, llm_failure

from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.engine import Engine, GeneratedCode, RecommendedSteps
from dataquorum.core.state import DataFrameOutput
from dataquorum.core.statuses import Status

SPEC = AgentSpec(
    name="test_cleaning",
    role="You are a data-cleaning agent.",
    recommend_guidance="Recommend cleaning steps.",
    code_guidance="Prefer vectorised pandas.",
)
REQUEST = "Make income numeric and title-case city. Keep all rows."
STEPS = RecommendedSteps(steps=["parse income", "title-case city"])

GOOD = GeneratedCode(
    python_code=(
        "import pandas as pd\n"
        "def transform(df):\n"
        "    out = df.copy()\n"
        "    out['income'] = pd.to_numeric(out['income'].str.replace('$', '', regex=False),"
        " errors='coerce')\n"
        "    out['city'] = out['city'].str.strip().str.title()\n"
        "    return out\n"
    )
)
CRASHES = GeneratedCode(python_code="def transform(df):\n    return df['no_such_column']\n")
EMPTY = GeneratedCode(python_code="def transform(df):\n    return df.iloc[0:0]\n")
DROPS_COLUMN = GeneratedCode(
    python_code="def transform(df):\n    return df.drop(columns=['city'])\n"
)


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame({"income": ["$1200", "$3400", "n/a"], "city": ["boston ", "NYC", "la"]})


def make_engine(llm: ScriptedLLM, tmp_path: Path, max_retries: int = 3) -> Engine:
    return Engine(SPEC, llm, InProcessExecutor(), runs_dir=tmp_path, max_retries=max_retries)


def test_happy_path_is_success(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(STEPS, GOOD)
    run = make_engine(llm, tmp_path).start(data, REQUEST)

    assert not run.paused and run.result is not None
    result = run.result
    assert result.status is Status.SUCCESS
    assert result.run_id == run.run_id
    assert result.steps == STEPS.steps
    assert result.code == GOOD.python_code
    assert (result.usage.calls, result.usage.retries) == (2, 0)
    assert isinstance(result.output, DataFrameOutput)
    out = result.output.load()
    assert out["city"].tolist() == ["Boston", "Nyc", "La"]
    assert out["income"].tolist()[:2] == [1200.0, 3400.0]


def test_request_is_passed_verbatim_and_summary_is_included(
    data: pd.DataFrame, tmp_path: Path
) -> None:
    llm = ScriptedLLM(STEPS, GOOD)
    make_engine(llm, tmp_path).start(data, REQUEST)
    recommend_prompt = llm.prompts_for("RecommendedSteps")[0]
    assert f"User request (verbatim):\n{REQUEST}" in recommend_prompt
    assert "Shape: 3 rows x 2 columns" in recommend_prompt
    assert "1. parse income" in llm.prompts_for("GeneratedCode")[0]


def test_crash_is_fed_back_and_fixed(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(STEPS, CRASHES, GOOD)
    result = make_engine(llm, tmp_path).start(data, REQUEST).result

    assert result is not None and result.status is Status.SUCCESS
    assert result.usage.retries == 1
    fix_prompt = llm.prompts_for("GeneratedCode")[1]
    assert "KeyError" in fix_prompt
    assert CRASHES.python_code in fix_prompt


def test_retries_exhausted_is_failed_with_error_and_usage(
    data: pd.DataFrame, tmp_path: Path
) -> None:
    llm = ScriptedLLM(STEPS, CRASHES, CRASHES, CRASHES)
    result = make_engine(llm, tmp_path, max_retries=2).start(data, REQUEST).result

    assert result is not None and result.status is Status.FAILED
    assert result.error is not None and "KeyError" in result.error
    assert result.output is None
    assert (result.usage.calls, result.usage.retries) == (4, 2)  # bug log #5


def test_empty_output_is_retried_then_failed(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(STEPS, EMPTY, EMPTY)
    result = make_engine(llm, tmp_path, max_retries=1).start(data, REQUEST).result

    assert result is not None and result.status is Status.FAILED  # bug log #6
    assert "has_rows_and_columns" in llm.prompts_for("GeneratedCode")[1]


def test_warning_gives_needs_review_without_retry(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(STEPS, DROPS_COLUMN)
    result = make_engine(llm, tmp_path).start(data, REQUEST).result

    assert result is not None and result.status is Status.NEEDS_REVIEW
    assert result.error is not None and "columns_dropped" in result.error
    assert result.usage.retries == 0


def test_llm_failure_while_recommending_is_failed(data: pd.DataFrame, tmp_path: Path) -> None:
    result = make_engine(ScriptedLLM(llm_failure()), tmp_path).start(data, REQUEST).result
    assert result is not None and result.status is Status.FAILED
    assert result.error is not None and "Recommending steps failed" in result.error
    assert result.usage.calls == 1  # the failed call's spend is still counted


def test_review_approve_then_finish(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(STEPS, GOOD)
    engine = make_engine(llm, tmp_path)
    paused = engine.start(data, REQUEST, human_review=True)

    assert paused.paused and paused.review is not None
    assert paused.review["steps"] == STEPS.steps
    assert paused.review["request"] == REQUEST
    assert llm.prompts_for("GeneratedCode") == []  # nothing generated before approval

    done = engine.resume(paused.run_id, "Yes.")  # bug log #7: "Yes." is an approval
    assert done.result is not None and done.result.status is Status.SUCCESS


def test_review_revise_reruns_recommend_with_separate_notes(
    data: pd.DataFrame, tmp_path: Path
) -> None:
    llm = ScriptedLLM(STEPS, RecommendedSteps(steps=["parse income", "lower-case city"]), GOOD)
    engine = make_engine(llm, tmp_path)
    paused = engine.start(data, REQUEST, human_review=True)

    again = engine.resume(paused.run_id, "Use lower case for city instead.")
    assert again.paused and again.review is not None
    assert again.review["review_notes"] == ["Use lower case for city instead."]
    second_prompt = llm.prompts_for("RecommendedSteps")[1]
    assert f"User request (verbatim):\n{REQUEST}" in second_prompt  # request untouched
    assert "1. Use lower case for city instead." in second_prompt

    done = engine.resume(paused.run_id, {"kind": "approve"})
    assert done.result is not None and done.result.steps == ["parse income", "lower-case city"]


def test_review_reject_ends_failed_without_generating(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(STEPS)
    engine = make_engine(llm, tmp_path)
    paused = engine.start(data, REQUEST, human_review=True)

    done = engine.resume(paused.run_id, "no: wrong columns")
    assert done.result is not None and done.result.status is Status.FAILED
    assert done.result.error == "Rejected by reviewer: wrong columns"
    assert llm.prompts_for("GeneratedCode") == []


def test_invalid_reply_raises_and_run_stays_paused(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(STEPS, GOOD)
    engine = make_engine(llm, tmp_path)
    paused = engine.start(data, REQUEST, human_review=True)

    with pytest.raises(ValueError, match="Empty review reply"):
        engine.resume(paused.run_id, "   ")
    done = engine.resume(paused.run_id, "approve")
    assert done.result is not None and done.result.status is Status.SUCCESS


def test_checkpoint_types_are_registered(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(STEPS, GOOD)
    engine = make_engine(llm, tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        paused = engine.start(data, REQUEST, human_review=True)
        engine.resume(paused.run_id, "ok")
    unregistered = [str(w.message) for w in caught if "unregistered type" in str(w.message)]
    assert unregistered == []


BLOCKED = RecommendedSteps(steps=[], blocker="The data has no product_category column.")


def test_blocker_stops_before_any_code_is_generated(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(BLOCKED)
    result = make_engine(llm, tmp_path).start(data, "revenue by product category").result

    assert result is not None and result.status is Status.FAILED
    assert result.error == (
        "Cannot do this with the given data: The data has no product_category column."
    )
    assert llm.prompts_for("GeneratedCode") == []
    assert (result.usage.calls, result.usage.retries) == (1, 0)


def test_blocker_skips_human_review(data: pd.DataFrame, tmp_path: Path) -> None:
    run = make_engine(ScriptedLLM(BLOCKED), tmp_path).start(data, "x", human_review=True)
    assert not run.paused
    assert run.result is not None and run.result.status is Status.FAILED


def test_blank_blocker_is_ignored(data: pd.DataFrame, tmp_path: Path) -> None:
    llm = ScriptedLLM(RecommendedSteps(steps=STEPS.steps, blocker="  "), GOOD)
    result = make_engine(llm, tmp_path).start(data, REQUEST).result
    assert result is not None and result.status is Status.SUCCESS
