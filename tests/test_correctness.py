"""Known-answer and negative controls for the audit, folds and binary diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataquorum.agents import CORRECTNESS
from dataquorum.agents.correctness import requested_audit
from dataquorum.core.validation import CheckContext, run_checks
from dataquorum.evals.audit_data import audit_dataset
from dataquorum.evals.oracle import check_task
from dataquorum.evals.tasks import build, load_tasks
from dataquorum.ml.calibration import calibration
from dataquorum.ml.correctness import audit
from dataquorum.ml.tuning import validation_folds


def test_all_week11_answer_keys_work_without_openml() -> None:
    tasks = [task for task in load_tasks() if "week11" in task.tags]
    assert len(tasks) == 12
    for task in tasks:
        assert check_task(build(task)) == [], task.id


def test_incorrect_or_duplicate_recommendation_fails_validation() -> None:
    frame = audit_dataset()[["code", "label"]]
    ctx = CheckContext("Audit classification. Target is label.")
    expected = requested_audit(frame, ctx.request)
    for wrong in (expected.iloc[:2], pd.concat([expected, expected.iloc[:1]])):
        checks = run_checks(CORRECTNESS.checks, frame, wrong, ctx)
        assert not checks[0].passed
    checks = run_checks(CORRECTNESS.checks, frame, expected, ctx)
    assert checks[0].passed and not checks[1].passed


def test_holdout_labels_do_not_change_proxy_or_metric_choice() -> None:
    frame = audit_dataset()[["signal", "rare", "split"]]
    expected = audit(frame, "rare", problem="classification")
    frame.loc[frame["split"].eq("test"), "rare"] = np.nan
    pd.testing.assert_frame_equal(audit(frame, "rare", problem="classification"), expected)


def test_overlap_ignores_label_and_index() -> None:
    frame = audit_dataset()[["repeated", "label", "split"]]
    frame.index = [0] * len(frame)
    frame.iloc[120, 1] = "changed"
    assert "row_overlap" in audit(frame, "label", problem="classification")["value"].tolist()


def test_threshold_proxy_and_constant_predictor() -> None:
    frame = audit_dataset()[["signal", "label"]].assign(constant=0)
    frame["range"] = frame["signal"] + 100 * frame["label"].eq("yes")
    report = audit(frame, "label", problem="classification")
    assert list(report.loc[report["kind"].eq("risk"), "subject"]) == ["range"]


def test_explicit_roles_and_tied_dates() -> None:
    frame = audit_dataset().rename(columns={"household_id": "house hold", "label": "out come"})
    frame = frame[["signal", "house hold", "day", "out come"]].sample(frac=1, random_state=42)
    frame["day"] = frame["day"].dt.to_period("W").dt.start_time
    report = requested_audit(frame, "classification; target is 'out come', group is `house hold`.")
    assert "grouped_forward_split" in list(report["value"])
    for tr, va in validation_folds(
        frame, "out come", problem="classification", group="house hold", time="day", seed=7
    ):
        a, b = frame.iloc[tr], frame.iloc[va]
        assert a["day"].max() < b["day"].min()
        assert set(a["house hold"]).isdisjoint(b["house hold"])


def test_impossible_grouped_forward_validation_is_rejected() -> None:
    frame = audit_dataset()[["day", "label", "signal"]].assign(group=1)
    with pytest.raises(ValueError, match="empty"):
        validation_folds(
            frame, "label", group="group", time="day", problem="classification", seed=7
        )


def test_audit_rejects_ambiguous_groups_and_bad_split_labels() -> None:
    frame = audit_dataset()[["label", "signal", "site_id", "household_id", "split"]]
    with pytest.raises(ValueError, match="Specify a group"):
        audit(frame, "label", problem="classification")
    with pytest.raises(ValueError, match="split"):
        audit(frame.assign(split="other"), "label", problem="classification")
    with pytest.raises(ValueError, match="target"):
        requested_audit(frame, "Audit classification.")


def test_calibration_curve_and_weighted_error() -> None:
    y = pd.Series(["n", "y", "y", "y"])
    p = pd.Series([0.0, 0.25, 0.25, 1.0])
    result = calibration(y, p, positive="y", bins=2)
    assert result.curve["count"].tolist() == [3, 1]
    assert result.curve["fraction_positive"].tolist() == pytest.approx([2 / 3, 1])
    assert result.brier == pytest.approx(2 * 0.75**2 / 4)
    assert result.ece == pytest.approx(0.375)
    # A small holdout can contain only negative observations.
    negatives = calibration(pd.Series(["n", "n"]), pd.Series([0.2, 0.4]), positive="y")
    assert negatives.brier == pytest.approx(0.1)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_probabilities_raise(bad: float) -> None:
    with pytest.raises(ValueError, match="Probabilities"):
        calibration(pd.Series([0, 1]), pd.Series([bad, 1.0]), positive=1)
