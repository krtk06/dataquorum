"""Exercise the public API with the production image and no LLM."""

import numpy as np
import pandas as pd
import pytest

from dataquorum.ml.report import forest_report

pytestmark = pytest.mark.docker


def sample() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    x = rng.normal(size=160)
    return pd.DataFrame(
        {
            "signal": x,
            "noise": rng.normal(size=len(x)),
            "label": (x + rng.normal(scale=0.6, size=len(x)) > 0).astype(int),
            "split": ["train"] * 120 + ["test"] * 40,
        }
    )


def test_tuning_uses_training_labels_only() -> None:
    frame = sample()
    first = forest_report(frame, "label", problem="classification", report="tuning", trials=2)
    frame.loc[frame["split"].eq("test"), "label"] = -1
    second = forest_report(frame, "label", problem="classification", report="tuning", trials=2)
    pd.testing.assert_frame_equal(first, second)
    assert first["trials"].iloc[0] == 2 and 0 <= first["cv_score"].iloc[0] <= 1


def test_shap_ranks_planted_signal_first() -> None:
    result = forest_report(sample(), "label", problem="classification", report="shap", trials=1)
    assert list(result["feature"]) == ["signal", "noise"]
    assert result["mean_abs_shap"].min() >= 0 and result["rows"].tolist() == [40, 40]


def test_held_out_calibration() -> None:
    result = forest_report(
        sample(), "label", problem="classification", report="calibration", trials=1
    )
    assert result["count"].sum() == 40
    assert 0 <= result["brier"].iloc[0] <= 1 and 0 <= result["ece"].iloc[0] <= 1
    assert result["positive_label"].unique().tolist() == [1]


def test_regression_with_time_and_groups() -> None:
    rng = np.random.default_rng(7)
    frame = sample().assign(
        household_id=np.arange(160) // 4, timestamp=pd.date_range("2022-01-01", periods=160)
    )
    frame["label"] = frame["signal"] + rng.normal(size=len(frame))
    result = forest_report(frame, "label", problem="regression", report="tuning", trials=1)
    assert result["metric"].iloc[0] == "rmse" and result["cv_score"].iloc[0] > 0


def test_calibration_rejects_cross_partition_overlap() -> None:
    frame = sample()
    frame.iloc[120, :2] = frame.iloc[0, :2].to_numpy()
    with pytest.raises(RuntimeError, match="row_overlap"):
        forest_report(frame, "label", problem="classification", report="calibration", trials=1)
