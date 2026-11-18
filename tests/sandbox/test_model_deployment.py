"""Exercise the sandbox, local registry and HTTP contract together."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from dataquorum.ml.artifact import load_forest, pack_forest
from dataquorum.ml.export import export_forest
from dataquorum.ml.registry import ForestRegistry
from dataquorum.ml.serving import create_app

pytestmark = pytest.mark.docker


def test_export_register_and_serve(tmp_path: Path) -> None:
    pytest.importorskip("skops")
    pytest.importorskip("mlflow")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    rng = np.random.default_rng(13)
    signal, noise = rng.normal(size=(2, 160))
    label = np.where(signal + rng.normal(scale=0.6, size=160) > 0, "yes", "no")
    frame = pd.DataFrame(
        {"signal": signal, "noise": noise, "label": label, "split": ["train"] * 120 + ["test"] * 40}
    )
    artifact = export_forest(
        frame, "label", tmp_path / "candidate.zip", problem="classification", trials=1
    )
    loaded = load_forest(artifact)
    assert loaded.metadata.features == ("signal", "noise")
    assert loaded.metadata.training_dtypes == ("float64", "float64")
    assert loaded.metadata.metric == "accuracy"
    with ZipFile(artifact) as bundle:
        assert set(bundle.namelist()) == {"manifest.json", "forest.skops"}
        manifest = bundle.read("manifest.json").decode()
        corrupt = tmp_path / "corrupt.zip"
        corrupt.write_bytes(pack_forest(manifest, b"corrupt payload"))
    with pytest.raises(ValueError, match="checksum"):
        load_forest(corrupt)

    registry = ForestRegistry(tmp_path / "registry")
    first = registry.register("demo", artifact)
    second = registry.register("demo", artifact)
    assert (first.version, second.version) == ("1", "2")
    assert (
        registry.client.get_run(first.run_id).data.metrics["cv_accuracy"]
        == loaded.metadata.cv_score
    )
    records = frame.loc[120:124, ["noise", "signal"]]
    expected = loaded.predict(records).tolist()

    with TestClient(create_app(registry, first.name, first.version)) as client:
        assert client.get("/health").json() == {"status": "ready", "model": "demo", "version": "1"}
        assert client.get("/schema").json()["features"] == {"signal": "float64", "noise": "float64"}
        response = client.post("/predict", json={"rows": records.to_dict("records")})
        assert response.status_code == 200
        assert response.json() == {"model": "demo", "version": "1", "predictions": expected}
        for invalid in (
            [{"signal": 1}],
            [{"signal": 1, "noise": "2"}],
            [{"signal": 1, "noise": 1e39}],
            [{"signal": 1, "noise": 1, "extra": 0}],
        ):
            assert client.post("/predict", json={"rows": invalid}).status_code == 422


def test_export_regression_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("skops")
    rng = np.random.default_rng(17)
    x = rng.normal(size=80)
    frame = pd.DataFrame(
        {"x": x, "value": 2 * x + rng.normal(size=80), "split": ["train"] * 60 + ["test"] * 20}
    )
    path = export_forest(
        frame, "value", tmp_path / "regression.zip", problem="regression", trials=1
    )
    model = load_forest(path)
    assert model.metadata.metric == "rmse"
    assert len(model.predict(frame.loc[60:, ["x"]])) == 20
