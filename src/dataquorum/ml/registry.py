"""Local MLflow tracking and registry for immutable forest versions."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataquorum.ml.artifact import ForestModel, load_forest

_ARTIFACT = "forest.zip"


@dataclass(frozen=True)
class RegisteredForest:
    name: str
    version: str
    run_id: str


class ForestRegistry:
    def __init__(self, root: Path) -> None:
        from mlflow import MlflowClient

        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{(self.root / 'registry.db').as_posix()}"
        self.client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    def _experiment_id(self) -> str:
        from mlflow.exceptions import MlflowException

        name = "DataQuorum forests"
        experiment = self.client.get_experiment_by_name(name)
        if experiment is None:
            try:
                return str(
                    self.client.create_experiment(
                        name, artifact_location=(self.root / "artifacts").as_uri()
                    )
                )
            except MlflowException as exc:
                if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
                experiment = self.client.get_experiment_by_name(name)
        if experiment is None:
            raise RuntimeError("Could not locate the DataQuorum tracking experiment")
        return str(experiment.experiment_id)

    def register(self, name: str, artifact: Path) -> RegisteredForest:
        from mlflow.exceptions import MlflowException

        # Stage a snapshot so validation and upload use the same bytes.
        with tempfile.TemporaryDirectory(prefix="dq-register-") as temp:
            staged = Path(temp) / _ARTIFACT
            staged.write_bytes(artifact.read_bytes())
            meta = load_forest(staged).metadata
            run = self.client.create_run(self._experiment_id(), run_name=name)
            run_id = str(run.info.run_id)
            try:
                self.client.log_artifact(run_id, str(staged))
                self.client.log_metric(run_id, f"cv_{meta.metric}", meta.cv_score)
                params: dict[str, Any] = {
                    **meta.params,
                    "trials": meta.trials,
                    "seed": meta.seed,
                    "target": meta.target,
                    "problem": meta.problem,
                    "sklearn_version": meta.sklearn_version,
                }
                for key, value in params.items():
                    self.client.log_param(run_id, key, value)
                try:
                    self.client.create_registered_model(name)
                except MlflowException as exc:
                    if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                        raise
                source = run.info.artifact_uri.rstrip("/") + "/" + _ARTIFACT
                model = self.client.create_model_version(name, source, run_id=run_id)
            except Exception:
                self.client.set_terminated(run_id, status="FAILED")
                raise
            self.client.set_terminated(run_id, status="FINISHED")
        return RegisteredForest(name, str(model.version), run_id)

    def load(self, name: str, version: str) -> ForestModel:
        model = self.client.get_model_version(name, version)
        if model.status != "READY" or not model.run_id:
            raise ValueError("Model version must be ready and associated with a tracking run")
        with tempfile.TemporaryDirectory(prefix="dq-registry-") as temp:
            path = self.client.download_artifacts(model.run_id, _ARTIFACT, dst_path=temp)
            return load_forest(Path(path))
