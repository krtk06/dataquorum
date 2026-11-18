"""Read and validate a versioned DataQuorum forest artifact."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from dataquorum.ml.correctness import Problem

MAX_ARTIFACT_BYTES = 100 * 1024 * 1024


class ForestMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1]
    problem: Problem
    target: str
    features: tuple[str, ...] = Field(min_length=1)
    training_dtypes: tuple[str, ...]
    sklearn_version: str
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric: Literal["accuracy", "balanced_accuracy", "rmse"]
    cv_score: FiniteFloat
    trials: int = Field(ge=1)
    params: dict[str, int]
    seed: int

    @model_validator(mode="after")
    def check_schema(self) -> ForestMetadata:
        if len(set(self.features)) != len(self.features) or self.target in self.features:
            raise ValueError("Expected unique predictors excluding the target")
        if len(self.features) != len(self.training_dtypes):
            raise ValueError("Every feature needs a training dtype")
        return self


@dataclass(frozen=True)
class ForestModel:
    metadata: ForestMetadata
    _estimator: Any

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Require every named feature once, then align columns to training order."""
        columns = self.metadata.features
        if frame.empty or not frame.columns.is_unique or set(frame.columns) != set(columns):
            raise ValueError(f"Expected a nonempty batch with exactly these features: {columns}")
        ordered = frame.loc[:, list(columns)]
        if not all(
            pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype)
            for dtype in ordered.dtypes
        ):
            raise ValueError("Predictors must be real numbers")
        values = ordered.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError("Predictors must be finite and within the float32 range")
        return pd.DataFrame(values, columns=list(columns), index=frame.index)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        x = self.prepare(frame)
        return pd.Series(self._estimator.predict(x), index=x.index, name="prediction")


def load_forest(path: Path) -> ForestModel:
    """Reload a forest from a trusted DataQuorum export, using the recorded sklearn version."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from skops.io import loads

    try:
        with ZipFile(path) as bundle:
            members = bundle.infolist()
            if (
                sorted(m.filename for m in members) != ["forest.skops", "manifest.json"]
                or sum(m.file_size for m in members) > MAX_ARTIFACT_BYTES
            ):
                raise ValueError("Invalid forest artifact members or size")
            metadata = ForestMetadata.model_validate_json(bundle.read("manifest.json"))
            payload = bundle.read("forest.skops")
    except BadZipFile as exc:
        raise ValueError("Invalid forest artifact archive") from exc
    if sha256(payload).hexdigest() != metadata.model_sha256:
        raise ValueError("Model checksum does not match the manifest")
    installed = version("scikit-learn")
    if metadata.sklearn_version != installed:
        raise ValueError(f"Model needs scikit-learn {metadata.sklearn_version}, found {installed}")
    # Local forests contain sklearn's compiled Tree state. Permit only that extra type.
    estimator = loads(payload, trusted=["sklearn.tree._tree.Tree"])
    expected = RandomForestRegressor if metadata.problem == "regression" else RandomForestClassifier
    if type(estimator) is not expected or tuple(estimator.feature_names_in_) != metadata.features:
        raise ValueError("Fitted model does not match the manifest")
    return ForestModel(metadata, estimator)


def pack_forest(manifest: str, payload: bytes) -> bytes:
    """Create a single archive so the manifest and fitted model travel together."""
    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as bundle:
        bundle.writestr("manifest.json", manifest)
        bundle.writestr("forest.skops", payload)
    return buffer.getvalue()
