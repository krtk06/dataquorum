"""OpenML datasets used by the eval suite (no Titanic or Iris).

Each dataset is pinned by OpenML id and version, cached once as Parquet under
`data/openml/` (git-ignored), and checked against a pinned hash of its *contents* (not the
file bytes, which change with pyarrow versions), so a changed upstream file cannot silently
change results.

    uv sync --group evals
    uv run python -m dataquorum.evals.datasets      # download and verify all
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd


@dataclass(frozen=True)
class DatasetRef:
    name: str
    openml_id: int
    version: int
    content_hash: str | None  # see content_hash(); None until pinned


DATASETS: dict[str, DatasetRef] = {
    d.name: d
    for d in (
        # Hashes pinned 2026-11-16; identical under pandas 2.3.3 and 3.1.1.
        DatasetRef(
            "adult", 1590, 2, "eee07dd8795e3ba6041019fdfa6913d5fa59c04310cfde9b6e0b184425abb493"
        ),
        DatasetRef(
            "credit-g", 31, 1, "c333214e7d8084b3c5bebd79917d6a15f34c245453c3b1af05d75d5585e0194b"
        ),
        DatasetRef(
            "telco-customer-churn",
            42178,
            1,
            "254ebf7c696910c56bb87943b20fd43060072f3c24d25fb904957f5e49ec8497",
        ),
        DatasetRef(
            "ames-housing",
            41211,
            1,
            "c2bc16c44280f2cfa625d099c197cf8c39c1f4d4346c3ef1035db25d1d5ccfc2",
        ),
        DatasetRef(
            "diabetes", 37, 1, "2e22f330ff35ccdb7565fc61d8b41f83f0e1ff33e45e9960c45c5a4665438eae"
        ),
    )
}


def data_dir() -> Path:
    return Path(os.getenv("DQ_DATA_DIR", "data/openml"))


def cache_path(name: str) -> Path:
    return data_dir() / f"{name}.parquet"


def _canonical(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Text columns as plain Python objects, so pandas 2 (object) and pandas 3 (str dtype)
    hash the same data identically."""
    out = frame.copy()
    kinds = []
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            out[col] = pd.Series([None if pd.isna(v) else str(v) for v in series], dtype=object)
            kinds.append((str(col), "text"))
        else:
            kinds.append((str(col), str(series.dtype)))
    return out, kinds


def content_hash(frame: pd.DataFrame) -> str:
    """SHA-256 over column names, normalised dtypes and pandas' per-row hashes."""
    canonical, kinds = _canonical(frame)
    digest = hashlib.sha256()
    digest.update(repr(kinds).encode())
    digest.update(pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def download(ref: DatasetRef) -> Path:
    """Fetch from OpenML (needs the `evals` dependency group) and cache as Parquet."""
    import openml

    dataset = openml.datasets.get_dataset(
        ref.openml_id,
        version=ref.version,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    data, *_ = dataset.get_data(dataset_format="dataframe")
    frame = cast(pd.DataFrame, data)
    # Categories become plain strings so faults and agents see ordinary text columns.
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            frame[col] = [None if pd.isna(v) else v for v in frame[col].astype(object)]
    path = cache_path(ref.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index(drop=True).to_parquet(path, index=False)
    return path


def load(name: str) -> pd.DataFrame:
    ref = DATASETS[name]
    path = cache_path(name)
    if not path.exists():
        download(ref)
    frame = pd.read_parquet(path)
    if ref.content_hash is not None and content_hash(frame) != ref.content_hash:
        raise ValueError(
            f"{path} does not match the pinned content hash for {name}. Delete it and "
            "re-download, or update the pin if the change is intended."
        )
    return frame


if __name__ == "__main__":
    for ref in DATASETS.values():
        frame = load(ref.name)
        print(f"{ref.name:<22} {frame.shape} content_hash={content_hash(frame)}")
