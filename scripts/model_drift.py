# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["evidently==0.7.23", "pandas==2.3.3", "pyarrow==26.0.1"]
# ///
"""Compare two Parquet batches of model inputs using Evidently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> Any:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    columns = list(reference.columns)
    if (
        not columns
        or not all(isinstance(col, str) for col in columns)
        or not reference.columns.is_unique
        or not current.columns.is_unique
        or set(columns) != set(current.columns)
    ):
        raise ValueError("Reference and current data need the same unique named features")
    batches = []
    for label, frame in (("reference", reference), ("current", current)):
        frame = frame.loc[:, columns]
        if len(frame) < 2 or not all(
            pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype)
            for dtype in frame.dtypes
        ):
            raise ValueError(f"Provide at least two rows of real numeric {label} data")
        values = frame.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"The {label} batch contains missing or non-finite values")
        batches.append(pd.DataFrame(values, columns=columns))
    definition = DataDefinition(numerical_columns=columns)
    base, batch = (Dataset.from_pandas(data, data_definition=definition) for data in batches)
    return Report([DataDriftPreset()]).run(current_data=batch, reference_data=base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("report", type=Path, help="Output HTML file")
    args = parser.parse_args()
    snapshot = drift_report(pd.read_parquet(args.reference), pd.read_parquet(args.current))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.report))
    print(json.dumps(snapshot.dict(), allow_nan=False))


if __name__ == "__main__":
    main()
