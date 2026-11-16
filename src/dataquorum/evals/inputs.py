"""Build non-DataFrame task inputs: a folder of files (loader) or a SQLite database (SQL).

Files are written from the task's clean sample with the quirks the loader must handle:
separators, decimal commas, note lines above the header, several parts to stack, Excel
workbooks with extra sheets, JSON, and decoy files that must be ignored. The expected output
is the clean rows of the non-decoy files, stacked in file-name order.

Databases hold the clean sample as one table. `lookups` move a text column into its own
table (`purpose` -> `purpose_id` + table `purposes(id, name)`), so answers need a JOIN while
the expected values are still computed from the clean sample.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

FileFormat = Literal["csv", "tsv", "json", "excel", "parquet", "text"]


class FileSpec(BaseModel):
    path: str  # relative to the input folder
    format: FileFormat = "csv"
    sep: str | None = None  # csv default ",", tsv "\t"
    decimal: str = "."
    preamble: list[str] = []  # note lines written above the header
    rows: tuple[int, int] | None = None  # slice [start, stop) of the clean sample
    sheet: str = "data"  # excel: sheet holding the rows
    extra_sheets: list[str] = []  # excel: decoy sheets with a short note
    decoy: bool = False  # different data; must not be loaded
    text: str = ""  # format "text": file content


class LookupSpec(BaseModel):
    column: str
    table: str  # e.g. "purposes"; main table gets `<column>_id`


class DatabaseSpec(BaseModel):
    table: str = "data"
    lookups: list[LookupSpec] = []


def write_files(files: list[FileSpec], clean: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """Write the files; return the expected loaded frame (non-decoy parts, path order)."""
    folder.mkdir(parents=True, exist_ok=True)
    parts: list[tuple[str, pd.DataFrame]] = []
    for spec in files:
        target = folder / spec.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if spec.format == "text":
            target.write_text(spec.text, encoding="utf-8")
            continue
        frame = clean.iloc[slice(*spec.rows)] if spec.rows else clean
        if spec.decoy:
            frame = _decoy(frame)
        _write(spec, frame.reset_index(drop=True), target)
        if not spec.decoy:
            parts.append((spec.path, frame))
    ordered = [frame for _, frame in sorted(parts, key=lambda p: p[0])]
    return pd.concat(ordered, ignore_index=True) if ordered else clean.iloc[0:0]


def _decoy(frame: pd.DataFrame) -> pd.DataFrame:
    """Same columns, clearly different values (numbers shifted, order reversed)."""
    out = frame.iloc[::-1].head(25).copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(float) + 1000  # float: uint8 columns would overflow
    return out


def _write(spec: FileSpec, frame: pd.DataFrame, target: Path) -> None:
    if spec.format in ("csv", "tsv"):
        sep = spec.sep or ("\t" if spec.format == "tsv" else ",")
        body = frame.to_csv(index=False, sep=sep, decimal=spec.decimal)
        preamble = "".join(f"{line}\n" for line in spec.preamble)
        target.write_text(preamble + body, encoding="utf-8")
    elif spec.format == "json":
        frame.to_json(target, orient="records", indent=1)
    elif spec.format == "parquet":
        frame.to_parquet(target, index=False)
    elif spec.format == "excel":
        with pd.ExcelWriter(target) as writer:
            for name in spec.extra_sheets:
                pd.DataFrame({"note": [f"{name}: not data, see the {spec.sheet} sheet"]}).to_excel(
                    writer, sheet_name=name, index=False
                )
            frame.to_excel(writer, sheet_name=spec.sheet, index=False)


def write_database(spec: DatabaseSpec, clean: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    main = clean.copy()
    with sqlite3.connect(path) as con:
        for lookup in spec.lookups:
            names = sorted(str(v) for v in main[lookup.column].dropna().unique())
            ids = {name: i for i, name in enumerate(names, start=1)}
            pd.DataFrame({"id": list(ids.values()), "name": list(ids)}).to_sql(
                lookup.table, con, index=False
            )
            main[f"{lookup.column}_id"] = [
                ids.get(str(v)) if pd.notna(v) else np.nan for v in main[lookup.column]
            ]
            main = main.drop(columns=[lookup.column])
        main.to_sql(spec.table, con, index=False)
    con.close()
