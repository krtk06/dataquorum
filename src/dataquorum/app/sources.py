"""Load local tabular sources and create isolated snapshots of browser uploads."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path, PureWindowsPath

import pandas as pd

from dataquorum.app.pipeline_model import Source
from dataquorum.core.inputs import DATABASE_SUFFIXES

TABLE_SUFFIXES = (".csv", ".parquet")
FILE_SUFFIXES = (*TABLE_SUFFIXES, ".tsv", ".xlsx", ".json", ".txt")


def _read_table(name: str, data: Path | io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(data) if Path(name).suffix.lower() == ".csv" else pd.read_parquet(data)


def open_source(path: Path) -> Source:
    path = path.resolve()
    if path.is_dir():
        return Source(path.name, "files", path)
    if not path.is_file():
        raise ValueError("Choose a local file or directory.")
    suffix = path.suffix.lower()
    if suffix in TABLE_SUFFIXES:
        return Source(path.name, "dataframe", _read_table(path.name, path))
    if suffix in DATABASE_SUFFIXES:
        return Source(path.name, "database", path)
    raise ValueError("Choose CSV, Parquet, a database file or a folder.")


def uploaded_source(kind: str, uploads: list[tuple[str, bytes]], root: Path) -> Source:
    if not uploads:
        raise ValueError("Select at least one file.")
    if len(uploads) > 50 or sum(len(data) for _, data in uploads) > 25 * 1024**2:
        raise ValueError("Use at most 50 files and 25 MB per data source.")
    names: set[str] = set()
    for name, data in uploads:
        if (
            not data
            or PureWindowsPath(name).name != name
            or name in (".", "..")
            or name.casefold() in names
            or ":" in name
            or name.rstrip(" .") != name
            or PureWindowsPath(name).is_reserved()
        ):
            raise ValueError("Use nonempty files with unique, plain filenames.")
        names.add(name.casefold())
    name, data = uploads[0]
    if kind == "Table" and len(uploads) == 1 and Path(name).suffix.lower() in TABLE_SUFFIXES:
        return Source(name, "dataframe", _read_table(name, io.BytesIO(data)))
    database = (
        kind == "Database" and len(uploads) == 1 and Path(name).suffix.lower() in DATABASE_SUFFIXES
    )
    files = kind == "Folder" and all(Path(n).suffix.lower() in FILE_SUFFIXES for n, _ in uploads)
    if not (database or files):
        raise ValueError("Select files matching the source format.")
    root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix="source-", dir=root))
    for filename, content in uploads:
        (snapshot / filename).write_bytes(content)
    if database:
        return Source(name, "database", snapshot / name)
    return Source(f"{len(uploads)} source files", "files", snapshot)
