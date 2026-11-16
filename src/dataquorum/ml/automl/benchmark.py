"""AutoML benchmark (roadmap week 10): H2O vs FLAML vs AutoGluon on 8 OpenML tasks.

    uv sync --group evals                     # openml, for the first download
    uv run python -m dataquorum.ml.automl.benchmark --budget 60 --seeds 0 1 2
    uv run python -m dataquorum.ml.automl.benchmark --resume runs/automl_bench/<id>
    uv run python -m dataquorum.ml.automl.benchmark --report runs/automl_bench/<id>

Fairness rules: same split per seed for every backend (stratified 75/25 for classification),
same time budget, same container limits (2 CPUs, 3 GB), one container at a time, offline.
Scores are computed on the host from each backend's test predictions: ROC AUC (binary),
accuracy (multiclass), RMSE (regression). Backends are compared with the Friedman test on
per-task mean scores, average ranks and the Nemenyi critical difference (Demsar, 2006).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from dataquorum.evals.stats import friedman
from dataquorum.ml.automl import Backend, TaskType, fit_predict


@dataclass(frozen=True)
class BenchTask:
    name: str
    openml_id: int
    version: int
    task_type: TaskType


TASKS = [
    BenchTask("credit-g", 31, 1, "binary"),
    BenchTask("diabetes", 37, 1, "binary"),
    BenchTask("blood-transfusion", 1464, 1, "binary"),
    BenchTask("phoneme", 1489, 1, "binary"),
    BenchTask("vehicle", 54, 1, "multiclass"),
    BenchTask("segment", 36, 1, "multiclass"),
    BenchTask("kin8nm", 189, 1, "regression"),
    BenchTask("cpu_act", 197, 1, "regression"),
]
METRIC = {
    "binary": ("roc_auc", True),
    "multiclass": ("accuracy", True),
    "regression": ("rmse", False),
}


def load_task(task: BenchTask, cache: Path = Path("data/openml/bench")) -> tuple[pd.DataFrame, str]:
    path, meta = cache / f"{task.name}.parquet", cache / f"{task.name}.json"
    if not path.exists():
        import openml

        ds = openml.datasets.get_dataset(
            task.openml_id,
            version=task.version,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        frame = cast(pd.DataFrame, ds.get_data(dataset_format="dataframe")[0])
        for col in frame.columns:
            if isinstance(frame[col].dtype, pd.CategoricalDtype):
                frame[col] = [None if pd.isna(v) else str(v) for v in frame[col]]
        cache.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        meta.write_text(json.dumps({"target": ds.default_target_attribute}), encoding="utf-8")
    target = json.loads(meta.read_text(encoding="utf-8"))["target"]
    return pd.read_parquet(path), str(target)


def split(
    frame: pd.DataFrame, target: str, task_type: TaskType, seed: int, test_frac: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded 75/25 split, stratified by class for classification."""
    rng = np.random.default_rng(seed)
    groups = (
        [frame.index.to_numpy()]
        if task_type == "regression"
        else [g.index.to_numpy() for _, g in frame.groupby(target)]
    )
    test_idx: list[int] = []
    for idx in groups:
        shuffled = rng.permutation(idx)
        test_idx += shuffled[: max(1, round(len(idx) * test_frac))].tolist()
    test_mask = frame.index.isin(test_idx)
    return frame[~test_mask].reset_index(drop=True), frame[test_mask].reset_index(drop=True)


def roc_auc(y_true: pd.Series, proba: np.ndarray, positive: Any) -> float:
    """Mann-Whitney form of ROC AUC (ties count half)."""
    is_pos = (y_true.astype(str) == str(positive)).to_numpy()
    ranks = pd.Series(proba).rank().to_numpy()
    n_pos, n_neg = int(is_pos.sum()), int((~is_pos).sum())
    return float((ranks[is_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def score(
    task_type: TaskType, y_true: pd.Series, preds: pd.DataFrame, positive: Any = None
) -> float:
    if task_type == "binary":
        return roc_auc(y_true, preds["proba"].to_numpy(dtype=float), positive)
    if task_type == "multiclass":
        return float((preds["pred"].astype(str).to_numpy() == y_true.astype(str).to_numpy()).mean())
    errors = preds["pred"].to_numpy(dtype=float) - y_true.to_numpy(dtype=float)
    return float(math.sqrt(np.mean(errors**2)))


def run(
    out_dir: Path, backends: list[Backend], seeds: list[int], budget: int, tasks: list[BenchTask]
) -> None:
    results = out_dir / "results.jsonl"
    done = {(r["task"], r["seed"], r["backend"]) for r in _rows(results)}
    for task in tasks:
        frame, target = load_task(task)
        for seed in seeds:
            train, test = split(frame, target, task.task_type, seed)
            for backend in backends:
                if (task.name, seed, backend) in done:
                    continue
                started = time.perf_counter()
                res = fit_predict(
                    train,
                    test,
                    target=target,
                    task_type=task.task_type,
                    backend=backend,
                    budget_s=budget,
                    seed=seed,
                    workdir=out_dir / "work" / f"{task.name}-{seed}-{backend}",
                )
                metric, _ = METRIC[task.task_type]
                positive = (
                    sorted(train[target].dropna().unique(), key=str)[1]
                    if task.task_type == "binary"
                    else None
                )
                value = (
                    score(task.task_type, test[target], res.predictions, positive)
                    if res.ok and res.predictions is not None
                    else None
                )
                row = {
                    "task": task.name,
                    "seed": seed,
                    "backend": backend,
                    "task_type": task.task_type,
                    "metric": metric,
                    "score": value,
                    "ok": res.ok,
                    "error": res.error,
                    "wall_seconds": round(time.perf_counter() - started, 1),
                    **res.meta,
                }
                with results.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                shown = (
                    f"{value:.4f}"
                    if value is not None
                    else f"FAILED ({(res.error or '')[-120:]!r})"
                )
                print(
                    f"  {task.name:<18} seed {seed} {backend:<9} {metric}={shown}  ({row['wall_seconds']}s)",
                    flush=True,
                )


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def report(out_dir: Path) -> str:
    rows = pd.DataFrame(_rows(out_dir / "results.jsonl"))
    ok = rows[rows["ok"]]
    means = ok.pivot_table(index="task", columns="backend", values="score", aggfunc="mean")
    seeds_done = ok.groupby(["task", "backend"]).size().unstack()
    backends = list(means.columns)
    complete = means.dropna()
    direction = [
        METRIC[rows.loc[rows["task"] == t, "task_type"].iloc[0]][1] for t in complete.index
    ]
    lines = [
        "| Task | Metric | " + " | ".join(backends) + " |",
        "|---|---|" + "---|" * len(backends),
    ]
    for task in means.index:
        metric = rows.loc[rows["task"] == task, "metric"].iloc[0]
        cells = [
            f"{means.loc[task, b]:.4f} ({int(seeds_done.loc[task, b]) if pd.notna(seeds_done.loc[task, b]) else 0})"
            if pd.notna(means.loc[task, b])
            else "failed"
            for b in backends
        ]
        lines.append(f"| {task} | {metric} | " + " | ".join(cells) + " |")
    failures = rows[~rows["ok"]].groupby("backend").size().to_dict()
    lines += [
        "",
        "Cells: mean score over seeds (number of successful seeds).",
        f"Failed runs: {failures or 'none'}",
    ]
    if len(complete) >= 2 and len(backends) >= 2:
        test = friedman(backends, complete.to_numpy().tolist(), direction)
        ranks = ", ".join(
            f"{m} {r:.2f}"
            for m, r in sorted(
                zip(test.methods, test.average_ranks, strict=True), key=lambda p: p[1]
            )
        )
        lines += [
            "",
            f"Average rank over {len(complete)} tasks (1 = best): {ranks}",
            f"Friedman chi2 = {test.statistic:.3f}, p = {test.p_value:.4f}; Nemenyi critical difference (alpha 0.05) = {test.critical_difference:.2f}",
        ]
    secs = ok.groupby("backend")[["fit_seconds", "wall_seconds"]].median()
    lines += [
        "",
        "Median seconds (fit / wall incl. container start): "
        + ", ".join(
            f"{b} {secs.loc[b, 'fit_seconds']:.0f}/{secs.loc[b, 'wall_seconds']:.0f}"
            for b in secs.index
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--backends", nargs="+", default=["flaml", "h2o", "autogluon"])
    p.add_argument("--tasks", nargs="+", help="task names (default: all 8)")
    p.add_argument("--resume", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    if args.report:
        print(report(args.report))
        return 0
    out_dir = args.resume or Path("runs/automl_bench") / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "meta.json"
    if not meta.exists():
        meta.write_text(
            json.dumps(
                {
                    "budget_s": args.budget,
                    "seeds": args.seeds,
                    "backends": args.backends,
                    "started": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )
    tasks = [t for t in TASKS if not args.tasks or t.name in args.tasks]
    print(
        f"Benchmark {out_dir.name}: {len(tasks)} tasks x {len(args.seeds)} seeds x {args.backends}, budget {args.budget}s",
        flush=True,
    )
    run(out_dir, args.backends, args.seeds, args.budget, tasks)
    print("\n" + report(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
