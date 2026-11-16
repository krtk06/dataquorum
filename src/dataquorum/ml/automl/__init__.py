"""AutoML behind one interface: H2O, FLAML and AutoGluon, each in its own Docker image.

    from dataquorum.ml.automl import fit_predict
    result = fit_predict(train, test, target="class", task_type="binary", backend="flaml",
                         budget_s=60, seed=0)
    result.predictions["proba"]   # host scores these; no backend grades itself

Each call runs `runner.py` in a container with no network, a read-only root, all
capabilities dropped and memory/CPU/PID limits. The runner is our own fixed code (not LLM
output); the container keeps heavy dependencies (JVM, torch) off the host and enforces equal
resources across backends. One container at a time keeps within the 4 GB WSL limit.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import docker
import pandas as pd
from docker.errors import DockerException
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout

Backend = Literal["flaml", "h2o", "autogluon"]
TaskType = Literal["binary", "multiclass", "regression"]

IMAGES: dict[str, str] = {
    "flaml": "dataquorum/automl:0.1",
    "h2o": "dataquorum/automl:0.1",
    "autogluon": "dataquorum/autogluon:1.6.3",
}
RUNNER = Path(__file__).with_name("runner.py")
LABEL = "dataquorum.automl"


@dataclass
class AutoMLResult:
    ok: bool
    predictions: pd.DataFrame | None
    meta: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    error: str | None = None


def positive_label(y: pd.Series) -> Any:
    """Binary tasks: the class whose probability is reported (second in sorted order)."""
    labels = sorted(y.dropna().unique(), key=str)
    if len(labels) != 2:
        raise ValueError(f"binary task needs 2 classes, got {labels}")
    return labels[1]


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    task_type: TaskType,
    backend: Backend,
    budget_s: int = 60,
    seed: int = 0,
    workdir: Path | None = None,
    cpus: int = 2,
    memory_gb: int = 3,
    client: docker.DockerClient | None = None,
) -> AutoMLResult:
    work = workdir or Path("runs/automl") / uuid.uuid4().hex[:12]
    in_dir, out_dir = work / "in", work / "out"
    shutil.rmtree(work, ignore_errors=True)
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    train.to_parquet(in_dir / "train.parquet", index=False)
    test.drop(columns=[target], errors="ignore").to_parquet(in_dir / "test.parquet", index=False)
    shutil.copy(RUNNER, in_dir / "runner.py")
    job: dict[str, Any] = {
        "backend": backend,
        "target": target,
        "task_type": task_type,
        "budget_s": budget_s,
        "seed": seed,
        "cpus": cpus,
        "mem_gb": max(1, memory_gb - 1),
    }
    if task_type == "binary":
        job["positive"] = positive_label(train[target])
    (in_dir / "job.json").write_text(json.dumps(job, default=str), encoding="utf-8")

    client = client or docker.from_env()
    start = time.perf_counter()
    container = client.containers.run(
        IMAGES[backend],
        ["python", "/io/in/runner.py"],
        detach=True,
        network_mode="none",
        read_only=True,
        tmpfs={"/tmp": "rw,nosuid,nodev,size=1536m,mode=1777"},
        volumes={
            str(in_dir.resolve()): {"bind": "/io/in", "mode": "ro"},
            str(out_dir.resolve()): {"bind": "/io/out", "mode": "rw"},
        },
        user="10001:10001",
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
        mem_limit=f"{memory_gb}g",
        memswap_limit=f"{memory_gb}g",
        nano_cpus=cpus * 10**9,
        pids_limit=2048,
        labels={LABEL: backend},
    )
    try:
        status = container.wait(timeout=budget_s + 300)
        logs = container.logs(tail=40).decode("utf-8", "replace")
    except (ReadTimeout, RequestsConnectionError):
        status, logs = {"StatusCode": -1}, f"timed out after {budget_s + 300} s"
    except DockerException as exc:
        status, logs = {"StatusCode": -1}, str(exc)
    finally:
        with contextlib.suppress(DockerException):
            container.remove(force=True)
    seconds = time.perf_counter() - start
    preds_file = out_dir / "predictions.parquet"
    if status.get("StatusCode") != 0 or not preds_file.exists():
        return AutoMLResult(False, None, seconds=seconds, error=logs[-1500:])
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return AutoMLResult(True, pd.read_parquet(preds_file), meta, seconds)


__all__ = ["IMAGES", "AutoMLResult", "Backend", "TaskType", "fit_predict", "positive_label"]
