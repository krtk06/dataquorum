"""Run or report the Week 14 paired evaluation with a saved plan and a spend stop."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

from docker.errors import DockerException

from dataquorum.core.budget import BudgetExceededError, BudgetGuard, JsonlLedger, SpendEntry
from dataquorum.core.cost import Usage
from dataquorum.core.llm import (
    DEFAULT_BASE_URL,
    LLMClient,
    LLMConfig,
    MissingReportedCostError,
    ReasoningEffort,
)
from dataquorum.evals import datasets
from dataquorum.evals.oracle import check_task
from dataquorum.evals.runner import RunOptions, run_task
from dataquorum.evals.study_report import record_trial, summarize_study
from dataquorum.evals.tasks import Task, build, load_tasks
from dataquorum.sandbox.docker_executor import DockerExecutor, SandboxUnavailableError
from dataquorum.tracing.recorder import DuckDBLedger, RunRecorder, git_commit

TASK_IDS = (
    "hard-clean-ames-four-faults",
    "hard-clean-telco-three-faults",
    "hard-wr-credit-top2-per-housing",
    "hard-wr-impossible-income-by-occupation",
    "hard-viz-telco-churn-grouped",
    "hard-viz-impossible-income-vs-age",
    "load-adult-semicolon",
    "load-telco-monthly-parts",
    "sql-credit-renters-long-loans-join",
    "sql-impossible-missing-table",
    "eda-ames-area-price-correlation",
    "eda-telco-churn-rate-by-contract",
    "fe-credit-leakage-trap",
    "fe-impossible-missing-target",
    "audit-clean",
    "audit-categorical-proxy",
)
BASELINE = "deepseek-v4.1-flash"
MAX_RETRIES = 1
SEED = 14


@dataclass(frozen=True)
class Arm:
    name: str
    model: str
    variant: str = "full"
    effort: ReasoningEffort = "default"

    def options(self) -> RunOptions:
        return RunOptions(
            max_retries=MAX_RETRIES,
            no_checks=self.variant == "no-checks",
            no_recommend=self.variant == "no-recommend",
        )


ARMS = (
    Arm(BASELINE, BASELINE, effort="low"),
    Arm("deepseek-no-checks", BASELINE, "no-checks", "low"),
    Arm("deepseek-no-recommend", BASELINE, "no-recommend", "low"),
    Arm("glm-5.3-flash", "glm-5.3-flash"),
    Arm("qwen3.8-flash", "qwen3.8-flash"),
    Arm("minimax-m3.1-flash", "minimax-m3.1-flash"),
    Arm("mimo-v2.6-flash", "mimo-v2.6-flash"),
)


def ordered_trials(tasks: list[Task]) -> list[tuple[Task, Arm]]:
    """Interleave the arms for each task to reduce time-of-day bias."""
    rng = random.Random(SEED)
    order = list(tasks)
    rng.shuffle(order)
    return [(task, arm) for task in order for arm in rng.sample(ARMS, k=len(ARMS))]


def study_tasks() -> list[Task]:
    registry = {t.id: t for t in load_tasks()}
    return [registry[key] for key in TASK_IDS]


def check_reference_answers(tasks: list[Task]) -> None:
    missing = [
        name
        for name in sorted({t.dataset for t in tasks} & datasets.DATASETS.keys())
        if not datasets.cache_path(name).is_file()
    ]
    if missing:
        raise ValueError(f"Cache the selected OpenML datasets before the study: {missing}")
    with tempfile.TemporaryDirectory(prefix="dq-study-oracle-") as temp:
        for task in tasks:
            errors = check_task(build(task, workdir=Path(temp) / task.id))
            if errors:
                raise ValueError(f"{task.id} failed its reference answer: {errors}")


def make_plan(tasks: list[Task]) -> dict[str, Any]:
    return {
        "version": 1,
        "seed": SEED,
        "max_retries": MAX_RETRIES,
        "arms": [asdict(arm) for arm in ARMS],
        "tasks": [t.model_dump(mode="json") for t in tasks],
        "order": [[arm.name, task.id] for task, arm in ordered_trials(tasks)],
    }


class StudyBudget(BudgetGuard):
    def __init__(self, recorder: RunRecorder, out: Path, max_allowance: float) -> None:
        super().__init__(DuckDBLedger(recorder), mode="eval")
        self.study_ledger = JsonlLedger(out / "spend.jsonl")
        self.max_allowance = max_allowance

    def total(self) -> float:
        return self.study_ledger.total_since(datetime.min.replace(tzinfo=UTC))

    def check(self) -> None:
        if self.total() >= self.max_allowance:
            raise BudgetExceededError(
                f"Study allowance reached the ${self.max_allowance:.2f} stop."
            )
        super().check()

    def record(self, usage: Usage) -> None:
        self.study_ledger.append(
            SpendEntry(self.clock(), usage.allowance_usd, ",".join(usage.models))
        )
        super().record(usage)


def _read_trials(out: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    result_path = out / "trials.jsonl"
    if not result_path.exists():
        return []
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    keys = {(r["arm"], r["task_id"]) for r in rows}
    if len(keys) != len(rows) or not keys <= set(map(tuple, plan["order"])):
        raise ValueError("Saved trials contain duplicate or unexpected task/arm keys")
    return rows


def _write_json(path: Path, data: dict[str, Any]) -> None:
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    staged.replace(path)


def report(out: Path) -> dict[str, Any]:
    meta = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    plan = meta["plan"]
    summary = summarize_study(_read_trials(out, plan), plan)
    summary["metered_allowance_usd"] = JsonlLedger(out / "spend.jsonl").total_since(
        datetime.min.replace(tzinfo=UTC)
    )
    summary["allowance_complete"] = not (out / "unmetered.json").exists()
    saved = out / "summary.json"
    if saved.is_file():
        summary["stopped"] = json.loads(saved.read_text(encoding="utf-8")).get("stopped")
    return summary


@contextmanager
def _study_lock(out: Path) -> Iterator[None]:
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "running.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"Another study run owns {lock}. Check the recorded process.") from exc
    try:
        with stream:
            stream.write(f"{os.getpid()}\n")
        yield
    finally:
        lock.unlink()


def _environment() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    files = sorted(
        p
        for dirname in ("agents", "core", "evals", "ml", "sandbox")
        for p in (package / dirname).glob("*.py")
    )
    digest = sha256()
    for source in files:
        digest.update(source.relative_to(package).as_posix().encode() + b"\0" + source.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        **{dep: version(dep) for dep in ("pandas", "numpy", "pyarrow", "duckdb", "openai")},
    }


def _run_live(out: Path, tasks: list[Task], max_allowance: float) -> int:
    if max_allowance <= 0 or not math.isfinite(max_allowance):
        raise ValueError("Use a positive, finite --max-allowance for live runs")
    check_reference_answers(tasks)
    config = LLMConfig.from_env()
    if config.base_url.rstrip("/") != DEFAULT_BASE_URL:
        raise ValueError("The live study uses the OpenCode Go OpenAI-compatible endpoint")
    executor = DockerExecutor()
    executor.client.ping()
    image_id = executor.client.images.get(executor.image).id

    plan = make_plan(tasks)
    environment = _environment()
    meta_path = out / "plan.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta["plan"], meta["sandbox_image_id"], meta["environment"]) != (
            plan,
            image_id,
            environment,
        ):
            raise ValueError(
                "The saved plan, Docker image or runtime changed. Use a new directory."
            )
    else:
        if (out / "trials.jsonl").exists() or (out / "spend.jsonl").exists():
            raise ValueError("Existing results or spend ledger lack a saved plan")
        meta = {
            "study_id": f"week14-{uuid.uuid4().hex[:10]}",
            "git_commit": git_commit(),
            "sandbox_image": executor.image,
            "sandbox_image_id": image_id,
            "environment": environment,
            "plan": plan,
        }
        _write_json(meta_path, meta)
    if (out / "unmetered.json").exists():
        raise ValueError(
            "A response had unknown allowance. Reconcile the provider usage before resuming."
        )

    rows = _read_trials(out, plan)
    done = {(r["arm"], r["task_id"]) for r in rows}
    recorder = RunRecorder()
    budget = StudyBudget(recorder, out, max_allowance)
    clients = {
        arm.name: LLMClient(
            config.model_copy(
                update={
                    "model": arm.model,
                    "reasoning_effort": arm.effort,
                    "fallback_model": None,
                    "require_reported_cost": True,
                    "max_retries": 0,
                }
            ),
            budget=budget,
            session_id=f"{meta['study_id']}-{arm.name}",
        )
        for arm in ARMS
    }
    stopped: str | None = None
    try:
        with (out / "trials.jsonl").open("a", encoding="utf-8") as log:
            for task, arm in ordered_trials(tasks):
                if (arm.name, task.id) in done:
                    continue
                budget.check()
                started = time.perf_counter()
                # A fresh attempt directory keeps partial loader/SQL fixtures out of retries.
                workdir = out / "work" / f"{arm.name}-{uuid.uuid4().hex[:8]}"
                scored, result = run_task(
                    task,
                    clients[arm.name],
                    executor,
                    workdir,
                    f"{meta['study_id']}-{arm.name}",
                    arm.options(),
                )
                row = record_trial(arm.name, scored, result, time.perf_counter() - started)
                row["workdir"] = str(workdir.resolve())
                recorder.record(result, eval_task_id=task.id)
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    f"{len(rows)}/{len(plan['order'])} {arm.name} {task.id} {scored.outcome} "
                    f"(study allowance ${budget.total():.5f})",
                    flush=True,
                )
    except (BudgetExceededError, SandboxUnavailableError, KeyboardInterrupt) as exc:
        stopped = str(exc) or "Interrupted"
    except MissingReportedCostError as exc:
        stopped = str(exc)
        _write_json(out / "unmetered.json", {"reason": stopped})
    finally:
        summary = summarize_study(rows, plan)
        summary.update(stopped=stopped, metered_allowance_usd=budget.total())
        _write_json(out / "summary.json", summary)
        recorder.close()
    if stopped:
        print(f"Stopped: {stopped}", file=sys.stderr)
    print(f"Completed {len(rows)}/{len(plan['order'])} trials. Summary: {out / 'summary.json'}")
    return 2 if stopped else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "report"), nargs="?", default="plan")
    parser.add_argument("--out", type=Path, default=Path("runs/week14-study"))
    parser.add_argument("--max-allowance", type=float)
    args = parser.parse_args(argv)
    if args.mode == "report":
        print(json.dumps(report(args.out), indent=2))
        return 0
    tasks = study_tasks()
    if args.mode == "plan":
        check_reference_answers(tasks)
        print(f"{len(tasks)} reference answers passed. {len(tasks) * len(ARMS)} planned trials.")
        print(json.dumps(make_plan(tasks)["arms"], indent=2))
        return 0
    if args.max_allowance is None:
        parser.error("run requires --max-allowance")
    try:
        with _study_lock(args.out):
            return _run_live(args.out, tasks, args.max_allowance)
    except (ValueError, DockerException, SandboxUnavailableError) as exc:
        print(f"Study could not start: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
