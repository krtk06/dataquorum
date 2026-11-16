"""Run the original ai-data-science-team on DataQuorum's eval tasks (spec S10, parity).

    docker build -t dataquorum/original:4ffeb7f -f images/original/Dockerfile .original/ai-data-science-team
    uv run python -m dataquorum.evals.parity.run --label original-baseline
    uv run python -m dataquorum.evals.compare runs/parity/<id> runs/evals/<id>

Each task runs `images/original/worker.py` in its own `dataquorum/original` container: same
built input, same request text, same model and reasoning effort, same retry budget. The
container needs network access (to call Go) but gets no host mount except the task's job
folder; root filesystem read-only, all capabilities dropped, non-root, memory/CPU/PID limits.

The original has no status field. It is mapped the way a user would read it: an error or no
output -> `failed`; an output without an error -> `success` (it reports the work as done).
Cost is estimated from token counts with our price table (the original's LangChain client
does not expose Go's allowance figure).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import docker
import pandas as pd
from docker.errors import DockerException
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout

from dataquorum.core.cost import PRICES, estimate_list_cost
from dataquorum.core.llm import LLMConfig
from dataquorum.evals.checkers import Output, run_checker
from dataquorum.evals.scoring import TaskResult, classify, format_summary, summarise
from dataquorum.evals.tasks import Task, build, load_tasks

IMAGE = "dataquorum/original:4ffeb7f"
WORKER = Path(__file__).resolve().parents[4] / "images" / "original" / "worker.py"
LABEL = "dataquorum.parity"


def _read_output(out: Path, kind: str | None) -> Output:
    if kind == "dataframe" and (out / "output.parquet").exists():
        return pd.read_parquet(out / "output.parquet")
    if kind == "figure" and (out / "figure.json").exists():
        figure: dict[str, Any] = json.loads((out / "figure.json").read_text(encoding="utf-8"))
        return figure
    return None


def run_one(
    task: Task, config: LLMConfig, workdir: Path, client: docker.DockerClient,
    max_retries: int, timeout: int,
) -> TaskResult:  # fmt: skip
    built = build(task)
    assert isinstance(built.messy, pd.DataFrame)
    job = workdir / task.id
    shutil.rmtree(job, ignore_errors=True)
    (job / "out").mkdir(parents=True)
    built.messy.to_parquet(job / "input.parquet", index=False)
    (job / "job.json").write_text(json.dumps({
        "agent": task.agent, "request": built.request, "model": config.model,
        "base_url": config.base_url, "reasoning_effort": config.reasoning_effort,
        "max_retries": max_retries, "session": f"dq-parity-{uuid.uuid4().hex[:12]}",
    }), encoding="utf-8")  # fmt: skip
    (job / "secret").write_text(config.api_key.get_secret_value(), encoding="utf-8")

    container = client.containers.run(
        IMAGE, ["python", "/worker/worker.py"], detach=True,
        volumes={str(job.resolve()): {"bind": "/job", "mode": "rw"},
                 str(WORKER.parent.resolve()): {"bind": "/worker", "mode": "ro"}},
        read_only=True, tmpfs={"/tmp": "rw,nosuid,nodev,noexec,size=512m,mode=1777"},
        user="10001:10001", cap_drop=["ALL"], security_opt=["no-new-privileges"],
        mem_limit="2g", memswap_limit="2g", nano_cpus=2_000_000_000, pids_limit=256,
        labels={LABEL: "1"},
    )  # fmt: skip
    start = time.perf_counter()
    try:
        container.wait(timeout=timeout)
        logs = ""
    except (ReadTimeout, RequestsConnectionError):
        logs = f"timed out after {timeout} s"
    finally:
        tail = container.logs(tail=30).decode("utf-8", "replace")
        container.remove(force=True)
        (job / "secret").unlink(missing_ok=True)
    seconds = time.perf_counter() - start

    result_file = job / "out" / "result.json"
    meta: dict[str, Any] = (
        json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
    )
    error = meta.get("error") or (None if meta else f"worker produced no result: {logs or tail}")
    output = _read_output(job / "out", meta.get("output_kind"))
    status = "failed" if error or output is None else "success"
    checks = [run_checker(c.kind, built, output, c.params) for c in task.checks]
    prompt, completion = int(meta.get("prompt_tokens", 0)), int(meta.get("completion_tokens", 0))
    list_cost = estimate_list_cost(config.model, prompt, completion, 0, datetime.now(UTC))
    return TaskResult(
        task_id=task.id, agent=task.agent, tags=task.tags, expect_status=task.expect_status,
        status=status, outcome=classify(status, task.expect_status, checks), checks=checks,
        error=str(error)[:500] if error else None, run_id=job.name, calls=int(meta.get("calls", 0)),
        completion_tokens=completion, reasoning_tokens=int(meta.get("reasoning_tokens", 0)),
        allowance_usd=list_cost * PRICES[config.model].allowance_ratio,
        seconds=round(float(meta.get("seconds", seconds)), 2),
        extra={"final_message": meta.get("final_message"), "cost_estimated": True},
    )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", action="append", default=[])
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--label", default="original")
    args = p.parse_args(argv)

    # The original's agents take a DataFrame; folder/database tasks have no counterpart here.
    comparable = {"data_cleaning", "data_wrangling", "data_visualization"}
    tasks = [t for t in load_tasks() if t.agent in comparable and not t.files and not t.database]
    tasks = [t for t in tasks if not args.only or any(s in t.id for s in args.only)]
    tasks = tasks[: args.limit]
    config = LLMConfig.from_env().model_copy(update={"fallback_model": None})
    try:
        client = docker.from_env()
        client.images.get(IMAGE)
    except DockerException as exc:
        print(f"Docker or {IMAGE} not available ({exc}). Build it first; see the docstring.")
        return 2

    eval_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out_dir = Path("runs/parity") / eval_id
    out_dir.mkdir(parents=True)
    meta: dict[str, Any] = {
        "eval_id": eval_id, "label": args.label, "system": IMAGE, "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "options": {"max_retries": args.max_retries, "no_checks": False, "no_recommend": False},
        "tasks": [t.id for t in tasks], "started": datetime.now(UTC).isoformat(),
    }  # fmt: skip
    print(
        f"Parity {eval_id}: {len(tasks)} tasks on {IMAGE}, {config.model}, effort={config.reasoning_effort}"
    )

    results: list[TaskResult] = []
    with (
        (out_dir / "results.jsonl").open("a", encoding="utf-8") as f,
        ThreadPoolExecutor(args.workers) as pool,
    ):
        futures = [
            pool.submit(
                run_one, t, config, out_dir / "work", client, args.max_retries, args.timeout
            )
            for t in tasks
        ]
        for future in futures:
            r = future.result()
            results.append(r)
            f.write(json.dumps(r.as_dict()) + "\n")
            f.flush()
            print(f"  {r.outcome:<15} {r.status:<8} {r.task_id}  ({r.calls} calls, {r.seconds}s)")

    summary = summarise(results)
    meta.update(summary=summary, finished=datetime.now(UTC).isoformat())
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n" + format_summary(summary) + f"\n\nResults: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
