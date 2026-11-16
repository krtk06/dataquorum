"""Run eval tasks through the real agents and score them against known answers.

    uv run python -m dataquorum.evals.runner --dry-run              # build + oracle only
    uv run python -m dataquorum.evals.runner --only clean-adult --limit 5
    uv run python -m dataquorum.evals.runner --label baseline
    uv run python -m dataquorum.evals.runner --no-checks --label ablate-checks
    uv run python -m dataquorum.evals.runner --repeats 3 --label pass-at-k

Ablations (recorded in summary.json, compare runs with `python -m dataquorum.evals.compare`):
- `--no-checks`     agents run without generic or agent-specific validation checks
- `--no-recommend`  code is generated straight from the request, without a step plan
- `--max-retries N` fix attempts after a crash or failed blocking check (0 = none)
- `--repeats N`     run every task N times (for pass@k and run-to-run variance)

Every run gets an id, results.jsonl and summary.json in runs/evals/<id>/; each task's
AgentResult is recorded in DuckDB with its eval task id. One model and effort per run (no
fallback). The budget guard runs in eval mode: hitting the limit stops the run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataquorum.agents import AGENTS
from dataquorum.core.budget import BudgetExceededError, BudgetGuard
from dataquorum.core.engine import Engine, StructuredLLM
from dataquorum.core.llm import LLMClient, LLMConfig
from dataquorum.core.state import AgentResult, DataFrameOutput, FigureOutput
from dataquorum.evals.checkers import Output, run_checker
from dataquorum.evals.oracle import check_task
from dataquorum.evals.scoring import TaskResult, classify, format_summary, summarise
from dataquorum.evals.tasks import Task, build, load_tasks
from dataquorum.sandbox.docker_executor import DockerExecutor, SandboxUnavailableError
from dataquorum.sandbox.executor import Executor
from dataquorum.tracing.recorder import DuckDBLedger, RunRecorder, git_commit


@dataclasses.dataclass(frozen=True)
class RunOptions:
    max_retries: int = 2
    no_checks: bool = False
    no_recommend: bool = False
    repeats: int = 1


def output_of(result: AgentResult) -> Output:
    if isinstance(result.output, DataFrameOutput):
        return result.output.load()
    if isinstance(result.output, FigureOutput):
        return result.output.figure
    return None


def run_task(
    task: Task, llm: StructuredLLM, executor: Executor, runs_dir: Path, eval_id: str,
    options: RunOptions, repeat: int = 0,
) -> tuple[TaskResult, AgentResult]:  # fmt: skip
    built = build(task)
    spec = AGENTS[task.agent]
    if options.no_checks:
        spec = dataclasses.replace(spec, generic_checks=(), extra_checks=())
    engine = Engine(spec, llm, executor, runs_dir=runs_dir, max_retries=options.max_retries,
                    skip_recommend=options.no_recommend)  # fmt: skip
    run_id = f"{eval_id}-{task.id}" + (f"-r{repeat}" if options.repeats > 1 else "")
    run = engine.start(built.messy, built.request, run_id=run_id)
    assert run.result is not None  # no human review in evals
    result = run.result
    output = output_of(result)
    checks = [run_checker(c.kind, built, output, c.params) for c in task.checks]
    u = result.usage
    scored = TaskResult(
        task_id=task.id, agent=task.agent, tags=task.tags, expect_status=task.expect_status,
        status=str(result.status), outcome=classify(str(result.status), task.expect_status, checks),
        checks=checks, error=result.error, run_id=result.run_id, calls=u.calls, retries=u.retries,
        completion_tokens=u.completion_tokens, reasoning_tokens=u.reasoning_tokens,
        allowance_usd=u.allowance_usd, seconds=round(u.seconds + u.execution_seconds, 2),
        llm_seconds=round(u.seconds, 2), execution_seconds=round(u.execution_seconds, 2), repeat=repeat,
    )  # fmt: skip
    return scored, result


def run_eval(
    tasks: list[Task],
    *,
    llm_factory: Callable[[], StructuredLLM],
    executor: Executor,
    recorder: RunRecorder | None,
    out_dir: Path,
    workers: int = 3,
    options: RunOptions | None = None,
    on_result: Callable[[TaskResult], None] | None = None,
) -> tuple[list[TaskResult], str | None]:
    """Run tasks (x repeats) in parallel. Returns results and, if stopped early, why."""
    options = options or RunOptions()
    eval_id = out_dir.name
    llm = llm_factory()
    results: list[TaskResult] = []
    stopped: str | None = None
    jobs = [(t, r) for r in range(options.repeats) for t in tasks]
    with (
        ThreadPoolExecutor(workers) as pool,
        (out_dir / "results.jsonl").open("a", encoding="utf-8") as f,
    ):
        pending = {
            pool.submit(run_task, t, llm, executor, out_dir / "work", eval_id, options, r)
            for t, r in jobs
        }
        while pending:
            done, pending = wait(pending, return_when=FIRST_EXCEPTION)
            for future in done:
                try:
                    scored, result = future.result()
                except (BudgetExceededError, SandboxUnavailableError) as exc:
                    stopped = f"{type(exc).__name__}: {exc}"
                    for other in pending:
                        other.cancel()
                    continue
                if recorder is not None:
                    recorder.record(result, eval_task_id=scored.task_id)
                results.append(scored)
                f.write(json.dumps(scored.as_dict()) + "\n")
                f.flush()
                if on_result:
                    on_result(scored)
            if stopped:
                pending = {p for p in pending if not p.cancelled() and not p.done()}
    return results, stopped


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", action="append", default=[], help="task id substring (repeatable)")
    p.add_argument("--tag", action="append", default=[], help="keep tasks with this tag")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--no-checks", action="store_true")
    p.add_argument("--no-recommend", action="store_true")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--model")
    p.add_argument("--reasoning-effort", choices=["default", "minimal", "low", "medium", "high"])
    p.add_argument("--label", default="")
    p.add_argument("--dry-run", action="store_true", help="build tasks and run the oracle only")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    tasks = [t for t in load_tasks() if (not args.only or any(s in t.id for s in args.only))]
    tasks = [t for t in tasks if not args.tag or set(args.tag) & set(t.tags)][: args.limit]
    if args.dry_run:
        bad = {t.id: probs for t in tasks if (probs := check_task(build(t)))}
        print(f"{len(tasks)} tasks built; {len(bad)} with problems: {bad or ''}")
        return 1 if bad else 0

    updates: dict[str, Any] = {"fallback_model": None}  # one model per run
    if args.model:
        updates["model"] = args.model
    if args.reasoning_effort:
        updates["reasoning_effort"] = args.reasoning_effort
    config = LLMConfig.from_env().model_copy(update=updates)
    options = RunOptions(args.max_retries, args.no_checks, args.no_recommend, args.repeats)

    eval_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out_dir = Path("runs/evals") / eval_id
    out_dir.mkdir(parents=True)
    recorder = RunRecorder()
    budget = BudgetGuard(DuckDBLedger(recorder), mode="eval")
    meta: dict[str, Any] = {
        "eval_id": eval_id, "label": args.label, "model": config.model,
        "reasoning_effort": config.reasoning_effort, "options": dataclasses.asdict(options),
        "tasks": [t.id for t in tasks], "git_commit": git_commit(),
        "started": datetime.now(UTC).isoformat(),
    }  # fmt: skip
    print(f"Eval {eval_id}: {len(tasks)} tasks x {options.repeats}, {config.model}, "
          f"effort={config.reasoning_effort}, options={dataclasses.asdict(options)}")  # fmt: skip

    def show(r: TaskResult) -> None:
        print(f"  {r.outcome:<15} {r.status:<13} {r.task_id}  ({r.calls} calls, {r.seconds}s)")

    start = time.perf_counter()
    results, stopped = run_eval(
        tasks,
        llm_factory=lambda: LLMClient(config, budget=budget, session_id=f"dq-eval-{eval_id}"),
        executor=DockerExecutor(),
        recorder=recorder,
        out_dir=out_dir,
        workers=args.workers,
        options=options,
        on_result=show,
    )
    summary = summarise(results)
    meta.update(
        summary=summary, stopped=stopped, wall_seconds=round(time.perf_counter() - start, 1)
    )
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n" + format_summary(summary))
    if stopped:
        print(f"\nSTOPPED EARLY: {stopped}")
    print(f"\nResults: {out_dir}")
    return 2 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
