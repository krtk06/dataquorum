"""Re-apply the current checkers to a finished eval run's saved outputs (no LLM calls).

    uv run python -m dataquorum.evals.rescore runs/evals/<id>

Use it after fixing a checker: the agent outputs stay exactly the same, only the grading
changes. Writes `results.rescored.jsonl` next to the original and prints both summaries.
Table outputs are read from each run's `result.parquet`; tasks whose output is not a saved
table (figures, failed runs) keep their original outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from dataquorum.evals.checkers import run_checker
from dataquorum.evals.scoring import TaskResult, classify, format_summary, summarise
from dataquorum.evals.tasks import build, load_tasks


def rescore(run_dir: Path) -> tuple[list[TaskResult], list[TaskResult]]:
    tasks = {t.id: t for t in load_tasks()}
    before: list[TaskResult] = []
    after: list[TaskResult] = []
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        row: dict[str, Any] = json.loads(line)
        old = _to_result(row)
        before.append(old)
        output_file = run_dir / "work" / str(row["run_id"]) / "result.parquet"
        task = tasks.get(row["task_id"])
        if task is None or not output_file.exists():
            after.append(old)
            continue
        built = build(task)
        output = pd.read_parquet(output_file)
        checks = [run_checker(c.kind, built, output, c.params) for c in task.checks]
        outcome = classify(row["status"], task.expect_status, checks)
        after.append(
            _to_result(
                {
                    **row,
                    "expect_status": task.expect_status,
                    "checks": [vars(c) for c in checks],
                    "outcome": outcome,
                }
            )
        )
    return before, after


def _to_result(row: dict[str, Any]) -> TaskResult:
    from dataquorum.evals.checkers import CheckOutcome

    fields = {k: v for k, v in row.items() if k in TaskResult.__dataclass_fields__}
    fields["checks"] = [CheckOutcome(**c) for c in row.get("checks", [])]
    return TaskResult(**fields)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    before, after = rescore(args.run_dir)
    out = args.run_dir / "results.rescored.jsonl"
    out.write_text("".join(json.dumps(r.as_dict()) + "\n" for r in after), encoding="utf-8")
    changed = [
        (b.task_id, b.outcome, a.outcome)
        for b, a in zip(before, after, strict=True)
        if b.outcome != a.outcome
    ]
    print("Before:\n" + format_summary(summarise(before)))
    print("\nAfter rescoring:\n" + format_summary(summarise(after)))
    print(
        "\nChanged:" + "".join(f"\n- {t}: {o} -> {n}" for t, o, n in changed)
        if changed
        else "\nNo outcome changed."
    )
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
