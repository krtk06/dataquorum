"""Compare two eval runs on the tasks they share.

    uv run python -m dataquorum.evals.compare runs/evals/<A> runs/evals/<B>

Reports, for each run, accuracy and false-success rate with 95% Wilson intervals; McNemar's
exact test on tasks whose correctness differs between the runs (paired, so it uses the
same tasks); a paired bootstrap interval for the change in cost and time per task; and the
list of tasks that flipped. With repeated runs, a task counts as correct in a run if it was
correct in the majority of its attempts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dataquorum.evals.stats import bootstrap_ci, mcnemar, wilson_interval


def load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    lines = (path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return meta, [json.loads(line) for line in lines if line.strip()]


def per_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    out = {}
    for task_id, attempts in grouped.items():
        correct = sum(a["outcome"] == "correct" for a in attempts)
        out[task_id] = {
            "correct": correct * 2 > len(attempts),
            "false_success": any(a["outcome"] == "false_success" for a in attempts),
            "cost": float(np.mean([a["allowance_usd"] for a in attempts])),
            "seconds": float(np.mean([a["seconds"] for a in attempts])),
            "outcomes": sorted({a["outcome"] for a in attempts}),
        }
    return out


def _rate(k: int, n: int) -> str:
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} ({k / n:.0%}, 95% CI {lo:.0%}-{hi:.0%})" if n else "-"


def _describe(meta: dict[str, Any]) -> str:
    opts = meta.get("options", {})
    flags = [k for k in ("no_checks", "no_recommend") if opts.get(k)]
    return (
        f"{meta.get('label') or meta['eval_id']}: {meta['model']}, effort={meta['reasoning_effort']}, "
        f"retries={opts.get('max_retries', meta.get('max_retries'))}"
        + (f", {', '.join(flags)}" if flags else "")
    )


def compare(a_dir: Path, b_dir: Path) -> str:
    meta_a, rows_a = load_run(a_dir)
    meta_b, rows_b = load_run(b_dir)
    a, b = per_task(rows_a), per_task(rows_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two runs share no tasks")
    n = len(shared)
    lines = [f"A = {_describe(meta_a)}", f"B = {_describe(meta_b)}", f"Shared tasks: {n}", ""]
    lines += ["| | A | B |", "|---|---|---|"]
    for label, key in (("Correct", "correct"), ("False success", "false_success")):
        ka, kb = sum(a[t][key] for t in shared), sum(b[t][key] for t in shared)
        lines.append(f"| {label} | {_rate(ka, n)} | {_rate(kb, n)} |")
    for label, key, fmt in (
        ("Allowance $ / task", "cost", ".5f"),
        ("Seconds / task", "seconds", ".1f"),
    ):
        ma = np.mean([a[t][key] for t in shared])
        mb = np.mean([b[t][key] for t in shared])
        lines.append(f"| {label} | {ma:{fmt}} | {mb:{fmt}} |")

    test = mcnemar({t: a[t]["correct"] for t in shared}, {t: b[t]["correct"] for t in shared})
    lines += [
        "",
        f"McNemar (exact): only A correct = {test.only_a}, only B correct = {test.only_b}, "
        f"p = {test.p_value:.3g}",
    ]
    for label, key in (("cost", "cost"), ("time", "seconds")):
        diffs = [b[t][key] - a[t][key] for t in shared]
        lo, hi = bootstrap_ci(diffs, seed=0)
        lines.append(
            f"B - A {label} per task: {np.mean(diffs):+.5g} (95% bootstrap CI {lo:+.5g} to {hi:+.5g})"
        )

    flipped = [t for t in shared if a[t]["correct"] != b[t]["correct"]]
    if flipped:
        lines += ["", "Tasks that changed:"]
        lines += [f"- {t}: A {a[t]['outcomes']} -> B {b[t]['outcomes']}" for t in flipped]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args(argv)
    print(compare(args.run_a, args.run_b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
