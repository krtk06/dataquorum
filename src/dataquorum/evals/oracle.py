"""Oracle check: validate the eval tasks themselves, before any agent runs them.

For every task:
- a reference ("oracle") answer must pass every checker -> the task is solvable and the
  checkers agree with the intended answer;
- the untouched input must fail at least one checker -> the task is not trivially passed by
  returning the input.

    uv run python -m dataquorum.evals.oracle
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
import plotly.express as px

from dataquorum.evals.checkers import Output, run_checker
from dataquorum.evals.tasks import BuiltTask, Task, build, load_tasks


def _spec(task: Task, kind: str) -> dict[str, Any] | None:
    for check in task.checks:
        if check.kind == kind:
            return check.params
    return None


def _aggregate_frame(b: BuiltTask, p: dict[str, Any]) -> pd.DataFrame:
    keys = [p["group_by"]] if isinstance(p["group_by"], str) else list(p["group_by"])
    source = b.clean.query(p["query"], engine="python") if p.get("query") else b.clean
    grouped = source.groupby(keys)[p["value"]]
    series = grouped.size() if p.get("agg") == "count" else grouped.agg(p.get("agg", "sum"))
    if p.get("top"):
        series = series.sort_values(ascending=False).head(p["top"])
    elif p.get("order"):
        series = series.sort_values(ascending=p["order"] != "desc")
    frame: pd.DataFrame = series.rename("result").reset_index()
    return frame


def oracle_output(b: BuiltTask) -> Output:
    t = b.task
    if t.expect_status == "failed":
        return None
    if t.agent == "data_cleaning":
        return b.expected
    if (p := _spec(t, "aggregate")) is not None:
        return _aggregate_frame(b, p)
    if (p := _spec(t, "filter_rows")) is not None:
        return b.clean.query(p["query"], engine="python")
    if (p := _spec(t, "derived_column")) is not None:
        return b.clean.assign(result=pd.Series(b.clean.eval(p["expression"])))
    if (p := _spec(t, "figure_aggregate")) is not None:
        frame = _aggregate_frame(b, p)
        return dict(px.bar(frame, x=p["group_by"], y="result").to_plotly_json())
    types = (_spec(t, "figure_type") or {}).get("types", [])
    first, second = b.clean.columns[0], b.clean.columns[1]
    if "histogram" in types:
        return dict(px.histogram(b.clean, x=first).to_plotly_json())
    if "box" in types:
        return dict(px.box(b.clean, x="class", y="plas").to_plotly_json())
    return dict(px.scatter(b.clean, x=first, y=second).to_plotly_json())


def _as_json(output: Output) -> Output:
    """Round-trip figures through JSON like real runs do (typed arrays, no numpy objects)."""
    if isinstance(output, dict):
        import json

        import plotly.io as pio

        return dict(json.loads(pio.to_json(output)))
    return output


def check_task(b: BuiltTask) -> list[str]:
    """Problems with one task; empty when the task is well formed."""
    problems = []
    oracle = _as_json(oracle_output(b))
    for spec in b.task.checks:
        outcome = run_checker(spec.kind, b, oracle, spec.params)
        if not outcome.passed:
            problems.append(f"oracle fails {outcome.name}: {outcome.detail}")
    if b.task.expect_status == "ok":
        naive = b.messy if b.task.agent != "data_visualization" else None
        if all(run_checker(s.kind, b, naive, s.params).passed for s in b.task.checks):
            problems.append("returning the input unchanged passes every check (task is trivial)")
    return problems


def main() -> int:
    bad = 0
    for task in load_tasks():
        problems = check_task(build(task))
        status = "ok " if not problems else "BAD"
        bad += bool(problems)
        print(f"{status} {task.id}" + "".join(f"\n      - {p}" for p in problems))
    print(f"\n{bad} problem task(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
