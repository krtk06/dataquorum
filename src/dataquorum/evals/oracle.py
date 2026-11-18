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
from collections import Counter
from typing import Any

import pandas as pd
import plotly.express as px

from dataquorum.evals.checkers import Output, expected_aggregate, run_checker
from dataquorum.evals.checkers_ml import STATS
from dataquorum.evals.tasks import BuiltTask, Task, build, load_tasks


def _spec(task: Task, kind: str) -> dict[str, Any] | None:
    for check in task.checks:
        if check.kind == kind:
            return check.params
    return None


def _aggregate_frame(
    b: BuiltTask, p: dict[str, Any], extra_keys: list[str] | None = None
) -> pd.DataFrame:
    keys = [p["group_by"]] if isinstance(p["group_by"], str) else list(p["group_by"])
    keys += extra_keys or []
    series = expected_aggregate(
        b, keys, p["value"], p.get("agg", "sum"), p.get("query"), p.get("value_expr"),
        p.get("min_count"), p.get("top"), p.get("top_per"),
    )  # fmt: skip
    if p.get("order") and not p.get("top_per"):
        series = series.sort_values(ascending=p["order"] != "desc")
    frame: pd.DataFrame = series.rename("result").reset_index()
    return frame


def oracle_output(b: BuiltTask) -> Output:
    t = b.task
    if t.expect_status == "failed":
        return None
    if t.agent in ("data_cleaning", "data_loader"):
        return b.expected
    if t.agent == "ml_correctness":
        from dataquorum.agents.correctness import requested_audit

        assert isinstance(b.messy, pd.DataFrame)
        return requested_audit(b.messy, b.request)
    if t.agent == "feature_engineering":
        return _features(b)
    aggregates = [c.params for c in t.checks if c.kind == "aggregate"]
    if aggregates:
        # Several aggregate checks (e.g. count, median, mean per group) -> one table.
        frame = _aggregate_frame(b, aggregates[0])
        for i, extra in enumerate(aggregates[1:], 1):
            by = extra["group_by"]
            keys = [by] if isinstance(by, str) else list(by)
            more = _aggregate_frame(b, extra).rename(columns={"result": f"result_{i}"})
            frame = frame.merge(more, on=keys, how="left")
        return frame
    if (p := _spec(t, "pivot")) is not None:
        table = b.clean.pivot_table(index=p["index"], columns=p["columns"], values=p["value"],
                                    aggfunc=p.get("agg", "mean"))  # fmt: skip
        flat: pd.DataFrame = table.reset_index()
        return flat
    if (p := _spec(t, "binned_count")) is not None:
        bands = pd.cut(
            b.clean[p["column"]], bins=p["bins"], labels=p["labels"], right=p.get("right", False)
        )
        counts: pd.DataFrame = pd.Series(bands).value_counts().rename("n").reset_index()
        return counts
    if (p := _spec(t, "per_column_stat")) is not None:
        source = b.messy if isinstance(b.messy, pd.DataFrame) else b.clean
        stat = STATS[p["stat"]]
        return pd.DataFrame(
            {"column": p["columns"], "value": [stat(source[c]) for c in p["columns"]]}
        )
    if (p := _spec(t, "correlation")) is not None:
        source = b.messy if isinstance(b.messy, pd.DataFrame) else b.clean
        r = source[p["x"]].corr(source[p["y"]], method=p.get("method", "pearson"))
        return pd.DataFrame({"x": [p["x"]], "y": [p["y"]], "corr": [r]})
    if (p := _spec(t, "filter_rows")) is not None:
        return b.clean.query(p["query"], engine="python")
    if (p := _spec(t, "derived_column")) is not None:
        return b.clean.assign(result=pd.Series(b.clean.eval(p["expression"])))
    if (p := _spec(t, "figure_aggregate")) is not None:
        color = p.get("color")
        frame = _aggregate_frame(b, p, [color] if color else None)
        wanted = (_spec(t, "figure_type") or {}).get("types", ["bar"])
        if "bar" in wanted or "histogram" in wanted:
            figure = px.bar(frame, x=p["group_by"], y="result", color=color, barmode="group")
        else:
            figure = px.line(frame, x=p["group_by"], y="result", color=color)
        return dict(figure.to_plotly_json())
    types = (_spec(t, "figure_type") or {}).get("types", [])
    first, second = b.clean.columns[0], b.clean.columns[1]
    if "histogram" in types:
        return dict(px.histogram(b.clean, x=first).to_plotly_json())
    if "box" in types:
        return dict(px.box(b.clean, x="class", y="plas").to_plotly_json())
    return dict(px.scatter(b.clean, x=first, y=second).to_plotly_json())


def _features(b: BuiltTask) -> pd.DataFrame:
    """Reference feature table built from the task's own checks."""
    assert isinstance(b.messy, pd.DataFrame)
    src, t = b.messy, b.task
    target = (_spec(t, "column_unchanged") or {}).get("column")
    scaled = {c: "standardized" for c in (_spec(t, "standardized") or {}).get("columns", [])}
    scaled |= {c: "minmax" for c in (_spec(t, "minmax") or {}).get("columns", [])}
    encoded = [c.params["column"] for c in t.checks if c.kind == "onehot"]
    parts: list[pd.Series | pd.DataFrame] = []
    for col in src.columns:
        if col == target or col in encoded:
            continue
        if pd.api.types.is_numeric_dtype(src[col]):
            v = src[col].astype(float).fillna(src[col].median())
            if scaled.get(col) == "standardized":
                v = (v - v.mean()) / v.std(ddof=0)
            elif scaled.get(col) == "minmax":
                v = (v - v.min()) / (v.max() - v.min())
            parts.append(v.rename(col))
    for col in encoded:
        parts.append(pd.get_dummies(src[col].astype(str), prefix=col, dtype=float))
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=src.index)
    if target:
        out[target] = src[target]
    return out


def _as_json(output: Output) -> Output:
    """Round-trip figures through JSON like real runs do (typed arrays, no numpy objects)."""
    if isinstance(output, dict):
        import json

        import plotly.io as pio

        return dict(json.loads(pio.to_json(output)))
    return output


def label_problems(b: BuiltTask) -> list[str]:
    """Text answers must be knowable from the input: for "use the spelling most rows use",
    each expected label has to be the strict-majority spelling among its case/space variants
    in the messy input. (Week 8: a rare label that only appeared as a variant made a correct
    agent look like a false success.)"""
    problems: list[str] = []
    if not isinstance(b.messy, pd.DataFrame):
        return problems
    for spec in b.task.checks:
        if spec.kind != "matches_expected":
            continue
        for col in spec.params.get("columns", []):
            # Text columns: object in pandas 2, `str` dtype in pandas 3.
            if col not in b.messy or pd.api.types.is_numeric_dtype(b.expected[col]):
                continue
            spellings: dict[str, Counter[str]] = {}
            for v in b.messy[col]:
                if isinstance(v, str):
                    spellings.setdefault(v.strip().lower(), Counter())[v] += 1
            for label in set(b.expected[col].dropna()):
                counts = spellings.get(str(label).strip().lower(), Counter())
                top = counts.most_common(2)
                if counts and (counts[label] == 0 or (len(top) > 1 and top[1][1] >= counts[label]
                               and top[0][0] != label) or counts[label] < max(counts.values())):  # fmt: skip
                    problems.append(
                        f"{col}: {label!r} is not the majority spelling in the input {dict(counts)}"
                    )
    return problems


def check_task(b: BuiltTask) -> list[str]:
    """Problems with one task; empty when the task is well formed."""
    problems = label_problems(b)
    oracle = _as_json(oracle_output(b))
    for spec in b.task.checks:
        outcome = run_checker(spec.kind, b, oracle, spec.params)
        if not outcome.passed:
            problems.append(f"oracle fails {outcome.name}: {outcome.detail}")
    if b.task.expect_status != "failed":
        naive: Output = None  # folders, databases and charts have no "unchanged input" answer
        if isinstance(b.messy, pd.DataFrame) and b.task.agent != "data_visualization":
            naive = b.messy
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
