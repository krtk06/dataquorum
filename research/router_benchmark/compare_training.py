"""Compare the original and expanded MiniLM training sets against a fresh holdout."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from benchmark_routes import Case, load_cases, predict_cases, score
from pydantic import BaseModel

from dataquorum.agents import AGENTS
from dataquorum.evals.stats import mcnemar
from dataquorum.sandbox.executor import InputKind
from dataquorum.supervisor.router_learned import LearnedRouter


class RequestGroup(BaseModel):
    kind: InputKind
    requests: list[str | tuple[InputKind, str]]


def read_requests(stem: str) -> list[Case]:
    path = Path(__file__).with_name(stem + ".yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(raw) != {*AGENTS, "refuse"}:
        raise ValueError(f"{path.name} must cover every route and refusal")
    cases = []
    for label, value in raw.items():
        data = RequestGroup.model_validate(value)
        for index, item in enumerate(data.requests):
            kind, request = (data.kind, item) if isinstance(item, str) else item
            if label != "refuse" and kind != AGENTS[label].input_kind:
                raise ValueError(f"Wrong input kind for {label}")
            cases.append(Case(f"{stem}-{label}-{index}", request, label, kind))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-dir", type=Path, default=Path("runs/router-model"))
    parser.add_argument("--out", type=Path, default=Path("runs/week13-more-examples"))
    args = parser.parse_args()

    original, original_holdout = load_cases()
    fresh = read_requests("fresh_holdout")
    additional = read_requests("additional_training")
    cases = original + original_holdout + fresh + additional
    normalized = {re.sub(r"\W+", " ", c.request.casefold()).strip() for c in cases}
    if len(normalized) != len(cases):
        raise ValueError("Unexpected duplicate request across the training and test sets")

    result: dict[str, Any] = {}
    for name, holdout in (("fresh", fresh), ("original", original_holdout)):
        result[name] = {"n": len(holdout)}
    outcomes: dict[str, dict[str, bool]] = {}
    predictions = []
    for name, training in (("baseline", original), ("expanded", original + additional)):
        router = LearnedRouter.fit(
            [c.request for c in training],
            [c.expected for c in training],
            encoder="minilm",
            directory=args.encoder_dir,
        )
        correct: dict[str, bool] = {}
        for holdout_name, holdout in (("fresh", fresh), ("original", original_holdout)):
            rows = predict_cases(router, holdout)
            result[holdout_name][name] = {"train": len(training), **score(holdout, rows)}
            for case, prediction in zip(holdout, rows, strict=True):
                correct[case.id] = prediction.label == case.expected
                predictions.append(
                    {
                        "train": name,
                        "holdout": holdout_name,
                        "expected": case.expected,
                        **asdict(prediction),
                    }
                )
        outcomes[name] = correct

    for name, holdout in (("fresh", fresh), ("original", original_holdout)):
        result[name]["paired"] = asdict(
            mcnemar(
                {c.id: outcomes["expanded"][c.id] for c in holdout},
                {c.id: outcomes["baseline"][c.id] for c in holdout},
            )
        )
    args.out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2) + "\n"
    (args.out / "summary.json").write_text(text, encoding="utf-8")
    (args.out / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
