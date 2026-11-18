"""Compare first-step routers on one synthetic holdout with fixed expected labels."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from dataquorum.agents import AGENTS
from dataquorum.core.budget import BudgetGuard
from dataquorum.core.llm import LLMClient, LLMConfig, LLMError
from dataquorum.evals.stats import mcnemar, wilson_interval
from dataquorum.sandbox.executor import InputKind
from dataquorum.supervisor.router_learned import LearnedRouter, download_encoder
from dataquorum.supervisor.router_llm import LLMRouter
from dataquorum.tracing.recorder import DuckDBLedger, RunRecorder

CORPUS = Path(__file__).with_name("corpus.yaml")


@dataclass(frozen=True)
class Case:
    id: str
    request: str
    expected: str
    kind: InputKind


class Examples(BaseModel):
    kind: InputKind
    values: list[str]
    train: list[str]
    test: list[str | tuple[InputKind, str]]


def load_cases() -> tuple[list[Case], list[Case]]:
    raw = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))
    if set(raw) != {*AGENTS, "refuse"}:
        raise ValueError("The corpus must cover the current agents and refusal")
    train, test = [], []
    for label, config in raw.items():
        examples = Examples.model_validate(config)
        if label in AGENTS and examples.kind != AGENTS[label].input_kind:
            raise ValueError(f"Wrong input kind for {label}")
        for i, template in enumerate(examples.train):
            for j, value in enumerate(examples.values):
                train.append(
                    Case(f"{label}-train-{i}-{j}", template.format(value), label, examples.kind)
                )
        for i, item in enumerate(examples.test):
            kind, request = (examples.kind, item) if isinstance(item, str) else item
            if label in AGENTS and kind != AGENTS[label].input_kind:
                raise ValueError(f"Wrong holdout input kind for {label}")
            test.append(Case(f"{label}-test-{i}", request, label, kind))
    if len({c.request.casefold() for c in train + test}) != len(train) + len(test):
        raise ValueError("Duplicate requests in the corpus")
    return train, test


@dataclass(frozen=True)
class Prediction:
    id: str
    label: str
    seconds: float
    allowance_usd: float = 0.0


def predict_cases(router: LearnedRouter, cases: list[Case]) -> list[Prediction]:
    rows = []
    for case in cases:
        started = time.perf_counter()
        label = router.predict(case.request, case.kind)
        rows.append(Prediction(case.id, label, time.perf_counter() - started))
    return rows


def _live(test: list[Case], path: Path) -> list[Prediction]:
    config = LLMConfig.from_env().model_copy(update={"fallback_model": None, "max_retries": 0})
    metadata = {
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "holdout_sha256": sha256(json.dumps([asdict(c) for c in test]).encode()).hexdigest(),
    }
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Model or holdout changed. Choose a new output directory.")
    else:
        if path.exists():
            raise ValueError("Existing predictions have no matching metadata")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            prior = Prediction(**json.loads(line))
            done[prior.id] = prior

    recorder = RunRecorder()
    try:
        router = LLMRouter(
            LLMClient(config, budget=BudgetGuard(DuckDBLedger(recorder), mode="eval"))
        )
        with path.open("a", encoding="utf-8") as log:
            for case in test:
                if case.id in done:
                    continue
                start = time.perf_counter()
                label: str
                try:
                    out = router.route(
                        case.request,
                        case.kind,
                        "Synthetic benchmark. Assume named columns, files and tables exist.",
                    )
                    label = "refuse" if out.value.refusal is not None else out.value.next
                    cost = out.usage.allowance_usd
                except LLMError as exc:
                    label, cost = "error", exc.usage.allowance_usd
                prediction = Prediction(case.id, label, time.perf_counter() - start, cost)
                done[case.id] = prediction
                log.write(json.dumps(asdict(prediction)) + "\n")
                log.flush()
                print(f"{len(done)}/{len(test)} {case.id}: {label}", flush=True)
    finally:
        recorder.close()
    return [done[c.id] for c in test]


def score(test: list[Case], rows: list[Prediction]) -> dict[str, Any]:
    labels = {p.id: p.label for p in rows}
    correct = {c.id: labels[c.id] == c.expected for c in test}
    counts = Counter(c.expected for c in test)
    hits = Counter(c.expected for c in test if correct[c.id])
    return {
        "correct": sum(correct.values()),
        "n": len(test),
        "wilson_95": wilson_interval(sum(correct.values()), len(test)),
        "dataframe_correct": sum(correct[c.id] for c in test if c.kind == "dataframe"),
        "dataframe_n": sum(c.kind == "dataframe" for c in test),
        "by_label": {name: [hits[name], n] for name, n in counts.items()},
        "median_seconds": statistics.median(p.seconds for p in rows),
        "allowance_usd": sum(p.allowance_usd for p in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/week13-routing"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("runs/router-model"))
    parser.add_argument("--download", action="store_true", help="Fetch the pinned MiniLM encoder")
    parser.add_argument(
        "--live", action="store_true", help="Run the held-out requests on OpenCode Go"
    )
    args = parser.parse_args()
    train, test = load_cases()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.download:
        download_encoder(args.encoder_dir)
    requests = [c.request for c in train]
    labels = [c.expected for c in train]
    results: dict[str, Any] = {"train": len(train), "test": len(test)}
    predictions: dict[str, list[Prediction]] = {}
    for encoder in ("tfidf", "minilm"):
        clock = time.perf_counter()
        learned = LearnedRouter.fit(requests, labels, encoder=encoder, directory=args.encoder_dir)
        fit_seconds = time.perf_counter() - clock
        rows = predict_cases(learned, test)
        results[encoder] = {**score(test, rows), "setup_and_fit_seconds": fit_seconds}
        predictions[encoder] = rows
    if args.live:
        rows = _live(test, args.out / "llm.jsonl")
        results["llm"] = score(test, rows)
        predictions["llm"] = rows
    expected = {c.id: c.expected for c in test}
    match = {
        name: {p.id: p.label == expected[p.id] for p in rows} for name, rows in predictions.items()
    }
    results["paired_vs_minilm"] = {
        name: asdict(mcnemar(match["minilm"], hits))
        for name, hits in match.items()
        if name != "minilm"
    }
    (args.out / "predictions.json").write_text(
        json.dumps(
            {name: [asdict(r) for r in rows] for name, rows in predictions.items()}, indent=2
        ),
        encoding="utf-8",
    )
    text = json.dumps(results, indent=2)
    (args.out / "summary.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
