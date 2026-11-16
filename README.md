# DataQuorum

**A team of AI data-science agents that must prove their work.**

> Status: **v0.1.0 released; main is at week 9 of 17.** Seven agents work end to end: cleaning,
> wrangling, visualization, loader (folders of files), SQL (read-only queries on SQLite or
> DuckDB), EDA (with findings checked against the numbers) and feature engineering
> (scikit-learn, with target-leakage checks). AutoML, a supervisor and the Studio app are
> still to come.

DataQuorum is a Python library of LLM agents for data-science work. An agent takes a pandas
DataFrame and a request in plain English, writes code, runs it in a locked-down Docker
sandbox, checks the result, and reports `success`, `failed` or `needs_review`. The status is
computed from checks, never chosen by the model, so an empty or invented result is never
reported as done.

## Why

DataQuorum rebuilds the ideas of
[business-science/ai-data-science-team](https://github.com/business-science/ai-data-science-team)
with a different focus: proving the work. On the same 55 tasks, model and retry budget
([details](docs/results/week7-parity.md)):

| | Original | DataQuorum |
|---|---|---|
| Correct | 47/55 (85%) | 55/55 (100%) |
| Reported success on a wrong or invented answer | 7/55 (13%) | 0/55 |
| Impossible requests refused | 0/3 | 3/3 |
| Cost per task | ~$0.0037 | $0.0008 |

McNemar exact test on the paired tasks: p = 0.008. The tasks were written alongside
DataQuorum, so read this as an early, home-advantage result.

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, and an OpenAI-compatible API
key (developed against DeepSeek V4.1 Flash via OpenCode Go).

```bash
git clone https://github.com/krtk06/dataquorum && cd dataquorum
uv sync
docker build -t dataquorum/sandbox:0.2 images/sandbox
cp .env.example .env   # then add OPENCODE_GO_API_KEY
```

```python
import pandas as pd
from dataquorum.agents import CLEANING, WRANGLING, VISUALIZATION, make_engine

df = pd.DataFrame({"price": ["$1,200", "n/a", "$3.4k"], "city": [" boston", "NYC", "boston"]})

run = make_engine(CLEANING).start(
    df, "Convert price to numbers ('$3.4k' = 3400, 'n/a' = missing). Strip and title-case city."
)
result = run.result
print(result.status, "-", result.message)   # success - data_cleaning finished and all checks passed.
print(result.output.load())                 # the cleaned DataFrame
print(result.code)                          # the code that produced it
print(result.usage.allowance_usd, result.usage.calls)
```

Ask for a plan first and approve it:

```python
engine = make_engine(WRANGLING)
run = engine.start(df, "Average price per city", human_review=True)
print(run.review["steps"])                  # the proposed steps
run = engine.resume(run.run_id, "Yes")      # or "Use the median instead", or "no: wrong column"
```

`VISUALIZATION` returns a Plotly figure (`result.output.figure`).

## How it proves its work

- **Structured output only.** The model returns a validated object with a `python_code`
  field; code is never scraped from free text.
- **Docker sandbox.** No network, read-only filesystem, no writable host folder, non-root,
  no capabilities, memory/CPU/process limits, time limit. 19 escape attempts are tested in CI.
- **Checks decide the status.** Generic checks (output not empty, rows or columns lost
  without being asked, missing values added, numbers turned into text) plus agent-specific
  ones. A failed blocking check triggers a fix attempt; warnings give `needs_review`.
- **Impossible requests stop early.** If the data cannot answer the request, the run ends
  `failed` after one call instead of inventing columns.
- **Every run is recorded** in DuckDB with tokens, cost, time, checks and code. A budget
  guard stops evals before the API plan's limits.

## Evals

102 tasks on five OpenML datasets (adult, credit-g, telco-customer-churn, ames-housing,
diabetes), each with a known answer produced by seeded fault injection. Checkers compare
outputs by value; an oracle check proves every task is solvable and not trivial.

```bash
uv sync --group evals
uv run python -m dataquorum.evals.runner --label my-run          # needs Docker + API key
uv run python -m dataquorum.evals.compare runs/evals/<A> runs/evals/<B>
```

Results so far: [week 6 baseline](docs/results/week6-baseline.md),
[week 7 parity](docs/results/week7-parity.md), [week 8 hard tasks](docs/results/week8-hard-tasks.md),
[week 9 new agents](docs/results/week9-new-agents.md),
[reasoning-effort experiment](research/reasoning_effort/README.md).

## Project layout

```
src/dataquorum/
  core/      llm (structured output, cost, budget), engine (LangGraph), state, checks, review
  agents/    cleaning, wrangling, visualization, loader, sql, eda, features (AgentSpecs)
  sandbox/   Docker executor
  tracing/   DuckDB run recorder
  evals/     datasets, faults, tasks (YAML), checkers, oracle, runner, stats, parity
images/      sandbox, original (parity), autogluon
docs/        spec, bug log of the original, results
```

## Development

```bash
uv run pytest            # Docker tests skip unless the sandbox image is built
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

CI runs lint, strict type checks and tests on Ubuntu and Windows, against pandas 2 and 3,
plus the sandbox escape tests in Docker.

## Credits

Inspired by [business-science/ai-data-science-team](https://github.com/business-science/ai-data-science-team)
by Matt Dancho (MIT). DataQuorum is an independent reimplementation and shares no code with
it; the parity eval runs the original unmodified in its own container.

## License

MIT
