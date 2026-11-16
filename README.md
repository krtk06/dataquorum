# DataQuorum

**A team of AI data-science agents that must prove their work.**

> Status: early development (week 7 of 17). Cleaning, wrangling and visualization agents work;
> more agents, AutoML and the Studio are still to come.

DataQuorum is a Python library of LLM agents for data-science work (cleaning, wrangling,
visualization, SQL, EDA, feature engineering, AutoML) plus a Streamlit pipeline studio.
Every agent runs its code in a locked-down Docker sandbox, validates its own output, and
reports `success`, `failed` or `needs_review`. It never claims success on an empty result.

## Early results

55 eval tasks on five OpenML datasets, each with a known correct answer, run against the
original project on the same model, effort and retry budget
([details](docs/results/week7-parity.md)):

| | Original | DataQuorum |
|---|---|---|
| Correct | 47/55 (85%) | 55/55 (100%) |
| Reported success on a wrong or invented answer | 7/55 (13%) | 0/55 |
| Impossible requests refused | 0/3 | 3/3 |
| Cost per task | ~$0.0037 | $0.0008 |

McNemar exact test on the paired tasks: p = 0.008. The suite was written alongside DataQuorum,
so treat this as an early, home-advantage result.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

Copy `.env.example` to `.env` and add your API key. Keys are never stored in code.

## Credits

Inspired by [business-science/ai-data-science-team](https://github.com/business-science/ai-data-science-team)
(MIT). DataQuorum is an independent reimplementation and shares no code with it.

## License

MIT
