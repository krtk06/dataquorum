# DataQuorum

**A team of AI data-science agents that must prove their work.**

> Status: early development (week 1 of 17). Nothing usable yet.

DataQuorum is a Python library of LLM agents for data-science work (cleaning, wrangling,
visualization, SQL, EDA, feature engineering, AutoML) plus a Streamlit pipeline studio.
Every agent runs its code in a locked-down Docker sandbox, validates its own output, and
reports `success`, `failed` or `needs_review`. It never claims success on an empty result.

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
