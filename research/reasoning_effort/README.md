# Reasoning effort vs cost, speed and correctness (week 2)

**Question.** DeepSeek V4.1 Flash spends most of its output on reasoning tokens. Which
`reasoning_effort` should DataQuorum use by default?

**Answer.** `low`. On this task every level produced correct code (40/40), and `low` was the
cheapest and fastest: about 40% less cost and time than the provider default, and less than
half of `medium` and `high`. Agents can still raise effort per step.

## Setup

- Date: 2026-11-16, 06:17–06:22 UTC (Go **peak** hours, so list cost was 2×).
- Model: `deepseek-v4.1-flash` via OpenCode Go, through `dataquorum.core.llm.LLMClient`
  (structured output, schema `CleaningPlan{steps, python_code}`), fallback model disabled.
- Levels: provider default (parameter omitted), `minimal`, `low`, `medium`, `high`.
  8 runs each, interleaved so all levels share the same time window. 4 parallel workers.
- Task: write `clean(df)` for a 12-row frame with tricky values
  (`"$3.4k"`, `" $2,050 "`, `"1,750.50"`, `"n/a"`, `""`, `None`, `"san  francisco"`).
  Exact prompt in `run.py`.
- Check: the generated code ran in the Docker sandbox (no network, read-only, non-root,
  512 MB, 60 s) and its output was compared with the known answer (`harness.py`): 12 rows
  kept, `income` float and exactly right, `city` stripped/collapsed/title case, `age`
  untouched. The harness was itself checked with a correct solution (pass), a wrong one
  (fail) and code that tries to use the network (blocked).

## Results

| Effort | Pass | 95% CI (Wilson) | Median out tokens | Median reasoning | Median s | Mean allowance $ |
|---|---|---|---|---|---|---|
| default | 8/8 | 68–100% | 2,580 | 2,080 | 24.3 | 0.00181 |
| minimal | 8/8 | 68–100% | 1,489 | 1,024 | 16.0 | 0.00111 |
| **low** | 8/8 | 68–100% | **1,443** | **847** | **14.0** | **0.00104** |
| medium | 8/8 | 68–100% | 3,430 | 3,009 | 33.7 | 0.00243 |
| high | 8/8 | 68–100% | 2,882 | 2,419 | 28.6 | 0.00244 |

Spread (min–max output tokens): default 1,502–3,178 · minimal 1,080–2,153 · low 739–2,181 ·
medium 2,129–5,420 · high 1,656–7,805. No schema failures and no repair attempts in 40 calls.
Total spend: $0.071 allowance ($0.118 list at peak prices).

Raw rows, including every generated function, are in `results.jsonl`.

## Caveats

- **Ceiling effect.** All 40 runs passed, so this task cannot show whether more reasoning
  helps on harder work. It only shows that `low` is enough for a well-specified cleaning
  step. The week-6 eval suite (about 50 tasks) is the real test; week 14 repeats this
  comparison there, with McNemar tests on paired results.
- n = 8 per level: the pass-rate intervals are wide (68–100%) and overlap completely.
- `minimal` and `low` are close; `low` had the lower median cost and latency here, but the
  difference is within run-to-run noise.
- One task, one model, one time window (peak).

## Reproduce

```bash
uv run python research/reasoning_effort/run.py --runs 8          # appends to results.jsonl
uv run python research/reasoning_effort/run.py --analyse-only
```

Needs Docker running, the `dataquorum/autogluon:1.6.3` image (any image with pandas works;
change `IMAGE` in `run.py`), and `OPENCODE_GO_API_KEY` in `.env`.
