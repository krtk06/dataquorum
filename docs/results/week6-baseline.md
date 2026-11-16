# Eval baseline, week 6 (2026-11-16)

Run `20261116T080743Z-728da3` (label `baseline-week6`), git `71fe36f`.
Model `deepseek-v4.1-flash` via OpenCode Go, `reasoning_effort=low`, no fallback model,
`max_retries=2`, Docker sandbox `dataquorum/sandbox:0.1`, 3 workers. Go peak hours.

## Result

| Agent | n | Correct | False success | Allowance $ | Median s |
|---|---|---|---|---|---|
| **overall** | 55 | **55 (100%)** | **0 (0%)** | 0.0465 | 12.7 |
| data_cleaning | 25 | 25 | 0 | 0.0309 | 17.2 |
| data_wrangling | 20 | 20 | 0 | 0.0094 | 8.9 |
| data_visualization | 10 | 10 | 0 | 0.0061 | 12.7 |

- 95% Wilson interval for accuracy: **93.5%–100%**, so the false-success rate on this suite
  is at most about 6.5% at 95% confidence.
- All 3 impossible requests were refused after one LLM call (`failed`, "cannot do this with
  the given data").
- 47,928 output tokens, 68% of them reasoning; 5 retries in 55 runs; mean ~2.1 LLM calls.
- Whole run: about $0.05 against the Go allowance.

## Findings

1. **Five correct runs were flagged `needs_review` by our own warning checks.** They were
   still "correct" (the answer was right), but each one would have asked a human to look:
   - `missing_values_increased` fired when the request explicitly asked to set values to
     missing (outliers, "0 means not measured"): 4 tasks.
   - `duplicate_rows_not_increased` fired after mode imputation made small rows identical
     without adding rows: 2 tasks (one overlaps the above).

   Both checks were fixed (requested blanking is not a warning; duplicates only warn when
   rows were added). Re-run `20261116T081509Z-9f3f7e`: all 5 now `success` and still correct.
2. **Ceiling effect.** 100% means this suite cannot tell configurations apart (as with the
   reasoning-effort experiment). It shows the pipeline is sound, but week 9 tasks must be
   harder: ambiguous requests, multi-step wrangling, wider frames, mixed faults per column,
   and requests where the "obvious" fix is wrong.
3. **One slow outlier:** `clean-credit-duration-placeholders-median` took 199 s for 1,162
   output tokens, so the time was not reasoning. Likely API queueing at peak; the runner
   records only total time, so LLM and sandbox time should be split in a later change.

Raw results: `runs/evals/20261116T080743Z-728da3/` (local) and DuckDB `runs` rows with
`eval_task_id`.
