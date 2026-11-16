# Parity: DataQuorum vs the original project (week 7, 2026-11-16)

Same 55 tasks, same built inputs and request text, same model (`deepseek-v4.1-flash` via
OpenCode Go), same `reasoning_effort=low`, same retry budget (2), same checkers.

- **Original:** business-science/ai-data-science-team at `4ffeb7f`, run by
  `python -m dataquorum.evals.parity.run` (run `20261116T083333Z-f56c01`), each task in its
  own `dataquorum/original` container. It has no status field, so "an output and no error"
  counts as `success` (that is how it reports work as done). Its code-explanation step was
  bypassed (it does not change the output).
- **DataQuorum:** week-6 baseline run `20261116T080743Z-728da3`.

## Result

| | Original | DataQuorum |
|---|---|---|
| Correct | 47/55 (85%, 95% CI 74–92%) | **55/55 (100%, 95% CI 93–100%)** |
| **False success** | **7/55 (13%, 95% CI 6–24%)** | **0/55 (0%, 95% CI 0–7%)** |
| Impossible requests refused | 0/3 | 3/3 |
| LLM calls | 120 | 112 |
| Output tokens (reasoning) | 254,194 (182,482) | 47,928 (32,704) |
| Allowance $ per task | 0.0037 (estimated) | 0.00084 |
| Seconds per task (mean / median) | 59.7 / 44.5 | 19.0 / 12.7 |

- **McNemar exact test** on the paired tasks: 8 tasks correct only for DataQuorum, 0 only
  for the original, **p = 0.0078**.
- Paired bootstrap, DataQuorum minus original: cost −$0.0029 per task (95% CI −0.0037 to
  −0.0021); time −40.6 s per task (95% CI −58.7 to −25.6).

## What the original got wrong

| Task | What happened | Reported as |
|---|---|---|
| wr-impossible-adult-income-by-state | Invented columns `state`, `average_yearly_income`; returned **0 rows** | success |
| wr-impossible-telco-revenue-by-region | Relabelled `Contract` as `product_line`, put every row in an invented region "All Regions", summed MonthlyCharges as "revenue" | success |
| viz-impossible-telco-revenue-by-region | Charted MonthlyCharges by Contract instead, retitled it | success |
| clean-diabetes-zeros | Asked to turn 0 into missing; **filled them with the column mean** instead (103 values) | success |
| clean-telco-tenure-outliers | Asked to set outliers to missing; filled them with the mean | success |
| clean-telco-total-charges | Asked to leave blanks missing; filled them with the mean | success |
| clean-credit-combined | Removed 11 rows it judged to be outliers, which nobody asked for | success |
| clean-adult-combined | Its sandbox failed on its own debug `print` ("Sandbox returned non-JSON output") | failed (honest) |

The pattern: the original's built-in cleaning defaults (impute, drop outliers) override explicit
instructions, and nothing checks the result against the request, so the run still reports
done. Impossible requests get an invented answer every time. These are bug log #6 and #13-style
behaviours, now measured.

## Caveats

- **Home advantage.** The tasks and checkers were written alongside DataQuorum. They are
  explicit and checked by value, but a suite written by the original's authors could favour
  different behaviour. Week 9 adds harder tasks; an independent task set would be stronger.
- **Cost for the original is estimated** from token counts and our price table (its LangChain
  client does not expose Go's allowance figure). Both runs were in Go peak hours.
- **Ceiling on our side:** DataQuorum scored 100%, so this suite cannot show where it fails.
- One run each; repeats (`--repeats`) would give run-to-run variance and pass@k.
