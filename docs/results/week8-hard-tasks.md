# Hard tasks and the first "false success" (week 8, 2026-11-16)

The week-6 suite scored 100%, so week 8 added 16 harder tasks (`evals/tasks/hard.yaml`):
mixed formats in one column, four faults in one table, near-duplicate rows, pivots, top-N per
group, rates, age bands, filter-then-round, grouped and sorted charts, a line chart, and two
impossible requests with tempting proxies (income in dollars, when the data only has an
income bracket).

## First run (`20261116T090505Z-0f2ca0`)

15/16 correct. One **false success**: `hard-clean-ames-four-faults` returned `success` with
one wrong Neighborhood label (`northpark_villa` instead of `Northpark_Villa`).

## Root cause: the task, not the agent

`Northpark_Villa` appeared once in the 300-row sample, and the fault injector turned that one
row into `northpark_villa`. The right spelling was **not in the input at all**, so "rewrite
each label to the spelling used by most rows" had exactly one defensible answer: the one the
agent gave. Two more labels were ambiguous (Briardale: variant 3 vs original 2; credit-g
"other": a 1-1 tie), and two week-6 baseline tasks had ambiguous labels that the agent got
right by luck.

Fixes:
- `category_variants` and `near_duplicates` now keep each label's original spelling a strict
  majority (labels seen once are never varied). Random draws are unchanged, so only the 5
  affected tasks got new inputs.
- The oracle gained `label_problems`: a task fails the oracle if any expected label is not
  the majority spelling in its input.
- Tests for both.

## After the fix (`20261116T091028Z-208868`, the 5 changed tasks)

All 5 correct. Combined with the unchanged tasks: **hard suite 16/16 correct, 0 false
successes** (95% CI 79–100%), including both tempting impossible requests refused after one
call.

## Other findings

- **Cleaning is the slow, expensive agent.** Median LLM time 48–64 s and 4–6k output tokens
  per cleaning task, against about 10 s and 500–600 tokens for wrangling and charts. Sandbox
  time is ~1.5 s everywhere (now recorded separately as `execution_seconds`). The cleaning
  agent's default checklist seems to make the model reason far more even for specific
  requests. Candidate for the week-14 ablations.
- **Still near the ceiling.** 71/71 across both suites after the fix. Agents added in week 9
  (loader, SQL, EDA, features) and their tasks are the next chance to find real failures.
- **Lesson:** a benchmark can produce false "false successes". Every generated answer key
  needs a check that it can actually be inferred from the input.
