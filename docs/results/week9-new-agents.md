# Week 9: loader, SQL, EDA and feature-engineering agents (2026-11-16)

Run `20261116T100158Z-2610bb` (label `week9-v1`): 31 new tasks, DeepSeek V4.1 Flash,
`reasoning_effort=low`, retries 2, sandbox image 0.2.

## Result (after fixing a checker; same saved outputs, no new LLM calls)

| Agent | n | Correct | False success | Allowance $ | Median s |
|---|---|---|---|---|---|
| **overall** | 31 | **31 (100%, 95% CI 89–100%)** | **0** | 0.0112 | 12.1 |
| data_loader | 7 | 7 | 0 | 0.0012 | 11.3 |
| sql_database | 9 | 9 | 0 | 0.0011 | 7.5 |
| eda | 7 | 7 | 0 | 0.0041 | 24.0 |
| feature_engineering | 8 | 8 | 0 | 0.0049 | 17.3 |

All four impossible requests ended `failed` before any code ran: a file that does not exist,
"delete everyone older than 60" (SQL), a table that does not exist, and a target column that
does not exist. The feature agent passed the target-leakage trap ("one-hot encode every text
column" when the target is a text column) in every run.

## The first score was wrong, again because of a checker

The raw score was 26/31 with **5 "false successes", all in feature engineering**. Every one
was a grading bug: scikit-learn's `ColumnTransformer` names its outputs `num__age` and
`cat__sex_Female`, and the checkers looked for `age` and `sex_Female`. The agent's features
were correct.

Fixes:
- Checkers accept the `prefix__` naming (`find_column`, `base_name`).
- The agent's own leakage check had the same blind spot (`cat__class_good` would have passed);
  fixed and tested.
- A new independent checker, `no_target_leakage`, is now on every feature task with a target.
- `python -m dataquorum.evals.rescore <run>` re-applies the current checkers to a run's
  saved outputs, so checker fixes are verified on identical outputs without new LLM calls.

## Lessons

- Two weeks in a row, the only "false successes" for DataQuorum were grading bugs (week 8:
  an unknowable answer key; week 9: column naming). Every checker change now gets a rescore
  of the run that exposed it, and checkers get unit tests with realistic library output.
- SQL is the cheapest agent (median 7.5 s, $0.00012 per task) because the query is short;
  EDA is the slowest of the new ones because of the extra findings call.
