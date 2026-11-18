# Week 11 ML correctness results

On November 18, 2026, the correctness agent produced the expected audit and final status
on **12 of 12** seeded cases. It marked all seven planted risks `needs_review` and all five
controls `success`. There were zero false successes. The Wilson 95% interval for accuracy
is 75.8%–100%. Twelve examples leave considerable uncertainty about performance on new data.

| Cases | Expected decision | Correct |
|---|---|---:|
| Balanced classification and rare-class classification | Stratified folds with accuracy or balanced accuracy | 2/2 |
| Group, chronological, and combined group/time controls | Respect the split structure | 3/3 |
| High-cardinality ID, categorical target copy, near-perfect regression proxy, post-outcome field | Name the suspect column and reason | 4/4 |
| Repeated records, shared groups, reversed time across partitions | Name the contaminated split or column | 3/3 |

DeepSeek V4.1 Flash ran at low reasoning effort through OpenCode Go, with one worker and
two retries available per task. The 25 calls used **$0.01337** of allowance. Wall time was
315.3 seconds and median task time was 25.6 seconds. The run ID was
`20261118T062226Z-a58c07`. The preceding 102 tasks retain their previously reported live results.

## What the retry caught

In the categorical-proxy task, the feature `code` encoded the class label under a different
name. On its first attempt the agent designated `code` as a group and recommended GroupKFold.
That designation removed the feature from the proxy screen. The host recomputed the audit
using the roles in the original request and rejected the result. The second attempt reported
`code` as a target proxy and returned `needs_review`.

The first live scoring pass used the suite's older rule that accepts either successful
completion or review for a solvable task. The week 11 answer keys now specify the exact
status for each case. Regrading all twelve saved tables and statuses gave the same twelve
correct results without additional model calls.

## How to reproduce

Run these commands in PowerShell from the repository root. The dry run checks the audit
against hand-labelled rows. The live run uses the configured Go allowance.

```powershell
docker build -t dataquorum/sandbox:0.3 -f images/sandbox/Dockerfile .
uv run python -m dataquorum.evals.runner --tag week11 --dry-run
uv run python -m dataquorum.evals.runner --tag week11 --workers 1 --label week11-correctness
uv run pytest -q tests/test_correctness.py tests/sandbox/test_model_reports.py
```

## Interpretation and limits

On Windows, 320 non-Docker tests and 39 Docker tests passed. Ruff formatting, lint and
strict mypy passed. All 114 tasks passed the offline reference-answer check.

The twelve cases exercise the agent and its validator. The model-report tests separately
exercise Optuna tuning, SHAP feature rankings, and calibration with held-out observations.
The SHAP test plants one informative feature and one noise feature. The calibration unit
test checks the bin counts and both scores against a hand calculation. Model fitting uses
the training partition and refuses flagged risks. The model-report API currently supports
a small random forest with finite numeric predictors and binary outcomes for calibration.

Proxy scores and column names are screening rules. A genuinely predictive feature may
trigger review, and a feature collected after the outcome can escape detection if its
name and values do not reveal it. The analyst still needs to establish when each feature
would be available at prediction time. The live tasks use one synthetic fixture, one model
and one run per case.
