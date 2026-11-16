# Week 10: AutoML benchmark, FLAML vs H2O vs AutoGluon (2026-11-16)

Run `runs/automl_bench/week10` (local, git-ignored). 8 OpenML tasks × 3 seeds × 3 backends =
72 runs, 60 s budget, each run alone in a container with 2 CPUs and 3 GB, no network.
Seeded 75/25 splits (stratified for classification), identical for every backend. Metrics
computed on the host from test predictions. No LLM calls (cost $0).

Reproduce: `uv run python -m dataquorum.ml.automl.benchmark --budget 60 --seeds 0 1 2`
(needs `dataquorum/automl:0.1` and `dataquorum/autogluon:1.6.3`, see `images/`).

## Result

| Task (OpenML id) | Metric | AutoGluon 1.6.3 | FLAML 2.8.0 | H2O 3.46.0.12 |
|---|---|---|---|---|
| credit-g (31) | ROC AUC ↑ | **0.7724** | 0.7501 | 0.7614 |
| diabetes (37) | ROC AUC ↑ | **0.8180** | 0.8160 | 0.8076 |
| blood-transfusion (1464) | ROC AUC ↑ | **0.7466** | 0.7102 | 0.7237 |
| phoneme (1489) | ROC AUC ↑ | 0.9605 | 0.9614 | **0.9620** |
| vehicle (54) | accuracy ↑ | 0.7741 | 0.7725 | **0.8009** |
| segment (36) | accuracy ↑ | 0.9797 | **0.9849** | 0.9814 |
| kin8nm (189) | RMSE ↓ | **0.0688** | 0.1076 | 0.1025 |
| cpu_act (197) | RMSE ↓ | **2.1341** | 2.2819 | 2.1934 |

Mean over 3 seeds. All 72 runs succeeded (after the fix below).

- **Average rank (1 = best): AutoGluon 1.62, H2O 1.88, FLAML 2.50.**
- **Friedman χ² = 3.25, p = 0.197.** Not significant. Nemenyi critical difference
  (α = 0.05) = 1.17, larger than the largest rank gap (0.88), so no pair differs
  significantly. With 8 tasks the test can only detect large, consistent gaps; this is the
  honest reading, not "AutoGluon wins".
- The clearest single gap is kin8nm (RMSE 0.069 vs ~0.10, seed spread ≤ 0.008): AutoGluon's
  neural nets fit this smooth simulated robot-arm function much better than tree models.
- Seed-to-seed spread is often as large as the gaps between backends (e.g. credit-g AUC std
  0.05–0.06), which is why per-task winners above should not be over-read.

## Cost of each backend

| | AutoGluon | FLAML | H2O |
|---|---|---|---|
| Median fit / wall seconds | **18 / 21** | 61 / 63 | 67 / 69 |
| Median models tried | 12 | 13 | 27 |
| Usual best model | WeightedEnsemble (24/24) | LightGBM (12/24), XGBoost (8/24) | StackedEnsemble (14/24) |

AutoGluon's `medium_quality` preset trains a fixed portfolio and stops, using about a third
of the budget and still ranking first. FLAML uses the whole budget for a search. H2O overruns
by ~7 s because `max_runtime_secs` excludes JVM start-up and prediction. Peak container
memory in the smoke test was ~1 GB for every backend.

## Fairness notes and caveats

- **AutoGluon has no fit-level seed** in 1.6: its models always use seed 0, so for AutoGluon
  the seed only changes the split. Its seed spread is therefore partly understated.
- Equal budget means an equal *cap*, not equal compute used (see AutoGluon above).
- Small, classic OpenML tasks (748–8,192 rows); results may not carry over to large data.
- One machine (Windows 11, WSL2, 2 CPUs per container); timings are indicative.

## Bug found during the run (ours, fixed)

First pass: 66/72 succeeded; all 6 failures were H2O on blood-transfusion and phoneme. Their
class labels are `"1"`/`"2"`, and H2O names probability columns for number-like levels
`p1`/`p2`, but our runner looked up a column named `2`. Training had worked; only reading the
probability failed. The runner now finds the column by the factor-level position. The 6 runs
were re-run with the fix (first pass kept as `results.first-pass.jsonl`); no other run was
affected (no binary tasks remained when the fix was made).
