# Changelog

## 1.0.0

### Studio

- Local Streamlit Studio with an offline retail workflow, versioned JSON recipes, and
  runnable Python export. The workflow diagram traces a selected step's inputs and
  switches to a readable list on smaller screens.
- Inspect step checks, preview source and verified tables, and export Plotly charts.
  Studio also includes model audits, sandbox reports, and local DuckDB run history.
- The captioned walkthrough shows the demo, its verification checks, and recipe export.
  [Studio walkthrough](docs/media/studio-demo.mp4)

### Supervisor and evaluation

- Resumable paired study for safeguard ablations and five OpenCode Go model configurations.
  Reports matched-task accuracy, false successes, metered allowance, timing and failure evidence.
  Offline controls are complete. The paired live study is pending.
- `--no-checks` disables EDA grounding and preserves the explanation call.

- LangGraph supervisor for multi-step requests with verbatim user instructions, exact
  routing, refusal handling, input compatibility checks and a bounded execution loop.
- Offline MiniLM and TF-IDF routing experiment with 225 original training requests and 72
  held-out requests. A follow-up adds 144 training examples and a fresh 72-request holdout.

### Model tooling and agents

- Sandbox 0.4 exports audited random forests with their feature schema and cross-validation
  metadata. The host verifies prediction parity before saving the artifact.
- Local MLflow registry and versioned FastAPI batch prediction, available via the `serving` extra.
- Evidently compares numeric feature batches and produces HTML reports and JSON metrics
  from its own locked environment.

- ML correctness agent with host-verified leakage checks, group and time validation,
  imbalance-aware metric choice and 12 new known-answer evaluations.
- Sandbox 0.3 adds seeded Optuna tuning, tree SHAP explanations and held-out binary
  calibration reports. Build from the repository root using the updated README command.

- AutoML behind one interface (`dataquorum.ml.automl.fit_predict`): FLAML, H2O and
  AutoGluon, each in a locked-down container, scored on the host from test predictions.
- AutoML benchmark on 8 OpenML tasks (`python -m dataquorum.ml.automl.benchmark`) with a
  Friedman test, average ranks and Nemenyi critical difference (`evals.stats.friedman`).
- Image `dataquorum/automl:0.1` (FLAML 2.8.0, H2O 3.46.0.12, Java 21).

- Loader agent (a folder of CSV/TSV/Excel/JSON/Parquet files, mounted read-only), SQL agent
  (sqlglot guard plus read-only SQLite/DuckDB connections), EDA agent (findings whose numbers
  must appear in the computed table) and feature-engineering agent (scikit-learn pipelines;
  rows kept, target unchanged, no target leakage).
- Engine inputs can be a DataFrame, a folder or a database file; code can be Python or SQL.
- Sandbox image 0.2 adds openpyxl and scikit-learn.
- 31 new eval tasks (102 total), checkers for statistics and features, and
  `dataquorum.evals.rescore` to re-grade saved outputs after a checker fix.

### Upgrade notes

- Python 3.12 is required. Use the `studio` extra for the browser app, `serving` for the model
  registry and API, and `routing` for the offline MiniLM experiment.
- Rebuild the live-agent and model-tool sandbox as `dataquorum/sandbox:0.4` from the repository
  root. The bundled Studio demo runs locally with the Studio extra.

## 0.1.0 (2026-11-16): first usable release

### Agents
- Cleaning, wrangling and visualization agents, defined as `AgentSpec`s and run by one
  shared engine. `dataquorum.agents.make_engine(spec)` wires the LLM, budget guard, sandbox
  and run recorder.

### Engine
- LangGraph flow: summarize → recommend → optional human review → generate → execute →
  validate → fix (retries) → finalize.
- Human review with `interrupt()` and a checkpointer; replies parsed into Approve / Revise /
  Reject (so "Yes." approves).
- Impossible requests end `failed` at the recommend step (`blocker`), before any code.
- `AgentResult` status is computed from checks; inconsistent results cannot be built.

### LLM layer
- Structured output only (strict JSON schema, JSON-object fallback, one repair).
- Cost meter using OpenCode Go's own allowance figure; budget guard ($18 / 5 h, $45 / week).
- Default `reasoning_effort="low"`, chosen by a measured experiment.

### Sandbox
- Docker SDK executor: no network, read-only root, no writable host mount (output read from a
  size-limited tmpfs), non-root, no capabilities, resource limits, timeout, capped logs.
- 19 escape probes plus resource and cleanup tests, run in CI.

### Tracing
- DuckDB run recorder (`runs` and `llm_spend` tables); LLM and sandbox time recorded
  separately.

### Evals
- 71 tasks on five OpenML datasets with seeded faults and known answers; value-based
  checkers; oracle check; outcomes with false-success rate as the headline metric.
- Statistics: Wilson intervals, bootstrap, pass@k, exact McNemar (verified against SciPy and
  statsmodels). Ablation flags and a run comparison tool.
- Parity eval that runs the original project in its own container on the same tasks.
