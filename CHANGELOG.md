# Changelog

## Unreleased (week 9)

- Loader agent (a folder of CSV/TSV/Excel/JSON/Parquet files, mounted read-only), SQL agent
  (sqlglot guard plus read-only SQLite/DuckDB connections), EDA agent (findings whose numbers
  must appear in the computed table) and feature-engineering agent (scikit-learn pipelines;
  rows kept, target unchanged, no target leakage).
- Engine inputs can be a DataFrame, a folder or a database file; code can be Python or SQL.
- Sandbox image 0.2 adds openpyxl and scikit-learn.
- 31 new eval tasks (102 total), checkers for statistics and features, and
  `dataquorum.evals.rescore` to re-grade saved outputs after a checker fix.

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
