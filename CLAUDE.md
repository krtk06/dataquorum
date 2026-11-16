# DataQuorum — project context

> Written 2026-11-15 from a planning discussion (held in `Projects\Descriptions`, which stays discussion-only).
> Code for DataQuorum lives **here**. Keep this file updated as decisions change.

## What we're building
**DataQuorum** is a from-scratch rebuild of [business-science/ai-data-science-team](https://github.com/business-science/ai-data-science-team) (MIT, ~5.5k★, beta v0.0.0.9017, last commit 2026-01-28, effectively one maintainer). It is a Python library of LLM agents for data-science work, plus a Streamlit "pipeline studio" app.

- **Match the original's features and behaviour, not its code.** Use our own name, branding and implementation.
- **Credit the original in the README** ("Inspired by …"). If any code is ever copied, keep their MIT notice.
- **Goal:** a 2026 portfolio project with depth in **ML and Data Science**. It also covers AI and SDE.
- Tagline ideas: *"A team of AI data-science agents that must prove their work."*

## Problems in the original we fix by design
- **No tests, no CI.** Unpinned dependencies, which broke under pandas 3.
- **Huge files:** the supervisor is ~4,900 lines and the app's `app.py` is ~15,500 lines. **Our rule: no file over ~400 lines.**
- **The sandbox is an import blocklist**, and its resource limits do nothing on Windows. We use Docker.
- **False success:** it reports "Workflow complete" on empty results.
- **Code scraped from `<think>` text.** We use structured output instead.
- **Routing bugs:** the router merges appended instructions into the question and discards refusals.

## Constraints
- **Windows 11**, 16 GB RAM, **no GPU**. No local models: Ollama was dropped on 2026-11-16 (Go calls cost ~$0.001–0.003 each, CI replays recorded calls, and 3–4B CPU models are too weak at structured output to be useful tests).
- **~15 hrs/week.** The user is new to LangGraph and agents but learns fast.
- **Main LLM:** **DeepSeek V4.1 Flash** (`deepseek-v4.1-flash`) through the user's **OpenCode Go** subscription.
  - OpenAI-compatible endpoint: `https://opencode.ai/inference/go/openai/v1/chat/completions` (check against the docs)
  - $100/month limit for this model. Rule of thumb: about 20% per 5 hours (~$20) and 50% per week (~$50).
  - Off-Peak $0.15 in / $0.60 out / $0.003 cached-in per 1M tokens. Peak costs 2×. Peak = 01:00–04:00 and 06:00–10:00 UTC, Mon–Fri.
  - Estimated project spend is about $20–35 in the busiest month, so it fits.
  - Send our own User-Agent and an `x-opencode-session` header. Go asks for "typical coding agent traffic", so space out big eval runs.
  - Keep Go's default privacy settings, which block models that train on prompts. DeepSeek's zero-retention agreement was valid through 2026-11-30 and is renewed monthly, so recheck it.
  - Fallback model in config: GLM-5.3-Flash. Lock model IDs per eval run.
- **API keys only in environment variables or `.env`** (git-ignored). Never in code.

## Stack
Python 3.12 · **uv** + `pyproject.toml` + lock file · ruff · mypy · pytest · recorded LLM calls replayed in CI via our own `tests/replay.py` (the `openai` SDK now uses `httpx2`, which vcrpy/`pytest-recording` cannot intercept; dropped 2026-11-16) · GitHub Actions
- **Agents:** `langgraph`, `langchain-openai` (base_url pointed at Go), `pydantic`
- **Data:** `pandas` (pin 2.x; CI also tests pandas 3), `duckdb`, `sqlalchemy`, `sqlglot` (enforces read-only SQL), `plotly`
- **Sandbox:** Docker via the `docker` Python SDK, running under WSL2
- **ML:** scikit-learn, xgboost, `h2o` (needs Java 17+), `flaml`, `autogluon.tabular` (run in WSL2/Docker), `optuna`, `shap`, `mlflow`, `fastapi`, `evidently`
- **Stats:** numpy, scipy, statsmodels
- **Datasets:** `openml` (no Titanic or Iris)
- **Learned router:** scikit-learn + `sentence-transformers` (small CPU model)
- **UI:** `streamlit`, `streamlit-flow-component`
- Check the latest versions in week 1, then lock them.

## Architecture
```
core/        llm.py (provider + cost meter + budget guard) · state.py · engine.py · validation.py · statuses.py
sandbox/     docker_executor.py · image/
agents/      cleaning · wrangling · visualization · loader · sql · eda · features
ml/          automl/ (h2o, flaml, autogluon behind one interface) · correctness · tuning · explain · registry · serving · drift
supervisor/  router_llm · router_learned · graph
evals/       tasks/ (YAML) · faults.py · checkers.py · runner.py · stats.py · parity/
research/    ablations · model_comparison · notebooks/
tracing/     recorder.py → DuckDB (one row per run)
app/         pages/ · components/ · pipeline_model.py
tests/
```
- **Engine (LangGraph):** `summarize → recommend → [human_review interrupt] → generate_code (structured output) → execute (sandbox) → validate → (fix → execute)* → finalize`. Each agent only supplies its prompts and validation rules.
- **Every agent returns the same shape:** `status` (success / failed / needs_review, **never a false success**), output (data / figure / model), `code`, `steps`, `validation`, `usage` (tokens, cost, retries, time).
- **Sandbox:** no network, read-only filesystem plus a small writable `/tmp`, memory, CPU and process limits, non-root user, all capabilities dropped, no-new-privileges, timeout. Data goes in and out as Parquet through a temp mount.
- **Budget guard:** pause evals at $18 per 5 hours or $45 per week. Schedule big runs off-peak. Put fixed prompt text first so it hits the cache.

## Roadmap (17 weeks, plus 1 buffer week)
| Wk | Work |
|---|---|
| 1 | Setup (uv, CI, WSL2 + Docker, Java, AutoGluon Docker image) · Go smoke test + **measure real tokens per call** · install the original in a **separate venv**, point it at Go/DeepSeek, log its bugs · write the spec · LLM basics + structured output |
| 2 | LangGraph learning · throwaway mini agent (write code → run → retry) · `core/llm.py`, `core/state.py` |
| 3 | Core engine + validation framework + human-review interrupt + replay tests |
| 4 | Cleaning, Wrangling, Visualization agents + run recorder |
| 5 | Docker sandbox + escape tests |
| 6 | Eval suite: OpenML datasets + fault injector + YAML tasks + checkers (~50 tasks) + stats learning |
| 7 | `stats.py` (Wilson/bootstrap CIs, pass@k, McNemar) · runner flags for ablations · parity adapter running the original in its venv |
| 8 | Fix top failures · README v1 · **v0.1.0 MVP** |
| 9 (+9b buffer) | Loader, SQL (sqlglot read-only), EDA, Feature Engineering (sklearn Pipelines) · +30 tasks |
| 10 | AutoML (H2O / FLAML / AutoGluon, one interface, run in Docker/WSL2) + benchmark (8 OpenML tasks, equal budgets, 3 seeds, Friedman test) |
| 11 | ML correctness agent (leakage, validation chooser, metric chooser) · Optuna · SHAP · calibration |
| 12 | MLflow registry · FastAPI serving · Evidently drift |
| 13 | Supervisor (LLM router, keeps instructions, respects refusals) + learned-router experiment (~300 labelled requests) |
| 14 | DS research: ablations · cost-vs-accuracy across 5–6 Go models · failure taxonomy |
| 15–16 | Studio app (Pipeline, Data, Charts, Model, Runs dashboard; script export; save/load) |
| 17 | README v2 · demo video · PyPI v1.0.0 · two write-ups · stretch: MCP server |

**Cut for time:** time-series agent, separate dashboard phase (now the Studio's Runs page), MCP server (stretch only).

## Open items
- **PyPI placeholder:** `dist/dataquorum-0.0.1` is built (`uv build`); the user uploads it with `uv publish` and their own PyPI token. Name was still free on 2026-11-16.
- **Go docs (checked 2026-11-16):** endpoint and model IDs (`deepseek-v4.1-flash`, `glm-5.3-flash`) match. Go docs say DeepSeek V4.1 Flash is **restricted by default** and needs Settings > Privacy > Regions > **Global** (default is Europe & US). **Decided 2026-11-16: enable Global** so DeepSeek V4.1 Flash is usable; keep the other default privacy settings (the block on models that train on prompts stays on). Recheck DeepSeek's zero-retention terms monthly. Auth header is not documented; assumed `Authorization: Bearer` (OpenAI-compatible).
- **Go smoke test results (2026-11-16, off-peak, one run each, so treat as rough):**
  - `Authorization: Bearer` works; `json_schema` structured output with `strict: true` is accepted by both models and returned schema-valid JSON.
  - **DeepSeek V4.1 Flash emits reasoning tokens**, billed as output, and returns the text in `reasoning_content`. On the cleaning prompt (75 in): default effort → 3,729 out, of which **2,919 reasoning (78%)**, 34 s, $0.00225 list. `reasoning_effort="low"` → 770 out (140 reasoning), 9 s, $0.00047. `"minimal"` → 1,795 out (1,142 reasoning), 18 s. Effort is accepted but noisy; measure properly before choosing a default (week 2, `core/llm.py`).
  - GLM-5.3-Flash: no reasoning tokens, 252 out, 4.7 s, $0.00013. Shorter code; quality not compared yet.
  - Go returns `usage.opencode.allowance` (what counts against the plan limit) and `list_cost`. For DeepSeek, allowance = **0.6 × list**; for GLM, allowance = list. The budget guard should use `allowance`. Our own price table matched `list_cost` exactly.
  - Cached tokens were 0 (short, first-time prompts).
- **Original vs Go (2026-11-16):** works through `ChatOpenAI(base_url=...)`, so parity is feasible. ~80% of output is reasoning; ~$0.008 and ~105 s per agent run. Reproduced a silent empty-result "success" (#6) and a crash whenever `invoke_agent` gets `user_instructions` (#18); the parity adapter must call `_compiled_graph.invoke` directly. Details in `docs/original-bug-log.md`; scratch runner to become `evals/parity/` in week 7.
- **Decided 2026-11-16:** the AutoGluon Docker image stays in week 1 (user's choice). It lives in `images/autogluon/` (Dockerfile + `smoke_fit.py`): AutoGluon Tabular 1.6.3 with LightGBM, CatBoost, XGBoost and fastai/torch NN, **CPU-only torch 2.13.0**, base image pinned by digest, non-root user, `HOME=/tmp` so it runs with a read-only root. Exact versions in `images/autogluon/requirements.lock` (111 packages). **Built and verified 2026-11-16:** image 3.1 GB, build ~4 min. Smoke fit (3,000 × 20 synthetic binary, `medium_quality`, 90 s limit) under `--network none --read-only --memory 3g --cpus 2 --cap-drop ALL`: 12 models trained (GBMs, CatBoost, XGBoost, RF/ET, NeuralNetTorch, NeuralNetFastAI, ensemble), fit 16 s, test accuracy 0.959, peak RSS ~0.66 GB, model files 49 MB.
- **Reasoning effort decided 2026-11-16: default `low`** (`research/reasoning_effort/`). 8 runs × 5 levels on a cleaning task, code checked in the sandbox: 40/40 correct at every level (ceiling effect), `low` cheapest and fastest (median 1,443 out tokens, 14 s, $0.00104 allowance vs default 2,580 / 24 s / $0.00181). Revisit on the week-6 eval suite and in week 14. `"default"` omits the parameter; `DQ_REASONING_EFFORT` overrides.
- **CI lesson:** ruff's formatter also formats Python blocks inside Markdown; `*.md` is excluded in `[tool.ruff.format]`. Run `uv run ruff format --check .` before pushing doc changes.
- **Installed 2026-11-16:** OpenJDK 21; Docker Desktop (engine 29.9.3, WSL2 backend, cgroup v2). WSL integration with Ubuntu is **off** and Docker does not start at login (RAM is tight: ~81% used at idle). `C:\Users\krtk2\.wslconfig` caps WSL at **memory=4GB, processors=4, swap=4GB**.
- **Sandbox flags verified on this machine (2026-11-16)** with `python:3.12-slim`: `--network none --read-only --tmpfs /tmp:size=64m --memory 1g --memory-swap 1g --cpus 1 --pids-limit 64 --user 65534 --cap-drop ALL --security-opt no-new-privileges` and a read-only input mount. Network, root-fs writes, input writes, chown, a 1.2 GB allocation and a 200-process fork were all blocked; `/tmp` writes worked; `docker kill` stopped a running container. Container start + probe took ~1.5 s.
- **Week 10 RAM (resolved):** the planned raise to 6 GB was not needed; one AutoML container at a time peaked at ~1 GB under the 4 GB WSL cap. Still close heavy apps during long runs.
- **Original project** is cloned to `.original/ai-data-science-team` (git-ignored) with its own venv `.original/venv` (pandas<3 via `.original/constraints.txt`, plus `ipython`, which it imports but does not declare).

## Status
**Week 2 started 2026-11-16: `core/llm.py` done** (with `core/cost.py`, `core/budget.py`). Structured output only (json_schema strict → json_object fallback → one repair → `StructuredOutputError`), fallback model on 5xx/429/connection errors, Go `allowance` used for cost, budget guard (eval raises / interactive warns) with a JSONL ledger in `runs/spend.jsonl` until the week-4 DuckDB recorder. Default `reasoning_effort` is `low` (experiment done 2026-11-16). **`core/state.py` + `core/statuses.py` done 2026-11-16** (spec S2): `AgentResult.from_run` decides status from facts (error/no output/empty output → failed; failed blocking check → failed; warning or review reason → needs_review); a validator makes inconsistent results impossible to construct; regression tests for bug log #5 and #6. **Throwaway mini agent done 2026-11-16** (`research/mini_agent/`, LangGraph 1.2.22): generate → execute (Docker) → validate → fix loop → `AgentResult`; normal run and a forced-failure retry both ended in success. Lesson: a status is only as honest as its checks (an unchecked double space passed). 

**Week 3 engine done 2026-11-16** (see spec S6): `core/engine.py` (330 lines), `agent_spec.py`, `prompts.py`, `review.py`, `validation.py`, `sandbox/executor.py` + `docker_executor.py`. 107 tests (scripted LLM + in-process executor for test-authored code only) plus one replay of a real Go run (2 calls, $0.0021 allowance; recorded with Docker, replayed in-process with the reviewed recorded code). A test caught `columns_dropped` treating any mention of a column as permission to drop it; fixed to require a removal word too. 

**Week 4 done 2026-11-16:** agents `cleaning`, `wrangling`, `visualization` (figure output, Plotly JSON) as `AgentSpec`s in `src/dataquorum/agents/`; `make_engine(spec)` wires Go + budget guard + Docker + recorder. New sandbox image `dataquorum/sandbox:0.1` (`images/sandbox/`, 700 MB: pandas 2.3.3, numpy, pyarrow, plotly 7.1.0) is now the executor default. `tracing/recorder.py`: DuckDB `runs` + `llm_spend` tables (`at` is reserved in DuckDB 2.0, hence `spent_at`). 125 tests; live replays for wrangling (totals; missing columns → honest `failed`, unlike the original's bug #6) and visualization (bar chart). Live recording cost ~$0.0035.
- **Early stop done 2026-11-16:** `RecommendedSteps.blocker` ends impossible requests after one call (missing-column replay: 3 calls / $0.00252 → 1 call / $0.00013). All four engine cassettes re-recorded with the new prompt; no false blocks on the normal requests. Watch false-block rate in the week-6 evals.


**Week 5 done 2026-11-16** (spec S5): `DockerExecutor` rewritten on the docker SDK. No writable host mount any more (week-4 gap: code could fill the host disk via the bind-mounted output dir): output goes to a 256 MB `noexec,nosuid` tmpfs and is read back with `exec head -c` (100 MB cap) while the container is alive; `docker cp` can't see tmpfs. Logs capped at 2 MB, `ipc_mode=none`, empty env, `nofile` 256, label `dataquorum.sandbox`. Docker down / image missing → `SandboxUnavailableError` (not an LLM retry). `tests/sandbox/`: 19 escape probes + capability/mount/timeout/flood/size tests (33 total, marker `docker`, auto-skip without Docker/image); CI job `sandbox` builds `images/sandbox` and runs them. Dockerfiles moved `docker/` → `images/` (ruff confused it with the `docker` package). 162 tests locally.


**Week 6 eval suite built 2026-11-16** (spec S10): 5 OpenML datasets (content-hash pinned, same under pandas 2/3), 9 fault kinds that produce messy input **and** expected output, 55 YAML tasks (25 cleaning / 20 wrangling / 10 viz, 3 impossible), 12 checkers, oracle check (all 55 solvable and non-trivial), scoring with **false-success rate** as headline, parallel runner (`python -m dataquorum.evals.runner`, `--dry-run`, `--only`, `--tag`, `--limit`, `--model`, `--reasoning-effort`). Found while building: an ambiguous combined task (median before/after dedup), a checker mutating shared output, a pandas-version-dependent dataset hash. 149 non-Docker tests.
- **Baseline run 2026-11-16** (`docs/results/week6-baseline.md`): 55/55 correct, 0 false successes (Wilson 95% 93.5–100%), $0.0465 allowance, median 12.7 s. Two warning checks flagged correct work for review (requested blanking; imputation creating identical rows) — fixed and re-verified live. **Ceiling effect:** week-9 tasks must be harder to discriminate configurations. Runner should split LLM vs sandbox time (one 199 s outlier at peak).


**Week 7 done 2026-11-16:** `evals/stats.py` (Wilson, bootstrap, pass@k, exact McNemar; verified vs SciPy/statsmodels), runner ablation flags (`--no-checks`, `--no-recommend`, `--max-retries`, `--repeats`), `evals/compare.py`, parity eval (`images/original/` Dockerfile + worker, `evals/parity/run.py`; one locked-down container per task, API key via read-once file). **Parity result** (`docs/results/week7-parity.md`): original 47/55 correct, **7 false successes** (invented answers to all 3 impossible requests; imputed means where told to leave NaN; dropped unrequested rows) vs DataQuorum 55/55, 0; McNemar p = 0.0078; ~4.4× cheaper, ~3× faster; original emits 5.3× the output tokens. Caveat: home-advantage suite; original cost estimated from tokens.


**Week 8 done 2026-11-16 (v0.1.0 prepared):** `Usage.execution_seconds` (sandbox) split from `seconds` (LLM), DuckDB migrated in place. 16 hard tasks (`evals/tasks/hard.yaml`; faults `mixed_numeric_text`, `near_duplicates`; checkers `pivot`, `binned_count`, aggregate `value_expr`/`min_count`/`top_per`, figure `color`/`order`). First "false success" was a **benchmark bug**: category variants could leave a label only in a variant spelling → fixed (original spelling stays a strict majority) + oracle `label_problems` guard; after fix hard suite 16/16 (`docs/results/week8-hard-tasks.md`). Cleaning agent is the slow one (48–64 s LLM, 4–6k tokens vs ~10 s for others) — week-14 ablation candidate. README v1 (quick start verified live), CHANGELOG, version 0.1.0.
- **Released v0.1.0 on 2026-11-16:** tag `v0.1.0` (commit `04fca69`), GitHub release https://github.com/krtk06/dataquorum/releases/tag/v0.1.0 with wheel + sdist attached. PyPI upload still pending (user: `uv build` then `uv publish` with their token; 0.0.1 placeholder was never uploaded).


**Week 9 done 2026-11-16** (`docs/results/week9-new-agents.md`): engine generalised for inputs (`core/inputs.py`: dataframe / folder / SQLite-DuckDB file) and language (python / sql); `core/findings.py` grounded findings (numbers must appear in the output table) with an engine `explain` step; checkpoint setup in `core/checkpoint.py` (engine.py ~400 lines — keep new logic out of it). Agents: `loader` (folder mounted read-only at `/io/data`), `sql` (`sandbox/sql_executor.py`: sqlglot guard + read-only connections, 16 attacks × 2 dialects blocked), `eda`, `features` (sklearn; blocking hygiene checks: rows kept, target unchanged, no target leakage, numeric + complete; warning for missing values one-hot encoded). Sandbox image **0.2** (+ openpyxl, scikit-learn). 31 tasks (`week9.yaml`, built folders/databases via `evals/inputs.py`, checkers in `evals/checkers_ml.py`) → **102 tasks total**. Live: 31/31 after fixing a checker that didn't understand sklearn's `num__`/`cat__` names (5 fake false successes); `evals/rescore.py` re-grades saved outputs without LLM calls. Rule: always pipe test runs with `set -o pipefail` (a `| tail` once hid a failing test before a push).
**Week 10 done 2026-11-16** (`docs/results/week10-automl.md`, spec S11): `ml/automl/` = `fit_predict` (FLAML / H2O / AutoGluon, one locked-down container per call, host-side scoring), `runner.py` (runs inside the container), `benchmark.py` (8 OpenML tasks, 3 seeds, 60 s, resumable JSONL, report). `evals/stats.py` gained `friedman` + `chi2_sf` (checked vs SciPy). Image `dataquorum/automl:0.1` built (3.1 GB; xgboost pinned 2.1.4 because flaml[automl] 2.8.0 needs <3, which drags in a 417 MB unused nvidia-nccl wheel). **Result:** avg rank AutoGluon 1.62, H2O 1.88, FLAML 2.50; Friedman p = 0.197, CD 1.17 → no significant difference; AutoGluon fastest (18 s median fit, stops early) and clearly best on kin8nm. Bug found: H2O renames number-like class levels (`2` → `p2`); runner now locates the proba column by level position, 6 runs re-run. AutoGluon 1.6 has no fit-level seed (seed only changes the split). Peak container memory ~1 GB, so the 4 GB WSL cap was enough.
- **Next: week 11** (ML correctness agent: leakage, validation chooser, metric chooser; Optuna; SHAP; calibration). Still open: week-14 ablations (`--no-checks`, `--no-recommend`); PyPI upload (user).
- **2026-11-16: work moves to Codex** (Claude subscription ending). `AGENTS.md` is the handoff file Codex reads; keep it and this file in sync.
- **Earlier plan for week 9:** — Loader, SQL (sqlglot read-only), EDA, Feature Engineering agents + ~30 tasks. Still to run: ablations (`--no-checks`, `--no-recommend`) for week 14.

**Week 1 (done except the user's PyPI upload).** GitHub repo: https://github.com/krtk06/dataquorum (public). Done: uv skeleton (`src/dataquorum/…`, Python 3.12, ruff, mypy strict, pytest), CI (Ubuntu + Windows, plus a pandas-3 job), MIT license, README with credit, Go smoke test (run 2026-11-16, see Open items), `docs/spec.md` (S1–S10), `docs/original-bug-log.md`, Docker sandbox flags verified. Original run against Go and AutoGluon image built (both 2026-11-16). Left in week 1: PyPI upload (user), LLM basics + LangGraph reading (user).
