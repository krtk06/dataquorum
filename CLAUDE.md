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
Python 3.12 · **uv** + `pyproject.toml` + lock file · ruff · mypy · pytest · `pytest-recording` (replay recorded LLM calls in CI) · GitHub Actions
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
- **Still to check:** does the original tool work against Go's endpoint (needed for the parity test)?
- **CI lesson:** ruff's formatter also formats Python blocks inside Markdown; `*.md` is excluded in `[tool.ruff.format]`. Run `uv run ruff format --check .` before pushing doc changes.
- **Installed 2026-11-16:** OpenJDK 21; Docker Desktop (engine 29.9.3, WSL2 backend, cgroup v2). WSL integration with Ubuntu is **off** and Docker does not start at login (RAM is tight: ~81% used at idle). `C:\Users\krtk2\.wslconfig` caps WSL at **memory=4GB, processors=4, swap=4GB**.
- **Sandbox flags verified on this machine (2026-11-16)** with `python:3.12-slim`: `--network none --read-only --tmpfs /tmp:size=64m --memory 1g --memory-swap 1g --cpus 1 --pids-limit 64 --user 65534 --cap-drop ALL --security-opt no-new-privileges` and a read-only input mount. Network, root-fs writes, input writes, chown, a 1.2 GB allocation and a 200-process fork were all blocked; `/tmp` writes worked; `docker kill` stopped a running container. Container start + probe took ~1.5 s.
- **Week 10 reminder:** AutoML needs more RAM. Raise `.wslconfig` memory to 6 GB, run `wsl --shutdown`, close Brave/ChatGPT/Teams during benchmark runs, and keep benchmark datasets small.
- **Original project** is cloned to `.original/ai-data-science-team` (git-ignored) with its own venv `.original/venv` (pandas<3 via `.original/constraints.txt`, plus `ipython`, which it imports but does not declare).

## Status
**Week 1 in progress.** GitHub repo: https://github.com/krtk06/dataquorum (public). Done: uv skeleton (`src/dataquorum/…`, Python 3.12, ruff, mypy strict, pytest), CI (Ubuntu + Windows, plus a pandas-3 job), MIT license, README with credit, Go smoke test (run 2026-11-16, see Open items), `docs/spec.md` (S1–S10), `docs/original-bug-log.md`, Docker sandbox flags verified. Left in week 1: PyPI upload (user), point the original at Go, AutoGluon Docker image, LLM basics.
