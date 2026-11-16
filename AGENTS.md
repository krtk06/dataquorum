# AGENTS.md: handoff for coding agents (Codex and others)

> Written 2026-11-16 when work moved from Claude Code to Codex. **Read `CLAUDE.md` next.** It
> holds the full plan, every decision with its reason, and a per-week status log. Keep both
> files up to date as you work (CLAUDE.md = history and decisions; this file = how to work
> here and what's next).

## The project in one paragraph

**DataQuorum** ("A team of AI data-science agents that must prove their work") is a
from-scratch rebuild of business-science/ai-data-science-team (MIT; credited in the README,
no code copied). It is a Python library of LLM agents that write pandas/SQL/sklearn code,
run it in a locked-down Docker sandbox, validate the result and report an honest
`success` / `failed` / `needs_review`. A 17-week portfolio project (ML + data science depth)
with ~15 h/week. Repo: https://github.com/krtk06/dataquorum (public). v0.1.0 is released on
GitHub; **weeks 1–10 are done; week 11 is next.**

## Where things stand (2026-11-16)

- Agents: cleaning, wrangling, visualization, loader, SQL, EDA, feature engineering
  (`src/dataquorum/agents/`), all run by one LangGraph engine (`core/engine.py`).
- Evals: 102 tasks, live score 102/102 with 0 false successes; parity vs the original
  (47/55 with 7 false successes, McNemar p = 0.0078). Results in `docs/results/`.
- Week 10 (just finished): `ml/automl/` runs FLAML, H2O and AutoGluon behind one
  `fit_predict` interface; benchmark on 8 OpenML tasks → no significant difference
  (Friedman p = 0.197), AutoGluon best average rank and fastest.
  See `docs/results/week10-automl.md`.
- Total LLM spend so far: under ~$0.40.

## Next: week 11 (from the CLAUDE.md roadmap)

**ML correctness agent** (leakage detection, validation-strategy chooser, metric chooser) ·
**Optuna** tuning · **SHAP** explanations · **calibration**. Suggested order:
1. `ml/correctness.py`: deterministic checks first (target leakage via near-perfect single
   feature / ID-like columns / post-outcome columns; train/test overlap; time order;
   class imbalance → metric choice; group columns → GroupKFold). Unit tests with seeded
   synthetic data where the right answer is known.
2. Wrap it as an `AgentSpec` (see `agents/features.py` for the pattern: prompts + blocking
   checks) so the LLM proposes and the checks verify.
3. `ml/tuning.py` (Optuna, seeded, time budget), `ml/explain.py` (SHAP), calibration
   (reliability curve + Brier/ECE). These run in the sandbox image, so add packages to
   `images/sandbox/requirements.txt` and bump the tag (currently `dataquorum/sandbox:0.2`).
4. ~10–15 eval tasks with known answers (e.g. a planted leaky column the agent must flag).
5. Write `docs/results/week11-*.md`, update CLAUDE.md, CHANGELOG ("Unreleased"), spec.

Still open from earlier: week-14 ablation runs (`--no-checks`, `--no-recommend`); the PyPI
upload (the **user** does it: `uv build`, then `uv publish` with their own token).

## Working with the user

- **Suggest the next step, then wait for a go-ahead.** The rhythm has been: finish a week,
  summarise it, propose the next week; the user replies "Yes, start week N". Don't start
  the next week, or a large new piece of work, on your own.
- **Ask before anything that spends money or takes long:** live LLM eval runs, re-recording
  replay cassettes, long benchmarks. Say the expected cost and time first (past runs cost
  well under $0.05 each; the week-10 benchmark took ~1 h). Unit tests and lint need no ask.
- **Manual steps for the user come as numbered, step-by-step instructions** (exact menu
  paths, exact commands, which shell). Steps like installs, settings, the PyPI upload.
- **Explain LangGraph, agent and stats ideas plainly:** the user is new to them and learning
  as the project goes. Short, concrete explanations beat jargon.
- **Respect the RAM limit.** The user pushed back on raising the WSL cap. Prefer approaches
  that fit 4 GB, and say up front if something needs more.
- Report results honestly, including failures and your own bugs (see the results docs for
  the tone: what went wrong, why, what changed).

## Local vs cloud

Everything here assumes **Codex running locally on the user's PC** in this folder. A Codex
**cloud** task gets only the git repo. It has none of these:
- the Docker images;
- the `.env` API key;
- the `.original/` clone;
- the `runs/` and `data/` caches.

In the cloud, Docker tests skip silently, evals and benchmarks can't run, and OpenML data
must be downloaded again. So don't claim anything is "verified" from a cloud task unless it
actually ran there; use it only for code and unit-test work.

## Hard rules

- **No AI attribution in git:** never add `Co-Authored-By` trailers for an AI, or
  "Generated with …" lines, to commits or PRs. (History was rewritten once to remove them.)
- **API keys only in env vars / `.env`** (git-ignored). Never in code, logs or commits.
  `.env` has `OPENCODE_GO_API_KEY`; see `.env.example` for optional overrides.
- **Never enter credentials or publish to PyPI.** The user does that.
- **No file over ~400 lines.** `core/engine.py` is at the limit, so put new logic elsewhere.
- **Never report a false success.** A status is only as honest as its checks. Three times a
  "false success" turned out to be a *benchmark/checker bug* (weeks 8, 9, 10). Investigate
  the grader before blaming the agent, and write it up either way.
- Ask before destructive or outward-facing actions (force-push, deleting data, releases).
  Committing and pushing to `main` after a week's work is checked has been the normal flow.

## Environment (user's machine)

- **Windows 11, 16 GB RAM (~81% used at idle), no GPU.** WSL2 capped at 4 GB / 4 CPUs / 4 GB
  swap in `C:\Users\krtk2\.wslconfig`. The user rejected raising it to 6 GB. Run one heavy
  container at a time.
- **Docker Desktop does not start at login.** Start it before Docker tests or evals. WSL
  integration with Ubuntu is off.
- **uv** came from winget. If `uv` isn't on PATH in Git Bash:
  `export PATH="/c/Users/krtk2/AppData/Local/Microsoft/WinGet/Packages/astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe:$PATH"`
- Git Bash rewrites `/tmp`-style args to Windows paths. Use `MSYS_NO_PATHCONV=1` for
  `docker run ... --tmpfs /tmp` and similar.
- Java: OpenJDK 21 on the host (H2O runs in its image anyway).
- Docker images (build from `images/<name>/`):
  `dataquorum/sandbox:0.2` (agent sandbox; Docker tests need it),
  `dataquorum/automl:0.1` (FLAML + H2O), `dataquorum/autogluon:1.6.3`,
  `dataquorum/original:4ffeb7f` (parity runs of the original project).
- The original project is cloned to `.original/` (git-ignored) with its own venv.

## LLM

- Model **DeepSeek V4.1 Flash** (`deepseek-v4.1-flash`) via the user's OpenCode Go
  subscription, OpenAI-compatible endpoint `https://opencode.ai/inference/go/openai/v1`.
  Fallback `glm-5.3-flash`. Default `reasoning_effort="low"` (measured; see CLAUDE.md).
- Structured output only (strict json_schema → json_object → one repair).
- Cost = Go's `usage.opencode.allowance`. **Budget guard:** stop at $18 per 5 h or $45 per
  week (ledger in DuckDB `llm_spend`). Peak hours (2× price): 01:00–04:00 and 06:00–10:00
  UTC Mon–Fri, so schedule big eval runs off-peak and space them out.

## Commands

```bash
uv sync --all-groups                      # dev + evals groups
uv run ruff format --check . && uv run ruff check .
uv run mypy                               # strict
set -o pipefail; uv run pytest -q         # Docker tests auto-skip without Docker/image
uv run pytest -q -m "not docker"          # fast suite (304 tests)
uv run python -m dataquorum.evals.runner --dry-run          # eval suite (needs key + Docker)
uv run python -m dataquorum.evals.rescore runs/evals/<id>   # re-grade saved outputs, no LLM
uv run python -m dataquorum.ml.automl.benchmark --report runs/automl_bench/week10
```

CI (GitHub Actions): lint + mypy + tests on Ubuntu and Windows (pandas 2), Ubuntu pandas 3,
and a Docker job that builds `images/sandbox` and runs the escape tests. Check CI after
every push (`gh run list`, `gh run watch <id>`).

## Gotchas learned the hard way

- **Always `set -o pipefail`** when piping test output (`| tail` once hid a failing test
  before a push).
- ruff's formatter also formats Python inside Markdown, so `*.md` is excluded in
  `[tool.ruff.format]`. Run `ruff format --check .` before pushing doc changes.
- Long one-line checker/report code is allowed via per-file `E501` ignores in
  `pyproject.toml`; untyped libs (openml, scipy, statsmodels, flaml, h2o, autogluon, sklearn)
  are in the mypy `ignore_missing_imports` override.
- The `openai` SDK uses `httpx2`, which vcrpy can't intercept, so tests replay recorded
  calls via our own `tests/replay.py` (cassettes in `tests/cassettes/`).
- DuckDB 2.0 reserves `at` (column is `spent_at`).
- pandas 3 uses a `str` dtype; guards must use `is_numeric_dtype` etc., not dtype == object.
- sklearn `ColumnTransformer` names outputs `num__age`; checkers use `find_column`/`base_name`.
- H2O renames number-like class levels (`2` → `p2`) in prediction columns.
- AutoGluon 1.6 has no fit-level seed. FLAML 2.8 needs xgboost < 3.
- Big benchmark/eval runs: run in the background and make them resumable (JSONL, skip done).

## Layout

```
src/dataquorum/
  core/      llm, cost, budget, state, statuses, engine (LangGraph), prompts, inputs,
             findings, checkpoint, review, validation, agent_spec
  agents/    cleaning, wrangling, visualization, loader, sql, eda, features, checks
  sandbox/   executor, docker_executor, sql_executor (sqlglot read-only guard)
  ml/automl/ fit_predict (__init__), runner (inside container), benchmark
  evals/     datasets, faults, tasks/*.yaml, inputs, checkers(_ml), oracle, scoring,
             runner, compare, rescore, stats (Wilson, bootstrap, pass@k, McNemar, Friedman),
             parity/
  tracing/   recorder (DuckDB runs + llm_spend)
  supervisor/, app/   placeholders (weeks 13, 15–16)
images/      sandbox, automl, autogluon, original
docs/        spec.md (S1–S11), original-bug-log.md, results/week*.md
research/    reasoning_effort/, mini_agent/
tests/       unit + replay + tests/sandbox (Docker escape probes)
runs/, data/ local outputs and dataset caches (git-ignored)
```
