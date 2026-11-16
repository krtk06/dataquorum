# DataQuorum spec (v0, week 1)

Status: draft, 2026-11-16. Sections are numbered S1-S10 so issues, tests and the bug log
(`docs/original-bug-log.md`) can cite them. Change this file when a decision changes.

## S1. Scope

DataQuorum matches the features of business-science/ai-data-science-team: single agents
(cleaning, wrangling, visualization, loader, SQL, EDA, feature engineering, ML), a
supervisor that routes between them, and a Streamlit studio. It does not copy code.

What DataQuorum adds: every agent proves its work. Generated code runs in Docker, outputs
are validated, results carry an honest status, and every run is recorded with its cost.

Out of scope for v1: time-series agent, a separate dashboard app, MCP server (stretch).

## S2. Agent result contract

Every agent, and the supervisor, returns the same Pydantic model:

```python
class Status(StrEnum):
    SUCCESS = "success"            # output exists and every blocking check passed
    FAILED = "failed"              # no usable output (errors, retries exhausted, refusal)
    NEEDS_REVIEW = "needs_review"  # output exists but a warning check failed or a human must decide

class AgentResult(BaseModel):
    run_id: str
    agent: str
    status: Status
    output: Output | None          # DataFrame ref (Parquet path), figure JSON, model ref, or text
    code: str | None               # final code that produced the output
    steps: list[str]               # recommended steps the code implements
    validation: list[CheckResult]  # see S4
    usage: Usage
    error: str | None              # last error, set whenever status != success
    message: str                   # one-paragraph human summary
```

`Usage`: model id, prompt / completion / reasoning / cached tokens, estimated cost (USD),
LLM calls, retries, wall time. `CheckResult`: name, passed, severity (`blocking` / `warning`),
detail.

Rules:
- **Never a false success.** `success` requires a non-empty output and all blocking checks
  passed. Retries exhausted means `failed`, with `error` set. (Bug log #5, #6.)
- Status is decided by the engine from the checks, never by the LLM.
- `output` for data is always a file reference, not an in-memory dict, so results are
  replayable and can be large.

## S3. LLM layer (`core/llm.py`)

- One provider interface over OpenAI-compatible endpoints (OpenCode Go). No local models.
- Config: base URL, model id, fallback model id, temperature, timeout, max tokens. Model ids
  are locked per eval run and recorded in every trace row.
- Headers on every Go request: `User-Agent: dataquorum/<version>` and a stable
  `x-opencode-session` per run.
- **Structured output only.** Every LLM step returns a Pydantic model: `json_schema` response
  format first; if the model rejects it, `json_object` + schema in the prompt + Pydantic
  validation; one repair attempt, then the step fails. Code is a field (`python_code: str`),
  never scraped from prose or `<think>` text. (Bug log #1.)
- Reasoning text, if the model returns it, is stored in the trace and never executed.
- **Cost meter:** price table per model (input, output, cached input; peak = 2x at
  01:00-04:00 and 06:00-10:00 UTC, Mon-Fri). Reasoning tokens are billed as output.
- **Budget guard:** rolling totals per 5 hours and per week, persisted in the trace DB. Evals
  pause at $18 / 5 h or $45 / week; interactive use warns at the same limits.
- **Prompt caching:** fixed text (role, rules, schema) first, variable data last.

Open (smoke test answers these): does Go accept `json_schema` for DeepSeek V4.1 Flash; does it
return reasoning tokens and cached-token counts.

## S4. Validation framework (`core/validation.py`)

A check is a function `(input, output, context) -> CheckResult`. Each agent declares its
checks; the engine runs them after every successful execution.

Generic checks for DataFrame agents:

| Check | Severity |
|---|---|
| Output is a DataFrame with at least 1 row and 1 column | blocking |
| Function returned without raising and within time limit | blocking |
| Row loss above threshold (default 50%) without the user asking for filtering | warning |
| Columns dropped that the user did not mention | warning |
| Missing values increased in any column | warning |
| Dtypes changed to `object` from numeric | warning |

Agent-specific checks live with the agent (for example, visualization: figure JSON parses and
has at least one trace; SQL: query is read-only per sqlglot and returns rows).

A failed blocking check sends the run to `fix` (counts as a retry). Warnings never trigger a
fix; they make the final status `needs_review`.

## S5. Sandbox (`sandbox/docker_executor.py`)

All generated code runs here. There is no `exec()` path in the host process. (Bug log #2-#4.)

Container settings:
- `network_mode="none"`
- read-only root filesystem; writable `/tmp` tmpfs (64 MB); data mounted at `/io`
  (input read-only, output dir writable)
- `mem_limit` 1 GB (configurable), `nano_cpus` 1 CPU, `pids_limit` 64
- non-root user, `cap_drop=["ALL"]`, `security_opt=["no-new-privileges"]`
- wall-clock timeout (default 60 s); container killed and removed on timeout
- image pinned by digest, built from `sandbox/image/` with pandas, numpy, scikit-learn,
  plotly, pyarrow only

Protocol: host writes `input.parquet` and `code.py` (defines one function with a fixed
name); the runner inside calls it and writes `output.parquet` (or `output.json` for
figures) plus `meta.json` (stdout, stderr tail, duration, exception type and message).

Escape tests (week 5) must show each of these fails: network access, writing outside
`/tmp` and `/io/out`, reading host files, fork bomb, memory bomb, infinite loop, importing
blocked-in-the-original modules (`os`, `subprocess`) to do any of the above.

## S6. Engine (`core/engine.py`, LangGraph)

```
summarize -> recommend -> [human_review] -> generate_code -> execute -> validate
                                                   ^                       |
                                                   +------- fix <----------+ (error or blocking check, retries left)
validate -> finalize   (pass, or retries exhausted -> failed)
```

- Each agent supplies: prompts, output schema, checks, and an optional post-processor.
  Agents contain no graph code.
- State is a Pydantic model; nodes never format a missing field (bug log #8).
- Retry policy: `max_retries` (default 3) in engine config; the count always starts at 0
  (bug log #9).
- Human review uses a LangGraph `interrupt` with a typed reply:
  `Approve`, `Revise(instructions: str)`, or `Reject(reason: str)`. Free text from a UI is
  parsed into this type, not matched against `"yes"` (bug log #7). Reject ends the run with
  `failed`.
- A checkpointer is always on, so runs can resume after review.

## S7. Supervisor (`supervisor/`)

The router returns a structured decision:

```python
class RouteDecision(BaseModel):
    next: Literal[<agent names>, "finish"]
    instructions: str        # what to tell that agent, kept separate from the user's question
    reason: str
    refusal: str | None      # set when the request cannot or should not be done
```

- The user's original request is passed unchanged to every agent; router instructions are
  a separate field (bug log #13).
- A refusal ends the run with `failed` and the refusal text as `message`. It is never
  converted to "finish" (bug log #12).
- Invalid router output gets one repair attempt, then the run fails. No substring fallback
  (bug log #10, #11).
- Hard cap on supervisor steps (default 8) and no agent repeated with identical inputs.
- The final supervisor status is the worst status among the agents that ran.
- Week 13 experiment: a learned router (sentence-transformers embeddings + scikit-learn
  classifier on about 300 labelled requests), compared with the LLM router on accuracy, cost
  and latency.

## S8. Tracing (`tracing/recorder.py`)

One DuckDB row per run: run id, parent run id (for supervisor runs), timestamp UTC, agent,
model id, status, retries, tokens (prompt / completion / reasoning / cached), cost, wall time,
check results (JSON), error, git commit, eval task id (if any). LLM calls get a child table.
The budget guard (S3) and the studio Runs page read from this database.

## S9. Engineering rules

- No file over about 400 lines; CI fails on any `.py` file over 450.
- Python 3.12, uv lock file committed, ruff + mypy strict + pytest in CI on Ubuntu and
  Windows, plus a pandas 3 job.
- LLM calls in tests are replayed from JSON cassettes through an `httpx2` transport
  (`tests/replay.py`); cassettes never store request headers. CI never calls a live API.
  Re-record with `DQ_RECORD=1 uv run pytest <test file>`.
- API keys only from environment variables or `.env`.
- Every bug in `docs/original-bug-log.md` gets a test that fails on the original's behaviour.

## S10. Evals (`evals/`)

- Task file (YAML): id, dataset (OpenML id + version), injected faults, user request, agent,
  checkers, tags. About 50 tasks in week 6, about 80 by week 9.
- Faults (`faults.py`): missing values, duplicates, mixed types, unit strings, outliers,
  inconsistent categories, all seeded.
- Checkers compare the output with a ground-truth function, not with the LLM's own claims.
- Metrics (`stats.py`): pass rate with Wilson 95% CIs, pass@k, McNemar test for paired
  comparisons, bootstrap CIs for cost and latency.
- Parity (`evals/parity/`): runs the original from its own venv on the same tasks, through a
  subprocess adapter, with the same model. Reports pass rate, false-success rate and cost
  for both.
