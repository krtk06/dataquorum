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

Implemented in `core/statuses.py` (`Status`, `Severity`, `CheckResult`, `decide_status`) and
`core/state.py` (`AgentResult` plus `parent_run_id` and `created_at`; outputs are
`DataFrameOutput` (Parquet path + shape + dtypes), `FigureOutput` (Plotly JSON),
`ModelOutput` (URI + framework + metrics) and `TextOutput`, told apart by `kind`). Agents
build results with `AgentResult.from_run(...)`, which computes the status; a validator
rejects any result that breaks the rules below, even one built by hand.

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

Container settings (week 5, docker SDK):
- `network_mode="none"`, `ipc_mode="none"`, empty environment, hostname `sandbox`
- read-only root filesystem; input mounted read-only at `/io/in`
- **no writable host mount**: `/tmp` is a 256 MB tmpfs with `noexec,nosuid,nodev`; the
  runner writes the result to `/tmp/out/`, and the host reads it with `exec head -c` while
  the container is still alive, capped at 100 MB. (`docker cp`/`get_archive` cannot see
  tmpfs, verified 2026-11-16.) This closes the week-4 gap where code could fill the host
  disk through a writable bind mount.
- `mem_limit` 1 GB with no swap, `nano_cpus` 1 CPU, `pids_limit` 64, `nofile` 256
- container logs capped at 2 MB (json-file, one file), so a stdout flood cannot fill disk
- non-root user 10001, `cap_drop=["ALL"]`, `security_opt=["no-new-privileges"]`
- wall-clock timeout (default 60 s); the container is always force-removed; containers
  carry the label `dataquorum.sandbox` so leftovers are easy to find
- generated-code failures return `ExecutionResult(ok=False)`; Docker down or image missing
  raise `SandboxUnavailableError` (the engine must not burn LLM retries on those)
- image `dataquorum/sandbox:0.1` built from `images/sandbox/` (base pinned by digest):
  pandas 2.3.3, numpy 2.5.5, pyarrow 26.0.1, plotly 7.1.0 only (~700 MB; full set in
  `requirements.lock`). scikit-learn is added when the feature-engineering agent needs it.

Protocol: host writes `input.parquet`, `code.py` and `runner.py` to the input mount. The
container runs `python runner.py <function> /tmp/out/<file>; echo __DQ_DONE__ $?; sleep`.
The runner prints one JSON line (`ok`, `error`); the host waits for the done marker, reads
the JSON line from the log tail, copies the output out, then removes the container.
Exit 137 is reported as "killed, most likely for using too much memory".

Escape tests (`tests/sandbox/test_escapes.py`, marker `docker`, run in the CI `sandbox`
job which builds the image): TCP connection, DNS lookup, writing to `/etc`, `/usr`, the input
data and the runner, `setuid(0)`, `chown` to root, raw sockets, `mount`, `su`, the Docker
socket, executing a script from `/tmp`, `/proc/sysrq-trigger`, seeing host environment
variables (including the API key), filling `/tmp`, forking 200 processes, allocating 3 GB.
Plus: runs as uid/gid 10001 with zero effective and bounding capabilities and
NoNewPrivs=1; `/` and `/io/in` are read-only; `/tmp` is noexec/nosuid; infinite loops time
out; a 20 MB stdout flood does not break a run; oversize output is rejected; no container is
left behind after any test.

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

Implemented (week 3): `core/engine.py` (`Engine.start` / `Engine.resume`, returning an
`EngineRun` that is either finished or paused with a review payload), `core/agent_spec.py`
(an agent is an `AgentSpec`: role, recommend guidance, code guidance, extra checks,
optional reasoning effort), `core/prompts.py` (fixed system text first for caching; the
request is passed verbatim in its own section, reviewer notes in another),
`core/review.py` (`parse_review`), `core/validation.py` (S4 checks) and
`sandbox/executor.py` + `docker_executor.py`. Revise sends the run back to `recommend`;
fix attempts see the previous code and the crash or failed blocking checks.
Checkpoint types are registered with LangGraph's msgpack allowlist. A
`BudgetExceededError` is not caught, so an eval batch stops instead of logging failures.
Output kinds (week 4): `AgentSpec.output_kind` is `"dataframe"` (generated
`transform(df) -> DataFrame`, stored as Parquet) or `"figure"` (generated
`make_figure(df) -> plotly Figure`, stored as Plotly JSON in `FigureOutput`). The kind picks
the code rules in the prompt and the default generic checks (S4: figure checks are
`figure_has_traces` and `traces_have_points`, blocking, and `figure_has_title`, warning).
Agents: `agents/cleaning.py`, `agents/wrangling.py` (shape changes expected, so row-loss and
column-drop warnings are off; must raise on missing columns rather than invent them),
`agents/visualization.py`. `agents.make_engine(spec)` wires Go, the budget guard, Docker and
the recorder.

Early stop (week 4): `RecommendedSteps` has a `blocker` field. When the request cannot be
done with the data (for example a missing column), recommend sets it, the run skips review,
code generation and retries, and ends `failed` with "Cannot do this with the given data:
...". On the missing-column replay this cut cost from 3 calls / $0.00252 to 1 call /
$0.00013. Strict schemas drop `default` keys, since every field is required.

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

Implemented (week 4): `RunRecorder` writes table `runs` (one row per `AgentResult`, replaced
if the same run id is recorded again; includes `failed_checks`, output kind and reference,
and the final code) and `llm_spend` (`spent_at`, model, allowance; one row per LLM call via
`DuckDBLedger`, which replaces the JSONL ledger). The engine records every finished run when
given a recorder. File: `runs/dataquorum.duckdb` (git-ignored); one writer process at a time.

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
