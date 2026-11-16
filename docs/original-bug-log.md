# Bug log: business-science/ai-data-science-team

Reference copy: commit `4ffeb7f` (2026-01-28), cloned to `.original/` (git-ignored) with its own
venv. Each entry says how it was found: **code** (read in source) or **run** (reproduced at
runtime). Entries marked *to verify* come from the planning notes and still need a repro.

Each bug links to the DataQuorum design rule that prevents it (see `docs/spec.md`).

| # | Area | Bug | Found by | DataQuorum rule |
|---|---|---|---|---|
| 1 | Code extraction | `PythonOutputParser` takes the first fenced block, falls back to a regex `python(.*?)'` that matches any text between the word "python" and a quote, and if both fail returns the **whole LLM reply as code** (`parsers/parsers.py:12-30`). Prose and `<think>` text can end up executed. | code | Structured output only (S3) |
| 2 | Sandbox | Isolation is an import **blocklist** plus a reduced builtins dict (`utils/sandbox.py:53-135`). `getattr`, `type` and `setattr` stay available, and pandas/numpy are pre-imported, so blocked modules are reachable through object attributes. | code | Docker sandbox (S5) |
| 3 | Sandbox | Resource limits use `resource.setrlimit`, which does not exist on Windows; failure is silently ignored (`utils/sandbox.py:38-51`). No memory or file limits on Windows. | code | Docker limits (S5) |
| 4 | Sandbox | Two execution paths bypass the sandbox and call plain `exec()` in the host process: the generic `node_func_execute_agent_code_on_data` (`templates/agent_templates.py:591`) and the SQL path (`:685`). | code | One executor for all code (S5) |
| 5 | False success | When retries run out, the graph still moves on to `explain_code` / `human_review` / `END` with the error left in state (`templates/agent_templates.py:393-430`). The caller gets no explicit failed status. | code | `status` field, never false success (S2) |
| 6 | False success | Supervisor reports "Workflow complete" on empty results. | *to verify* | Validation + status (S2, S4) |
| 7 | Human review | Approval is an exact match on `"yes"` after strip/lower (`agent_templates.py:493`). "Yes." or "yes please" is treated as a change request and appended to the instructions. | code | Typed review decision (S6) |
| 8 | Human review | `"```python\n" + state.get(code_snippet_key)` raises `TypeError` if no code was generated yet (`agent_templates.py:483`). | code | Engine never formats missing fields (S6) |
| 9 | Retry | `fix` increments `retry_count` with `state.get(...) + 1`; a missing count raises `TypeError`, and `error_and_can_retry` silently disables retries when either key is `None` (`agent_templates.py:393-399, 789`). | code | Retry policy in engine config (S6) |
| 10 | Routing | Router text fallback checks route options in order with a substring test, and `FINISH` is first (`supervisor_ds_team.py:258, 314-317`). A reply like "before we finish, use Data_Cleaning_Agent" routes to FINISH. | code | Structured routing decision (S7) |
| 11 | Routing | The `next: X` fallback regex is a raw string with doubled backslashes (`r"(?:next|route)\\s*[:=]\\s*..."`, line 323), so it matches a literal backslash and never fires. | code | Structured routing decision (S7) |
| 12 | Routing | Empty or unparseable router output silently becomes `FINISH` (lines 301-302, 332). Refusals and errors are discarded instead of surfaced. | code | Refusal is a first-class outcome (S7) |
| 13 | Routing | Router merges appended instructions into the question. | *to verify* | Instructions kept separate (S7) |
| 14 | Packaging | `requirements.txt` has no version pins (pandas, langchain, langgraph, ...). Breaks under pandas 3. | code | Lock file + pandas 2/3 CI (S9) |
| 14b | Packaging | `import ai_data_science_team` fails on a clean install: `agents/data_cleaning_agent.py:22` imports `IPython.display`, but IPython is not in `requirements.txt`. Workaround in our parity venv: install `ipython`. | run | Lock file + import test in CI (S9) |
| 15 | Size | `apps/ai-pipeline-studio-app/app.py` is 15,574 lines; `multiagents/supervisor_ds_team.py` is 4,922 lines. | code | ~400-line file limit (S9) |
| 16 | Quality | No tests and no CI in the repository. | code | Tests + replay CI (S9) |

## Runtime checks still to do (need the Go API key)

- Point the original at Go (`langchain_openai.ChatOpenAI(base_url=..., model="deepseek-v4.1-flash")`) and run the cleaning, wrangling and visualization examples.
- Reproduce #6 (empty result reported as complete) and #13 (instruction merging).
- Record the token use of one cleaning run, for the cost comparison.
