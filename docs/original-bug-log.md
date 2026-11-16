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
| 6 | False success | Asked for "total revenue by sales region and product category" on a table with only `age`, `income`, `city`, `PandasDataAnalyst` **invented the columns**, returned a **0-row table with no error**, and its final message was just "Wrangled table shape: 0 rows x 3 cols." (2026-11-16, DeepSeek V4.1 Flash). With no wrangled data at all, `finalize_output` falls back to "Workflow completed." (`multiagents/pandas_data_analyst.py:466`). The cleaning agent's validation also has no empty-output check (`agents/data_cleaning_agent.py:731-760`). | run + code | Validation + status (S2, S4) |
| 7 | Human review | Approval is an exact match on `"yes"` after strip/lower (`agent_templates.py:493`). "Yes." or "yes please" is treated as a change request and appended to the instructions. | code | Typed review decision (S6) |
| 8 | Human review | `"```python\n" + state.get(code_snippet_key)` raises `TypeError` if no code was generated yet (`agent_templates.py:483`). | code | Engine never formats missing fields (S6) |
| 9 | Retry | `fix` increments `retry_count` with `state.get(...) + 1`; a missing count raises `TypeError`, and `error_and_can_retry` silently disables retries when either key is `None` (`agent_templates.py:393-399, 789`). | code | Retry policy in engine config (S6) |
| 10 | Routing | Router text fallback checks route options in order with a substring test, and `FINISH` is first (`supervisor_ds_team.py:258, 314-317`). A reply like "before we finish, use Data_Cleaning_Agent" routes to FINISH. | code | Structured routing decision (S7) |
| 11 | Routing | The `next: X` fallback regex is a raw string with doubled backslashes (`r"(?:next|route)\\s*[:=]\\s*..."`, line 323), so it matches a literal backslash and never fires. Confirmed: it does not match `"next: Data_Cleaning_Agent"`. | code + run | Structured routing decision (S7) |
| 12 | Routing | Empty or unparseable router output silently becomes `FINISH` (lines 301-302, 332). Refusals and errors are discarded instead of surfaced. | code | Refusal is a first-class outcome (S7) |
| 13 | Routing | Router merges appended instructions into the question. Not found in the library: the supervisor passes the last human message to agents unchanged (`supervisor_ds_team.py:3325, 3356`). May live in the Studio app (`apps/ai-pipeline-studio-app/app.py`); check in week 13. | *to verify* | Instructions kept separate (S7) |
| 14 | Packaging | `requirements.txt` has no version pins (pandas, langchain, langgraph, ...). Breaks under pandas 3. | code | Lock file + pandas 2/3 CI (S9) |
| 14b | Packaging | `import ai_data_science_team` fails on a clean install: `agents/data_cleaning_agent.py:22` imports `IPython.display`, but IPython is not in `requirements.txt`. Workaround in our parity venv: install `ipython`. | run | Lock file + import test in CI (S9) |
| 15 | Size | `apps/ai-pipeline-studio-app/app.py` is 15,574 lines; `multiagents/supervisor_ds_team.py` is 4,922 lines. | code | ~400-line file limit (S9) |
| 16 | Quality | No tests and no CI in the repository. | code | Tests + replay CI (S9) |
| 17 | Routing | The keyword router's `has_word` uses `rf"\\b{w}\\b"` (line 388), which matches a literal backslash, so it **never matches**. Intent words checked only through it (`"predict"` line 502, `"visualize"` line 693) are never detected. | code + run | Structured routing decision (S7) |
| 18 | Crash | `DataCleaningAgent.invoke_agent(user_instructions=...)` puts `("user", text)` tuples into `messages`, whose reducer is plain `operator.add` (line 508), so they are never converted to message objects. `BaseAgent.invoke` then calls `remove_consecutive_duplicates`, which reads `.content` and raises `AttributeError: 'tuple' object has no attribute 'content'` (`utils/regex.py:158`). The crash happens **after** recommend, generate and execute have run, so the LLM cost is spent and the result is lost. Reproduced twice. | run | Typed state, no raw tuples (S6) |

## Runtime results against Go (2026-11-16, DeepSeek V4.1 Flash, default reasoning effort)

Script: kept outside the repo for now; becomes `evals/parity/` in week 7. Data: 12-row frame
with missing `age`, `income` strings like `"$1,200"` and `"n/a"`, mixed-case `city`.

| Scenario | Outcome | Output tokens (reasoning) | List cost | Time |
|---|---|---|---|---|
| Cleaning, no instructions | Worked: 12 rows kept, `income` → int64, no retries | 12,980 (10,571 = 81%) | $0.0081 | 103 s |
| Cleaning to an empty table ("drop rows with any missing value, keep income > 1,000,000") | **Crashed** (#18) after all steps ran; result lost | not captured | ~$0.008 | 107 s |
| Analyst asked for columns that do not exist | **Silent false success** (#6): invented columns, 0 rows, no error | 11,488 (9,446 = 82%) | $0.0072 | 111 s |

Conclusions:
- The original **does work against Go's endpoint** through `langchain_openai.ChatOpenAI(base_url=...)`, so the week-7 parity adapter is feasible.
- Its agents call `invoke_agent` with instructions in most real use; #18 means the parity adapter must call `agent._compiled_graph.invoke(...)` directly (or patch the tuple) to get results.
- About 80% of output tokens are reasoning at default effort, matching our smoke test. Cost per agent run is under a cent.
