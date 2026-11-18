# Week 14 ablation and model study

## Offline results

The offline oracle passed all **114 answer keys**, including all 16 selected cases. Three
scripted examples then exercised the real engine and independent scoring. Each example
fixes the generated replies so it can demonstrate the safeguard's behavior.
All 385 local tests passed, including 41 Docker tests. Ruff and strict mypy also passed.

| Case | Full engine | Ablated engine |
|---|---|---|
| Audit omits a categorical target proxy | Blocking check rejects the first report. A retry adds the risk and returns the expected review status. | With checks disabled, the incomplete report returns success and the oracle grades it as a false success. |
| A request names an absent column | The planning step gives an explicit blocker and stops. | Without planning, the scripted code invents the column and the oracle grades the success as false. |
| EDA claims a value absent from its correct summary table | The grounding check marks the explanation for review. | With checks disabled, the same explanation returns success. The study grader checks the narrative and records a false success. |

The original `--no-checks` implementation kept EDA's grounding check active. The flag now
disables that check while preserving the call that writes the findings. The same structural
checks on execution, output type and empty results apply to both arms.

Earlier records illustrate why every apparent failure needs triage. Regrading the 31 saved
Week 9 outputs converted all five feature-engineering false-success flags to passes. The
checker had discarded valid scikit-learn output names with transformer prefixes. The
Week 8 false-success flag came from a fixture that removed the only correct neighborhood
spelling. That fixture changed before a new run. The 12 saved Week 11 outputs retained
their scores when regraded. [Week 8 fixture diagnosis](week8-hard-tasks.md) ·
[Week 9 checker diagnosis](week9-new-agents.md) · [Week 11 results](week11-correctness.md)

These cases demonstrate mechanisms and grader corrections. The live study is required
to estimate how often the failures occur under each model and safeguard configuration.

## Paired live protocol

Use 16 fixed cases drawn from the existing suite, two for each of the eight specialist
agents. The list includes four impossible requests and a leakage audit that must return
`needs_review`. Its accuracy describes this deliberately challenging slice. The
supervisor routes are studied in Week 13. This study calls specialists directly.

| Arm | Model and settings |
|---|---|
| Full baseline | DeepSeek V4.1 Flash with low reasoning effort |
| No checks | DeepSeek V4.1 Flash with low reasoning effort and output validation disabled |
| No recommend | DeepSeek V4.1 Flash with low reasoning effort and no planning call |
| Other full-engine models | GLM-5.3-Flash, Qwen3.8 Flash, MiniMax M3.1 Flash and MiMo-V2.6-Flash at provider defaults |

All seven arms receive the same requests, input seeds and one engine fix attempt. A
seeded schedule interleaves arms within tasks. One worker runs at a time and provider
fallback is disabled. The model comparison measures these deployed configurations.
[OpenCode Go model catalog](https://opencode.ai/inference/go/v1/models)

Report verified correctness, false successes, allowance per verified task and median wall
time on tasks completed by all arms in each comparison. For unavailable tasks, verified
correctness requires a declared blocker and the expected lack of output. The legacy
status-based grading remains visible. For EDA, the study also grades whether the narrative
contains unsupported numbers or is missing. Pairwise comparisons include exact McNemar
tests and paired bootstrap intervals for allowance and wall-time differences. This small,
preselected benchmark supports exploratory comparisons.

The failure queue captures declared blockers, provider/request errors, invalid structured
responses, agent validation failures, execution or empty-output failures, reference
mismatches, status mismatches and grader exceptions. Each row retains the evidence for
review. A grader exception removes that task from the matched comparison until adjudicated.
Earlier benchmark bugs motivate this check.

## Running and resuming

Run from the repository root in PowerShell. The plan command verifies the 16 cases using
cached data without calling an LLM.

```powershell
uv run --locked python -m dataquorum.evals.study plan
```

The live run requires approval for the provider allowance. Its expected range is
**$0.15–$0.60 of allowance** and **35–75 minutes** with one worker for all 112 trials.
The most recent archived runs for the same 16 tasks used $0.01511 in aggregate and took
337.9 seconds of inference plus execution. That figure provides a rough anchor across
different days, model prices and retry behavior.

```powershell
uv run --locked python -m dataquorum.evals.study run --max-allowance 0.60
uv run --locked python -m dataquorum.evals.study report
```

The runner saves the plan, sandbox image ID, execution source fingerprint, package
versions, a row per completed trial, and a per-call Go allowance ledger. Resume with the
same output directory and runtime. Completed task/arm pairs are skipped. Interrupted
trials can incur a new charge on retry. Both attempts count toward the spend stop.
The stop threshold is checked between calls and a call in flight may cross it. Missing
provider cost stops the study and leaves a marker for reconciliation. If forced
termination leaves a lock file, confirm the recorded process has stopped before clearing it.

**Live status:** awaiting allowance approval.
