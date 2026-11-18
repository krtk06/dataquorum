# Week 13 supervisor and routing experiment

The supervisor routes one step at a time through the existing agents. It preserves the
original user request, sends the current step instructions separately, and records each
child under the parent run ID. Validated tables become the next input. A refusal ends the
run and preserves its explanation. The supervisor also stops on an invalid decision after
one repair, an incompatible input, an identical repeated operation, or the step limit.

All 377 local tests passed, including 16 supervisor tests. An additional smoke test used
scripted model responses through the default supervisor factory, ran the transformation
in Docker, and verified both parent and child records in DuckDB. Ruff and strict mypy
passed, including mypy in a base install with optional dependencies absent.

## Offline experiment

The [corpus](../../research/router_benchmark/corpus.yaml) defines 225 training requests
from 45 templates and 72 separately written holdout requests. The holdout includes eight
cases per label across eight agents and refusal. Both classifiers use the same eligible
agents for each input kind. For folders and databases there is one eligible agent plus
refusal, so the table below also reports the DataFrame-only subset.

| First route | Correct | Wilson 95% interval | DataFrame only | Median warm prediction | API allowance |
|---|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 51/72 (70.8%) | 59.5–80.1% | 36/51 | 0.41 ms | $0 |
| MiniLM + logistic regression | 57/72 (79.2%) | 68.4–86.9% | 40/51 | 5.16 ms | $0 |

MiniLM got 14 cases right that TF-IDF missed and missed eight that TF-IDF got right.
The exact paired McNemar test gave p = 0.286. Warm time covers one request at a time on the
local Windows CPU, after model loading and classifier training. The small synthetic test
set measures first-step routing. It gives limited evidence for live traffic or multi-step
workflow quality.

MiniLM recognized six of eight refusals. It routed both unsupported folder mutations to
the loader and got only four of eight wrangling requests right. The learned router remains
an experiment. The runtime supervisor uses structured LLM decisions and host-side checks.

The MiniLM encoder uses the published ONNX weights and tokenizer. The download pins a
revision and verifies SHA-256 digests. It uses a two-thread CPU session without PyTorch.
[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Follow-up with more varied training examples

We kept the encoder and classifier settings fixed and added 144 individually written
training requests, 16 per label, to the original 225. The fresh 72-request holdout was
written before the training additions. We checked for duplicate requests across both
training and test sets after normalizing case, punctuation and whitespace. The original
holdout had already been inspected, so it serves as a secondary diagnostic.

| Holdout | 225 training requests | 369 training requests | Newly correct | Newly incorrect |
|---|---:|---:|---:|---:|
| Fresh | 57/72 (79.2%) | 67/72 (93.1%) | 10 | 0 |
| Original | 57/72 (79.2%) | 68/72 (94.4%) | 11 | 0 |

On the fresh holdout, wrangling rose from 3/8 to 6/8, ML correctness from 5/8 to 7/8,
and refusals from 4/8 to 6/8. The exact paired McNemar test gave p = 0.00195.
The expanded model still routed an external texting request to EDA and a database
deletion request to SQL. More varied training improved routing on these synthetic
requests. The experiment changes training volume and variety together, and the holdouts
remain synthetic. The runtime supervisor continues to use structured LLM routing.

Run the comparison after caching the encoder with the command below. It writes
`summary.json` and `predictions.json` under `runs/week13-more-examples`.

```powershell
uv run --locked --extra routing python research/router_benchmark/compare_training.py
```

## Reproduce

Run from the repository root in PowerShell. The first run downloads about 91 MB of public
encoder files. Later runs can use the local cache and omit `--download`.

```powershell
uv run --locked --extra routing python research/router_benchmark/benchmark_routes.py --download
uv run --locked pytest -q tests/test_supervisor.py
```

Add `--live` for the paired OpenCode Go comparison after approving API spend. It sends the
72 synthetic holdout requests to the router, with zero agent execution. It records each
prediction, allowance and wall time in JSONL, then reports both scores on the same cases.
The live comparison is pending approval.
