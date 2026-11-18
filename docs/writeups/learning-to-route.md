# What more training examples changed in agent routing

DataQuorum's supervisor chooses an agent for the current request and passes forward
verified tables. Routing a request well is one decision in that workflow. A routing model
can still choose the wrong task even when the downstream agent executes correctly.

The Week 13 experiment asked a narrower question. Given the request and the eligible
agents for the input type, can an offline classifier choose the right first action?
The labels cover the eight agents and a refusal action. The runtime supervisor
continues to route through structured LLM decisions and host-side compatibility checks.
[Experiment and methods](../results/week13-routing.md)

## Compare the encodings on the same requests

The baseline used 225 training requests and a 72-request synthetic holdout. Both models
used logistic regression. One represented text with TF-IDF word features. The other used
a frozen MiniLM encoder. Each classifier received the same eligible labels for a given
input type.

| Representation | Correct routes | DataFrame subset | Median warm CPU prediction |
| --- | ---: | ---: | ---: |
| TF-IDF | 51/72 | 36/51 | 0.41 ms |
| MiniLM | 57/72 | 40/51 | 5.16 ms |

MiniLM got 14 requests right that TF-IDF missed and missed eight that TF-IDF got right.
The exact paired test gave p = 0.286. The six-case improvement alone gives limited
evidence that switching encoders helps on new requests.

## Keep the encoder fixed and add examples

The follow-up kept the MiniLM weights and classifier settings fixed. It added 144
individually written training requests, 16 for each label. The fresh 72-request holdout
was written before the additions. Normalized duplicate checks verified that its
requests did not appear in training.

| Training set | Fresh holdout | Accuracy |
| --- | ---: | ---: |
| 225 requests | 57/72 | 79.2% |
| 369 requests | 67/72 | 93.1% |

Ten requests improved and none regressed on that holdout. The exact paired test gave
p = 0.00195. Examples covered more varied phrasing, and the sample size increased at the
same time. The experiment cannot tell how much of the gain came from each change.

Wrangling improved from three to six correct routes out of eight, and refusals improved
from four to six out of eight. The expanded classifier still routed an external texting
request to EDA and a database deletion request to SQL. The routing test measures the
choice of label. DataQuorum's SQL execution layer separately enforces read-only queries.
[First-step failures and limits](../results/week13-routing.md)

## Reproduce the comparison

Run this from the repository root in PowerShell after caching the public encoder as
described in the Week 13 results. Both runs are local and use zero model API allowance.

```powershell
uv run --locked --extra routing python research/router_benchmark/compare_training.py
```

The improvement applies to a synthetic first-step holdout. An evaluation on independent
user requests and multi-step runs would measure a different outcome.
