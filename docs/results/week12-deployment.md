# Week 12 model deployment

A fitted forest now survives the training container. The archive carries the forest, feature
order and dtypes, tuning settings, and cross-validation score. Before saving it, the host
reloads it and checks predictions against the sandbox on up to 32 held-out rows.

## Local integration checks

All 361 local tests passed, including 41 Docker tests. Ruff and strict mypy passed.
The documented Uvicorn command also ran on loopback. It reported the model ready and
returned the same predictions as the registry-loaded model for three synthetic rows.
MLflow emitted a SQLAlchemy deprecation warning during registry initialization.

| Check | Expected result |
|---|---|
| Classification and regression export | Reloaded predictions match the training sandbox |
| Corrupted archive | Model digest mismatch is rejected |
| Two registrations under one name | MLflow creates versions 1 and 2 and records the CV score |
| Prediction requests | FastAPI serves the selected version, accepts reordered features, and rejects invalid rows |
| Unchanged drift control | Neither feature is flagged |
| Shifted drift batch | Evidently flags only the feature that changed |

The first export tests stopped at the skops trust check. The fitted random forest uses
scikit-learn's compiled tree type. The loader now allows that specific type for models
produced by the local workflow. Callers supply trusted artifacts from their training
pipeline. The digest detects changes to the model payload relative to the manifest.

Evidently required Plotly 5, so its locked script runs in a separate environment. The
drift comparison used 400 seeded observations with two numeric features. The control
compared the baseline with itself. The shifted batch added six to `signal` and kept
`noise` identical. Evidently flagged 0 of 2 columns in the control and 1 of 2 in the
shifted batch. For the shifted batch, its Kolmogorov–Smirnov test returned a p-value of
about 1.06 × 10⁻²³⁹ for `signal` and 1 for `noise`. This checks that the report
distinguishes a large planted shift from identical data.

## Reproduce the deployment test

Run these commands in PowerShell from the repository root.

```powershell
uv sync --locked --extra serving
docker build -t dataquorum/sandbox:0.4 -f images/sandbox/Dockerfile .
uv run --locked --extra serving pytest -q tests/sandbox/test_model_deployment.py
```

Use the local API and drift commands in the [README](../../README.md#register-and-serve-a-trained-model)
to repeat the workflow on your data. This phase uses the sandbox random forest and a
local SQLite registry. The serving process binds to loopback and loads one version at
startup. No LLM calls are involved.
