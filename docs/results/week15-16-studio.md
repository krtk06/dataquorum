# Weeks 15–16 Studio

DataQuorum Studio is a local Streamlit app for composing and inspecting agent pipelines.
It opens with a retail-orders demo that cleans 192 rows, summarizes revenue by region,
and draws a verified Plotly chart. The demo runs checked-in transformations through the
existing validation functions. It uses zero model calls.

| View | Workflow |
|---|---|
| Pipeline | Add or remove steps, select a node, edit its request, and link it to the source or an earlier table. Save and load versioned JSON recipes or export runnable Python. |
| Data | Load CSV or Parquet, upload a SQLite or DuckDB database file, or stage several files for the loader agent. Inspect row counts, missing values, column types, and verified tables. Download CSV. |
| Charts | View and export pipeline figures. When a pipeline has no figure, preview a bar, line, or scatter chart from a table. |
| Model | Audit targets and splits, then run Optuna tuning, SHAP, or binary calibration with the existing sandbox. Export a fitted forest and register a local MLflow version with the serving extra. |
| Runs | Inspect the current pipeline's outcomes and browse up to 200 recorded DuckDB runs with status, allowance, retries, checks, and generated code. |

## Design and run behavior

An ordered recipe can branch from one source. Each agent takes one input. Table-producing
steps can feed later steps. A failed or review-flagged result blocks its dependents and
independent branches still run. Step code and findings appear with the status awarded by the
existing engine. The app accepts user requests as data and sends generated Python through
Docker. SQL uses the existing read-only executor.

A recipe JSON file stores only the pipeline definition. It has no uploaded data, secrets,
generated code, or results. Opening it validates the schema and dependencies. The Python
export calls the same live pipeline runner against a file or directory provided on the
command line. It generates fresh results using the configured model. It carries the
recipe forward without packaging the source or previous results.

Uploaded folders and databases are staged under `runs/studio/sources`. Model exports go
under `runs/studio/models`. These paths and the DuckDB trace database are git-ignored.
Each browser session has its own active recipe and source. The launch command binds to
`127.0.0.1`.

## Start in PowerShell

1. From the repository root, install the Studio dependency group.

   ```powershell
   uv sync --locked --extra studio
   ```

2. Start Studio.

   ```powershell
   uv run --locked --extra studio dataquorum-studio
   ```

3. Open `http://127.0.0.1:8501` and select **Run pipeline**. View the result in **Charts**
   and the validated tables in **Data**. Open **Step results** for the checks and
   **Files & export** for the JSON recipe or runnable Python.
   [Captioned walkthrough](../media/studio-demo.mp4)

For live agents, start Docker Desktop, build sandbox 0.4 with the README command, and
open **Demo mode** to enable **Live agents**. Go reads its key from the existing environment or `.env`. Live
runs send the agent's data preview to Go and use allowance. Local model reports also
require Docker. Add `--extra serving` when starting Studio for forest export and MLflow
registration. Use `dataquorum-studio --port 8502` if the default port is occupied.

## Verification

All 396 local tests passed, including 41 Docker tests and 11 Studio tests. Ruff,
strict mypy, and the dependency lock check passed. Strict mypy also passed in a
base environment with Studio and serving extras absent.

The automated checks cover recipe validation, script syntax, safe upload names, demo
results, branch blocking, and a Streamlit walkthrough across all five views. Browser
verification selected a node, created a step, downloaded both export formats, reopened
the saved recipe, and uploaded a two-row CSV. It verified the demo Plotly chart and ran
the tuning report on synthetic model data through Docker. Step creation now uses an
inline form because the initial dialog could close during a canvas rerun. The live
agent UI uses the existing engine and recorder. This Studio verification used no
paid model calls.

### Studio redesign

Studio uses warm charcoal, off-white, and a muted brass accent for primary actions.
The top navigation connects all five views. The workflow diagram highlights the path
that feeds a selected step. On narrow screens it defaults to a list that names each
step's input and links to the inspector below it. The trace and inspector transitions
respect reduced motion.

Edge verified the offline demo, consecutive keyboard and mouse step selections,
diagram zoom, layouts from 320 to 1440 CSS pixels, both export formats, recipe import
with Unicode names, CSV upload, and a Docker tuning report. Impeccable's rendered
scan reported zero findings. The full suite passed 396 tests and the 11 Studio tests
passed again after the final review fixes. The built wheel contains the CSS and
JavaScript required for the workflow component.
