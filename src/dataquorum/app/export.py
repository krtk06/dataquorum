"""Export a Studio pipeline as a stand-alone Python entry script."""

from dataquorum.app.pipeline_model import Pipeline


def export_script(pipeline: Pipeline) -> str:
    return f'''"""Run an exported DataQuorum pipeline against a local source."""
import argparse
from pathlib import Path

from dataquorum.app.pipeline_model import Pipeline, run_live_pipeline
from dataquorum.app.sources import open_source
from dataquorum.core.statuses import Status

recipe = Pipeline.model_validate_json({pipeline.model_dump_json()!r})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="CSV, Parquet, database file or folder")
    args = parser.parse_args()
    run = run_live_pipeline(recipe, open_source(args.source))
    for step in run.steps:
        detail = step.result.message if step.result else step.blocked
        print(f"{{step.step.label}}: {{step.status}} | {{detail}}")
        if step.result and step.result.output:
            output = step.result.output
            if output.kind == "dataframe":
                print(f"  Table: {{output.path}}")
    print(f"Pipeline: {{run.status}}")
    raise SystemExit({{Status.SUCCESS: 0, Status.NEEDS_REVIEW: 1, Status.FAILED: 2}}[run.status])
'''
