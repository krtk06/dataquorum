"""Serve one selected model version over a local FastAPI endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from dataquorum.ml.registry import ForestRegistry

if TYPE_CHECKING:
    from fastapi import FastAPI


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[dict[str, StrictInt | StrictFloat]] = Field(min_length=1, max_length=1000)


def create_app(registry: ForestRegistry, name: str, version: str) -> FastAPI:
    from fastapi import FastAPI, HTTPException

    model = registry.load(name, version)
    metadata = model.metadata
    features = set(metadata.features)
    app = FastAPI(title="DataQuorum Model Serving")

    def health() -> dict[str, str]:
        return {"status": "ready", "model": name, "version": version}

    def schema() -> dict[str, object]:
        return {
            "model": name,
            "version": version,
            "problem": metadata.problem,
            "target": metadata.target,
            "features": dict(zip(metadata.features, metadata.training_dtypes, strict=True)),
        }

    def predict(request: PredictionRequest) -> dict[str, object]:
        if any(set(row) != features for row in request.rows):
            raise HTTPException(status_code=422, detail=f"Expected features {metadata.features}")
        try:
            predictions = model.predict(pd.DataFrame(request.rows)).tolist()
        except (ValueError, OverflowError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"model": name, "version": version, "predictions": predictions}

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/schema", schema, methods=["GET"])
    app.add_api_route("/predict", predict, methods=["POST"])
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("version")
    parser.add_argument("--registry", type=Path, default=Path("runs/registry"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(ForestRegistry(args.registry), args.name, args.version)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
