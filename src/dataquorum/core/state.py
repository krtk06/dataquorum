"""The result contract every agent returns (spec S2).

`AgentResult` is the single shape that agents, the supervisor, the run recorder and the
Studio all read. Two rules are enforced by validation, so an inconsistent result cannot
even be constructed:

- `success` needs a non-empty output, no failed blocking check, and no error.
- Any other status needs an `error` explaining why.

Build results with `AgentResult.from_run(...)`, which computes the status with
`decide_status`. Constructing `AgentResult(status=...)` directly is allowed (for loading
saved runs) but is still checked against the rules above.

Outputs are references, not in-memory objects: tables are Parquet files, models are URIs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from dataquorum.core.cost import Usage
from dataquorum.core.statuses import CheckResult, Severity, Status, decide_status


class DataFrameOutput(BaseModel):
    kind: Literal["dataframe"] = "dataframe"
    path: Path  # Parquet file
    rows: int
    columns: list[str]
    dtypes: dict[str, str]

    @property
    def is_empty(self) -> bool:
        return self.rows == 0 or not self.columns

    @classmethod
    def save(cls, df: pd.DataFrame, path: Path) -> DataFrameOutput:
        """Write `df` to Parquet and describe it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return cls(
            path=path,
            rows=len(df),
            columns=[str(c) for c in df.columns],
            dtypes={str(c): str(t) for c, t in df.dtypes.items()},
        )

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)


class FigureOutput(BaseModel):
    kind: Literal["figure"] = "figure"
    figure: dict[str, Any]  # Plotly figure as JSON (fig.to_dict())

    @property
    def is_empty(self) -> bool:
        return not self.figure.get("data")


class ModelOutput(BaseModel):
    kind: Literal["model"] = "model"
    uri: str  # file path or MLflow model URI
    framework: str  # e.g. "sklearn", "h2o", "autogluon"
    metrics: dict[str, float] = {}

    @property
    def is_empty(self) -> bool:
        return not self.uri


class TextOutput(BaseModel):
    kind: Literal["text"] = "text"
    text: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


Output = Annotated[
    DataFrameOutput | FigureOutput | ModelOutput | TextOutput, Field(discriminator="kind")
]


def _new_run_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentResult(BaseModel):
    run_id: str = Field(default_factory=_new_run_id)
    parent_run_id: str | None = None  # set for agents run by the supervisor
    agent: str
    status: Status
    output: Output | None = None
    code: str | None = None  # final code that produced the output
    steps: list[str] = []  # recommended steps the code implements
    validation: list[CheckResult] = []
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None
    message: str = ""
    created_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _status_matches_facts(self) -> AgentResult:
        if self.status is Status.SUCCESS:
            if self.output is None or self.output.is_empty:
                raise ValueError("A success result needs a non-empty output.")
            if self.error:
                raise ValueError("A success result cannot carry an error.")
            if any(not c.passed and c.severity is Severity.BLOCKING for c in self.validation):
                raise ValueError("A success result cannot have failed blocking checks.")
        elif not self.error:
            raise ValueError(f"A {self.status} result needs an error explaining why.")
        if self.status is Status.NEEDS_REVIEW and (self.output is None or self.output.is_empty):
            raise ValueError("A needs_review result needs a non-empty output to review.")
        return self

    @classmethod
    def from_run(
        cls,
        *,
        agent: str,
        output: DataFrameOutput | FigureOutput | ModelOutput | TextOutput | None,
        checks: list[CheckResult] | None = None,
        error: str | None = None,
        review_reason: str | None = None,
        code: str | None = None,
        steps: list[str] | None = None,
        usage: Usage | None = None,
        message: str | None = None,
        parent_run_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentResult:
        """Build a result whose status is decided from the facts, never passed in."""
        checks = checks or []
        decision = decide_status(
            has_output=output is not None,
            output_empty=output.is_empty if output is not None else True,
            checks=checks,
            error=error,
            review_reason=review_reason,
        )
        return cls(
            run_id=run_id or _new_run_id(),
            agent=agent,
            parent_run_id=parent_run_id,
            status=decision.status,
            output=output,
            code=code,
            steps=steps or [],
            validation=checks,
            usage=usage or Usage(),
            error=decision.error,
            message=message or _default_message(agent, decision.status, decision.error),
        )


def _default_message(agent: str, status: Status, error: str | None) -> str:
    if status is Status.SUCCESS:
        return f"{agent} finished and all checks passed."
    if status is Status.NEEDS_REVIEW:
        return f"{agent} produced a result that needs review. {error}"
    return f"{agent} failed. {error}"
