"""Structured LLM routing with exact agent names and explicit refusals."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from dataquorum.agents import AGENTS
from dataquorum.core.engine import StructuredLLM
from dataquorum.core.llm import LLMRefusal, StructuredResult
from dataquorum.core.state import AgentResult
from dataquorum.sandbox.executor import InputKind

Route = Literal[
    "data_cleaning",
    "data_wrangling",
    "data_visualization",
    "data_loader",
    "sql_database",
    "eda",
    "feature_engineering",
    "ml_correctness",
    "finish",
]


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    next: Route
    instructions: str
    reason: str
    refusal: str | None

    @field_validator("refusal")
    @classmethod
    def refusal_has_reason(cls, refusal: str | None) -> str | None:
        if refusal is not None and not refusal.strip():
            raise ValueError("Provide a refusal explanation or null")
        return refusal


_DESCRIPTIONS = {
    "data_cleaning": "Repair messy values, types, missing entries, outliers or duplicates.",
    "data_wrangling": "Filter, join within the table, reshape or compute requested aggregations.",
    "data_visualization": "Create a chart from a table.",
    "data_loader": "Load and combine data from a folder of files into a table.",
    "sql_database": "Read from a SQLite or DuckDB database using SQL.",
    "eda": "Explore an existing table and report descriptive statistics or findings.",
    "feature_engineering": "Transform predictors for modeling while preserving the target.",
    "ml_correctness": "Audit leakage, split integrity, validation strategy or metric choice.",
}
_CATALOG = "\n".join(
    f"- {name} ({spec.input_kind}): {_DESCRIPTIONS[name]}" for name, spec in AGENTS.items()
)
_SYSTEM = f"""Route one step of a data-science workflow.
Choose the next specialist among the eligible agents. The selected agent receives the exact
original user request and your step instructions separately. Specify only work this agent
should do now. Keep all user constraints and dependencies. The active input becomes the
latest validated table if one is produced. A chart does not replace the active table.
If the request cannot be completed with the available input or capabilities, set refusal
to a clear explanation and next to finish. Otherwise set refusal to null. Only choose
finish when completed steps fulfill the request. Return the structured decision.

{_CATALOG}"""


class LLMRouter:
    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def route(
        self,
        request: str,
        kind: InputKind,
        summary: str,
        completed: tuple[AgentResult, ...] = (),
    ) -> StructuredResult[RouteDecision]:
        eligible = [name for name, spec in AGENTS.items() if spec.input_kind == kind]
        history = [
            {
                "agent": result.agent,
                "status": result.status,
                "output": result.output.kind if result.output else None,
                "steps": result.steps,
                "note": result.error,
            }
            for result in completed
        ]
        user = (
            f"User request (verbatim):\n{request}\n\nCurrent input kind: {kind}\n"
            f"Eligible agents: {', '.join(eligible)}\n"
            f"Current input summary:\n{summary}\n\nCompleted steps:\n{json.dumps(history)}"
        )
        try:
            return self.llm.structured(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
                RouteDecision,
            )
        except LLMRefusal as exc:
            return StructuredResult(
                value=RouteDecision(
                    next="finish", instructions="", reason="Provider refusal.", refusal=str(exc)
                ),
                usage=exc.usage,
                model=exc.usage.models[-1] if exc.usage.models else "unknown",
            )
