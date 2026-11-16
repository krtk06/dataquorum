"""What an agent supplies to the engine: prompts and checks, nothing else (spec S6).

The engine owns the graph, retries, review, execution and status. An agent is just data:

    CLEANING = AgentSpec(
        name="data_cleaning",
        role="You are a careful data-cleaning agent ...",
        recommend_guidance="Recommend cleaning steps ...",
        code_guidance="Prefer vectorised pandas ...",
        extra_checks=(Check("no_duplicate_rows", Severity.WARNING, ...),),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataquorum.core.llm import ReasoningEffort
from dataquorum.core.validation import GENERIC_DATAFRAME_CHECKS, Check


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str  # who the agent is; first line of every system prompt
    recommend_guidance: str  # what good recommended steps look like for this agent
    code_guidance: str  # agent-specific coding rules
    extra_checks: tuple[Check, ...] = ()
    generic_checks: tuple[Check, ...] = GENERIC_DATAFRAME_CHECKS
    reasoning_effort: ReasoningEffort | None = None  # None = LLMConfig default
    tags: tuple[str, ...] = field(default=())

    @property
    def checks(self) -> tuple[Check, ...]:
        return self.generic_checks + self.extra_checks
