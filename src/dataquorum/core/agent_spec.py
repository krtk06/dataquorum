"""What an agent supplies to the engine: prompts and checks, nothing else (spec S6).

The engine owns the graph, retries, review, execution and status. An agent is just data:

    CLEANING = AgentSpec(
        name="data_cleaning",
        role="You are a careful data-cleaning agent ...",
        recommend_guidance="Recommend cleaning steps ...",
        code_guidance="Prefer vectorised pandas ...",
        extra_checks=(Check("no_duplicate_rows", Severity.WARNING, ...),),
    )

`output_kind` picks the generated function (`transform` or `make_figure`), how the result
is stored, and the default generic checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataquorum.core.llm import ReasoningEffort
from dataquorum.core.validation import GENERIC_DATAFRAME_CHECKS, GENERIC_FIGURE_CHECKS, Check
from dataquorum.sandbox.executor import InputKind, Language, OutputKind, function_name


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str  # who the agent is; first line of every system prompt
    recommend_guidance: str  # what good recommended steps look like for this agent
    code_guidance: str  # agent-specific coding rules
    output_kind: OutputKind = "dataframe"
    input_kind: InputKind = "dataframe"  # dataframe, a folder of files, or a database file
    language: Language = "python"  # sql only for database input
    extra_checks: tuple[Check, ...] = ()
    generic_checks: tuple[Check, ...] | None = None  # None = defaults for output_kind
    reasoning_effort: ReasoningEffort | None = None  # None = LLMConfig default
    explain_findings: bool = False  # EDA: write findings grounded in the output table
    verify_findings: bool = True

    @property
    def function_name(self) -> str:
        return function_name(self.input_kind, self.output_kind)

    @property
    def checks(self) -> tuple[Check, ...]:
        if self.generic_checks is not None:
            generic = self.generic_checks
        elif self.output_kind == "figure":
            generic = GENERIC_FIGURE_CHECKS
        else:
            generic = GENERIC_DATAFRAME_CHECKS
        return generic + self.extra_checks
