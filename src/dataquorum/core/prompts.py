"""Prompt building for the engine.

Layout rule (for prompt caching): the system message is fixed per agent and step, so it is
identical across runs and gets cached. Everything that changes (request, data summary,
review notes, errors) goes in the user message, after the fixed text.

The user's request is always passed **unchanged** in its own section. Reviewer notes and
fix feedback are separate sections and are never merged into it (bug log #13).
"""

from __future__ import annotations

import pandas as pd

from dataquorum.core.agent_spec import AgentSpec
from dataquorum.core.llm import Message
from dataquorum.core.validation import missing_like
from dataquorum.sandbox.executor import FUNCTION_NAME

_SHARED_CODE_RULES = f"""Code rules:
- Define exactly one public function: `{FUNCTION_NAME}(df: pd.DataFrame) -> pd.DataFrame`.
- Import only pandas and numpy. Put imports at the top of the code.
- Do not read or write files, use the network, print, or call input().
- Do not modify `df` in place; work on a copy and return a new DataFrame.
- The code runs in a sandbox with no network and a time limit."""


def summarize_frame(df: pd.DataFrame, max_rows: int = 8) -> str:
    """Deterministic data summary for prompts (no LLM involved)."""
    lines = [f"Shape: {len(df)} rows x {len(df.columns)} columns", "Columns:"]
    for col in df.columns:
        series = df[col]
        info = f"- {col}: dtype={series.dtype}, missing_or_placeholder={missing_like(series)}"
        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            info += f", distinct={series.nunique(dropna=True)}"
        lines.append(info)
    sample = df.head(max_rows).to_dict(orient="records")
    lines.append(f"First {min(max_rows, len(df))} rows: {sample}")
    return "\n".join(lines)


def _context(request: str, data_summary: str, review_notes: list[str]) -> str:
    parts = [f"User request (verbatim):\n{request}", f"Data summary:\n{data_summary}"]
    if review_notes:
        notes = "\n".join(f"{i}. {n}" for i, n in enumerate(review_notes, 1))
        parts.append(f"Reviewer notes, in order (apply all, later notes win on conflict):\n{notes}")
    return "\n\n".join(parts)


def recommend_messages(
    spec: AgentSpec, request: str, data_summary: str, review_notes: list[str]
) -> list[Message]:
    system = (
        f"{spec.role}\n\n{spec.recommend_guidance}\n\n"
        "Return a short ordered list of concrete steps. Each step should be one action a "
        "pandas function can perform. Do not include code."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _context(request, data_summary, review_notes)},
    ]


def code_messages(
    spec: AgentSpec,
    request: str,
    data_summary: str,
    review_notes: list[str],
    steps: list[str],
    previous_code: str | None = None,
    problem: str | None = None,
) -> list[Message]:
    system = f"{spec.role}\n\n{_SHARED_CODE_RULES}\n\n{spec.code_guidance}"
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    user = _context(request, data_summary, review_notes) + f"\n\nSteps to implement:\n{numbered}"
    if previous_code is not None and problem:
        user += (
            f"\n\nYour previous code did not work.\nProblem: {problem}\n\n"
            f"Previous code:\n{previous_code}\n\nReturn corrected code."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
