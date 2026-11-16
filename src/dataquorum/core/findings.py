"""Grounded findings: plain-English takeaways that may only quote numbers from the output.

Used by agents with `explain_findings=True` (EDA). After the code has run and passed its
checks, the LLM reads the output table and writes 2-5 findings. `ungrounded_numbers` then
checks every number in the text against the table. A number is grounded if it equals some
table value after rounding to the decimals it was written with ("38.4" matches 38.42), either
as is or as a percentage of a 0-1 value ("25%" matches 0.2531). Small integers (<= 12) are
skipped ("top 3", "5 groups"). Ungrounded numbers become a warning, so the run ends
`needs_review` instead of passing on invented statistics.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from dataquorum.core.llm import Message
from dataquorum.core.statuses import CheckResult, Severity


class Findings(BaseModel):
    findings: list[str] = Field(
        description="2 to 5 short findings, each quoting numbers from the table."
    )


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?")

SYSTEM = """You summarise an analysis table for a busy reader.
Write 2 to 5 findings, one sentence each, that answer the user's request.
Rules:
- Quote only numbers that appear in the table. You may round them (keep at least 2
  significant digits) and write a fraction like 0.25 as 25%.
- Do not compute new numbers (no differences, ratios or totals that are not in the table).
- Do not guess at causes. Name columns and groups exactly as they appear."""


def findings_messages(request: str, table: pd.DataFrame, max_rows: int = 60) -> list[Message]:
    shown = table.head(max_rows).to_csv(index=False)
    note = f"(first {max_rows} of {len(table)} rows)" if len(table) > max_rows else ""
    user = f"User request (verbatim):\n{request}\n\nAnalysis table {note}:\n{shown}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _parse(token: str) -> float:
    return float(token.rstrip("%").replace(",", ""))


def table_numbers(table: pd.DataFrame) -> np.ndarray:
    """Every number in the table: numeric cells, numbers inside text cells, and the row count."""
    values: list[float] = [float(len(table))]
    for col in table.columns:
        series = table[col]
        if pd.api.types.is_numeric_dtype(series):
            values += [float(v) for v in series.dropna()]
        else:
            for cell in series.dropna().astype(str):
                values += [_parse(n) for n in _NUMBER.findall(cell)]
    return np.asarray(values, dtype=float)


def _grounded(token: str, known: np.ndarray) -> bool:
    body = token.rstrip("%")
    decimals = len(body.split(".")[1]) if "." in body else 0
    half_unit = 0.5 * 10.0**-decimals + 1e-9
    value = _parse(token)
    as_is = np.abs(known - value) <= half_unit
    as_percent = np.abs(known * 100 - value) <= half_unit
    return bool(np.any(as_is) or np.any(as_percent))


def ungrounded_numbers(findings: list[str], table: pd.DataFrame) -> list[str]:
    known = table_numbers(table)
    missing = []
    for text in findings:
        for token in _NUMBER.findall(text):
            value = _parse(token)
            if value.is_integer() and abs(value) <= 12:
                continue
            if not _grounded(token, known):
                missing.append(token)
    return missing


def grounding_check(findings: list[str], table: pd.DataFrame) -> CheckResult:
    missing = ungrounded_numbers(findings, table)
    detail = (
        "every number appears in the table"
        if not missing
        else f"numbers not in the table: {missing}"
    )
    return CheckResult(
        name="findings_grounded", passed=not missing, severity=Severity.WARNING, detail=detail
    )
