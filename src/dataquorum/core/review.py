"""Typed human-review decisions (spec S6).

The engine pauses with LangGraph's `interrupt()` and resumes with one of three decisions.
A UI may send free text; `parse_review` turns it into a decision. Unlike the original
(bug log #7), "Yes." or "yes please" means approve, not "please change the code to 'Yes.'".

Text rules, after trimming and lower-casing:
- An approval word, optionally followed by punctuation or a polite tail, -> Approve
  ("yes", "y", "ok", "approve", "approved", "lgtm", "looks good", "go ahead", ...)
- "reject", "no", "stop", "cancel", optionally with ": reason" -> Reject
- Anything else is treated as revision instructions -> Revise(text)
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class Approve(BaseModel):
    kind: Literal["approve"] = "approve"


class Revise(BaseModel):
    kind: Literal["revise"] = "revise"
    instructions: str = Field(min_length=1)


class Reject(BaseModel):
    kind: Literal["reject"] = "reject"
    reason: str = "Rejected by the reviewer."


ReviewDecision = Annotated[Approve | Revise | Reject, Field(discriminator="kind")]
_ADAPTER: TypeAdapter[Approve | Revise | Reject] = TypeAdapter(ReviewDecision)

_APPROVE = re.compile(
    r"^(yes|y|yep|yeah|ok|okay|approve|approved|lgtm|looks good|go ahead|proceed|"
    r"sounds good|do it)\b[\s!.,]*(please|thanks|thank you)?[\s!.]*$"
)
_REJECT = re.compile(r"^(reject|rejected|no|nope|stop|cancel|abort)\b[\s!.,:;-]*(?P<reason>.*)$")


def parse_review(value: Any) -> Approve | Revise | Reject:
    """Accept a decision object, a dict like {"kind": "revise", ...}, or free text."""
    if isinstance(value, Approve | Revise | Reject):
        return value
    if isinstance(value, dict):
        return _ADAPTER.validate_python(value)
    if not isinstance(value, str):
        raise TypeError(f"Cannot read a review decision from {type(value).__name__}.")

    text = value.strip()
    lowered = text.lower()
    if not lowered:
        raise ValueError("Empty review reply: approve, reject, or say what to change.")
    if _APPROVE.match(lowered):
        return Approve()
    rejected = _REJECT.match(lowered)
    if rejected:
        reason = text[len(text) - len(rejected.group("reason")) :].strip()
        return Reject(reason=reason) if reason else Reject()
    return Revise(instructions=text)
