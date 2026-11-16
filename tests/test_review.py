import pytest

from dataquorum.core.review import Approve, Reject, Revise, parse_review


@pytest.mark.parametrize(
    "text",
    ["yes", "Yes.", "YES!", "y", "ok", "Okay", "approve", "Approved.", "LGTM", "looks good",
     "Go ahead", "yes please", "Yes, thanks!", "sounds good", "do it"],
)  # fmt: skip
def test_approvals(text: str) -> None:
    assert parse_review(text) == Approve()


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("no", None),
        ("Reject", None),
        ("no: wrong columns", "wrong columns"),
        ("Reject - the plan drops data", "the plan drops data"),
        ("stop", None),
    ],
)
def test_rejections(text: str, reason: str | None) -> None:
    decision = parse_review(text)
    assert isinstance(decision, Reject)
    if reason:
        assert decision.reason == reason


@pytest.mark.parametrize(
    "text",
    [
        "Yes, but keep the age column",  # an approval word followed by real instructions
        "Use lower case for city",
        "nothing should be dropped",  # starts with "no" but is not a rejection
        "okay so also parse dates",
    ],
)
def test_everything_else_is_a_revision(text: str) -> None:
    assert parse_review(text) == Revise(instructions=text)


def test_structured_inputs() -> None:
    assert parse_review({"kind": "approve"}) == Approve()
    assert parse_review({"kind": "revise", "instructions": "x"}) == Revise(instructions="x")
    assert parse_review(Reject(reason="r")) == Reject(reason="r")


def test_empty_and_wrong_types_raise() -> None:
    with pytest.raises(ValueError):
        parse_review("  ")
    with pytest.raises(TypeError):
        parse_review(42)
