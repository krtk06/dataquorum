import pytest

from dataquorum.core.statuses import CheckResult, Severity, Status, decide_status

OK_BLOCKING = CheckResult(name="has_rows", passed=True, severity=Severity.BLOCKING)
BAD_BLOCKING = CheckResult(
    name="has_rows", passed=False, severity=Severity.BLOCKING, detail="0 rows"
)
BAD_WARNING = CheckResult(
    name="row_loss", passed=False, severity=Severity.WARNING, detail="lost 60% of rows"
)


@pytest.mark.parametrize(
    ("kwargs", "status", "error_part"),
    [
        (
            {"has_output": True, "output_empty": False, "checks": [OK_BLOCKING]},
            Status.SUCCESS,
            None,
        ),
        ({"has_output": True, "output_empty": False, "checks": []}, Status.SUCCESS, None),
        ({"has_output": False, "output_empty": True, "checks": []}, Status.FAILED, "No output"),
        ({"has_output": True, "output_empty": True, "checks": []}, Status.FAILED, "empty"),
        (
            {"has_output": True, "output_empty": False, "checks": [], "error": "KeyError: 'x'"},
            Status.FAILED,
            "KeyError",
        ),
        (
            {"has_output": True, "output_empty": False, "checks": [BAD_BLOCKING, BAD_WARNING]},
            Status.FAILED,
            "has_rows (0 rows)",
        ),
        (
            {"has_output": True, "output_empty": False, "checks": [OK_BLOCKING, BAD_WARNING]},
            Status.NEEDS_REVIEW,
            "row_loss (lost 60% of rows)",
        ),
        (
            {
                "has_output": True,
                "output_empty": False,
                "checks": [OK_BLOCKING],
                "review_reason": "User asked to approve the plan.",
            },
            Status.NEEDS_REVIEW,
            "approve the plan",
        ),
    ],
)
def test_decide_status(kwargs: dict[str, object], status: Status, error_part: str | None) -> None:
    decision = decide_status(**kwargs)  # type: ignore[arg-type]
    assert decision.status is status
    if error_part is None:
        assert decision.error is None
    else:
        assert decision.error is not None and error_part in decision.error


def test_error_wins_over_passing_checks() -> None:
    decision = decide_status(
        has_output=True, output_empty=False, checks=[OK_BLOCKING], error="timeout"
    )
    assert decision.status is Status.FAILED
