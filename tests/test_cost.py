from datetime import UTC, datetime

import pytest

from dataquorum.core.cost import (
    UnknownModelPriceError,
    Usage,
    estimate_list_cost,
    is_peak,
    usage_from_response,
)

MONDAY_PEAK = datetime(2026, 11, 16, 7, 30, tzinfo=UTC)
MONDAY_OFF_PEAK = datetime(2026, 11, 16, 5, 30, tzinfo=UTC)
SATURDAY_MORNING = datetime(2026, 11, 21, 7, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (MONDAY_PEAK, True),
        (MONDAY_OFF_PEAK, False),
        (SATURDAY_MORNING, False),
        (datetime(2026, 11, 16, 1, 0, tzinfo=UTC), True),
        (datetime(2026, 11, 16, 4, 0, tzinfo=UTC), False),
        (datetime(2026, 11, 16, 9, 59, tzinfo=UTC), True),
        (datetime(2026, 11, 16, 10, 0, tzinfo=UTC), False),
    ],
)
def test_is_peak(when: datetime, expected: bool) -> None:
    assert is_peak(when) is expected


def test_deepseek_list_cost_matches_observed_smoke_test() -> None:
    # Smoke test 2026-11-16 off-peak: 75 in, 3,729 out -> Go list_cost 0.00224865.
    cost = estimate_list_cost("deepseek-v4.1-flash", 75, 3729, 0, MONDAY_OFF_PEAK)
    assert cost == pytest.approx(0.00224865)


def test_deepseek_peak_doubles_and_cache_is_cheaper() -> None:
    off = estimate_list_cost("deepseek-v4.1-flash", 1000, 0, 0, MONDAY_OFF_PEAK)
    peak = estimate_list_cost("deepseek-v4.1-flash", 1000, 0, 0, MONDAY_PEAK)
    cached = estimate_list_cost("deepseek-v4.1-flash", 1000, 0, 1000, MONDAY_OFF_PEAK)
    assert peak == pytest.approx(2 * off)
    assert cached < off / 10


def test_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelPriceError):
        estimate_list_cost("some-new-model", 1, 1, 0, MONDAY_OFF_PEAK)


def test_usage_prefers_go_reported_costs() -> None:
    raw = {
        "prompt_tokens": 75,
        "completion_tokens": 3729,
        "completion_tokens_details": {"reasoning_tokens": 2919},
        "prompt_tokens_details": {"cached_tokens": 0},
        "opencode": {"list_cost": 0.00224865, "allowance": 0.00134919},
    }
    usage = usage_from_response("deepseek-v4.1-flash", raw, seconds=34.2, when=MONDAY_PEAK)
    assert usage.reasoning_tokens == 2919
    assert usage.list_cost_usd == pytest.approx(0.00224865)  # not doubled: Go's number wins
    assert usage.allowance_usd == pytest.approx(0.00134919)
    assert usage.calls == 1


def test_usage_falls_back_to_price_table_with_allowance_ratio() -> None:
    raw = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
    usage = usage_from_response("deepseek-v4.1-flash", raw, seconds=1, when=MONDAY_OFF_PEAK)
    assert usage.list_cost_usd == pytest.approx(0.15)
    assert usage.allowance_usd == pytest.approx(0.09)


def test_usage_add_combines_counts_and_models() -> None:
    a = Usage(models=["m1"], prompt_tokens=1, completion_tokens=2, calls=1, allowance_usd=0.1)
    b = Usage(models=["m1", "m2"], prompt_tokens=3, retries=1, calls=1, allowance_usd=0.2)
    total = a + b
    assert total.models == ["m1", "m2"]
    assert (total.prompt_tokens, total.completion_tokens) == (4, 2)
    assert (total.calls, total.retries) == (2, 1)
    assert total.allowance_usd == pytest.approx(0.3)
