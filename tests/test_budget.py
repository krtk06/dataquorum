from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dataquorum.core.budget import (
    BudgetExceededError,
    BudgetGuard,
    BudgetLimits,
    BudgetWarning,
    InMemoryLedger,
    JsonlLedger,
    SpendEntry,
)
from dataquorum.core.cost import Usage

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


def guard(ledger: InMemoryLedger, mode: str = "eval") -> BudgetGuard:
    return BudgetGuard(
        ledger,
        mode="eval" if mode == "eval" else "interactive",
        limits=BudgetLimits(per_5_hours_usd=1.0, per_week_usd=2.0),
        clock=lambda: NOW,
    )


def test_under_limits_passes() -> None:
    ledger = InMemoryLedger()
    ledger.append(SpendEntry(NOW - timedelta(hours=1), 0.5, "m"))
    guard(ledger).check()


def test_five_hour_window_only_counts_recent_spend() -> None:
    ledger = InMemoryLedger()
    ledger.append(SpendEntry(NOW - timedelta(hours=6), 0.9, "m"))  # outside 5 h window
    ledger.append(SpendEntry(NOW - timedelta(hours=1), 0.5, "m"))
    g = guard(ledger)
    five_h, week = g.spent()
    assert five_h == pytest.approx(0.5)
    assert week == pytest.approx(1.4)
    g.check()


def test_eval_mode_raises_when_five_hour_limit_reached() -> None:
    ledger = InMemoryLedger()
    ledger.append(SpendEntry(NOW - timedelta(minutes=5), 1.0, "m"))
    with pytest.raises(BudgetExceededError, match="in 5 h"):
        guard(ledger).check()


def test_eval_mode_raises_when_week_limit_reached() -> None:
    ledger = InMemoryLedger()
    for days in (1, 2, 3):
        ledger.append(SpendEntry(NOW - timedelta(days=days), 0.7, "m"))
    with pytest.raises(BudgetExceededError, match="in 7 days"):
        guard(ledger).check()


def test_interactive_mode_warns_instead_of_raising() -> None:
    ledger = InMemoryLedger()
    ledger.append(SpendEntry(NOW, 5.0, "m"))
    with pytest.warns(BudgetWarning):
        guard(ledger, mode="interactive").check()


def test_record_appends_allowance() -> None:
    ledger = InMemoryLedger()
    guard(ledger).record(Usage(models=["a", "b"], allowance_usd=0.25))
    assert ledger.entries == [SpendEntry(NOW, 0.25, "a,b")]


def test_jsonl_ledger_round_trip(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "runs" / "spend.jsonl")
    assert ledger.total_since(NOW - timedelta(days=7)) == 0.0
    ledger.append(SpendEntry(NOW - timedelta(days=8), 1.0, "m"))
    ledger.append(SpendEntry(NOW - timedelta(hours=2), 0.3, "m"))
    assert ledger.total_since(NOW - timedelta(days=7)) == pytest.approx(0.3)
