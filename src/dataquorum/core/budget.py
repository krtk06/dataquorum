"""Budget guard: stops spending before the OpenCode Go plan limits are hit.

Go limits each model to about 20% of the monthly allowance per 5 hours and 50% per week.
We stop earlier, at $18 per 5 hours and $45 per week of *allowance* (what Go counts),
so there is headroom for interactive work.

Spend is kept in a ledger. `JsonlLedger` writes one line per call to `runs/spend.jsonl`
(git-ignored). The run recorder in week 4 will replace it with DuckDB.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from dataquorum.core.cost import Usage

Mode = Literal["eval", "interactive"]


class BudgetExceededError(RuntimeError):
    """Raised in eval mode when a spending window is over its limit."""


class BudgetWarning(UserWarning):
    """Issued in interactive mode when a spending window is over its limit."""


@dataclass(frozen=True)
class BudgetLimits:
    per_5_hours_usd: float = 18.0
    per_week_usd: float = 45.0


@dataclass(frozen=True)
class SpendEntry:
    at: datetime
    allowance_usd: float
    model: str


class Ledger(Protocol):
    def append(self, entry: SpendEntry) -> None: ...
    def total_since(self, since: datetime) -> float: ...


class InMemoryLedger:
    def __init__(self) -> None:
        self.entries: list[SpendEntry] = []

    def append(self, entry: SpendEntry) -> None:
        self.entries.append(entry)

    def total_since(self, since: datetime) -> float:
        return sum(e.allowance_usd for e in self.entries if e.at >= since)


class JsonlLedger:
    def __init__(self, path: Path = Path("runs/spend.jsonl")) -> None:
        self.path = path

    def append(self, entry: SpendEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "at": entry.at.isoformat(),
            "allowance_usd": entry.allowance_usd,
            "model": entry.model,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def total_since(self, since: datetime) -> float:
        if not self.path.exists():
            return 0.0
        total = 0.0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if datetime.fromisoformat(row["at"]) >= since:
                    total += float(row["allowance_usd"])
        return total


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BudgetGuard:
    """Check before each call, record after each call."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        mode: Mode = "interactive",
        limits: BudgetLimits | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.ledger = ledger
        self.mode = mode
        self.limits = limits or BudgetLimits()
        self.clock = clock

    def spent(self) -> tuple[float, float]:
        """Allowance spent in the last 5 hours and the last 7 days."""
        now = self.clock()
        return (
            self.ledger.total_since(now - timedelta(hours=5)),
            self.ledger.total_since(now - timedelta(days=7)),
        )

    def check(self) -> None:
        five_h, week = self.spent()
        problems = []
        if five_h >= self.limits.per_5_hours_usd:
            problems.append(f"${five_h:.2f} in 5 h (limit ${self.limits.per_5_hours_usd:.2f})")
        if week >= self.limits.per_week_usd:
            problems.append(f"${week:.2f} in 7 days (limit ${self.limits.per_week_usd:.2f})")
        if not problems:
            return
        message = "LLM budget reached: " + "; ".join(problems)
        if self.mode == "eval":
            raise BudgetExceededError(message)
        warnings.warn(message, BudgetWarning, stacklevel=2)

    def record(self, usage: Usage) -> None:
        model = ",".join(usage.models) or "unknown"
        self.ledger.append(SpendEntry(self.clock(), usage.allowance_usd, model))
