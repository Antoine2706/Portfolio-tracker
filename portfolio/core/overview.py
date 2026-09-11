"""The small arithmetic the overview needs, kept out of the API layer.

Day change, a sparkline, and the portfolio-level sums are each a few lines.
They live here rather than in `api/` because the rule is that the API projects
and never computes: a subtraction in a route handler is untested arithmetic,
and untested arithmetic is how a day change ends up comparing today's close
with the close from a different holiday calendar.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal

import pandas as pd

from .money import BASE_CURRENCY, FxRates, Money, MissingRate
from .positions import Position

__all__ = ["DayChange", "day_change", "sparkline", "PositionTotals", "position_totals"]


@dataclasses.dataclass(frozen=True)
class DayChange:
    """Movement between the last two closes, in the quote currency and as a fraction."""
    amount: Decimal            # per unit, quote currency
    pct: float
    last_date: dt.date
    previous_date: dt.date


def day_change(closes: pd.Series) -> DayChange | None:
    """Last close against the one before it. None with fewer than two closes.

    >>> import pandas as pd
    >>> s = pd.Series([100.0, 102.0], index=pd.to_datetime(["2026-09-03", "2026-09-04"]))
    >>> dc = day_change(s)
    >>> (float(dc.amount), round(dc.pct, 4), dc.last_date.isoformat())
    (2.0, 0.02, '2026-09-04')
    """
    s = closes.dropna()
    if len(s) < 2:
        return None
    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
    if prev == 0:
        return None
    amount = Decimal(str(round(last - prev, 6)))
    return DayChange(amount=amount, pct=last / prev - 1.0,
                     last_date=pd.Timestamp(s.index[-1]).date(),
                     previous_date=pd.Timestamp(s.index[-2]).date())


def sparkline(closes: pd.Series, points: int = 30) -> list[float]:
    """The last `points` closes as plain floats, oldest first.

    >>> import pandas as pd
    >>> sparkline(pd.Series([1.0, 2.0, 3.0, 4.0]), points=3)
    [2.0, 3.0, 4.0]
    """
    s = closes.dropna()
    return [float(v) for v in s.iloc[-points:]]


@dataclasses.dataclass(frozen=True)
class PositionTotals:
    """Sums over open positions, in the reporting currency.

    Realised, dividends and fees are summed over every position that has a
    ledger history, closed ones included: money you took out of a position you
    no longer hold is still money you took out.
    """
    realised: Money
    dividends: Money
    fees: Money
    cost_basis: Money
    day_change: Money | None
    day_change_pct: float | None


def position_totals(positions: dict[str, Position],
                    day_changes: dict[str, DayChange],
                    rates: FxRates | None = None,
                    on: dt.date | None = None,
                    base: str = BASE_CURRENCY) -> PositionTotals:
    """Portfolio-level sums, with the day change converted at today's rate.

    The day change is only reported when every open, priced holding has one;
    a total that silently omits one holding's move is wrong in a way nobody
    notices, so it is None instead. The percentage is the move against the
    previous day's value of the same holdings.
    """
    on = on or dt.date.today()
    realised = Money.zero(base)
    dividends = Money.zero(base)
    fees = Money.zero(base)
    cost = Money.zero(base)
    change = Money.zero(base)
    previous_value = Decimal("0")
    complete = True

    for isin, pos in positions.items():
        if pos.transaction_count == 0 or not pos.is_derivable:
            continue
        realised = realised + pos.realised_pnl
        dividends = dividends + pos.dividends
        fees = fees + pos.fees_paid
        if not pos.is_open:
            continue
        cost = cost + pos.cost_basis
        dc = day_changes.get(isin)
        if dc is None or pos.quote is None:
            complete = False
            continue
        move = Money(dc.amount * pos.quantity, pos.quote.price.currency)
        last_value = Money(pos.quote.price.amount * pos.quantity, pos.quote.price.currency)
        if move.currency != base:
            if rates is None:
                complete = False
                continue
            try:
                rate = rates.rate(move.currency, base, on)
            except MissingRate:
                complete = False
                continue
            move = Money(move.amount * rate, base)
            last_value = Money(last_value.amount * rate, base)
        change = change + move
        previous_value += last_value.amount - move.amount

    if not complete or previous_value == 0:
        return PositionTotals(realised, dividends, fees, cost, None, None)
    return PositionTotals(realised, dividends, fees, cost, change,
                          float(change.amount / previous_value))
