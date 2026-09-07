"""core.overview: the arithmetic behind the KPI strip."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from portfolio.core.models import Transaction, TransactionType
from portfolio.core.money import FxTable, Money
from portfolio.core.overview import (DayChange, day_change, position_totals,
                                     sparkline)
from portfolio.core.positions import PriceQuote, derive_positions

D = dt.date


def _series(values, end="2026-09-04"):
    return pd.Series(values, index=pd.bdate_range(end=end, periods=len(values)))


class TestDayChange:
    def test_uses_the_last_two_closes(self):
        dc = day_change(_series([98.0, 100.0, 103.0]))
        assert dc.amount == Decimal("3")
        assert dc.pct == pytest.approx(0.03)
        assert dc.last_date == D(2026, 9, 4)
        assert dc.previous_date == D(2026, 9, 3)

    def test_one_close_is_not_a_change(self):
        assert day_change(_series([100.0])) is None

    def test_nan_rows_are_skipped_not_compared(self):
        s = _series([100.0, float("nan"), 101.0])
        dc = day_change(s)
        assert dc.amount == Decimal("1")

    def test_zero_previous_close_is_refused(self):
        assert day_change(_series([0.0, 5.0])) is None


class TestSparkline:
    def test_last_n_oldest_first(self):
        assert sparkline(_series([1.0, 2.0, 3.0, 4.0, 5.0]), points=3) == [3.0, 4.0, 5.0]

    def test_shorter_than_n_returns_everything(self):
        assert sparkline(_series([1.0, 2.0]), points=30) == [1.0, 2.0]


class TestPositionTotals:
    A, B = "IE0002Y8CX98", "IE000IAXNM41"

    def _positions(self, fx=None, quote_b_ccy="EUR"):
        txns = [
            Transaction(D(2025, 1, 10), self.A, TransactionType.BUY,
                        Decimal("100"), Decimal("10"), "EUR", Decimal("5")),
            Transaction(D(2025, 1, 10), self.B, TransactionType.BUY,
                        Decimal("10"), Decimal("20"), "EUR", Decimal("0")),
            Transaction(D(2025, 2, 1), self.B, TransactionType.DIVIDEND,
                        Decimal("10"), Decimal("0.5"), "EUR", Decimal("0")),
        ]
        quotes = {
            self.A: PriceQuote(Money(Decimal("12"), "EUR"), D(2026, 9, 4)),
            self.B: PriceQuote(Money(Decimal("22"), quote_b_ccy), D(2026, 9, 4)),
        }
        return derive_positions(txns, rates=fx, quotes=quotes, strict=False)

    def test_sums_are_hand_computable(self):
        positions = self._positions()
        changes = {
            self.A: DayChange(Decimal("0.5"), 0.05, D(2026, 9, 4), D(2026, 9, 3)),
            self.B: DayChange(Decimal("-1"), -0.05, D(2026, 9, 4), D(2026, 9, 3)),
        }
        t = position_totals(positions, changes)
        assert t.cost_basis == Money(Decimal("1205"), "EUR")      # 1005 + 200
        assert t.dividends == Money(Decimal("5"), "EUR")
        assert t.fees == Money(Decimal("5"), "EUR")
        assert t.realised == Money(Decimal("0"), "EUR")
        # A moved +0.5 x 100 = +50, B moved -1 x 10 = -10.
        assert t.day_change == Money(Decimal("40"), "EUR")
        # Previous value: A 11.5 x 100 = 1150, B 23 x 10 = 230 -> 1380.
        assert t.day_change_pct == pytest.approx(40 / 1380)

    def test_a_missing_day_change_blanks_the_total_rather_than_omitting(self):
        positions = self._positions()
        changes = {self.A: DayChange(Decimal("0.5"), 0.05, D(2026, 9, 4), D(2026, 9, 3))}
        t = position_totals(positions, changes)
        assert t.day_change is None and t.day_change_pct is None
        assert t.cost_basis == Money(Decimal("1205"), "EUR")

    def test_foreign_quote_converts_at_the_reporting_date(self):
        fx = FxTable().add("USD", "EUR", D(2026, 9, 1), "0.5")
        positions = self._positions(fx=fx, quote_b_ccy="USD")
        changes = {
            self.A: DayChange(Decimal("1"), 0.1, D(2026, 9, 4), D(2026, 9, 3)),
            self.B: DayChange(Decimal("2"), 0.1, D(2026, 9, 4), D(2026, 9, 3)),
        }
        t = position_totals(positions, changes, rates=fx, on=D(2026, 9, 4))
        # A: +100 EUR. B: +2 USD x 10 units x 0.5 = +10 EUR.
        assert t.day_change == Money(Decimal("110"), "EUR")

    def test_foreign_quote_without_a_rate_is_not_added_at_par(self):
        positions = self._positions(fx=None, quote_b_ccy="USD")
        changes = {
            self.A: DayChange(Decimal("1"), 0.1, D(2026, 9, 4), D(2026, 9, 3)),
            self.B: DayChange(Decimal("2"), 0.1, D(2026, 9, 4), D(2026, 9, 3)),
        }
        t = position_totals(positions, changes, rates=None, on=D(2026, 9, 4))
        assert t.day_change is None
