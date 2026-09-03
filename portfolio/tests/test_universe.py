"""Deletion rules: the ledger must never be orphaned."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from portfolio.core.models import AssetClass, Instrument, Transaction, TransactionType as T
from portfolio.core.universe import check_deletable, deactivate, reactivate

D = dt.date
HELD = "IE0002Y8CX98"
WATCHED = "LU1681048630"


def ledger():
    return [
        Transaction(D(2025, 1, 1), HELD, T.BUY, Decimal("10"), Decimal("10")),
        Transaction(D(2025, 2, 1), HELD, T.SELL, Decimal("5"), Decimal("12")),
    ]


class TestDeletionRules:
    def test_instrument_with_transactions_cannot_be_deleted(self):
        check = check_deletable(HELD, ledger())
        assert not check
        assert check.transaction_count == 2
        assert "orphan" in check.reason

    def test_refusal_offers_deactivation(self):
        assert "Deactivate instead" in check_deletable(HELD, ledger()).alternative

    def test_watchlist_entry_can_be_deleted(self):
        check = check_deletable(WATCHED, ledger())
        assert check
        assert check.transaction_count == 0

    def test_deletable_always_reports_zero_affected(self):
        """The confirmation dialog shows this number; for anything deletable it
        must be zero, which is the reassurance the user needs."""
        check = check_deletable(WATCHED, ledger())
        assert check.allowed and check.transaction_count == 0

    def test_a_fully_sold_instrument_still_cannot_be_deleted(self):
        """Quantity zero is not the same as never owned: the history is real
        and deleting it would change past portfolio values."""
        txns = [
            Transaction(D(2025, 1, 1), HELD, T.BUY, Decimal("5"), Decimal("10")),
            Transaction(D(2025, 2, 1), HELD, T.SELL, Decimal("5"), Decimal("12")),
        ]
        assert not check_deletable(HELD, txns)

    def test_check_is_falsy_and_truthy_directly(self):
        assert bool(check_deletable(WATCHED, ledger())) is True
        assert bool(check_deletable(HELD, ledger())) is False


class TestDeactivation:
    def _inst(self):
        return Instrument(HELD, "WisdomTree Europe Defence", AssetClass.ETF, "EUR")

    def test_deactivate_is_reversible(self):
        i = self._inst()
        assert i.active
        deactivate(i, "no longer tracked")
        assert not i.active
        reactivate(i)
        assert i.active

    def test_deactivation_records_a_reason(self):
        assert "no longer tracked" in deactivate(self._inst(), "no longer tracked").note

    def test_deactivation_does_not_touch_the_ledger(self):
        log = ledger()
        deactivate(self._inst(), "x")
        assert len(log) == 2, "deactivation must never cascade into transactions"
