"""Positions derived from the ledger, with values verifiable by hand."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from portfolio.core.models import Instrument, Transaction, TransactionType as T
from portfolio.core.money import FxTable, Money
from portfolio.core.positions import (InsufficientUnits, PriceQuote,
                                      derive_positions, total_market_value,
                                      weights)

D = dt.date
A = "IE0002Y8CX98"
B = "IE000IAXNM41"
USD_ETC = "JE00BN7KB664"


def q(price: str, ccy: str = "EUR") -> PriceQuote:
    return PriceQuote(Money(Decimal(price), ccy), as_of=D(2025, 12, 31),
                      source="test", delay_minutes=15)


class TestHandComputable:
    """Every number here is checked against arithmetic written in the docstring."""

    def test_single_buy(self, two_asset_ledger):
        # A: 100 @ 10.00 + 5.00 fee = 1005.00 basis, 10.05/unit
        p = derive_positions(two_asset_ledger)[A]
        assert p.quantity == Decimal("100")
        assert p.cost_basis == Money(Decimal("1005.00"), "EUR")
        assert p.average_cost == Money(Decimal("10.05"), "EUR")

    def test_fees_are_capitalised_into_basis(self, two_asset_ledger):
        """Excluding fees from basis overstates every subsequent gain."""
        p = derive_positions(two_asset_ledger)[A]
        assert p.cost_basis.amount == Decimal("1005.00") != Decimal("1000.00")

    def test_partial_sell_realises_average_cost(self):
        """
        Buy 100 @ 10.00 + 5 fee -> basis 1005.00
        Buy 100 @ 12.00 + 5 fee -> basis 2210.00 over 200 units = 11.05/unit
        Sell 50 @ 15.00, 5 fee  -> proceeds 750.00
                                   basis sold 50 * 11.05 = 552.50
                                   realised  750.00 - 552.50 - 5.00 = 192.50
                                   remaining basis 2210.00 - 552.50 = 1657.50
        """
        txns = [
            Transaction(D(2025, 1, 10), A, T.BUY, Decimal("100"), Decimal("10.00"),
                        "EUR", Decimal("5")),
            Transaction(D(2025, 3, 10), A, T.BUY, Decimal("100"), Decimal("12.00"),
                        "EUR", Decimal("5")),
            Transaction(D(2025, 6, 10), A, T.SELL, Decimal("50"), Decimal("15.00"),
                        "EUR", Decimal("5")),
        ]
        p = derive_positions(txns)[A]
        assert p.quantity == Decimal("150")
        assert p.average_cost.amount == Decimal("11.05")
        assert p.realised_pnl == Money(Decimal("192.50"), "EUR")
        assert p.cost_basis == Money(Decimal("1657.50"), "EUR")

    def test_full_exit_leaves_exactly_zero_basis(self):
        txns = [
            Transaction(D(2025, 1, 1), A, T.BUY, Decimal("3"), Decimal("10"),
                        "EUR", Decimal("1")),
            Transaction(D(2025, 2, 1), A, T.SELL, Decimal("3"), Decimal("12"),
                        "EUR", Decimal("1")),
        ]
        p = derive_positions(txns)[A]
        assert p.quantity == 0
        assert p.cost_basis.amount == Decimal("0"), "no phantom residual basis"
        # proceeds 36 - basis 31 - fee 1 = 4
        assert p.realised_pnl.amount == Decimal("4")

    def test_selling_more_than_held_is_refused(self):
        txns = [
            Transaction(D(2025, 1, 1), A, T.BUY, Decimal("10"), Decimal("10")),
            Transaction(D(2025, 2, 1), A, T.SELL, Decimal("11"), Decimal("10")),
        ]
        with pytest.raises(InsufficientUnits, match="only 10 held"):
            derive_positions(txns)


class TestCurrency:
    def test_cost_basis_uses_the_transaction_date_rate(self, fx):
        """USD buy on 2025-01-01 at 0.90, not the later 0.80. Using today's rate
        would silently rewrite what a past purchase cost."""
        txns = [Transaction(D(2025, 1, 1), USD_ETC, T.BUY, Decimal("100"),
                            Decimal("10.00"), "USD", Decimal("0"))]
        p = derive_positions(txns, rates=fx)[USD_ETC]
        assert p.cost_basis == Money(Decimal("900.00"), "EUR")

    def test_two_purchases_at_different_rates(self, fx):
        # 100 @ 10 USD * 0.90 = 900 ; 100 @ 10 USD * 0.80 = 800 -> 1700 total
        txns = [
            Transaction(D(2025, 1, 1), USD_ETC, T.BUY, Decimal("100"), Decimal("10"), "USD"),
            Transaction(D(2025, 6, 1), USD_ETC, T.BUY, Decimal("100"), Decimal("10"), "USD"),
        ]
        p = derive_positions(txns, rates=fx)[USD_ETC]
        assert p.cost_basis == Money(Decimal("1700.00"), "EUR")
        assert p.average_cost == Money(Decimal("8.50"), "EUR")

    def test_pence_purchase_lands_in_pounds_then_euros(self, fx):
        # 90 units @ 742.50 GBp = 7.425 GBP -> 668.25 GBP * 1.20 = 801.90 EUR
        txns = [Transaction(D(2025, 1, 1), B, T.BUY, Decimal("90"),
                            Decimal("742.50"), "GBp", Decimal("0"))]
        p = derive_positions(txns, rates=fx)[B]
        assert p.cost_basis == Money(Decimal("801.90"), "EUR")

    def test_foreign_transaction_without_rates_refuses(self):
        txns = [Transaction(D(2025, 1, 1), USD_ETC, T.BUY, Decimal("1"), Decimal("1"), "USD")]
        with pytest.raises(ValueError, match="refusing to add unconverted"):
            derive_positions(txns)

    def test_market_value_in_another_currency_needs_rates(self, fx):
        txns = [Transaction(D(2025, 1, 1), USD_ETC, T.BUY, Decimal("10"),
                            Decimal("10"), "USD")]
        pos = derive_positions(txns, rates=fx,
                               quotes={USD_ETC: q("12.00", "USD")})[USD_ETC]
        assert pos.market_value(rates=None) is None, "must not add USD to EUR"
        assert pos.market_value(fx, D(2025, 6, 1)) == Money(Decimal("96.00"), "EUR")


class TestWatchlist:
    def test_instrument_with_no_transactions_is_a_watchlist_entry(self, instruments,
                                                                  two_asset_ledger):
        positions = derive_positions(two_asset_ledger, instruments)
        watch = positions["LU1681048630"]
        assert watch.is_watchlist
        assert not watch.is_open
        assert watch.quantity == 0

    def test_watchlist_entries_are_not_dropped(self, instruments, two_asset_ledger):
        """v2's simulator reads these; omitting them would hide exactly the
        instruments it needs."""
        positions = derive_positions(two_asset_ledger, instruments)
        assert set(positions) == set(instruments)

    def test_a_held_instrument_is_not_a_watchlist_entry(self, instruments,
                                                        two_asset_ledger):
        assert not derive_positions(two_asset_ledger, instruments)[A].is_watchlist

    def test_fully_sold_position_is_not_a_watchlist_entry(self):
        """It has history, so it is closed, not never-owned."""
        txns = [
            Transaction(D(2025, 1, 1), A, T.BUY, Decimal("1"), Decimal("1")),
            Transaction(D(2025, 2, 1), A, T.SELL, Decimal("1"), Decimal("1")),
        ]
        p = derive_positions(txns)[A]
        assert not p.is_open and not p.is_watchlist


class TestTotalsAndWeights:
    def _positions(self, two_asset_ledger, instruments):
        # A: 100 @ 12.00 = 1200 ; B: 200 @ 4.00 = 800 ; total 2000
        return derive_positions(two_asset_ledger, instruments,
                                quotes={A: q("12.00"), B: q("4.00")})

    def test_total_market_value(self, two_asset_ledger, instruments):
        total, missing = total_market_value(self._positions(two_asset_ledger, instruments))
        assert total == Money(Decimal("2000.00"), "EUR")
        assert missing == []

    def test_weights_sum_to_exactly_one(self, two_asset_ledger, instruments):
        w = weights(self._positions(two_asset_ledger, instruments))
        assert sum(w.values()) == Decimal("1")
        assert w[A] == Decimal("0.6") and w[B] == Decimal("0.4")

    def test_watchlist_entries_carry_no_weight(self, two_asset_ledger, instruments):
        assert "LU1681048630" not in weights(self._positions(two_asset_ledger, instruments))

    def test_unpriced_holdings_are_reported_not_silently_dropped(self, two_asset_ledger,
                                                                 instruments):
        """A total that quietly omits a holding is wrong in a way nobody notices."""
        positions = derive_positions(two_asset_ledger, instruments, quotes={A: q("12.00")})
        total, missing = total_market_value(positions)
        assert total == Money(Decimal("1200.00"), "EUR")
        assert missing == [B]

    def test_empty_portfolio_gives_empty_weights(self):
        assert weights({}) == {}


class TestIncomeAndFees:
    def test_dividend_is_income_not_a_gain(self):
        txns = [
            Transaction(D(2025, 1, 1), A, T.BUY, Decimal("40"), Decimal("100")),
            Transaction(D(2025, 6, 1), A, T.DIVIDEND, Decimal("40"), Decimal("0.62")),
        ]
        p = derive_positions(txns)[A]
        assert p.dividends == Money(Decimal("24.80"), "EUR")
        assert p.realised_pnl.amount == Decimal("0")
        assert p.cost_basis.amount == Decimal("4000")

    def test_standalone_fee_reduces_realised_pnl(self):
        txns = [
            Transaction(D(2025, 1, 1), A, T.BUY, Decimal("10"), Decimal("10")),
            Transaction(D(2025, 12, 31), A, T.FEE, Decimal("0"), Decimal("0"),
                        "EUR", Decimal("12.00"), "custody"),
        ]
        p = derive_positions(txns)[A]
        assert p.realised_pnl == Money(Decimal("-12.00"), "EUR")
        assert p.cost_basis.amount == Decimal("100"), "a fee is not part of unit cost"
