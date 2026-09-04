"""Currency: the GBX trap, refusal to mix, and dated conversion."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from portfolio.core.money import (CurrencyMismatch, FxTable, MissingRate, Money,
                                  convert, normalise_currency)

D = dt.date


class TestMinorUnits:
    """GBp/GBX are pence. Treating them as pounds overstates by 100x."""

    def test_gbp_lowercase_p_is_pence(self):
        assert normalise_currency("GBp", Decimal("250")) == ("GBP", Decimal("2.50"))

    def test_gbx_is_pence(self):
        assert normalise_currency("GBX", Decimal("250")) == ("GBP", Decimal("2.50"))

    def test_uppercase_gbp_is_pounds_not_pence(self):
        # The regression that matters: fold case too early and this becomes 2.50.
        assert normalise_currency("GBP", Decimal("250")) == ("GBP", Decimal("250"))

    def test_money_normalises_on_construction(self):
        assert Money(Decimal("742.50"), "GBp") == Money(Decimal("7.425"), "GBP")

    def test_rejects_nonsense_code(self):
        with pytest.raises(ValueError):
            normalise_currency("EUROS")


class TestArithmetic:
    def test_same_currency_adds(self):
        assert (Money(Decimal("1.10"), "EUR") + Money(Decimal("2.20"), "EUR")).amount \
            == Decimal("3.30")

    def test_different_currencies_refuse_to_add(self):
        with pytest.raises(CurrencyMismatch):
            Money(Decimal("1"), "EUR") + Money(Decimal("1"), "USD")

    def test_different_currencies_refuse_to_subtract(self):
        with pytest.raises(CurrencyMismatch):
            Money(Decimal("1"), "EUR") - Money(Decimal("1"), "USD")


class TestConversion:
    def test_converts_at_stated_date(self, ):
        t = FxTable().add("USD", "EUR", D(2025, 1, 1), "0.90")
        got = convert(Money(Decimal("100"), "USD"), "EUR", D(2025, 1, 1), t)
        assert got == Money(Decimal("90.00"), "EUR")

    def test_uses_most_recent_earlier_rate_for_a_holiday(self):
        t = FxTable().add("USD", "EUR", D(2025, 1, 1), "0.90")
        assert convert(Money(Decimal("100"), "USD"), "EUR", D(2025, 3, 15), t).amount \
            == Decimal("90.00")

    def test_picks_the_rate_of_the_day_not_the_latest(self):
        """Cost basis converts at transaction date; using today's rate would
        silently rewrite what a past purchase cost."""
        t = FxTable().add("USD", "EUR", D(2025, 1, 1), "0.90")
        t.add("USD", "EUR", D(2025, 6, 1), "0.80")
        assert convert(Money(Decimal("100"), "USD"), "EUR", D(2025, 2, 1), t).amount \
            == Decimal("90.00")
        assert convert(Money(Decimal("100"), "USD"), "EUR", D(2025, 7, 1), t).amount \
            == Decimal("80.00")

    def test_inverts_a_reversed_pair(self):
        t = FxTable().add("EUR", "USD", D(2025, 1, 1), "1.25")
        assert convert(Money(Decimal("100"), "USD"), "EUR", D(2025, 1, 1), t).amount \
            == Decimal("80")

    def test_missing_rate_raises_rather_than_assuming_parity(self):
        """A silent 1.0 would turn a USD position into EUR at par, invisibly."""
        with pytest.raises(MissingRate):
            convert(Money(Decimal("1"), "JPY"), "EUR", D(2025, 1, 1), FxTable())
