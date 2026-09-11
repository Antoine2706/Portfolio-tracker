"""Daily FX history: the pair table, bisected lookups, and the same refusals
`core.money.FxTable` makes -- plus a speed guard, because the whole point of
this class is the lookup that FxTable does by scanning."""

from __future__ import annotations

import datetime as dt
import random
import time
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from portfolio.core.models import AssetClass, Instrument, Transaction, TransactionType as T
from portfolio.core.money import FxRates, MissingRate, Money, convert
from portfolio.data.fx import FX_PAIRS, DailyFxTable, currencies_needed, pair_symbol
from portfolio.data.providers.fixture import FIXTURE_END, FixtureProvider

D = dt.date
A, B = "IE0002Y8CX98", "IE000IAXNM41"


def week(values=(0.90, 0.91, 0.92, 0.93, 0.94)) -> pd.Series:
    """Mon 6 Jan 2025 to Fri 10 Jan 2025: a known weekend follows."""
    return pd.Series(list(values), index=pd.bdate_range("2025-01-06", periods=len(values)))


@pytest.fixture
def usd() -> DailyFxTable:
    t = DailyFxTable()
    t.add_series("USD", "EUR", week())
    return t


class TestPairTable:
    def test_every_pair_converts_into_eur(self):
        for ccy, symbol in FX_PAIRS.items():
            assert symbol == f"{ccy}EUR=X"
        assert {"USD", "GBP", "CHF", "SEK", "DKK", "NOK", "JPY", "CAD", "AUD"} == set(FX_PAIRS)

    def test_pair_symbol_normalises_pence(self):
        assert pair_symbol("GBp") == "GBPEUR=X"

    def test_no_pair_for_an_unknown_currency_or_another_base(self):
        assert pair_symbol("XXX") is None
        assert pair_symbol("USD", base="GBP") is None


class TestLookups:
    def test_implements_the_core_protocol(self, usd):
        assert isinstance(usd, FxRates)

    def test_exact_date(self, usd):
        assert usd.rate("USD", "EUR", D(2025, 1, 8)) == Decimal("0.92")

    def test_weekend_uses_the_friday_before(self, usd):
        """A Saturday trade converts at Friday's close, not Monday's."""
        assert usd.rate("USD", "EUR", D(2025, 1, 11)) == Decimal("0.94")
        assert usd.rate("USD", "EUR", D(2025, 1, 12)) == Decimal("0.94")

    def test_later_than_the_last_date_uses_the_last_rate(self, usd):
        assert usd.rate("USD", "EUR", D(2026, 6, 1)) == Decimal("0.94")

    def test_before_the_first_date_raises_naming_pair_and_date(self, usd):
        with pytest.raises(MissingRate) as exc:
            usd.rate("USD", "EUR", D(2025, 1, 5))
        assert "USD->EUR" in str(exc.value) and "2025-01-05" in str(exc.value)

    def test_pair_never_loaded_raises_a_different_sentence(self, usd):
        with pytest.raises(MissingRate) as exc:
            usd.rate("JPY", "EUR", D(2025, 1, 8))
        assert "JPY->EUR" in str(exc.value) and "never loaded" in str(exc.value)

    def test_inverse_pair_is_inverted(self):
        t = DailyFxTable()
        t.add_series("EUR", "USD", week((1.25, 1.25, 1.25, 1.25, 1.25)))
        assert t.rate("USD", "EUR", D(2025, 1, 8)) == Decimal("0.8")
        assert t.rate("USD", "EUR", D(2025, 1, 11)) == Decimal("0.8")

    def test_direct_pair_wins_over_the_inverse(self):
        t = DailyFxTable()
        t.add_series("USD", "EUR", week())
        t.add_series("EUR", "USD", week((2.0, 2.0, 2.0, 2.0, 2.0)))
        assert t.rate("USD", "EUR", D(2025, 1, 8)) == Decimal("0.92")

    def test_same_currency_is_one_without_any_data(self):
        assert DailyFxTable().rate("EUR", "EUR", D(2025, 1, 1)) == Decimal("1")

    def test_pence_normalise_to_pounds(self):
        t = DailyFxTable()
        t.add_series("GBp", "EUR", week((1.2, 1.2, 1.2, 1.2, 1.2)))
        assert t.pairs() == [("GBP", "EUR")]
        assert t.rate("GBp", "EUR", D(2025, 1, 8)) == Decimal("1.2")
        assert t.rate("GBX", "eur", D(2025, 1, 8)) == Decimal("1.2")

    def test_datetime_lookup_is_treated_as_its_date(self, usd):
        assert usd.rate("USD", "EUR", dt.datetime(2025, 1, 8, 15, 30)) == Decimal("0.92")
        assert usd.rate("USD", "EUR", pd.Timestamp("2025-01-08")) == Decimal("0.92")

    def test_ten_significant_digits_not_float_noise(self):
        t = DailyFxTable()
        t.add_series("USD", "EUR", week((0.123456789012, 0.1, 0.1, 0.1, 0.1)))
        assert t.rate("USD", "EUR", D(2025, 1, 6)) == Decimal("0.123456789")
        assert t.rate("USD", "EUR", D(2025, 1, 7)) == Decimal("0.1")

    def test_returns_decimal_so_money_arithmetic_stays_exact(self, usd):
        rate = usd.rate("USD", "EUR", D(2025, 1, 8))
        assert isinstance(rate, Decimal)
        assert Money(Decimal("100"), "USD").amount * rate == Decimal("92.00")


class TestLoading:
    def test_second_series_merges_and_overwrites_the_overlap(self):
        t = DailyFxTable()
        t.add_series("USD", "EUR", week())
        t.add_series("USD", "EUR", pd.Series([0.5, 0.6],
                                              index=pd.bdate_range("2025-01-10", periods=2)))
        assert t.observations("USD", "EUR") == 6
        assert t.rate("USD", "EUR", D(2025, 1, 10)) == Decimal("0.5")
        assert t.rate("USD", "EUR", D(2025, 1, 13)) == Decimal("0.6")
        assert t.rate("USD", "EUR", D(2025, 1, 9)) == Decimal("0.93")

    def test_add_is_a_drop_in_for_fx_table(self):
        """The conversion tests in test_money.py, verbatim, on this class."""
        t = DailyFxTable().add("USD", "EUR", D(2025, 1, 1), "0.90")
        t.add("USD", "EUR", D(2025, 6, 1), "0.80")
        assert convert(Money(Decimal("100"), "USD"), "EUR", D(2025, 2, 1), t).amount \
            == Decimal("90.00")
        assert convert(Money(Decimal("100"), "USD"), "EUR", D(2025, 7, 1), t).amount \
            == Decimal("80.00")
        with pytest.raises(MissingRate):
            convert(Money(Decimal("1"), "JPY"), "EUR", D(2025, 1, 1), t)

    def test_nan_rows_are_skipped(self):
        t = DailyFxTable()
        t.add_series("USD", "EUR", week((0.9, np.nan, 0.92, np.nan, 0.94)))
        assert t.observations("USD", "EUR") == 3
        assert t.rate("USD", "EUR", D(2025, 1, 7)) == Decimal("0.9")

    def test_tz_aware_index_is_read_by_date(self):
        s = week()
        s.index = s.index.tz_localize("Europe/London")
        t = DailyFxTable()
        t.add_series("USD", "EUR", s)
        assert t.rate("USD", "EUR", D(2025, 1, 8)) == Decimal("0.92")

    def test_latest_and_pairs(self, usd):
        assert usd.latest("USD", "EUR") == (D(2025, 1, 10), Decimal("0.94"))
        assert usd.latest("EUR", "USD") == (D(2025, 1, 10), Decimal("1.063829787"))
        assert usd.latest("JPY", "EUR") is None
        assert usd.pairs() == [("USD", "EUR")]
        assert len(usd) == 1

    def test_fixture_provider_series_loads_directly(self):
        t = DailyFxTable()
        t.add_series("USD", "EUR", FixtureProvider().history("USDEUR=X"))
        # The fixture's random walk is anchored so its last value is the spot.
        assert t.rate("USD", "EUR", FIXTURE_END) == Decimal("0.92")
        assert t.observations("USD", "EUR") == 520


class TestSpeed:
    def test_ten_thousand_lookups_are_fast(self):
        """value_history does a lookup per date per foreign holding. FxTable's
        O(n) scan makes that seconds; this must stay well under one."""
        t = DailyFxTable()
        t.add_series("USD", "EUR", FixtureProvider().history("USDEUR=X"))
        rng = random.Random(7)
        dates = [FIXTURE_END - dt.timedelta(days=rng.randint(0, 600)) for _ in range(10_000)]
        started = time.perf_counter()
        for on in dates:
            t.rate("USD", "EUR", on)
        assert time.perf_counter() - started < 0.5


class TestCurrenciesNeeded:
    @pytest.fixture
    def instruments(self):
        return {
            A: Instrument(A, "EUR line", AssetClass.ETF, "EUR", quote_currency="EUR"),
            # USD-base fund quoted in EUR: no USD rate is needed to value it.
            B: Instrument(B, "USD base, EUR quote", AssetClass.ETF, "USD", quote_currency="EUR"),
            "LU1681048630": Instrument("LU1681048630", "Pence line", AssetClass.ETF, "USD",
                                       quote_currency="GBp"),
        }

    def test_quote_currencies_and_ledger_currencies_minus_base(self, instruments):
        txns = [Transaction(D(2025, 1, 1), A, T.BUY, Decimal("1"), Decimal("1"), "CHF")]
        assert currencies_needed(instruments, txns) == {"GBP", "CHF"}

    def test_base_currency_of_a_fund_is_not_a_conversion(self, instruments):
        assert "USD" not in currencies_needed(instruments, [])

    def test_pence_reported_as_pounds(self, instruments):
        assert "GBP" in currencies_needed(instruments, [])
        assert "GBp" not in currencies_needed(instruments, [])

    def test_another_base_excludes_itself(self, instruments):
        """Reporting in GBP: the pence line needs nothing, the EUR lines do."""
        assert currencies_needed(instruments, [], base="GBP") == {"EUR"}

    def test_accepts_a_list_of_instruments(self, instruments):
        assert currencies_needed(list(instruments.values()), []) == {"GBP"}
