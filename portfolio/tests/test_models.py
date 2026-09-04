"""ISIN validation, ledger validation, append-only amendments, overrides."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from portfolio.core.models import (Amendment, AssetClass, Instrument, Transaction,
                                   TransactionType, ValidationError,
                                   apply_amendments, is_valid_isin)

D = dt.date

# Every ISIN in the reference set. All real, so this is a genuine check of the
# check-digit implementation rather than a self-fulfilling one.
REAL_ISINS = [
    "IE0002Y8CX98", "IE000IAXNM41", "IE000I7E6HL0", "IE000OJ5TQP4",
    "IE00B6R52143", "GB00B15KYL00", "JE00BN7KB664", "GB00B15KYB02",
    "IE00BMW42637", "LU1681048630",
    "GB00B15KY765", "DE000A1JS9B2",          # the liquidated predecessors
    "US97717Y3374", "US8829277677", "US88166A8707",   # the US collisions
]


class TestIsin:
    @pytest.mark.parametrize("isin", REAL_ISINS)
    def test_real_isins_pass(self, isin):
        assert is_valid_isin(isin)

    @pytest.mark.parametrize("bad", [
        "IE0002Y8CX97",        # wrong check digit
        "IE0002Y8CX9",         # too short
        "IE0002Y8CX988",       # too long
        "0E0002Y8CX98",        # country code must be alphabetic
        "",
        "NOTANISIN123",
    ])
    def test_bad_isins_fail(self, bad):
        assert not is_valid_isin(bad)

    def test_transposed_characters_are_caught(self):
        """The check digit exists for exactly this: a typo makes a second,
        empty instrument rather than failing loudly."""
        assert is_valid_isin("IE00B6R52143")
        assert not is_valid_isin("IE00B6R52134")

    def test_transaction_rejects_invalid_isin(self):
        with pytest.raises(ValidationError, match="primary key"):
            Transaction(D(2025, 1, 1), "IE0002Y8CX97", TransactionType.BUY,
                        Decimal("1"), Decimal("1"))


class TestTransactionValidation:
    def test_buy_needs_positive_quantity(self):
        with pytest.raises(ValidationError, match="positive quantity"):
            Transaction(D(2025, 1, 1), REAL_ISINS[0], TransactionType.BUY,
                        Decimal("-5"), Decimal("10"))

    def test_sell_is_positive_quantity_of_type_sell(self):
        with pytest.raises(ValidationError, match="not a negative BUY"):
            Transaction(D(2025, 1, 1), REAL_ISINS[0], TransactionType.SELL,
                        Decimal("-5"), Decimal("10"))

    def test_negative_fees_rejected(self):
        with pytest.raises(ValidationError, match="fees cannot be negative"):
            Transaction(D(2025, 1, 1), REAL_ISINS[0], TransactionType.BUY,
                        Decimal("5"), Decimal("10"), "EUR", Decimal("-1"))

    def test_pence_price_normalised_at_construction(self):
        t = Transaction(D(2025, 1, 1), REAL_ISINS[0], TransactionType.BUY,
                        Decimal("10"), Decimal("742.50"), "GBp")
        assert t.currency == "GBP"
        assert t.price_per_unit == Decimal("7.4250")
        assert t.gross.amount == Decimal("74.250")

    def test_iso_date_string_accepted(self):
        assert Transaction("2025-01-15", REAL_ISINS[0], TransactionType.BUY,
                           Decimal("1"), Decimal("1")).date == D(2025, 1, 15)


class TestCashFlow:
    def test_buy_is_money_out(self):
        t = Transaction(D(2025, 1, 1), REAL_ISINS[0], TransactionType.BUY,
                        Decimal("10"), Decimal("20"), "EUR", Decimal("5"))
        assert t.cash_flow().amount == Decimal("-205")

    def test_sell_is_money_in_net_of_fees(self):
        t = Transaction(D(2025, 1, 1), REAL_ISINS[0], TransactionType.SELL,
                        Decimal("10"), Decimal("20"), "EUR", Decimal("5"))
        assert t.cash_flow().amount == Decimal("195")


class TestAppendOnly:
    def test_void_removes_from_the_derived_view_not_the_log(self):
        good = Transaction(D(2025, 1, 1), REAL_ISINS[0], TransactionType.BUY,
                           Decimal("10"), Decimal("20"))
        typo = Transaction(D(2025, 1, 2), REAL_ISINS[0], TransactionType.BUY,
                           Decimal("1000"), Decimal("20"))
        log = [good, typo]
        live = apply_amendments(log, [Amendment(target_id=typo.id)])
        assert [t.id for t in live] == [good.id]
        assert len(log) == 2, "the ledger itself must not be rewritten"

    def test_amendment_needs_a_target(self):
        with pytest.raises(ValidationError):
            Amendment(target_id="")


class TestInstrumentOverrides:
    def _inst(self) -> Instrument:
        return Instrument("JE00BN7KB664", "WisdomTree Wheat", AssetClass.ETC, "USD",
                          provider_symbols={"eodhd": "WEAT.LSE"})

    def test_override_provider_symbol_sticks(self):
        i = self._inst().override("provider_symbols.eodhd", "OD7S.XETRA")
        assert i.provider_symbols["eodhd"] == "OD7S.XETRA"
        assert i.is_overridden("provider_symbols.eodhd")

    def test_re_resolution_does_not_clobber_a_manual_correction(self):
        """The point of marking overrides: a resolver that silently reverts your
        fix is worse than one that never runs."""
        i = self._inst().override("provider_symbols.eodhd", "OD7S.XETRA")
        skipped = i.apply_resolution({"provider_symbols": {"eodhd": "WEAT.LSE",
                                                           "yfinance": "WEAT.L"}})
        assert i.provider_symbols["eodhd"] == "OD7S.XETRA"
        assert i.provider_symbols["yfinance"] == "WEAT.L", "unprotected fields still update"
        assert skipped == ["provider_symbols.eodhd"]

    def test_re_resolution_reports_what_it_skipped(self):
        i = self._inst().override("name", "WisdomTree Wheat ETC (mine)")
        skipped = i.apply_resolution({"name": "WisdomTree Wheat", "issuer": "WisdomTree"})
        assert skipped == ["name"]
        assert i.name == "WisdomTree Wheat ETC (mine)"
        assert i.issuer == "WisdomTree"

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            self._inst().override("nonsense", 1)

    def test_etc_is_not_folded_into_etf(self):
        assert AssetClass.ETC != AssetClass.ETF
        assert self._inst().asset_class is AssetClass.ETC
