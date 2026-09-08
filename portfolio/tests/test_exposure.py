"""Exposure slices: grouping and summing, tested because a view would not."""

from __future__ import annotations

import pytest

from portfolio.core.exposure import DIMENSIONS, UNKNOWN, exposure_by, exposures
from portfolio.core.models import AssetClass, Instrument

A, B, C = "IE0002Y8CX98", "IE000IAXNM41", "JE00BN7KB664"


@pytest.fixture
def instruments():
    return {
        A: Instrument(A, "WisdomTree Europe Defence", AssetClass.ETF, "EUR",
                      issuer="WisdomTree", exchange="XETR", quote_currency="EUR"),
        B: Instrument(B, "iShares Europe Defence", AssetClass.ETF, "EUR",
                      issuer="", exchange="XETR", quote_currency="EUR"),
        C: Instrument(C, "WisdomTree Copper", AssetClass.ETC, "USD",
                      issuer="WisdomTree", exchange="XLON", quote_currency="GBp"),
    }


VALUES = {A: 1200.0, B: 800.0, C: 500.0}


class TestExposureBy:
    def test_asset_class_by_hand(self, instruments):
        """ETF 1200 + 800 = 2000 of 2500 = 80%; ETC 500 = 20%."""
        slices = exposure_by(VALUES, instruments, "asset_class")
        assert [(s.key, s.value, s.count) for s in slices] == [("ETF", 2000.0, 2), ("ETC", 500.0, 1)]
        assert slices[0].weight == pytest.approx(0.8) and slices[1].weight == pytest.approx(0.2)
        assert slices[0].isins == (A, B)

    def test_enum_key_is_the_string_value(self, instruments):
        keys = {s.key for s in exposure_by(VALUES, instruments, "asset_class")}
        assert keys == {"ETF", "ETC"} and all(isinstance(k, str) for k in keys)

    def test_asset_class_labels(self, instruments):
        labels = {s.key: s.label for s in exposure_by(VALUES, instruments, "asset_class")}
        assert labels == {"ETF": "ETF", "ETC": "ETC (commodity)"}

    def test_blank_issuer_is_backfilled_from_the_fund_name(self, instruments):
        """B was constructed with issuer="", and is an iShares fund by name.

        An issuer breakdown with a large Unknown slice is a data quality
        report, not a portfolio insight, so the model derives what it can.
        """
        assert instruments[B].issuer == "iShares"
        slices = exposure_by(VALUES, instruments, "issuer")
        assert [(s.key, s.value) for s in slices] == [("WisdomTree", 1700.0), ("iShares", 800.0)]

    def test_unrecognised_issuer_is_still_named_unknown(self):
        """The fallback must survive: derivation is a heuristic, not a promise."""
        odd = Instrument(B, "Some Unbranded Tracker", AssetClass.ETF, "EUR", issuer="")
        assert odd.issuer == ""
        slices = exposure_by({B: 100.0}, {B: odd}, "issuer")
        assert [(s.key, s.value) for s in slices] == [(UNKNOWN, 100.0)]

    def test_missing_instrument_is_unknown(self):
        slices = exposure_by({A: 100.0}, {}, "issuer")
        assert slices[0].key == UNKNOWN and slices[0].isins == (A,)

    def test_currencies_and_exchange(self, instruments):
        assert [(s.key, s.value) for s in exposure_by(VALUES, instruments, "base_currency")] == \
            [("EUR", 2000.0), ("USD", 500.0)]
        assert [(s.key, s.value) for s in exposure_by(VALUES, instruments, "quote_currency")] == \
            [("EUR", 2000.0), ("GBP", 500.0)], "pence normalise to pounds at the instrument"
        assert [(s.key, s.value) for s in exposure_by(VALUES, instruments, "exchange")] == \
            [("XETR", 2000.0), ("XLON", 500.0)]

    def test_sorted_by_value_then_key(self, instruments):
        slices = exposure_by({A: 500.0, B: 500.0, C: 500.0}, instruments, "issuer")
        assert [s.key for s in slices] == ["WisdomTree", "iShares"]
        assert slices[0].value >= slices[1].value

    def test_tie_broken_by_key(self, instruments):
        instruments[B].issuer = "Amundi"
        instruments[C].issuer = "Zed"
        slices = exposure_by({A: 100.0, B: 100.0, C: 100.0}, instruments, "issuer")
        assert [s.key for s in slices] == ["Amundi", "WisdomTree", "Zed"]

    def test_weights_sum_to_one(self, instruments):
        for dimension in DIMENSIONS:
            assert sum(s.weight for s in exposure_by(VALUES, instruments, dimension)) == pytest.approx(1.0)

    def test_zero_and_negative_values_are_skipped(self, instruments):
        slices = exposure_by({A: 100.0, B: 0.0, C: -5.0}, instruments, "asset_class")
        assert [(s.key, s.count) for s in slices] == [("ETF", 1)]

    def test_empty(self, instruments):
        assert exposure_by({}, instruments, "issuer") == []

    def test_unknown_dimension_refused(self, instruments):
        with pytest.raises(ValueError, match="unknown exposure dimension"):
            exposure_by(VALUES, instruments, "colour")

    def test_isins_are_sorted_within_a_slice(self, instruments):
        slices = exposure_by({B: 100.0, A: 100.0}, instruments, "asset_class")
        assert slices[0].isins == (A, B)


class TestExposures:
    def test_every_dimension(self, instruments):
        got = exposures(VALUES, instruments)
        assert tuple(got) == DIMENSIONS
        assert got["asset_class"] == exposure_by(VALUES, instruments, "asset_class")
