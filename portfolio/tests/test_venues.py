"""Venue code translation, validated against the stored venue map.

The seed instruments carry ten Yahoo symbols confirmed by the 3 September 2026
matrix run. Rather than trusting a hand-written code table, the table is
asserted to reproduce every one of them. A mismatch means either the mapping is
wrong or the stored symbol is, and both are worth knowing -- which makes the
seed data a regression test for the resolver rather than only demo data.
"""

from __future__ import annotations

import pathlib

import pytest

from portfolio.data.store import DataStore, DataMode
from portfolio.data.venues import (PRIMARY_VENUES, VenueClass, bloomberg_to_yahoo,
                                   classify_bloomberg_code, is_us_yahoo_exchange,
                                   yahoo_suffix_to_mic)

SEED_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data_store"

# The (ticker, Bloomberg exchange code) OpenFIGI is expected to return for each
# instrument's confirmed primary listing. Verified live by spike/check_openfigi.py.
EXPECTED_FIGI_IDENTITY = {
    "IE0002Y8CX98": ("EUDF", "GR"),
    "IE000IAXNM41": ("DFNC", "GR"),
    "IE000I7E6HL0": ("8RMY", "GR"),
    "IE000OJ5TQP4": ("ASWC", "GR"),
    "IE00B6R52143": ("ISAE", "NA"),
    "GB00B15KYL00": ("AIGG", "IM"),
    "JE00BN7KB664": ("WEAT", "IM"),
    "GB00B15KYB02": ("AIGE", "IM"),
    "IE00BMW42637": ("ESIE", "GR"),
    "LU1681048630": ("GLUX", "FP"),
}


@pytest.fixture(scope="module")
def seed_instruments():
    return DataStore(mode=DataMode.SEED, root=SEED_ROOT).load_instruments()


class TestVenueMapRegression:
    def test_every_stored_symbol_is_reproducible(self, seed_instruments):
        for isin, (ticker, code) in EXPECTED_FIGI_IDENTITY.items():
            built = bloomberg_to_yahoo(ticker, code)
            stored = seed_instruments[isin].provider_symbols["yfinance"]
            assert built == stored, f"{isin}: {ticker}+{code} -> {built}, stored {stored}"

    def test_every_stored_mic_is_reproducible(self, seed_instruments):
        for isin, (ticker, code) in EXPECTED_FIGI_IDENTITY.items():
            built = bloomberg_to_yahoo(ticker, code)
            assert yahoo_suffix_to_mic(built) == seed_instruments[isin].exchange, isin

    def test_the_map_covers_every_seed_instrument(self, seed_instruments):
        assert set(EXPECTED_FIGI_IDENTITY) == set(seed_instruments)

    def test_all_four_venues_in_the_map_are_exercised(self):
        codes = {c for _, c in EXPECTED_FIGI_IDENTITY.values()}
        assert codes == {"GR", "NA", "IM", "FP"}


class TestClassification:
    @pytest.mark.parametrize("code", ["LN", "IM", "GR", "FP", "NA", "SW", "SE"])
    def test_primary_venues(self, code):
        assert classify_bloomberg_code(code) is VenueClass.PRIMARY

    @pytest.mark.parametrize("code", ["GF", "GD", "GS", "GM", "GH", "GT", "GZ", "TH"])
    def test_german_regionals_and_tradegate_excluded(self, code):
        assert classify_bloomberg_code(code) is VenueClass.GERMAN_REGIONAL
        assert bloomberg_to_yahoo("EUDF", code) is None

    @pytest.mark.parametrize("code", ["EP", "EZ", "EO", "X2", "XH", "XF", "XJ"])
    def test_mtfs_and_dark_venues_excluded(self, code):
        assert classify_bloomberg_code(code) is VenueClass.MTF
        assert bloomberg_to_yahoo("WDEFEUR", code) is None

    @pytest.mark.parametrize("code", ["US", "UN", "UQ", "UP", "UW"])
    def test_us_bloomberg_codes_classified_us(self, code):
        assert classify_bloomberg_code(code) is VenueClass.US
        assert bloomberg_to_yahoo("WEAT", code) is None

    def test_unknown_code_is_other_and_yields_nothing(self):
        assert classify_bloomberg_code("ZZ") is VenueClass.OTHER
        assert bloomberg_to_yahoo("WDEF", "ZZ") is None

    def test_allowlist_not_blocklist(self):
        """A code nobody has classified must still fail closed."""
        assert bloomberg_to_yahoo("WDEF", "QQ") is None


class TestUsGate:
    @pytest.mark.parametrize("exchange", ["PCX", "NGM", "NYQ", "NMS", "PNK"])
    def test_the_five_venues_from_the_matrix(self, exchange):
        """Every bare colliding ticker resolved to one of these with healthy data."""
        assert is_us_yahoo_exchange(exchange)

    @pytest.mark.parametrize("exchange", ["GER", "MIL", "AMS", "PAR", "LSE", "EBS"])
    def test_european_venues_pass(self, exchange):
        assert not is_us_yahoo_exchange(exchange)

    def test_case_and_spacing_insensitive(self):
        assert is_us_yahoo_exchange("nyse") and is_us_yahoo_exchange("NYSE ARCA")

    def test_missing_exchange_is_not_treated_as_us(self):
        """Absence of a venue is a data gap, not a refusal reason."""
        assert not is_us_yahoo_exchange(None) and not is_us_yahoo_exchange("")
