"""The add-instrument resolution pipeline.

Probe results below are the ones actually observed on 2026-09-03 wherever the
matrix recorded them, including the five colliding US tickers and the two-row
LSE listings. Inventing gentler data would test a problem we do not have.
"""

from __future__ import annotations

import datetime as dt

import pytest

from portfolio.core.models import AssetClass, Instrument
from portfolio.data.provider import FigiListing, IdentityProvider, ListingProbe, MarketDataProvider
from portfolio.data.providers.openfigi import parse_mapping_response
from portfolio.data.resolve import (Verdict, candidates_from_listings,
                                    instrument_from_candidate, rank_candidates,
                                    resolve_isin)
from portfolio.tests.fixtures_openfigi import (NO_MATCH_RESPONSE,
                                               WDEF_MAPPING_RESPONSE, wdef_listings)

WDEF_ISIN = "IE0002Y8CX98"
WHEAT_ISIN = "JE00BN7KB664"
D = dt.date


# Observed probe results, keyed by Yahoo symbol.
PROBES: dict[str, ListingProbe] = {
    # The confirmed primary listing: 377 days, Xetra, EUR.
    "EUDF.DE": ListingProbe("EUDF.DE", True, "GER", "EUR", "WisdomTree Europe Defence",
                            377, D(2025, 3, 3), D(2026, 9, 3), adjusted=True),
    # London and Milan lines: real, but foreign currency.
    "WDEF.L": ListingProbe("WDEF.L", True, "LSE", "USD", "WisdomTree Europe Defence",
                           500, D(2024, 9, 3), D(2026, 9, 3), adjusted=True),
    "WDEF.MI": ListingProbe("WDEF.MI", True, "MIL", "EUR", "WisdomTree Europe Defence",
                            300, D(2025, 6, 3), D(2026, 9, 3), adjusted=True),
    # The pence line.
    "WDEP.L": ListingProbe("WDEP.L", True, "LSE", "GBp", "WisdomTree Europe Defence",
                           420, D(2024, 12, 3), D(2026, 9, 3), adjusted=True),
    "WDEFM.L": ListingProbe("WDEFM.L", False, error="no history returned"),
    "WDEFL.L": ListingProbe("WDEFL.L", False, error="no history returned"),
    "WDEPL.L": ListingProbe("WDEPL.L", False, error="no history returned"),
    "EUDFD.DE": ListingProbe("EUDFD.DE", False, error="no history returned"),
    # Trap 2: resolves, European venue, right shape, two rows.
    "AIGG.L": ListingProbe("AIGG.L", True, "LSE", "USD", "WisdomTree Grains",
                           2, D(2026, 7, 17), D(2026, 9, 3), adjusted=True),
    "AIGE.L": ListingProbe("AIGE.L", True, "LSE", "USD", "WisdomTree Energy",
                           2, D(2026, 7, 17), D(2026, 9, 3), adjusted=True),
    "AIGG.MI": ListingProbe("AIGG.MI", True, "MIL", "EUR", "WisdomTree Grains",
                            503, D(2024, 9, 3), D(2026, 9, 3), adjusted=True),
    # Trap 3: ETCs return nothing on Xetra.
    "D7Y0.DE": ListingProbe("D7Y0.DE", False, error="no data, 404 on quoteSummary"),
    # Trap 1: bare colliding tickers, all healthy, all wrong.
    "WEAT": ListingProbe("WEAT", True, "PCX", "USD", "Teucrium Wheat Fund",
                         502, D(2024, 9, 3), D(2026, 9, 3), adjusted=True),
    "GLUX": ListingProbe("GLUX", True, "PNK", "USD", "Great Lakes Aviation, Ltd.",
                         502, D(2024, 9, 3), D(2026, 9, 3), adjusted=True),
    "NATO": ListingProbe("NATO", True, "NGM", "USD", "Themes Transatlantic Defense ETF",
                         474, D(2024, 11, 3), D(2026, 9, 3), adjusted=True),
    "WEAT.MI": ListingProbe("WEAT.MI", True, "MIL", "EUR", "WisdomTree Wheat",
                            503, D(2024, 9, 3), D(2026, 9, 3), adjusted=True),
}


class FakeIdentity(IdentityProvider):
    name = "openfigi"

    def __init__(self, listings=None, raises=None):
        self._listings = listings if listings is not None else wdef_listings()
        self._raises = raises

    def listings_for_isin(self, isin):
        if self._raises:
            raise self._raises
        return self._listings


class FakePrices(MarketDataProvider):
    name = "yfinance"

    def __init__(self):
        self.probed: list[str] = []

    def probe(self, symbol, lookback_days=252):
        self.probed.append(symbol)
        return PROBES.get(symbol, ListingProbe(symbol, False, error="unknown symbol"))

    def history(self, symbol, start=None):
        raise NotImplementedError

    def quote(self, symbol):
        raise NotImplementedError


class TestParsingAndFiltering:
    def test_the_fixture_is_realistically_noisy(self):
        listings = parse_mapping_response(WDEF_MAPPING_RESPONSE)
        assert len(listings) > 60, "a tidy fixture would not test the filtering"

    def test_no_match_response_yields_nothing(self):
        assert parse_mapping_response(NO_MATCH_RESPONSE) == []

    def test_seventy_listings_collapse_to_a_handful(self):
        candidates, filtered = candidates_from_listings(WDEF_ISIN, wdef_listings())
        assert len(candidates) <= 8, [c.yahoo_symbol for c in candidates]
        assert sum(filtered.values()) > 40

    def test_the_confirmed_primary_survives_filtering(self):
        candidates, _ = candidates_from_listings(WDEF_ISIN, wdef_listings())
        assert "EUDF.DE" in {c.yahoo_symbol for c in candidates}

    def test_filtered_counts_are_reported_by_class(self):
        _, filtered = candidates_from_listings(WDEF_ISIN, wdef_listings())
        assert filtered["german_regional"] == 8
        assert filtered["mtf"] > 30

    def test_duplicate_symbols_collapse(self):
        listings = [FigiListing("A", "EUDF", "GR"), FigiListing("B", "EUDF", "GR")]
        candidates, _ = candidates_from_listings(WDEF_ISIN, listings)
        assert len(candidates) == 1

    def test_us_listings_never_become_candidates(self):
        listings = [FigiListing("X", "WEAT", "UN"), FigiListing("Y", "WEAT", "IM")]
        candidates, filtered = candidates_from_listings(WHEAT_ISIN, listings)
        assert [c.yahoo_symbol for c in candidates] == ["WEAT.MI"]
        assert filtered["us"] == 1


class TestHardUsGate:
    """Trap 1. Every one of these returned a real security with clean history."""

    @pytest.mark.parametrize("symbol,exchange,days", [
        ("WEAT", "PCX", 502), ("GLUX", "PNK", 502), ("NATO", "NGM", 474)])
    def test_us_listing_refused_however_good_the_series(self, symbol, exchange, days):
        known = {WHEAT_ISIN: Instrument(WHEAT_ISIN, "WisdomTree Wheat", AssetClass.ETC,
                                        "USD", provider_symbols={"yfinance": symbol})}
        res = resolve_isin(WHEAT_ISIN, FakeIdentity(), FakePrices(), known=known)
        assert res.candidates[0].verdict is Verdict.REFUSED
        assert not res.candidates[0].selectable_as_primary
        assert res.recommended is None

    def test_refusal_explains_that_a_good_series_is_the_danger(self):
        known = {WHEAT_ISIN: Instrument(WHEAT_ISIN, "WisdomTree Wheat", AssetClass.ETC,
                                        "USD", provider_symbols={"yfinance": "WEAT"})}
        res = resolve_isin(WHEAT_ISIN, FakeIdentity(), FakePrices(), known=known)
        assert "different fund sharing the ticker" in res.candidates[0].reasons[0]

    def test_refused_is_the_lowest_rank_not_merely_penalised(self):
        from portfolio.data.resolve import Candidate
        refused = Candidate(WHEAT_ISIN, "WEAT", "", "WEAT", "US", None,
                            Verdict.REFUSED, observations=502, currency="USD")
        thin = Candidate(WHEAT_ISIN, "AIGG", "LN", "AIGG.L", "London", "XLON",
                         Verdict.THIN, observations=2, currency="USD")
        assert rank_candidates([refused, thin])[0] is thin


class TestThinVerdict:
    """Trap 2. AIGG.L and AIGE.L resolve cleanly and return two rows."""

    def test_two_row_listing_is_thin_not_pass(self):
        listings = [FigiListing("A", "AIGG", "LN")]
        res = resolve_isin("GB00B15KYL00", FakeIdentity(listings), FakePrices())
        assert res.candidates[0].verdict is Verdict.THIN

    def test_thin_is_never_selectable_as_primary(self):
        listings = [FigiListing("A", "AIGG", "LN")]
        res = resolve_isin("GB00B15KYL00", FakeIdentity(listings), FakePrices())
        assert not res.candidates[0].selectable_as_primary
        assert res.recommended is None
        assert res.blocked

    def test_a_full_listing_outranks_a_thin_one(self):
        listings = [FigiListing("A", "AIGG", "LN"), FigiListing("B", "AIGG", "IM")]
        res = resolve_isin("GB00B15KYL00", FakeIdentity(listings), FakePrices())
        assert res.recommended.yahoo_symbol == "AIGG.MI"

    def test_short_of_lookback_is_thin_even_when_usable(self):
        """300 days is above the 60-observation floor but below a 400 lookback."""
        listings = [FigiListing("A", "WDEF", "IM")]
        res = resolve_isin(WDEF_ISIN, FakeIdentity(listings), FakePrices(), lookback=400)
        assert res.candidates[0].verdict is Verdict.THIN
        assert "constrain the whole window" in res.candidates[0].reasons[0]

    def test_block_reason_names_the_longest_thin_option(self):
        listings = [FigiListing("A", "AIGG", "LN")]
        res = resolve_isin("GB00B15KYL00", FakeIdentity(listings), FakePrices())
        assert "AIGG.L" in res.block_reason() and "2 observations" in res.block_reason()


class TestRanking:
    def test_eur_outranks_foreign_currency_even_with_less_history(self):
        """EUDF.DE has 377 days in EUR; WDEF.L has 500 in USD. EUR wins."""
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices())
        assert res.recommended.yahoo_symbol == "EUDF.DE"

    def test_foreign_currency_candidate_is_flagged_not_hidden(self):
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices())
        usd = next(c for c in res.candidates if c.yahoo_symbol == "WDEF.L")
        assert any("FX conversion" in r for r in usd.reasons)

    def test_pence_line_ranks_below_the_euro_line(self):
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices())
        symbols = [c.yahoo_symbol for c in res.candidates]
        assert symbols.index("EUDF.DE") < symbols.index("WDEP.L")

    def test_failed_probes_rank_below_everything_usable(self):
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices())
        verdicts = [c.verdict for c in res.candidates]
        assert verdicts == sorted(verdicts, key=lambda v: [Verdict.PASS, Verdict.THIN,
                                                           Verdict.FAILED].index(v))

    def test_only_primary_venues_are_probed(self):
        """Probing costs a call each; the 60+ filtered rows must never be probed."""
        prices = FakePrices()
        resolve_isin(WDEF_ISIN, FakeIdentity(), prices)
        assert len(prices.probed) <= 8


class TestStoredMapShortCircuit:
    def _known(self, overridden=False):
        inst = Instrument(WDEF_ISIN, "WisdomTree Europe Defence", AssetClass.ETF, "EUR",
                          exchange="XETR", quote_currency="EUR",
                          provider_symbols={"yfinance": "EUDF.DE"})
        if overridden:
            inst.override("provider_symbols.yfinance", "EUDF.DE")
        return {WDEF_ISIN: inst}

    def test_known_isin_does_not_call_the_identity_provider(self):
        identity = FakeIdentity(raises=AssertionError("must not be called"))
        res = resolve_isin(WDEF_ISIN, identity, FakePrices(), known=self._known())
        assert res.from_stored_map
        assert res.recommended.yahoo_symbol == "EUDF.DE"

    def test_stored_symbol_is_still_probed_not_blindly_trusted(self):
        prices = FakePrices()
        resolve_isin(WDEF_ISIN, FakeIdentity(), prices, known=self._known())
        assert prices.probed == ["EUDF.DE"]

    def test_manual_override_is_announced(self):
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices(),
                           known=self._known(overridden=True))
        assert any("set by hand" in r for r in res.candidates[0].reasons)

    def test_unknown_isin_falls_through_to_identity_lookup(self):
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices(), known={})
        assert not res.from_stored_map


class TestRefusalsAndErrors:
    def test_no_listings_at_all(self):
        res = resolve_isin(WDEF_ISIN, FakeIdentity([]), FakePrices())
        assert res.blocked and "No listing found" in res.block_reason()

    def test_only_non_primary_listings(self):
        listings = [FigiListing("A", "EUDF", "TH"), FigiListing("B", "WDEFEUR", "X2")]
        res = resolve_isin(WDEF_ISIN, FakeIdentity(listings), FakePrices())
        assert res.blocked
        reason = res.block_reason()
        assert "german regional" in reason and "mtf" in reason

    def test_identity_provider_failure_is_reported_not_raised(self):
        res = resolve_isin(WDEF_ISIN, FakeIdentity(raises=RuntimeError("boom")),
                           FakePrices())
        assert res.blocked and res.errors and "boom" in res.errors[0]

    def test_probe_exception_becomes_a_failed_candidate(self):
        class Exploding(FakePrices):
            def probe(self, symbol, lookback_days=252):
                raise RuntimeError("probe exploded")
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), Exploding())
        assert all(c.verdict is Verdict.FAILED for c in res.candidates)
        assert res.blocked

    def test_etc_absent_on_xetra_falls_through(self):
        """Trap 3: a venue that works for every ETF returns nothing for the ETCs."""
        listings = [FigiListing("A", "D7Y0", "GR"), FigiListing("B", "AIGG", "IM")]
        res = resolve_isin("GB00B15KYL00", FakeIdentity(listings), FakePrices())
        assert res.recommended.yahoo_symbol == "AIGG.MI"
        failed = next(c for c in res.candidates if c.yahoo_symbol == "D7Y0.DE")
        assert failed.verdict is Verdict.FAILED


class TestConfirmScreenPayload:
    def test_describe_carries_everything_the_wdef_case_needs(self):
        """Name and sector cannot separate the two WDEF funds. ISIN and
        exchange must both be on screen."""
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices())
        line = res.recommended.describe()
        for fragment in ["EUDF.DE", "GER", "EUR", "377 days", WDEF_ISIN]:
            assert fragment in line

    def test_summary_states_what_was_discarded(self):
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices())
        assert "listings seen" in res.summary() and "filtered" in res.summary()

    def test_instrument_built_only_from_a_confirmed_candidate(self):
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices())
        inst = instrument_from_candidate(res.recommended, "yfinance", base_currency="EUR",
                                         issuer="WisdomTree", asset_class="ETF")
        assert inst.isin == WDEF_ISIN
        assert inst.provider_symbols == {"yfinance": "EUDF.DE"}
        assert inst.exchange == "XETR" and inst.quote_currency == "EUR"

    def test_base_currency_is_collected_not_inferred(self):
        """ISAE.AS quotes EUR for a USD-base fund; the listing cannot tell you."""
        res = resolve_isin(WDEF_ISIN, FakeIdentity(), FakePrices())
        inst = instrument_from_candidate(res.recommended, "yfinance", base_currency="USD")
        assert inst.base_currency == "USD" and inst.quote_currency == "EUR"
