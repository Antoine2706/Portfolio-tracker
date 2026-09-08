"""From a provider's bars to a number the cost model will charge.

`test_spread.py` tests the estimator. This tests everything between it and
the cost model: fetching bars, caching them, deciding whether the estimate is
good enough to use, and the plausibility check on the result as a whole.

The end-to-end test here is the one worth reading. The fixture provider
imposes a per-instrument spread derived from the symbol's hash, and nothing
downstream of it is told what that spread was; the assertion is that the
number coming out the far end of the cache and the tier logic is the number
that went in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.spreads import (ASSUMED, ESTIMATED, decide_spread,
                                      ranking_is_plausible)
from portfolio.core.spread import SpreadEstimate, TickSize, edge, infer_tick_size
from portfolio.data.cache import PriceCache
from portfolio.data.provider import MarketDataProvider, ProviderError
from portfolio.data.providers.fixture import FixtureProvider

SYMBOLS = ["IWDA.AS", "EUDF.DE", "AIGG.MI", "WEAT.MI", "GLUX.PA", "ISAE.AS",
           "MEUD.PA", "SMEA.MI", "ESIE.DE", "AIGE.MI"]


@pytest.fixture
def provider() -> FixtureProvider:
    return FixtureProvider()


class TestTheProviderSuppliesBars:
    def test_the_shape_is_ohlcv(self, provider):
        frame = provider.bars("IWDA.AS")
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(frame.index, pd.DatetimeIndex)

    def test_the_range_contains_the_open_and_the_close(self, provider):
        f = provider.bars("AIGG.MI")
        assert (f["high"] >= f[["open", "close"]].max(axis=1)).all()
        assert (f["low"] <= f[["open", "close"]].min(axis=1)).all()

    def test_a_provider_without_bars_declines_rather_than_not_existing(self):
        """`bars` is concrete on the interface so a close-only provider stays
        a valid provider. It must say so rather than returning something."""

        class CloseOnly(MarketDataProvider):
            name = "close-only"

            def probe(self, symbol, lookback_days=252): ...
            def history(self, symbol, start=None): ...
            def quote(self, symbol): ...

        with pytest.raises(ProviderError, match="does not supply"):
            CloseOnly().bars("ANY")

    def test_bars_are_unadjusted_so_the_tick_grid_survives(self, provider):
        """The whole reason `bars` is a separate call from `history`."""
        frame = provider.bars("IWDA.AS")
        tick = infer_tick_size(
            frame[["open", "high", "low", "close"]].to_numpy().ravel())
        assert tick.usable and tick.size == pytest.approx(0.01)


class TestTheCache:
    def test_a_round_trip_returns_what_went_in(self, tmp_path, provider):
        cache = PriceCache(tmp_path / "bars.sqlite")
        original = provider.bars("AIGG.MI")
        cache.put_bars("AIGG.MI", original)
        back, _ = cache.get_bars("AIGG.MI")
        pd.testing.assert_frame_equal(back, original, check_freq=False,
                                      check_names=False)
        cache.close()

    def test_an_upsert_keeps_the_rows_it_was_not_given(self, tmp_path, provider):
        cache = PriceCache(tmp_path / "bars.sqlite")
        whole = provider.bars("AIGG.MI")
        cache.put_bars("AIGG.MI", whole)
        cache.put_bars("AIGG.MI", whole.tail(5) * 1.0)
        back, _ = cache.get_bars("AIGG.MI")
        assert len(back) == len(whole)
        cache.close()

    def test_a_bar_missing_a_price_is_not_stored(self, tmp_path, provider):
        """Copying the close into the other three would manufacture a
        zero-range day, which the estimator reads as 'did not trade'. Same
        answer, reached by inventing data."""
        cache = PriceCache(tmp_path / "bars.sqlite")
        frame = provider.bars("AIGG.MI").copy()
        frame.iloc[3, frame.columns.get_loc("high")] = np.nan
        cache.put_bars("AIGG.MI", frame)
        back, _ = cache.get_bars("AIGG.MI")
        assert len(back) == len(frame) - 1
        cache.close()

    def test_a_bar_missing_only_volume_is_kept(self, tmp_path, provider):
        """Volume feeds a check on the estimate. Dropping the bar would trade
        the estimate away for the thing that verifies it."""
        cache = PriceCache(tmp_path / "bars.sqlite")
        frame = provider.bars("AIGG.MI").copy()
        frame["volume"] = np.nan
        cache.put_bars("AIGG.MI", frame)
        back, _ = cache.get_bars("AIGG.MI")
        assert len(back) == len(frame)
        assert back["volume"].isna().all()
        cache.close()

    def test_rows_written_before_volume_existed_are_distinguishable(
            self, tmp_path, provider):
        """Three states, not two: has volume, venue reports none, and written
        before the column existed. A cache that cannot tell the last two apart
        either refetches every run for ever or never refetches at all.
        """
        path = tmp_path / "bars.sqlite"
        cache = PriceCache(path)
        cache.put_bars("AIGG.MI", provider.bars("AIGG.MI"))
        assert not cache.bars_predate_volume("AIGG.MI")

        no_volume = provider.bars("WEAT.MI").copy()
        no_volume["volume"] = np.nan
        cache.put_bars("WEAT.MI", no_volume)
        assert not cache.bars_predate_volume("WEAT.MI")
        cache.close()

        # Now forge the pre-upgrade state: rows present, flag never set.
        import sqlite3
        conn = sqlite3.connect(path)
        conn.execute("UPDATE bars_meta SET had_volume = NULL "
                     "WHERE symbol = 'AIGG.MI'")
        conn.commit()
        conn.close()
        reopened = PriceCache(path)
        assert reopened.bars_predate_volume("AIGG.MI")
        assert not reopened.bars_predate_volume("WEAT.MI")
        reopened.close()

    def test_a_cache_written_without_the_column_still_opens(self, tmp_path):
        """CREATE TABLE IF NOT EXISTS does nothing to a table that exists, so
        an old cache keeps the old shape and every query naming the new column
        fails. This is how that was found, and it must not come back."""
        import sqlite3
        path = tmp_path / "old.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript(
            "CREATE TABLE bars (symbol TEXT NOT NULL, date TEXT NOT NULL, "
            "open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, "
            "close REAL NOT NULL, PRIMARY KEY (symbol, date));"
            "CREATE TABLE bars_meta (symbol TEXT PRIMARY KEY, "
            "fetched_at TEXT NOT NULL, first_date TEXT, last_date TEXT, "
            "rows INTEGER);"
            "INSERT INTO bars VALUES ('X', '2026-01-02', 1, 2, 0.5, 1.5);"
            "INSERT INTO bars_meta VALUES ('X', '2026-01-02T00:00:00.000000+00:00',"
            " '2026-01-02', '2026-01-02', 1);")
        conn.commit()
        conn.close()

        cache = PriceCache(path)
        back, _ = cache.get_bars("X")
        assert len(back) == 1 and back["volume"].isna().all()
        assert cache.bars_predate_volume("X")
        cache.close()


class TestTheTiers:
    def test_a_resolved_estimate_is_charged(self):
        good = SpreadEstimate(0.0040, 1.6e-5, 0.0004, 500, 499,
                              square_standard_error=3.2e-6)
        d = decide_spread(good, price=50.0)
        assert d.source == ESTIMATED and d.is_evidence
        assert d.half_spread_bps == pytest.approx(20.0)

    def test_an_unresolved_estimate_is_discarded_not_shaded(self):
        """The number is thrown away rather than used with a caveat. A 4 bps
        estimate at the noise floor and a genuine 4 bps spread are the same
        number, and only one of them is a measurement."""
        weak = SpreadEstimate(0.0008, 6.4e-7, 0.0006, 500, 499,
                              square_standard_error=9.6e-7)
        d = decide_spread(weak, price=50.0, fallback_bps=8.0)
        assert d.source == ASSUMED and not d.is_evidence
        assert d.half_spread_bps == 8.0

    def test_a_refusal_passes_through_as_the_constant(self):
        nothing = SpreadEstimate(None, None, None, 30, 29,
                                 refusal="only 29 usable bars")
        d = decide_spread(nothing, price=50.0, fallback_bps=8.0)
        assert d.source == ASSUMED and "29 usable bars" in d.reason

    def test_the_tick_floor_binds_and_is_recorded(self):
        """A 4 EUR instrument on a cent grid cannot quote under 12.5 bps
        half-spread. An estimate of 6 there is the estimator finding no
        bounce, not a tight market."""
        narrow = SpreadEstimate(0.0012, 1.44e-6, 0.00006, 500, 499,
                                square_standard_error=1.4e-7)
        tick = TickSize(size=0.01, agreement=1.0, prices=2000)
        d = decide_spread(narrow, price=4.0, tick=tick)
        assert d.clamped_to_tick
        assert d.half_spread_bps == pytest.approx(12.5)
        assert d.source == ESTIMATED           # still evidence, but bounded

    def test_the_tick_floor_does_not_bind_where_it_should_not(self):
        wide = SpreadEstimate(0.0060, 3.6e-5, 0.0004, 500, 499,
                              square_standard_error=4.8e-6)
        tick = TickSize(size=0.01, agreement=1.0, prices=2000)
        d = decide_spread(wide, price=100.0, tick=tick)
        assert not d.clamped_to_tick
        assert d.half_spread_bps == pytest.approx(30.0)

    def test_resolution_and_the_tick_are_independent_gates(self):
        """Each catches instruments the other does not, which is why there
        are two. Resolution is about the sample; the tick is about the venue.
        """
        narrow_but_certain = SpreadEstimate(0.0012, 1.44e-6, 0.00006, 500, 499,
                                            square_standard_error=1.4e-7)
        tick = TickSize(size=0.01, agreement=1.0, prices=2000)
        # Resolved, clamped by the tick.
        assert decide_spread(narrow_but_certain, price=4.0,
                             tick=tick).clamped_to_tick
        # Same estimate, a price where the tick is irrelevant: untouched.
        assert not decide_spread(narrow_but_certain, price=400.0,
                                 tick=tick).clamped_to_tick


class TestTheRankingCheck:
    def test_it_passes_when_liquidity_predicts_the_ordering(self):
        check = ranking_is_plausible(
            {"A": 4.0, "B": 8.0, "C": 20.0, "D": 55.0},
            {"A": 900.0, "B": 400.0, "C": 90.0, "D": 5.0})
        assert check.passed and check.rho == pytest.approx(-1.0)

    def test_it_fails_when_the_most_liquid_are_the_widest(self):
        """Which is what a symbol mapped to the wrong listing looks like."""
        check = ranking_is_plausible(
            {"A": 4.0, "B": 8.0, "C": 20.0, "D": 55.0},
            {"A": 5.0, "B": 90.0, "C": 400.0, "D": 900.0})
        assert not check.passed
        assert "backwards" in check.verdict

    def test_it_reports_weakness_rather_than_claiming_a_pass(self):
        """Unranked is not the same as confirmed. Over four instruments
        Spearman has almost no power, and the check says so rather than
        letting a pass stand in for evidence."""
        check = ranking_is_plausible(
            {"A": 4.0, "B": 8.0, "C": 20.0, "D": 55.0},
            {"A": 400.0, "B": 900.0, "C": 90.0, "D": 500.0})
        assert check.rho == pytest.approx(0.0)
        assert check.passed and "weak evidence" in check.verdict

    def test_too_few_instruments_says_so(self):
        check = ranking_is_plausible({"A": 4.0, "B": 8.0}, {"A": 9.0, "B": 4.0})
        assert check.rho is None and "too few" in check.verdict


class TestEndToEnd:
    """The fixture imposes a spread; nothing downstream is told what it was."""

    # Above the estimator's measured noise floor of roughly 4 bps, with room
    # to spare. Below this the estimator is not unbiased and the controls say
    # so; asserting otherwise here would be asserting the opposite of what was
    # measured, in a place where it would look like a stronger result.
    RESOLVABLE_BPS = 8.0

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_the_number_out_is_the_number_in(self, provider, symbol):
        imposed = provider.fixture_half_spread_bps(symbol)
        if imposed < self.RESOLVABLE_BPS:
            pytest.skip(f"{imposed:.1f} bps is at the estimator's noise floor; "
                        f"covered by the test below instead")
        frame = provider.bars(symbol)
        estimate = edge(frame["open"].to_numpy(), frame["high"].to_numpy(),
                        frame["low"].to_numpy(), frame["close"].to_numpy())
        assert estimate.spread is not None
        off = abs(estimate.half_spread_bps - imposed) / estimate.half_spread_error_bps
        assert off < 3.0, (f"{symbol}: imposed {imposed:.1f}, got "
                           f"{estimate.half_spread_bps:.1f} +/- "
                           f"{estimate.half_spread_error_bps:.1f} ({off:.1f} SE)")

    def test_a_spread_below_the_floor_is_declined_rather_than_reported(
            self, provider):
        """The other half, and the one that matters more.

        ESIE.DE has 4.7 bps imposed and estimates at 10.3 -- more than three
        standard errors out, because below the noise floor the estimator is
        not centred on the truth and the controls measured that. What must
        hold is not that the number is right; it is that the number never
        reaches the cost model. It does not: its squared estimate comes out
        negative, so the significance test on s^2 rejects it outright.
        """
        below = [s for s in SYMBOLS
                 if provider.fixture_half_spread_bps(s) < self.RESOLVABLE_BPS]
        assert below, "the fixture must contain at least one sub-floor spread"
        for symbol in below:
            frame = provider.bars(symbol)
            e = edge(frame["open"].to_numpy(), frame["high"].to_numpy(),
                     frame["low"].to_numpy(), frame["close"].to_numpy())
            d = decide_spread(e, price=float(frame["close"].iloc[-1]))
            assert d.source == ASSUMED, (
                f"{symbol}: {provider.fixture_half_spread_bps(symbol):.1f} bps "
                f"imposed came back as {d.half_spread_bps:.1f} bps of "
                f"'{d.source}', which the floor says it cannot be")

    def test_it_survives_the_cache(self, tmp_path, provider):
        """Bars that went through SQLite must give the same estimate as bars
        that did not. A float column stored and read back is exactly the sort
        of thing that quietly loses a digit."""
        cache = PriceCache(tmp_path / "bars.sqlite")
        direct = provider.bars("WEAT.MI")
        cache.put_bars("WEAT.MI", direct)
        stored, _ = cache.get_bars("WEAT.MI")
        cache.close()

        def estimate(f):
            return edge(f["open"].to_numpy(), f["high"].to_numpy(),
                        f["low"].to_numpy(), f["close"].to_numpy()).spread

        assert estimate(stored) == pytest.approx(estimate(direct), rel=1e-12)

    def test_the_fixture_spreads_are_not_all_the_same(self, provider):
        """If they were, the whole survey would pass while proving nothing."""
        imposed = [provider.fixture_half_spread_bps(s) for s in SYMBOLS]
        assert max(imposed) > 3.0 * min(imposed)

    def test_the_book_splits_across_the_tiers(self, provider):
        """The offline demo must contain instruments the estimator resolves
        and instruments it declines. A fixture where everything resolves would
        never exercise the tier that keeps the constant."""
        sources = []
        for symbol in SYMBOLS:
            frame = provider.bars(symbol)
            prices = frame[["open", "high", "low", "close"]]
            e = edge(prices["open"].to_numpy(), prices["high"].to_numpy(),
                     prices["low"].to_numpy(), prices["close"].to_numpy())
            sources.append(decide_spread(
                e, price=float(frame["close"].iloc[-1]),
                tick=infer_tick_size(prices.to_numpy().ravel())).source)
        assert ESTIMATED in sources and ASSUMED in sources

    def test_the_ranking_check_finds_the_relation_the_fixture_imposed(
            self, provider):
        """End to end and independent: the fixture makes volume inversely
        related to spread, and the check has to find that through the
        estimator without being told."""
        spreads, liquidity = {}, {}
        for symbol in SYMBOLS:
            frame = provider.bars(symbol)
            e = edge(frame["open"].to_numpy(), frame["high"].to_numpy(),
                     frame["low"].to_numpy(), frame["close"].to_numpy())
            d = decide_spread(e, price=float(frame["close"].iloc[-1]))
            if d.is_evidence:
                spreads[symbol] = d.half_spread_bps
                liquidity[symbol] = float(
                    (frame["volume"] * frame["close"]).median())
        check = ranking_is_plausible(spreads, liquidity)
        assert check.passed
        assert check.rho is not None and check.rho < -0.4, check.rho
