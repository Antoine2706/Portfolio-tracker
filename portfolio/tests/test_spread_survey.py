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

from portfolio.agents.spreads import (ASSUMED, BOUNDED, ESTIMATED,
                                      decide_spread, ranking_is_plausible)
from portfolio.core.spread import SpreadEstimate, TickSize, edge, infer_tick_size
from portfolio.data.cache import PriceCache
from portfolio.data.provider import MarketDataProvider, ProviderError
from portfolio.data.providers.fixture import FixtureProvider
from portfolio.eval.spread_controls import simulate_bars
from portfolio.research import tick_for

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

    def test_an_unresolved_estimate_becomes_a_ceiling_not_a_number(self):
        """The point estimate is not charged -- a 4 bps reading at the noise
        floor and a genuine 4 bps spread are the same number, and only one is
        a measurement. But the sample still bounds the spread from above, and
        that bound is per instrument, so it beats the constant."""
        weak = SpreadEstimate(0.0008, 6.4e-7, 0.0006, 500, 499,
                              square_standard_error=9.6e-7)
        d = decide_spread(weak, price=50.0, fallback_bps=8.0)
        assert d.source == BOUNDED
        assert d.is_evidence and not d.is_measurement
        assert d.half_spread_bps != pytest.approx(weak.half_spread_bps)
        assert d.half_spread_bps > weak.half_spread_bps      # it is a ceiling
        # sqrt(s^2 + 2 SE) in half-bps: sqrt(6.4e-7 + 1.92e-6) * 1e4 / 2
        assert d.half_spread_bps == pytest.approx(8.0, abs=0.05)

    def test_the_bound_is_per_instrument_which_is_the_whole_point(self):
        """Two instruments that both fail the significance test must still
        get different numbers, or the exercise has delivered nothing on a
        book where nothing resolves."""
        # Both unresolved: t = 1.6 and t = 0.4, either side of nothing and
        # both under the threshold of 2. Only the standard error differs.
        quiet = SpreadEstimate(0.0008, 6.4e-7, 0.0006, 2000, 1999,
                               square_standard_error=4.0e-7)
        noisy = SpreadEstimate(0.0008, 6.4e-7, 0.0006, 300, 299,
                               square_standard_error=1.6e-6)
        a = decide_spread(quiet, price=50.0)
        b = decide_spread(noisy, price=50.0)
        assert a.source == b.source == BOUNDED
        assert b.half_spread_bps > 1.5 * a.half_spread_bps, (
            f"{a.half_spread_bps:.1f} vs {b.half_spread_bps:.1f}: the "
            f"instrument with four times the standard error must carry the "
            f"wider ceiling, or the bound is not per instrument")

    def test_a_significantly_negative_square_is_not_a_tight_spread(self):
        """s^2 + k SE below zero is the data contradicting the estimator.
        Charging sqrt of it, or charging zero, would both be inventing."""
        impossible = SpreadEstimate(0.0008, -6.4e-6, 0.0006, 500, 499,
                                    square_standard_error=9.6e-7)
        d = decide_spread(impossible, price=50.0, fallback_bps=8.0)
        assert d.source == ASSUMED and not d.is_evidence
        assert d.half_spread_bps == 8.0
        assert "contradicting the estimator" in d.reason

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

        Below the noise floor the estimator is not centred on the truth and
        the controls measured that. What must hold is not that the number
        is right; it is that it is never called a measurement. On the
        fixture as first written every sub-floor line came back with a
        negative squared spread and fell to the constant; with the fixture's
        open moved to the previous close (see `FixtureProvider.bars`) the
        same lines come back as ceilings, 6 bps on a 4 bps line, which is
        the other honest answer. Either is allowed here; `estimated` is
        not.
        """
        below = [s for s in SYMBOLS
                 if provider.fixture_half_spread_bps(s) < self.RESOLVABLE_BPS]
        assert below, "the fixture must contain at least one sub-floor spread"
        for symbol in below:
            frame = provider.bars(symbol)
            e = edge(frame["open"].to_numpy(), frame["high"].to_numpy(),
                     frame["low"].to_numpy(), frame["close"].to_numpy())
            d = decide_spread(e, price=float(frame["close"].iloc[-1]))
            assert d.source in (BOUNDED, ASSUMED), (
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
        and instruments it does not. A fixture where everything resolves
        would never exercise the verdicts that are not measurements."""
        sources = []
        for symbol in SYMBOLS:
            frame = provider.bars(symbol)
            prices = frame[["open", "high", "low", "close"]]
            e = edge(prices["open"].to_numpy(), prices["high"].to_numpy(),
                     prices["low"].to_numpy(), prices["close"].to_numpy())
            sources.append(decide_spread(
                e, price=float(frame["close"].iloc[-1]),
                tick=infer_tick_size(prices.to_numpy().ravel())).source)
        assert ESTIMATED in sources
        assert BOUNDED in sources or ASSUMED in sources

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


class TestAnObservedSpreadOutranksAnEstimatedOne:
    """A quote screen beats an inference from daily bars.

    The survey computes an estimate for every instrument, and a watched
    half-spread must outrank it in the verdict as it does in the cost model,
    where a watched spread is the only kind charged. The tier ordering is
    the whole point of having tiers, and it has to bind in the direction
    that costs something.
    """

    def _estimate(self, half_bps: float, *, certain: bool = True):
        s = 2.0 * half_bps / 10_000.0
        return SpreadEstimate(s, s * s, s / 20.0, 500, 499,
                              square_standard_error=(s * s) / (5.0 if certain
                                                               else 0.5))

    def test_the_observed_value_is_what_is_charged(self):
        d = decide_spread(self._estimate(45.0), price=50.0, observed_bps=6.0)
        assert d.source == "observed"
        assert d.half_spread_bps == pytest.approx(6.0)

    def test_it_wins_even_against_a_confidently_resolved_estimate(self):
        """Not "whichever is better supported". The tiers are an ordering of
        kinds of evidence, not a contest between error bars."""
        confident = self._estimate(45.0, certain=True)
        assert confident.resolved()
        d = decide_spread(confident, price=50.0, observed_bps=6.0)
        assert d.half_spread_bps == pytest.approx(6.0)

    def test_a_large_disagreement_is_printed_rather_than_swallowed(self):
        """Two independent readings a factor apart is a finding about one of
        them, and the reader is the one who can tell which."""
        d = decide_spread(self._estimate(45.0), price=50.0, observed_bps=6.0)
        assert "7.5x apart" in d.reason and "which is stale" in d.reason

    def test_agreement_is_also_said(self):
        d = decide_spread(self._estimate(7.0), price=50.0, observed_bps=6.0)
        assert "agrees" in d.reason

    def test_it_still_wins_when_there_is_no_estimate_at_all(self):
        nothing = SpreadEstimate(None, None, None, 20, 19, refusal="too short")
        d = decide_spread(nothing, price=50.0, observed_bps=6.0)
        assert d.source == "observed" and d.half_spread_bps == 6.0


class TestTheTickIsReadOffRecentBarsOnly:
    """The guard that a whole-series read would break, and nothing caught.

    A split divides every older price by the ratio and takes that section off
    any grid, so reading the whole series finds no tick and the instrument is
    refused for having had a split -- which is not a reason to refuse
    anything. A dividend adjustment does the same to the section before the
    last ex-date, and that one IS a reason to refuse, because an adjusted
    price never traded.

    The two look identical in a whole-series read and different in a recent
    one, which is the entire argument for reading recent bars.
    """

    def frame(self, *, bars_count: int = 800, seed: int = 3):
        o, h, l, c = simulate_bars(0.004, bars=bars_count, seed=seed,
                                   tick_size=0.01)
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": c},
                            index=pd.bdate_range("2020-01-01",
                                                 periods=bars_count))

    def test_a_split_does_not_cost_the_instrument_its_tick(self):
        frame = self.frame()
        frame.iloc[:500] /= 3.0                    # a 3:1 split, 300 bars ago
        assert tick_for(frame).usable, (
            "a split took the tick away; the inference is reading more than "
            "the recent tail")
        # ...and the whole-series read is what would have failed.
        assert not infer_tick_size(frame.to_numpy().ravel()).usable

    def test_a_dividend_adjusted_series_is_still_caught(self):
        """The guard must not become so permissive that it stops catching the
        thing it is for."""
        frame = self.frame()
        frame *= 0.98317
        assert not tick_for(frame).usable

    def test_an_adjustment_inside_the_recent_window_is_caught(self):
        """The realistic case for FR0000121972, which pays every year: the
        last ex-date is recent, so part of the tail is scaled and part is
        not."""
        frame = self.frame()
        frame.iloc[:-80] *= 0.98317
        assert not tick_for(frame).usable

    def test_a_clean_series_reads_its_grid(self):
        assert tick_for(self.frame()).size == pytest.approx(0.01)

    def test_a_series_shorter_than_the_window_still_reads(self):
        assert tick_for(self.frame(bars_count=120)).usable


# --------------------------------------------------------------------------
# The survey after the real book failed its ranking check
# --------------------------------------------------------------------------

import json
import pathlib

from portfolio.core.spread import classify_bars
from portfolio.data.store import DataMode, DataStore
from portfolio.research import load_book, record_survey, survey_spreads

# Five fixture instruments with imposed half-spreads 11.3, 12.1, 17.0, 21.5
# and 26.8 bps. The fixture makes volume inversely related to spread, so the
# tightest is the most liquid, which is the ordering a real book has. The
# bias a carried bar adds scales with the daily variance, so the tight lines
# were chosen among the fixture's more volatile ones; a low-volatility line
# would need a still larger sabotage to overtake the wide ones.
BOOK = (
    ("IE00B579F325", "Gold ETC", "SGLD.AS"),          # 11.3 bps, most liquid
    ("IE0031442068", "S&P 500 tracker", "IUSA.AS"),   # 12.1
    ("IE00BK5BQT80", "All-World tracker", "VWCE.DE"),  # 17.0
    ("IE00B4K48X80", "Europe tracker", "SMEA.MI"),     # 21.5
    ("IE00BKM4GZ66", "EM tracker", "IEMA.AS"),         # 26.8, least liquid
)


def book_at(root: pathlib.Path):
    store = DataStore(mode=DataMode.USER, root=root)
    store.directory.mkdir(parents=True, exist_ok=True)
    rows = ["isin,name,issuer,asset_class,base_currency,primary_symbol,"
            "exchange,quote_currency,provider_symbols,active,manual_overrides,"
            "note,broker,tradeable,tob_rate,tob_observed,half_spread_bps,"
            "spread_observed,buy_tax_rate,buyable"]
    ledger = ["id,date,isin,type,quantity,price_per_unit,currency,fees,note"]
    for i, (isin, name, symbol) in enumerate(BOOK):
        rows.append(f"{isin},{name},X,ETF,EUR,{symbol.split('.')[0]},XAMS,EUR,"
                    f"yfinance={symbol},true,,,MeDirect,true,0.0012,false,"
                    f"8.0,false,0.0,true")
        ledger.append(f"t{i},2025-02-0{i + 1},{isin},BUY,10,20.0,EUR,0,")
    store.instruments_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    store.transactions_path.write_text("\n".join(ledger) + "\n",
                                       encoding="utf-8")
    return load_book(mode="user", data_root=root, provider="fixture",
                     lookback=400)


class CarriesBarsForward(FixtureProvider):
    """The fixture, with whole bars copied forward on chosen symbols.

    The kind of bar measured to inflate the estimate most, imposed on the
    most liquid lines most heavily -- which is the shape of the real book's
    failure, where the fault anti-correlated with liquidity.
    """

    def __init__(self, carry: dict[str, float]) -> None:
        super().__init__()
        self.carry = carry

    def bars(self, symbol, start=None, *, period="2y"):
        # Contaminate the whole series and slice afterwards, so that a
        # fetch of the tail returns the same rows the full fetch did. The
        # survey tops up a history whose last bar is older than a few days,
        # and the fixture's last bar always is.
        frame = super().bars(symbol, None, period=period).copy()
        fraction = self.carry.get(symbol, 0.0)
        if fraction:
            rng = np.random.default_rng(len(symbol) * 7919)
            n = len(frame)
            picked = np.sort(rng.choice(np.arange(1, n), int(fraction * n),
                                        replace=False))
            cols = ["open", "high", "low", "close"]
            values = frame[cols].to_numpy().copy()
            for i in picked:
                values[i] = values[i - 1]
            frame[cols] = values
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        return frame


# Whole bars carried forward on the three most liquid lines. Thirty-five per
# cent is a sabotage level: the bias a carried bar adds to s^2 is about half
# the daily variance per carried bar, so on a five-instrument book whose true
# spreads span 11 to 27 bps it takes this much to push the tight lines past
# the wide ones. It says nothing about what a real feed does, which is the
# survey's job to count. Measured, the specification test refuses all three
# at this level (p < 0.001 each; at 30/30/25 it refused two of the three and
# let VWCE.DE through at 36 bps against 17), so the ranking check never sees
# them: what a carried bar does to the four moment conditions is load the
# two that use the overnight return, and J reads that as four numbers that
# are not one spread.
CARRY = {"SGLD.AS": 0.35, "IUSA.AS": 0.35, "VWCE.DE": 0.35}


class DisplacesBothEnds(FixtureProvider):
    """The fixture, with the open AND the close pushed off the mid on
    independent days, kept inside the day's range and on the cent grid.

    The one contamination the specification test cannot see, measured in
    `eval.spread_controls.specification_control`: both ends displaced move
    all four moment conditions alike, exactly as a genuine spread does. Two
    per cent on half the days lifts the liquid lines from 10-18 bps to over
    50 while J passes on every one of them (p from 0.03 to 0.5), which
    inverts the ranking and is what the ranking check is for.
    """

    def __init__(self, push: dict[str, float], size: float = 0.02) -> None:
        super().__init__()
        self.push = push
        self.size = size

    def bars(self, symbol, start=None, *, period="2y"):
        frame = super().bars(symbol, None, period=period).copy()
        fraction = self.push.get(symbol, 0.0)
        if fraction:
            rng = np.random.default_rng(len(symbol) * 7919)
            n = len(frame)
            o, h, l, c = (frame[k].to_numpy().copy()
                          for k in ("open", "high", "low", "close"))
            count = int(fraction * n)
            i = rng.choice(np.arange(1, n), count, replace=False)
            j = rng.choice(np.arange(1, n), count, replace=False)
            up = 1.0 + self.size
            down = 1.0 - self.size
            o[i] = np.clip(np.round(o[i] * np.where(rng.random(count) < 0.5,
                                                    up, down) / 0.01) * 0.01,
                           l[i], h[i])
            c[j] = np.clip(np.round(c[j] * np.where(rng.random(count) < 0.5,
                                                    up, down) / 0.01) * 0.01,
                           l[j], h[j])
            frame["open"], frame["close"] = o, c
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        return frame


PUSH = {"SGLD.AS": 0.5, "IUSA.AS": 0.5, "VWCE.DE": 0.4}


@pytest.fixture(scope="module")
def contaminated(tmp_path_factory):
    root = tmp_path_factory.mktemp("contaminated")
    book = book_at(root)
    return survey_spreads(book, mode="user", data_root=root,
                          provider=CarriesBarsForward(CARRY))


@pytest.fixture(scope="module")
def displaced(tmp_path_factory):
    root = tmp_path_factory.mktemp("displaced")
    book = book_at(root)
    return survey_spreads(book, mode="user", data_root=root,
                          provider=DisplacesBothEnds(PUSH))


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    root = tmp_path_factory.mktemp("clean")
    book = book_at(root)
    return survey_spreads(book, mode="user", data_root=root,
                          provider="fixture")


class TestTheSurveyRunsTwiceAndSaysWhy:
    """T3 and T4 end to end: the two-column survey on a contaminated feed."""

    CARRY = CARRY

    def test_on_the_clean_feed_both_columns_agree_and_rank_right(self, clean):
        assert clean.ranking.passed and clean.ranking.rho < -0.5
        assert clean.clean_ranking.passed
        for isin, d in clean.decisions.items():
            c = clean.clean[isin]
            assert c.half_spread_bps == pytest.approx(d.half_spread_bps, rel=0.15), (
                f"{isin}: {d.half_spread_bps:.1f} on every bar, "
                f"{c.half_spread_bps:.1f} with a handful set aside")

    def test_carried_bars_are_refused_before_the_ranking_check_sees_them(
            self, contaminated):
        """The real book's failure, reproduced, and caught one gate earlier
        than it was the first time: the specification test refuses every
        carried line as four numbers that are not one spread, so the every-
        bar column has too few instruments left to rank. The check cannot
        fail on what it never sees, and says so rather than passing."""
        for symbol in self.CARRY:
            isin = next(i for i, s in contaminated.symbols.items() if s == symbol)
            d = contaminated.decisions[isin]
            assert d.source == ASSUMED, (symbol, d.source, d.reason)
            assert "moment conditions disagree" in d.reason, d.reason
            assert d.estimate.specification.rejects(0.01)
        check = contaminated.ranking
        assert check.rho is None and check.instruments == 2
        assert "too few to rank" in check.verdict

    def test_and_the_column_with_them_set_aside_ranks_right_again(self, contaminated):
        """The attribution. Same feed, same estimator, the counted bars
        excluded, and liquidity predicts the ordering once more."""
        check = contaminated.clean_ranking
        assert check.passed, check.line(contaminated.names)
        assert check.rho < -0.2

    def test_the_counts_name_the_contamination_per_instrument(self, contaminated):
        by_symbol = {contaminated.symbols[i]: q
                     for i, q in contaminated.quality.items()}
        for symbol, fraction in self.CARRY.items():
            expected = int(fraction * 505)
            got = by_symbol[symbol].counts()["repeated"]
            assert got == expected, (symbol, got, expected)
        for symbol in ("SMEA.MI", "IEMA.AS"):
            assert by_symbol[symbol].counts()["repeated"] == 0

    def test_the_report_shows_both_columns_and_the_pairs(self, contaminated):
        text = "\n".join(contaminated.lines())
        assert "on every bar" in text and "with those bars set aside" in text
        assert "whole bars carried forward" in text
        assert "liquidity proxy: median daily traded value" in text
        assert "liquidity rank  spread rank" in text
        assert "(SGLD.AS)" in text                 # the symbol, per instrument
        assert "With the" in text and "bars set aside above excluded" in text
        assert "REJECTED: not four readings of one spread" in text
        assert "None of these is charged" in text

    def test_what_the_specification_test_cannot_see_the_ranking_check_catches(
            self, displaced):
        """Both ends of the day displaced from the mid moves all four moment
        conditions alike, so J passes on every line and the estimates on the
        liquid lines triple. The ranking check is the gate that catches it:
        the most liquid instruments come out widest and it refuses, without
        guessing which of its two inputs is wrong."""
        for symbol in PUSH:
            isin = next(i for i, s in displaced.symbols.items() if s == symbol)
            d = displaced.decisions[isin]
            assert d.source == ESTIMATED, (symbol, d.source, d.reason)
            assert not d.estimate.specification.rejects(0.01)
            assert d.half_spread_bps > 40.0, (symbol, d.half_spread_bps)
        check = displaced.ranking
        assert not check.passed, check.line(displaced.names)
        assert check.rho > 0.5
        assert "suspect the symbol mapping" not in check.verdict
        assert "cannot say which is wrong" in check.verdict
        assert "SGLD.AS" not in check.verdict          # keys are ISINs here...
        assert check.pairs[0][0] == "IE00B579F325"     # ...most liquid first

    def test_a_ceiling_above_the_constant_is_not_called_safe(
            self, clean, contaminated, displaced):
        for survey in (clean, contaminated, displaced):
            text = "\n".join(survey.lines())
            assert "intended direction" not in text
            assert "wrong error" not in text


class TestTheRunIsRecorded:
    def test_every_run_is_appended_with_its_numbers(self, tmp_path):
        root = tmp_path / "book"
        book = book_at(root)
        survey = survey_spreads(book, mode="user", data_root=root,
                                provider=CarriesBarsForward({"SGLD.AS": 0.2}))
        log = root / "spread-runs.jsonl"
        assert record_survey(survey, log) == 1
        assert record_survey(survey, log) == 2
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(entries) == 2
        entry = entries[0]
        assert entry["provider"] == "fixture" and "recorded_at" in entry
        assert entry["ranking"]["rho"] == pytest.approx(survey.ranking.rho)
        assert entry["clean_ranking"]["rho"] == pytest.approx(
            survey.clean_ranking.rho)
        gold = entry["instruments"]["IE00B579F325"]
        assert gold["symbol"] == "SGLD.AS"
        assert gold["bars_set_aside"]["repeated"] == int(0.2 * 505)
        # Both verdicts are in the record, with what each would have implied
        # and why: on every bar the carried line is refused by the
        # specification test; with the carried bars set aside it resolves.
        assert gold["every_bar"]["source"] == "assumed"
        assert gold["specification"]["p_value"] < 0.01
        assert gold["clean"]["source"] in ("estimated", "bounded")
        assert gold["clean"]["verdict_bps"] > gold["every_bar"]["verdict_bps"]
        assert gold["every_bar"]["estimate"]["usable_bars"] > 0
        assert "range_ratio" in gold and gold["range_ratio"]["ratio"] > 0

    def test_the_command_records_a_failed_run_before_refusing(self, tmp_path, capsys,
                                                             monkeypatch):
        """The point of the log: the run that refuses to write is the one
        whose numbers must not be lost."""
        from portfolio import cli
        root = tmp_path / "book"
        book_at(root)
        monkeypatch.setattr("portfolio.research._provider_named",
                            lambda name: DisplacesBothEnds(PUSH))
        code = cli.main(["spreads", "--mode", "user", "--data-root", str(root),
                         "--provider", "fixture"])
        captured = capsys.readouterr()
        assert code == 1
        assert "Recorded as run 1" in captured.out
        assert "did not pass" in captured.err
        assert "spread-ladder" in captured.err
        # Under the mode's directory, beside the instruments it describes.
        assert (root / "user" / "spread-runs.jsonl").exists()
        assert not (root / "spread-runs.jsonl").exists()
        # And nothing was written to the instruments.
        store = DataStore(mode=DataMode.USER, root=root)
        assert all(not i.spread_observed and i.spread_source == ""
                   and i.half_spread_bps == 8.0
                   for i in store.load_instruments().values())

    def test_a_passing_run_is_recorded_and_still_not_applied(self, tmp_path,
                                                             capsys):
        """The closed result, end to end: the survey's verdicts are printed
        and logged, the instruments keep the declared constant, and the
        command says so. `--write` is gone rather than refused."""
        from portfolio import cli
        root = tmp_path / "book"
        book_at(root)
        code = cli.main(["spreads", "--mode", "user", "--data-root", str(root),
                         "--provider", "fixture"])
        captured = capsys.readouterr()
        assert code == 0
        assert "Recorded as run 1" in captured.out
        assert "Recorded, not applied" in captured.out
        assert "declared constant of" in captured.out and "8 bps" in captured.out
        store = DataStore(mode=DataMode.USER, root=root)
        assert all(not i.spread_observed and i.spread_source == ""
                   and i.half_spread_bps == 8.0
                   for i in store.load_instruments().values())
        with pytest.raises(SystemExit):
            cli.main(["spreads", "--mode", "user", "--data-root", str(root),
                      "--provider", "fixture", "--write"])


class TestTheBarsCacheCannotLieAboutWhatItHolds:
    """Three ways the previous `_bars_for` served the wrong rows, each found
    by a reviewer probing it rather than by a test, and each now a test."""

    def _cache(self, tmp_path):
        return PriceCache(tmp_path / "bars.sqlite")

    def test_a_short_history_cached_without_a_period_is_refetched_whole(self, tmp_path):
        """The real-book case: the first wiring fetched two years and set
        every flag a later version checked, so 'max' was never fetched."""
        from portfolio.research import _bars_for
        provider = FixtureProvider()
        cache = self._cache(tmp_path)
        whole = provider.bars("IUSA.AS")
        cache.put_bars("IUSA.AS", whole.tail(120))        # no period recorded
        assert cache.bars_period("IUSA.AS") is None
        served = _bars_for("IUSA.AS", provider, cache,
                           today=whole.index[-1].date())
        assert len(served) == len(whole), (len(served), len(whole))
        assert cache.bars_period("IUSA.AS") == "max"
        cache.close()

    def test_a_whole_history_is_served_from_the_cache_without_a_fetch(self, tmp_path):
        from portfolio.research import _bars_for

        class Counting(FixtureProvider):
            calls = 0

            def bars(self, symbol, start=None, *, period="2y"):
                Counting.calls += 1
                return super().bars(symbol, start, period=period)

        provider = Counting()
        cache = self._cache(tmp_path)
        first = _bars_for("IUSA.AS", provider, cache,
                          today=provider.bars("IUSA.AS").index[-1].date())
        calls_after_first = Counting.calls
        again = _bars_for("IUSA.AS", provider, cache, today=first.index[-1].date())
        assert Counting.calls == calls_after_first, "a whole, fresh history was refetched"
        pd.testing.assert_frame_equal(first, again)
        cache.close()

    def test_a_stale_tail_is_topped_up_and_a_failed_top_up_is_not_fatal(self, tmp_path):
        from portfolio.research import STALE_BARS_DAYS, _bars_for

        class Tail(FixtureProvider):
            starts: list = []
            fail_tail = False

            def bars(self, symbol, start=None, *, period="2y"):
                if start is not None:
                    Tail.starts.append(start)
                    if Tail.fail_tail:
                        raise ProviderError(self.name, "network down")
                return super().bars(symbol, start, period=period)

        provider = Tail()
        cache = self._cache(tmp_path)
        whole = provider.bars("IUSA.AS")
        last = whole.index[-1].date()
        # Fresh enough: no top-up.
        _bars_for("IUSA.AS", provider, cache, today=last)
        assert Tail.starts == []
        # Stale: a top-up from just before the last cached bar.
        import datetime as dt
        _bars_for("IUSA.AS", provider, cache,
                  today=last + dt.timedelta(days=STALE_BARS_DAYS + 1))
        assert len(Tail.starts) == 1 and Tail.starts[0] < last
        assert cache.bars_period("IUSA.AS") == "max"     # not demoted by the tail
        # A top-up that fails still serves the cached rows.
        Tail.fail_tail = True
        served = _bars_for("IUSA.AS", provider, cache,
                           today=last + dt.timedelta(days=STALE_BARS_DAYS + 1))
        assert len(served) == len(whole)
        cache.close()

    def test_the_first_run_and_every_later_run_read_the_same_rows(self, tmp_path):
        """The cache drops a bar with a missing price, so the provider's
        frame and the cache's copy pair different days. Every run must
        read the cache's copy."""
        from portfolio.research import _bars_for

        class Gappy(FixtureProvider):
            def bars(self, symbol, start=None, *, period="2y"):
                frame = super().bars(symbol, start, period=period).copy()
                frame.iloc[10:35, frame.columns.get_loc("high")] = np.nan
                return frame

        provider = Gappy()
        cache = self._cache(tmp_path)
        today = provider.bars("IUSA.AS").index[-1].date()
        first = _bars_for("IUSA.AS", provider, cache, today=today)
        second = _bars_for("IUSA.AS", provider, cache, today=today)
        assert len(first) == len(second) == len(provider.bars("IUSA.AS")) - 25
        pd.testing.assert_frame_equal(first, second)
        cache.close()

    def test_the_survey_says_which_bars_it_read(self, clean):
        text = "\n".join(clean.lines())
        assert " bars, 20" in text and " to 2026-" in text
        record = clean.record()
        first = next(iter(record["instruments"].values()))
        assert first["bars"]["rows"] > 400 and first["bars"]["last"].startswith("2026")


class TestTheRecordIsJson:
    def test_a_non_finite_number_is_written_as_null_not_nan(self, tmp_path):
        from portfolio.research import _json_ready
        import math
        ready = _json_ready({"a": float("nan"), "b": np.float64("inf"),
                             "c": np.int64(3), "d": (np.bool_(True), 1.5),
                             "e": {"f": math.nan}})
        assert ready == {"a": None, "b": None, "c": 3, "d": [True, 1.5],
                         "e": {"f": None}}
        text = json.dumps(ready, allow_nan=False)
        assert "NaN" not in text and "Infinity" not in text


class TestDailyVolatilityIsAnRmsThatSurvivesOneBadBar:
    def test_it_is_not_disabled_by_a_coarse_grid(self):
        """More than half the closes unchanged used to make it None, which
        silently switched the ceiling check off for exactly the thin
        instruments it exists for."""
        from portfolio.research import daily_volatility
        closes = np.r_[np.full(40, 7.5), 8.0, np.full(30, 8.0), 7.9, np.full(20, 7.9)]
        assert daily_volatility(closes) is not None

    def test_one_impossible_bar_does_not_set_it(self):
        from portfolio.research import daily_volatility
        rng = np.random.default_rng(3)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
        clean = daily_volatility(closes)
        spiked = closes.copy()
        spiked[400] = closes[400] / 100.0
        assert daily_volatility(spiked) < 1.3 * clean

    def test_it_tracks_the_second_moment_under_clustering(self):
        """Half the days at 0.5%, half at 2%: the RMS is what the per-bar
        series' variance scales with, and a median-based scale sits well
        under it. The reference ceiling built from the median was too low,
        and every ratio printed against it too high."""
        from portfolio.research import daily_volatility
        rng = np.random.default_rng(4)
        sig = np.where(rng.random(2000) < 0.5, 0.005, 0.02)
        returns = rng.normal(0, 1, 2000) * sig
        closes = 100 * np.exp(np.cumsum(returns))
        rms = float(np.sqrt(np.mean(returns ** 2)))
        mad = 1.4826 * float(np.median(np.abs(returns - np.median(returns))))
        got = daily_volatility(closes)
        assert abs(got - rms) < 0.15 * rms, (got, rms)
        assert mad < 0.85 * rms                 # the thing being avoided
