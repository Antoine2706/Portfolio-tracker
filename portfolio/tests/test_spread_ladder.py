"""The ladder: the estimator's whole data path on instruments whose spread is known.

What is tested here is the verdict logic and that it bites, on a provider
that returns simulated bars at spreads chosen per symbol. What is NOT tested
here, because it cannot be from a machine without market data, is the thing
the ladder exists for -- whether Yahoo's European bars are what the estimator
assumes. That run is the user's, with `portfolio controls --spread-ladder`.

The scenarios below are the decision table in `research._ladder_verdict`,
one test per row, each constructed so that exactly one row applies. The
last test is the one that matters most: the ladder run against the fixture
provider, whose spreads bear no relation to the bands, must fail with the
pipeline verdict. A control that has never been seen to fail has not been
shown to check anything.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from portfolio.core.spread import SpreadEstimate
from portfolio.data.provider import MarketDataProvider, ProviderError
from portfolio.eval.spread_controls import simulate_bars
from portfolio.research import (LADDER, PAIRS, Rung, run_ladder,
                                rung_verdict)

BY_SYMBOL = {r.symbol: r for r in LADDER}
CONSISTENT = {"SPY": 1.0, "AAPL": 1.5, "SU.PA": 1.5, "MEUD.PA": 3.0,
              "IPRP.AS": 20.0, "VVSM.DE": 12.0, "SMH.L": 15.0}


class KnownSpreads(MarketDataProvider):
    """Bars simulated at a half-spread chosen per symbol, in bps."""
    name = "known"

    def __init__(self, spreads: dict[str, float], *, bars: int = 2500,
                 missing: frozenset[str] = frozenset(),
                 carry: dict[str, float] | None = None) -> None:
        self.spreads = spreads
        self.bars_count = bars
        self.missing = missing
        self.carry = carry or {}

    def probe(self, symbol, lookback_days=252): ...
    def history(self, symbol, start=None): ...
    def quote(self, symbol): ...

    def bars(self, symbol: str, start=None, *, period: str = "2y"):
        if symbol in self.missing or symbol not in self.spreads:
            raise ProviderError(self.name, f"no bars for {symbol}")
        seed = sum(ord(ch) for ch in symbol)
        o, h, l, c = simulate_bars(2.0 * self.spreads[symbol] / 10_000.0,
                                   bars=self.bars_count, seed=seed,
                                   tick_size=0.01)
        fraction = self.carry.get(symbol, 0.0)
        if fraction:
            rng = np.random.default_rng(seed + 1)
            picked = np.sort(rng.choice(np.arange(1, self.bars_count),
                                        int(fraction * self.bars_count),
                                        replace=False))
            for i in picked:
                o[i], h[i], l[i], c[i] = o[i - 1], h[i - 1], l[i - 1], c[i - 1]
        # Ending today, so the survey's staleness rule does not top it up;
        # a top-up would be the same rows again, but a control on the
        # verdict logic should not also be exercising the cache.
        index = pd.bdate_range(end=pd.Timestamp.today().normalize(),
                               periods=self.bars_count)
        frame = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                              "volume": np.full(self.bars_count, 1e5)},
                             index=index)
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        return frame


def ladder(tmp_path, spreads, **kw):
    return run_ladder(provider=KnownSpreads(spreads, **kw), data_root=tmp_path)


class TestTheVerdictOnOneRung:
    """`rung_verdict` compares an interval on s^2 against a band."""

    def test_an_unresolved_tight_instrument_passes_on_its_ceiling(self):
        """SPY off daily bars cannot be measured, only bounded. A ceiling of
        3 bps against a band of 0.1 to 2 is consistent: the floor of the
        interval is zero, which is inside."""
        e = SpreadEstimate(0.0003, 9e-8, 0.0002, 3000, 2999,
                           square_standard_error=1.2e-7)
        status, detail, lower, upper = rung_verdict(e, BY_SYMBOL["SPY"],
                                                    sigma_day=0.01)
        assert status == "consistent" and lower == 0.0
        assert "not resolved" in detail

    def test_a_wide_reading_on_a_tight_instrument_fails_on_its_ceiling(self):
        """The real-book case: 41.85 bps at t = 1.93 on a one-bp instrument.

        This test found the first version of the verdict dead for exactly
        this case. Unresolved means the interval on s^2 reaches below zero,
        so the floor is zero and "inside any band". What is wrong is the
        ceiling: 5000 bars at 1.5% a day put a ceiling of a few bps on a
        tight spread, and this one reports 42 -- an error bar tens of times
        what the sample allows. Without the daily volatility the verdict
        still cannot see that, and says so by passing; with it, it fails.
        """
        # These constants are the real run's: a point estimate of 42 bps at
        # t = 1.9 whose ceiling is 60, the top of the book's 15-to-60 range.
        e = SpreadEstimate(0.0084, 7.0e-5, 0.0004, 5000, 4999,
                           square_standard_error=3.6e-5)
        assert e.half_spread_bps == pytest.approx(42.0)
        assert e.t_statistic < 2.0                 # a ceiling, not a reading
        blind = rung_verdict(e, BY_SYMBOL["SU.PA"])
        assert blind[0] == "unchecked"             # no volatility: says so, does not pass
        status, detail, lower, upper = rung_verdict(e, BY_SYMBOL["SU.PA"],
                                                    sigma_day=0.015)
        assert status == "too wide", detail
        assert lower == 0.0 and upper == pytest.approx(59.6, abs=0.1)
        assert "error bar" in detail

    def test_an_honest_ceiling_on_a_tight_instrument_still_passes(self):
        """The other half: the ceiling check must not fail a clean sample.
        Simulated at a true 1 bp with the same length and volatility, the
        ceiling reported is the one the check expects, and passes."""
        o, h, l, c = simulate_bars(2.0 * 1.0 / 10_000.0, bars=3000, sigma=0.015,
                                   seed=11)
        from portfolio.core.spread import edge
        e = edge(o, h, l, c)
        assert not e.resolved()
        status, detail, _, _ = rung_verdict(e, BY_SYMBOL["SU.PA"],
                                            sigma_day=0.015)
        assert status == "consistent", detail

    def test_a_thin_instrument_read_tight_fails_the_other_way(self):
        e = SpreadEstimate(0.0002, 4e-8, 0.0001, 3000, 2999,
                           square_standard_error=1.0e-7)
        assert rung_verdict(e, BY_SYMBOL["IPRP.AS"])[0] == "too narrow"

    def test_a_refusal_is_not_a_verdict(self):
        e = SpreadEstimate(None, None, None, 10, 9, refusal="only 9 usable bars")
        assert rung_verdict(e, BY_SYMBOL["SPY"])[0] == "no estimate"
        assert rung_verdict(None, BY_SYMBOL["SPY"])[0] == "no estimate"


class TestTheLadderDecisionTable:
    """One test per row of `_ladder_verdict`, each reaching exactly that row."""

    def test_everything_inside_its_band_is_consistent(self, tmp_path):
        report = ladder(tmp_path, CONSISTENT)
        assert report.passed, "\n".join(report.lines())
        assert report.verdict.startswith("CONSISTENT.")
        assert all(r.status == "consistent" for r in report.rungs)
        assert all(p.consistent for p in report.pairs)

    def test_us_tight_and_europe_wide_blames_the_european_bars(self, tmp_path):
        """The outcome the real book's numbers predict, if the estimator is
        fine and the provider's European bars are not."""
        spreads = dict(CONSISTENT, **{"SU.PA": 30.0, "MEUD.PA": 30.0})
        report = ladder(tmp_path, spreads)
        assert not report.passed
        assert report.verdict.startswith("THIS PROVIDER'S EUROPEAN BARS")
        assert {r.rung.symbol for r in report.rungs if r.status == "too wide"} \
            == {"SU.PA", "MEUD.PA"}

    def test_us_wide_blames_the_pipeline_whatever_europe_says(self, tmp_path):
        """If SPY reads at 40 bps nothing about Europe has been learned."""
        spreads = dict(CONSISTENT, **{"SPY": 40.0, "SU.PA": 30.0})
        report = ladder(tmp_path, spreads)
        assert not report.passed
        assert report.verdict.startswith("THE PIPELINE.")
        assert "SPY" in report.verdict

    def test_a_thin_line_read_tight_is_under_reading(self, tmp_path):
        spreads = dict(CONSISTENT, **{"IPRP.AS": 1.0})
        report = ladder(tmp_path, spreads)
        assert not report.passed
        assert report.verdict.startswith("THE PIPELINE, UNDER-READING")

    def test_two_listings_far_apart_is_the_data_for_one_of_them(self, tmp_path):
        spreads = dict(CONSISTENT, **{"VVSM.DE": 8.0, "SMH.L": 40.0})
        report = ladder(tmp_path, spreads)
        assert not report.passed
        assert report.verdict.startswith("THE DATA FOR ONE LISTING")
        pair = report.pairs[0]
        assert pair.consistent is False and pair.ratio > 3.0

    def test_two_listings_moderately_apart_is_allowed(self, tmp_path):
        """Two venues do differ. A factor under three, however significant,
        is not called a fault; the threshold is a choice and is printed."""
        spreads = dict(CONSISTENT, **{"VVSM.DE": 12.0, "SMH.L": 24.0})
        report = ladder(tmp_path, spreads)
        assert report.passed
        assert "factor of 3" in report.pairs[0].detail

    def test_a_missing_us_rung_leaves_the_verdict_qualified(self, tmp_path):
        report = ladder(tmp_path, CONSISTENT, missing=frozenset({"SPY", "AAPL"}))
        assert not report.passed
        assert report.verdict.startswith("CONSISTENT WHERE IT COULD BE CHECKED")
        assert {r.rung.symbol for r in report.rungs
                if r.status == "unavailable"} == {"SPY", "AAPL"}

    def test_nothing_reachable_says_nothing_was_checked(self, tmp_path):
        every = frozenset(CONSISTENT)
        report = ladder(tmp_path, CONSISTENT, missing=every)
        assert not report.passed
        assert report.verdict.startswith("NOTHING TO SAY")


class TestTheLadderCarriesTheBarCounts:
    """T3 on the ladder: the counts are how a European-bars verdict is
    attributed to a kind of bar rather than to a suspicion."""

    def test_carried_bars_on_a_european_rung_are_counted_and_undone(self, tmp_path):
        """Contaminate SU.PA with 15% carried bars. It reads wide on every
        bar, is counted, and reads inside its band with those set aside --
        which is the shape of the finding the survey is looking for."""
        report = ladder(tmp_path, CONSISTENT, carry={"SU.PA": 0.15})
        su = next(r for r in report.rungs if r.rung.symbol == "SU.PA")
        assert su.status == "too wide", su.detail
        assert su.quality.counts()["repeated"] == int(0.15 * 2500)
        assert su.clean_status == "consistent", su.clean_detail
        assert report.verdict.startswith("THIS PROVIDER'S EUROPEAN BARS")
        text = "\n".join(report.lines())
        assert "whole bars carried forward" in text
        assert "with those set aside" in text


class TestTheCommandLine:
    def test_a_rung_parses_and_a_bad_one_is_refused(self):
        from portfolio.cli import _parse_pair, _parse_rung
        rung = _parse_rung("VWCE.DE=1-6:EU")
        assert (rung.symbol, rung.low_bps, rung.high_bps, rung.region) \
            == ("VWCE.DE", 1.0, 6.0, "EU")
        assert _parse_rung("QQQ=0.2-3").region == "US"
        assert _parse_rung("X.MI=10-50:thin").region == "thin"
        assert _parse_rung("EURUSD=X=1-5").symbol == "EURUSD=X"   # Yahoo's own '='
        for bad in ("VWCE.DE", "VWCE.DE=6-1", "VWCE.DE=a-b", "X=1-2:MARS",
                    "X=1-inf", "X=nan-1"):
            with pytest.raises(argparse.ArgumentTypeError):
                _parse_rung(bad)
        assert _parse_pair("A.DE, B.L") == ("A.DE", "B.L")
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_pair("A.DE")

    def test_the_fixture_fails_the_ladder_and_says_so_first(self, tmp_path, capsys):
        """The check biting. The fixture imposes spreads by hash of the
        symbol, so SPY comes back at tens of bps and the verdict must be
        that the pipeline is wrong -- which, for these bars, it is not; the
        bars are. A control that passed on invented data would be worthless,
        and this is the run that shows it does not."""
        from portfolio.cli import main
        code = main(["controls", "--spread-ladder", "--provider", "fixture",
                     "--data-root", str(tmp_path)])
        out = capsys.readouterr().out
        assert code == 1
        assert "expected to FAIL" in out
        assert "THE PIPELINE" in out
        assert "[WIDE] SPY" in out

    def test_extra_rungs_are_added_and_can_replace(self, tmp_path, monkeypatch):
        from portfolio import cli
        seen = {}

        def fake(**kw):
            seen.update(kw)
            from portfolio.research import LadderReport
            return LadderReport((), (), "stub", True, "known")

        monkeypatch.setattr("portfolio.research.run_ladder", fake)
        cli.main(["controls", "--spread-ladder", "--provider", "fixture",
                  "--data-root", str(tmp_path), "--rung", "VWCE.DE=1-6:EU",
                  "--pair", "A.DE,B.L"])
        assert [r.symbol for r in seen["rungs"]] == \
            [r.symbol for r in LADDER] + ["VWCE.DE"]
        assert seen["pairs"] == (("A.DE", "B.L"),)
        cli.main(["controls", "--spread-ladder", "--provider", "fixture",
                  "--data-root", str(tmp_path), "--rung", "VWCE.DE=1-6:EU",
                  "--replace-ladder"])
        assert [r.symbol for r in seen["rungs"]] == ["VWCE.DE"]
        assert seen["pairs"] == PAIRS


class TestThePairRuleIsNotDeadWhenNothingResolves:
    """Found by the review of the first version, which compared two lines on
    s^2 whatever their resolution. When the wider line is unresolved,
    |z| <= t < 2 whatever the ratio, so two listings a factor of thirteen
    apart were 'agreeing within error'. The real book resolved nothing."""

    REAL = SpreadEstimate(0.0084, 7.0e-5, 0.0004, 5000, 4999,
                          square_standard_error=3.6e-5)      # 42 bps, t = 1.9
    TIGHT = SpreadEstimate(0.00064, 4.1e-7, 0.0004, 5000, 4999,
                           square_standard_error=2.7e-7)     # 3.2 bps, t = 1.5

    def test_the_real_book_pair_is_not_called_consistent(self):
        from portfolio.research import _compare_pair
        pair = _compare_pair(("A.DE", "B.L"), (self.REAL, self.TIGHT),
                             (None, None), ("", ""), sigmas=(0.015, 0.015))
        assert pair.consistent is False, pair.detail
        assert "A.DE is the inconsistent line" in pair.detail
        assert "cannot explain" in pair.detail

    def test_without_volatility_it_says_not_comparable_rather_than_passing(self):
        from portfolio.research import _compare_pair
        pair = _compare_pair(("A.DE", "B.L"), (self.REAL, self.TIGHT),
                             (None, None), ("", ""))
        assert pair.consistent is None
        assert "not comparable" in pair.detail

    def test_an_honest_unresolved_line_does_not_contradict_a_resolved_one(self):
        """Simulated at 12 bps, one line resolves and the other is cut to
        too few bars to resolve; the pair must pass."""
        from portfolio.core.spread import edge
        from portfolio.research import _compare_pair
        o, h, l, c = simulate_bars(2.0 * 12.0 / 10_000.0, bars=3000, sigma=0.012,
                                   seed=21)
        resolved = edge(o, h, l, c)
        assert resolved.resolved()
        # The first short clean sample that does not resolve; which seed
        # that is does not matter, and a fixed one would be a coin toss.
        for seed in range(22, 60):
            o2, h2, l2, c2 = simulate_bars(2.0 * 12.0 / 10_000.0, bars=70,
                                           sigma=0.012, seed=seed)
            short = edge(o2, h2, l2, c2)
            if not short.resolved():
                break
        assert not short.resolved()
        pair = _compare_pair(("A", "B"), (resolved, short), (None, None),
                             ("", ""), sigmas=(0.012, 0.012))
        assert pair.consistent is True, pair.detail

    def test_the_ladder_reaches_the_listing_verdict_through_an_unresolved_line(self, tmp_path):
        """End to end, through the unresolved branch. Nothing simulated
        produces the real book's shape -- an unresolved line whose ceiling
        far exceeds what its sample allows -- so this reaches the branch
        the other way round: VVSM.DE simulated at 1 bp over 400 bars does
        not resolve, and its ceiling sits below the floor of SMH.L resolved
        at 12 bps. One fund cannot be both. The old rule compared them on
        s^2 and, with the short line's t under 2, called them agreeing."""
        class Mixed(KnownSpreads):
            def bars(self, symbol, start=None, *, period="2y"):
                if symbol == "VVSM.DE":
                    self.bars_count, saved = 400, self.bars_count
                    try:
                        return super().bars(symbol, start, period=period)
                    finally:
                        self.bars_count = saved
                return super().bars(symbol, start, period=period)

        provider = Mixed(dict(CONSISTENT, **{"VVSM.DE": 1.0, "SMH.L": 12.0}))
        report = run_ladder(provider=provider, data_root=tmp_path)
        pair = report.pairs[0]
        assert not pair.estimates[0].resolved() and pair.estimates[1].resolved()
        assert pair.consistent is False, pair.detail
        assert "below the other line's floor" in pair.detail
        assert report.verdict.startswith("THE DATA FOR ONE LISTING"), report.verdict


class TestAnUncheckableCeilingIsSaidNotPassed:
    def test_a_rung_without_volatility_is_unchecked(self):
        e = SpreadEstimate(0.0003, 9e-8, 0.0002, 3000, 2999,
                           square_standard_error=1.2e-7)
        status, detail, _, _ = rung_verdict(e, BY_SYMBOL["SPY"])
        assert status == "unchecked" and "could not be checked" in detail

    def test_the_stated_figure_is_printed_beside_the_band(self, tmp_path):
        report = ladder(tmp_path, CONSISTENT)
        text = "\n".join(report.lines())
        assert "band 0.3 to 4 bps (stated: 1 to 2 bps)" in text
        assert "usable bars in the window taken" in text
