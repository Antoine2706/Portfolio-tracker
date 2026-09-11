"""The spread estimator, and the properties it must have to be believed.

The controls in `eval/spread_controls.py` are the substantive validation:
they impose spreads on simulated markets and check they come back. This file
holds the faster things -- the algebraic identities, the refusals, the
behaviour at the awkward edges -- plus a conformance check against the
reference implementation for anyone who has it installed.

The division is deliberate. A control that takes four minutes cannot run on
every commit; a test that takes four seconds can. So the tests here are the
ones that would catch a transcription error, and the controls are the ones
that would catch a misunderstanding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from portfolio.core.spread import (FLOAT32_EPSILON, MINIMUM_BARS, SpreadEstimate,
                                   TickSize, classify_bars, exclude_bars,
                                   edge, infer_tick_size, sweep_windows)
from portfolio.eval.spread_controls import simulate_bars


def bars(spread_bps: float, *, n: int = 500, ticks: int = 60, seed: int = 0):
    """OHLC with a known HALF-spread in bps."""
    return simulate_bars(2.0 * spread_bps / 10_000.0, bars=n, ticks=ticks,
                         seed=seed)


class TestItRecoversASpreadItWasNotTold:
    """The one thing that matters. Kept small here; swept in the controls."""

    @pytest.mark.parametrize("imposed", [10.0, 25.0, 60.0])
    def test_within_three_standard_errors_of_the_truth(self, imposed):
        o, h, l, c = bars(imposed, n=1000, seed=int(imposed))
        e = edge(o, h, l, c)
        assert e.spread is not None and e.standard_error is not None
        off = abs(e.half_spread_bps - imposed) / e.half_spread_error_bps
        assert off < 3.0, f"{e.half_spread_bps:.2f} vs {imposed} ({off:.1f} SE)"

    def test_a_wider_spread_reads_wider(self):
        """Monotone in the imposed spread, which a constant would not be."""
        got = [edge(*bars(s, n=800, seed=11)).half_spread_bps
               for s in (10.0, 20.0, 40.0, 80.0)]
        assert got == sorted(got)

    def test_it_is_scale_free(self):
        """A spread is a fraction, so multiplying every price leaves it alone."""
        o, h, l, c = bars(30.0, seed=5)
        one = edge(o, h, l, c).half_spread_bps
        other = edge(o * 7.5, h * 7.5, l * 7.5, c * 7.5).half_spread_bps
        assert one == pytest.approx(other, rel=1e-12)

    def test_drift_does_not_leak_into_the_estimate(self):
        """The de-meaning earns its place: a trend must not read as a spread.
        Without it the products pick up E[X]E[Y] rather than Cov(X, Y).

        The drift is per bar in logs, so 0.002 is about 65% a year -- a strong
        trend for an equity fund and a fair test. It is worth saying that the
        first version of this test used 1.5 per bar, which is e^1200 over the
        window; the estimator refused it for want of finite prices, which is
        the right behaviour and no evidence about drift at all.
        """
        flat = edge(*simulate_bars(0.004, bars=800, seed=3, drift=0.0))
        rising = edge(*simulate_bars(0.004, bars=800, seed=3, drift=0.002))
        assert rising.spread is not None
        assert rising.half_spread_bps == pytest.approx(
            flat.half_spread_bps, abs=3.0 * flat.half_spread_error_bps)


class TestTheInfrequentTradingCorrection:
    """`p_o` and `p_c` are what this estimator has over its predecessors."""

    def test_it_still_works_when_bars_hold_very_few_trades(self):
        """At three trades a bar the open is frequently the extreme, which is
        the case the correction exists for."""
        e = edge(*bars(40.0, n=1500, ticks=3, seed=21))
        assert e.spread is not None
        assert abs(e.half_spread_bps - 40.0) < 3.0 * e.half_spread_error_bps

    def test_the_correction_is_what_makes_that_work(self):
        """Pin p_o and p_c at their continuous-trading value of 2 -- which is
        what an implementation without the correction computes -- and the
        estimate is biased low on thin bars. This is the sabotage run for the
        one part of the formula that is not shared with its predecessors.
        """
        o, h, l, c = bars(40.0, n=1500, ticks=3, seed=21)
        proper = edge(o, h, l, c).half_spread_bps
        naive = _edge_without_the_correction(o, h, l, c)
        assert naive < 0.9 * proper, (
            f"pinning p_o = p_c = 2 gave {naive:.1f} bps against "
            f"{proper:.1f}; if these agree the correction is not wired in")
        assert abs(proper - 40.0) < abs(naive - 40.0)


class TestItRefusesRatherThanGuessing:
    def test_too_few_bars(self):
        e = edge(*bars(20.0, n=MINIMUM_BARS - 5))
        assert e.spread is None and "usable bars" in e.refusal

    def test_a_price_that_never_moved(self):
        flat = np.full(300, 42.0)
        e = edge(flat, flat, flat, flat)
        assert e.spread is None and "never changed" in e.refusal

    def test_all_nan(self):
        nan = np.full(300, np.nan)
        assert edge(nan, nan, nan, nan).spread is None

    def test_mismatched_lengths_are_a_programming_error(self):
        with pytest.raises(ValueError, match="same length"):
            edge(np.ones(5), np.ones(5), np.ones(5), np.ones(4))

    def test_a_non_positive_price_is_excluded_not_fatal(self):
        """A zero or negative print is not a price. It must not become -inf
        and poison every mean downstream of it."""
        o, h, l, c = bars(30.0, n=600, seed=9)
        o = o.copy()
        o[100] = 0.0
        o[200] = -1.0
        e = edge(o, h, l, c)
        assert e.spread is not None and math.isfinite(e.spread)
        assert e.usable_bars < 599

    def test_gaps_are_dropped_rather_than_carried(self):
        """A missing bar must not be filled, and must not take the estimate
        with it."""
        o, h, l, c = bars(30.0, n=800, seed=13)
        clean = edge(o, h, l, c)
        o, h, l, c = (x.copy() for x in (o, h, l, c))
        for i in range(50, 800, 37):
            o[i] = h[i] = l[i] = c[i] = np.nan
        holed = edge(o, h, l, c)
        assert holed.usable_bars < clean.usable_bars
        assert holed.half_spread_bps == pytest.approx(
            clean.half_spread_bps, abs=3.0 * holed.half_spread_error_bps)


class TestSignificanceIsTestedOnTheSquare:
    """The defect the resolution control found, held in place by a test."""

    def test_the_t_statistic_is_on_the_square(self):
        e = SpreadEstimate(spread=0.002, signed_square=4e-6,
                           standard_error=0.0005, bars=500, usable_bars=499,
                           square_standard_error=1e-6)
        assert e.t_statistic == pytest.approx(4.0)
        assert e.resolved(2.0) and not e.resolved(5.0)

    def test_a_negative_square_is_never_resolved(self):
        """sqrt(|s^2|) turns a negative estimate into a positive spread. The
        significance test must see through that; if it is applied to the root
        it cannot, because the root has forgotten the sign."""
        e = SpreadEstimate(spread=0.002, signed_square=-4e-6,
                           standard_error=0.0005, bars=500, usable_bars=499,
                           square_standard_error=1e-6)
        assert e.t_statistic == pytest.approx(-4.0)
        assert not e.resolved()
        # ...whereas the test on the root would wave it through.
        assert e.spread >= 2.0 * e.standard_error

    def test_the_two_tests_are_not_the_same_test(self):
        """`s >= k SE(s)` rearranges to `|s^2| >= (k/2) SE(s^2)`. Worked
        through on the arithmetic rather than asserted, because the whole
        point is that the two forms read identically.

        Sitting exactly on the boundary of the root test: s^2 is one standard
        error from zero, so the root test at k = 2 passes by construction and
        the square test at k = 2 fails by a factor of two.
        """
        se2 = 2e-6
        s2 = 1.0 * se2                      # one standard error from zero
        s = math.sqrt(s2)
        se = se2 / (2 * s)                  # the delta method
        assert s == pytest.approx(2.0 * se)         # exactly on the boundary
        estimate = SpreadEstimate(s, s2, se, 500, 499,
                                  square_standard_error=se2)
        assert estimate.t_statistic == pytest.approx(1.0)
        assert not estimate.resolved(2.0)

    def test_a_resolvable_spread_is_resolved(self):
        """The gate must open, or it is a constant with extra steps."""
        e = edge(*bars(50.0, n=800, seed=17))
        assert e.resolved()

    def test_pure_noise_is_not(self):
        e = edge(*simulate_bars(0.0, bars=800, seed=17))
        assert e.spread is not None            # it returns a number...
        assert e.half_spread_bps > 1.0         # ...and the number looks real...
        assert not e.resolved()                # ...and it is not a measurement


class TestTheStandardError:
    def test_it_shrinks_with_the_sample(self):
        wide = edge(*bars(30.0, n=200, seed=31)).half_spread_error_bps
        narrow = edge(*bars(30.0, n=2000, seed=31)).half_spread_error_bps
        assert narrow < wide / 2.0

    def test_the_default_is_the_iid_error(self):
        """`lags` defaults to 0. The docstring argues at length for why the
        structural case for a Newey-West correction does not survive being
        measured; this pins the conclusion so it cannot drift back."""
        o, h, l, c = bars(30.0, n=1000, seed=41)
        assert (edge(o, h, l, c).half_spread_error_bps
                == edge(o, h, l, c, lags=0).half_spread_error_bps)

    def test_the_series_it_averages_is_not_autocorrelated(self):
        """The measurement that settled it, at one seed. The per-bar series
        shares a bar between neighbours and still shows no autocorrelation,
        which is why the i.i.d. error is the honest one."""
        rho = [edge(*bars(30.0, n=1500, seed=s)).autocorrelation
               for s in range(12)]
        assert all(abs(r) < 0.10 for r in rho), rho
        assert abs(float(np.mean(rho))) < 0.03, float(np.mean(rho))

    def test_raising_the_lag_adds_noise_rather_than_removing_bias(self):
        """The cost of the correction, on a single sample, which is the only
        sample a real instrument gets. If this ever falls to nothing the
        series has become autocorrelated and the default should be revisited.
        """
        moves = []
        for seed in range(20):
            o, h, l, c = bars(30.0, n=1000, seed=1000 + seed)
            base = edge(o, h, l, c, lags=0).half_spread_error_bps
            moves.append(abs(edge(o, h, l, c, lags=8).half_spread_error_bps
                             / base - 1.0))
        assert max(moves) > 0.02, (
            f"lag 8 moved the standard error by at most {max(moves):.1%}; "
            f"the docstring claims it is worth up to 15% on one sample")


def _edge_without_the_correction(o, h, l, c) -> float:
    """EDGE with `p_o` and `p_c` pinned at their continuous-trading value.

    Not a second implementation to maintain -- it exists only so the test
    above can show that the correction is doing something. It is the same
    arithmetic as `core.spread.edge` with the two probabilities frozen.
    """
    lo, lh, ll, lc = (np.log(np.asarray(x, float)) for x in (o, h, l, c))
    lm = (lh + ll) / 2.0
    h1, l1, c1, m1 = lh[:-1], ll[:-1], lc[:-1], lm[:-1]
    lo, lh, ll, lc, lm = lo[1:], lh[1:], ll[1:], lc[1:], lm[1:]
    r1, r2, r3, r4, r5 = lm - lo, lo - m1, lm - c1, c1 - m1, lo - c1
    tau = ((lh != ll) | (ll != c1)).astype(float)
    p_tau = tau.mean()
    d1 = r1 - r1.mean() / p_tau * tau
    d3 = r3 - r3.mean() / p_tau * tau
    d5 = r5 - r5.mean() / p_tau * tau
    x1 = -2.0 * d1 * r2 + -2.0 * d3 * r4     # 4/p with p pinned at 2
    x2 = -2.0 * d1 * r5 + -2.0 * d5 * r4
    v1, v2 = x1.var(), x2.var()
    w = v2 / (v1 + v2) if v1 + v2 > 0 else 0.5
    s2 = w * x1.mean() + (1 - w) * x2.mean()
    return float(math.sqrt(abs(s2)) * 10_000.0 / 2.0)


class TestItAgreesWithTheReferenceImplementation:
    """Optional: skipped unless the authors' package happens to be installed.

    Not a dependency -- the whole point of writing the estimator out is that
    it can be read here. But when the package is present, agreeing with it to
    floating-point precision is the strongest available evidence that the
    transcription is faithful, so the check runs when it can.

    Measured on 300 random panels spanning 3 to 60 trades a bar, with and
    without gaps: worst disagreement 3.5e-18.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_signed_estimates_agree(self, seed):
        reference = pytest.importorskip(
            "bidask", reason="the authors' package is not installed")
        rng = np.random.default_rng(seed)
        n, ticks = int(rng.integers(200, 600)), int(rng.integers(3, 60))
        spread = float(rng.uniform(0.0, 0.02))
        o, h, l, c = simulate_bars(spread, bars=n, ticks=ticks, seed=seed)
        if seed % 2:
            o, h, l, c = (x.copy() for x in (o, h, l, c))
            gaps = rng.choice(n, size=n // 20, replace=False)
            for x in (o, h, l, c):
                x[gaps] = np.nan
        theirs = reference.edge(o, h, l, c, sign=True)
        mine = edge(o, h, l, c, minimum_bars=0).signed_square
        ours = math.copysign(math.sqrt(abs(mine)), mine)
        assert ours == pytest.approx(theirs, abs=1e-12)


class TestTheTickIsMeasuredNotLookedUp:
    """The floor under a credible spread estimate.

    A venue rulebook would be a table of numbers nobody here can check, kept
    correct by hand, and wrong in a way that looks right -- which is the
    failure this cost model was rebuilt to escape. The grid is in the prices,
    so it is read off them.
    """

    @pytest.mark.parametrize("tick", [0.001, 0.005, 0.01, 0.05])
    def test_it_reads_the_grid_off_simulated_prices(self, tick):
        o, h, l, c = simulate_bars(0.004, bars=400, seed=3, tick_size=tick)
        got = infer_tick_size(np.concatenate([o, h, l, c]))
        assert got.usable and got.size == pytest.approx(tick)

    def test_it_takes_the_coarsest_fitting_grid(self):
        """Every price on a 0.01 grid also sits on the 0.001 grid, so the
        finest candidate always fits and would say nothing."""
        prices = np.round(np.linspace(20.0, 25.0, 400) / 0.01) * 0.01
        assert infer_tick_size(prices).size == pytest.approx(0.01)

    def test_adjusted_prices_are_declined_rather_than_guessed(self):
        """Adjusting for a distribution multiplies by a non-round factor and
        takes every price off the grid. The right answer is 'I cannot tell',
        not the finest candidate on the list."""
        o, h, l, c = simulate_bars(0.004, bars=400, seed=3, tick_size=0.01)
        adjusted = np.concatenate([o, h, l, c]) * 0.983172
        got = infer_tick_size(adjusted)
        assert not got.usable and got.size is None

    def test_too_few_prices_is_a_refusal(self):
        assert infer_tick_size(np.array([1.0, 2.0, 3.0])).size is None

    def test_the_floor_it_implies(self):
        """One tick on a 50 EUR price is 2 bps, so half a tick is 1 bp."""
        tick = TickSize(size=0.01, agreement=1.0, prices=1000)
        assert tick.floor_bps(50.0) == pytest.approx(1.0)
        assert tick.floor_bps(5.0) == pytest.approx(10.0)
        assert TickSize(None, 0.4, 1000).floor_bps(50.0) is None

    def test_the_floor_bites_on_a_price_where_the_tick_is_coarse(self):
        """A 4 EUR instrument on a 1 cent grid cannot have a half-spread
        under 12.5 bps, which is above the estimator's own noise floor. On
        such an instrument the tick, not the sample, is the binding limit."""
        tick = TickSize(size=0.01, agreement=1.0, prices=1000)
        assert tick.floor_bps(4.0) == pytest.approx(12.5)


class TestTheFiniteTradeBias:
    """The estimator runs slightly high on wide spreads at realistic trade
    counts, and the effect goes away as trading becomes continuous.

    Measured rather than assumed, and pinned here because it is the one
    systematic error the controls found that is *not* corrected. If a change
    ever makes it larger, or makes it stop shrinking with the trade count,
    that is a different effect wearing the same name and the module docstring
    would be describing something that no longer happens.
    """

    def _bias(self, half_bps: float, ticks: int, runs: int = 40) -> float:
        got = [edge(*bars(half_bps, n=500, ticks=ticks, seed=800_000 + i),
                    minimum_bars=0).half_spread_bps for i in range(runs)]
        return (float(np.mean(got)) - half_bps) / half_bps

    def test_it_shrinks_as_trading_becomes_continuous(self):
        """The claim that separates a finite-trade effect from a defect."""
        few = self._bias(100.0, ticks=10)
        many = self._bias(100.0, ticks=600)
        assert few > 0.02, f"expected a visible bias on thin bars, got {few:+.1%}"
        assert many < 0.5 * few, (
            f"bias at 600 trades/bar is {many:+.1%} against {few:+.1%} at 10; "
            f"it is meant to shrink towards zero, and if it does not this is "
            f"not the effect the module docstring describes")

    def test_it_is_small_next_to_the_correction_it_is_not(self):
        """The infrequent-trading bias `p_o` and `p_c` remove is 21% on
        three-trade bars. This one is a few percent. Confusing the two would
        make the correction look unnecessary."""
        assert abs(self._bias(100.0, ticks=10)) < 0.10

    def test_it_errs_in_the_harmless_direction(self):
        """A cost model that overstates cost declines trades it could have
        afforded. One that understates it takes trades it could not."""
        assert self._bias(100.0, ticks=30) > 0


class TestChoosingTheWindow:
    """The window is a choice, and it is not the covariance window's choice.

    A covariance window must be short because correlations move with regime.
    A spread moves slowly and has nothing to do with when the holding was
    bought, so the whole available history is the right starting point --
    which matters because the noise floor thins only as the fourth root of
    the sample, and on one year of bars almost nothing in a European ETF book
    resolves at all.
    """

    def test_a_stable_spread_takes_the_whole_history(self):
        sweep = sweep_windows(*bars(30.0, n=1200, seed=5))
        assert sweep.drifted_at is None
        assert sweep.chosen_bars == 1200
        assert "nothing is gained by throwing history away" in sweep.reason

    def test_the_longer_window_is_the_tighter_estimate(self):
        """The entire reason for the change: more bars, smaller error bar,
        and therefore more instruments that resolve at all."""
        sweep = sweep_windows(*bars(30.0, n=1200, seed=5))
        short = next(r for r in sweep.rungs if r.bars == 250)
        assert (sweep.chosen.half_spread_error_bps
                < 0.6 * short.estimate.half_spread_error_bps)

    def test_a_spread_that_halved_is_not_averaged_across(self):
        """A ten-year window on a fund whose spread has halved is an average
        of a market that no longer exists."""
        old = simulate_bars(0.012, bars=700, seed=6)
        new = simulate_bars(0.004, bars=500, seed=7)
        o, h, l, c = (np.concatenate([a, b]) for a, b in zip(old, new))
        sweep = sweep_windows(o, h, l, c)
        assert sweep.drifted_at is not None
        assert sweep.chosen_bars < 1200
        assert "a market that is gone" in sweep.reason
        # And it took the recent regime, not the old one.
        assert abs(sweep.chosen.half_spread_bps - 20.0) < 15.0

    def test_the_drift_test_uses_disjoint_blocks(self):
        """Nested windows share their data, so the difference of two nested
        estimates has a variance smaller than the sum of theirs. Comparing
        them as if independent would call drift on stable instruments."""
        sweep = sweep_windows(*bars(30.0, n=1200, seed=5))
        assert len(sweep.blocks) > 1
        # Every block after the first starts where the previous window ended.
        assert [b.ends_bars_ago for b in sweep.blocks] == [0, 250, 500, 1000]
        assert sum(b.bars for b in sweep.blocks) == 1200

    def test_a_short_series_still_produces_one_rung(self):
        sweep = sweep_windows(*bars(30.0, n=300, seed=9))
        assert sweep.chosen_bars == 300 and sweep.chosen.spread is not None

    def test_the_sequence_does_not_drift_when_the_truth_is_constant(self):
        """The free diagnostic. An estimator that trended with window length
        on a constant spread would be a bug, and this is where it would show.
        """
        sweep = sweep_windows(*bars(30.0, n=2000, seed=11))
        got = [r.estimate.half_spread_bps for r in sweep.rungs]
        assert max(got) - min(got) < 4.0, got


class TestAnAdjustedSeriesIsDetected:
    """The tick grid is the check on whether the right series arrived.

    yfinance returns adjusted closes by default, and adjustment multiplies
    every historical price by a factor that is not a multiple of the tick. On
    an adjusted series the grid is destroyed. The accumulating ETFs pay no
    distribution so adjustment is a no-op for them and the grid survives;
    FR0000121972 pays a dividend every year, so its adjusted history is off
    grid at every point before the most recent ex-date.
    """

    def test_an_adjusted_series_does_not_silently_produce_a_tick(self):
        o, h, l, c = simulate_bars(0.004, bars=500, seed=3, tick_size=0.01)
        raw = np.concatenate([o, h, l, c])
        assert infer_tick_size(raw).usable
        # One dividend adjustment factor applied to the whole series.
        assert not infer_tick_size(raw * 0.98317).usable

    def test_a_dividend_adjusted_tail_is_caught_too(self):
        """The realistic shape: everything before the last ex-date is scaled
        and everything after it is not."""
        o, h, l, c = simulate_bars(0.004, bars=500, seed=3, tick_size=0.01)
        raw = np.concatenate([o, h, l, c]).reshape(4, -1)
        raw[:, :400] *= 0.98317
        assert not infer_tick_size(raw.ravel()).usable

    def test_the_recent_tail_of_a_split_series_still_reads(self):
        """A split divides older prices by the ratio and takes them off grid,
        which is why the inference is given recent bars rather than all of
        them. The recent tail is post-split and reads cleanly."""
        o, h, l, c = simulate_bars(0.004, bars=800, seed=3, tick_size=0.01)
        frame = np.stack([o, h, l, c])
        frame[:, :500] /= 3.0                       # a 3:1 split, 300 bars ago
        assert not infer_tick_size(frame.ravel()).usable
        assert infer_tick_size(frame[:, -250:].ravel()).usable


class TestTheTickSurvivesTheWireFormat:
    """Yahoo returns float32. The first version of the inference did not.

    On a real seven-holding book it rejected every instrument with fit rates
    of 0, 1, 3, 3, 6, 8 and 25 per cent, and told the reader the series had
    been adjusted for distributions. Four of the seven are accumulating ETFs
    that have never made one, and the single dividend-paying holding had the
    *highest* fit rate of the seven -- the explanation was not merely
    unproven, its ordering was backwards.

    The cause was the wire format. `100.06` stored as float32 and upcast to
    float64 is `100.05999755859375`, and an exact-divisibility test against a
    tick fails on essentially every price.
    """

    @pytest.mark.parametrize("tick", [0.001, 0.005, 0.01, 0.05])
    def test_a_float32_round_trip_still_finds_the_grid(self, tick):
        """The test that reproduces it, on a series built on a known grid."""
        o, h, l, c = simulate_bars(0.004, bars=500, seed=3, tick_size=tick)
        clean = np.concatenate([o, h, l, c])
        assert infer_tick_size(clean).size == pytest.approx(tick)

        via32 = clean.astype(np.float32).astype(np.float64)
        assert not np.array_equal(clean, via32), (
            "the round trip changed nothing, so this proves nothing about "
            "float32")
        got = infer_tick_size(via32)
        assert got.usable and got.size == pytest.approx(tick), (
            f"{got.agreement:.0%} fit after a float32 round trip; this is the "
            f"0-8% the real book showed")

    def test_the_old_exact_test_is_what_failed(self):
        """Pinned so the mechanism cannot be argued about later: an exact
        divisibility test rejects nearly every float32-carried price."""
        o, h, l, c = simulate_bars(0.004, bars=500, seed=3, tick_size=0.01)
        via32 = np.concatenate([o, h, l, c]).astype(np.float32).astype(float)
        exact = np.mean(np.abs(via32 / 0.01 - np.round(via32 / 0.01)) < 1e-6)
        assert exact < 0.2, f"exact test fit {exact:.0%}, expected near zero"

    def test_a_grid_finer_than_the_data_is_not_claimed(self):
        """float32 leaves about 1.2e-5 of uncertainty on a EUR 100 price, so a
        0.0001 grid cannot be established from it. Claiming one anyway is how
        an earlier attempt at this fix made a dividend-adjusted series 'fit'
        at 93%: it manufactured the grid it reported."""
        o, h, l, c = simulate_bars(0.004, bars=500, seed=3, tick_size=0.01)
        adjusted = np.concatenate([o, h, l, c]) * 0.98317
        got = infer_tick_size(adjusted)
        assert not got.usable
        assert got.size is None


class TestTheRefusalDoesNotAssertACauseItDidNotTest:
    """The defect this shares with "an ETC is a debt security" about a
    property fund: a confident, specific, untested explanation attached to a
    real refusal. The refusal may be right; the reason sends the reader
    somewhere there is nothing to find."""

    def scattered(self):
        o, h, l, c = simulate_bars(0.004, bars=500, seed=3, tick_size=0.01)
        return np.concatenate([o, h, l, c]) * 0.98317

    def part_scattered(self):
        o, h, l, c = simulate_bars(0.004, bars=500, seed=3, tick_size=0.01)
        whole = np.concatenate([o, h, l, c])
        return np.concatenate([whole[:1000] / 3.0, whole[1000:]])

    def test_a_clean_series_says_nothing(self):
        o, h, l, c = simulate_bars(0.004, bars=500, seed=3, tick_size=0.01)
        assert infer_tick_size(np.concatenate([o, h, l, c])).why_not_a_grid() == ""

    def test_a_rescaled_series_is_described_by_its_residual(self):
        got = infer_tick_size(self.scattered())
        message = got.why_not_a_grid()
        assert "of a tick" in message and "rescaling" in message
        # Named as candidates, not asserted as the cause.
        assert "not established here" in message

    def test_it_distinguishes_a_whole_series_from_part_of_one(self):
        """A split applied to the older history leaves most prices fitting
        exactly. Reporting the median over ALL prices called that "floating
        point" -- the wrong statistic for a bimodal series."""
        assert "part of the series" in infer_tick_size(
            self.part_scattered()).why_not_a_grid()
        assert "part of the series" not in infer_tick_size(
            self.scattered()).why_not_a_grid()

    def test_it_never_claims_a_distribution_it_cannot_see(self):
        """The specific false sentence. An accumulating ETF has never made a
        distribution, so a message that names one is wrong before it is
        unproven."""
        for series in (self.scattered(), self.part_scattered()):
            message = infer_tick_size(series).why_not_a_grid()
            assert "has been adjusted for distributions" not in message
            assert "dividend adjustment" in message  # offered, among others


class TestBarsThatCannotBeTrusted:
    """`classify_bars`, and the measurement behind which kinds it excludes.

    The estimator passed six controls on simulated bars and then estimated
    a CAC 40 mega-cap at 41.85 bps on a real feed. So each kind of bar a feed
    manufactures was imposed on simulated bars and measured, and the table
    is in `core/spread.py` above `classify_bars`. These tests pin the three
    findings that decide what the survey sets aside:

      * a carried close or a carried whole bar inflates the estimate by more
        than half at five per cent of bars, and setting them aside recovers
        the imposed spread;
      * a flat bar does nothing, which is why it is counted and not excluded;
      * contaminating the high and the low, symmetrically or on one side,
        does nothing either -- the hypothesis that reading (h+l)/2 makes the
        estimator "maximally sensitive to high/low contamination" is wrong
        as stated, and this is the test that says so.
    """

    IMPOSED = 10.0
    N = 1500

    def clean(self, seed: int = 0):
        return bars(self.IMPOSED, n=self.N, seed=seed)

    def picked(self, seed: int, fraction: float) -> np.ndarray:
        rng = np.random.default_rng(1000 + seed)
        return np.sort(rng.choice(np.arange(1, self.N), int(fraction * self.N),
                                  replace=False))

    def test_a_clean_series_has_nothing_to_set_aside(self):
        o, h, l, c = self.clean()
        q = classify_bars(o, h, l, c)
        assert q.excluded_count == 0
        assert q.counts()["flat"] == 0 and q.counts()["impossible"] == 0

    def test_each_kind_is_counted_exactly_once(self):
        o, h, l, c = self.clean()
        v = np.full(self.N, 500.0)
        # Ten whole bars carried, ten closes carried, five flat bars at the
        # previous close, three impossible bars, four with zero volume, two
        # with no volume figure. Disjoint rows, so each is one thing.
        for i in range(100, 110):
            o[i], h[i], l[i], c[i] = o[i - 1], h[i - 1], l[i - 1], c[i - 1]
        for i in range(200, 210):
            c[i] = c[i - 1]
            h[i], l[i] = max(h[i], c[i]), min(l[i], c[i])
        for i in range(300, 305):
            o[i] = h[i] = l[i] = c[i] = c[i - 1]
        for i in range(400, 403):
            c[i] = h[i] * 1.01
        v[500:504] = 0.0
        v[600:602] = np.nan
        q = classify_bars(o, h, l, c, v)
        n = q.counts()
        assert n["repeated"] == 10 and n["stale_close"] == 10
        assert n["flat"] == 5 and n["flat_at_previous_close"] == 5
        assert n["impossible"] == 3
        assert n["zero_volume"] == 4 and n["no_volume"] == 2
        assert n["excluded"] == 10 + 10 + 3 + 4

    def test_a_carried_whole_bar_inflates_the_estimate_and_exclusion_undoes_it(self):
        """Five per cent of bars, the measured +84% on average. One sample
        here, so the assertion is a floor on the effect, not its size."""
        o, h, l, c = self.clean(seed=3)
        for i in self.picked(3, 0.05):
            o[i], h[i], l[i], c[i] = o[i - 1], h[i - 1], l[i - 1], c[i - 1]
        contaminated = edge(o, h, l, c)
        assert contaminated.half_spread_bps > 1.4 * self.IMPOSED, (
            f"{contaminated.half_spread_bps:.1f} bps: the carried bars did "
            f"not inflate the estimate, and the whole exclusion rests on "
            f"the claim that they do")
        q = classify_bars(o, h, l, c)
        assert q.counts()["repeated"] == int(0.05 * self.N)
        recovered = edge(*exclude_bars(o, h, l, c, q.excluded))
        off = abs(recovered.half_spread_bps - self.IMPOSED) / recovered.half_spread_error_bps
        assert off < 3.0, f"{recovered.describe()} against {self.IMPOSED}"

    def test_a_carried_close_alone_does_too(self):
        o, h, l, c = self.clean(seed=4)
        for i in self.picked(4, 0.05):
            c[i] = c[i - 1]
            h[i], l[i] = max(h[i], c[i]), min(l[i], c[i])
        assert edge(o, h, l, c).half_spread_bps > 1.3 * self.IMPOSED
        q = classify_bars(o, h, l, c)
        assert q.counts()["stale_close"] == int(0.05 * self.N)
        recovered = edge(*exclude_bars(o, h, l, c, q.excluded))
        off = abs(recovered.half_spread_bps - self.IMPOSED) / recovered.half_spread_error_bps
        assert off < 3.0, recovered.describe()

    def test_a_flat_bar_is_harmless_which_is_why_it_is_kept(self):
        """Fifteen per cent of bars collapsed to one price. The estimator's
        own tau already treats these as untraded; excluding them would
        change the bar count and nothing else."""
        o, h, l, c = self.clean(seed=5)
        for i in self.picked(5, 0.15):
            o[i] = h[i] = l[i] = c[i]
        e = edge(o, h, l, c)
        off = abs(e.half_spread_bps - self.IMPOSED) / e.half_spread_error_bps
        assert off < 3.0, e.describe()
        q = classify_bars(o, h, l, c)
        assert q.counts()["flat"] == int(0.15 * self.N)
        assert q.excluded_count == 0

    def test_widening_the_range_is_harmless_symmetric_or_not(self):
        """The pushback. (h+l)/2 is unmoved by a symmetric widening, and the
        de-meaning absorbs a one-sided one; neither is what broke the real
        book, whatever it looked like from the formula."""
        for side in ("both", "high"):
            o, h, l, c = self.clean(seed=6)
            for i in self.picked(6, 0.15):
                h[i] *= 1.003
                if side == "both":
                    l[i] /= 1.003
            e = edge(o, h, l, c)
            off = abs(e.half_spread_bps - self.IMPOSED) / e.half_spread_error_bps
            assert off < 3.0, f"{side}: {e.describe()}"

    def test_an_impossible_bar_is_flagged_but_a_float32_ulp_is_not(self):
        o, h, l, c = self.clean()
        c[7] = h[7] * (1.0 + 2.0 * FLOAT32_EPSILON)     # a representation error
        c[8] = h[8] * 1.001                              # a bar that cannot be
        q = classify_bars(o, h, l, c)
        assert not q.impossible[7] and q.impossible[8]

    def test_exclusion_is_by_nan_so_pairs_are_not_manufactured(self):
        """Deleting a bar would pair two days that were never consecutive."""
        o, h, l, c = self.clean()
        mask = np.zeros(self.N, dtype=bool)
        mask[10] = True
        o2, h2, l2, c2 = exclude_bars(o, h, l, c, mask)
        assert len(o2) == self.N and np.isnan(o2[10]) and o2[11] == o[11]
        # Two pairs lost, (9,10) and (10,11), not one.
        assert edge(o2, h2, l2, c2).usable_bars == edge(o, h, l, c).usable_bars - 2

    def test_the_carried_bars_push_the_autocorrelation_positive_not_negative(self):
        """Recorded because it was NOT expected. The real book's seven
        instruments all read negative, four beyond -0.10; a carried bar
        moves it the other way, so whatever produced those is not this."""
        rhos = []
        for seed in range(6):
            o, h, l, c = self.clean(seed=20 + seed)
            for i in self.picked(20 + seed, 0.15):
                o[i], h[i], l[i], c[i] = o[i - 1], h[i - 1], l[i - 1], c[i - 1]
            rhos.append(edge(o, h, l, c).autocorrelation)
        assert np.mean(rhos) > 0.02, rhos


class TestTheContaminationControlRegeneratesTheTable:
    """The table above `classify_bars` lived in a comment until a reviewer
    had to re-derive it to trust it. Now it is a control."""

    def test_it_passes_and_says_what_it_measured(self):
        from portfolio.eval.spread_controls import contamination_control
        report = contamination_control(runs=10, bars=800)
        assert report.passed, "\n".join(report.lines())
        text = "\n".join(report.lines())
        assert "whole bar carried forward" in text
        assert "daily reversal in the price itself" in text
        rows = {(r["kind"], r["share"]): r for r in report.numbers["rows"]}
        base = report.numbers["baseline_bps"]
        assert rows[("whole bar carried forward", "5%")]["contaminated_bps"] > 1.35 * base
        assert rows[("high and low widened by 30 bps", "15%")]["set_aside"] == 0.0
        reversal = rows[("daily reversal in the price itself", "phi -0.15")]
        assert reversal["contaminated_bps"] > 1.35 * base
        assert reversal["set_aside"] == 0.0

    def test_it_bites_when_the_mask_stops_catching_carried_bars(self, monkeypatch):
        """Sabotage: a mask that flags nothing. The control must fail on
        its recovery claim, or it is a table with a PASS stamp."""
        from portfolio.eval import spread_controls
        from portfolio.core import spread as core_spread

        real = core_spread.classify_bars

        def blind(*args, **kwargs):
            q = real(*args, **kwargs)
            return type(q)(bars=q.bars, repeated=np.zeros_like(q.repeated),
                           stale_close=np.zeros_like(q.stale_close),
                           flat=q.flat, flat_at_previous_close=q.flat_at_previous_close,
                           impossible=q.impossible, zero_volume=q.zero_volume,
                           no_volume=q.no_volume)

        monkeypatch.setattr(spread_controls, "classify_bars", blind)
        report = spread_controls.contamination_control(runs=6, bars=600)
        assert not report.passed
        assert "recovers the imposed spread: NO" in "\n".join(report.lines())


class TestTheSpecificationTest:
    """Hansen's J on the four moment conditions. The test the estimator was
    missing: on the first real ladder run the four read +14.2, +0.9, +13.0
    and -5.4 bps in one era and the point estimate came back with a tight
    error bar anyway."""

    def test_the_chi_square_survival_matches_the_tables(self):
        from portfolio.core.spread import chi2_survival, chi2_survival_3dof
        # Critical values of chi-square with three degrees of freedom.
        assert chi2_survival_3dof(7.815) == pytest.approx(0.05, abs=5e-4)
        assert chi2_survival_3dof(11.345) == pytest.approx(0.01, abs=5e-4)
        assert chi2_survival_3dof(16.266) == pytest.approx(0.001, abs=2e-4)
        assert chi2_survival_3dof(0.0) == 1.0
        assert chi2_survival_3dof(2.366) == pytest.approx(0.5, abs=1e-3)
        # Two, the test's own, and one.
        assert chi2_survival(5.991, 2) == pytest.approx(0.05, abs=5e-4)
        assert chi2_survival(9.210, 2) == pytest.approx(0.01, abs=5e-4)
        assert chi2_survival(13.816, 2) == pytest.approx(0.001, abs=2e-4)
        assert chi2_survival(3.841, 1) == pytest.approx(0.05, abs=5e-4)
        assert chi2_survival(6.635, 1) == pytest.approx(0.01, abs=5e-4)
        with pytest.raises(ValueError):
            chi2_survival(1.0, 4)

    def test_every_estimate_carries_it(self):
        o, h, l, c = bars(10.0, n=800, seed=1)
        e = edge(o, h, l, c)
        assert e.specification is not None
        assert e.specification.dof == 2 and e.specification.bars == e.usable_bars
        assert len(e.specification.moments) == 4

    def test_the_four_products_satisfy_one_exact_identity(self):
        """r2 = r4 + r5 and r3 = r1 + r5, so p_o (a12 - a15) = p_c (a34 - a54)
        bar by bar: four conditions, three independent, two degrees of
        freedom. The first version of the test read J against chi-square(3)
        on a singular covariance and got the wrong answer whenever the
        pseudo-inverse's cutoff fell on the null direction."""
        from portfolio.core.spread import _four_products, _per_bar_terms
        o, h, l, c = bars(10.0, n=800, seed=1)
        t = _per_bar_terms(o, h, l, c)
        a12, a34, a15, a54 = _four_products(t)
        lhs, rhs = t.p_o * (a12 - a15), t.p_c * (a34 - a54)
        ok = np.isfinite(lhs) & np.isfinite(rhs)
        assert np.allclose(lhs[ok], rhs[ok], rtol=1e-9, atol=1e-18)

    def test_a_residual_along_the_identity_does_not_enter_j(self):
        """The sabotage that found the identity, reduced to its mechanism.
        With a few bars set aside the de-meaning of r3 no longer equals
        that of r1 plus r5, the three series having lost different rows,
        and the identity holds only to parts per million. A test that
        inverts the full four-by-four covariance then carries a direction
        whose variance is of order epsilon squared and whose mean is of
        order epsilon, so its contribution to J does not vanish with
        epsilon: on the old fixture's IEMA.AS bars p went from 0.04 to
        under 0.01 on 29 of 30 random three-bar exclusions. Here the
        residual is one part in a billion along the identity, with a
        t-ratio of about twenty, and J must not see it. The pseudo-inverse
        version read J = 479 on this matrix; the projection reads 1.4
        with or without the residual."""
        from portfolio.core.spread import _specification_test
        rng = np.random.default_rng(3)
        k, p_o, p_c = 500, 1.3, 1.4
        a12, a34, a15 = (rng.standard_t(4, k) * 1e-6 for _ in range(3))
        a54 = a34 - (p_o / p_c) * (a12 - a15)         # the exact identity
        loud = 1.0 + rng.normal(0.0, 1.0, k)           # t-ratio ~ sqrt(500)
        clean = _specification_test(np.column_stack([a12, a34, a15, a54]),
                                    p_o, p_c)
        sabotaged = _specification_test(
            np.column_stack([a12, a34, a15, a54 + 1e-9 * loud]), p_o, p_c)
        assert clean.dof == sabotaged.dof == 2
        assert sabotaged.statistic == pytest.approx(clean.statistic, abs=0.05)
        assert not sabotaged.rejects(0.05)
        # And a residual that is NOT small is a disagreement, and is seen.
        real = _specification_test(
            np.column_stack([a12, a34, a15, a54 + 1e-6 * loud]), p_o, p_c)
        assert real.rejects(0.01)

    def test_it_is_calibrated_on_clean_bars(self):
        """A test that rejects the model it was derived from is not a test.
        Forty clean samples: at 5% about two rejections, at 1% about none."""
        rejected_5 = rejected_1 = 0
        for seed in range(40):
            o, h, l, c = bars(10.0, n=1000, seed=100 + seed)
            t = edge(o, h, l, c).specification
            rejected_5 += t.p_value < 0.05
            rejected_1 += t.p_value < 0.01
        assert rejected_5 <= 6, rejected_5          # 15% ceiling on a 5% rate
        assert rejected_1 <= 2, rejected_1

    def test_a_contaminated_open_is_rejected_while_the_estimate_looks_plausible(self):
        """The sabotage. Five bps imposed; an open off the market by 30 bps
        on a fifth of days lifts the estimate to something a real instrument
        could show, and the four moment conditions no longer agree."""
        from portfolio.agents.spreads import ASSUMED, decide_spread
        # A one-bp instrument over 5000 bars, the SPY shape. The open is
        # pushed 50 bps off the market on a fifth of days but kept inside
        # the day's range, so the mid is untouched. Pushing it outside and
        # widening the range to cover it was tried first and moved all
        # four products together, which is a different contamination.
        o, h, l, c = bars(1.0, n=5000, seed=7)
        rng = np.random.default_rng(7)
        picked = rng.choice(np.arange(1, 4999), 1000, replace=False)
        o2 = o.copy()
        o2[picked] = np.clip(o2[picked] * np.where(rng.random(1000) < 0.5, 1.005, 0.995),
                             l[picked], h[picked])
        e = edge(o2, h, l, c)
        assert 5.0 < e.half_spread_bps < 20.0, e.describe()   # plausible-looking
        assert e.resolved()                                     # and "significant"
        assert e.specification.rejects(0.01), e.specification.line()
        roots = e.specification.moments_half_bps
        assert roots[1] < 3.0 < min(roots[0], roots[2]), roots  # r3 r4 alone clean
        # The decision refuses it, with the four numbers in the reason.
        d = decide_spread(e, price=100.0)
        assert d.source == ASSUMED
        assert "moment conditions disagree" in d.reason
        assert "J = " in d.reason

    def test_a_contaminated_close_leaves_the_open_product_alone(self):
        """The mirror image, which is what tells the two apart on real bars."""
        o, h, l, c = bars(1.0, n=5000, seed=8)
        rng = np.random.default_rng(8)
        picked = rng.choice(np.arange(1, 4999), 1000, replace=False)
        c2 = c.copy()
        c2[picked] = np.clip(c2[picked] * np.where(rng.random(1000) < 0.5, 1.005, 0.995),
                             l[picked], h[picked])
        e = edge(o, h, l, c2)
        assert e.specification.rejects(0.01), e.specification.line()
        roots = e.specification.moments_half_bps
        assert roots[0] < 3.0 < min(roots[1], roots[3]), roots  # r1 r2 alone clean

    def test_a_reversal_is_caught_and_wears_the_close_signature(self):
        """Written first as 'a reversal moves all four alike and J passes',
        from the argument. Measured, a reversal lives in the overnight
        step, which r3 and r5 span and r1 does not, so it loads on the two
        products that use the previous close and leaves the open-based two
        at zero. J rejects it and cannot tell it from a carried close.

        The bound on the open side is relative, not absolute: the readings
        are signed square roots, and a moment that is one eighth of the
        close side's in bps squared (7 against 20 on seed 1) reads as a
        third of it once rooted. Measured over eight seeds the ratio of
        roots sits between 0.12 and 0.37; the bound is one half."""
        from portfolio.eval.spread_controls import simulate_reversal_bars
        rejected = 0
        for seed in range(6):
            o, h, l, c = simulate_reversal_bars(0.0002, bars=3000, phi=-0.15,
                                                seed=seed)
            e = edge(o, h, l, c)
            rejected += e.specification.rejects(0.01)
            r12, r34, r15, r54 = e.specification.moments_half_bps
            assert min(r34, r54) > 15.0, (seed, e.specification.line())
            assert max(abs(r12), abs(r15)) < 0.5 * min(r34, r54), (seed, e.specification.line())
        assert rejected == 6, rejected

    def test_both_ends_displaced_is_the_limit_nothing_on_these_moments_sees(self):
        """Push the open AND the close off the mid, on independent days, and
        all four products rise alike: the signature of a genuine spread,
        and J passes. Pinned so that nobody reads a passing J as 'a
        spread': it means only that the four agree."""
        rejected, estimates = 0, []
        for seed in range(6):
            o, h, l, c = bars(1.0, n=5000, seed=700 + seed)
            rng = np.random.default_rng(1700 + seed)
            i = rng.choice(np.arange(1, 4999), 1000, replace=False)
            j = rng.choice(np.arange(1, 4999), 1000, replace=False)
            o2, c2 = o.copy(), c.copy()
            o2[i] = np.clip(o2[i] * np.where(rng.random(1000) < 0.5, 1.005, 0.995),
                            l[i], h[i])
            c2[j] = np.clip(c2[j] * np.where(rng.random(1000) < 0.5, 1.005, 0.995),
                            l[j], h[j])
            e = edge(o2, h, l, c2)
            rejected += e.specification.rejects(0.01)
            estimates.append(e.half_spread_bps)
            roots = e.specification.moments_half_bps
            assert max(roots) - min(roots) < 5.0, roots      # all four alike
        assert rejected <= 1, rejected
        assert min(estimates) > 8.0                           # and wrong by 8x

    def test_a_clean_resolved_estimate_still_passes_the_gate(self):
        from portfolio.agents.spreads import ESTIMATED, decide_spread
        o, h, l, c = bars(20.0, n=1000, seed=3)
        d = decide_spread(edge(o, h, l, c), price=100.0)
        assert d.source == ESTIMATED, d.reason

    def test_the_control_regenerates_all_of_it(self):
        from portfolio.eval.spread_controls import specification_control
        report = specification_control(runs=12, bars=2000)
        assert report.passed, "\n".join(report.lines())
        rows = {r["kind"]: r for r in report.numbers["rows"]}
        assert rows["contaminated open"]["rejected_alpha"] >= 0.9
        assert rows["daily reversal, phi -0.15"]["rejected_alpha"] >= 0.9
        assert rows["open AND close displaced"]["rejected_alpha"] <= 0.15
        assert rows["clean"]["rejected_alpha"] <= 0.1


class TestTheRangeRatio:
    """Parkinson over close-to-close: is the range carrying prices the
    closes never see? The prediction on record for SPY is near or below
    one before 2000 and above one after."""

    def test_clean_simulated_bars_sit_a_little_under_one(self):
        """Written first as 'at one', from the argument that the simulator's
        walk is continuous across the close and so has no overnight gap.
        Measured, seeds 5 to 7 read 0.82 to 0.85. The reason is sampling,
        not a gap: the simulator draws sixty ticks per bar, and the maximum
        of a discretely sampled walk falls short of the continuous maximum
        by about 0.58 standard deviations of one step per side (Broadie,
        Glasserman and Kou, 1997), which at sixty steps puts the squared
        range about eighteen percent low. The bound is set where the
        simulator reads, and the line's 'overnight gap' wording is the
        explanation for real daily bars, not for these."""
        from portfolio.core.spread import range_ratio
        for seed in (5, 6, 7):
            o, h, l, c = bars(1.0, n=2000, seed=seed)
            r = range_ratio(h, l, c)
            assert r is not None and 0.70 < r.ratio < 0.95, r.line()
            assert "not above 1" in r.line() and "overnight gap" in r.line()

    def test_a_widened_range_goes_above_one_and_says_so(self):
        from portfolio.core.spread import range_ratio
        o, h, l, c = bars(1.0, n=2000, seed=5)
        r = range_ratio(h * 1.004, l / 1.004, c)
        assert r.ratio > 1.0
        assert "prices the closes never see" in r.line()

    def test_too_few_bars_is_none(self):
        from portfolio.core.spread import range_ratio
        o, h, l, c = bars(1.0, n=15, seed=5)
        assert range_ratio(h, l, c) is None
