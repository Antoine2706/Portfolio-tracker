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

from portfolio.core.spread import (MINIMUM_BARS, SpreadEstimate, TickSize,
                                   edge, infer_tick_size)
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
