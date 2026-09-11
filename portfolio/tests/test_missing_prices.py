"""A day an instrument did not trade is not a 0.00% return.

The defect, found by running the harness on a real book
-------------------------------------------------------
Six dates in a one-year window were venue holidays: Amsterdam open while
Xetra was shut, or the reverse. The panel is an outer join across venues, so
those dates exist as rows with a NaN in one column. The harness carried the
NaN at a zero return -- correct for the *valuation*, since the position had
not vanished -- and then let that zero into the Sharpe ratio and the
volatility as if it were an observed daily return.

Two things followed, and the second was worse than the first.

A fabricated zero suppresses variance. Roughly 4% of the sample was an
invented number in a place where invented numbers look exactly like measured
ones. That is the failure this project has now caught five times.

And the money vanished. `prev_close` was the previous *row*, so the day after
a gap divided by a NaN and was also carried at zero -- one missing print
producing two dead days and destroying the two-day move rather than deferring
it. On the fixture below that silently removed 0.39 percentage points from a
4.35% total return.

The rule now, stated once
------------------------
*Value* against the last price that printed. *Estimate* only from returns
whose two endpoint prices were both observed, on consecutive panel dates.

That keeps the money (the missed move lands on the next print, so the
compounded return is exactly what a buy-and-hold investor experienced) and
throws away the observation (neither the gap day nor the day after it is a
one-day return). Both halves are necessary: keeping the money without
dropping the observation is the original bug; dropping both loses real
performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.base import MarketView
from portfolio.eval.harness import Execution, Hold, Panel, walk_forward

PERIODS = 140
GAP = 100                       # the row an instrument fails to print on


def prices(seed: int = 3, columns=("A", "B")) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=PERIODS)
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(rng.normal(0.0004, 0.011, (PERIODS, len(columns))),
                        index=dates, columns=list(columns))
    rets.iloc[0] = 0.0
    return 100.0 * (1.0 + rets).cumprod()


def with_gap(frame: pd.DataFrame, row: int = GAP, col: int = 0) -> pd.DataFrame:
    """The same price path, with one venue shut for a day."""
    out = frame.copy()
    out.iloc[row, col] = np.nan
    return out


def hold(frame: pd.DataFrame):
    """Buy-and-hold: no trades, so the answer is a pure function of prices."""
    start = {c: 1.0 / len(frame.columns) for c in frame.columns}
    return walk_forward(Panel(closes=frame), Hold(), warmup=60,
                        rebalance_every=len(frame) + 1, cost_model=None,
                        execution=Execution.NEXT_CLOSE, initial_weights=start,
                        frozen=frozenset(frame.columns))


class TestTheMoneyIsNotLost:
    """The half that is an invariance, and can therefore be stated exactly.

    A buy-and-hold portfolio's total return depends only on the endpoint
    prices of each holding. A day one of them did not print changes neither
    endpoint, so it cannot change the answer. Before the fix it changed it by
    0.39 percentage points.
    """

    def test_a_venue_holiday_does_not_change_the_total_return(self):
        clean, gapped = prices(), with_gap(prices())
        a, b = hold(clean).track(), hold(gapped).track()
        assert b.total_return == pytest.approx(a.total_return, abs=1e-12), (
            "a day an instrument did not trade changed the compounded return, "
            "so a real move has been destroyed rather than deferred")

    def test_the_missed_move_is_deferred_rather_than_destroyed(self):
        """Over the pair of days the compounding is exactly right, even though
        neither day on its own is."""
        clean = prices()
        assert abs(clean.iloc[GAP, 0] / clean.iloc[GAP - 1, 0] - 1.0) > 1e-4, (
            "the fixture's gap day is flat, so it would pass either way")
        a, b = hold(clean).returns, hold(with_gap(clean)).returns
        pair = clean.index[[GAP, GAP + 1]]
        assert float((1.0 + b.loc[pair]).prod()) == pytest.approx(
            float((1.0 + a.loc[pair]).prod()), abs=1e-12)
        assert b.loc[pair[0]] != pytest.approx(a.loc[pair[0]], abs=1e-9), (
            "the gap day is unchanged, so nothing was deferred and this test "
            "is not exercising the case it describes")

    @pytest.mark.parametrize("row", [61, 90, 100, PERIODS - 2])
    def test_wherever_the_gap_falls(self, row):
        clean = prices()
        a, b = hold(clean).track(), hold(with_gap(clean, row=row)).track()
        assert b.total_return == pytest.approx(a.total_return, abs=1e-12)

    def test_several_gaps_at_once(self):
        clean = prices()
        gapped = clean.copy()
        for row, col in ((70, 0), (71, 1), (95, 0), (96, 0), (120, 1)):
            gapped.iloc[row, col] = np.nan
        a, b = hold(clean).track(), hold(gapped).track()
        assert b.total_return == pytest.approx(a.total_return, abs=1e-12)
        assert b.excluded_observations >= 5


class TestTheObservationIsThrownAway:
    def test_the_gap_day_and_the_day_after_are_both_excluded(self):
        """The gap day cannot be marked; the day after carries two days."""
        result = hold(with_gap(prices()))
        excluded = result.returns.index[~result.measured.to_numpy(dtype=bool)]
        assert list(excluded) == list(prices().index[[GAP, GAP + 1]])

    def test_every_admitted_return_is_a_real_one_day_return(self):
        """The property the fabricated zero violated.

        Whatever survives into the estimator must be a return the portfolio
        genuinely earned over exactly one day of the true price path.
        """
        clean = prices()
        a, b = hold(clean), hold(with_gap(clean))
        admitted = b.estimable
        truth = a.returns.loc[admitted.index]
        assert np.allclose(admitted.to_numpy(), truth.to_numpy(),
                           rtol=0, atol=1e-15), (
            "an admitted return differs from the one the clean panel produced, "
            "so the estimator is being fed something that did not happen")

    def test_no_zero_is_smuggled_into_the_variance(self):
        result = hold(with_gap(prices()))
        assert not (result.estimable.abs() < 1e-15).any(), (
            "a fabricated 0.00% return reached the estimator")

    def test_the_volatility_is_not_diluted_by_the_gap(self):
        """The direction of the original error: an extra zero pulls it down."""
        clean = prices()
        honest = hold(with_gap(clean)).track()
        naive = hold(with_gap(clean))
        diluted = float(naive.returns.std(ddof=1) * np.sqrt(252))
        assert honest.volatility > diluted, (
            "including the fabricated zeros no longer suppresses volatility, "
            "so this fixture is not exercising the defect")

    def test_the_excluded_days_are_counted_and_reported(self):
        record = hold(with_gap(prices())).track()
        assert record.excluded_observations == 2
        assert record.observations + record.excluded_observations == \
            len(hold(prices()).returns)

    def test_a_zero_weight_holding_costs_no_days(self):
        """Its missing price cannot move the portfolio, so flagging it would
        throw away good observations for nothing."""
        frame = prices(columns=("A", "B", "C"))
        gapped = with_gap(frame, col=2)
        start = {"A": 0.5, "B": 0.5}          # C is in the panel, not the book
        result = walk_forward(Panel(closes=gapped), Hold(), warmup=60,
                              rebalance_every=len(frame) + 1, cost_model=None,
                              execution=Execution.NEXT_CLOSE,
                              initial_weights=start, frozen=frozenset(start))
        assert result.excluded_days == 0
        assert not result.warnings


class TestTheCovarianceWindow:
    """The same rule, applied where the policy estimates rather than the harness.

    This path never fabricated a zero -- it dropped the missing row. What it
    did instead was let the *next* row's `pct_change` reach back over the gap,
    so a two-day move entered the sample as a one-day observation. That
    inflates the variance of whichever instrument was shut, and through it
    every covariance it appears in.
    """

    def _view(self, frame):
        return MarketView(as_of=frame.index[-1], _closes=frame, held={},
                          min_observations=10)

    def test_no_return_spans_a_gap(self):
        clean = prices(columns=("A", "B", "C"))
        gapped = with_gap(clean)
        admitted = self._view(gapped).returns(120)
        truth = self._view(clean).returns(120).loc[admitted.index]
        assert np.array_equal(admitted.to_numpy(), truth.to_numpy()), (
            "a return in the covariance window is not the one-day return the "
            "clean panel gives for that date, so it spans the gap")

    def test_both_contaminated_dates_are_dropped(self):
        clean = prices(columns=("A", "B", "C"))
        dropped = (self._view(clean).returns(120).index
                   .difference(self._view(with_gap(clean)).returns(120).index))
        assert list(dropped) == list(clean.index[[GAP, GAP + 1]])

    def test_the_spanning_return_is_wrong_on_average_and_unpredictable_on_one(self):
        """Why it is excluded rather than corrected.

        A two-day return carries twice the variance of a one-day one, so over
        many samples the old estimator sits about 1/n above the honest one --
        with n = 120 admitted returns, roughly 0.8%. On any single sample it
        lands either side, because the two days can offset as easily as
        reinforce. A bias nobody can sign is not one a reader can allow for.
        """
        ratios, above = [], 0
        for seed in range(120):
            clean = prices(seed=seed, columns=("A", "B", "C"))
            gapped = with_gap(clean)
            honest = float(self._view(gapped).returns(120)["A"].var(ddof=1))
            old = gapped.iloc[-121:].dropna(how="any").pct_change().iloc[1:]
            spanning = float(old["A"].var(ddof=1))
            ratios.append(spanning / honest)
            above += spanning > honest
        mean_ratio = float(np.mean(ratios))
        assert mean_ratio == pytest.approx(1.0 + 1.0 / 120, abs=0.01), (
            f"the spanning return's effect on the variance is {mean_ratio:.4f} "
            f"on average, not the ~1/n inflation the argument predicts")
        assert 0.25 < above / len(ratios) < 0.75, (
            "the effect has a predictable sign after all, which would make "
            "excluding it a choice rather than the only honest option")

    def test_a_column_nobody_asked_about_costs_nothing(self):
        clean = prices(columns=("A", "B", "C"))
        gapped = with_gap(clean, col=2)
        view = self._view(gapped)
        assert len(view.returns(120, ["A", "B"])) == \
            len(self._view(clean).returns(120, ["A", "B"]))
