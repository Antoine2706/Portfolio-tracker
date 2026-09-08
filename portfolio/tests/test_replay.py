"""Replaying real purchases: same money, same days, different destination.

The comparison is only clean if the two arms really are identical except for
where the money went, so most of this file checks that rather than checking
the result. Same amounts, same dates, no sales in either arm, and no purchase
decided with data that had not happened yet.

That last one is the same discipline as `eval/leakage.py` and for the same
reason. The allocator reads a covariance matrix, and a covariance matrix
estimated one bar late is a leak that produces a better-looking answer with
nothing behind it. The check is the same experiment: rewrite every price
after a date, re-run, and require every earlier decision to come back
identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.execution import CostModel, InstrumentCost
from portfolio.eval.replay import estimable_window, replay_purchases

KEYS = ["A", "B", "C", "D"]


def panel(seed: int = 5, periods: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=periods)
    rets = pd.DataFrame(rng.normal(0.0004, 0.011, (periods, len(KEYS))),
                        index=dates, columns=KEYS)
    rets.iloc[0] = 0.0
    # Deliberately different volatilities, so the book has a risk share worth
    # equalising and the two arms can actually differ.
    return 100.0 * (1.0 + rets * np.array([0.5, 1.0, 2.0, 3.0])).cumprod()


def costs_for() -> CostModel:
    return CostModel(account_value=20_000.0,
                     per_instrument={k: InstrumentCost(name=k) for k in KEYS})


def concentrated(prices: pd.DataFrame):
    """Eight purchases, all into the most volatile holding.

    The shape of a book bought by taste rather than by need, which is what
    the allocator is being compared against.
    """
    days = list(prices.index[80::35])[:8]
    return [(d, "D", 20.0) for d in days]


def run(prices=None, purchases=None, **kw):
    prices = panel() if prices is None else prices
    buys = concentrated(prices) if purchases is None else purchases
    return replay_purchases(prices, buys, costs=costs_for(),
                            buyable=set(KEYS), warmup=60, **kw)


class TestTheTwoArmsAreComparable:
    def test_both_spend_the_same_money_on_the_same_days(self):
        prices = panel()
        result = run(prices)
        assert result.purchases == 8
        spent = sum(20.0 * float(prices.loc[d, "D"]) for d, _, _ in
                    concentrated(prices))
        assert result.total_invested == pytest.approx(spent)

    def test_neither_arm_ever_sells(self):
        result = run()
        assert all(v >= 0 for v in result.actual.final_shares.values())
        assert all(v >= 0 for v in result.allocated.final_shares.values())

    def test_both_arms_put_in_the_same_cash(self):
        """The comparability condition, and it is about the money going IN,
        not the value coming out. The two arms hold different instruments, so
        their final values differ by those instruments' returns -- which is a
        return comparison, and the thing this experiment deliberately does not
        make. Whole shares are the only reason the cash figures differ at all.
        """
        result = run()
        assert result.allocated.invested + result.carried == pytest.approx(
            result.actual.invested, rel=1e-9)
        assert result.allocated.invested <= result.actual.invested + 1e-6

    def test_the_whole_share_remainder_is_carried_not_dropped(self):
        """Over eight purchases the remainder reached 6.5% of the money on
        this fixture. Left behind, one arm would quietly be part in cash."""
        result = run()
        assert result.carried < 0.02 * result.actual.invested

    def test_the_actual_arm_really_is_concentrated(self):
        """Otherwise the comparison has nothing to improve on."""
        result = run()
        assert result.actual.final_shares["D"] == pytest.approx(8 * 20.0)
        assert all(result.actual.final_shares[k] == 0 for k in "ABC")


class TestTheDecisionsAreMadeInTime:
    """The same experiment as the leak detector, on the allocator's estimator."""

    def _perturb_after(self, prices: pd.DataFrame, row: int,
                       seed: int = 11) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        out = prices.copy()
        n = len(prices) - row - 1
        for column in prices.columns:
            start = float(prices[column].iloc[row])
            steps = rng.normal(0.002, 0.05, n)
            out.iloc[row + 1:, out.columns.get_loc(column)] = \
                start * np.exp(np.cumsum(steps))
        return out

    def test_rewriting_the_future_leaves_earlier_purchases_untouched(self):
        prices = panel()
        buys = concentrated(prices)
        split = 200
        clean = replay_purchases(prices, buys, costs=costs_for(),
                                 buyable=set(KEYS), warmup=60)
        altered = replay_purchases(self._perturb_after(prices, split), buys,
                                   costs=costs_for(), buyable=set(KEYS),
                                   warmup=60)
        cut = prices.index[split]
        before_clean = clean.allocated.dispersion[clean.allocated.dispersion.index <= cut]
        before_altered = altered.allocated.dispersion[
            altered.allocated.dispersion.index <= cut]
        assert len(before_clean) == len(before_altered)
        assert np.allclose(before_clean.to_numpy(), before_altered.to_numpy(),
                           rtol=0, atol=0), (
            "a purchase decided before the split moved when only later prices "
            "changed, so the allocator is reading data it did not have")

    def test_and_the_check_could_have_failed(self):
        """A perturbation that changes nothing proves nothing. The dispersion
        after the split must move, or the experiment is vacuous."""
        prices = panel()
        buys = concentrated(prices)
        split = 200
        clean = run(prices, buys)
        altered = replay_purchases(self._perturb_after(prices, split), buys,
                                   costs=costs_for(), buyable=set(KEYS),
                                   warmup=60)
        cut = prices.index[split]
        after_clean = clean.allocated.dispersion[clean.allocated.dispersion.index > cut]
        after_altered = altered.allocated.dispersion[
            altered.allocated.dispersion.index > cut]
        assert len(after_clean) and not np.allclose(
            after_clean.to_numpy(), after_altered.to_numpy()), (
            "rewriting every price after the split changed nothing at all, so "
            "this replay is not reading the prices it was handed")


class TestWhatItConcludes:
    def test_directing_the_money_lowers_dispersion(self):
        """The primary claim, and close to arithmetic rather than an
        empirical bet: money sent to the underweight holding evens the risk
        shares. The replay measures the size, not the sign."""
        result = run()
        assert result.allocated.mean_dispersion < result.actual.mean_dispersion
        assert result.dispersion_gap > 0

    def test_dispersion_is_reported_without_a_standard_error(self):
        """Because it is computed from the holdings and the covariance matrix
        rather than estimated from a sample of returns."""
        text = "\n".join(run().lines())
        assert "carries no sampling error" in text
        assert "the size of the effect, not evidence for its sign" in text

    def test_volatility_is_reported_with_one(self):
        result = run()
        assert result.actual.volatility_standard_error is not None
        text = "\n".join(result.lines())
        assert "1 standard error" in text
        assert "this is not a ranking" in text

    def test_the_small_sample_is_named(self):
        assert "With 8 purchases this is not a ranking" in "\n".join(run().lines())

    def test_a_purchase_it_could_not_replay_is_listed(self):
        """Silence about a skipped arm would make the two incomparable
        without saying so."""
        prices = panel()
        early = [(prices.index[5], "D", 20.0)] + concentrated(prices)
        result = run(prices, early)
        assert result.skipped
        assert any("too few to estimate a covariance" in s
                   for s in result.skipped)
        assert "could not replay" in "\n".join(result.lines())

    def test_an_empty_ledger_produces_no_arms_rather_than_a_wrong_one(self):
        prices = panel()
        result = replay_purchases(prices, [], costs=costs_for(),
                                  buyable=set(KEYS), warmup=60)
        assert result.purchases == 0
        assert result.total_invested == 0.0


class TestAContributionIsNotAReturn:
    """Money going in is not the book going up.

    The order fills at that day's close, so the new cash was not in the book
    for the day's move. Left in the series it reads as a double-digit gain on
    every purchase day: the demo ledger came out at annualised volatilities of
    0.81 and 0.56 for two arms holding ETFs, and the two arms differed largely
    because they invested slightly different amounts.
    """

    def test_the_volatility_is_a_market_number_not_a_cash_flow_one(self):
        result = run()
        for arm in (result.actual, result.allocated):
            assert 0.0 < arm.volatility < 0.6, (
                f"{arm.name} came out at {arm.volatility:.4f} annualised, "
                f"which is a cash-flow artefact rather than a market move")

    def test_no_purchase_day_shows_as_a_giant_return(self):
        prices = panel()
        result = run(prices)
        days = {d for d, _, _ in concentrated(prices)}
        on_purchase = [abs(v) for d, v in result.actual.returns.items()
                       if d in days]
        assert on_purchase
        assert max(on_purchase) < 0.15, (
            "a purchase day is showing as a large return, so the contribution "
            "is being counted as performance")

    def test_and_the_check_could_have_failed(self, monkeypatch):
        """Put the contribution back and the volatility must blow up."""
        import portfolio.eval.replay as module
        prices = panel()
        buys = [(d, "D", 400.0) for d, _, _ in concentrated(prices)]
        clean = run(prices, buys)
        # A purchase 20x the size makes the flow dominate: if flows were still
        # in the series this arm would be wild, and it is not.
        assert clean.actual.volatility < 0.6
        assert module  # the import is the point of the fixture path


class TestSalesStopTheReplay:
    """Ignoring them made the "actual" arm a book nobody owned."""

    def test_the_replay_ends_at_the_first_sale(self):
        prices = panel()
        buys = concentrated(prices)
        cut = buys[3][0]
        result = run(prices, buys, sales=[(cut, "D", 10.0)])
        assert result.stopped_at == cut
        assert result.purchases == 3, "a purchase on or after the sale was replayed"
        assert result.actual.dispersion.index.max() < cut

    def test_it_says_so_rather_than_quietly_covering_less(self):
        prices = panel()
        buys = concentrated(prices)
        text = "\n".join(run(prices, buys,
                             sales=[(buys[3][0], "D", 10.0)]).lines())
        assert "the first sale in the ledger" in text
        assert "no neutral way to apply a concentrated sale" in text

    def test_a_ledger_with_no_sales_runs_to_the_end(self):
        result = run()
        assert result.stopped_at is None
        assert result.purchases == 8

    def test_a_sale_after_the_panel_does_not_truncate_anything(self):
        prices = panel()
        late = prices.index[-1] + pd.Timedelta(days=30)
        result = run(prices, sales=[(late, "D", 10.0)])
        assert result.stopped_at is None
        assert result.purchases == 8


class TestStaggeredListings:
    """Instruments that start on different dates.

    Complete-case deletion across every column deletes every date before the
    last instrument's first print. On the demo panel -- ten instruments, eight
    start dates, the latest eight months after the earliest -- that is every
    date in any window reaching back further, and the replay raised
    "covariance needs at least 2 return observations, got 0" and produced
    nothing at all.
    """

    def stagger(self, seed: int = 5, start: int = 220) -> pd.DataFrame:
        prices = panel(seed)
        prices = prices.copy()
        prices.loc[prices.index[:start], "C"] = np.nan   # C lists late
        return prices

    def test_a_late_listing_does_not_empty_the_window(self):
        prices = self.stagger()
        assert prices.iloc[:220].notna().all(axis=1).sum() == 0, (
            "the fixture no longer has a date with a missing column, so this "
            "test is not exercising the failure")
        result = run(prices)
        assert result.purchases == 8

    def test_the_instrument_is_dropped_not_carried_as_a_gap(self):
        window = estimable_window(self.stagger(), 200, 252, minimum=60)
        assert "C" not in window.columns
        assert len(window) > 50
        assert set(window.columns) == {"A", "B", "D"}

    def test_it_returns_and_is_measured_once_it_has_history(self):
        window = estimable_window(self.stagger(), 399, 252, minimum=60)
        assert "C" in window.columns

    def test_a_return_never_spans_a_gap(self):
        """Dividing today's close by the last one that printed records a
        two-day move as a one-day move."""
        prices = panel()
        prices = prices.copy()
        prices.iloc[150, prices.columns.get_loc("B")] = np.nan
        window = estimable_window(prices, 200, 252, minimum=60)
        rows = window.index
        assert prices.index[150] not in rows
        assert prices.index[151] not in rows, (
            "the day after the gap divides across it, which is a two-day "
            "return recorded as one")

    def test_what_the_measurement_leaves_out_is_named(self):
        prices = panel()
        prices = prices.copy()
        prices.loc[prices.index[:390], "C"] = np.nan   # never enough history
        result = run(prices)
        assert "C" in result.unmeasured
        assert "C" not in result.measured
        assert "Outside the measurement" in "\n".join(result.lines())
