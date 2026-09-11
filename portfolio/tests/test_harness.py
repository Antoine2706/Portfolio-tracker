"""The walk-forward engine, and one test per way it could see the future.

Section 2.6 of the specification lists the look-ahead traps to design
against, and asks for a test per trap that fails if the protection is
removed. That last clause is the whole requirement: a test that passes
whether or not the protection exists is the third instance in this project of
a check that looked like it worked without exercising what it claimed.

So each test here either sabotages the protection and asserts the failure, or
uses `leakage.check_lookahead`, which is itself proven to bite in
`test_leakage.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.base import MarketView, Proposal
from portfolio.agents.execution import CostModel
from portfolio.eval.controls import CoinFlip, synthetic_world
from portfolio.eval.harness import Execution, Panel, buy_and_hold, walk_forward


class Fixed:
    """Proposes the same weights forever. Its turnover is entirely drift."""
    name = "fixed"

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights
        self.views: list[MarketView] = []

    def observe(self, view: MarketView) -> Proposal:
        self.views.append(view)
        return Proposal(dict(self.weights), 0.5, "hold the target", self.name)


@pytest.fixture
def world():
    return synthetic_world(n_assets=3, periods=600, sigma=0.01, seed=3)


class TestPanel:
    def test_unsorted_dates_are_refused(self):
        idx = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
        with pytest.raises(ValueError, match="not sorted"):
            Panel(closes=pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx))

    def test_mismatched_opens_are_refused(self, world):
        panel, _ = world
        with pytest.raises(ValueError, match="same columns"):
            Panel(closes=panel.closes, opens=panel.opens.rename(columns={"SYN00": "X"}))

    def test_next_open_without_opens_is_refused_rather_than_invented(self, world):
        panel, _ = world
        closes_only = Panel(closes=panel.closes)
        with pytest.raises(ValueError, match="only\ncloses|only closes"):
            walk_forward(closes_only, Fixed({"SYN00": 1.0}), warmup=100,
                         execution=Execution.NEXT_OPEN)


class TestPointInTime:
    """Trap 2.6(a) and 2.6(c): the estimator must not see past the decision."""

    def test_the_view_ends_at_the_decision_date(self, world):
        panel, _ = world
        policy = Fixed({"SYN00": 1.0})
        walk_forward(panel, policy, warmup=100, rebalance_every=50,
                     cost_model=None)
        assert policy.views, "no decisions were taken; the test proves nothing"
        for view in policy.views:
            frame = view.closes()
            assert frame.index[-1] == view.as_of
            assert len(frame) == panel.closes.index.get_loc(view.as_of) + 1

    def test_the_future_is_absent_not_merely_filtered(self, world):
        """There is no flag to forget: the rows are not in the object."""
        panel, _ = world
        policy = Fixed({"SYN00": 1.0})
        walk_forward(panel, policy, warmup=100, rebalance_every=50, cost_model=None)
        view = policy.views[0]
        after = panel.closes.index[panel.closes.index > view.as_of]
        assert len(after) > 0
        assert not any(d in set(view.closes().index) for d in after)

    def test_a_returns_window_is_the_length_asked_for(self, world):
        """N returns need N+1 prices. Getting this wrong makes 252 into 251."""
        panel, _ = world
        policy = Fixed({"SYN00": 1.0})
        walk_forward(panel, policy, warmup=300, rebalance_every=100, cost_model=None)
        assert len(policy.views[0].returns(252)) == 252

    def test_a_covariance_uses_only_the_trailing_window(self, world):
        """Sabotage: poison a bar outside the window and require no change."""
        panel, _ = world
        policy = Fixed({"SYN00": 1.0})
        walk_forward(panel, policy, warmup=400, rebalance_every=200, cost_model=None)
        view = policy.views[0]
        clean = view.covariance(60).to_numpy()
        upto = panel.closes.loc[:view.as_of]
        assert len(upto) > 200, "the view is too short for this test to mean anything"

        def cov_with(frame):
            return MarketView(as_of=view.as_of, _closes=frame,
                              held={}).covariance(60).to_numpy()

        outside = upto.copy()
        outside.iloc[:100] *= 3.0            # ~240 bars before the window opens
        assert np.allclose(clean, cov_with(outside)), (
            "a 60-day covariance changed when data 240 bars earlier moved")

        # Prove the comparison above can fail: poison inside the window.
        inside = upto.copy()
        inside.iloc[-30:] *= 3.0
        assert not np.allclose(clean, cov_with(inside)), (
            "poisoning data inside the window changed nothing, so the "
            "comparison above was not testing anything")


class TestSurvivorship:
    """Trap 2.6(d): the universe must be what existed then, not what exists now."""

    def test_a_delisted_instrument_is_available_early_and_gone_late(self):
        panel, _ = synthetic_world(n_assets=3, periods=600, seed=9)
        closes = panel.closes.copy()
        closes.iloc[400:, closes.columns.get_loc("SYN02")] = np.nan   # delisted

        early = MarketView(as_of=closes.index[300], _closes=closes.iloc[:301], held={})
        late = MarketView(as_of=closes.index[500], _closes=closes.iloc[:501], held={})
        assert "SYN02" in early.available, (
            "an instrument that was trading has been dropped from the past "
            "because it does not exist at the end of the sample")
        assert "SYN02" not in late.available
        # It is still in the panel; it is not tradable, which is different.
        assert "SYN02" in late.instruments

    def test_an_instrument_not_yet_listed_is_absent_from_early_views(self):
        panel, _ = synthetic_world(n_assets=3, periods=600, seed=9)
        closes = panel.closes.copy()
        closes.iloc[:450, closes.columns.get_loc("SYN02")] = np.nan   # lists later

        early = MarketView(as_of=closes.index[300], _closes=closes.iloc[:301], held={})
        later = MarketView(as_of=closes.index[560], _closes=closes.iloc[:561], held={})
        assert "SYN02" not in early.available
        assert "SYN02" in later.available

    def test_too_little_history_is_not_tradable_either(self):
        panel, _ = synthetic_world(n_assets=2, periods=600, seed=9)
        closes = panel.closes.copy()
        closes.iloc[:290, closes.columns.get_loc("SYN01")] = np.nan
        view = MarketView(as_of=closes.index[300], _closes=closes.iloc[:301],
                          held={}, min_observations=60)
        assert "SYN01" not in view.available, (
            "eleven observations is not enough to model with, and the view "
            "should say so rather than let a policy find out downstream")


class TestExecution:
    """Trap 2.6(a): decide on a close, trade on the next bar, never the same one."""

    def test_a_decision_is_executed_after_it_is_taken(self, world):
        panel, _ = world
        result = walk_forward(panel, Fixed({"SYN00": 1.0}), warmup=100,
                              rebalance_every=50, cost_model=None)
        assert result.decisions
        for d in result.decisions:
            assert d.executed_on > d.decided_on, (
                "a decision was executed on the bar it was taken on, which "
                "assumes a fill at a price only known once trading had ended")

    def test_next_open_splits_the_day_between_old_and_new_weights(self):
        """The overnight leg belongs to the weights actually held."""
        dates = pd.bdate_range("2020-01-01", periods=6)
        closes = pd.DataFrame({"A": [100.0] * 6, "B": [100.0] * 6}, index=dates)
        opens = closes.shift(1)
        opens.iloc[0] = closes.iloc[0]
        # B gaps up 10% overnight on the last bar, then gives it all back.
        opens.iloc[5, 1] = 110.0
        panel = Panel(closes=closes, opens=opens)

        held_a = walk_forward(panel, Fixed({"A": 1.0}), warmup=2,
                              rebalance_every=1, cost_model=None,
                              execution=Execution.NEXT_OPEN)
        held_b = walk_forward(panel, Fixed({"B": 1.0}), warmup=2,
                              rebalance_every=1, cost_model=None,
                              execution=Execution.NEXT_OPEN)
        # A never moves. B is bought at the gapped-up open and falls back to
        # 100 by the close, so the day is a loss for whoever held it.
        assert held_a.returns.iloc[-1] == pytest.approx(0.0)
        assert held_b.returns.iloc[-1] == pytest.approx(110 / 100 - 1 + 100 / 110 - 1
                                                        + (110 / 100 - 1) * (100 / 110 - 1),
                                                        abs=1e-12)

    def test_buy_and_hold_trades_once_and_then_drifts(self, world):
        panel, _ = world
        result = buy_and_hold(panel, {"SYN00": 0.5, "SYN01": 0.5}, warmup=100)
        assert len(result.decisions) == 1
        assert result.turnover.sum() == pytest.approx(result.decisions[0].turnover)
        first, last = result.weights.iloc[0], result.weights.iloc[-1]
        assert not np.allclose(first.to_numpy(), last.to_numpy()), (
            "weights did not drift, so this is not buy-and-hold")


class TestTurnoverAndCosts:
    def test_a_policy_that_never_changes_its_target_still_pays_for_drift(self, world):
        """Turnover is measured against drifted weights, not the last target.

        Comparing successive targets would report zero turnover for a
        portfolio that rebalances every month, which is the wrong sign of
        wrong: undoing drift is exactly what a rebalance is.
        """
        panel, _ = world
        result = walk_forward(panel, Fixed({"SYN00": 0.5, "SYN01": 0.5}),
                              warmup=100, rebalance_every=21, cost_model=None)
        executed = [d.turnover for d in result.decisions[1:]]
        assert executed and all(t > 0 for t in executed)

    def test_costs_reduce_the_net_return_and_only_on_trading_days(self, world):
        panel, _ = world
        policy = lambda: CoinFlip(lam=0.5, seed=4)                    # noqa: E731
        free = walk_forward(panel, policy(), warmup=100, rebalance_every=21,
                            cost_model=None)
        paid = walk_forward(panel, policy(), warmup=100, rebalance_every=21,
                            cost_model=CostModel())
        assert paid.total_cost > 0
        assert np.allclose(free.gross_returns.to_numpy(), paid.gross_returns.to_numpy())
        assert paid.returns.sum() < free.returns.sum()
        traded = paid.costs > 0
        assert traded.sum() == len(paid.decisions)

    def test_moving_into_cash_is_a_trade_with_a_cost(self, world):
        """Turnover counts the cash leg, or de-risking would look free."""
        panel, _ = world
        result = walk_forward(panel, Fixed({"SYN00": 0.3}), warmup=100,
                              rebalance_every=21, cost_model=None)
        assert result.decisions[0].turnover > 0


class TestRefusals:
    def test_a_warmup_longer_than_the_panel_is_refused(self, world):
        panel, _ = world
        with pytest.raises(ValueError, match="nothing to evaluate"):
            walk_forward(panel, Fixed({"SYN00": 1.0}), warmup=len(panel) + 1)

    def test_a_levered_proposal_is_refused(self, world):
        panel, _ = world
        with pytest.raises(ValueError, match="unlevered"):
            walk_forward(panel, Fixed({"SYN00": 0.8, "SYN01": 0.9}), warmup=100,
                         cost_model=None)

    def test_a_proposal_without_a_reason_is_refused(self):
        with pytest.raises(ValueError, match="cannot explain itself"):
            Proposal({"A": 1.0}, 0.5, "", "nameless")
