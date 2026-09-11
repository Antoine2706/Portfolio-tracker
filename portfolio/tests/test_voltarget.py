"""Volatility targeting, the de-risking half: the policy does what the
pre-registration says and nothing it forbids.

Every number here is on synthetic prices. The pre-registered run is against
the real book, and it is the user's to run and register.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.base import MarketView, Proposal
from portfolio.agents.voltarget import (TARGET_VOLATILITY, VolatilityTarget,
                                        scale_factor)


def _closes(daily_sigma, *, periods=400, seed=21, cols=("GOLD", "A", "B", "C")):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=periods)
    rets = pd.DataFrame(rng.normal(0.0002, daily_sigma, (periods, len(cols))),
                        index=dates, columns=list(cols))
    rets.iloc[0] = 0.0
    return 100.0 * (1.0 + rets).cumprod()


def _view(daily_sigma=0.011, held=None, **kw):
    closes = _closes(daily_sigma, **kw)
    return MarketView(as_of=closes.index[-1], _closes=closes,
                      held=held or {"GOLD": 0.2, "A": 0.3, "B": 0.3, "C": 0.2})


def _portfolio_vol(weights, cov):
    w = np.array([weights.get(c, 0.0) for c in cov.columns])
    return math.sqrt(float(w @ cov.to_numpy() @ w) * 252)


class TestTheFormula:
    def test_without_a_frozen_part_it_is_the_one_line_formula(self):
        """k = min(1, target / forecast), to the last decimal."""
        view = _view(0.02)
        cov = view.covariance(252, view.available)
        held = dict(view.held)
        s = scale_factor(cov, held, {}, target=0.15)
        assert s.k == pytest.approx(min(1.0, 0.15 / _portfolio_vol(held, cov)))
        assert s.achieved == pytest.approx(0.15)

    def test_it_never_levers(self):
        """Calm markets: the root exceeds one and the answer is one. The
        thing the whole pre-registration is about."""
        view = _view(0.003)
        cov = view.covariance(252, view.available)
        s = scale_factor(cov, dict(view.held), {}, target=0.15)
        assert s.unconstrained > 1.0 and s.k == 1.0 and s.capped

    def test_a_frozen_part_is_solved_for_not_scaled(self):
        """The quadratic: the achieved volatility is the target, with the
        frozen weight where it was and only the tradeable part scaled."""
        view = _view(0.02)
        cov = view.covariance(252, view.available)
        invested = {"A": 0.35, "B": 0.25, "C": 0.2}
        frozen = {"GOLD": 0.2}
        s = scale_factor(cov, invested, frozen, target=0.15)
        assert 0.0 < s.k < 1.0
        proposed = {k: v * s.k for k, v in invested.items()}
        proposed.update(frozen)
        assert _portfolio_vol(proposed, cov) == pytest.approx(0.15)
        assert s.achieved == pytest.approx(0.15)

    def test_a_frozen_part_above_the_target_leaves_only_cash(self):
        view = _view(0.02)
        cov = view.covariance(252, view.available)
        s = scale_factor(cov, {"A": 0.5}, {"GOLD": 0.5}, target=0.02)
        assert s.k == 0.0 and s.frozen_alone > 0.02

    def test_a_non_positive_target_is_refused(self):
        view = _view(0.02)
        cov = view.covariance(252, view.available)
        with pytest.raises(ValueError):
            scale_factor(cov, dict(view.held), {}, target=0.0)


class TestThePolicy:
    def test_the_default_target_is_the_pre_registered_one(self):
        assert TARGET_VOLATILITY == 0.15
        assert VolatilityTarget().target == 0.15
        assert VolatilityTarget().lookback == 252

    def test_it_scales_the_held_mix_and_never_reweights_it(self):
        """Among the tradeable lines the policy is buy-and-hold: the
        proportions proposed are the proportions held."""
        view = _view(0.02, held={"A": 0.5, "B": 0.3, "C": 0.2})
        proposal = VolatilityTarget().observe(view)
        assert 0.0 < sum(proposal.weights.values()) < 1.0
        for a, b in (("A", "B"), ("B", "C")):
            assert (proposal.weights[a] / proposal.weights[b]
                    == pytest.approx(view.held[a] / view.held[b]))
        assert "scaled by" in proposal.reason and "in cash" in proposal.reason

    def test_the_cash_left_by_the_last_decision_is_not_treated_as_the_book(self):
        """After de-risking, the held tradeable weights sum to less than
        one. The fully invested composition is what is scaled, so a calm
        market brings the book back to fully invested rather than leaving
        it at last time's scale."""
        view = _view(0.003, held={"A": 0.3, "B": 0.2, "C": 0.1})   # 40% cash held
        proposal = VolatilityTarget().observe(view)
        assert sum(proposal.weights.values()) == pytest.approx(1.0)
        assert proposal.weights["A"] == pytest.approx(0.5)
        assert "cannot lever" in proposal.reason

    def test_the_frozen_holding_is_pinned_where_it_is_and_named(self):
        view = _view(0.02, held={"GOLD": 0.17, "A": 0.4, "B": 0.3, "C": 0.13})
        proposal = VolatilityTarget(fixed={"GOLD": 0.99}).observe(view)
        assert proposal.weights["GOLD"] == pytest.approx(0.17)
        assert "outside the scaling: GOLD at 17.0%" in proposal.reason
        # and the achieved volatility, frozen part included, is the target
        cov = view.covariance(252, view.available)
        assert _portfolio_vol(proposal.weights, cov) == pytest.approx(0.15, rel=1e-6)

    def test_the_referee_accepts_every_proposal(self):
        """The policy respects the account's constraints on its own; the
        referee is the independent check that it did."""
        from portfolio.agents.execution import CostModel, InstrumentCost
        from portfolio.agents.referee import Referee
        costs = CostModel(account_value=15_000.0, per_instrument={
            "GOLD": InstrumentCost(broker="Keytrade", tob_rate=None),
            "A": InstrumentCost(), "B": InstrumentCost(), "C": InstrumentCost()})
        referee = Referee(costs=costs, tradeable={"A", "B", "C"},
                          enforce_minimum_trade=False)
        for sigma in (0.003, 0.011, 0.02, 0.04):
            view = _view(sigma)
            proposal = VolatilityTarget(fixed={"GOLD": 0.0}).observe(view)
            verdict = referee.review(proposal, dict(view.held))
            assert verdict.allowed, verdict.explain()

    def test_a_levered_answer_cannot_leave_the_policy(self):
        """Sabotage: force the factor above one and the proposal itself
        refuses, because weights over one are not a portfolio this account
        can hold. The cap is not the only thing standing between the
        symmetric literature version and this book."""
        with pytest.raises(ValueError, match="unlevered"):
            Proposal({"A": 0.8, "B": 0.5}, 0.5, "levered", "volatility-target")

    def test_too_little_history_holds_rather_than_guessing(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        closes = pd.DataFrame({"A": np.linspace(100, 110, 10)}, index=dates)
        view = MarketView(as_of=dates[-1], _closes=closes, held={"A": 1.0})
        proposal = VolatilityTarget().observe(view)
        assert proposal.weights == {"A": 1.0}
        assert "holding" in proposal.reason


class TestEndToEnd:
    """Through the harness on a panel whose second half is five times as
    volatile as its first, so that the policy binds where it should."""

    def _panel(self, seed=5):
        from portfolio.eval.harness import Panel
        rng = np.random.default_rng(seed)
        n = 900
        dates = pd.bdate_range("2021-01-01", periods=n)
        # 0.6% a day on three uncorrelated lines at 40/30/30 is 5.6%
        # annualised, well under target; 3% a day is 27.8%, so the scalar
        # should settle near 0.15 / 0.278 = 0.54 once the window is full.
        sigma = np.where(np.arange(n) < 450, 0.006, 0.03)[:, None]
        rets = rng.normal(0.0003, 1.0, (n, 3)) * sigma
        rets[0] = 0.0
        closes = pd.DataFrame(100.0 * (1.0 + rets).cumprod(axis=0),
                              index=dates, columns=["A", "B", "C"])
        return Panel(closes=closes)

    def test_it_de_risks_the_volatile_half_and_holds_the_calm_one(self):
        from portfolio.eval.harness import Execution, buy_and_hold, walk_forward
        panel = self._panel()
        weights = {"A": 0.4, "B": 0.3, "C": 0.3}
        policy = VolatilityTarget()
        result = walk_forward(panel, policy, warmup=253, rebalance_every=21,
                              execution=Execution.NEXT_CLOSE,
                              initial_weights=weights)
        benchmark = buy_and_hold(panel, weights, warmup=253,
                                 execution=Execution.NEXT_CLOSE)
        invested = result.weights.sum(axis=1)
        # Calm regime, measured on a full calm window: fully invested.
        assert invested.iloc[:150].min() > 0.999
        # Volatile regime, once the window has seen it: scaled down.
        assert invested.iloc[-100:].max() < 0.7
        # And the realised volatility over the volatile half is below the
        # benchmark's, which is the one claim the policy can make.
        late = result.returns.index[-300:]
        assert (result.returns.loc[late].std()
                < 0.8 * benchmark.returns.loc[late].std())
        assert all("volatility target 15.0%" in d.reason for d in result.decisions)

    def test_the_run_reports_the_three_pre_registered_criteria(self):
        """`research.run_volatility_target` on a synthetic book: the
        report carries the three criteria, each with its number, and the
        verdict is all three, not any one."""
        from portfolio.research import VolatilityTargetRun, criteria_lines
        from portfolio.eval.harness import Execution, buy_and_hold, walk_forward
        from portfolio.eval.report import compare
        panel = self._panel()
        weights = {"A": 0.4, "B": 0.3, "C": 0.3}
        result = walk_forward(panel, VolatilityTarget(), warmup=253,
                              rebalance_every=21,
                              execution=Execution.NEXT_CLOSE,
                              initial_weights=weights)
        benchmark = buy_and_hold(panel, weights, warmup=253,
                                 execution=Execution.NEXT_CLOSE)
        comparison = compare(result, benchmark, overlap=21, weights=weights)
        run = VolatilityTargetRun(comparison, result, target=0.15)
        text = "\n".join(criteria_lines(run))
        assert "realised volatility" in text and "within 15%" in text
        assert "matched-risk gap" in text and "|t| < 2" in text
        assert "cost" in text and "0.50% a year" in text
        assert "all three" in text
        assert run.binding_share is not None and 0.0 < run.binding_share < 1.0
