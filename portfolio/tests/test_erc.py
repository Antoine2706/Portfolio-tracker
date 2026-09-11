"""Equal risk contribution: the closed forms, the constraint, and the solver.

Three kinds of check, in increasing order of what they are worth:

1.  Cases with a known answer. Two uncorrelated assets give weights inversely
    proportional to volatility; identical assets give identical weights. These
    are checks on the solver rather than restatements of it, because the
    answer was available before the code was.
2.  The post-condition. Whatever the covariance matrix, the risk
    contributions of the assets it solved for must actually be equal --
    recomputed from `core.risk`, not from anything the solver carried out.
3.  The case that would have made a naive implementation quietly wrong: a
    holding that hedges the rest of the book, where the standard
    w <- b / (Sigma w) iteration sends weights negative. Gold is exactly such
    a holding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.risk import (EqualRiskContribution, equal_risk_weights,
                                   risk_contribution_spread)
from portfolio.core.risk import risk_decomposition


def cov_from(vols, corr) -> pd.DataFrame:
    v = np.asarray(vols, dtype=float)
    c = np.asarray(corr, dtype=float)
    keys = [f"A{i}" for i in range(len(v))]
    return pd.DataFrame(np.outer(v, v) * c, index=keys, columns=keys)


def random_cov(n: int, seed: int, hedge: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 500)) * rng.uniform(0.004, 0.03, (n, 1))
    if hedge:
        a[-1] = -0.6 * a[:-1].mean(axis=0) + 0.01 * rng.normal(size=500)
    keys = [f"A{i}" for i in range(n)]
    return pd.DataFrame(np.cov(a), index=keys, columns=keys)


class TestKnownAnswers:
    def test_inverse_volatility_when_uncorrelated(self):
        """Uncorrelated assets: equal risk means weight proportional to 1/sigma."""
        cov = cov_from([0.1, 0.2, 0.4], np.eye(3))
        w = equal_risk_weights(cov)
        inverse = np.array([1 / 0.1, 1 / 0.2, 1 / 0.4])
        assert np.allclose(w.to_numpy(), inverse / inverse.sum())

    def test_identical_assets_get_identical_weights(self):
        for rho in (-0.4, 0.0, 0.5, 0.95):
            cov = cov_from([0.2, 0.2], [[1, rho], [rho, 1]])
            assert np.allclose(equal_risk_weights(cov).to_numpy(), [0.5, 0.5])

    def test_a_more_volatile_asset_always_gets_less(self):
        cov = cov_from([0.1, 0.3], [[1, 0.3], [0.3, 1]])
        w = equal_risk_weights(cov)
        assert w.iloc[0] > w.iloc[1]


class TestThePostCondition:
    @pytest.mark.parametrize("n,seed", [(2, 1), (3, 2), (5, 3), (7, 4), (12, 5)])
    def test_risk_contributions_really_are_equal(self, n, seed):
        cov = random_cov(n, seed)
        w = equal_risk_weights(cov)
        assert risk_contribution_spread(w.to_dict(), cov) < 1e-8
        assert float(w.sum()) == pytest.approx(1.0)
        assert (w > 0).all()

    def test_it_matches_the_decomposition_the_tracker_reports(self):
        """The solver and the risk page must agree, or the app contradicts itself."""
        cov = random_cov(6, 11)
        w = equal_risk_weights(cov)
        shares = risk_decomposition(w.to_dict(), cov).percent
        assert np.allclose(shares, 1 / 6, atol=1e-8)

    def test_a_hedging_asset_does_not_break_it(self):
        """The case the naive iteration gets wrong by sending weights negative."""
        cov = random_cov(7, 99, hedge=True)
        assert (cov.to_numpy()[-1, :-1] < 0).all(), "the fixture is not a hedge"
        w = equal_risk_weights(cov)
        assert (w > 0).all(), "a hedging asset produced a non-positive weight"
        assert risk_contribution_spread(w.to_dict(), cov) < 1e-8

    def test_the_naive_iteration_would_have_failed_here(self):
        """Proving the design choice was necessary, not defensive."""
        cov = random_cov(7, 99, hedge=True)
        sigma, w = cov.to_numpy(), np.full(7, 1 / 7)
        for _ in range(50):
            with np.errstate(divide="ignore", invalid="ignore"):
                w = (1 / 7) / (sigma @ w)
            w = w / w.sum()
        assert (w < 0).any(), (
            "the naive update no longer fails on this fixture, so the "
            "justification for the closed-form root needs rechecking")


class TestFixedWeights:
    def test_the_pinned_weight_is_exactly_preserved(self):
        cov = random_cov(5, 7)
        w = equal_risk_weights(cov, fixed={"A4": 0.13})
        assert float(w["A4"]) == pytest.approx(0.13, abs=1e-12)
        assert float(w.sum()) == pytest.approx(1.0)

    def test_the_free_assets_equalise_among_themselves(self):
        cov = random_cov(6, 8)
        free = [f"A{i}" for i in range(5)]
        w = equal_risk_weights(cov, fixed={"A5": 0.25})
        assert risk_contribution_spread(w.to_dict(), cov, among=free) < 1e-8

    def test_the_fixed_asset_still_shapes_the_answer(self):
        """It stays in the covariance matrix: excluding it would equalise
        contributions to a portfolio that does not exist."""
        cov = random_cov(4, 12)
        with_gold = equal_risk_weights(cov, fixed={"A3": 0.20})
        without = equal_risk_weights(cov.iloc[:3, :3])
        rescaled = without * 0.80
        assert not np.allclose(with_gold.iloc[:3].to_numpy(),
                               rescaled.to_numpy(), atol=1e-4), (
            "pinning an asset gave the same answer as deleting it, so it is "
            "not affecting the other holdings' marginal risk")

    def test_pinning_everything_returns_the_input(self):
        cov = random_cov(3, 13)
        fixed = {"A0": 0.2, "A1": 0.3, "A2": 0.5}
        assert equal_risk_weights(cov, fixed=fixed).to_dict() == fixed

    def test_over_allocated_fixed_weights_are_refused(self):
        with pytest.raises(ValueError, match="nothing left to allocate"):
            equal_risk_weights(random_cov(3, 14), fixed={"A0": 0.7, "A1": 0.5})

    def test_an_unknown_instrument_is_refused(self):
        with pytest.raises(ValueError, match="not in the covariance matrix"):
            equal_risk_weights(random_cov(3, 15), fixed={"NOPE": 0.1})


class TestThePolicy:
    def _view(self, cov_seed=21, periods=400, held=None):
        from portfolio.agents.base import MarketView
        rng = np.random.default_rng(cov_seed)
        cols = ["GOLD", "A", "B", "C"]
        dates = pd.bdate_range("2024-01-01", periods=periods)
        rets = pd.DataFrame(rng.normal(0.0002, 0.011, (periods, 4)),
                            index=dates, columns=cols)
        rets.iloc[0] = 0.0
        closes = 100.0 * (1.0 + rets).cumprod()
        return MarketView(as_of=dates[-1], _closes=closes,
                          held=held or {"GOLD": 0.2, "A": 0.3, "B": 0.3, "C": 0.2})

    def test_it_pins_the_frozen_holding_where_it_actually_is(self):
        """Not at a target: a frozen weight drifts and cannot be corrected."""
        view = self._view(held={"GOLD": 0.17, "A": 0.4, "B": 0.3, "C": 0.13})
        proposal = EqualRiskContribution(lookback=252,
                                         fixed={"GOLD": 0.99}).observe(view)
        assert proposal.weights["GOLD"] == pytest.approx(0.17), (
            "the policy used the value in `fixed` rather than the weight held")

    def test_it_equalises_the_others(self):
        view = self._view()
        proposal = EqualRiskContribution(lookback=252,
                                         fixed={"GOLD": 0.0}).observe(view)
        cov = view.covariance(252, view.available)
        assert risk_contribution_spread(proposal.weights, cov,
                                        among=["A", "B", "C"]) < 1e-6

    def test_the_reason_names_what_the_constraint_cost(self):
        """If equal risk wants more gold and cannot have it, that is a finding
        about the account's structure and should be stated, not inferred."""
        view = self._view()
        proposal = EqualRiskContribution(lookback=252,
                                         fixed={"GOLD": 0.0}).observe(view)
        assert "frozen at" in proposal.reason
        assert "unconstrained equal risk would hold" in proposal.reason

    def test_without_a_constraint_it_says_nothing_about_one(self):
        view = self._view()
        proposal = EqualRiskContribution(lookback=252).observe(view)
        assert "frozen" not in proposal.reason
        assert "equal risk contribution across 4" in proposal.reason

    def test_too_little_history_holds_rather_than_guessing(self):
        from portfolio.agents.base import MarketView
        dates = pd.bdate_range("2024-01-01", periods=10)
        closes = pd.DataFrame({"A": np.linspace(100, 110, 10)}, index=dates)
        view = MarketView(as_of=dates[-1], _closes=closes, held={"A": 1.0})
        proposal = EqualRiskContribution().observe(view)
        assert proposal.weights == {"A": 1.0}
        assert "enough history" in proposal.reason
