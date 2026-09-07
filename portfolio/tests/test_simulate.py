"""The what-if simulator and the weight generators.

The before/after figures are produced by the same functions as the Risk
view, so the tests here check two things: the arithmetic of the change, and
that the CCTR invariant still holds on the simulated book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.core.risk import (annualise_volatility, covariance_matrix,
                                 portfolio_volatility, risk_decomposition)
from portfolio.core.simulate import (equal_weights, min_variance_weights,
                                     risk_parity_weights, simulate_cash_changes,
                                     simulate_target_weights, trades_to_target)

A, B, C = "IE0002Y8CX98", "IE000IAXNM41", "LU1681048630"


@pytest.fixture
def two_asset_cov() -> pd.DataFrame:
    """sigma_A = 0.02, sigma_B = 0.01, rho = 0.5 (the test_risk fixture)."""
    return pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0001]], index=[A, B], columns=[A, B])


@pytest.fixture
def three_asset_cov() -> pd.DataFrame:
    """Two correlated defence ETFs and an uncorrelated commodity."""
    return pd.DataFrame([[0.0004, 0.00036, 0.0], [0.00036, 0.0004, 0.0], [0.0, 0.0, 0.0009]],
                        index=[A, B, C], columns=[A, B, C])


def random_cov(seed: int, n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = pd.DataFrame(rng.normal(0, 0.01, (252, n)), columns=[f"X{i}" for i in range(n)])
    return covariance_matrix(returns)


class TestEqualWeights:
    def test_quarter_each(self):
        assert equal_weights([A, B, C, "D"]) == {A: 0.25, B: 0.25, C: 0.25, "D": 0.25}

    def test_empty(self):
        assert equal_weights([]) == {}


class TestCashChanges:
    def test_by_hand(self, two_asset_cov):
        """A 600, B 400; add 200 to B -> 600 / 600, half each."""
        r = simulate_cash_changes({A: 600.0, B: 400.0}, two_asset_cov, {B: 200.0})
        assert (r.total_before, r.total_after) == (1000.0, 1200.0)
        by = {h.isin: h for h in r.holdings}
        assert (by[A].value_before, by[A].value_after) == (600.0, 600.0)
        assert (by[B].value_before, by[B].value_after) == (400.0, 600.0)
        assert (by[A].weight_before, by[B].weight_before) == (0.6, 0.4)
        assert (by[A].weight_after, by[B].weight_after) == (0.5, 0.5)

    def test_selling_more_than_held_clips_at_zero(self, two_asset_cov):
        r = simulate_cash_changes({A: 600.0, B: 400.0}, two_asset_cov, {B: -1000.0})
        by = {h.isin: h for h in r.holdings}
        assert by[B].value_after == 0.0
        assert r.total_after == 600.0

    def test_no_change_is_identical_before_and_after(self, two_asset_cov):
        r = simulate_cash_changes({A: 600.0, B: 400.0}, two_asset_cov, {})
        assert r.volatility_before == r.volatility_after
        assert r.effective_before == r.effective_after
        for h in r.holdings:
            assert h.risk_before == h.risk_after

    def test_watchlist_isin_enters_from_nothing(self, three_asset_cov):
        r = simulate_cash_changes({A: 600.0, B: 400.0}, three_asset_cov, {C: 500.0})
        by = {h.isin: h for h in r.holdings}
        assert by[C].value_before == 0.0 and by[C].weight_before == 0.0
        assert by[C].value_after == 500.0 and by[C].weight_after == pytest.approx(1 / 3)
        assert by[C].risk_before == 0.0 and by[C].risk_after > 0

    def test_isin_outside_the_risk_model_is_refused_by_name(self, two_asset_cov):
        with pytest.raises(ValueError, match=C):
            simulate_cash_changes({A: 600.0, B: 400.0}, two_asset_cov, {C: 100.0})

    def test_negative_holding_is_refused(self, two_asset_cov):
        with pytest.raises(ValueError, match="negative"):
            simulate_cash_changes({A: -1.0, B: 400.0}, two_asset_cov, {})

    def test_volatility_is_the_annualised_portfolio_volatility(self, two_asset_cov):
        """After: 600 / 600 -> w = (0.5, 0.5); sigma_p = sqrt(0.25 x 0.0004 +
        0.25 x 0.0001 + 2 x 0.25 x 0.0001) = sqrt(0.000175)."""
        r = simulate_cash_changes({A: 600.0, B: 400.0}, two_asset_cov, {B: 200.0})
        assert r.volatility_after == pytest.approx(annualise_volatility(np.sqrt(0.000175)))
        assert r.volatility_before == pytest.approx(
            annualise_volatility(portfolio_volatility([0.6, 0.4], two_asset_cov)))

    def test_adding_the_uncorrelated_asset_lowers_max_risk_share(self, three_asset_cov):
        r = simulate_cash_changes({A: 500.0, B: 500.0}, three_asset_cov, {C: 500.0})
        assert r.max_risk_share_after < r.max_risk_share_before
        assert r.effective_after > r.effective_before
        assert r.diversification_after > r.diversification_before

    @pytest.mark.parametrize("seed", range(8))
    def test_risk_shares_sum_to_one_after_any_change(self, seed):
        """The CCTR invariant on the simulated book, for random books and changes."""
        rng = np.random.default_rng(seed)
        cov = random_cov(seed)
        values = {c: float(v) for c, v in zip(cov.columns, rng.uniform(100, 1000, 5))}
        changes = {c: float(v) for c, v in zip(cov.columns, rng.uniform(-300, 300, 5))}
        r = simulate_cash_changes(values, cov, changes)
        assert sum(h.risk_after for h in r.holdings) == pytest.approx(1.0)
        assert sum(h.risk_before for h in r.holdings) == pytest.approx(1.0)
        assert sum(h.weight_after for h in r.holdings) == pytest.approx(1.0)

    def test_marginal_after_is_the_annualised_mctr(self, two_asset_cov):
        r = simulate_cash_changes({A: 600.0, B: 400.0}, two_asset_cov, {B: 200.0})
        d = risk_decomposition([0.5, 0.5], two_asset_cov)
        by = {h.isin: h for h in r.holdings}
        assert by[A].marginal_after == pytest.approx(annualise_volatility(d.marginal[0]))
        assert by[B].marginal_after == pytest.approx(annualise_volatility(d.marginal[1]))

    def test_holdings_sorted_by_value_after(self, two_asset_cov):
        r = simulate_cash_changes({A: 600.0, B: 400.0}, two_asset_cov, {B: 500.0})
        assert [h.isin for h in r.holdings] == [B, A]

    def test_selling_everything_gives_a_riskless_empty_book(self, two_asset_cov):
        r = simulate_cash_changes({A: 600.0, B: 400.0}, two_asset_cov, {A: -600.0, B: -400.0})
        assert r.total_after == 0.0 and r.volatility_after == 0.0
        assert r.effective_after == 0.0 and r.max_risk_share_after == 0.0


class TestTargetWeights:
    def test_total_is_kept(self, two_asset_cov):
        r = simulate_target_weights({A: 600.0, B: 400.0}, two_asset_cov, {A: 0.5, B: 0.5})
        assert r.total_after == r.total_before == 1000.0
        by = {h.isin: h for h in r.holdings}
        assert by[A].value_after == 500.0 and by[B].value_after == 500.0

    def test_targets_are_rescaled_to_sum_to_one(self, two_asset_cov):
        a = simulate_target_weights({A: 600.0, B: 400.0}, two_asset_cov, {A: 0.5, B: 0.5})
        b = simulate_target_weights({A: 600.0, B: 400.0}, two_asset_cov, {A: 3.0, B: 3.0})
        assert [h.value_after for h in a.holdings] == [h.value_after for h in b.holdings]

    def test_absent_holding_is_sold_out(self, two_asset_cov):
        r = simulate_target_weights({A: 600.0, B: 400.0}, two_asset_cov, {A: 1.0})
        by = {h.isin: h for h in r.holdings}
        assert by[B].value_after == 0.0 and by[A].value_after == 1000.0
        assert r.effective_after == pytest.approx(1.0)

    def test_negative_target_refused(self, two_asset_cov):
        with pytest.raises(ValueError, match="negative"):
            simulate_target_weights({A: 600.0, B: 400.0}, two_asset_cov, {A: 1.2, B: -0.2})

    def test_zero_targets_refused(self, two_asset_cov):
        with pytest.raises(ValueError, match="sum to zero"):
            simulate_target_weights({A: 600.0, B: 400.0}, two_asset_cov, {A: 0.0, B: 0.0})

    def test_equal_weights_give_effective_count_n(self, three_asset_cov):
        r = simulate_target_weights({A: 900.0, B: 50.0, C: 50.0}, three_asset_cov,
                                    equal_weights([A, B, C]))
        assert r.effective_after == pytest.approx(3.0)
        assert sum(h.risk_after for h in r.holdings) == pytest.approx(1.0)


class TestRiskParity:
    def test_diagonal_closed_form(self):
        """sigma = (0.01, 0.02, 0.04): w proportional to 1/sigma = (4, 2, 1)/7."""
        cov = pd.DataFrame(np.diag([0.0001, 0.0004, 0.0016]), index=[A, B, C], columns=[A, B, C])
        w = risk_parity_weights(cov)
        assert w[A] == pytest.approx(4 / 7, abs=1e-8)
        assert w[B] == pytest.approx(2 / 7, abs=1e-8)
        assert w[C] == pytest.approx(1 / 7, abs=1e-8)

    def test_equal_risk_contributions_on_a_correlated_matrix(self, three_asset_cov):
        w = risk_parity_weights(three_asset_cov)
        d = risk_decomposition(w, three_asset_cov)
        assert np.allclose(d.percent, 1 / 3, atol=1e-8)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_the_uncorrelated_asset_gets_more_than_its_volatility_suggests(self, three_asset_cov):
        """C is the most volatile but uncorrelated; it still earns a third of
        the risk with a weight above what inverse volatility alone gives."""
        w = risk_parity_weights(three_asset_cov)
        inverse_vol = (1 / 0.03) / (1 / 0.02 + 1 / 0.02 + 1 / 0.03)
        assert w[C] > inverse_vol

    @pytest.mark.parametrize("seed", range(6))
    def test_converges_on_random_matrices(self, seed):
        cov = random_cov(seed, n=6)
        w = risk_parity_weights(cov)
        d = risk_decomposition(w, cov)
        assert np.allclose(d.percent, 1 / 6, atol=1e-8)
        assert all(v > 0 for v in w.values())

    def test_two_identical_assets_split_evenly(self):
        cov = pd.DataFrame([[0.0004, 0.0004], [0.0004, 0.0004]], index=[A, B], columns=[A, B])
        w = risk_parity_weights(cov)
        assert w[A] == pytest.approx(0.5) and w[B] == pytest.approx(0.5)

    def test_single_asset(self):
        assert risk_parity_weights(pd.DataFrame([[0.0004]], index=[A], columns=[A])) == {A: 1.0}

    def test_empty(self):
        assert risk_parity_weights(pd.DataFrame()) == {}

    def test_zero_variance_asset_refused_by_name(self):
        cov = pd.DataFrame([[0.0004, 0.0], [0.0, 0.0]], index=[A, B], columns=[A, B])
        with pytest.raises(ValueError, match=B):
            risk_parity_weights(cov)


class TestMinVariance:
    def test_diagonal_closed_form(self):
        """variances (0.0001, 0.0003): w proportional to 1/var = (3, 1)/4."""
        cov = pd.DataFrame(np.diag([0.0001, 0.0003]), index=[A, B], columns=[A, B])
        w = min_variance_weights(cov)
        assert w[A] == pytest.approx(0.75, abs=1e-9)
        assert w[B] == pytest.approx(0.25, abs=1e-9)

    def test_two_asset_closed_form_with_correlation(self):
        """w_A = (sigma_B^2 - sigma_AB) / (sigma_A^2 + sigma_B^2 - 2 sigma_AB)
        = (0.0001 - 0.00005) / (0.0004 + 0.0001 - 0.0001) = 0.125."""
        cov = pd.DataFrame([[0.0004, 0.00005], [0.00005, 0.0001]], index=[A, B], columns=[A, B])
        w = min_variance_weights(cov)
        assert w[A] == pytest.approx(0.125, abs=1e-9)

    def test_long_only_constraint_binds(self):
        """sigma_AB above sigma_B^2 makes the unconstrained answer short A;
        long-only puts everything in B."""
        cov = pd.DataFrame([[0.0004, 0.00015], [0.00015, 0.0001]], index=[A, B], columns=[A, B])
        w = min_variance_weights(cov)
        assert w[A] == pytest.approx(0.0, abs=1e-9) and w[B] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("seed", range(6))
    def test_beats_random_long_only_portfolios(self, seed):
        cov = random_cov(seed, n=6)
        best = portfolio_volatility(min_variance_weights(cov), cov)
        rng = np.random.default_rng(100 + seed)
        for _ in range(50):
            probe = rng.random(6)
            assert best <= portfolio_volatility(probe / probe.sum(), cov) + 1e-12

    def test_weights_are_a_long_only_full_allocation(self):
        w = min_variance_weights(random_cov(3))
        assert all(v >= 0 for v in w.values())
        assert sum(w.values()) == pytest.approx(1.0)

    def test_single_asset(self):
        assert min_variance_weights(pd.DataFrame([[0.0004]], index=[A], columns=[A])) == {A: 1.0}

    def test_all_zero_covariance_gives_equal_weights(self):
        cov = pd.DataFrame(np.zeros((2, 2)), index=[A, B], columns=[A, B])
        assert min_variance_weights(cov) == {A: 0.5, B: 0.5}

    def test_empty(self):
        assert min_variance_weights(pd.DataFrame()) == {}


class TestTradesToTarget:
    def test_by_hand(self):
        """A 600, B 400 to 50/50 of 1000: sell 100 A, buy 100 B."""
        trades = trades_to_target({A: 600.0, B: 400.0}, {A: 0.5, B: 0.5})
        by = {t.isin: t for t in trades}
        assert (by[A].action, by[A].amount) == ("SELL", 100.0)
        assert (by[B].action, by[B].amount) == ("BUY", 100.0)
        assert (by[A].weight_before, by[A].weight_after) == (0.6, 0.5)
        assert by[A].units is None

    def test_units_from_prices(self):
        trades = trades_to_target({A: 600.0, B: 400.0}, {A: 0.5, B: 0.5}, prices={A: 12.5, B: 4.0})
        by = {t.isin: t for t in trades}
        assert by[A].units == pytest.approx(8.0) and by[B].units == pytest.approx(25.0)

    def test_sorted_by_amount_descending(self):
        trades = trades_to_target({A: 700.0, B: 200.0, C: 100.0}, equal_weights([A, B, C]))
        assert [t.isin for t in trades] == [A, C, B]          # 366.67, 233.33, 133.33
        assert trades[0].action == "SELL"

    def test_min_amount_drops_small_trades(self):
        trades = trades_to_target({A: 501.0, B: 499.0}, {A: 0.5, B: 0.5}, min_amount=5.0)
        assert trades == []

    def test_total_override_deposits_and_rebalances(self):
        """Total 2000 from a 1000 book: A to 1000 (+400), B to 1000 (+600)."""
        trades = trades_to_target({A: 600.0, B: 400.0}, {A: 0.5, B: 0.5}, total=2000.0)
        by = {t.isin: t for t in trades}
        assert (by[A].action, by[A].amount) == ("BUY", 400.0)
        assert (by[B].action, by[B].amount) == ("BUY", 600.0)

    def test_absent_from_targets_is_sold_out(self):
        trades = trades_to_target({A: 600.0, B: 400.0}, {A: 1.0})
        by = {t.isin: t for t in trades}
        assert (by[B].action, by[B].amount, by[B].weight_after) == ("SELL", 400.0, 0.0)

    def test_new_isin_is_bought_from_nothing(self):
        trades = trades_to_target({A: 1000.0}, {A: 0.5, C: 0.5})
        by = {t.isin: t for t in trades}
        assert (by[C].action, by[C].amount, by[C].weight_before) == ("BUY", 500.0, 0.0)

    def test_targets_rescaled(self):
        a = trades_to_target({A: 600.0, B: 400.0}, {A: 0.5, B: 0.5})
        b = trades_to_target({A: 600.0, B: 400.0}, {A: 2.0, B: 2.0})
        assert a == b

    def test_already_on_target_means_no_trades(self):
        assert trades_to_target({A: 500.0, B: 500.0}, {A: 0.5, B: 0.5}) == []

    def test_zero_price_gives_no_units(self):
        trades = trades_to_target({A: 600.0, B: 400.0}, {A: 0.5, B: 0.5}, prices={A: 0.0})
        assert all(t.units is None for t in trades)
