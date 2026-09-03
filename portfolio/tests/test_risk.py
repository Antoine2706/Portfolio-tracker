"""Risk mathematics, anchored on the CCTR invariant.

Where a value can be computed by hand it is, with the arithmetic written in the
docstring so the test is checkable without running it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.core.risk import (HIGH_CORRELATION_THRESHOLD, annualise_volatility,
                                 beta, concentration, correlation_matrix,
                                 covariance_matrix, diversification_ratio,
                                 drawdown, high_correlation_pairs,
                                 portfolio_volatility, risk_decomposition)

A, B, C = "IE0002Y8CX98", "IE000IAXNM41", "LU1681048630"


@pytest.fixture
def two_asset_cov() -> pd.DataFrame:
    """σ_A = 0.02, σ_B = 0.01, ρ = 0.5.

    Σ_AA = 0.02²        = 0.0004
    Σ_BB = 0.01²        = 0.0001
    Σ_AB = 0.5x0.02x0.01 = 0.0001
    """
    return pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0001]],
                        index=[A, B], columns=[A, B])


@pytest.fixture
def random_cov() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0, 0.01, (252, 5)),
                           columns=[f"IE0002Y8CX9{i}" for i in range(5)])
    return covariance_matrix(returns)


class TestPortfolioVolatilityByHand:
    def test_two_asset_case(self, two_asset_cov):
        """
        w = (0.6, 0.4)
        σ_p² = 0.36(0.0004) + 0.16(0.0001) + 2(0.6)(0.4)(0.0001)
             = 0.000144 + 0.000016 + 0.000048
             = 0.000208
        σ_p  = sqrt(0.000208) = 0.0144222051018...
        """
        got = portfolio_volatility([0.6, 0.4], two_asset_cov)
        assert got == pytest.approx(np.sqrt(0.000208), rel=1e-12)
        assert round(got, 9) == 0.014422205

    def test_single_asset_volatility_is_its_own(self, two_asset_cov):
        assert portfolio_volatility([1.0, 0.0], two_asset_cov) == pytest.approx(0.02)

    def test_perfectly_correlated_assets_do_not_diversify(self):
        """ρ = 1: σ_p is exactly the weighted average, DR is exactly 1."""
        cov = pd.DataFrame([[0.0004, 0.0002], [0.0002, 0.0001]],
                           index=[A, B], columns=[A, B])
        w = [0.5, 0.5]
        assert portfolio_volatility(w, cov) == pytest.approx(0.5 * 0.02 + 0.5 * 0.01)
        assert diversification_ratio(w, cov) == pytest.approx(1.0)

    def test_annualisation_is_sqrt_252(self):
        assert annualise_volatility(0.01) == pytest.approx(0.01 * np.sqrt(252))

    def test_weights_are_matched_by_name_not_position(self, two_asset_cov):
        """A dict in the wrong order must not silently price a different portfolio."""
        by_name = portfolio_volatility({B: 0.4, A: 0.6}, two_asset_cov)
        by_order = portfolio_volatility([0.6, 0.4], two_asset_cov)
        assert by_name == pytest.approx(by_order)

    def test_missing_weight_is_refused(self, two_asset_cov):
        with pytest.raises(ValueError, match="no weight supplied"):
            portfolio_volatility({A: 1.0}, two_asset_cov)

    def test_wrong_length_vector_is_refused(self, two_asset_cov):
        with pytest.raises(ValueError, match="expected"):
            portfolio_volatility([0.5, 0.3, 0.2], two_asset_cov)


class TestTheInvariant:
    """CCTR must sum to σ_p. Euler's theorem, not an approximation.

    This is the single best check that the risk code is correct.
    """

    def test_two_asset(self, two_asset_cov):
        d = risk_decomposition([0.6, 0.4], two_asset_cov)
        assert d.component.sum() == pytest.approx(d.portfolio_volatility, abs=1e-15)
        assert d.check_invariant()

    def test_five_asset_random(self, random_cov):
        rng = np.random.default_rng(3)
        w = rng.random(5)
        w = w / w.sum()
        d = risk_decomposition(w, random_cov)
        assert d.check_invariant(), (d.component.sum(), d.portfolio_volatility)

    @pytest.mark.parametrize("seed", range(25))
    def test_holds_for_arbitrary_portfolios(self, seed):
        """Twenty-five random covariance matrices and weight vectors."""
        rng = np.random.default_rng(seed)
        n = rng.integers(1, 9)
        returns = pd.DataFrame(rng.normal(0, 0.015, (300, n)),
                               columns=[f"X{i}" for i in range(n)])
        cov = covariance_matrix(returns)
        w = rng.random(n)
        w = w / w.sum()
        d = risk_decomposition(w, cov)
        assert d.check_invariant(), f"seed {seed}: {d.component.sum()} != {d.portfolio_volatility}"

    def test_percentages_sum_to_one(self, random_cov):
        d = risk_decomposition(np.full(5, 0.2), random_cov)
        assert d.percent.sum() == pytest.approx(1.0)

    def test_holds_with_a_zero_weight_holding(self, two_asset_cov):
        d = risk_decomposition([1.0, 0.0], two_asset_cov)
        assert d.check_invariant()
        assert d.component[1] == 0.0

    def test_zero_variance_portfolio_does_not_divide_by_zero(self):
        cov = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=[A, B], columns=[A, B])
        d = risk_decomposition([0.5, 0.5], cov)
        assert d.portfolio_volatility == 0.0
        assert d.check_invariant()


class TestRiskVersusCapital:
    def test_risk_share_diverges_from_capital_weight(self, two_asset_cov):
        """The whole point of the risk view: A is 60% of capital but more of the risk."""
        d = risk_decomposition([0.6, 0.4], two_asset_cov)
        frame = d.as_frame()
        assert frame.loc[A, "weight"] == pytest.approx(0.6)
        assert frame.loc[A, "pct_of_risk"] > 0.6
        assert frame.loc[B, "pct_of_risk"] < 0.4

    def test_marginal_contribution_can_be_below_standalone_volatility(self):
        """Why a volatile, weakly-correlated asset can reduce total risk --
        the reason the v2 simulator exists."""
        cov = pd.DataFrame([[0.0004, 0.0], [0.0, 0.0009]], index=[A, B], columns=[A, B])
        d = risk_decomposition([0.9, 0.1], cov)
        sigma_b = np.sqrt(0.0009)                       # 0.03, the more volatile
        assert d.marginal[1] < sigma_b


class TestDiversificationRatio:
    def test_single_asset_is_exactly_one(self):
        cov = pd.DataFrame([[0.0004]], index=[A], columns=[A])
        assert diversification_ratio([1.0], cov) == pytest.approx(1.0)

    def test_rises_with_genuine_diversification(self, two_asset_cov):
        uncorrelated = two_asset_cov.copy()
        uncorrelated.iloc[0, 1] = uncorrelated.iloc[1, 0] = 0.0
        assert diversification_ratio([0.6, 0.4], uncorrelated) > \
               diversification_ratio([0.6, 0.4], two_asset_cov)


class TestConcentration:
    def test_equal_weights_give_the_holding_count(self):
        assert concentration([0.25] * 4).effective_holdings == pytest.approx(4.0)

    def test_the_headline_case(self):
        """Ten holdings behaving like three is the insight worth surfacing."""
        w = [0.30, 0.28, 0.25, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03]
        stats = concentration(w)
        assert stats.actual_holdings == 10
        assert stats.effective_holdings < 4.5
        assert "10 holdings" in stats.summary()

    def test_empty_portfolio(self):
        assert concentration([]).effective_holdings == 0.0


class TestCorrelation:
    def test_diagonal_is_one(self, random_cov):
        assert np.allclose(np.diag(correlation_matrix(random_cov).to_numpy()), 1.0)

    def test_matches_the_covariance_it_came_from(self, two_asset_cov):
        corr = correlation_matrix(two_asset_cov)
        assert corr.loc[A, B] == pytest.approx(0.5)

    def test_flags_a_duplicate_holding(self):
        """Two European defence ETFs holding the same underlyings."""
        cov = pd.DataFrame([[0.0004, 0.00038], [0.00038, 0.0004]],
                           index=[A, B], columns=[A, B])
        pairs = high_correlation_pairs(correlation_matrix(cov))
        assert len(pairs) == 1
        assert {pairs[0].a, pairs[0].b} == {A, B}
        assert pairs[0].correlation >= HIGH_CORRELATION_THRESHOLD

    def test_flag_names_both_funds_in_a_sentence(self):
        cov = pd.DataFrame([[0.0004, 0.00038], [0.00038, 0.0004]],
                           index=[A, B], columns=[A, B])
        pair = high_correlation_pairs(correlation_matrix(cov))[0]
        sentence = pair.sentence({A: "WisdomTree Europe Defence",
                                  B: "iShares Europe Defence"})
        assert "WisdomTree Europe Defence" in sentence
        assert "iShares Europe Defence" in sentence

    def test_no_false_positives(self, two_asset_cov):
        assert high_correlation_pairs(correlation_matrix(two_asset_cov)) == []


class TestDrawdown:
    def test_max_and_current(self):
        """100 -> 120 -> 90 -> 110: max drawdown (90/120 - 1) = -25%,
        current (110/120 - 1) = -8.333%"""
        idx = pd.bdate_range("2026-01-01", periods=4)
        v = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx)
        d = drawdown(v)
        assert d.max_drawdown == pytest.approx(-0.25)
        assert d.current_drawdown == pytest.approx(-1 / 12)
        assert d.max_drawdown_peak == idx[1]
        assert d.max_drawdown_trough == idx[2]

    def test_monotonic_rise_has_no_drawdown(self):
        v = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2026-01-01", periods=3))
        d = drawdown(v)
        assert d.max_drawdown == 0.0
        assert d.current_drawdown == 0.0

    def test_empty_series(self):
        assert drawdown(pd.Series(dtype=float)).max_drawdown == 0.0


class TestBeta:
    def test_identical_series_has_beta_one(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0, 0.01, 200),
                      index=pd.bdate_range("2026-01-01", periods=200))
        assert beta(r, r) == pytest.approx(1.0)

    def test_double_the_benchmark_has_beta_two(self):
        rng = np.random.default_rng(1)
        b = pd.Series(rng.normal(0, 0.01, 200),
                      index=pd.bdate_range("2026-01-01", periods=200))
        assert beta(b * 2, b) == pytest.approx(2.0)

    def test_uses_only_overlapping_dates(self):
        idx = pd.bdate_range("2026-01-01", periods=200)
        rng = np.random.default_rng(2)
        b = pd.Series(rng.normal(0, 0.01, 200), index=idx)
        p = (b * 1.5).iloc[50:]
        assert beta(p, b) == pytest.approx(1.5)

    def test_too_little_overlap_refuses(self):
        a = pd.Series([0.1], index=pd.bdate_range("2026-01-01", periods=1))
        with pytest.raises(ValueError, match="at least 2"):
            beta(a, a)

    def test_zero_variance_benchmark_refuses(self):
        idx = pd.bdate_range("2026-01-01", periods=10)
        flat = pd.Series([0.0] * 10, index=idx)
        with pytest.raises(ValueError, match="zero variance"):
            beta(pd.Series(np.arange(10.0), index=idx), flat)


class TestCovarianceRefusals:
    def test_one_observation_refused(self):
        with pytest.raises(ValueError, match="at least 2"):
            covariance_matrix(pd.DataFrame({A: [0.01]}))

    def test_no_instruments_refused(self):
        with pytest.raises(ValueError, match="at least one instrument"):
            covariance_matrix(pd.DataFrame(index=[0, 1]))
