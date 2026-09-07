"""VaR, expected shortfall, worst periods and stress scenarios, by hand.

Where a quantile can be placed exactly it is: 21 observations put the 5%
quantile on the second-lowest value, 101 put the 1% quantile on the second
lowest, and an alternating +10%/-10% series makes every two-day window
compound to exactly -1%.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from portfolio.core.var import (cornish_fisher_quantile, cornish_fisher_var,
                                historical_var, norm_cdf, norm_pdf, norm_ppf,
                                parametric_var, sample_skew_kurtosis, stress_scenarios,
                                var_report, worst_periods)

A, B = "IE0002Y8CX98", "IE000IAXNM41"


def daily(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2026-01-01", periods=len(values)), dtype=float)


class TestNormalQuantile:
    def test_the_two_textbook_values(self):
        assert norm_ppf(0.95) == pytest.approx(1.6448536269514722, abs=1e-12)
        assert norm_ppf(0.99) == pytest.approx(2.3263478740408408, abs=1e-12)
        assert norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-12)

    def test_median_is_zero_and_symmetric(self):
        assert norm_ppf(0.5) == 0.0
        assert norm_ppf(0.05) == pytest.approx(-norm_ppf(0.95), abs=1e-14)

    @pytest.mark.parametrize("p", [1e-9, 0.001, 0.02, 0.02425, 0.3, 0.5, 0.7, 0.97575, 0.999, 1 - 1e-9])
    def test_round_trip_through_the_cdf_in_every_region(self, p):
        """Acklam's three regions and their boundaries, polished to machine precision."""
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-14)

    def test_pdf_at_zero(self):
        assert norm_pdf(0.0) == pytest.approx(1 / math.sqrt(2 * math.pi))

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_is_refused(self, p):
        with pytest.raises(ValueError, match="between 0 and 1"):
            norm_ppf(p)


class TestHistoricalVar:
    def test_95_by_hand(self):
        """21 values: the 5% quantile sits exactly on the 2nd lowest (-0.05).
        ES is the mean of everything at or below it: (-0.10 - 0.05) / 2."""
        r = daily([-0.10, -0.05] + [0.01] * 19)
        v = historical_var(r, 0.95)
        assert v.loss == pytest.approx(0.05)
        assert v.expected_shortfall == pytest.approx(0.075)
        assert v.method == "historical" and v.confidence == 0.95
        assert v.horizon_days == 1 and v.observations == 21

    def test_99_by_hand(self):
        """101 values: the 1% quantile is the 2nd lowest."""
        r = daily([-0.20, -0.08] + [0.0] * 99)
        v = historical_var(r, 0.99)
        assert v.loss == pytest.approx(0.08)
        assert v.expected_shortfall == pytest.approx(0.14)

    def test_quantile_interpolates_between_order_statistics(self):
        """4 values at 95%: position 0.15 of the way from the lowest to the next."""
        r = daily([-0.10, 0.0, 0.0, 0.0])
        assert historical_var(r, 0.95).loss == pytest.approx(0.10 - 0.15 * 0.10)

    def test_two_day_horizon_compounds_overlapping_windows(self):
        """+10%, -10% alternating: every 2-day window is 1.1 x 0.9 - 1 = -1%."""
        r = daily([0.1, -0.1] * 10)
        v = historical_var(r, 0.95, horizon_days=2)
        assert v.loss == pytest.approx(0.01)
        assert v.expected_shortfall == pytest.approx(0.01)
        assert v.observations == 19, "20 returns give 19 overlapping 2-day windows"

    def test_ten_day_horizon_window_count(self):
        v = historical_var(daily(np.linspace(-0.02, 0.02, 60)), 0.95, horizon_days=10)
        assert v.observations == 51

    def test_loss_is_a_positive_fraction(self):
        rng = np.random.default_rng(0)
        v = historical_var(daily(rng.normal(0, 0.01, 252)))
        assert 0 < v.loss < 0.05

    def test_99_is_at_least_95_and_es_is_at_least_var(self):
        rng = np.random.default_rng(1)
        r = daily(rng.standard_t(4, 500) * 0.01)
        v95, v99 = historical_var(r, 0.95), historical_var(r, 0.99)
        assert v99.loss >= v95.loss
        assert v95.expected_shortfall >= v95.loss
        assert v99.expected_shortfall >= v99.loss

    def test_nans_are_dropped(self):
        r = daily([-0.10, float("nan"), -0.05] + [0.01] * 19)
        assert historical_var(r, 0.95).observations == 21

    def test_too_few_observations_refused(self):
        with pytest.raises(ValueError, match="at least"):
            historical_var(daily([0.01]))
        with pytest.raises(ValueError, match="at least 3"):
            historical_var(daily([0.01, 0.02]), horizon_days=2)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
    def test_bad_confidence_refused(self, confidence):
        with pytest.raises(ValueError, match="confidence"):
            historical_var(daily([0.01, 0.02]), confidence)

    def test_bad_horizon_refused(self):
        with pytest.raises(ValueError, match="horizon"):
            historical_var(daily([0.01, 0.02]), horizon_days=0)


class TestParametricVar:
    SYMMETRIC = daily([0.01, -0.01] * 100)          # mean exactly 0

    def test_loss_is_z_times_sigma_when_the_mean_is_zero(self):
        sigma = float(self.SYMMETRIC.std(ddof=1))
        v = parametric_var(self.SYMMETRIC, 0.95)
        assert v.loss == pytest.approx(norm_ppf(0.95) * sigma)
        assert parametric_var(self.SYMMETRIC, 0.99).loss == pytest.approx(norm_ppf(0.99) * sigma)

    def test_expected_shortfall_is_the_normal_tail_mean(self):
        """ES / sigma = phi(z) / (1 - c): 0.10314 / 0.05 = 2.0627 at 95%."""
        sigma = float(self.SYMMETRIC.std(ddof=1))
        v = parametric_var(self.SYMMETRIC, 0.95)
        assert v.expected_shortfall / sigma == pytest.approx(2.0627, abs=1e-4)
        assert v.expected_shortfall > v.loss

    def test_mean_shifts_the_loss(self):
        """loss = z sigma - mu."""
        r = daily([0.03, 0.01] * 50)          # mean 0.02
        sigma = float(r.std(ddof=1))
        assert parametric_var(r, 0.95).loss == pytest.approx(norm_ppf(0.95) * sigma - 0.02)

    def test_ten_day_horizon_scales_by_sqrt_time_and_time(self):
        r = daily([0.03, 0.01] * 50)
        one, ten = parametric_var(r, 0.95, 1), parametric_var(r, 0.95, 10)
        sigma = float(r.std(ddof=1))
        assert ten.loss == pytest.approx(norm_ppf(0.95) * sigma * math.sqrt(10) - 0.02 * 10)
        assert ten.horizon_days == 10 and one.observations == ten.observations == 100

    def test_too_few_observations_refused(self):
        with pytest.raises(ValueError, match="at least 2"):
            parametric_var(daily([0.01]))


class TestCornishFisher:
    def test_quantile_formula_by_hand(self):
        """z = -1.6449, S = -0.5, K = 1:
        (z^2 - 1) S / 6 + (z^3 - 3z) K / 24 - (2z^3 - 5z) S^2 / 36."""
        z = -1.6449
        expected = (z + (z ** 2 - 1) * (-0.5) / 6 + (z ** 3 - 3 * z) * 1.0 / 24
                    - (2 * z ** 3 - 5 * z) * 0.25 / 36)
        assert cornish_fisher_quantile(z, -0.5, 1.0) == pytest.approx(expected)
        assert expected < z, "negative skew and fat tails push the quantile out"

    def test_no_correction_at_zero_moments(self):
        assert cornish_fisher_quantile(-2.0, 0.0, 0.0) == -2.0

    def test_moments_by_hand(self):
        """[1, 2, 3]: m2 = 2/3, m3 = 0, m4 = 2/3 -> S = 0, K = (2/3)/(4/9) - 3 = -1.5."""
        assert sample_skew_kurtosis(np.array([1.0, 2.0, 3.0])) == (0.0, -1.5)

    def test_moments_of_a_constant_are_zero_not_nan(self):
        assert sample_skew_kurtosis(np.array([0.01, 0.01, 0.01])) == (0.0, 0.0)

    def test_skewed_sample_has_negative_skew(self):
        s, _ = sample_skew_kurtosis(np.array([-0.10, 0.01, 0.01, 0.01, 0.01]))
        assert s < 0

    def test_var_matches_the_formula_written_out(self):
        rng = np.random.default_rng(2)
        r = daily(rng.standard_t(4, 300) * 0.01)
        x = r.to_numpy()
        mu, sigma = x.mean(), x.std(ddof=1)
        skew, kurt = sample_skew_kurtosis(x)
        z = cornish_fisher_quantile(norm_ppf(0.05), skew, kurt)
        assert cornish_fisher_var(r, 0.95).loss == pytest.approx(-(mu + sigma * z))

    def test_symmetric_sample_reduces_to_parametric_up_to_kurtosis(self):
        """[+1%, -1%] alternating has S = 0 exactly and K = -2 (two-point
        distribution), so only the kurtosis term separates it from normal."""
        r = daily([0.01, -0.01] * 100)
        sigma = float(r.std(ddof=1))
        z = norm_ppf(0.05)
        expected = -sigma * (z + (z ** 3 - 3 * z) * (-2.0) / 24)
        assert cornish_fisher_var(r, 0.95).loss == pytest.approx(expected)

    def test_fat_left_tail_raises_var_above_parametric(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 0.01, 500)
        x[:10] = -0.06                                   # a crash cluster
        r = daily(x)
        assert cornish_fisher_var(r, 0.95).loss > parametric_var(r, 0.95).loss

    def test_expected_shortfall_closed_form(self):
        """ES = -(mu + sigma E[g(Z) | Z <= z]) with
        E = -phi(z)/alpha [1 + zS/6 - (1 - z^2)K/24 + (1 - 2z^2)S^2/36]."""
        rng = np.random.default_rng(4)
        r = daily(rng.standard_t(5, 250) * 0.01)
        x = r.to_numpy()
        mu, sigma = x.mean(), x.std(ddof=1)
        s, k = sample_skew_kurtosis(x)
        z, alpha = norm_ppf(0.05), 0.05
        tail = -norm_pdf(z) / alpha * (1 + z * s / 6 - (1 - z * z) * k / 24
                                       + (1 - 2 * z * z) * s * s / 36)
        assert cornish_fisher_var(r, 0.95).expected_shortfall == pytest.approx(-(mu + sigma * tail))

    def test_expected_shortfall_equals_parametric_when_moments_vanish(self):
        """A sample with S = 0 and K = 0 exactly, so the correction is nil.

        A symmetric three-point sample, +-a with probability 1/6 each and 0
        otherwise: m2 = a^2/3, m4 = a^4/3, so m4 / m2^2 = 3 and the excess
        kurtosis is 0; symmetry makes the skew 0. (Midpoint normal quantiles
        would not do: their truncated tails give K about -0.06 at n = 401.)
        """
        x = np.array([-0.03] * 50 + [0.0] * 200 + [0.03] * 50)
        s, k = sample_skew_kurtosis(x)
        assert abs(s) < 1e-12 and abs(k) < 1e-12
        r = daily(x)
        cf, normal = cornish_fisher_var(r, 0.95), parametric_var(r, 0.95)
        assert cf.loss == pytest.approx(normal.loss, rel=1e-9)
        assert cf.expected_shortfall == pytest.approx(normal.expected_shortfall, rel=1e-9)

    def test_horizon_scales_the_moments_too(self):
        """Under independence skew shrinks by sqrt(h) and excess kurtosis by h."""
        rng = np.random.default_rng(5)
        r = daily(rng.standard_t(4, 300) * 0.01)
        x = r.to_numpy()
        mu, sigma = x.mean() * 10, x.std(ddof=1) * math.sqrt(10)
        s, k = sample_skew_kurtosis(x)
        z = cornish_fisher_quantile(norm_ppf(0.01), s / math.sqrt(10), k / 10)
        assert cornish_fisher_var(r, 0.99, 10).loss == pytest.approx(-(mu + sigma * z))


class TestVarReport:
    def test_every_method_confidence_and_horizon(self):
        rng = np.random.default_rng(6)
        rows = var_report(daily(rng.normal(0, 0.01, 252)))
        assert len(rows) == 12
        assert {r.method for r in rows} == {"historical", "parametric", "cornish_fisher"}
        assert {r.confidence for r in rows} == {0.95, 0.99}
        assert {r.horizon_days for r in rows} == {1, 10}

    def test_ordering_is_confidence_then_horizon_then_method(self):
        rows = var_report(daily(np.linspace(-0.03, 0.03, 100)))
        assert [(r.confidence, r.horizon_days, r.method) for r in rows[:4]] == [
            (0.95, 1, "historical"), (0.95, 1, "parametric"), (0.95, 1, "cornish_fisher"),
            (0.95, 10, "historical")]

    def test_custom_grid(self):
        rows = var_report(daily(np.linspace(-0.03, 0.03, 100)), confidences=(0.9,), horizons=(1,))
        assert len(rows) == 3 and all(r.confidence == 0.9 for r in rows)


class TestWorstPeriods:
    R = daily([0.01, -0.05, 0.02, -0.03, 0.01, -0.10, 0.02])

    def test_single_days_worst_first(self):
        got = worst_periods(self.R, 1, n=3)
        assert [p.ret for p in got] == [-0.10, -0.05, -0.03]
        assert got[0].start == got[0].end == self.R.index[5].date()
        assert all(p.length_days == 1 for p in got)

    def test_two_day_windows_do_not_overlap(self):
        """Windows: -4.05% (0-1), -3.1% (1-2), -1.06% (2-3), -2.03% (3-4),
        -9.1% (4-5), -8.2% (5-6). Worst is 4-5; 5-6 and 3-4 overlap it; 0-1
        is next; 1-2 overlaps that; 2-3 stands alone."""
        got = worst_periods(self.R, 2, n=5)
        assert [round(p.ret, 6) for p in got] == [round(1.01 * 0.90 - 1, 6),
                                                   round(1.01 * 0.95 - 1, 6),
                                                   round(1.02 * 0.97 - 1, 6)]
        assert (got[0].start, got[0].end) == (self.R.index[4].date(), self.R.index[5].date())

    def test_fewer_than_n_when_the_series_is_short(self):
        assert len(worst_periods(self.R, 1, n=20)) == 7

    def test_window_longer_than_the_series_is_empty(self):
        assert worst_periods(self.R, 8) == []

    def test_zero_n_is_empty(self):
        assert worst_periods(self.R, 1, n=0) == []

    def test_bad_length_refused(self):
        with pytest.raises(ValueError, match="length_days"):
            worst_periods(self.R, 0)


class TestStressScenarios:
    FRAME = pd.DataFrame({A: [0.01, -0.05, 0.02, -0.02], B: [0.00, -0.01, 0.03, -0.04]},
                         index=pd.bdate_range("2026-01-01", periods=4))
    W = {A: 0.6, B: 0.4}

    def _by_key(self, frame=None):
        return {s.key: s for s in stress_scenarios(self.FRAME if frame is None else frame, self.W)}

    def test_worst_day_is_the_portfolios_worst_day(self):
        """Portfolio: 0.006, -0.034, 0.024, -0.028 -> day 2, where A -5%, B -1%."""
        s = self._by_key()["worst_day"]
        assert s.portfolio_return == pytest.approx(-0.034)
        assert s.per_holding == {A: -0.05, B: -0.01}
        assert "2026-01-02" in s.description

    def test_own_worst_day(self):
        """0.6 x -0.05 + 0.4 x -0.04 = -0.046."""
        s = self._by_key()["own_worst_day"]
        assert s.per_holding == {A: -0.05, B: -0.04}
        assert s.portfolio_return == pytest.approx(-0.046)

    def test_all_minus_ten(self):
        s = self._by_key()["all_minus_10"]
        assert s.portfolio_return == pytest.approx(-0.10)
        assert s.per_holding == {A: -0.10, B: -0.10}

    def test_largest_holding_minus_25_alone(self):
        s = self._by_key()["largest_minus_25"]
        assert s.per_holding == {A: -0.25, B: 0.0}
        assert s.portfolio_return == pytest.approx(-0.15)
        assert A in s.description

    def test_correlation_goes_to_one(self):
        """Each holding falls z_99 x its own daily sigma; the portfolio is the
        weighted sum, and the description quotes the diversified figure."""
        s = self._by_key()["correlation_one"]
        z = norm_ppf(0.99)
        sa, sb = float(self.FRAME[A].std(ddof=1)), float(self.FRAME[B].std(ddof=1))
        assert s.per_holding[A] == pytest.approx(-z * sa)
        assert s.per_holding[B] == pytest.approx(-z * sb)
        assert s.portfolio_return == pytest.approx(-0.6 * z * sa - 0.4 * z * sb)
        assert "diversification" in s.description

    def test_short_window_skips_multi_day_scenarios(self):
        keys = list(self._by_key())
        assert "worst_5d" not in keys and "worst_21d" not in keys
        assert keys == ["worst_day", "own_worst_day", "all_minus_10", "largest_minus_25",
                        "correlation_one"]

    def test_multi_day_windows_compound(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(0, 0.01, (40, 2)), columns=[A, B],
                             index=pd.bdate_range("2026-01-01", periods=40))
        by_key = self._by_key(frame)
        s = by_key["worst_21d"]
        p = frame[A] * 0.6 + frame[B] * 0.4
        windows = [float(np.prod(1 + p.iloc[i:i + 21]) - 1) for i in range(20)]
        assert s.portfolio_return == pytest.approx(min(windows))
        i = int(np.argmin(windows))
        assert s.per_holding[A] == pytest.approx(float(np.prod(1 + frame[A].iloc[i:i + 21]) - 1))
        assert "worst_5d" in by_key

    def test_missing_weight_is_refused(self):
        with pytest.raises(ValueError, match="no weight supplied"):
            stress_scenarios(self.FRAME, {A: 1.0})

    def test_every_scenario_names_every_holding(self):
        for s in stress_scenarios(self.FRAME, self.W):
            assert set(s.per_holding) == {A, B}
            assert s.label and s.description
