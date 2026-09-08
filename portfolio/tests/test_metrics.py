"""Track-record statistics: the properties the doctests cannot state.

The formulae themselves are doctested against values computed independently
of the implementation. What is here is the behaviour around them -- the
monotonicity, the refusals, and above all the rule that the harness must not
report what the sample cannot support.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from portfolio.eval.metrics import (SUSPICIOUS_ANNUAL_SHARPE, annualise_sharpe,
                                    deannualise_sharpe, deflated_sharpe_ratio,
                                    deflation_threshold, effective_observations,
                                    expected_maximum_sharpe,
                                    minimum_track_record_length,
                                    probabilistic_sharpe_ratio,
                                    sharpe_standard_error,
                                    sharpe_variance_factor, track_record,
                                    turnover_series, years_to_detect)


def series(mean: float, sd: float, n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, n),
                     index=pd.bdate_range("2020-01-01", periods=n))


class TestConventions:
    def test_annualisation_round_trips(self):
        assert deannualise_sharpe(annualise_sharpe(0.05)) == pytest.approx(0.05)

    def test_the_normal_case_collapses_to_the_textbook_factor(self):
        for sr in (0.0, 0.05, 0.2, 1.0):
            assert sharpe_variance_factor(sr) == pytest.approx(1 + sr * sr / 2)

    def test_excess_kurtosis_is_the_convention(self):
        """A normal has excess kurtosis 0. Passing 3 by mistake would inflate
        the factor by 3/4 SR^2, and this is the assertion that catches it."""
        assert sharpe_variance_factor(0.5, 0.0, 0.0) == pytest.approx(1.125)
        assert sharpe_variance_factor(0.5, 0.0, 3.0) == pytest.approx(1.3125)

    def test_fat_tails_and_negative_skew_widen_the_error_bars(self):
        base = sharpe_standard_error(0.06, 252)
        assert sharpe_standard_error(0.06, 252, excess_kurtosis=4.0) > base
        assert sharpe_standard_error(0.06, 252, skewness=-0.8) > base
        assert sharpe_standard_error(0.06, 252, skewness=0.8) < base

    def test_more_observations_shrink_the_error(self):
        assert (sharpe_standard_error(0.06, 1008)
                == pytest.approx(sharpe_standard_error(0.06, 252) / 2, rel=0.01))

    def test_the_detection_table_matches_the_specification(self):
        assert [round(years_to_detect(s), 3) for s in (0.5, 1.0, 2.0)] == [16.0, 4.0, 1.0]
        assert years_to_detect(0.0) == math.inf


class TestSelectionBias:
    def test_the_threshold_grows_with_the_number_of_trials(self):
        got = [expected_maximum_sharpe(n, 0.02) for n in (1, 2, 5, 20, 100, 1000)]
        assert got == sorted(got)
        assert got[0] == 0.0

    def test_one_trial_has_nothing_to_deflate(self):
        psr = probabilistic_sharpe_ratio(0.06, 252)
        assert deflated_sharpe_ratio(0.06, 252, 1, 0.02) == pytest.approx(psr)

    def test_deflation_only_ever_lowers_the_figure(self):
        psr = probabilistic_sharpe_ratio(0.06, 252)
        for n in (2, 10, 50, 500):
            assert deflated_sharpe_ratio(0.06, 252, n, 0.02) <= psr

    def test_a_missing_trial_spread_does_not_silently_undeflate(self):
        """The most dangerous default in the module: a zero spread with many
        trials would set the threshold to zero and print the most optimistic
        number available exactly when the evidence is weakest."""
        threshold, fell_back = deflation_threshold(252, 20, 0.0)
        assert fell_back and threshold > 0
        assert deflated_sharpe_ratio(0.06, 252, 20, 0.0) < \
            probabilistic_sharpe_ratio(0.06, 252)

    def test_a_recorded_spread_is_used_as_given(self):
        assert deflation_threshold(252, 20, 0.02) == (
            pytest.approx(expected_maximum_sharpe(20, 0.02)), False)


class TestRefusingToOverstate:
    def test_minimum_track_record_is_undefined_below_the_benchmark(self):
        assert minimum_track_record_length(0.0) is None
        assert minimum_track_record_length(0.05, benchmark=0.06) is None

    def test_a_smaller_edge_needs_a_longer_record(self):
        needs = [minimum_track_record_length(sr) for sr in (0.02, 0.05, 0.1)]
        assert needs == sorted(needs, reverse=True)

    def test_a_noise_series_is_reported_as_unsupported(self):
        tr = track_record(series(0.0002, 0.01, 252, seed=3))
        assert not tr.supported
        assert "cannot establish" in tr.verdict() or "nothing to establish" in tr.verdict()

    def test_a_strong_long_record_is_supported(self):
        # An annualised Sharpe near 1.0 over ten years: strong enough to
        # establish, and below the threshold at which it should be disbelieved.
        tr = track_record(series(0.00063, 0.01, 2520, seed=4))
        assert tr.supported and not tr.suspicious
        assert "deflated Sharpe" in tr.verdict()

    def test_an_absurd_sharpe_is_called_a_bug_not_a_discovery(self):
        tr = track_record(series(0.004, 0.005, 504, seed=5))
        assert tr.sharpe > SUSPICIOUS_ANNUAL_SHARPE and tr.suspicious
        assert "assumed broken" in tr.verdict()

    def test_two_observations_produce_no_ratio_at_all(self):
        tr = track_record(pd.Series([0.01], index=pd.bdate_range("2020-01-01", periods=1)))
        assert tr.sharpe is None and not tr.supported
        assert "too few" in tr.verdict()


class TestOverlap:
    def test_overlapping_windows_shrink_the_independent_count(self):
        assert effective_observations(252, 1) == 252
        assert effective_observations(252, 21) == 12
        assert effective_observations(10, 21) == 1

    def test_the_t_statistic_falls_when_the_overlap_is_declared(self):
        r = series(0.0006, 0.01, 1008, seed=6)
        plain = track_record(r)
        overlapped = track_record(r, overlap=21)
        assert overlapped.independent_observations < plain.independent_observations
        assert abs(overlapped.t_statistic) < abs(plain.t_statistic), (
            "declaring that observations overlap must weaken the claim, not "
            "leave it unchanged")
        assert plain.sharpe == pytest.approx(overlapped.sharpe), (
            "the point estimate is unchanged; only its uncertainty moves")


class TestTurnover:
    def test_drift_is_what_a_rebalance_undoes(self):
        targets = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]})
        drifted = pd.DataFrame({"a": [0.5, 0.62], "b": [0.5, 0.38]})
        assert list(turnover_series(targets, drifted).round(10)) == [0.0, 0.12]

    def test_mismatched_columns_are_refused(self):
        with pytest.raises(ValueError, match="share their columns"):
            turnover_series(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"b": [1.0]}))

    def test_cost_drag_is_annualised(self):
        tr = track_record(series(0.0002, 0.01, 504, seed=7), costs=0.04)
        assert tr.cost_drag == pytest.approx(0.02, rel=1e-6)
