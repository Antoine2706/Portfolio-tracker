"""Testing the raw active return is a tautology for a de-risking policy.

The defect
----------
Equal risk contribution's function is to hold less of the volatile assets. Run
it through a rising year and it underperforms by construction: a book carrying
14% less risk gives up 14% of the benchmark's return before any question of
selection arises. A t-statistic on the raw active return then reports, with
confidence, that a de-risking policy de-risked -- and the identical policy in
a falling year comes out significantly *better*. The sign belongs to the
window, not to the strategy.

That is a confidently correct number measuring something other than what it
claims, which is the defect this project keeps finding. It had just been
introduced into the verdict, in the fix for the previous one.

`TestTheWindowDecidesTheSign` is the demonstration. `Deleverage` holds a fixed
fraction of the benchmark's book and the rest in cash: no selection at all,
zero skill by construction, and a Sharpe ratio identical to the benchmark's up
to rebalancing drift. The raw test calls it significant in both directions.
The risk-adjusted test calls it nothing, twice, which is the truth.

The quantity that generalises
-----------------------------
With the risk-free rate at zero an annualised arithmetic return is Sharpe
times volatility, so the raw gap splits exactly:

    R_p - R_b = (SR_p - SR_b) x sigma_b  +  SR_p x (sigma_p - sigma_b)
                \\____ selection ____/       \\_____ mandate _____/

and the difference of Sharpe ratios is tested with Jobson-Korkie plus
Memmel's correction, at a correlation this file insists is measured.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.base import Proposal
from portfolio.eval.harness import Execution, Hold, Panel, walk_forward
from portfolio.eval.metrics import (SUSPICIOUS_ACTIVE_SHARPE,
                                    sharpe_difference_standard_error)
from portfolio.eval.report import DECISIVE_T, compare


class Deleverage:
    """Holds `k` of the benchmark's book and the rest in cash.

    A policy with no view on anything: it cannot select, time, or tilt. Its
    return is k times the benchmark's, so its Sharpe ratio is the benchmark's
    and its volatility is k times it. Any statistic that calls this policy
    better or worse than doing nothing is measuring the window.
    """
    name = "de-lever"

    def __init__(self, k: float):
        self.k = k

    def observe(self, view) -> Proposal:
        held = {i: w for i, w in view.held.items() if w > 0}
        total = sum(held.values()) or 1.0
        return Proposal({i: self.k * w / total for i, w in held.items()},
                        0.5, f"hold {self.k:.0%} of the book, rest in cash",
                        self.name)


def prices(drift: float, seed: int = 5, periods: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=periods)
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(rng.normal(drift, 0.009, (periods, 3)),
                        index=dates, columns=["A", "B", "C"])
    rets.iloc[0] = 0.0
    return 100.0 * (1.0 + rets).cumprod()


def run(frame: pd.DataFrame, policy=None):
    start = {c: 1.0 / len(frame.columns) for c in frame.columns}
    if policy is None:
        return walk_forward(Panel(closes=frame), Hold(), warmup=60,
                            rebalance_every=len(frame) + 1, cost_model=None,
                            execution=Execution.NEXT_CLOSE,
                            initial_weights=start,
                            frozen=frozenset(frame.columns))
    return walk_forward(Panel(closes=frame), policy, warmup=60,
                        rebalance_every=21, cost_model=None,
                        execution=Execution.NEXT_CLOSE, initial_weights=start)


def compared(drift: float, k: float = 0.7):
    frame = prices(drift)
    return compare(run(frame, Deleverage(k)), run(frame), overlap=21)


class TestTheWindowDecidesTheSign:
    """The whole argument, in two runs of the same policy."""

    @pytest.fixture(scope="class")
    def rising(self):
        return compared(drift=0.0012)

    @pytest.fixture(scope="class")
    def falling(self):
        return compared(drift=-0.0012)

    def test_the_fixture_really_is_a_pure_de_risking_policy(self, rising):
        """No selection: it holds a scaled copy of the benchmark's book."""
        ra = rising.risk_adjusted
        assert ra.correlation > 0.99, (
            "the policy is not tracking the benchmark closely enough to be a "
            "pure de-levering, so this file is testing something else")
        assert ra.volatility_policy < 0.85 * ra.volatility_benchmark

    def test_the_raw_test_calls_it_significantly_worse_when_the_book_rises(
            self, rising):
        assert rising.active.t_statistic < -DECISIVE_T

    def test_and_significantly_better_when_the_book_falls(self, falling):
        assert falling.active.t_statistic > DECISIVE_T

    def test_which_is_the_same_policy_twice(self, rising, falling):
        """The two verdicts are opposite and neither is about the strategy."""
        assert rising.active.t_statistic * falling.active.t_statistic < 0

    def test_the_risk_adjusted_test_says_nothing_in_either_window(
            self, rising, falling):
        """Correctly. A scaled copy of the benchmark has the benchmark's
        return per unit of risk, and no sample can show otherwise."""
        for result in (rising, falling):
            assert abs(result.risk_adjusted.t_statistic) < DECISIVE_T
            assert not result.risk_adjusted.decisive

    def test_the_report_labels_the_raw_figure_as_window_dependent(self, rising):
        text = " ".join(rising.paired_lines())
        assert "realised cost of the mandate in THIS window" in text
        assert "sign is the window's direction" in text
        assert "Not tested" in text

    def test_the_verdict_applies_the_two_sigma_rule_to_the_risk_adjusted_one(
            self, rising):
        text = " ".join(rising.paired_lines())
        assert "return per unit of risk is INDISTINGUISHABLE" in text
        assert "carries out of sample" in text


class TestTheAlarmMovedToo:
    """The implausibility threshold had the same fault as the verdict."""

    def test_a_de_risking_policy_in_a_falling_market_no_longer_trips_it(self):
        """It would have: raw information ratio above 3 on a policy that does
        nothing but hold cash. That is arithmetic, not a leak."""
        result = compared(drift=-0.0012)
        assert result.active.sharpe > SUSPICIOUS_ACTIVE_SHARPE, (
            "the fixture's raw information ratio is no longer large enough to "
            "have tripped the old alarm, so this proves nothing")
        assert result.risk_adjusted.difference < SUSPICIOUS_ACTIVE_SHARPE
        assert "Find the leak" not in result.verdict()

    def test_but_a_real_leak_still_does(self):
        """The alarm must not have been disarmed, only re-aimed."""
        from portfolio.eval.controls import Oracle, synthetic_world
        panel, _ = synthetic_world(n_assets=4, periods=400, sigma=0.01, seed=5)
        start = {c: 0.25 for c in panel.closes.columns}
        peeking = walk_forward(panel, Oracle(panel.closes, skill=1.0, seed=0),
                               warmup=60, rebalance_every=1, cost_model=None,
                               execution=Execution.NEXT_OPEN,
                               initial_weights=start)
        benchmark = walk_forward(panel, Hold(), warmup=60,
                                 rebalance_every=len(panel) + 1,
                                 cost_model=None, execution=Execution.NEXT_OPEN,
                                 initial_weights=start,
                                 frozen=frozenset(panel.closes.columns))
        result = compare(peeking, benchmark, overlap=1)
        assert result.risk_adjusted.difference > SUSPICIOUS_ACTIVE_SHARPE
        assert "Find the leak" in result.verdict()
        assert "neither can holding more or less risk" in result.verdict()


class TestTheDecomposition:
    def test_the_two_terms_sum_to_the_raw_gap(self):
        """An identity, so it must hold to floating point, not approximately."""
        for drift in (0.0012, -0.0012, 0.0):
            ra = compared(drift).risk_adjusted
            assert ra.selection + ra.mandate == pytest.approx(ra.raw_gap,
                                                              abs=1e-12)

    def test_a_pure_de_risking_policy_puts_almost_everything_in_mandate(self):
        """Which is the point: four fifths of the gap is the job, not the
        selection, and a reader seeing only the total cannot tell."""
        ra = compared(drift=0.0012).risk_adjusted
        assert abs(ra.mandate) > 3 * abs(ra.selection)

    def test_the_split_is_reported_with_both_numbers_and_the_caveat(self):
        text = " ".join(compared(drift=0.0012).paired_lines())
        assert "is selection" in text and "is mandate" in text
        assert "which a cash account cannot do" in text
        assert "the raw gap is what the account lost" in text

    def test_matched_risk_uses_the_benchmark_volatility(self):
        ra = compared(drift=0.0012).risk_adjusted
        assert ra.selection == pytest.approx(
            ra.difference * ra.volatility_benchmark, rel=1e-12)


class TestTheCorrelationIsMeasured:
    def test_it_is_reported_beside_the_standard_error(self):
        text = " ".join(compared(drift=0.0012).paired_lines())
        assert "measured correlation" in text

    def test_it_matches_the_two_series(self):
        frame = prices(0.0012)
        policy, benchmark = run(frame, Deleverage(0.7)), run(frame)
        dates = policy.returns.index.intersection(benchmark.returns.index)
        clean = (policy.measured.loc[dates].to_numpy(dtype=bool)
                 & benchmark.measured.loc[dates].to_numpy(dtype=bool))
        expected = float(np.corrcoef(policy.returns.loc[dates].to_numpy()[clean],
                                     benchmark.returns.loc[dates].to_numpy()[clean])[0, 1])
        assert compare(policy, benchmark, overlap=21).risk_adjusted.correlation \
            == pytest.approx(expected, rel=1e-12)

    def test_the_standard_error_depends_on_it_strongly(self):
        """Which is why assuming a value would not be a small approximation."""
        paired = sharpe_difference_standard_error(0.1, 0.11, 0.99, 253)
        loose = sharpe_difference_standard_error(0.1, 0.11, 0.90, 253)
        assert loose > 3 * paired


class TestTheMemmelFormula:
    def test_it_predicts_the_spread_of_the_estimated_difference(self):
        """Simulation, because a standard error is a claim about a
        distribution. 20,000 draws per case."""
        rng = np.random.default_rng(7)
        T = 256
        for rho, mu_a, mu_b in ((0.99, 0.0004, 0.00045), (0.90, 0.0004, 0.0004),
                                (0.50, 0.0003, 0.0006)):
            cov = np.array([[1.0, rho], [rho, 1.0]]) * 0.01 ** 2
            draws = rng.multivariate_normal([mu_a, mu_b], cov, size=(20000, T))
            ratios = draws.mean(axis=1) / draws.std(axis=1, ddof=1)
            empirical = float(np.std(ratios[:, 0] - ratios[:, 1], ddof=1))
            predicted = sharpe_difference_standard_error(
                mu_a / 0.01, mu_b / 0.01, rho, T)
            assert empirical / predicted == pytest.approx(1.0, abs=0.03), (
                f"at rho {rho} the formula gives {predicted:.6f} against an "
                f"empirical spread of {empirical:.6f}")

    def test_identical_perfectly_correlated_series_have_nothing_to_test(self):
        assert sharpe_difference_standard_error(0.1, 0.1, 1.0, 253) == 0.0

    def test_pairing_is_what_makes_the_test_possible(self):
        """At the correlation two legs of one book actually run at, the paired
        error is a fraction of either marginal one."""
        from portfolio.eval.metrics import sharpe_standard_error
        paired = sharpe_difference_standard_error(0.1, 0.11, 0.99, 253)
        marginal = sharpe_standard_error(0.1, 253)
        assert paired < 0.2 * marginal

    def test_frequency_consistency_is_not_optional(self):
        """Annualised ratios with T in years is the natural thing to write and
        gives an answer about twice too large, because an asymptotic variance
        at T = 1 approximates nothing."""
        daily_a, daily_b, rho, days = 0.101403, 0.106361, 0.99, 256
        consistent = ((daily_b - daily_a)
                      / sharpe_difference_standard_error(daily_a, daily_b, rho, days))
        annual = math.sqrt(252)
        at_one_year = ((daily_b - daily_a) * annual
                       / sharpe_difference_standard_error(daily_a * annual,
                                                          daily_b * annual, rho, 2))
        assert consistent > 1.5 * abs(at_one_year), (
            "mixing annualised ratios with a yearly count no longer changes "
            "the answer, so the warning in the docstring is stale")

    def test_a_correlation_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError, match="correlation must be"):
            sharpe_difference_standard_error(0.1, 0.1, 1.5, 100)
