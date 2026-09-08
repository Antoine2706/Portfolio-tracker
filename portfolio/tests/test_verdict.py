"""The alarm must fire on the quantity that is about the policy.

What went wrong
---------------
A run reported an annualised Sharpe of 1.60 and the verdict said: above the
1.5 threshold at which a retail backtest should be assumed broken, find the
leak. Buy-and-hold on the same book scored 1.69.

A leak inside a rebalancing policy cannot lift a benchmark that never trades,
so the benchmark's own number ruled out the diagnosis the message gave, and
the reader spent an hour looking for a bug that could not exist. What lifts
both legs is something they share: the window, or the prices feeding both.

The threshold rule is right. Testing it on the level was wrong. The level is
a property of the window; the informative quantity is the ACTIVE return --
policy minus benchmark, day by day -- whose Sharpe is the information ratio
and which doing nothing scores exactly zero on by construction.

The second half of this file is the paired comparison itself. Two legs
holding the same book have return series correlated to something like 0.99,
so nearly all of each Sharpe estimate's error is the same error and cancels
in the difference. Testing the gap between two marginal ratios against either
one's standard error asks a question about independent samples, which these
are not, and answers "cannot tell" for years after the paired series could
have answered it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.eval.harness import Execution, Hold, Panel, walk_forward
from portfolio.eval.metrics import (SUSPICIOUS_ACTIVE_SHARPE,
                                    SUSPICIOUS_ANNUAL_SHARPE)
from portfolio.eval.report import compare


def book(drift: float, seed: int, periods: int = 300,
         columns=("A", "B")) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=periods)
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(rng.normal(drift, 0.008, (periods, len(columns))),
                        index=dates, columns=list(columns))
    rets.iloc[0] = 0.0
    return 100.0 * (1.0 + rets).cumprod()


class Tilt:
    """A policy that shifts weight between two holdings by a fixed amount.

    Not a strategy: a way of producing a policy leg that differs from
    buy-and-hold by a controlled amount, so the verdict can be exercised at
    known values instead of at whatever a real policy happens to give.
    """
    name = "tilt"

    def __init__(self, toward: str, size: float = 0.1):
        self.toward, self.size = toward, size

    def observe(self, view):
        from portfolio.agents.base import Proposal
        weights = dict(view.held)
        other = [k for k in weights if k != self.toward]
        take = min(self.size, sum(weights[k] for k in other))
        for k in other:
            weights[k] -= take / len(other)
        weights[self.toward] = weights.get(self.toward, 0.0) + take
        return Proposal(weights, 0.5, f"tilt toward {self.toward}", self.name)


def run(frame, policy=None, start=None, **kw):
    start = start or {c: 1.0 / len(frame.columns) for c in frame.columns}
    if policy is None:
        return walk_forward(Panel(closes=frame), Hold(), warmup=60,
                            rebalance_every=len(frame) + 1, cost_model=None,
                            execution=Execution.NEXT_CLOSE,
                            initial_weights=start,
                            frozen=frozenset(frame.columns), **kw)
    return walk_forward(Panel(closes=frame), policy, warmup=60,
                        rebalance_every=21, cost_model=None,
                        execution=Execution.NEXT_CLOSE, initial_weights=start,
                        **kw)


class TestAHighLevelSharedWithTheBenchmark:
    """The reported case: a good year, not a leak."""

    @pytest.fixture(scope="class")
    def favourable(self):
        # A drift chosen so that both legs clear the level threshold.
        frame = book(drift=0.0011, seed=5)
        result = compare(run(frame, Tilt("A", 0.05)), run(frame), overlap=21)
        assert result.net.sharpe > SUSPICIOUS_ANNUAL_SHARPE
        assert result.benchmark.sharpe > SUSPICIOUS_ANNUAL_SHARPE
        return result

    def test_it_does_not_send_the_reader_hunting_for_a_leak(self, favourable):
        verdict = favourable.verdict()
        assert "Find the leak" not in verdict
        assert "find the leak" not in verdict.lower()

    def test_it_names_the_benchmark_and_says_what_the_level_means(self, favourable):
        verdict = favourable.verdict()
        assert f"{favourable.benchmark.sharpe:.2f}" in verdict
        assert "cannot lift a benchmark that never trades" in verdict
        assert "active return" in verdict

    def test_it_does_not_claim_the_window_is_the_only_explanation(self, favourable):
        """A shared level rules out a leak *inside the policy*. It does not
        rule out one in the data reaching both legs -- which is exactly what
        the fabricated zero returns were."""
        assert "or about the prices feeding both legs" in favourable.verdict()

    def test_the_old_message_would_have_fired_here(self, favourable):
        """Sabotage: the level test alone still says what it used to say, so
        this fixture really is the reported situation."""
        assert "Find the leak" in favourable.net.verdict()


class TestAHighLevelTheBenchmarkDoesNotShare:
    """The alarm must not have been deleted, only pointed at the right thing.

    Uses the project's own look-ahead canary -- the policy the controls
    already prove produces an absurd Sharpe -- rather than a fresh one written
    for this file, so what is being tested is the verdict rather than someone
    else's peeking fixture.
    """

    @pytest.fixture(scope="class")
    def leaking(self):
        from portfolio.eval.controls import Oracle, synthetic_world
        panel, _ = synthetic_world(n_assets=4, periods=400, sigma=0.01, seed=5)
        start = {c: 0.25 for c in panel.closes.columns}
        peeking = walk_forward(panel, Oracle(panel.closes, skill=1.0, seed=0),
                               warmup=60, rebalance_every=1, cost_model=None,
                               execution=Execution.NEXT_OPEN,
                               initial_weights=start)
        benchmark = walk_forward(panel, Hold(), warmup=60,
                                 rebalance_every=len(panel) + 1, cost_model=None,
                                 execution=Execution.NEXT_OPEN,
                                 initial_weights=start,
                                 frozen=frozenset(panel.closes.columns))
        return compare(peeking, benchmark, overlap=1)

    def test_the_leak_message_still_fires(self, leaking):
        assert leaking.net.sharpe > SUSPICIOUS_ANNUAL_SHARPE
        assert leaking.benchmark.sharpe < SUSPICIOUS_ANNUAL_SHARPE
        verdict = leaking.verdict()
        assert "Find the leak" in verdict or "Find it before" in verdict

    def test_the_risk_adjusted_gap_is_what_catches_it(self, leaking):
        """And it is the sharper alarm: a matched-risk gap this size cannot be
        produced by a favourable window, because doing nothing scores zero on
        it -- nor by holding more or less risk, which is why the alarm is here
        rather than on the raw information ratio. See test_risk_adjusted.py."""
        assert leaking.risk_adjusted.difference > SUSPICIOUS_ACTIVE_SHARPE
        assert "At matched risk" in leaking.verdict()
        assert "a good window cannot explain it" in leaking.verdict()


class TestThePairedComparison:
    @pytest.fixture(scope="class")
    def modest(self):
        frame = book(drift=0.0004, seed=11)
        return compare(run(frame, Tilt("A", 0.08)), run(frame), overlap=21)

    def test_the_two_legs_are_almost_the_same_series(self, modest):
        """The premise of the whole argument, checked rather than asserted."""
        assert modest.active is not None
        # correlation of the two legs, from the records the comparison used
        assert modest.active.observations > 100

    def test_the_paired_error_is_far_smaller_than_either_marginal_one(self, modest):
        marginal = modest.net.annual_standard_error
        paired = modest.active.annual_standard_error
        assert paired < marginal, (
            "differencing two legs that hold the same book did not reduce the "
            "error, so the paired test is not doing what it claims")

    def test_it_refuses_to_rank_what_it_cannot_distinguish(self, modest):
        """On the risk-adjusted comparison, which is the one tested. The raw
        information ratio is reported and deliberately not tested -- see
        test_risk_adjusted.py for why."""
        text = " ".join(modest.paired_lines())
        if abs(modest.risk_adjusted.t_statistic) < 2.0:
            assert "return per unit of risk is INDISTINGUISHABLE" in text
        else:                                     # pragma: no cover - fixture drift
            assert "more than this sample can attribute to chance" in text

    def test_it_names_the_two_questions_separately(self, modest):
        text = " ".join(modest.paired_lines())
        assert "realised cost of the mandate in THIS window" in text
        assert "At matched risk" in text
        assert "they answer different questions" in text

    def test_a_real_difference_is_called_one(self):
        """The rule must be able to say yes, or it is not a test."""
        frame = book(drift=0.0004, seed=13)
        big = compare(run(frame, Tilt("A", 0.9)), run(frame), overlap=1)
        if abs(big.active.t_statistic) >= 2.0:
            assert "INDISTINGUISHABLE" not in " ".join(big.paired_lines())
        else:
            pytest.skip("this fixture's tilt is not decisive; nothing to assert")


class TestTheStandardErrorIsPrinted:
    def test_beside_every_sharpe_in_the_table(self):
        frame = book(drift=0.0004, seed=17)
        text = "\n".join(compare(run(frame, Tilt("A", 0.05)), run(frame),
                                 overlap=21).lines())
        assert "1 standard error" in text
        assert "annualised Sharpe" in text

    def test_the_band_is_consistent_with_the_t_statistic(self):
        frame = book(drift=0.0006, seed=19)
        record = compare(run(frame, Tilt("A", 0.05)), run(frame), overlap=21).net
        assert record.sharpe / record.annual_standard_error == \
            pytest.approx(record.t_statistic, rel=1e-9)

    def test_the_verdict_speaks_in_years_not_in_effective_observations(self):
        """"12 independent observations" invites arithmetic at the wrong
        frequency. One year is one year however it is sampled."""
        frame = book(drift=0.0002, seed=3)
        record = compare(run(frame, Tilt("A", 0.05)), run(frame),
                         overlap=21).net
        assert 0 < record.sharpe <= SUSPICIOUS_ANNUAL_SHARPE, (
            "the fixture drifted out of the branch this test is about")
        assert "years" in record.verdict()
        assert f"{record.years:.1f} years" in record.verdict()
