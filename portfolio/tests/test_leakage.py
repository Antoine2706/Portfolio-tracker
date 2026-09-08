"""The leak detector, tested in all three directions it can be wrong.

A leak detector that has only ever been run against clean code is
indistinguishable from one with a typo in it. So this file asserts three
things, and the second and third matter more than the first:

    1. it stays quiet on a policy that reads only its view
    2. it catches a policy that reads tomorrow's close
    3. it refuses to run at all when wired up so that it could not catch
       anything

The third came out of building it. The first version of this file wired the
leaky policy to close over the *original* panel, so the perturbation never
reached it and the detector reported "clean" on a policy that was openly
reading the future. That is precisely the failure mode the detector exists to
prevent, reproduced inside the detector's own test, and `VacuousLeakCheck`
exists because of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.eval.controls import CoinFlip, Oracle, synthetic_world
from portfolio.eval.harness import Execution, walk_forward
from portfolio.eval.leakage import (VacuousLeakCheck, check_lookahead,
                                    perturb_after)

WARMUP = 252
SPLITS = [500, 520, 611, 700]


@pytest.fixture(scope="module")
def panel():
    return synthetic_world(n_assets=4, periods=800, sigma=0.01, seed=5)[0]


def run_clean(p, every=21):
    """Reads only what the view offers. Built from the panel it is given."""
    return walk_forward(p, CoinFlip(lam=0.3, seed=0), warmup=WARMUP,
                        rebalance_every=every, cost_model=None,
                        execution=Execution.NEXT_CLOSE)


def run_leaky(p, every=21):
    """Reads tomorrow's close. The bug takes exactly this shape in real code:
    a frame from the enclosing scope where a slice was intended."""
    return walk_forward(p, Oracle(p.closes, skill=1.0, seed=0), warmup=WARMUP,
                        rebalance_every=every, cost_model=None,
                        execution=Execution.NEXT_OPEN)


class TestPerturbation:
    def test_the_past_is_untouched_and_the_future_is_not(self, panel):
        altered = perturb_after(panel, 500)
        assert np.array_equal(altered.closes.iloc[:501].to_numpy(),
                              panel.closes.iloc[:501].to_numpy())
        assert not np.allclose(altered.closes.iloc[501:].to_numpy(),
                               panel.closes.iloc[501:].to_numpy())

    def test_the_join_is_continuous(self, panel):
        """A discontinuous jump would be detectable as an outlier return, and
        a policy might then react to the perturbation for the wrong reason."""
        altered = perturb_after(panel, 500)
        step = altered.closes.iloc[501] / altered.closes.iloc[500] - 1.0
        assert float(step.abs().max()) < 0.4

    def test_missing_data_stays_missing(self):
        """Or the perturbed run would differ because the universe changed."""
        p, _ = synthetic_world(n_assets=3, periods=600, seed=2)
        closes = p.closes.copy()
        closes.iloc[450:, 2] = np.nan
        from portfolio.eval.harness import Panel
        altered = perturb_after(Panel(closes=closes), 400)
        assert bool(altered.closes.iloc[450:, 2].isna().all())

    def test_it_refuses_a_split_with_nothing_after_it(self, panel):
        with pytest.raises(ValueError, match="at least one row"):
            perturb_after(panel, len(panel.closes) - 1)


class TestDetection:
    @pytest.mark.parametrize("split", SPLITS)
    def test_a_clean_policy_is_reported_clean(self, panel, split):
        report = check_lookahead(run_clean, panel, split)
        assert not report.leaked, report.detail
        assert report.compared_returns > 0 and report.compared_decisions > 0
        assert bool(report), "LeakReport should be truthy when clean"

    @pytest.mark.parametrize("split", SPLITS)
    def test_a_policy_reading_tomorrow_is_caught(self, panel, split):
        report = check_lookahead(run_leaky, panel, split)
        assert report.leaked, (
            "a policy that decides using tomorrow's close was reported clean; "
            "the detector is not detecting")
        assert "read a price that had not happened yet" in report.detail

    def test_it_is_caught_under_daily_rebalancing_too(self, panel):
        assert check_lookahead(lambda p: run_leaky(p, every=1), panel, 600).leaked
        assert not check_lookahead(lambda p: run_clean(p, every=1), panel, 600).leaked

    def test_the_split_snaps_back_to_a_decision_day(self, panel):
        """Why: a one-bar peek only touches perturbed data if a decision sits
        on the split. Without the snap the detector misses it nineteen times
        in twenty under monthly rebalancing -- which it did, before this."""
        report = check_lookahead(run_leaky, panel, 519)
        decisions = [d.decided_on for d in run_leaky(panel).decisions]
        assert report.split in decisions
        assert report.leaked


class TestTheDetectorCannotPassVacuously:
    def test_a_run_that_ignores_its_panel_is_refused(self, panel):
        def miswired(_):
            return run_clean(panel)          # ignores the argument entirely
        with pytest.raises(VacuousLeakCheck, match="ignoring its argument"):
            check_lookahead(miswired, panel, 500)

    def test_a_policy_built_from_a_stale_panel_is_refused(self, panel):
        """The exact mistake this file was first written with."""
        stale = Oracle(panel.closes, skill=1.0, seed=0)

        def miswired(p):
            return walk_forward(p, stale, warmup=WARMUP, rebalance_every=21,
                                cost_model=None, execution=Execution.NEXT_OPEN)
        # The oracle consults the original frame, so its decisions do not move;
        # the realised returns still do, because those come from the panel the
        # harness was given. So this is not vacuous -- but it is also not a
        # detection, and the report must say clean rather than pretend.
        report = check_lookahead(miswired, panel, 500)
        assert not report.leaked, (
            "a policy holding a stale copy of the prices cannot be caught by "
            "perturbation, and the report should not claim otherwise")

    def test_no_decision_before_the_split_is_refused(self, panel):
        with pytest.raises(ValueError, match="no decision was taken"):
            check_lookahead(run_clean, panel, 10)
