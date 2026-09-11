"""The three controls, as tests.

These are fast versions, sized to run in a suite. The full calibration run --
200 seeds for the negative control, a skill sweep for the positive one -- is
`portfolio controls`, and its output is what should be read before trusting
any strategy result.

What each test is really asserting:

    negative   the harness rejects a true null at close to the nominal rate,
               and the t-statistics it produces have standard deviation 1.
               The second is the sharper check: if the standard error formula
               were wrong by a factor, the spread of t would be wrong by that
               factor while the rejection rate might still look plausible.

    positive   the measured Sharpe matches the injected one, at several
               skill levels, within sampling error.

    canary     perfect foresight produces a Sharpe two orders of magnitude
               above anything real, and is flagged as suspicious.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.eval.controls import (CoinFlip, Oracle, calibrate_turnover,
                                     max_moments, oracle_sharpe,
                                     synthetic_world)
from portfolio.eval.harness import Execution, walk_forward
from portfolio.eval.metrics import SUSPICIOUS_ANNUAL_SHARPE

WARMUP = 126
PERIODS = 700


def run_oracle(seed: int, skill: float, n_assets: int = 2):
    panel, _ = synthetic_world(n_assets=n_assets, periods=PERIODS, sigma=0.01,
                               seed=seed)
    return walk_forward(panel, Oracle(panel.closes, skill=skill, seed=seed),
                        warmup=WARMUP, rebalance_every=1, cost_model=None,
                        execution=Execution.NEXT_OPEN).track()


class TestTheClosedForms:
    def test_the_maximum_agrees_with_its_two_known_values(self):
        assert max_moments(2)[0] == pytest.approx(1 / np.sqrt(np.pi), abs=1e-9)
        assert max_moments(2)[1] == pytest.approx(1.0, abs=1e-9)
        assert max_moments(1) == pytest.approx((0.0, 1.0), abs=1e-9)

    def test_the_quadrature_agrees_with_simulation(self):
        """Independent check on the number the positive control is judged by."""
        rng = np.random.default_rng(0)
        draws = rng.standard_normal((400_000, 7)).max(axis=1)
        m1, m2 = max_moments(7)
        assert m1 == pytest.approx(float(draws.mean()), abs=0.01)
        assert m2 == pytest.approx(float((draws ** 2).mean()), abs=0.02)

    def test_sharpe_rises_with_skill_and_is_zero_without_it(self):
        assert oracle_sharpe(0.0)[0] == 0.0
        sweep = [oracle_sharpe(p)[1] for p in (0.0, 0.05, 0.1, 0.3, 1.0)]
        assert sweep == sorted(sweep)

    def test_the_effect_size_does_not_depend_on_volatility(self):
        """Sigma cancels, so the harness cannot appear to recover the effect
        merely by getting the volatility right."""
        a = run_oracle(11, 0.25)
        panel, _ = synthetic_world(n_assets=2, periods=PERIODS, sigma=0.05, seed=11)
        b = walk_forward(panel, Oracle(panel.closes, skill=0.25, seed=11),
                         warmup=WARMUP, rebalance_every=1, cost_model=None,
                         execution=Execution.NEXT_OPEN).track()
        assert a.sharpe == pytest.approx(b.sharpe, rel=0.05)
        assert b.volatility > 4 * a.volatility, "the two worlds are not different"


class TestCanary:
    def test_perfect_foresight_screams(self):
        tr = run_oracle(1, 1.0)
        assert tr.sharpe > 8.0, (
            f"the canary produced {tr.sharpe:.2f}; a strategy that reads "
            f"tomorrow's close should be off the scale, and if it is not the "
            f"future is not reaching the strategy at all")
        assert tr.sharpe > 5 * SUSPICIOUS_ANNUAL_SHARPE
        assert tr.suspicious
        assert "assumed broken" in tr.verdict()

    def test_it_lands_near_its_closed_form(self):
        expected = oracle_sharpe(1.0, 2)[1]
        got = [run_oracle(s, 1.0).sharpe for s in range(4)]
        assert float(np.mean(got)) == pytest.approx(expected, rel=0.10)

    def test_more_assets_make_foresight_worth_more(self):
        assert run_oracle(2, 1.0, n_assets=7).sharpe > run_oracle(2, 1.0).sharpe


class TestPositiveControl:
    @pytest.mark.parametrize("skill", [0.10, 0.25, 0.50])
    def test_the_injected_effect_size_is_recovered(self, skill):
        expected = oracle_sharpe(skill, 2)[1]
        got = np.array([run_oracle(300 + s, skill).sharpe for s in range(10)])
        se = float(got.std(ddof=1)) / np.sqrt(len(got))
        z = (float(got.mean()) - expected) / se
        assert abs(z) < 3.0, (
            f"measured {got.mean():.3f} against an injected {expected:.3f}, "
            f"which is {z:.1f} standard errors away; the harness is not "
            f"recovering the effect it was given")

    def test_zero_skill_recovers_zero(self):
        got = np.array([run_oracle(400 + s, 0.0).sharpe for s in range(10)])
        se = float(got.std(ddof=1)) / np.sqrt(len(got))
        assert abs(float(got.mean())) < 3 * se


@pytest.fixture(scope="module")
def runs():
    """Forty no-skill runs, shared across the assertions about them."""
    out = []
    for seed in range(40):
        panel, _ = synthetic_world(n_assets=7, periods=PERIODS, sigma=0.011,
                                   seed=5000 + seed)
        out.append(walk_forward(panel, CoinFlip(lam=0.375, seed=seed),
                                warmup=WARMUP, rebalance_every=21,
                                cost_model=None,
                                execution=Execution.NEXT_CLOSE).track())
    return out


class TestNegativeControl:
    """The harness must stay quiet when there is nothing there."""

    def test_no_significant_edge_on_average(self, runs):
        sharpes = np.array([r.sharpe for r in runs])
        se = float(sharpes.std(ddof=1)) / np.sqrt(len(sharpes))
        assert abs(float(sharpes.mean())) < 3 * se, (
            f"a no-skill strategy showed a mean annualised Sharpe of "
            f"{sharpes.mean():.3f} ({sharpes.mean() / se:.1f} standard "
            f"errors). Either the harness is broken or the control is not "
            f"no-skill; both void every result that follows.")

    def test_the_t_statistics_are_standard_normal(self, runs):
        """The sharper check. A standard error wrong by a factor k makes the
        spread of t wrong by 1/k, while the rejection rate could still look
        roughly plausible."""
        ts = np.array([r.t_statistic for r in runs])
        assert 0.7 < float(ts.std(ddof=1)) < 1.4, (
            f"t-statistics under a true null have standard deviation "
            f"{ts.std(ddof=1):.3f}, not 1; the standard error formula is wrong")

    def test_it_almost_never_claims_support(self, runs):
        """`supported` is what the harness reports as a real finding."""
        claimed = sum(1 for r in runs if r.supported)
        assert claimed <= 2, (
            f"{claimed} of {len(runs)} no-skill runs were reported as "
            f"supported findings")

    def test_confident_wording_appears_only_when_the_sample_supports_it(self, runs):
        """The verdict is the part a reader acts on, so it is what is tested.

        Every no-skill run must land on one of the honest branches: too short
        to say, at or below zero, or undetermined. The confident phrasing --
        which quotes a t-statistic and a deflated Sharpe as findings -- may
        appear only where `supported` is true.
        """
        for r in runs:
            assert r.independent_observations <= r.observations
            if r.suspicious:
                # The calibration rule outranks everything: an implausible
                # figure is called implausible before it is called a finding.
                assert "assumed broken" in r.verdict()
            elif r.supported:
                assert "deflated Sharpe" in r.verdict()
            else:
                assert any(phrase in r.verdict() for phrase in (
                    "cannot establish it", "nothing to establish",
                    "too few")), r.verdict()

    def test_deflation_removes_the_best_of_forty_no_skill_runs(self, runs):
        """The demonstration that the trial count is doing real work.

        Run forty worthless strategies and the luckiest will look good --
        that is arithmetic, not bad luck. Undeflated, the best of these forty
        clears the conventional 0.95 bar. Told that it was the best of forty,
        the same number stops being evidence. This is why `registry.py` is a
        prerequisite rather than paperwork.
        """
        import numpy as np
        from portfolio.eval.metrics import (deflated_sharpe_ratio,
                                            probabilistic_sharpe_ratio)
        per_period = np.array([r.sharpe_per_period for r in runs])
        spread = float(per_period.std(ddof=1))
        best = max(runs, key=lambda r: r.sharpe_per_period)

        undeflated = probabilistic_sharpe_ratio(
            best.sharpe_per_period, best.independent_observations,
            best.skewness, best.excess_kurtosis)
        deflated = deflated_sharpe_ratio(
            best.sharpe_per_period, best.independent_observations,
            len(runs), spread, best.skewness, best.excess_kurtosis)

        assert undeflated > 0.95, (
            f"the best of {len(runs)} no-skill runs only reached PSR "
            f"{undeflated:.3f}; this demonstration needs a luckier sample")
        assert deflated < undeflated - 0.2
        assert deflated < 0.95, (
            f"deflation left the best of {len(runs)} worthless strategies "
            f"still looking significant (DSR {deflated:.3f})")


class TestTurnoverMatching:
    def test_calibration_hits_the_target(self):
        panel, _ = synthetic_world(n_assets=7, periods=PERIODS, sigma=0.011, seed=77)
        lam, got = calibrate_turnover(panel, target=0.15, warmup=WARMUP,
                                      rebalance_every=21, seed=0)
        assert 0.0 < lam <= 1.0
        assert got == pytest.approx(0.15, abs=0.01)

    def test_turnover_rises_with_lambda(self):
        panel, _ = synthetic_world(n_assets=7, periods=PERIODS, sigma=0.011, seed=78)

        def turnover(lam):
            return walk_forward(panel, CoinFlip(lam=lam, seed=0), warmup=WARMUP,
                                rebalance_every=21, cost_model=None).mean_turnover
        got = [turnover(x) for x in (0.05, 0.25, 0.5, 1.0)]
        assert got == sorted(got), f"turnover is not monotone in lambda: {got}"

    def test_a_target_above_the_maximum_is_reported_not_faked(self):
        panel, _ = synthetic_world(n_assets=7, periods=PERIODS, sigma=0.011, seed=79)
        lam, got = calibrate_turnover(panel, target=0.99, warmup=WARMUP,
                                      rebalance_every=21, seed=0)
        assert lam == 1.0 and got < 0.99
