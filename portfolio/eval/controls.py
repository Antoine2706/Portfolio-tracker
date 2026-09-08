"""Calibrating the instrument before trusting a reading from it.

A backtest produces a confident number whether or not it is measuring
anything. Before any strategy is evaluated, the harness has to be shown to
discriminate -- that it stays quiet when there is nothing there, speaks up
when there is, and screams when it is handed the future. Those are the three
controls, and they are the reason this file exists before `agents/risk.py`
does.

    negative   A no-skill policy with turnover matched to a real one. The
               harness must report no significant edge. If it does report
               one, the harness is broken and every result after it is void.

    positive   A policy with a known injected edge, in a synthetic world
               where its true Sharpe ratio is available in closed form. The
               harness must detect it and roughly recover the injected
               effect size. Without this, a negative result on a real
               strategy means nothing at all -- "we found no edge" and "we
               could not have found an edge" are indistinguishable.

    canary     A policy that decides today using tomorrow's price. It must
               produce an absurd Sharpe ratio. If it does not, the data feed
               leaks and every number the harness has ever produced is
               suspect.

The synthetic world, and why the closed form works
--------------------------------------------------
`synthetic_world` builds a panel whose opens equal the previous close, so
there is no overnight gap. Under `Execution.NEXT_OPEN` that makes the
arithmetic exact: a decision taken at the close of day t is executed at the
open of day t+1 at the same price, the overnight leg earns nothing, and the
new weights earn precisely r_{t+1}. The harness's realised return series is
then exactly the process the closed form describes, so a discrepancy between
them is a defect in the harness rather than a modelling gap. It also exercises
the two-leg execution path rather than the simpler one.

Returns are iid Normal(0, sigma^2) and independent across assets. That is not
a claim about markets -- it is the point. The control has to have a *known*
answer, and the only distribution with a known answer is one we chose.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from ..agents.base import MarketView, Proposal
from ..core.returns import TRADING_DAYS_PER_YEAR
from .harness import Execution, Panel, walk_forward
from .metrics import annualise_sharpe

__all__ = [
    "synthetic_world", "CoinFlip", "Oracle", "max_moments", "oracle_sharpe",
    "calibrate_turnover", "NEGATIVE_CONTROL_SEEDS",
]

# Enough seeds that the *rate* at which the harness rejects a true null can be
# measured rather than eyeballed. One no-skill run proves nothing: a single
# t-statistic above 2 is expected 5% of the time by construction, so seeing
# one tells you nothing about whether the harness is calibrated. With 200
# runs the standard error on an observed 5% rejection rate is about 1.5
# points, which separates "correctly calibrated" from "rejecting three times
# too often".
#
# There is an UPPER bound as well, and it is the more interesting one.
# Rebalancing toward a random point on the simplex is not perfectly
# null: repeatedly trading back toward a diversified mix harvests a small
# genuine diversification return, worth an annualised Sharpe of roughly 0.04
# at these volatilities and scaling with lambda. That is far below the noise
# of a single run, but averaging over S runs shrinks the noise as 1/sqrt(S)
# while leaving the bias untouched, so at
#
#     S >= (1.96 / 0.038)^2  ~=  2650
#
# the pooled negative control would show a *statistically significant*
# positive edge -- and a correctly working harness would look broken. The
# effect is real, not an artefact, which is why the answer is to stay well
# below that ceiling rather than to explain the result away. 200 sits at
# t ~= 0.54, comfortably inside the noise.
NEGATIVE_CONTROL_SEEDS = 200


# --------------------------------------------------------------------------
# The synthetic world
# --------------------------------------------------------------------------


def synthetic_world(n_assets: int = 2, periods: int = 1260, sigma: float = 0.01,
                    seed: int = 0, drift: float = 0.0,
                    ) -> tuple[Panel, pd.DataFrame]:
    """A panel of iid Normal(drift, sigma^2) returns, with no overnight gap.

    Returns the panel and the return matrix it was built from, so a control
    can be checked against the process rather than against another estimate
    of it.

    >>> panel, rets = synthetic_world(n_assets=3, periods=500, seed=1)
    >>> panel.closes.shape, panel.opens.shape
    ((500, 3), (500, 3))
    >>> bool((panel.opens.iloc[1:].to_numpy()
    ...       == panel.closes.iloc[:-1].to_numpy()).all())
    True

    The realised moments are close to the ones asked for, which is the only
    property the closed forms below depend on:

    >>> _, r = synthetic_world(n_assets=2, periods=200_000, sigma=0.01, seed=7)
    >>> abs(float(r.mean().mean())) < 1e-4, round(float(r.std().mean()), 4)
    (True, 0.01)
    """
    if n_assets < 1:
        raise ValueError(f"need at least one asset, got {n_assets}")
    if periods < 3:
        raise ValueError(f"need at least 3 periods, got {periods}")
    rng = np.random.default_rng(seed)
    cols = [f"SYN{i:02d}" for i in range(n_assets)]
    dates = pd.bdate_range("2015-01-01", periods=periods)
    rets = pd.DataFrame(rng.normal(drift, sigma, (periods, n_assets)),
                        index=dates, columns=cols)
    rets.iloc[0] = 0.0
    closes = 100.0 * (1.0 + rets).cumprod()
    # opens[t] = closes[t-1]: no overnight gap, so next-open execution puts
    # the whole of day t's move behind the weights decided at t-1.
    opens = closes.shift(1)
    opens.iloc[0] = closes.iloc[0]
    return Panel(closes=closes, opens=opens), rets


# --------------------------------------------------------------------------
# Moments of the maximum, written out
# --------------------------------------------------------------------------


def max_moments(n: int, grid: int = 200_001, span: float = 12.0
                ) -> tuple[float, float]:
    """E[M] and E[M^2] for M the maximum of n iid standard normals.

    The density of the maximum is n phi(x) Phi(x)^(n-1), so

        E[M^k] = integral of x^k n phi(x) Phi(x)^(n-1) dx

    computed here by Simpson's rule on [-span, span]. Written out rather than
    taken from a table because the positive control's whole value is that its
    expected effect size was derived independently of the code that measures
    it, and a number copied from somewhere is not independent of anything.

    Two closed forms check the quadrature. For n = 2 the maximum of two iid
    standard normals has mean 1/sqrt(pi):

    >>> m1, m2 = max_moments(2)
    >>> round(m1, 9), round(1 / math.sqrt(math.pi), 9)
    (0.564189584, 0.564189584)

    and by symmetry E[max^2] + E[min^2] = E[X^2] + E[Y^2] = 2, with the two
    equal, so E[max^2] = 1:

    >>> round(m2, 9)
    1.0

    For n = 1 the maximum is just the variable itself:

    >>> [round(v, 9) for v in max_moments(1)]
    [0.0, 1.0]
    >>> [round(v, 6) for v in max_moments(7)]
    [1.352178, 2.220304]

    Both figures agree with eight million Monte Carlo draws to four decimal
    places, which is the check that the quadrature itself is right.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if grid % 2 == 0:
        grid += 1                       # Simpson needs an even number of panels
    x = np.linspace(-span, span, grid)
    phi = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    # Phi via erf, elementwise; math.erf is scalar so vectorise it once.
    Phi = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    density = n * phi * np.power(Phi, n - 1)
    h = x[1] - x[0]
    w = np.ones(grid)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    simpson = lambda f: float((w * f).sum() * h / 3.0)      # noqa: E731
    return simpson(x * density), simpson(x * x * density)


def oracle_sharpe(skill: float, n_assets: int = 2,
                  periods_per_year: int = TRADING_DAYS_PER_YEAR
                  ) -> tuple[float, float]:
    """(per-period, annualised) true Sharpe of the oracle with a given skill.

    The oracle is fully invested in exactly one asset each period. With
    probability p it foresees which will do best and takes it; otherwise it
    picks one of the n uniformly at random. Writing M for the maximum of the
    n standardised returns:

        E[r]   = p sigma E[M]                    a random pick has mean zero
        E[r^2] = sigma^2 (p E[M^2] + (1 - p))    the random pick contributes 1
        Var    = sigma^2 (p E[M^2] + 1 - p - p^2 E[M]^2)
        SR     = p E[M] / sqrt(p E[M^2] + 1 - p - p^2 E[M]^2)

    Sigma cancels, which is what we want from a control: the effect size
    depends on the skill injected and on nothing else, so the harness cannot
    appear to recover it merely by getting the volatility right.

    At zero skill there is no edge, whatever the number of assets:

    >>> oracle_sharpe(0.0)[0]
    0.0

    A tenth of a period's foresight over two assets is worth an annualised
    0.90 -- which would be an excellent real strategy, and is what one day in
    ten of perfect knowledge buys:

    >>> [round(v, 6) for v in oracle_sharpe(0.1, 2)]
    [0.056509, 0.897052]

    Perfect foresight is the canary, and it is not subtle:

    >>> [round(v, 4) for v in oracle_sharpe(1.0, 2)]
    [0.6833, 10.8476]
    >>> [round(v, 4) for v in oracle_sharpe(1.0, 7)]
    [2.1599, 34.2876]

    An annualised Sharpe of 34 is the number the canary must shout, and 10.8
    over two assets. Both are twenty times the 1.5 above which a retail
    backtest should be assumed broken -- which is the point. Anything
    quieter means the future is not reaching the strategy, and the test that
    proves the harness can see a leak is itself broken.

    Note that more assets make perfect foresight worth more but each unit of
    partial skill worth less per asset: the maximum of seven is further out
    in the tail than the maximum of two, but a one-in-ten chance of finding
    it is diluted across seven ways to be wrong.
    """
    if not 0.0 <= skill <= 1.0:
        raise ValueError(f"skill must be in [0, 1], got {skill}")
    m1, m2 = max_moments(n_assets)
    mean = skill * m1
    variance = skill * m2 + (1.0 - skill) - skill * skill * m1 * m1
    if variance <= 0:
        return 0.0, 0.0
    per_period = mean / math.sqrt(variance)
    return float(per_period), float(annualise_sharpe(per_period, periods_per_year))


# --------------------------------------------------------------------------
# The control policies
# --------------------------------------------------------------------------


@dataclasses.dataclass
class CoinFlip:
    """A no-skill policy whose turnover can be dialled to match a real one.

    Each rebalance it blends the weights it is holding toward a fresh point
    drawn uniformly from the simplex:

        w_new = (1 - lambda) w_drifted + lambda u,     u ~ Dirichlet(1, ..., 1)

    Lambda is the single knob. At 0 it never trades; at 1 it jumps to a random
    portfolio every time. Turnover rises monotonically and continuously in
    between, so `calibrate_turnover` can bisect on it.

    Blending toward a *fresh* random point rather than toward a fixed target
    is deliberate. Rebalancing repeatedly toward one target harvests a real
    diversification return, and a negative control containing a real effect
    would make a working harness look broken. Redrawing u each time removes
    the persistent target while keeping the trading.

    It sees the view, as every policy must, and ignores it -- which is the
    definition of no skill.
    """
    lam: float = 0.2
    seed: int = 0
    name: str = "coin-flip"

    def __post_init__(self) -> None:
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")
        self._rng = np.random.default_rng(self.seed)

    def observe(self, view: MarketView) -> Proposal:
        available = view.available or view.instruments
        n = len(available)
        if n == 0:
            return Proposal({}, 0.0, "nothing tradable yet", self.name)
        u = self._rng.dirichlet(np.ones(n))
        held = {k: float(v) for k, v in view.held.items()}
        total_held = sum(held.values())
        if total_held <= 0:
            weights = {isin: float(u[i]) for i, isin in enumerate(available)}
        else:
            weights = {}
            for i, isin in enumerate(available):
                weights[isin] = ((1.0 - self.lam) * held.get(isin, 0.0)
                                 + self.lam * float(u[i]))
        # Anything held but no longer available is left to be sold; renormalise
        # so the book stays fully invested and unlevered.
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return Proposal(
            weights, 0.0,
            f"no-skill control: blended {self.lam:.0%} toward a random point "
            f"on the simplex, ignoring every price", self.name)


@dataclasses.dataclass
class Oracle:
    """A policy that is told, with probability `skill`, what happens next.

    This is deliberately a closure over the whole price history rather than a
    policy that reads its `MarketView`. It could not be written any other way
    -- the view physically ends at the decision date -- and that is the point:
    it is the shape a real look-ahead bug takes, an outer-scope frame captured
    where a slice was intended. `leakage.check_lookahead` is what has to
    notice, and `test_leakage.py` asserts that it does.

    At skill 1 it is the look-ahead canary. At skill p it is the positive
    control, with the true Sharpe `oracle_sharpe(p, n)` gives.
    """
    closes: pd.DataFrame
    skill: float = 0.1
    seed: int = 0
    name: str = "oracle"

    def __post_init__(self) -> None:
        if not 0.0 <= self.skill <= 1.0:
            raise ValueError(f"skill must be in [0, 1], got {self.skill}")
        self._rng = np.random.default_rng(self.seed)
        self._pos = {d: i for i, d in enumerate(self.closes.index)}

    def observe(self, view: MarketView) -> Proposal:
        cols = list(self.closes.columns)
        i = self._pos[view.as_of]
        if i + 1 >= len(self.closes):
            pick = int(self._rng.integers(len(cols)))
            reason = "no next bar to see; picked at random"
        else:
            # The return the weights decided now will actually earn, which
            # under a no-gap panel and next-open execution is exactly this.
            nxt = (self.closes.iloc[i + 1] / self.closes.iloc[i] - 1.0).to_numpy()
            if self._rng.random() < self.skill:
                pick = int(np.argmax(nxt))
                reason = (f"saw tomorrow and took the best of {len(cols)} "
                          f"({nxt[pick]:+.2%})")
            else:
                pick = int(self._rng.integers(len(cols)))
                reason = "no foresight this period; picked at random"
        return Proposal({cols[pick]: 1.0}, self.skill, reason, self.name)


# --------------------------------------------------------------------------
# Matching the control's turnover to the strategy it stands in for
# --------------------------------------------------------------------------


def calibrate_turnover(panel: Panel, target: float, *, warmup: int,
                       rebalance_every: int, seed: int = 0,
                       tolerance: float = 0.005, max_iterations: int = 30,
                       execution: Execution = Execution.NEXT_CLOSE,
                       ) -> tuple[float, float]:
    """Find the lambda whose realised mean turnover matches `target`.

    Returns (lambda, realised turnover). Bisection rather than a formula:
    the map from lambda to turnover depends on the drift between rebalances,
    which depends on the panel, so an analytic expression would be an
    approximation dressed up as an identity. Bisection converges because
    turnover is continuous and monotone in lambda, and it is cheap because a
    control run costs nothing.

    Comparing a control against a strategy without matching turnover compares
    two things at once. Since cost is roughly linear in turnover and the whole
    question is whether a policy earns more than it spends, an unmatched
    control answers a different question from the one asked.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target turnover must be in (0, 1], got {target}")

    def realised(lam: float) -> float:
        result = walk_forward(panel, CoinFlip(lam=lam, seed=seed), warmup=warmup,
                              rebalance_every=rebalance_every, cost_model=None,
                              execution=execution)
        return result.mean_turnover

    lo, hi = 0.0, 1.0
    hi_turnover = realised(hi)
    if hi_turnover < target:
        # Even jumping to a fresh random portfolio every time is calmer than
        # the strategy being matched. Say so rather than returning a lambda
        # that silently misses the target.
        return 1.0, hi_turnover
    best = (1.0, hi_turnover)
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        got = realised(mid)
        best = (mid, got)
        if abs(got - target) <= tolerance:
            return best
        if got < target:
            lo = mid
        else:
            hi = mid
    return best


# --------------------------------------------------------------------------
# Running all three, and saying whether the harness may be trusted
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ControlOutcome:
    """One control, its expected behaviour, and whether it behaved."""
    name: str
    passed: bool
    detail: str
    numbers: dict


@dataclasses.dataclass(frozen=True)
class CalibrationReport:
    """The three controls plus the leak detector. Read before anything else."""
    outcomes: tuple[ControlOutcome, ...]

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.outcomes)

    def lines(self) -> list[str]:
        out = ["Harness calibration", "=" * 60, ""]
        for o in self.outcomes:
            out.append(f"[{'PASS' if o.passed else 'FAIL'}] {o.name}")
            for line in o.detail.splitlines():
                out.append(f"       {line}")
            out.append("")
        out.append("-" * 60)
        out.append("The harness discriminates: it stays quiet on a no-skill"
                   if self.passed else
                   "AT LEAST ONE CONTROL FAILED.")
        if self.passed:
            out.append("strategy, recovers a known edge, and screams when handed")
            out.append("the future. Results from it can be read as measurements.")
        else:
            out.append("No result from this harness means anything until the")
            out.append("failure above is understood. Do not evaluate strategies.")
        return out


def run_calibration(*, seeds: int = NEGATIVE_CONTROL_SEEDS, periods: int = 1260,
                    n_assets: int = 7, warmup: int = 252,
                    rebalance_every: int = 21, target_turnover: float = 0.15,
                    skills: "tuple[float, ...]" = (0.0, 0.05, 0.10, 0.25),
                    positive_seeds: int = 12) -> CalibrationReport:
    """Run the three controls and the leak detector, and judge each one.

    This is what `portfolio controls` prints, and it is the gate on
    everything after it in the build order. It deliberately reports numbers
    rather than only pass or fail: the rejection rate and the spread of the
    t-statistics say *how* well calibrated the harness is, and a bare tick
    would hide a harness that passes for the wrong reason.
    """
    from .leakage import VacuousLeakCheck, check_lookahead
    from .metrics import SUSPICIOUS_ANNUAL_SHARPE

    outcomes: list[ControlOutcome] = []

    # ---- negative: no skill, matched turnover -----------------------------
    calib_panel, _ = synthetic_world(n_assets=n_assets, periods=periods,
                                     sigma=0.011, seed=11)
    lam, realised_turnover = calibrate_turnover(
        calib_panel, target=target_turnover, warmup=warmup,
        rebalance_every=rebalance_every, seed=0)

    # Both overlap settings, because only one of them was ever calibrated.
    #
    # The real backtest declares `overlap = rebalance_every`, this control ran
    # at the default of 1, and the two went down different code paths. The
    # declared-overlap path deflated every t-statistic by roughly the square
    # root of the overlap and nothing noticed, because the only check on the
    # standard error was here, on the path that did not use it. That is the
    # fifth time in this project a check has passed without exercising what it
    # claimed, so the control now covers the setting the tool actually runs.
    ts, ts_overlapped, sharpes = [], [], []
    for seed in range(seeds):
        panel, _ = synthetic_world(n_assets=n_assets, periods=periods,
                                   sigma=0.011, seed=1000 + seed)
        result = walk_forward(panel, CoinFlip(lam=lam, seed=seed), warmup=warmup,
                              rebalance_every=rebalance_every, cost_model=None,
                              execution=Execution.NEXT_CLOSE)
        record = result.track()
        ts.append(record.t_statistic)
        ts_overlapped.append(result.track(overlap=rebalance_every).t_statistic)
        sharpes.append(record.sharpe)
    ts_arr, sharpe_arr = np.array(ts), np.array(sharpes)
    rejection = float(np.mean(np.abs(ts_arr) > 1.959964))
    se_mean = float(sharpe_arr.std(ddof=1)) / math.sqrt(seeds)
    t_of_mean = float(sharpe_arr.mean()) / se_mean
    spread_of_t = float(ts_arr.std(ddof=1))
    spread_overlapped = float(np.array(ts_overlapped).std(ddof=1))
    rejection_se = math.sqrt(0.05 * 0.95 / seeds)
    negative_ok = (abs(t_of_mean) < 3.0
                   and abs(rejection - 0.05) < 4 * rejection_se
                   and 0.75 < spread_of_t < 1.35
                   and 0.70 < spread_overlapped < 1.35)
    outcomes.append(ControlOutcome(
        "negative control - a no-skill policy with matched turnover",
        negative_ok,
        f"{seeds} seeds, {n_assets} assets, {periods} bars, rebalanced every "
        f"{rebalance_every}\n"
        f"turnover matched at {realised_turnover:.3f} one-way per rebalance "
        f"(lambda {lam:.3f}, target {target_turnover:.2f})\n"
        f"mean annualised Sharpe {sharpe_arr.mean():+.4f}, "
        f"t of that mean {t_of_mean:+.2f}  (must be within 3)\n"
        f"spread of the t-statistics {spread_of_t:.4f}  (must be near 1: a "
        f"standard error\n"
        f"  wrong by a factor k shows up here as 1/k)\n"
        f"  the same, declaring overlap {rebalance_every}: "
        f"{spread_overlapped:.4f}  (the setting a real\n"
        f"  backtest runs at, and the one nothing used to check)\n"
        f"rejected the true null in {rejection:.1%} of runs  "
        f"(nominal 5.0%, standard error {rejection_se:.1%})",
        {"seeds": seeds, "lambda": lam, "turnover": realised_turnover,
         "mean_sharpe": float(sharpe_arr.mean()), "t_of_mean": t_of_mean,
         "spread_of_t": spread_of_t, "spread_of_t_overlapped": spread_overlapped,
         "rejection_rate": rejection}))

    # ---- positive: a known injected edge ----------------------------------
    rows, positive_ok = [], True
    for skill in skills:
        expected = oracle_sharpe(skill, 2)[1]
        got = []
        for seed in range(positive_seeds):
            panel, _ = synthetic_world(n_assets=2, periods=periods, sigma=0.01,
                                       seed=100 + seed)
            got.append(walk_forward(
                panel, Oracle(panel.closes, skill=skill, seed=seed),
                warmup=warmup, rebalance_every=1, cost_model=None,
                execution=Execution.NEXT_OPEN).track().sharpe)
        arr = np.array(got)
        se = float(arr.std(ddof=1)) / math.sqrt(len(arr))
        z = (float(arr.mean()) - expected) / se
        positive_ok &= abs(z) < 3.0
        rows.append(f"  skill {skill:>4.2f}   true {expected:>7.4f}   "
                    f"measured {arr.mean():>7.4f} +/- {se:.4f}   z {z:+5.2f}")
    outcomes.append(ControlOutcome(
        "positive control - a known edge, recovered", positive_ok,
        f"{positive_seeds} seeds per skill level, two assets, "
        f"iid normal returns, no costs\n"
        f"true Sharpe from the closed form in `oracle_sharpe`, which was "
        f"derived\nindependently of the code that measures it\n"
        + "\n".join(rows),
        {"skills": list(skills)}))

    # ---- canary: perfect foresight ----------------------------------------
    canary_rows, canary_ok = [], True
    for assets in (2, n_assets):
        panel, _ = synthetic_world(n_assets=assets, periods=periods, sigma=0.01,
                                   seed=1)
        record = walk_forward(panel, Oracle(panel.closes, skill=1.0, seed=0),
                              warmup=warmup, rebalance_every=1, cost_model=None,
                              execution=Execution.NEXT_OPEN).track()
        expected = oracle_sharpe(1.0, assets)[1]
        loud = record.sharpe > 5 * SUSPICIOUS_ANNUAL_SHARPE and record.suspicious
        canary_ok &= loud
        canary_rows.append(
            f"  {assets} assets   true {expected:>8.3f}   measured "
            f"{record.sharpe:>8.3f}   flagged suspicious: "
            f"{'yes' if record.suspicious else 'NO'}")
    outcomes.append(ControlOutcome(
        "look-ahead canary - decides today using tomorrow's close", canary_ok,
        f"must be absurd, and must be flagged. The threshold above which a\n"
        f"retail backtest is assumed broken is "
        f"{SUSPICIOUS_ANNUAL_SHARPE}.\n" + "\n".join(canary_rows), {}))

    # ---- the leak detector itself ----------------------------------------
    leak_panel, _ = synthetic_world(n_assets=4, periods=800, sigma=0.01, seed=5)

    def clean(p):
        return walk_forward(p, CoinFlip(lam=0.3, seed=0), warmup=252,
                            rebalance_every=21, cost_model=None,
                            execution=Execution.NEXT_CLOSE)

    def leaky(p):
        return walk_forward(p, Oracle(p.closes, skill=1.0, seed=0), warmup=252,
                            rebalance_every=21, cost_model=None,
                            execution=Execution.NEXT_OPEN)

    clean_report = check_lookahead(clean, leak_panel)
    leaky_report = check_lookahead(leaky, leak_panel)
    try:
        check_lookahead(lambda _: clean(leak_panel), leak_panel)
        refused = False
    except VacuousLeakCheck:
        refused = True
    detector_ok = (not clean_report.leaked) and leaky_report.leaked and refused
    outcomes.append(ControlOutcome(
        "leak detector - rewrite the future and see what moves", detector_ok,
        f"a policy reading only its view:   "
        f"{'reported clean' if not clean_report.leaked else 'FALSE ALARM'}\n"
        f"a policy reading tomorrow:        "
        f"{'CAUGHT' if leaky_report.leaked else 'MISSED - the detector is blind'}\n"
        f"a check wired so it cannot fail:  "
        f"{'refused' if refused else 'ACCEPTED - it would pass vacuously'}\n"
        f"compared {clean_report.compared_returns} returns and "
        f"{clean_report.compared_decisions} decisions per split", {}))

    return CalibrationReport(tuple(outcomes))
