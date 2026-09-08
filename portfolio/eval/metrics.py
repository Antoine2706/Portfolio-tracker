"""Statistics for a track record, chosen so the numbers cannot flatter.

A backtest always produces a Sharpe ratio. The question this module exists to
answer is whether that number means anything, and the honest answer at 252
daily observations is usually no. So every estimate here is reported with the
uncertainty attached, and the headline summary refuses to make a claim the
sample cannot support.

The three facts the design is built on
--------------------------------------
1.  **A Sharpe ratio is an estimate with a standard error**, and for iid
    returns that error is roughly 1/sqrt(T) in per-period units. Annualised,
    the t-statistic is approximately SR x sqrt(years). Detecting a true Sharpe
    of 0.5 at t = 2 therefore takes sixteen years; 1.0 takes four.

2.  **Trying many strategies inflates the best one's Sharpe** even when none
    of them works. The expected maximum of N independent Sharpe estimates
    drawn from a zero-mean null is strictly positive and grows with N, so the
    winner of a search looks good by construction. The deflated Sharpe ratio
    subtracts that selection effect, which is why `registry.py` exists: the
    correction needs an honest count of how many things were tried.

3.  **Returns are not normal.** Skewness and excess kurtosis both change the
    standard error of a Sharpe estimate, and a strategy that sells tails looks
    best precisely where the normal approximation is worst. Every formula here
    carries the third and fourth moments rather than assuming them away.

Conventions, stated once
------------------------
Every function takes and returns **per-period** quantities unless the name
says `annual`. Mixing the two is the classic bug in this material: an
annualised Sharpe of 0.9 and a daily one of 0.0565 are the same number, and
substituting one for the other in the PSR formula changes the answer by
orders of magnitude. Annualisation is always explicit, always
`x sqrt(periods_per_year)`, and never implicit in an argument.

Kurtosis is **excess** kurtosis throughout: 0 for a normal distribution, not
3. `core.var.sample_skew_kurtosis` returns it in that convention and this
module never converts. Where the published formulae are written with raw
kurtosis g4, the substitution used here is g4 - 1 = excess + 2, and the
normal case is doctested at every step so the convention cannot silently
drift.

Sources
-------
Lo (2002), "The Statistics of Sharpe Ratios", for the standard error and the
autocorrelation correction. Mertens (2002) for the non-normal terms. Bailey
and Lopez de Prado (2012, 2014) for the probabilistic and deflated Sharpe
ratios and the minimum track record length. The formulae are written out here
rather than imported so that each one can be checked against those papers.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from ..core.returns import TRADING_DAYS_PER_YEAR
from ..core.risk import drawdown
from ..core.var import norm_cdf, norm_ppf, sample_skew_kurtosis

__all__ = [
    "EULER_MASCHERONI", "SUSPICIOUS_ANNUAL_SHARPE", "SUSPICIOUS_ACTIVE_SHARPE",
    "TrackRecord",
    "sharpe_variance_factor", "sharpe_standard_error", "sharpe_t_statistic",
    "probabilistic_sharpe_ratio", "expected_maximum_sharpe",
    "deflated_sharpe_ratio", "deflation_threshold",
    "minimum_track_record_length", "years_to_detect",
    "annualise_sharpe", "deannualise_sharpe", "rescale_sharpe",
    "turnover_series", "effective_observations", "track_record",
]

# Appears in the expected maximum of N independent normals. Not a fitted
# constant: it comes out of the asymptotic expansion of the Gumbel limit.
EULER_MASCHERONI = 0.5772156649015328606

# The calibration rule. The best documented hedge fund records in history run
# at an annualised Sharpe of 2 to 3, on infrastructure and information no
# retail investor has. A backtest on a seven-ETF retail portfolio reporting
# more than this is a bug until proven otherwise -- most often a look-ahead
# leak, a missing cost model, or a survivorship-filtered universe. The harness
# flags it rather than celebrating it.
SUSPICIOUS_ANNUAL_SHARPE = 1.5

# The same rule applied to the quantity that is actually about the policy.
#
# A level above 1.5 says something about the window, and the benchmark shares
# the window, so on a rising book both legs clear it and the alarm fires on
# neither's account. The informative quantity is the ACTIVE series -- policy
# minus benchmark, day by day -- whose Sharpe is the information ratio. A
# retail rebalancing rule with an out-of-sample information ratio above 1 is
# implausible for the same reason, and unlike the level it cannot be produced
# by a good year: doing nothing scores exactly zero on it by construction.
SUSPICIOUS_ACTIVE_SHARPE = 1.0


# --------------------------------------------------------------------------
# The Sharpe ratio as an estimate
# --------------------------------------------------------------------------


def annualise_sharpe(per_period: float,
                     periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """SR_annual = SR_period x sqrt(periods per year).

    >>> round(annualise_sharpe(0.0565), 6)
    0.89691
    """
    return float(per_period * math.sqrt(periods_per_year))


def deannualise_sharpe(annual: float,
                       periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """The inverse, because every formula below wants per-period units.

    >>> round(deannualise_sharpe(0.89691), 6)
    0.0565
    """
    return float(annual / math.sqrt(periods_per_year))


def sharpe_variance_factor(sharpe: float, skewness: float = 0.0,
                           excess_kurtosis: float = 0.0) -> float:
    """The bracket shared by the standard error, PSR, DSR and MinTRL.

        1 - g3 SR + (g4 - 1)/4 SR^2         with g4 the *raw* kurtosis

    written here with excess kurtosis K = g4 - 3, so (g4 - 1)/4 = (K + 2)/4:

        1 - skew x SR + (K + 2)/4 x SR^2

    One function rather than four copies, because the single most likely way
    for this material to go quietly wrong is for the four formulae that share
    this term to stop agreeing with each other.

    For normally distributed returns it collapses to the familiar 1 + SR^2/2:

    >>> round(sharpe_variance_factor(0.0565), 12)
    1.001596125
    >>> round(1 + 0.0565 ** 2 / 2, 12)
    1.001596125

    Negative skew and fat tails both raise it, which widens the error bars --
    the correct direction, since a strategy that sells tails has a Sharpe
    ratio that is harder to pin down, not easier:

    >>> round(sharpe_variance_factor(0.0565, skewness=-0.3, excess_kurtosis=2.0), 10)
    1.02014225

    `sharpe` is per period. Passing an annualised one produces a number that
    is wrong by roughly a factor of the periods per year.
    """
    return float(1.0 - skewness * sharpe
                 + ((excess_kurtosis + 2.0) / 4.0) * sharpe * sharpe)


def sharpe_standard_error(sharpe: float, observations: int,
                          skewness: float = 0.0,
                          excess_kurtosis: float = 0.0) -> float:
    """Standard error of a per-period Sharpe estimate (Lo 2002, Mertens 2002).

        SE = sqrt( (1 - g3 SR + (g4-1)/4 SR^2) / (T - 1) )

    T - 1 rather than T follows Bailey and Lopez de Prado, so that this and
    the PSR below are the same statement; at T = 252 the choice moves the
    answer by 0.2%.

    >>> round(sharpe_standard_error(0.0565, 252), 8)
    0.06316979
    >>> round(sharpe_standard_error(0.0565, 252, -0.3, 2.0), 8)
    0.06375195

    Fat tails and negative skew widen it, so the same estimate is less
    significant than a normal assumption would claim.
    """
    if observations < 2:
        raise ValueError(
            f"a standard error needs at least 2 observations, got {observations}")
    return float(math.sqrt(
        sharpe_variance_factor(sharpe, skewness, excess_kurtosis)
        / (observations - 1)))


def rescale_sharpe(sharpe: float, observations: int, effective: int) -> float:
    """A per-period Sharpe restated at the effective observation frequency.

    Needed because a Sharpe ratio and an observation count are only comparable
    at the same frequency, and `effective_observations` changes the count
    without changing the ratio. A daily Sharpe used with a monthly-equivalent
    count is not a conservative approximation, it is a different quantity:

        SR(h periods) = SR(1 period) x sqrt(h)

    so coarsening 256 daily observations into 12 effective ones means each of
    those 12 carries sqrt(256/12) times the daily ratio.

    >>> round(rescale_sharpe(0.10064, 256, 12), 6)
    0.464836
    >>> rescale_sharpe(0.10064, 256, 256)
    0.10064

    This existed implicitly and wrongly: `track_record` fed a daily Sharpe to
    a standard error computed on the effective count, which deflated every
    t-statistic by roughly the square root of the overlap. Under a true null
    the t-statistics then had standard deviation 0.21 instead of 1 -- an error
    in the under-claiming direction, which is why it survived: a harness that
    finds nothing is not obviously broken.
    """
    if observations < 1 or effective < 1:
        raise ValueError("observation counts must be positive")
    return float(sharpe * math.sqrt(observations / effective))


def sharpe_t_statistic(sharpe: float, observations: int, skewness: float = 0.0,
                       excess_kurtosis: float = 0.0) -> float:
    """t = SR / SE(SR), on per-period inputs.

    >>> round(sharpe_t_statistic(0.0565, 252), 6)
    0.894415
    >>> round(sharpe_t_statistic(0.0565, 252, -0.3, 2.0), 6)
    0.886247

    An annualised Sharpe of 0.9 over a single year is not significant. That is
    not a defect in the estimator, it is the sample size.
    """
    return float(sharpe / sharpe_standard_error(
        sharpe, observations, skewness, excess_kurtosis))


def years_to_detect(annual_sharpe: float, target_t: float = 2.0) -> float:
    """Years of data needed for a true annual Sharpe to reach `target_t`.

    From t ~ SR_annual x sqrt(years), so years = (t / SR)^2. This is the table
    that decides what is worth building at all:

    >>> [round(years_to_detect(s), 2) for s in (0.5, 1.0, 2.0)]
    [16.0, 4.0, 1.0]

    Sixteen years to distinguish a genuinely good strategy from luck is why
    this project evaluates policies that require no return forecast first, and
    why the one return-based hypothesis is pre-registered with its expected
    outcome written down in advance.
    """
    if annual_sharpe <= 0:
        return float("inf")
    return float((target_t / annual_sharpe) ** 2)


# --------------------------------------------------------------------------
# Selection bias: PSR, the expected maximum, and DSR
# --------------------------------------------------------------------------


def probabilistic_sharpe_ratio(sharpe: float, observations: int,
                               skewness: float = 0.0,
                               excess_kurtosis: float = 0.0,
                               benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), given the estimate and its moments.

        PSR = Z[ (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2) ]

    All Sharpe ratios per period. The result is a probability, which is the
    point: it says "there is a 81% chance this strategy's true Sharpe is above
    zero" rather than printing 0.897 and letting the reader assume certainty.

    >>> round(probabilistic_sharpe_ratio(0.0565, 252, -0.3, 2.0), 8)
    0.81225787

    A PSR of 0.81 is *not* significance. The conventional bar is 0.95, and
    reaching it here would need roughly three and a half times the data.
    """
    if observations < 2:
        raise ValueError(
            f"PSR needs at least 2 observations, got {observations}")
    factor = sharpe_variance_factor(sharpe, skewness, excess_kurtosis)
    if factor <= 0:
        raise ValueError(
            f"the variance factor is {factor:.4g}, which is not positive; the "
            f"moments supplied are not internally consistent")
    z = (sharpe - benchmark) * math.sqrt(observations - 1) / math.sqrt(factor)
    return float(norm_cdf(z))


def expected_maximum_sharpe(trials: int, trial_sharpe_sd: float) -> float:
    """The Sharpe the *best* of N worthless strategies is expected to show.

        SR* = sd x [ (1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)) ]

    with g the Euler-Mascheroni constant. This is the standard approximation
    to the expectation of the maximum of N independent standard normals,
    scaled by the dispersion of the Sharpe estimates actually observed across
    the trials. It is the whole reason a pre-registration log is a
    mathematical requirement rather than paperwork: N is an input to the
    correction, and an N that quietly omits the failures produces a threshold
    that is too low and a discovery that is not one.

    With a single trial there is no selection to correct for, and the formula
    is undefined anyway (Z^-1(0) is minus infinity), so the threshold is zero:

    >>> expected_maximum_sharpe(1, 0.02)
    0.0

    Twenty trials whose Sharpe estimates scatter with a standard deviation of
    0.02 per day mean the best of them is expected to show 0.038 per day --
    about 0.60 annualised -- from nothing but chance:

    >>> round(expected_maximum_sharpe(20, 0.02), 8)
    0.03801416
    >>> round(annualise_sharpe(expected_maximum_sharpe(20, 0.02)), 4)
    0.6035

    The threshold grows slowly -- it is driven by the tail of a normal, so it
    goes roughly as sqrt(2 log N) -- but it grows without bound, and it is
    invisible without the log:

    >>> round(annualise_sharpe(expected_maximum_sharpe(100, 0.02)), 4)
    0.8034
    >>> round(annualise_sharpe(expected_maximum_sharpe(1000, 0.02)), 4)
    1.0335

    A thousand attempts make an annualised Sharpe of 1.03 the *expected* best
    result from a set of strategies that do nothing at all.
    """
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if trial_sharpe_sd < 0:
        raise ValueError(
            f"the spread of Sharpe estimates cannot be negative, got {trial_sharpe_sd}")
    if trials == 1:
        return 0.0
    n = float(trials)
    return float(trial_sharpe_sd * (
        (1.0 - EULER_MASCHERONI) * norm_ppf(1.0 - 1.0 / n)
        + EULER_MASCHERONI * norm_ppf(1.0 - 1.0 / (n * math.e))))


def deflation_threshold(observations: int, trials: int,
                        trial_sharpe_sd: float) -> tuple[float, bool]:
    """(SR* per period, whether a fallback spread had to be substituted).

    Guards the failure mode that matters most here: `trial_sharpe_sd` coming
    back as zero while `trials` is greater than one. That happens whenever
    several variants were registered but their Sharpe ratios were not all
    recorded, and it sets SR* to zero -- which silently turns the deflated
    Sharpe back into the undeflated one, printing the most optimistic number
    available at exactly the moment the evidence is weakest.

    So when it happens the spread falls back to 1/sqrt(T-1), the standard
    error of a Sharpe estimate under the null, and the caller is told:

    >>> deflation_threshold(252, 1, 0.0)
    (0.0, False)
    >>> t, fell_back = deflation_threshold(252, 20, 0.0)
    >>> fell_back, round(t, 6)
    (True, 0.119972)
    >>> deflation_threshold(252, 20, 0.02)[1]
    False
    """
    if trials <= 1:
        return 0.0, False
    fallback = trial_sharpe_sd <= 0.0
    spread = (1.0 / math.sqrt(max(observations - 1, 1))) if fallback else trial_sharpe_sd
    return expected_maximum_sharpe(trials, spread), fallback


def deflated_sharpe_ratio(sharpe: float, observations: int, trials: int,
                          trial_sharpe_sd: float, skewness: float = 0.0,
                          excess_kurtosis: float = 0.0) -> float:
    """PSR measured against the expected maximum of the trials, not against 0.

    The number to report when a strategy was selected from a set. Bailey and
    Lopez de Prado's point is that PSR against zero answers the wrong
    question once you have searched: the relevant null is not "is this better
    than nothing" but "is this better than the best thing chance would have
    handed me for the same amount of searching".

    Same estimate as the PSR doctest above, but now as the winner of twenty
    pre-registered attempts:

    >>> round(deflated_sharpe_ratio(0.0565, 252, 20, 0.02, -0.3, 2.0), 8)
    0.61407853

    0.81 becomes 0.61. Nothing about the strategy changed; what changed is the
    admission of how many were tried.
    """
    threshold, _ = deflation_threshold(observations, trials, trial_sharpe_sd)
    return probabilistic_sharpe_ratio(sharpe, observations, skewness,
                                      excess_kurtosis, benchmark=threshold)


def minimum_track_record_length(sharpe: float, skewness: float = 0.0,
                                excess_kurtosis: float = 0.0,
                                benchmark: float = 0.0,
                                confidence: float = 0.95) -> float | None:
    """Observations needed before this Sharpe would beat `benchmark` at `confidence`.

        MinTRL = 1 + (1 - g3 SR + (g4-1)/4 SR^2) x (Z_a / (SR - SR*))^2

    Returns None when the estimate is at or below the benchmark, since no
    amount of data makes it significant in the claimed direction.

    A daily Sharpe of 0.0565 -- 0.90 annualised, which would be a good result
    -- needs 866 observations, about three and a half years of daily data,
    before it is distinguishable from zero at 95%:

    >>> round(minimum_track_record_length(0.0565, -0.3, 2.0), 2)
    865.61
    >>> minimum_track_record_length(0.0, 0.0, 0.0) is None
    True

    This is what the harness reports instead of a p-value when the window is
    too short: not "no effect", but "this window cannot tell, and here is the
    one that could".
    """
    if sharpe <= benchmark:
        return None
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    z = norm_ppf(confidence)
    factor = sharpe_variance_factor(sharpe, skewness, excess_kurtosis)
    return float(1.0 + factor * (z / (sharpe - benchmark)) ** 2)


# --------------------------------------------------------------------------
# Overlapping windows
# --------------------------------------------------------------------------


def effective_observations(observations: int, overlap: int) -> int:
    """Independent observations remaining when each one spans `overlap` periods.

    A decision taken every day on a 252-day estimation window shares 251 of
    those 252 days with the decision before it. The resulting track record has
    `observations` rows but nothing like that many independent facts in it,
    and a t-statistic computed on the raw count is inflated by roughly
    sqrt(overlap).

    The correction used here is the conventional one for non-overlapping
    blocks, T_eff = T / overlap, floored at 1. It is deliberately crude: the
    exact factor depends on the autocorrelation the overlap induces, and a
    precise-looking number here would be false precision. Crude and
    conservative beats exact and wrong.

    >>> effective_observations(252, 1)
    252
    >>> effective_observations(252, 21)
    12
    >>> effective_observations(10, 21)
    1
    """
    if observations < 0:
        raise ValueError(f"observations cannot be negative, got {observations}")
    if overlap < 1:
        raise ValueError(f"overlap must be at least 1, got {overlap}")
    return max(1, int(observations // overlap))


# --------------------------------------------------------------------------
# Trading cost and activity
# --------------------------------------------------------------------------


def turnover_series(targets: pd.DataFrame, drifted: pd.DataFrame) -> pd.Series:
    """One-way turnover per rebalance: 0.5 x sum |w_target - w_drifted|.

    The half is what makes it one-way: moving 10% of the book from A to B
    changes two weights by 0.10 each, and that is one 10% trade, not two.

    Both frames are indexed by rebalance date and share their columns; the
    drifted frame is the previous target carried forward by realised returns,
    which is the weight actually held immediately before the trade. Comparing
    a new target against the previous *target* instead would report zero
    turnover for a portfolio that never trades, which is the wrong sign of
    wrong -- drift is precisely what a rebalance has to undo.

    >>> import pandas as pd
    >>> t = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]})
    >>> d = pd.DataFrame({"a": [0.5, 0.6], "b": [0.5, 0.4]})
    >>> [round(x, 10) for x in turnover_series(t, d)]
    [0.0, 0.1]
    """
    if list(targets.columns) != list(drifted.columns):
        raise ValueError("targets and drifted weights must share their columns")
    diff = (targets.astype(float) - drifted.reindex_like(targets).astype(float)).abs()
    return 0.5 * diff.sum(axis=1)


# --------------------------------------------------------------------------
# The whole picture
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrackRecord:
    """Everything worth saying about one out-of-sample return series.

    Constructed by `track_record`. Read `verdict()` before reading `sharpe`:
    the ratio on its own is the number this class exists to stop anyone
    quoting unqualified.
    """
    observations: int                    # admitted to the estimator
    excluded_observations: int           # real money, but not clean one-day returns
    independent_observations: int
    periods_per_year: int
    total_return: float                  # over everything realised
    mean_per_period: float
    volatility: float                    # annualised
    sharpe: float | None                 # annualised
    sharpe_per_period: float | None
    sharpe_effective: float | None       # at the effective observation frequency
    skewness: float
    excess_kurtosis: float
    standard_error: float | None         # of the per-period Sharpe
    t_statistic: float | None
    probabilistic_sharpe: float | None
    trials: int
    expected_max_sharpe: float | None    # per period, the deflation threshold
    deflated_sharpe: float | None
    minimum_track_record: float | None   # observations required
    max_drawdown: float
    hit_rate: float
    turnover: float | None               # mean one-way turnover per rebalance
    annual_turnover: float | None
    cost_drag: float | None              # annualised return lost to costs
    suspicious: bool
    # True when several trials were registered but their Sharpe ratios were
    # not, so the deflation had to substitute the null standard error for the
    # missing spread. The deflated figure is then indicative, not measured.
    deflation_fallback: bool = False

    @property
    def realised_observations(self) -> int:
        """Every period actually lived through, admitted to the estimator or not."""
        return self.observations + self.excluded_observations

    @property
    def years(self) -> float:
        return float(self.realised_observations / self.periods_per_year)

    @property
    def annual_standard_error(self) -> float | None:
        """Standard error of the ANNUALISED Sharpe, in the same units as it.

        The number to print beside a Sharpe ratio, because a ratio without one
        invites a ranking the sample cannot support. Derived from the
        per-observation error at the effective frequency:

            SR_annual = SR_eff x sqrt(independent / years)

        so the error scales the same way, and SR_annual / SE_annual is exactly
        the reported t-statistic.
        """
        if self.standard_error is None or self.years <= 0:
            return None
        return float(self.standard_error
                     * math.sqrt(self.independent_observations / self.years))

    @property
    def supported(self) -> bool:
        """True when the sample is long enough to support the claim."""
        if self.minimum_track_record is None:
            return False
        return self.independent_observations >= self.minimum_track_record

    def band(self) -> str:
        """`1.60 +/- 1.09` -- the estimate and what the sample can pin it to."""
        if self.sharpe is None:
            return "n/a"
        se = self.annual_standard_error
        return f"{self.sharpe:.2f}" if se is None else f"{self.sharpe:.2f} +/- {se:.2f}"

    def verdict(self) -> str:
        """A sentence that refuses to overstate what the sample can carry."""
        if self.sharpe is None:
            return (f"{self.observations} observations is too few to estimate "
                    f"anything. No Sharpe ratio is reported.")
        if self.suspicious:
            return (f"Annualised Sharpe {self.sharpe:.2f} is above the "
                    f"{SUSPICIOUS_ANNUAL_SHARPE} threshold at which a retail "
                    f"backtest should be assumed broken. Find the leak before "
                    f"reporting this as a result.")
        if not self.supported:
            need = self.minimum_track_record
            if need is None:
                return (f"Annualised Sharpe {self.band()} is at or below zero, "
                        f"so there is nothing to establish.")
            years = need / self.independent_observations * self.years
            return (f"Annualised Sharpe {self.band()}, and {self.years:.1f} "
                    f"years of data cannot establish it: about {years:.1f} "
                    f"years would be needed at this ratio. Reported as "
                    f"undetermined, not as a result.")
        return (f"Annualised Sharpe {self.band()} over {self.years:.1f} years "
                f"({self.independent_observations} independent observations), "
                f"t = {self.t_statistic:.2f}, deflated Sharpe "
                f"{self.deflated_sharpe:.2f} against {self.trials} "
                f"pre-registered trials.")


def track_record(returns: pd.Series, *, trials: int = 1,
                 trial_sharpe_sd: float = 0.0,
                 periods_per_year: int = TRADING_DAYS_PER_YEAR,
                 overlap: int = 1, costs: float | None = None,
                 turnover: float | None = None,
                 rebalances_per_year: float | None = None,
                 realised_returns: "pd.Series | None" = None,
                 confidence: float = 0.95) -> TrackRecord:
    """Summarise an out-of-sample return series, uncertainty included.

    `returns` are per-period simple returns, already net of costs. `costs` is
    the total cost paid over the series as a fraction of average capital, used
    only to report the drag; it is not subtracted again here.

    `realised_returns` is the full series actually experienced, when `returns`
    is a subset of it. The split exists because a day on which a held
    instrument did not trade is real money -- it belongs in the total return
    and the drawdown -- but is not a clean one-day observation, so it must not
    enter a variance. Passing the same series twice, or omitting it, keeps the
    old behaviour.

    `overlap` is the number of periods each decision's estimation window
    shares with the next, and feeds `effective_observations`. Leave it at 1
    for a track record whose observations really are independent.

    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> r = pd.Series(rng.normal(0.0004, 0.01, 252))
    >>> tr = track_record(r)
    >>> tr.observations
    252
    >>> tr.sharpe is not None and tr.probabilistic_sharpe is not None
    True

    A pure-noise series of this length is correctly reported as unsupported
    rather than as a small positive result:

    >>> tr.supported
    False
    """
    r = returns.dropna().astype(float)
    n = int(len(r))
    arr = r.to_numpy()

    if n < 2:
        full = r if realised_returns is None else realised_returns.dropna().astype(float)
        return TrackRecord(
            observations=n, excluded_observations=int(len(full) - n),
            independent_observations=n,
            periods_per_year=periods_per_year,
            total_return=float(np.prod(1.0 + full.to_numpy()) - 1.0) if len(full) else 0.0,
            mean_per_period=float(arr.mean()) if n else 0.0,
            volatility=0.0, sharpe=None, sharpe_per_period=None,
            sharpe_effective=None,
            skewness=0.0, excess_kurtosis=0.0, standard_error=None,
            t_statistic=None, probabilistic_sharpe=None, trials=trials,
            expected_max_sharpe=None, deflated_sharpe=None,
            minimum_track_record=None, max_drawdown=0.0, hit_rate=0.0,
            turnover=turnover, annual_turnover=None, cost_drag=None,
            suspicious=False)

    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    skew, exk = sample_skew_kurtosis(arr)
    independent = effective_observations(n, overlap)

    sr_period = float(mean / sd) if sd > 0 else None
    sr_annual = annualise_sharpe(sr_period, periods_per_year) if sr_period is not None else None

    # Every statistic below is computed at the EFFECTIVE frequency: a Sharpe
    # restated for `independent` observations, used with that same count. The
    # ratio and the count have to describe the same sampling interval or the
    # answer is neither the daily one nor the monthly one. Skew and kurtosis
    # are the daily figures, which shrink under aggregation -- so the error
    # bars come out slightly wide, which is the right direction.
    se = t = psr = dsr = mintrl = threshold = None
    sr_effective = None
    deflation_fallback = False
    if sr_period is not None and independent >= 2:
        sr_effective = rescale_sharpe(sr_period, n, independent)
        se = sharpe_standard_error(sr_effective, independent, skew, exk)
        t = sr_effective / se
        psr = probabilistic_sharpe_ratio(sr_effective, independent, skew, exk)
        threshold, deflation_fallback = deflation_threshold(
            independent, trials, trial_sharpe_sd)
        dsr = probabilistic_sharpe_ratio(sr_effective, independent, skew, exk,
                                         benchmark=threshold)
        mintrl = minimum_track_record_length(sr_effective, skew, exk,
                                             benchmark=threshold,
                                             confidence=confidence)

    realised = r if realised_returns is None else realised_returns.dropna().astype(float)
    values = (1.0 + realised).cumprod()
    dd = drawdown(values)
    years = len(realised) / periods_per_year
    annual_turnover = None
    if turnover is not None:
        per_year = (rebalances_per_year if rebalances_per_year is not None
                    else (periods_per_year if n else 0.0))
        annual_turnover = float(turnover * per_year)
    cost_drag = float(costs / years) if costs is not None and years > 0 else None

    return TrackRecord(
        observations=n,
        excluded_observations=int(len(realised) - n),
        independent_observations=independent,
        periods_per_year=periods_per_year,
        total_return=float(np.prod(1.0 + realised.to_numpy()) - 1.0),
        mean_per_period=mean,
        volatility=float(sd * math.sqrt(periods_per_year)),
        sharpe=sr_annual,
        sharpe_per_period=sr_period,
        sharpe_effective=sr_effective,
        skewness=float(skew),
        excess_kurtosis=float(exk),
        standard_error=se,
        t_statistic=t,
        probabilistic_sharpe=psr,
        trials=trials,
        expected_max_sharpe=threshold,
        deflated_sharpe=dsr,
        minimum_track_record=mintrl,
        max_drawdown=float(dd.max_drawdown),
        hit_rate=float(np.mean(arr > 0)),
        turnover=turnover,
        annual_turnover=annual_turnover,
        cost_drag=cost_drag,
        suspicious=bool(sr_annual is not None
                        and sr_annual > SUSPICIOUS_ANNUAL_SHARPE),
        deflation_fallback=deflation_fallback,
    )
