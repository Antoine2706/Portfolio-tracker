"""Value at risk, expected shortfall, worst periods and stress scenarios.

Three VaR methods are reported side by side because each is wrong in a
different, known way, and the disagreement between them is information:

    historical      the empirical quantile of what actually happened. Honest
                    about fat tails, but it can only show what the window
                    contained -- 252 days that held no crash report no crash.
    parametric      assumes normal returns with the sample mean and standard
                    deviation. Smooth and stable, and known to understate
                    tails: daily equity returns have more mass three standard
                    deviations out than a normal allows.
    Cornish-Fisher  the parametric figure corrected for the sample's skewness
                    and excess kurtosis. Sits between the two; it is the
                    method that says how much of the gap is the tails.

All three are written out in numpy. The normal quantile is a rational
approximation (Acklam) polished with one Newton step against `math.erfc`,
because the standard library has no inverse normal CDF and scipy is not a
dependency of `core`. Losses are reported as positive fractions: 0.031 is a
3.1% loss. A number is never quietly reduced to a "safer" one -- an
expected shortfall below the VaR, or a negative loss under a strong positive
drift, is reported as computed, because it is the estimator's answer.

Horizon
-------
Parametric and Cornish-Fisher scale the mean by h and the standard deviation
by sqrt(h), the square-root-of-time rule that also underlies annualisation,
with the same caveat: it assumes serially uncorrelated returns. Under that
same independence, skewness shrinks by 1/sqrt(h) and excess kurtosis by 1/h,
so the Cornish-Fisher correction is scaled too rather than applied at its
daily size to a ten-day figure. The historical method does not scale at all:
it compounds overlapping h-day windows from the daily series and takes their
quantile, which is what "a ten-day loss" literally means in the data. The
windows overlap, so the effective sample is smaller than the count suggests;
`observations` reports the number of windows.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import numpy as np
import pandas as pd

from .returns import TRADING_DAYS_PER_YEAR
from .risk import portfolio_return_series

__all__ = [
    "VarEstimate", "WorstPeriod", "Scenario",
    "historical_var", "parametric_var", "cornish_fisher_var", "var_report",
    "worst_periods", "stress_scenarios", "norm_ppf", "norm_pdf",
    "cornish_fisher_quantile", "sample_skew_kurtosis",
]


# --------------------------------------------------------------------------
# The normal distribution, without scipy
# --------------------------------------------------------------------------

# Peter Acklam's rational approximation to the inverse normal CDF. Relative
# error about 1.15e-9 on its own; the Newton step below takes it to machine
# precision using erfc, which the standard library does have.
_ACKLAM_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACKLAM_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
_ACKLAM_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACKLAM_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00)
_ACKLAM_LOW = 0.02425


def norm_pdf(x: float) -> float:
    """Standard normal density.

    >>> round(norm_pdf(0.0), 6)
    0.398942
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the complementary error function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF: the z with P(Z <= z) = p.

    >>> round(norm_ppf(0.95), 4)
    1.6449
    >>> round(norm_ppf(0.99), 4)
    2.3263
    >>> norm_ppf(0.5)
    0.0
    >>> round(norm_ppf(0.05), 4)
    -1.6449
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be strictly between 0 and 1, got {p}")
    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D
    if p < _ACKLAM_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
             / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    elif p > 1.0 - _ACKLAM_LOW:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
              / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    else:
        q = p - 0.5
        r = q * q
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
             / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    # One Newton step on the CDF residual, Acklam's own refinement.
    e = norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


# --------------------------------------------------------------------------
# Moments and the Cornish-Fisher expansion
# --------------------------------------------------------------------------

def sample_skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    """Skewness and *excess* kurtosis from central moments.

        S = m3 / m2^1.5      K = m4 / m2^2 - 3      m_k = mean((x - mean)^k)

    The plain moment estimators, with no small-sample adjustment, so the
    figure can be recomputed by hand; the adjustments matter at n = 20, not
    at n = 252. Both are 0 for a constant series rather than a division by
    zero.

    >>> sample_skew_kurtosis(np.array([1.0, 2.0, 3.0]))     # symmetric: S = 0
    (0.0, -1.5)
    """
    x = np.asarray(x, dtype=float)
    centred = x - x.mean()
    m2 = float(np.mean(centred ** 2))
    if m2 == 0.0:
        return 0.0, 0.0
    m3 = float(np.mean(centred ** 3))
    m4 = float(np.mean(centred ** 4))
    return m3 / m2 ** 1.5, m4 / m2 ** 2 - 3.0


def cornish_fisher_quantile(z: float, skew: float, excess_kurtosis: float) -> float:
    """The normal quantile z corrected for skewness S and excess kurtosis K.

        z_cf = z + (z^2 - 1) S/6 + (z^3 - 3z) K/24 - (2z^3 - 5z) S^2/36

    For a left-tail z (negative), negative skew and positive excess kurtosis
    both push z_cf further out, which is the whole point: a normal quantile
    understates a fat, left-leaning tail. The expansion is only monotone for
    moderate S and K; outside that region (roughly |S| < 1, K < 4) the
    correction can fold back on itself and the figure should be read with
    the sample moments beside it.

    >>> cornish_fisher_quantile(-1.6449, 0.0, 0.0)
    -1.6449
    """
    return (z + (z * z - 1.0) * skew / 6.0
            + (z ** 3 - 3.0 * z) * excess_kurtosis / 24.0
            - (2.0 * z ** 3 - 5.0 * z) * skew * skew / 36.0)


def _cornish_fisher_tail_mean(z: float, skew: float, excess_kurtosis: float,
                              alpha: float) -> float:
    """E[g(Z) | Z <= z] for the Cornish-Fisher transform g, closed form.

    With g(Z) = Z + (Z^2-1)S/6 + (Z^3-3Z)K/24 - (2Z^3-5Z)S^2/36 and the
    conditional normal moments below z (a = phi(z)/alpha):

        E[Z | Z<=z] = -a,  E[Z^2 | .] = 1 - z a,  E[Z^3 | .] = -(z^2 + 2) a

    the tail mean collapses to

        -a [1 + z S/6 - (1 - z^2) K/24 + (1 - 2z^2) S^2/36]

    which is exactly -phi(z)/alpha, the normal result, when S = K = 0.
    """
    a = norm_pdf(z) / alpha
    return -a * (1.0 + z * skew / 6.0
                 - (1.0 - z * z) * excess_kurtosis / 24.0
                 + (1.0 - 2.0 * z * z) * skew * skew / 36.0)


# --------------------------------------------------------------------------
# VaR
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VarEstimate:
    method: str             # "historical" | "parametric" | "cornish_fisher"
    confidence: float       # 0.95, 0.99
    horizon_days: int
    loss: float             # positive fraction: 0.031 means a 3.1% loss
    expected_shortfall: float
    observations: int


def _clean(daily: pd.Series) -> np.ndarray:
    return daily.dropna().astype(float).to_numpy()


def _check(confidence: float, horizon_days: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")


def _overlapping_returns(r: np.ndarray, horizon_days: int) -> np.ndarray:
    """Compounded return of every window of `horizon_days` consecutive days."""
    if horizon_days == 1:
        return r
    growth = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    return growth[horizon_days:] / growth[:-horizon_days] - 1.0


def _compound(block: np.ndarray) -> float:
    """Compounded return of one block of daily returns.

    A single day is returned as itself rather than as (1 + r) - 1, which
    would turn -0.05 into -0.050000000000000044: the day's return is a
    figure the reader can see in the price history, and it should match.
    """
    if block.size == 1:
        return float(block[0])
    return float(np.prod(1.0 + block) - 1.0)


def historical_var(daily: pd.Series, confidence: float = 0.95,
                   horizon_days: int = 1) -> VarEstimate:
    """Empirical VaR: minus the (1 - confidence) quantile of the returns.

    The quantile is numpy's default, linear interpolation between order
    statistics. Expected shortfall is minus the mean of every return at or
    below that quantile. For a horizon above one day the returns are the
    overlapping compounded windows described in the module docstring.

    >>> import pandas as pd
    >>> r = pd.Series([-0.10, -0.05] + [0.01] * 19)          # 21 values
    >>> v = historical_var(r, 0.95)
    >>> round(v.loss, 6), round(v.expected_shortfall, 6)     # 2nd lowest; mean of the two
    (0.05, 0.075)
    """
    _check(confidence, horizon_days)
    windows = _overlapping_returns(_clean(daily), horizon_days)
    if windows.size < 2:
        raise ValueError(f"historical VaR over {horizon_days} day(s) needs at least "
                         f"{horizon_days + 1} returns, got {_clean(daily).size}")
    quantile = float(np.quantile(windows, 1.0 - confidence))
    tail = windows[windows <= quantile]
    return VarEstimate("historical", confidence, horizon_days, -quantile,
                       -float(tail.mean()), int(windows.size))


def parametric_var(daily: pd.Series, confidence: float = 0.95,
                   horizon_days: int = 1) -> VarEstimate:
    """Normal VaR from the sample mean and standard deviation (ddof=1).

        loss = z sigma sqrt(h) - mu h,            z = norm_ppf(confidence)
        ES   = sigma sqrt(h) phi(z) / (1 - c) - mu h

    >>> import pandas as pd
    >>> r = pd.Series([0.01, -0.01] * 100)                  # mean 0
    >>> v = parametric_var(r, 0.95)
    >>> round(v.loss / float(r.std(ddof=1)), 4)              # = z_0.95
    1.6449
    """
    _check(confidence, horizon_days)
    r = _clean(daily)
    if r.size < 2:
        raise ValueError(f"parametric VaR needs at least 2 returns, got {r.size}")
    mu = float(r.mean()) * horizon_days
    sigma = float(r.std(ddof=1)) * math.sqrt(horizon_days)
    z = norm_ppf(confidence)
    loss = z * sigma - mu
    shortfall = sigma * norm_pdf(z) / (1.0 - confidence) - mu
    return VarEstimate("parametric", confidence, horizon_days, loss, shortfall, int(r.size))


def cornish_fisher_var(daily: pd.Series, confidence: float = 0.95,
                       horizon_days: int = 1) -> VarEstimate:
    """Parametric VaR with the quantile corrected for skew and kurtosis.

        loss = -(mu h + sigma sqrt(h) z_cf),   z_cf from cornish_fisher_quantile
               at z = norm_ppf(1 - c), S / sqrt(h), K / h
        ES   = -(mu h + sigma sqrt(h) E[g(Z) | Z <= z])   (closed form, see
               _cornish_fisher_tail_mean)

    With zero skew and excess kurtosis this is exactly `parametric_var`.
    """
    _check(confidence, horizon_days)
    r = _clean(daily)
    if r.size < 2:
        raise ValueError(f"Cornish-Fisher VaR needs at least 2 returns, got {r.size}")
    mu = float(r.mean()) * horizon_days
    sigma = float(r.std(ddof=1)) * math.sqrt(horizon_days)
    skew, kurt = sample_skew_kurtosis(r)
    skew /= math.sqrt(horizon_days)
    kurt /= horizon_days
    alpha = 1.0 - confidence
    z = norm_ppf(alpha)
    loss = -(mu + sigma * cornish_fisher_quantile(z, skew, kurt))
    shortfall = -(mu + sigma * _cornish_fisher_tail_mean(z, skew, kurt, alpha))
    return VarEstimate("cornish_fisher", confidence, horizon_days, loss, shortfall, int(r.size))


_METHODS = (historical_var, parametric_var, cornish_fisher_var)


def var_report(daily: pd.Series, confidences=(0.95, 0.99), horizons=(1, 10)
               ) -> list[VarEstimate]:
    """Every method at every confidence and horizon, in that nesting order."""
    out: list[VarEstimate] = []
    for confidence in confidences:
        for horizon in horizons:
            for method in _METHODS:
                out.append(method(daily, confidence, horizon))
    return out


# --------------------------------------------------------------------------
# Worst periods
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class WorstPeriod:
    start: dt.date
    end: dt.date
    length_days: int
    ret: float


def worst_periods(daily: pd.Series, length_days: int = 1, n: int = 5) -> list[WorstPeriod]:
    """The n worst non-overlapping windows of `length_days` trading days.

    Every window's compounded return is computed, then windows are taken
    from the worst upward, skipping any that shares a day with one already
    taken -- otherwise a single crash would fill the list five times over as
    five windows shifted by a day. `start` and `end` are the dates of the
    first and last return in the window.
    """
    if length_days < 1:
        raise ValueError(f"length_days must be at least 1, got {length_days}")
    r = daily.dropna().astype(float).sort_index()
    values = r.to_numpy()
    count = values.size - length_days + 1
    if count <= 0 or n <= 0:
        return []
    windows = _overlapping_returns(values, length_days)
    taken: list[tuple[int, int]] = []
    out: list[WorstPeriod] = []
    for i in np.argsort(windows, kind="stable"):
        i = int(i)
        lo, hi = i, i + length_days - 1
        if any(lo <= t_hi and t_lo <= hi for t_lo, t_hi in taken):
            continue
        taken.append((lo, hi))
        out.append(WorstPeriod(r.index[lo].date(), r.index[hi].date(), length_days,
                               float(windows[i])))
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------
# Stress scenarios
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    description: str
    portfolio_return: float
    per_holding: dict[str, float]     # ISIN -> return applied


def _historical_window(frame: pd.DataFrame, portfolio: np.ndarray, length: int,
                       key: str, label: str) -> Scenario | None:
    """The worst `length`-day window the portfolio actually had."""
    if portfolio.size < length:
        return None
    windows = _overlapping_returns(portfolio, length)
    i = int(np.argmin(windows))
    block = frame.iloc[i:i + length]
    per_holding = {str(c): _compound(block[c].to_numpy()) for c in frame.columns}
    start, end = frame.index[i].date(), frame.index[i + length - 1].date()
    when = (f"on {start.isoformat()}" if length == 1
            else f"from {start.isoformat()} to {end.isoformat()}")
    return Scenario(
        key, label,
        f"What happened {when}, the worst {length}-day stretch in the window, "
        f"applied to today's weights. Each holding's own move over those days "
        f"is shown; the portfolio figure compounds the daily weighted returns.",
        float(windows[i]), per_holding)


def stress_scenarios(returns: pd.DataFrame, weights: dict[str, float]) -> list[Scenario]:
    """Historical and hypothetical shocks applied to the current weights.

    Historical (from the return window):
        worst day, worst 5 days, worst 21 days of the *portfolio*, each
        holding shown with its own move over the same dates. A window
        shorter than the scenario is skipped rather than approximated.
    Hypothetical:
        every holding repeats its own worst day at once; every holding -10%;
        the largest holding -25% alone; and "correlation goes to 1", in
        which every holding falls by its own one-day 99% parametric loss
        (2.33 standard deviations, drift ignored) on the same day. The last
        is the honest stress: it is what the measured correlations are
        protecting against, and the description quotes the diversified
        figure beside it.

    Every hypothetical portfolio return is the weighted sum of the
    per-holding shocks. Weights are applied by name.
    """
    frame = returns.astype(float).dropna(how="any")
    if frame.shape[1] == 0:
        return []
    columns = [str(c) for c in frame.columns]
    portfolio = portfolio_return_series(frame, weights).to_numpy()
    w = np.array([float(weights[c]) for c in frame.columns], dtype=float)
    out: list[Scenario] = []

    for length, key, label in ((1, "worst_day", "Worst day"),
                               (5, "worst_5d", "Worst 5 days"),
                               (21, "worst_21d", "Worst 21 days")):
        scenario = _historical_window(frame, portfolio, length, key, label)
        if scenario is not None:
            out.append(scenario)

    if frame.shape[0] >= 1:
        own_worst = {c: float(frame[c].min()) for c in columns}
        out.append(Scenario(
            "own_worst_day", "Every holding repeats its worst day",
            "Each holding's single worst day in the window, all on the same day. "
            "They did not happen together, which is why this is worse than any "
            "day the portfolio actually had.",
            float(np.dot(w, list(own_worst.values()))), own_worst))

    out.append(Scenario(
        "all_minus_10", "Every holding -10%",
        "A uniform 10% fall across the book: the loss of a broad sell-off in "
        "which nothing diversifies anything.",
        float(-0.10 * w.sum()), {c: -0.10 for c in columns}))

    largest = columns[int(np.argmax(w))]
    out.append(Scenario(
        "largest_minus_25", "Largest holding -25% alone",
        f"{largest}, the largest position at {w.max():.1%} of the book, falls by a "
        f"quarter while everything else is unchanged: the cost of one holding's "
        f"specific risk.",
        float(-0.25 * w.max()), {c: (-0.25 if c == largest else 0.0) for c in columns}))

    if frame.shape[0] >= 2:
        z = norm_ppf(0.99)
        sigma = frame.std(ddof=1)
        shocks = {c: float(-z * sigma[c]) for c in columns}
        sigma_p = float(np.std(portfolio, ddof=1))
        out.append(Scenario(
            "correlation_one", "Correlation goes to 1",
            f"Every holding falls by its own one-day 99% parametric loss ({z:.2f} "
            f"standard deviations at its current volatility, drift ignored) on the "
            f"same day, which is what a correlation of 1 means. With the measured "
            f"correlations the same one-day 99% loss is {z * sigma_p:.1%}; the "
            f"difference is what diversification is currently worth.",
            float(np.dot(w, list(shocks.values()))), shocks))
    return out
