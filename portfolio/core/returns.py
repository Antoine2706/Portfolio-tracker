"""Price series to an aligned return matrix.

This is the step most likely to be silently wrong, so almost all of the code
here is about reporting what happened rather than computing anything.

The holdings trade on Xetra, Amsterdam, Milan and Paris. Those calendars do not
agree -- 15 August is a holiday in Milan and Paris but a trading day in
Frankfurt and Amsterdam -- so the price series do not share dates. Take the
intersection and you lose rows; fail to take it and you silently pair Monday's
price for one fund with Tuesday's for another, which fabricates correlation.

Two findings from the 3 September 2026 provider matrix shape the design:

1. **Thin series look identical to good ones.** AIGG.L and AIGE.L both resolve
   on a European venue in the right currency and return exactly 2 rows. Nothing
   about the call fails. Intersecting a 2-row series with nine 500-row series
   yields a 2-row matrix and a covariance estimate built from one return, which
   is worse than useless because it still produces a number. So instruments
   below `min_observations` are EXCLUDED before the intersection is taken, and
   named in the report.

2. **A short window is the expected state, not an edge case.** The defence ETFs
   are recently launched: DFNC.DE has 320 trading days, 8RMY.DE 356, EUDF.DE
   377, against ~500 for everything else. Any portfolio holding one of them
   caps the intersection near 320 before holidays take more. So the window
   degrades gracefully rather than failing, but never silently: the report
   names the binding instrument and the actual window used, and the UI is
   expected to show both beside the risk numbers.

Simple returns, not log returns
-------------------------------
r_t = P_t / P_{t-1} - 1, because simple returns aggregate correctly across
assets: the return of a portfolio is the weighted sum of its holdings' simple
returns, which is exactly what the risk decomposition needs. Log returns do not
have that property across assets.

Log returns would be preferable for aggregating over *time* -- they sum across
periods, so a multi-period compound return is a plain sum -- and for assuming
normality. If a time-aggregation feature is added later (annualised trailing
returns, say), convert with log(1 + r) at that point rather than switching the
matrix that feeds the covariance.

Adjusted close
--------------
The caller must pass adjusted closes. Unadjusted prices show a distribution as
a price drop, which registers as a large negative return and inflates measured
volatility; several of these instruments distribute. This module cannot detect
that, which is why the provider layer checks for an adjusted-close field.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_LOOKBACK", "MIN_OBSERVATIONS", "TRADING_DAYS_PER_YEAR",
    "InsufficientHistory", "ExcludedInstrument", "AlignmentReport",
    "align_returns", "simple_returns",
]

# Configurable, per the matrix finding. 252 is the conventional count of
# trading days in a year and is the default; it is not a floor.
DEFAULT_LOOKBACK = 252

# Below this many aligned observations a covariance estimate is not worth
# reporting: with 60 daily returns the standard error on a correlation is
# already around 0.13. Refusing is better than returning a confident number.
MIN_OBSERVATIONS = 60

TRADING_DAYS_PER_YEAR = 252


class InsufficientHistory(ValueError):
    """Not enough aligned history to compute a covariance matrix.

    Raised rather than returning a degraded number, because the spec's rule is
    to say which instrument is the problem instead of quietly producing
    something that looks like an answer.
    """


@dataclasses.dataclass(frozen=True)
class ExcludedInstrument:
    """An instrument left out of the matrix, and why."""
    isin: str
    observations: int
    reason: str


@dataclasses.dataclass(frozen=True)
class AlignmentReport:
    """Everything the UI needs to say how the risk numbers were produced.

    This is not diagnostics. The spec requires the effective window and the
    instrument that constrained it to appear next to the risk figures, because
    a volatility computed over 300 days and one computed over 252 are different
    claims and the user has to be able to tell them apart.
    """
    instruments: tuple[str, ...]
    requested_lookback: int
    effective_lookback: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    raw_observations: dict[str, int]
    dropped_in_alignment: dict[str, int]
    binding_instrument: str | None
    binding_observations: int | None
    excluded: tuple[ExcludedInstrument, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def window_was_shortened(self) -> bool:
        return self.effective_lookback < self.requested_lookback

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped_in_alignment.values())

    def summary(self) -> str:
        """One line for display beside the risk numbers."""
        base = (f"{self.effective_lookback} daily returns "
                f"({self.first_date:%Y-%m-%d} to {self.last_date:%Y-%m-%d})"
                if self.first_date is not None else "no aligned history")
        if self.window_was_shortened and self.binding_instrument:
            base += (f", short of the {self.requested_lookback} requested; "
                     f"constrained by {self.binding_instrument} "
                     f"({self.binding_observations} observations)")
        if self.excluded:
            base += f"; {len(self.excluded)} instrument(s) excluded"
        return base


def simple_returns(prices: "pd.DataFrame | pd.Series"):
    """r_t = P_t / P_{t-1} - 1, dropping the undefined first row.

    Accepts a Series as well as a frame so a single benchmark series can be
    converted here rather than in a view. Returning the same type it was given
    keeps the caller from having to unwrap anything.

    >>> import pandas as pd
    >>> px = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    >>> [round(r, 10) for r in simple_returns(px)["A"]]
    [0.1, -0.1]
    >>> [round(r, 10) for r in simple_returns(pd.Series([100.0, 110.0, 99.0]))]
    [0.1, -0.1]
    """
    return prices.astype(float).pct_change().iloc[1:]


def _clean(series: pd.Series) -> pd.Series:
    """Sorted, de-duplicated, NaN-free, with a real DatetimeIndex."""
    s = series.dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def align_returns(prices: dict[str, pd.Series],
                  lookback: int = DEFAULT_LOOKBACK,
                  min_observations: int = MIN_OBSERVATIONS,
                  ) -> tuple[pd.DataFrame, AlignmentReport]:
    """Build the aligned daily return matrix, with a full account of the cost.

    Returns the matrix (rows = dates, columns = ISINs) and a report naming what
    was excluded, how many observations alignment cost, and which instrument
    constrained the window.

    Raises `InsufficientHistory` when no usable matrix can be built, rather than
    returning a small one.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be at least 2, got {lookback}")

    cleaned = {isin: _clean(s) for isin, s in prices.items()}
    raw_counts = {isin: len(s) for isin, s in cleaned.items()}

    if not cleaned:
        raise InsufficientHistory(
            "no price series supplied; a portfolio with no holdings has no "
            "covariance matrix")

    # Step 1: drop the THIN ones BEFORE intersecting. A 2-row series would
    # otherwise truncate every other instrument to 2 rows and still yield a
    # number. Excluding is explicit and recoverable; contaminating is neither.
    excluded: list[ExcludedInstrument] = []
    usable: dict[str, pd.Series] = {}
    for isin, s in cleaned.items():
        if len(s) < min_observations:
            excluded.append(ExcludedInstrument(
                isin=isin, observations=len(s),
                reason=(f"only {len(s)} observations, below the {min_observations} "
                        f"minimum; excluded so it cannot truncate the others")))
        else:
            usable[isin] = s

    if not usable:
        raise InsufficientHistory(
            "every instrument has too little history for a covariance matrix: "
            + "; ".join(f"{e.isin} ({e.observations} obs)" for e in excluded))

    # Step 2: intersect the trading calendars of what remains.
    common: pd.DatetimeIndex | None = None
    for s in usable.values():
        common = s.index if common is None else common.intersection(s.index)
    assert common is not None
    common = common.sort_values()

    if len(common) < min_observations:
        # Name the culprit rather than returning a number.
        starts = {isin: s.index.min() for isin, s in usable.items()}
        latest_start = max(starts, key=lambda k: starts[k])
        raise InsufficientHistory(
            f"only {len(common)} dates are common to all {len(usable)} "
            f"instruments, below the {min_observations} minimum. The shortest "
            f"history belongs to {latest_start}, starting "
            f"{starts[latest_start]:%Y-%m-%d}. Exclude it, or shorten the "
            f"holding period under analysis.")

    # Step 3: take the window. N returns need N+1 prices.
    wanted_rows = min(lookback + 1, len(common))
    window = common[-wanted_rows:]

    frame = pd.DataFrame({isin: usable[isin].reindex(window)
                          for isin in usable}, index=window)
    returns = simple_returns(frame)

    dropped = {isin: raw_counts[isin] - len(window) for isin in usable}

    # Step 4: name what constrained the window. The binding instrument is the
    # one whose own history starts latest -- it is what truncates the
    # intersection at the front.
    binding = None
    binding_obs = None
    if len(returns) < lookback:
        starts = {isin: s.index.min() for isin, s in usable.items()}
        binding = max(starts, key=lambda k: starts[k])
        binding_obs = raw_counts[binding]

    warnings: list[str] = []
    if len(returns) < lookback:
        warnings.append(
            f"Window shortened to {len(returns)} returns from the requested "
            f"{lookback}. Constrained by {binding}, which has only "
            f"{binding_obs} observations. The same window is used for every "
            f"instrument -- window lengths are never mixed.")
    for e in excluded:
        warnings.append(f"{e.isin} excluded: {e.reason}")
    total_dropped = sum(dropped.values())
    if total_dropped:
        warnings.append(
            f"Calendar alignment discarded {total_dropped} observations across "
            f"{len(usable)} instruments (non-overlapping trading days).")

    report = AlignmentReport(
        instruments=tuple(returns.columns),
        requested_lookback=lookback,
        effective_lookback=len(returns),
        first_date=returns.index[0] if len(returns) else None,
        last_date=returns.index[-1] if len(returns) else None,
        raw_observations=raw_counts,
        dropped_in_alignment=dropped,
        binding_instrument=binding,
        binding_observations=binding_obs,
        excluded=tuple(excluded),
        warnings=tuple(warnings),
    )
    return returns, report
