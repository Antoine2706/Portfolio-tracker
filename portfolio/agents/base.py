"""The contract every policy implements, and the world it is allowed to see.

An "agent" here is a policy object with one method. It is not a language
model, nothing in this package makes a network call, and a decision costs
nothing. The word is used because the referee combines several of them and
they disagree.

Two types carry the whole contract:

`MarketView`   what a policy may look at. Constructed by the harness for one
               decision date and containing *only* data up to that date. The
               future is not hidden behind a check that could be forgotten --
               it is not in the object.

`Proposal`     what a policy returns: target weights, a confidence, and a
               written reason. The reason is required and validated. A month
               of paper decisions is only auditable if each one says why it
               was made, and a field that is optional is a field that ends up
               empty.

Why the view lives here rather than in `eval/`
----------------------------------------------
A policy must not know it is being backtested; if it could tell, it could
behave differently in evaluation than in production, which is the single
most expensive bug in this domain. So `agents/` defines both its input and
its output type, `eval/` imports `agents/`, and the arrow never points back.
The layering test enforces it.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = ["MarketView", "Proposal", "Policy", "LookAheadError"]


class LookAheadError(RuntimeError):
    """A policy asked for data at or after its decision date.

    Raised rather than returning an empty frame, because a silent empty is how
    a leak becomes a plausible number instead of a crash.
    """


@dataclasses.dataclass(frozen=True)
class MarketView:
    """The world as known at the close of `as_of`, and not one bar later.

    Built by slicing the price panel, so the frames this object holds simply
    end at the decision date. There is no filtering step to bypass and no flag
    to forget: a policy that wants tomorrow's price has nowhere to get it.

    `held` is the weight vector actually in force immediately before this
    decision -- the previous targets carried forward by realised returns. A
    policy needs it to reason about its own turnover, and giving it here means
    the policy never has to reconstruct the portfolio's history to find out
    what it owns.
    """
    as_of: pd.Timestamp
    _closes: pd.DataFrame                # ends at as_of, inclusive
    held: dict[str, float]
    min_observations: int = 60

    # -- what exists, as of now --------------------------------------------

    @property
    def instruments(self) -> tuple[str, ...]:
        """Every column in the panel, listed or not, alive or not."""
        return tuple(str(c) for c in self._closes.columns)

    @property
    def available(self) -> tuple[str, ...]:
        """Instruments tradable *today* with enough history to model.

        Computed from the point-in-time panel rather than from a list of what
        exists at the end of the sample, which is what makes the backtest
        survivorship-honest in both directions: an instrument that has not
        listed yet is absent from early views, and one that has been delisted
        is absent from late ones while remaining present in the earlier views
        where it really was tradable.
        """
        out = []
        for col in self._closes.columns:
            series = self._closes[col]
            if pd.isna(series.iloc[-1]):
                continue                 # not listed yet, or already gone
            if int(series.notna().sum()) < self.min_observations:
                continue
            out.append(str(col))
        return tuple(out)

    # -- prices and returns ------------------------------------------------

    def closes(self, lookback: int | None = None,
               instruments: "tuple[str, ...] | list[str] | None" = None) -> pd.DataFrame:
        """Adjusted closes up to and including the decision date.

        `lookback` counts rows back from the decision date. Asking for more
        than exists returns what exists; asking for a non-positive number is
        a programming error and raises.
        """
        if lookback is not None and lookback < 1:
            raise ValueError(f"lookback must be at least 1, got {lookback}")
        frame = self._closes if instruments is None else self._closes[list(instruments)]
        return frame if lookback is None else frame.iloc[-lookback:]

    def returns(self, lookback: int | None = None,
                instruments: "tuple[str, ...] | list[str] | None" = None) -> pd.DataFrame:
        """Simple daily returns over dates where every column really printed.

        `lookback` is the number of *returns* wanted, so it asks for one more
        row of prices. Getting that off by one is how a window silently
        becomes 251 days long, so it is done here once rather than at each
        call site.

        A return is admitted only when both of its endpoint prices were
        observed, on consecutive panel dates. Dropping the missing row alone
        is not enough, and that is what this used to do: the next row's
        `pct_change` then reached back over the gap and entered the sample as
        a one-day move when it was two days of one.

        A two-day return carries twice the variance of a one-day one, so in
        expectation each such row inflates the estimate by roughly 1/n -- but
        on any single sample it can land either way, since the two days may
        offset as easily as reinforce. That is the case for excluding it
        rather than correcting it: the number is not biased in a direction
        anyone could reason about, it simply is not the quantity being
        estimated.

        Complete-case rather than pairwise: a covariance matrix assembled from
        pairs with different sample sets can fail to be positive semi-definite,
        and an optimiser handed such a matrix returns a confident answer to a
        problem that has none. The cost is that one instrument's holiday
        removes that date for every pair, which is a few rows out of 252.
        """
        rows = None if lookback is None else lookback + 1
        prices = self.closes(rows, instruments)
        if len(prices) < 2:
            return prices.iloc[0:0]
        printed = prices.notna().all(axis=1)
        admitted = (printed & printed.shift(1, fill_value=False)).to_numpy()
        return prices.pct_change()[admitted]

    def covariance(self, lookback: int,
                   instruments: "tuple[str, ...] | list[str] | None" = None) -> pd.DataFrame:
        """Sample covariance of daily returns over the trailing window.

        Delegates to `core.risk.covariance_matrix` so the backtest and the
        live risk page cannot disagree about what a covariance matrix is.
        """
        from ..core.risk import covariance_matrix
        return covariance_matrix(self.returns(lookback, instruments))

    def last_price(self, instrument: str) -> float | None:
        value = self._closes[instrument].iloc[-1]
        return None if pd.isna(value) else float(value)

    def __len__(self) -> int:
        return int(len(self._closes))


@dataclasses.dataclass(frozen=True)
class Proposal:
    """Target weights from one policy, with its reasoning attached.

    Long-only and unlevered by construction: weights are non-negative and sum
    to at most 1, with the remainder held as cash. That is not a placeholder
    for a margin model, it is the constraint this portfolio actually operates
    under, and volatility targeting needs the cash leg to scale exposure down
    into.

    >>> p = Proposal({"A": 0.6, "B": 0.4}, 0.5, "equal risk", "risk")
    >>> round(p.cash, 10)
    0.0
    >>> Proposal({"A": 0.3}, 0.5, "de-risked", "voltarget").cash
    0.7

    The reason is validated, not merely encouraged:

    >>> Proposal({"A": 1.0}, 0.5, "   ", "x")
    Traceback (most recent call last):
        ...
    ValueError: policy 'x' returned a proposal with no reason; a decision that cannot explain itself cannot be audited

    So is the leverage constraint, which is the one that would quietly turn a
    modest backtest into an impressive one:

    >>> Proposal({"A": 0.8, "B": 0.5}, 0.5, "levered", "x")
    Traceback (most recent call last):
        ...
    ValueError: policy 'x' proposed weights summing to 1.3; this portfolio is long-only and unlevered, so they may not exceed 1
    """
    weights: dict[str, float]
    confidence: float
    reason: str
    agent: str

    def __post_init__(self) -> None:
        if not self.reason or not self.reason.strip():
            raise ValueError(
                f"policy {self.agent!r} returned a proposal with no reason; a "
                f"decision that cannot explain itself cannot be audited")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"policy {self.agent!r} gave confidence {self.confidence}, "
                f"which is outside [0, 1]")
        negative = {k: v for k, v in self.weights.items() if v < 0}
        if negative:
            raise ValueError(
                f"policy {self.agent!r} proposed negative weights {negative}; "
                f"this portfolio is long-only")
        total = sum(self.weights.values())
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"policy {self.agent!r} proposed weights summing to "
                f"{total:.6g}; this portfolio is long-only and unlevered, so "
                f"they may not exceed 1")

    @property
    def cash(self) -> float:
        """The unallocated remainder. Earns the risk-free rate, zero by default."""
        return float(max(0.0, 1.0 - sum(self.weights.values())))

    def vector(self, columns) -> np.ndarray:
        """Weights ordered to match a frame's columns, missing ones as zero.

        By name, never by position: a dict iterated in insertion order against
        an alphabetically sorted panel gives a plausible number for the wrong
        portfolio.
        """
        return np.array([float(self.weights.get(str(c), 0.0)) for c in columns],
                        dtype=float)


@runtime_checkable
class Policy(Protocol):
    """What the harness requires of anything it can evaluate."""

    name: str

    def observe(self, view: MarketView) -> Proposal:
        """Look at the world as of the decision date and propose weights."""
        ...
