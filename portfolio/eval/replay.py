"""Replaying real purchases with only the destination changed.

Why this and not a Sharpe ratio
-------------------------------
The buy-only allocator's claim is about risk structure, not return. It has no
view on which holding will do well and neither does anyone here. Evaluating
it on returns would be answering a question it never asked, and answering it
badly: step 3 established that one year of this book cannot distinguish a
Sharpe of 1.60 from 1.69, so a return test would come back "cannot tell"
whatever the allocator did.

So the experiment holds the dates and the amounts fixed -- the real purchases,
on the real days, for the real money -- and changes only where the money went.
Two arms, same cash flows:

    actual        what was bought
    allocator     what the allocator would have said, same day, same amount

Both arms buy the same amount on the same day, so neither pays extra turnover
and the cost model barely enters. That is what makes the comparison clean: it
isolates the destination, which is the only thing the policy chooses.

What the two arms are measured on
---------------------------------
*Risk-share dispersion* is computed from the holdings and the covariance
matrix. It is not estimated from a sample of returns, so the comparison
carries no sampling error in the quantity that matters. If the allocator's
arm has lower dispersion, it has lower dispersion.

*Realised volatility* is estimated, and carries its standard error like
everything else here. With a handful of purchases it will not resolve, and
the report says so rather than ranking two numbers a sample cannot separate.

The primary claim -- that directing money at the underweight holding lowers
dispersion -- is close to arithmetic rather than an empirical bet. The replay
is here to measure the size of the effect, not to establish its sign.

Point in time
-------------
The covariance is estimated from prices strictly before the purchase date and
the order is filled at that day's close, mirroring the harness: you know
yesterday's covariance and today's price when you place the order. Estimating
on the day of the trade would be a one-bar peek -- small, since it moves only
the destination, and exactly the kind of small that this project has twice
found to be load-bearing.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from ..agents.allocate import allocate_buy_only, dispersion_of
from ..core.returns import TRADING_DAYS_PER_YEAR
from ..core.risk import covariance_matrix

__all__ = ["Arm", "ReplayResult", "replay_purchases"]


@dataclasses.dataclass(frozen=True)
class Arm:
    """One way the same money could have been spent."""
    name: str
    dispersion: pd.Series               # through time
    returns: pd.Series                  # daily, from the arm's own holdings
    final_shares: "dict[str, float]"
    invested: float                     # cash actually put in, across the replay

    @property
    def volatility(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        return float(self.returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def volatility_standard_error(self) -> "float | None":
        """SE(sigma) ~ sigma / sqrt(2T), which assumes normal returns.

        Stated rather than hidden: daily equity returns are fat-tailed, so the
        true error is wider than this and the comparison is if anything less
        conclusive than the number suggests. That is the safe direction for a
        figure whose job is to stop two arms being ranked.
        """
        n = len(self.returns)
        if n < 2:
            return None
        return float(self.volatility / math.sqrt(2.0 * n))

    @property
    def mean_dispersion(self) -> float:
        return float(self.dispersion.mean()) if len(self.dispersion) else 0.0

    @property
    def final_dispersion(self) -> float:
        return float(self.dispersion.iloc[-1]) if len(self.dispersion) else 0.0


@dataclasses.dataclass(frozen=True)
class ReplayResult:
    """The two arms, and what may honestly be concluded from the pair."""
    actual: Arm
    allocated: Arm
    purchases: int
    total_invested: float
    carried: float                      # unspent at the end of the replay
    skipped: tuple[str, ...]

    @property
    def dispersion_gap(self) -> float:
        """Mean dispersion, actual minus allocator. Positive favours the tool."""
        return float(self.actual.mean_dispersion - self.allocated.mean_dispersion)

    def lines(self) -> list[str]:
        out = [
            "Replaying real purchases, destination changed",
            "=" * 64,
            "",
            f"{self.purchases} purchases, {self.total_invested:,.0f} EUR, the "
            f"same amounts on the same days in both arms. Neither arm pays "
            f"extra turnover, so this isolates the destination.",
            f"The allocator arm put in {self.allocated.invested:,.0f} EUR of "
            f"that and still holds {self.carried:,.0f} EUR it could not turn "
            f"into whole shares; the two together are the same money, not a "
            f"different bet.",
            "",
            f"{'':24}{'actual':>14}{'allocator':>14}",
            "-" * 64,
            f"{'mean dispersion':24}{self.actual.mean_dispersion:>14.4f}"
            f"{self.allocated.mean_dispersion:>14.4f}",
            f"{'final dispersion':24}{self.actual.final_dispersion:>14.4f}"
            f"{self.allocated.final_dispersion:>14.4f}",
            f"{'annualised volatility':24}{self.actual.volatility:>14.4f}"
            f"{self.allocated.volatility:>14.4f}",
        ]
        se_a = self.actual.volatility_standard_error
        se_b = self.allocated.volatility_standard_error
        if se_a is not None and se_b is not None:
            out.append(f"{'  1 standard error':24}{se_a:>14.4f}{se_b:>14.4f}")
        out += [
            "",
            f"Dispersion is computed from the holdings and the covariance "
            f"matrix, not estimated from a sample, so the "
            f"{self.dispersion_gap:+.4f} difference in the mean carries no "
            f"sampling error. It is the size of the effect, not evidence for "
            f"its sign: directing money at an underweight holding lowers "
            f"dispersion close to arithmetically.",
        ]
        if se_a is not None:
            gap = abs(self.actual.volatility - self.allocated.volatility)
            out.append(
                f"Volatility is estimated and does carry error. The two arms "
                f"differ by {gap:.4f} against a standard error near "
                f"{se_a:.4f} on each, and the two series are nearly the same "
                f"portfolio, so this is not a ranking. With {self.purchases} "
                f"purchases it was never going to be.")
        if self.skipped:
            out += ["", "Purchases the allocator could not replay", "-" * 64]
            out += [f"  - {s}" for s in self.skipped]
        return out


def replay_purchases(closes: pd.DataFrame, purchases, *, costs,
                     buyable, lookback: int = 252, warmup: int = 60,
                     among=None) -> ReplayResult:
    """Run both arms through the same cash flows.

    `purchases` is an iterable of (date, isin, shares) -- the real ledger,
    with quantities rather than amounts, because the amount is the quantity
    times that day's price and using a recorded amount at a different price
    would put the two arms on different money.
    """
    keys = [str(c) for c in closes.columns]
    dates = closes.index
    filled = closes.ffill()

    by_date: "dict[pd.Timestamp, list[tuple[str, float]]]" = {}
    for when, isin, shares in purchases:
        stamp = pd.Timestamp(when)
        if stamp not in dates:
            later = dates[dates >= stamp]
            if not len(later):
                continue
            stamp = later[0]
        by_date.setdefault(stamp, []).append((str(isin), float(shares)))

    actual = {k: 0.0 for k in keys}
    model = {k: 0.0 for k in keys}
    rows_a, rows_m, index = [], [], []
    skipped: list[str] = []
    count, invested, invested_model = 0, 0.0, 0.0
    # Whole shares leave a remainder on every purchase, and over a ledger it
    # compounds: on the fixture in `test_replay.py` it reached 6.5% of the
    # money, which would leave one arm quietly holding cash and make the two
    # incomparable. Carrying it to the next purchase is both the honest
    # bookkeeping and what a person with 125 EUR left over actually does.
    carry = 0.0

    for position, day in enumerate(dates):
        price = filled.loc[day]
        for isin, shares in by_date.get(day, []):
            if isin not in actual or not np.isfinite(price.get(isin, np.nan)):
                skipped.append(f"{day:%Y-%m-%d} {isin}: no price in the panel")
                continue
            cash = float(shares) * float(price[isin])
            actual[isin] += float(shares)
            count += 1
            invested += cash
            budget = cash + carry

            # Decide on data strictly before today, fill at today's close.
            history = closes.iloc[max(0, position - lookback):position]
            if position < warmup or len(history) < warmup:
                model[isin] += float(shares)
                invested_model += cash
                skipped.append(f"{day:%Y-%m-%d} {isin}: only {len(history)} "
                               f"rows of history, too few to estimate a "
                               f"covariance; the actual purchase was copied")
                continue
            window = history.pct_change().dropna(how="any")
            cov = covariance_matrix(window)
            values = {k: model[k] * float(price[k]) for k in keys
                      if np.isfinite(price.get(k, np.nan))}
            prices = {k: float(price[k]) for k in keys
                      if np.isfinite(price.get(k, np.nan)) and price[k] > 0}
            allocation = allocate_buy_only(
                values=values, prices=prices, cov=cov, cash=budget, costs=costs,
                buyable=set(buyable) & set(cov.columns), among=among)
            if not allocation.purchases:
                model[isin] += float(shares)
                invested_model += cash
                skipped.append(f"{day:%Y-%m-%d}: the allocator proposed "
                               f"nothing for {cash:,.0f} EUR; the actual "
                               f"purchase was copied")
                continue
            for bought in allocation.purchases:
                model[bought.isin] += bought.shares
            invested_model += allocation.invested
            carry = float(allocation.leftover)

        value_a = sum(actual[k] * float(price[k]) for k in keys
                      if np.isfinite(price.get(k, np.nan)))
        value_m = sum(model[k] * float(price[k]) for k in keys
                      if np.isfinite(price.get(k, np.nan)))
        if value_a <= 0 or value_m <= 0:
            continue
        index.append(day)
        rows_a.append((value_a, {k: actual[k] * float(price[k]) for k in keys}))
        rows_m.append((value_m, {k: model[k] * float(price[k]) for k in keys}))

    stamps = pd.DatetimeIndex(index)
    window = closes.pct_change()

    def arm(name, rows, shares, spent) -> Arm:
        values = pd.Series([v for v, _ in rows], index=stamps)
        dispersion = pd.Series(
            [dispersion_of(np.array([holdings.get(k, 0.0) for k in keys]),
                           covariance_matrix(
                               closes.loc[:day].iloc[-lookback:]
                               .pct_change().dropna(how="any")), among)
             for day, (_, holdings) in zip(stamps, rows)], index=stamps)
        return Arm(name=name, dispersion=dispersion,
                   returns=values.pct_change().dropna(), final_shares=shares,
                   invested=float(spent))

    return ReplayResult(
        actual=arm("actual", rows_a, dict(actual), invested),
        allocated=arm("allocator", rows_m, dict(model), invested_model),
        purchases=count, total_invested=invested, carried=carry,
        skipped=tuple(dict.fromkeys(skipped)))
