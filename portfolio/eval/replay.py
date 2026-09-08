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

__all__ = ["Arm", "ReplayResult", "estimable_window", "replay_purchases"]


def estimable_window(closes: pd.DataFrame, position: int, lookback: int, *,
                     minimum: int) -> pd.DataFrame:
    """Returns to estimate a covariance from, as of `position`, strictly before.

    Two filters, and neither is a precaution: without them the replay of the
    demo ledger raised `covariance needs at least 2 return observations, got 0`
    and produced no result at all.

    *Instruments that have not listed yet are dropped, not carried as gaps.*
    The panel's ten instruments start on eight different dates, the earliest in
    September 2024 and the latest in June 2025. Complete-case deletion across
    all ten therefore deletes every date before the last of those, which for
    any window reaching back further is every date in it. Dropping the column
    is also the point-in-time answer: an instrument with no history is not a
    destination the allocator could have chosen, and pretending otherwise would
    be the same peek the harness exists to prevent.

    *A return that spans a gap is not a return.* Dividing today's close by the
    last one that printed produces a two-day move recorded as a one-day move,
    which inflates the variance in expectation and is the defect the harness
    carries its `measured` mask for. A return survives only if both of its
    endpoint prices printed on consecutive panel dates.

    What is left is complete-case over the surviving instruments, which is what
    the equal risk contribution path does, so the two cannot disagree about
    what the book's covariance is.

    Empty when nothing survives, so the caller decides what to do rather than
    being handed an exception mid-ledger.
    """
    history = closes.iloc[max(0, position - lookback):position]
    if len(history) < 2:
        return history.iloc[:0]
    enough = [c for c in history.columns
              if int(history[c].notna().sum()) >= minimum]
    if len(enough) < 2:
        return history.iloc[:0]
    kept = history[enough].astype(float)
    observed = kept.notna()
    spanning = ~(observed & observed.shift(1))
    return kept.pct_change().mask(spanning).iloc[1:].dropna(how="any")


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
    measured: tuple[str, ...]           # instruments the dispersion covers, at the end
    unmeasured: tuple[str, ...]         # in the panel, too little history to
    opened_with: int                    # how many it covered on the first day
    ledger_purchases: int               # purchases in the ledger the panel covers
    withdrawals: int                    # disposals applied to both arms
    on_sale: str                        # "prorata" or "stop"
    stopped_at: "pd.Timestamp | None"   # the first sale, where the replay ends
    skipped: tuple[str, ...]

    @property
    def dispersion_gap(self) -> float:
        """Mean dispersion, actual minus allocator. Positive favours the tool."""
        return float(self.actual.mean_dispersion - self.allocated.mean_dispersion)

    @property
    def correlation(self) -> "float | None":
        """How alike the two arms' returns actually are.

        Measured, not assumed. The report used to assert that the arms were
        "nearly the same portfolio" as its reason for refusing to rank two
        volatilities, which is the same move the risk-adjusted comparison was
        corrected for: rho decides how much of each marginal standard error
        cancels, so it has to be a number on the page rather than an
        adjective. Two arms that share few holdings are not nearly the same
        portfolio and the reader should be able to see that.
        """
        a, b = self.actual.returns.align(self.allocated.returns, join="inner")
        if len(a) < 3:
            return None
        value = float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])
        return value if np.isfinite(value) else None

    def lines(self) -> list[str]:
        out = [
            "Replaying real purchases, destination changed",
            "=" * 64,
            "",
            f"{self.purchases} of the ledger's {self.ledger_purchases} "
            f"purchases, {self.total_invested:,.0f} EUR, the same amounts on "
            f"the same days in both arms. Neither arm pays extra turnover, so "
            f"this isolates the destination.",
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
            f"Both arms are measured on the same day against the same "
            f"covariance, estimated from the window ending that day, so no "
            f"reported number depends on a price that had not printed. It "
            f"decides nothing in either arm either: every purchase was chosen "
            f"on a window strictly *before* its own date.",
        ]
        if self.opened_with != len(self.measured):
            out.append(
                f"The metric covered {self.opened_with} instruments on the "
                f"first day and {len(self.measured)} on the last, because "
                f"instruments list at different dates and one with no history "
                f"has no risk share. A range over "
                f"{self.opened_with} risk shares and a range over "
                f"{len(self.measured)} are not the same statistic, so read the "
                f"final row rather than the mean where the two differ.")
        if self.unmeasured:
            out.append(
                f"Outside the measurement even at the end: "
                f"{', '.join(self.unmeasured)}. Too little price history to "
                f"estimate a covariance, so neither arm's dispersion counts "
                f"{'them' if len(self.unmeasured) > 1 else 'it'}.")
        if se_a is not None:
            gap = abs(self.actual.volatility - self.allocated.volatility)
            rho = self.correlation
            out.append(
                f"Volatility is estimated and does carry error: the two arms "
                f"differ by {gap:.4f} against a standard error near "
                f"{se_a:.4f} on each. Those are MARGINAL errors and the two "
                f"series are correlated at "
                f"{'unknown' if rho is None else format(rho, '.3f')}, so how "
                f"much of each cancels is not read off them -- the same reason "
                f"the policy comparison is made at matched risk rather than on "
                f"a difference of levels. With {self.purchases} purchases this "
                f"is not a ranking, and it was never going to be.")
        if self.purchases < 5:
            out += ["", f"READ THIS AS {self.purchases} POINTS, NOT AS A "
                    f"SERIES. The mean dispersion above is an average over "
                    f"the days between {self.purchases} decisions, and days "
                    f"are not independent observations of a policy -- the "
                    f"holdings only change when a purchase happens. Nothing "
                    f"here is a distribution to test; it is a worked example "
                    f"of what would have been bought."]
        if self.withdrawals:
            out += ["", f"{self.withdrawals} disposal"
                    f"{'s' if self.withdrawals > 1 else ''} in the ledger, "
                    f"applied to BOTH arms as the same proportional "
                    f"withdrawal. Risk shares are homogeneous of degree zero, "
                    f"so a pro-rata withdrawal changes none of them in either "
                    f"arm: it is neutral to the quantity being compared. The "
                    f"cost is that the actual arm is no longer the book that "
                    f"was held -- it is the purchases that were made, with "
                    f"disposals taken pro-rata. Selling a particular holding "
                    f"is a decision of yours that the allocator never makes, "
                    f"and crediting one arm for it would measure that instead "
                    f"of the destination of the purchases. Use "
                    f"`on_sale=\"stop\"` for the strict version, which "
                    f"describes a book that really was held and covers less "
                    f"of the ledger."]
        if self.stopped_at is not None:
            out += ["", f"The replay stops on {self.stopped_at:%Y-%m-%d}, the "
                    f"first sale in the ledger. Both arms have to hold the "
                    f"same money throughout, and there is no neutral way to "
                    f"apply a concentrated sale to the arm that did not make "
                    f"it: pro-rata leaves risk shares untouched, so the "
                    f"allocator arm would carry a free disposal while the "
                    f"actual arm carries one that moves its shares a long "
                    f"way. Everything above covers the span before it."]
        if self.skipped:
            out += ["", "Purchases the allocator could not replay", "-" * 64]
            out += [f"  - {s}" for s in self.skipped]
        return out


def _on_panel(when, dates) -> "pd.Timestamp | None":
    """The panel date a ledger date falls on, or the next one that trades."""
    stamp = pd.Timestamp(when)
    if stamp in dates:
        return stamp
    later = dates[dates >= stamp]
    return later[0] if len(later) else None


def replay_purchases(closes: pd.DataFrame, purchases, *, costs,
                     buyable, lookback: int = 252, warmup: int = 60,
                     among=None, sales=(), on_sale: str = "prorata"
                     ) -> ReplayResult:
    """Run both arms through the same cash flows.

    `purchases` is an iterable of (date, isin, shares) -- the real ledger,
    with quantities rather than amounts, because the amount is the quantity
    times that day's price and using a recorded amount at a different price
    would put the two arms on different money.

    `sales` is the same for disposals, and `on_sale` says what to do with
    them. Neither answer is free, so both are available and the report names
    the one it used.

    "prorata" (the default) applies each disposal to *both* arms as a
    proportional withdrawal of the same euro amount. Risk shares are
    homogeneous of degree zero, so scaling a book down leaves every one of
    them unchanged: the withdrawal is exactly neutral to the quantity being
    compared, in both arms, and the two stay on identical cash flows
    throughout. The whole ledger is then covered.

    What that costs is worth stating plainly, because it is not nothing: the
    actual arm is no longer the book that was held. It is "the purchases that
    were made, with disposals taken pro-rata". Selling a specific holding is
    itself a portfolio decision, and a good or bad one; letting the actual arm
    execute its real concentrated sale would credit or debit the comparison
    for selling skill, which is not what the allocator does and not what this
    experiment asks. Removing that decision from both arms is what makes the
    remaining difference attributable to the destination of the purchases.
    A side effect is fractional share counts after a disposal, since a
    proportional withdrawal is not a whole-share transaction; the purchases on
    top of it are still whole shares.

    "stop" ends the replay at the first disposal instead. Every figure then
    describes a book that really was held, and the window can be much shorter
    -- on a ledger that sells in month five, everything after month five is
    gone.

    Ignoring disposals is the one option that is simply wrong, and it is what
    this did: the demo ledger sells a 500-share position in October and the
    arm labelled *what was bought* went on holding it to the end of the panel,
    so it was a book nobody ever owned.
    """
    keys = [str(c) for c in closes.columns]
    dates = closes.index
    filled = closes.ffill()

    if on_sale not in ("prorata", "stop"):
        raise ValueError(f"on_sale must be 'prorata' or 'stop', not {on_sale!r}")
    sold: "dict[pd.Timestamp, list[tuple[str, float]]]" = {}
    for when, isin, shares in sales:
        stamp = _on_panel(when, dates)
        if stamp is not None:
            sold.setdefault(stamp, []).append((str(isin), float(shares)))
    stop = min(sold) if (sold and on_sale == "stop") else None

    ledger_purchases = 0
    by_date: "dict[pd.Timestamp, list[tuple[str, float]]]" = {}
    for when, isin, shares in purchases:
        stamp = _on_panel(when, dates)
        if stamp is None:
            continue
        ledger_purchases += 1
        if stop is not None and stamp >= stop:
            continue
        by_date.setdefault(stamp, []).append((str(isin), float(shares)))

    actual = {k: 0.0 for k in keys}
    model = {k: 0.0 for k in keys}
    rows_a, rows_m, index = [], [], []
    flows_a: "list[float]" = []
    flows_m: "list[float]" = []
    skipped: list[str] = []
    count, invested, invested_model = 0, 0.0, 0.0
    withdrawals = 0
    # Whole shares leave a remainder on every purchase, and over a ledger it
    # compounds: on the fixture in `test_replay.py` it reached 6.5% of the
    # money, which would leave one arm quietly holding cash and make the two
    # incomparable. Carrying it to the next purchase is both the honest
    # bookkeeping and what a person with 125 EUR left over actually does.
    carry = 0.0

    for position, day in enumerate(dates):
        if stop is not None and day >= stop:
            break
        price = filled.loc[day]
        # Money that arrived today, per arm. Needed because a contribution is
        # not a return and has to be backed out before the series is divided.
        paid_a = paid_m = 0.0
        for isin, shares in by_date.get(day, []):
            if isin not in actual or not np.isfinite(price.get(isin, np.nan)):
                skipped.append(f"{day:%Y-%m-%d} {isin}: no price in the panel")
                continue
            cash = float(shares) * float(price[isin])
            actual[isin] += float(shares)
            count += 1
            invested += cash
            paid_a += cash
            budget = cash + carry

            # Decide on data strictly before today, fill at today's close.
            history = closes.iloc[max(0, position - lookback):position]
            window = estimable_window(closes, position, lookback,
                                      minimum=warmup)
            if position < warmup or len(history) < warmup or len(window) < 2:
                model[isin] += float(shares)
                invested_model += cash
                paid_m += cash
                skipped.append(f"{day:%Y-%m-%d} {isin}: {len(history)} rows of "
                               f"history leave {len(window)} usable return "
                               f"observations across {window.shape[1]} "
                               f"instruments, too few to estimate a covariance; "
                               f"the actual purchase was copied")
                continue
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
                paid_m += cash
                skipped.append(f"{day:%Y-%m-%d}: the allocator proposed "
                               f"nothing for {cash:,.0f} EUR; the actual "
                               f"purchase was copied")
                continue
            for bought in allocation.purchases:
                model[bought.isin] += bought.shares
            invested_model += allocation.invested
            paid_m += float(allocation.invested)
            carry = float(allocation.leftover)

        # Disposals, applied to BOTH arms as the same proportional
        # withdrawal. Risk shares are homogeneous of degree zero, so scaling a
        # book down changes none of them: the withdrawal is neutral to the
        # quantity being compared, in both arms, and neither gains or loses a
        # decision the allocator never makes.
        for isin, shares in sold.get(day, []):
            if isin not in actual or not np.isfinite(price.get(isin, np.nan)):
                skipped.append(f"{day:%Y-%m-%d} {isin}: sold, but no price in "
                               f"the panel, so the withdrawal was not applied")
                continue
            cash_out = float(shares) * float(price[isin])
            withdrawals += 1
            for book, paid in ((actual, "a"), (model, "m")):
                worth = sum(book[k] * float(price[k]) for k in keys
                            if np.isfinite(price.get(k, np.nan)))
                if worth <= 0:
                    continue
                taken = min(cash_out, worth)
                keep = 1.0 - taken / worth
                for k in keys:
                    book[k] *= keep
                if paid == "a":
                    paid_a -= taken
                else:
                    paid_m -= taken

        value_a = sum(actual[k] * float(price[k]) for k in keys
                      if np.isfinite(price.get(k, np.nan)))
        value_m = sum(model[k] * float(price[k]) for k in keys
                      if np.isfinite(price.get(k, np.nan)))
        if value_a <= 0 or value_m <= 0:
            continue
        index.append(day)
        rows_a.append((value_a, {k: actual[k] * float(price[k]) for k in keys}))
        rows_m.append((value_m, {k: model[k] * float(price[k]) for k in keys}))
        flows_a.append(paid_a)
        flows_m.append(paid_m)

    stamps = pd.DatetimeIndex(index)

    # The measurement covariance is point in time, one per reported day, from
    # the window ending on that day. Both arms get the same matrix on the same
    # day, so it cannot favour either, and no reported number depends on a
    # price that had not printed yet.
    #
    # One matrix over the whole panel would read better -- the series would
    # move only because the holdings moved, rather than partly because the
    # estimator wandered -- and it was written that way first. The leak check
    # in `test_replay.py` rejected it, correctly: rewriting every price after a
    # date changed the reported dispersion *before* that date, because the
    # matrix had seen the rewrite. The measurement enters no decision, so it is
    # not a leak that could flatter the policy, but a number that moves when
    # the future is rewritten is not a number about the past, and this project
    # has twice found that kind of small to be load-bearing.
    #
    # The price is that the instrument set grows as instruments list, so a
    # range over five risk shares and a range over ten both appear in one
    # series. `measured` and `unmeasured` name the set at the end, and the
    # report says how it changed.
    positions = {day: i for i, day in enumerate(dates)}
    per_day: "list[tuple[pd.Timestamp, pd.DataFrame, list[str], object]]" = []
    for day in stamps:
        window = estimable_window(closes, positions[day] + 1, lookback,
                                  minimum=warmup)
        if len(window) < 2:
            continue
        cov = covariance_matrix(window)
        names = [str(c) for c in cov.columns]
        scope = ([k for k in among if k in cov.columns]
                 if among is not None else None)
        per_day.append((day, cov, names, scope))
    # An empty ledger has no days to measure, which is not an error: the
    # caller gets two empty arms and a purchase count of zero. Days that exist
    # but none of them estimable IS an error, because it means the panel cannot
    # support the comparison at all and a silent empty series would read as
    # "no difference".
    if len(stamps) and not per_day:
        raise ValueError(
            "no day in the replay has enough price history to estimate a "
            "covariance, so neither arm's dispersion can be measured.")
    measure_keys = per_day[-1][2] if per_day else []
    first_keys = per_day[0][2] if per_day else []

    def arm(name, rows, flows, shares, spent) -> Arm:
        values = pd.Series([v for v, _ in rows], index=stamps)
        # A contribution is not a return. The order fills at that day's close,
        # so the new money was not in the book for the day's move and has to
        # come back out before the series is divided. Left in, every purchase
        # day reads as a double-digit gain: this replay reported annualised
        # volatilities of 0.81 and 0.56 for two arms holding six ETFs between
        # them, and the two arms differed because they invested slightly
        # different amounts, not because they held different things.
        grown = values - pd.Series(flows, index=stamps)
        holdings_at = {day: holdings for day, (_, holdings) in zip(stamps, rows)}
        dispersion = pd.Series(
            [dispersion_of(
                np.array([holdings_at[day].get(k, 0.0) for k in names]),
                cov, scope)
             for day, cov, names, scope in per_day],
            index=pd.DatetimeIndex([d for d, _, _, _ in per_day]))
        returns = (grown / values.shift(1) - 1.0).iloc[1:].dropna()
        return Arm(name=name, dispersion=dispersion, returns=returns,
                   final_shares=shares, invested=float(spent))

    return ReplayResult(
        actual=arm("actual", rows_a, flows_a, dict(actual), invested),
        allocated=arm("allocator", rows_m, flows_m, dict(model), invested_model),
        purchases=count, total_invested=invested, carried=carry,
        measured=tuple(measure_keys), unmeasured=tuple(
            k for k in keys if k not in measure_keys),
        opened_with=len(first_keys), ledger_purchases=ledger_purchases,
        withdrawals=withdrawals, on_sale=on_sale,
        stopped_at=stop, skipped=tuple(dict.fromkeys(skipped)))
