"""Portfolio value and return history, reconstructed from the ledger.

Why a separate module from `positions`
--------------------------------------
`positions` answers "what do I hold and what did it cost" as of one date.
Performance asks the same question on every date of the price history, and
then asks what the answer *did* between dates. That is a time-series problem
with its own traps -- non-overlapping calendars, price gaps, flows that land on
a weekend -- and folding it into the position replay would cost the latter its
hand-verifiable simplicity. So this module replays the ledger onto a calendar
and leaves the per-transaction money arithmetic to `positions`, which it calls
for the cost basis so the two can never disagree.

Time-weighted versus money-weighted
-----------------------------------
Two returns are computed and both are shown, because they answer different
questions:

    time-weighted   what the *holdings* did, with the effect of deposits and
                    withdrawals removed. Comparable to a benchmark. This is
                    the return index the charts draw.
    money-weighted  what *your money* did, timing included: the IRR of your
                    flows against the terminal value. Not comparable to a
                    benchmark, but it is the return you actually experienced.

A portfolio that doubled the year before a large deposit and then fell 10%
has a strong time-weighted return and a negative money-weighted one. Showing
one without the other invites the wrong conclusion.

Flow convention
---------------
There is no cash account: the portfolio is the holdings and nothing else, so
every transaction is an external flow. Signs are from the portfolio's point of
view, + = money in:

    BUY       +(gross + fees)   money came in and became units; the fees
                                left your pocket for these units too
    SELL      -(gross - fees)   what actually reached you
    DIVIDEND  -(net)            paid out to you, not reinvested
    FEE       +(fee)            paid from outside; nothing changed in the
                                holdings, so the return absorbs it as a loss

Inflows count from the start of the day and outflows from the end of it. The
first half is the contract's convention; the second is forced by full exits
and is explained under `time_weighted_return`.

Adjusted closes and dividends: a trap on record
-----------------------------------------------
The price histories are adjusted closes, which is what the provider returns
and what the covariance needs. An adjusted close folds distributions back into
the price, so a distributing holding shows no ex-dividend drop; recording the
same dividend as an outflow then counts it twice in the time-weighted return.
The instruments this tool was built for are overwhelmingly accumulating, so
the effect is nil in practice, but it is a limitation of the inputs rather
than of this module, and it is written down here so nobody rediscovers it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from decimal import Decimal

import numpy as np
import pandas as pd

from .models import Transaction, TransactionType
from .money import BASE_CURRENCY, FxRates, MissingRate, Money, convert, normalise_currency
from .positions import derive_positions
from .returns import TRADING_DAYS_PER_YEAR
from .risk import beta as _beta
from .risk import drawdown as _drawdown

__all__ = [
    "ValueHistory", "HoldingContribution", "PerformanceSummary",
    "value_history", "time_weighted_return", "return_index", "annualised_return",
    "xirr", "money_weighted_return", "period_returns", "monthly_table",
    "drawdown_series", "rolling_volatility", "rolling_beta", "trailing_returns",
    "holding_contributions", "performance_summary", "TRAILING_KEYS",
]

TRAILING_KEYS = ("1w", "1m", "3m", "6m", "ytd", "1y", "all")


# --------------------------------------------------------------------------
# Value history
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ValueHistory:
    """The portfolio on every date of its price history.

    Every series shares one `pd.DatetimeIndex`. The per-holding frames share
    the same index and the same columns: the ISINs that could be valued.
    Anything that could not is in `missing`, with the reason in `warnings`,
    and is absent from *all* the figures -- a value series that quietly drops
    one holding's flows but keeps its value is wrong in a way no chart shows.
    """
    value: pd.Series                # holdings market value in base, per date
    cost_basis: pd.Series           # cost basis of open positions, per date
    invested: pd.Series             # cumulative net external flow, per date
    flows: pd.Series                # external flow per date (+ = money in), base
    per_holding: pd.DataFrame       # value per ISIN per date (columns = ISIN)
    quantities: pd.DataFrame        # units per ISIN per date
    flows_per_holding: pd.DataFrame
    missing: tuple[str, ...]        # ISINs with no usable price history, excluded and named
    warnings: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return self.value.empty


def _empty_history(missing, warnings) -> ValueHistory:
    idx = pd.DatetimeIndex([])

    def series() -> pd.Series:
        return pd.Series(dtype=float, index=idx)

    def frame() -> pd.DataFrame:
        return pd.DataFrame(index=idx, dtype=float)

    return ValueHistory(series(), series(), series(), series(), frame(), frame(), frame(),
                        tuple(missing), tuple(warnings))


def _clean_prices(series: pd.Series) -> pd.Series:
    """Float, NaN-free, sorted, de-duplicated, on a naive date-only index.

    Providers return tz-aware timestamps for some venues and naive ones for
    others; a union of the two raises, so everything is stripped to dates.
    """
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _quantity_steps(txns: list[Transaction]) -> pd.Series:
    """Units held after each transaction date, as a step function.

    Indexed by transaction date, so a weekend trade sits on the weekend and
    is picked up by the next trading day when reindexed with a forward fill.
    """
    deltas: dict[dt.date, Decimal] = {}
    for t in txns:
        if t.type is TransactionType.BUY:
            deltas[t.date] = deltas.get(t.date, Decimal(0)) + t.quantity
        elif t.type is TransactionType.SELL:
            deltas[t.date] = deltas.get(t.date, Decimal(0)) - t.quantity
    dates = sorted(deltas)
    running = Decimal(0)
    levels: list[float] = []
    for d in dates:
        running += deltas[d]
        levels.append(float(running))
    return pd.Series(levels, index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]),
                     dtype=float)


def _flow_money(t: Transaction) -> Money:
    """The external flow of one transaction, in its own currency, + = in."""
    if t.type is TransactionType.BUY:
        return t.gross + t.fee_money
    if t.type is TransactionType.SELL:
        return -(t.gross - t.fee_money)
    if t.type is TransactionType.DIVIDEND:
        gross = t.gross
        if gross.amount == 0:
            # Same convention as positions._replay: a dividend recorded with
            # no quantity carries its total in price_per_unit.
            gross = Money(t.price_per_unit, t.currency)
        return -(gross - t.fee_money)
    return t.gross + t.fee_money       # FEE: a charge, with or without a fee


def _to_base_float(money: Money, on: dt.date, rates: FxRates | None, base: str) -> float:
    if money.currency == base:
        return float(money.amount)
    if rates is None:
        raise MissingRate(f"no FX rates supplied to convert {money.currency}->{base}")
    return float(convert(money, base, on, rates).amount)


def _fx_series(ccy: str, base: str, calendar: pd.DatetimeIndex, rates: FxRates | None,
               cache: dict) -> pd.Series:
    """Units of base per unit of `ccy` on each calendar date; NaN where unknown.

    NaN rather than 1.0: a missing rate must surface as a gap that is counted
    and named, never as a silent identity conversion.
    """
    if ccy == base:
        return pd.Series(1.0, index=calendar)
    if rates is None:
        raise MissingRate(f"no FX rates supplied to convert {ccy}->{base}")
    out = np.full(len(calendar), np.nan)
    for i, ts in enumerate(calendar):
        key = (ccy, ts)
        if key not in cache:
            try:
                cache[key] = float(rates.rate(ccy, base, ts.date()))
            except MissingRate:
                cache[key] = np.nan
        out[i] = cache[key]
    return pd.Series(out, index=calendar)


def value_history(transactions: list[Transaction],
                  histories: dict[str, pd.Series],
                  quote_currencies: dict[str, str],
                  rates: FxRates | None = None,
                  base: str = BASE_CURRENCY,
                  start: dt.date | None = None) -> ValueHistory:
    """Replay the ledger onto the union of the price calendars.

    Quantity per date is a step function of the ledger (a transaction counts
    from its own date; a weekend trade counts from the next trading day).
    Value per date is quantity x adjusted close x FX(quote -> base) on that
    date. Price and FX gaps are forward-filled and counted in `warnings`.
    Flows are converted at the transaction date, per the convention in the
    module docstring, and `invested` is their running sum. The cost basis is
    `positions.derive_positions` replayed up to each transaction date, so it
    agrees with the Holdings view to the cent.

    Only ISINs with transactions are valued; extra price series are ignored.
    The per-holding frames share one column order, the ledger's: the ISIN
    whose first transaction is earliest comes first, and two first bought on
    the same day keep the order they were recorded in. Deterministic, and
    it reads as the story of the portfolio; an alphabetical order would be
    neither. Excluded, and named in `missing`:

    - an ISIN with no price series, or none inside the window;
    - an ISIN whose quote currency cannot be converted (no rates supplied, or
      no rate on or before any date). A missing quote currency is taken from
      the ISIN's own transactions when they agree, with a warning.

    Where the history starts: at `start`, else the first transaction date,
    pushed later to the first date on which every open holding has a price
    (a purchase before the provider's history begins cannot be valued, and a
    zero there would read as a loss). Flows before that date are carried into
    it so `invested` stays complete. The same rule applies per holding to a
    later purchase whose series starts after the trade.
    """
    warnings: list[str] = []
    missing: list[str] = []
    # Stable sort on the date alone. A tie-break on `id` would look natural
    # and is a trap: ids are random, so two holdings bought the same day
    # would swap columns from one run to the next.
    txns = sorted(transactions, key=lambda t: t.date)
    if not txns:
        return _empty_history((), ())

    by_isin: dict[str, list[Transaction]] = {}
    for t in txns:
        by_isin.setdefault(t.isin, []).append(t)

    # Step 1: which ISINs have a price series and a currency for it.
    prices: dict[str, pd.Series] = {}
    currencies: dict[str, str] = {}
    for isin, own in by_isin.items():
        raw = histories.get(isin)
        series = _clean_prices(raw) if raw is not None else pd.Series(dtype=float)
        if series.empty:
            missing.append(isin)
            warnings.append(f"{isin}: no price history; excluded from the value "
                            f"history together with its {len(own)} transaction(s)")
            continue
        ccy = (quote_currencies.get(isin) or "").strip()
        if not ccy:
            seen = sorted({t.currency for t in own})
            if len(seen) != 1:
                missing.append(isin)
                warnings.append(f"{isin}: no quote currency supplied and its transactions "
                                f"use {seen}; excluded rather than guessed")
                continue
            ccy = seen[0]
            warnings.append(f"{isin}: no quote currency supplied; using {ccy}, the "
                            f"currency of its transactions")
        code, factor = normalise_currency(ccy, Decimal(1))
        # A pence-quoted series is rescaled here, exactly as Transaction does
        # for a pence-quoted price. Doing it in one place keeps GBX out of
        # every downstream formula.
        prices[isin] = series * float(factor)
        currencies[isin] = code

    if not prices:
        warnings.append("no holding has a usable price history")
        return _empty_history(missing, warnings)

    begin = pd.Timestamp(start if start is not None else txns[0].date).normalize()
    calendar: pd.DatetimeIndex | None = None
    for s in prices.values():
        calendar = s.index if calendar is None else calendar.union(s.index)
    assert calendar is not None
    calendar = calendar[calendar >= begin]
    if calendar.empty:
        warnings.append(f"no price dates on or after {begin.date().isoformat()}")
        return _empty_history(missing, warnings)

    # Step 2: per holding -- quantity steps, unit value in base, and flows in
    # base at the transaction date. Any conversion failure excludes the whole
    # holding: a value series with some of its flows is worse than none.
    quantities: dict[str, pd.Series] = {}
    unit_values: dict[str, pd.Series] = {}
    known: dict[str, pd.Series] = {}
    flow_lists: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    fx_cache: dict = {}
    for isin in list(prices):
        try:
            fx = _fx_series(currencies[isin], base, calendar, rates, fx_cache)
            flows_of = [(pd.Timestamp(t.date), _to_base_float(_flow_money(t), t.date, rates, base))
                        for t in by_isin[isin]]
        except MissingRate as exc:
            missing.append(isin)
            warnings.append(f"{isin}: {exc}; excluded from the value history")
            continue
        unit = prices[isin].reindex(calendar) * fx
        if unit.notna().sum() == 0:
            missing.append(isin)
            warnings.append(f"{isin}: no price with an FX rate for {currencies[isin]}->{base} "
                            f"on any date from {calendar[0].date().isoformat()}; excluded")
            continue
        quantities[isin] = _quantity_steps(by_isin[isin]).reindex(
            calendar, method="ffill").fillna(0.0)
        known[isin] = unit.notna()
        unit_values[isin] = unit.ffill()
        flow_lists[isin] = flows_of

    if not quantities:
        warnings.append("no holding could be valued")
        return _empty_history(missing, warnings)
    included = list(quantities)

    # Step 3: where the history can honestly start. A held holding without a
    # price yet is a zero that would read as a loss, so the start moves to
    # the first date on which every open holding is priced.
    held_unpriced = pd.DataFrame({i: (quantities[i] > 0) & unit_values[i].isna()
                                  for i in included}, index=calendar)
    ok_rows = ~held_unpriced.any(axis=1)
    first_ok = ok_rows.idxmax() if ok_rows.any() else None
    if first_ok is None:
        warnings.append("no date on which every open holding has a price")
        return _empty_history(missing, warnings)
    if first_ok != calendar[0]:
        culprits = sorted(i for i in included if held_unpriced.loc[:first_ok, i].iloc[:-1].any())
        warnings.append(f"history starts {first_ok.date().isoformat()} rather than "
                        f"{calendar[0].date().isoformat()}: {', '.join(culprits)} held "
                        f"without a price before then; earlier flows are carried into "
                        f"the first date")
        calendar = calendar[calendar >= first_ok]
        for i in included:
            quantities[i] = quantities[i].loc[calendar]
            unit_values[i] = unit_values[i].loc[calendar]
            known[i] = known[i].loc[calendar]
            held_unpriced[i] = held_unpriced[i].loc[calendar]
        held_unpriced = held_unpriced.loc[calendar]

    # Step 4: flows onto the calendar. A flow lands on the first calendar date
    # on or after the later of its own date and the holding's first priced
    # date; one dated after the calendar ends has nothing to land on.
    flows_ph = pd.DataFrame(0.0, index=calendar, columns=included)
    late: list[str] = []
    for isin in included:
        vstart = unit_values[isin].first_valid_index()
        for when, amount in flow_lists[isin]:
            target = max(when, vstart)
            pos = int(calendar.searchsorted(target, side="left"))
            if pos >= len(calendar):
                late.append(isin)
                continue
            flows_ph.iloc[pos, flows_ph.columns.get_loc(isin)] += amount
        unpriced_days = int(held_unpriced[isin].sum())
        if unpriced_days:
            warnings.append(f"{isin}: held on {unpriced_days} date(s) before its first "
                            f"price on {vstart.date().isoformat()}; valued at zero there "
                            f"and its flows carried to that date")
        filled = int(((quantities[isin] > 0) & ~known[isin] & unit_values[isin].notna()).sum())
        if filled:
            last_known = known[isin][known[isin]].index[-1]
            tail = (f"; last price {last_known.date().isoformat()}"
                    if last_known < calendar[-1] else "")
            warnings.append(f"{isin}: {filled} price gap(s) forward-filled while held{tail}")
    if late:
        warnings.append(f"{len(late)} transaction(s) dated after the last price "
                        f"({calendar[-1].date().isoformat()}) are not in the history: "
                        f"{', '.join(sorted(set(late)))}")

    # Step 5: the cost basis, by the same arithmetic as the Holdings view.
    included_txns = [t for t in txns if t.isin in quantities]
    levels: dict[pd.Timestamp, float] = {}
    for d in sorted({t.date for t in included_txns}):
        positions = derive_positions([t for t in included_txns if t.date <= d],
                                     rates=rates, base=base, strict=False)
        levels[pd.Timestamp(d)] = float(sum(
            (p.cost_basis.amount for p in positions.values() if p.is_open and p.is_derivable),
            Decimal(0)))
    cost_basis = pd.Series(levels, dtype=float).reindex(calendar, method="ffill").fillna(0.0)

    per_holding = pd.DataFrame({i: quantities[i] * unit_values[i].fillna(0.0)
                                for i in included}, index=calendar)
    flows = flows_ph.sum(axis=1)
    return ValueHistory(
        value=per_holding.sum(axis=1),
        cost_basis=cost_basis,
        invested=flows.cumsum(),
        flows=flows,
        per_holding=per_holding,
        quantities=pd.DataFrame(quantities, index=calendar),
        flows_per_holding=flows_ph,
        missing=tuple(missing),
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------
# Returns
# --------------------------------------------------------------------------

def time_weighted_return(value: pd.Series, flows: pd.Series) -> pd.Series:
    """Daily return with the effect of external flows removed.

        r_t = (V_t + out_t) / (V_{t-1} + in_t) - 1

    where in_t is the day's inflow (money that arrived at the start of the
    day and earned the day's return) and out_t the day's outflow (units that
    earned the day's return and then left at the price they fetched). With
    inflows only this is the contract's r_t = V_t / (V_{t-1} + F_t) - 1.

    Why outflows are end-of-day: put a sale at the start of the day and a
    full exit divides by V_{t-1} - proceeds, which is a few euros of noise.
    Selling 100 of holdings for 99 would read as a -100% day. The proceeds
    *are* the day's valuation of the units sold, and that is exactly what an
    end-of-day flow says.

    The first day's return is 0, and so is any day whose denominator is 0
    (nothing held and nothing added: there is no return to measure).

    >>> import pandas as pd
    >>> idx = pd.bdate_range("2026-01-01", periods=3)
    >>> v = pd.Series([100.0, 220.0, 209.0], index=idx)
    >>> f = pd.Series([100.0, 100.0, 0.0], index=idx)
    >>> [round(r, 4) for r in time_weighted_return(v, f)]     # 220/(100+100)-1, 209/220-1
    [0.0, 0.1, -0.05]
    """
    v = value.astype(float)
    f = flows.reindex(v.index).fillna(0.0).astype(float)
    prev = v.shift(1)
    inflow = f.clip(lower=0.0)
    outflow = -f.clip(upper=0.0)
    denominator = (prev + inflow).to_numpy()
    numerator = (v + outflow).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denominator > 0, numerator / denominator - 1.0, 0.0)
    if len(r):
        r[0] = 0.0
    return pd.Series(r, index=v.index, dtype=float)


def return_index(daily: pd.Series, start: float = 100.0) -> pd.Series:
    """Chain daily returns into a level series: start x prod(1 + r).

    >>> import pandas as pd
    >>> return_index(pd.Series([0.0, 0.1, -0.1])).round(6).tolist()
    [100.0, 110.0, 99.0]
    """
    return start * (1.0 + daily.astype(float)).cumprod()


def annualised_return(daily: pd.Series, trading_days: int = TRADING_DAYS_PER_YEAR
                      ) -> float | None:
    """Geometric annualisation: (prod(1 + r)) ^ (252 / n) - 1.

    Geometric, because it is the rate that actually compounds to the observed
    total; the arithmetic mean x 252 overstates it by about half the variance
    and is used only where the textbook ratio calls for it (Sharpe, Sortino;
    see `performance_summary`). None for an empty series or one that lost
    everything, where no rate compounds to the result.

    >>> import pandas as pd
    >>> round(annualised_return(pd.Series([0.01] * 252)), 10) == round(1.01 ** 252 - 1, 10)
    True
    """
    r = daily.dropna().astype(float).to_numpy()
    if r.size == 0:
        return None
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return None
    return growth ** (trading_days / r.size) - 1.0


def xirr(cashflows: list[tuple[dt.date, float]]) -> float | None:
    """Internal rate of return of dated flows, investor's sign convention.

    Money paid in is negative, money received is positive; the rate r solves

        sum_i cf_i / (1 + r) ^ t_i = 0,   t_i in years, actual/365

    Newton's method from 0.1, which converges in a handful of steps on any
    ordinary ledger, with bisection on [-0.9999, 10] when it does not (a
    flat NPV curve, a step past -100%). None when there are fewer than two
    non-zero flows, when every flow has the same sign (no rate makes them
    net to zero), or when neither method converges. A ledger with several
    sign changes can have several roots; the one nearest 0.1 is reported,
    which is the conventional and almost always the meaningful one.

    >>> import datetime as dt
    >>> round(xirr([(dt.date(2025, 1, 1), -1000.0), (dt.date(2026, 1, 1), 1100.0)]), 6)
    0.1
    """
    flows = [(d, float(a)) for d, a in cashflows if float(a) != 0.0]
    if len(flows) < 2:
        return None
    amounts = np.array([a for _, a in flows], dtype=float)
    if np.all(amounts > 0) or np.all(amounts < 0):
        return None
    t0 = min(d for d, _ in flows)
    years = np.array([(d - t0).days / 365.0 for d, _ in flows], dtype=float)
    scale = float(np.abs(amounts).sum())
    tolerance = 1e-10 * scale

    def npv(rate: float) -> float:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return float(np.sum(amounts * (1.0 + rate) ** (-years)))

    def slope(rate: float) -> float:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return float(np.sum(-years * amounts * (1.0 + rate) ** (-years - 1.0)))

    rate = 0.1
    for _ in range(100):
        f = npv(rate)
        if not math.isfinite(f):
            break
        if abs(f) <= tolerance:
            return rate
        d = slope(rate)
        if not math.isfinite(d) or d == 0.0:
            break
        step = f / d
        nxt = rate - step
        if not math.isfinite(nxt) or nxt <= -1.0:
            break
        if abs(step) < 1e-12:
            return nxt
        rate = nxt

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo * f_hi > 0:
        return None
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if abs(f_mid) <= tolerance or hi - lo < 1e-12:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return None


def money_weighted_return(flows: pd.Series, terminal_value: float,
                          terminal_date: dt.date) -> float | None:
    """Annualised IRR of the portfolio's external flows against its value now.

    `flows` uses the portfolio's sign (+ = money in); the IRR uses the
    investor's (paying in is negative), so each flow is negated and the
    terminal value is what the investor would receive by selling everything.

    >>> import datetime as dt, pandas as pd
    >>> f = pd.Series([1000.0], index=pd.DatetimeIndex(["2025-01-01"]))
    >>> round(money_weighted_return(f, 1100.0, dt.date(2026, 1, 1)), 6)
    0.1
    """
    cashflows = [(pd.Timestamp(ts).date(), -float(v))
                 for ts, v in flows.items() if float(v) != 0.0]
    cashflows.append((terminal_date, float(terminal_value)))
    return xirr(cashflows)


_PERIOD_CODES = {"M": "M", "ME": "M", "Y": "Y", "YE": "Y", "A": "Y"}


def period_returns(daily: pd.Series, freq: str) -> pd.Series:
    """Compounded return per calendar month ("M") or year ("Y").

    Only periods that contain observations appear, indexed by their last
    calendar day. A resample would fill an empty month with a zero return,
    which is a claim about a month nothing was measured in.

    >>> import pandas as pd
    >>> r = pd.Series([0.1, -0.1, 0.05], index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-03-02"]))
    >>> period_returns(r, "M").round(6).tolist()      # 1.1 x 0.9 - 1, then 0.05
    [-0.01, 0.05]
    >>> [d.strftime("%Y-%m-%d") for d in period_returns(r, "M").index]
    ['2026-01-31', '2026-03-31']
    """
    code = _PERIOD_CODES.get(freq.upper())
    if code is None:
        raise ValueError(f"freq must be 'M' or 'Y', got {freq!r}")
    r = daily.dropna().astype(float).sort_index()
    if r.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    compounded = (1.0 + r).groupby(r.index.to_period(code)).prod() - 1.0
    compounded.index = compounded.index.to_timestamp(how="end").normalize()
    return compounded.astype(float)


def monthly_table(daily: pd.Series) -> pd.DataFrame:
    """Calendar of monthly returns: index = year, columns = 1..12, NaN gaps."""
    months = period_returns(daily, "M")
    columns = pd.Index(range(1, 13), name="month")
    if months.empty:
        return pd.DataFrame(index=pd.Index([], name="year", dtype=int), columns=columns,
                            dtype=float)
    long = pd.DataFrame({"year": months.index.year, "month": months.index.month,
                         "ret": months.to_numpy()})
    table = long.pivot(index="year", columns="month", values="ret").reindex(columns=columns)
    table.index.name = "year"
    return table.astype(float)


def drawdown_series(index: pd.Series) -> pd.Series:
    """Distance below the running peak, value / cummax - 1, so never above 0.

    >>> import pandas as pd
    >>> drawdown_series(pd.Series([100.0, 120.0, 90.0, 110.0])).round(6).tolist()
    [0.0, 0.0, -0.25, -0.083333]
    """
    v = index.astype(float)
    return v / v.cummax() - 1.0


def rolling_volatility(daily: pd.Series, window: int = 63,
                       trading_days: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Annualised standard deviation (ddof=1) over a trailing window.

    The first window-1 dates have no estimate and are dropped rather than
    carried as NaN, so the result is a series of numbers a chart can draw.
    """
    return (daily.astype(float).rolling(window).std(ddof=1).dropna()
            * math.sqrt(trading_days))


def rolling_beta(daily: pd.Series, benchmark: pd.Series, window: int = 63) -> pd.Series:
    """Trailing beta on shared dates: rolling Cov(p, b) / Var(b).

    Both rolling estimators are Bessel-corrected, so the correction cancels
    in the ratio. Dates before the window fills, and any window with a
    zero-variance benchmark, are dropped.
    """
    joined = pd.concat([daily.astype(float).rename("p"),
                        benchmark.astype(float).rename("b")], axis=1, sort=True).dropna()
    cov = joined["p"].rolling(window).cov(joined["b"])
    var = joined["b"].rolling(window).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan).dropna()


def trailing_returns(index: pd.Series, as_of: dt.date | None = None
                     ) -> dict[str, float | None]:
    """Return from a calendar anchor to `as_of`, for the standard windows.

    Each window's base is the last index value on or before its anchor:
    7 days, 1/3/6 months and 12 months before `as_of` (calendar offsets, so
    a month is a month whatever the weekends), and for "ytd" 1 January of
    the `as_of` year -- so the base is the last value of the previous year.
    None when the index does not reach back to the anchor: a "1y" return on
    ten months of history would be a different number wearing the label.
    "all" is from the first value.
    """
    v = index.dropna().astype(float).sort_index()
    out: dict[str, float | None] = {k: None for k in TRAILING_KEYS}
    if v.empty:
        return out
    end_ts = pd.Timestamp(as_of) if as_of is not None else v.index[-1]
    if end_ts < v.index[0]:
        return out
    end_value = float(v.asof(end_ts))
    anchors = {
        "1w": end_ts - pd.Timedelta(days=7),
        "1m": end_ts - pd.DateOffset(months=1),
        "3m": end_ts - pd.DateOffset(months=3),
        "6m": end_ts - pd.DateOffset(months=6),
        "ytd": pd.Timestamp(year=end_ts.year, month=1, day=1),
        "1y": end_ts - pd.DateOffset(months=12),
    }
    for key, anchor in anchors.items():
        if anchor < v.index[0]:
            continue
        base_value = float(v.asof(anchor))
        if math.isfinite(base_value) and base_value > 0:
            out[key] = end_value / base_value - 1.0
    first = float(v.iloc[0])
    out["all"] = end_value / first - 1.0 if first > 0 else None
    return out


# --------------------------------------------------------------------------
# Contributions
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HoldingContribution:
    """One holding's share of the portfolio's gain over a window."""
    isin: str
    pnl: float                  # base currency: value change net of flows
    contribution: float         # pnl / portfolio value at the start


def holding_contributions(history: ValueHistory,
                          start: dt.date | None = None,
                          end: dt.date | None = None) -> list[HoldingContribution]:
    """P&L per holding between two dates, and its share of the starting value.

        pnl_i = V_i(end) - V_i(start) - sum of holding i's flows in (start, end]

    Flows dated on `start` are already inside V(start) (they count from the
    start of the day) and so are not subtracted. The contribution divides by
    the *portfolio's* value at start, not the holding's, so that the
    contributions sum to the portfolio's own gain over its starting capital
    -- that additivity is what makes them a decomposition rather than a list
    of individual returns. A window that begins before anything was held has
    no starting value; the inflows over the window stand in for it.

    Sorted by pnl, largest gain first.
    """
    values = history.per_holding
    if values.empty:
        return []
    start_ts = pd.Timestamp(start) if start is not None else values.index[0]
    end_ts = pd.Timestamp(end) if end is not None else values.index[-1]
    before_start = values.loc[:start_ts]
    at_start = before_start.iloc[-1] if len(before_start) else pd.Series(0.0, index=values.columns)
    before_end = values.loc[:end_ts]
    at_end = before_end.iloc[-1] if len(before_end) else pd.Series(0.0, index=values.columns)
    window = history.flows_per_holding.loc[
        (history.flows_per_holding.index > start_ts) & (history.flows_per_holding.index <= end_ts)]
    flows = window.sum() if len(window) else pd.Series(0.0, index=values.columns)
    pnl = at_end - at_start - flows

    denominator = float(at_start.sum())
    if denominator <= 0:
        denominator = float(window.clip(lower=0.0).sum().sum()) if len(window) else 0.0
    rows = [HoldingContribution(str(isin), float(p),
                                float(p) / denominator if denominator > 0 else 0.0)
            for isin, p in pnl.items()]
    return sorted(rows, key=lambda c: (-c.pnl, c.isin))


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PerformanceSummary:
    """Headline figures. Anything that cannot be computed is None, never 0."""
    total_return: float | None
    annualised_return: float | None
    money_weighted: float | None
    volatility: float | None
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    max_drawdown: float | None
    current_drawdown: float | None
    best_day: float | None
    worst_day: float | None
    best_month: float | None
    worst_month: float | None
    positive_months: int
    negative_months: int
    observations: int
    first_date: dt.date | None
    last_date: dt.date | None
    # Against a benchmark; all None without one.
    benchmark_total_return: float | None
    beta: float | None
    alpha: float | None
    correlation: float | None
    tracking_error: float | None
    information_ratio: float | None


# Below this, an annualised standard deviation is floating-point noise rather
# than a measurement: ten identical returns have a sample std near 1e-17, not
# 0, and a Sharpe ratio over it would be a number in the quadrillions. Such a
# figure is reported as 0 and treated as an undefined denominator.
_NOISE = 1e-12


def _snap(value: float | None) -> float | None:
    """0.0 for a value below the noise floor, else the value unchanged."""
    if value is None:
        return None
    return 0.0 if abs(value) <= _NOISE else value


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= _NOISE:
        return None
    return float(numerator / denominator)


def performance_summary(daily: pd.Series, benchmark: pd.Series | None = None,
                        risk_free: float = 0.0, flows: pd.Series | None = None,
                        terminal_value: float | None = None) -> PerformanceSummary:
    """Every headline statistic, with its convention stated here once.

    - total return:        prod(1 + r) - 1 over the whole series
    - annualised return:   geometric, see `annualised_return`
    - volatility:          std(r, ddof=1) x sqrt(252)
    - Sharpe:              (mean(r) x 252 - rf) / volatility. The textbook
                           estimator uses the arithmetic mean; it is stated
                           here because the annualised return above is
                           geometric and the two differ.
    - Sortino:             (mean(r) x 252 - rf) / downside deviation, where
                           downside deviation = sqrt(mean(min(r, 0)^2)) x
                           sqrt(252) -- the full-sample root mean square of
                           returns below zero, target 0, not the std of the
                           negative days alone.
    - Calmar:              annualised return / |max drawdown|
    - drawdowns:           on the chained return index, <= 0
    - months:              compounded per calendar month; positive/negative
                           counts exclude exactly-zero months
    - money-weighted:      `money_weighted_return` when flows and a terminal
                           value are supplied, else None
    - benchmark figures:   on the dates both series share. beta as in
                           `risk.beta`; alpha is the intercept of the daily
                           regression of excess returns (r - rf/252 on
                           b - rf/252) x 252; tracking error std(r - b) x
                           sqrt(252); information ratio mean(r - b) x 252 /
                           tracking error.

    `risk_free` is an annual rate and defaults to 0: the tool has no rate
    source, a guessed one is a claim, and euro short rates were near zero for
    most of the period this was written against. Pass one to change it.

    Every ratio is None with fewer than 2 observations or a zero denominator.
    """
    r = daily.dropna().astype(float).sort_index()
    n = int(len(r))
    first = r.index[0].date() if n else None
    last = r.index[-1].date() if n else None
    arr = r.to_numpy()

    total = float(np.prod(1.0 + arr)) - 1.0 if n else None
    annual = annualised_return(r) if n >= 2 else None
    volatility = (_snap(float(np.std(arr, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)))
                  if n >= 2 else None)
    mean_annual = float(arr.mean() * TRADING_DAYS_PER_YEAR) if n >= 2 else None
    excess = None if mean_annual is None else mean_annual - risk_free
    sharpe = _ratio(excess, volatility)
    downside = (_snap(float(np.sqrt(np.mean(np.minimum(arr, 0.0) ** 2))
                            * math.sqrt(TRADING_DAYS_PER_YEAR)))
                if n >= 2 else None)
    sortino = _ratio(excess, downside)

    if n:
        dd = _drawdown(return_index(r))
        max_dd, current_dd = float(dd.max_drawdown), float(dd.current_drawdown)
    else:
        max_dd = current_dd = None
    calmar = _ratio(annual, abs(max_dd)) if max_dd is not None else None

    months = period_returns(r, "M") if n else pd.Series(dtype=float)
    money_weighted = (money_weighted_return(flows, terminal_value, last)
                      if flows is not None and terminal_value is not None and last is not None
                      else None)

    bench_total = beta_value = alpha = correlation = tracking_error = information = None
    if benchmark is not None:
        joined = pd.concat([r.rename("p"), benchmark.dropna().astype(float).rename("b")],
                           axis=1, sort=True).dropna()
        if len(joined) >= 2:
            p, b = joined["p"].to_numpy(), joined["b"].to_numpy()
            bench_total = float(np.prod(1.0 + b)) - 1.0
            try:
                beta_value = _beta(joined["p"], joined["b"])
            except ValueError:
                beta_value = None
            if beta_value is not None:
                rf_daily = risk_free / TRADING_DAYS_PER_YEAR
                intercept = (p - rf_daily).mean() - beta_value * (b - rf_daily).mean()
                alpha = float(intercept * TRADING_DAYS_PER_YEAR)
            # A zero-variance series makes the correlation 0/0; numpy would
            # warn about it before pandas hands back the NaN that becomes None.
            with np.errstate(invalid="ignore", divide="ignore"):
                rho = float(joined["p"].corr(joined["b"]))
            correlation = rho if math.isfinite(rho) else None
            active = p - b
            tracking_error = _snap(float(np.std(active, ddof=1)
                                         * math.sqrt(TRADING_DAYS_PER_YEAR)))
            information = _ratio(float(active.mean() * TRADING_DAYS_PER_YEAR), tracking_error)

    return PerformanceSummary(
        total_return=total,
        annualised_return=annual,
        money_weighted=money_weighted,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        current_drawdown=current_dd,
        best_day=float(arr.max()) if n else None,
        worst_day=float(arr.min()) if n else None,
        best_month=float(months.max()) if len(months) else None,
        worst_month=float(months.min()) if len(months) else None,
        positive_months=int((months > 0).sum()),
        negative_months=int((months < 0).sum()),
        observations=n,
        first_date=first,
        last_date=last,
        benchmark_total_return=bench_total,
        beta=beta_value,
        alpha=alpha,
        correlation=correlation,
        tracking_error=tracking_error,
        information_ratio=information,
    )
