"""Wiring a real book into the evaluation harness.

The composition root for backtests, and the only module that touches the data
layer, the policies and the harness at once. It exists so that `eval/` and
`agents/` can stay offline and pure: they are handed a price panel and a cost
model and never learn where either came from.

What it takes care of, and why each one is not incidental
---------------------------------------------------------
**The panel is built by outer join, not intersection.** An instrument that
listed late, or was sold and has no recent prices, appears as NaN outside its
own life. `MarketView.available` reads those NaNs at each decision date, so
the backtest is survivorship-honest in both directions without anything here
having to know about it. Intersecting instead would silently truncate the
whole history to the shortest series.

**Costs come from the instrument records.** Which broker holds a position,
what tax band it sits in, whether that band was read off a contract note or
assumed -- all of it is per holding, because at this account it genuinely
differs per holding, and a single global rate was wrong by a factor of eleven
until a contract note said so.

**Frozen holdings are pinned at the weight actually held.** A position at a
second broker cannot be traded against the rest of the book: that is a cash
transfer between institutions taking about a week, not a rebalance. Its weight
still drifts with the market, so it is pinned to whatever it currently is
rather than to a target, and both the policy and the referee are told.

A holding whose trading cost cannot be established is frozen the same way,
and the note says which of the two reasons applies. Refusing to price it
stays absolute -- nothing here invents a rate -- but refusing to produce any
result at all was a different and much stronger claim, and it is what one
unrecorded tax band used to do to a whole book.

There are no open prices here. The providers return adjusted closes, so
execution is next-close: a full extra day of delay against the next-open
model, which is the conservative direction.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import pathlib

import numpy as np
import pandas as pd

from .agents.execution import CostModel, cost_table
from .agents.referee import Referee, Refereed
from .agents.risk import EqualRiskContribution
from .core.positions import derive_positions, weights as position_weights
from .core.returns import MIN_OBSERVATIONS
from .data.store import DataMode, DataStore
from .eval.harness import Execution, Panel, buy_and_hold, walk_forward
from .eval.report import Comparison, compare

__all__ = ["Book", "load_book", "run_equal_risk_contribution",
           "allocate_new_money", "replay_the_ledger", "names_the_window",
           "dominant_holding", "survey_spreads", "SpreadSurvey",
           "record_survey", "Rung", "LADDER", "PAIRS", "RungResult",
           "PairResult", "LadderReport", "run_ladder", "rung_verdict",
           "ReferenceReport", "DateBlock", "reference_check"]


@dataclasses.dataclass(frozen=True)
class Book:
    """A real portfolio, ready to evaluate."""
    panel: Panel
    weights: dict[str, float]            # what is actually held, by value
    costs: CostModel
    tradeable: frozenset
    frozen: dict[str, float]             # non-tradeable holdings at held weight
    instruments: dict
    notes: tuple[str, ...]
    # For the buy-only allocator, which works in euros rather than weights: a
    # purchase is a number of shares at a price, and the whole point of it is
    # that it cannot be expressed as a weight change without also saying how
    # much money there is.
    values: dict[str, float] = dataclasses.field(default_factory=dict)
    prices: dict[str, float] = dataclasses.field(default_factory=dict)
    buyable: frozenset = frozenset()

    @property
    def account_value(self) -> float:
        return self.costs.account_value


def load_book(*, mode: str = "user", data_root: "pathlib.Path | None" = None,
              provider: str = "yfinance", lookback: int = 750,
              account_value: float | None = None) -> Book:
    """Load instruments, ledger and prices, and assemble everything downstream needs."""
    from .data.cache import PriceCache
    from .data.market import MarketData

    root = pathlib.Path(data_root) if data_root else None
    store = DataStore.open(DataMode(mode), root=root)
    instruments = store.load_instruments()
    transactions = store.load_transactions()
    if not instruments:
        raise ValueError(
            f"no instruments in {store.directory}. Add them first, or point "
            f"--data-root at the directory that holds them.")

    if provider == "fixture":
        from .data.providers.fixture import FixtureProvider
        market_provider = FixtureProvider()
    else:
        from .data.providers.yahoo import YahooProvider
        market_provider = YahooProvider()

    cache_dir = store.root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    market = MarketData(market_provider,
                        PriceCache(cache_dir / f"prices-{market_provider.name}.sqlite"))
    snapshot = market.load(instruments)

    # Outer join: every date any instrument traded, NaN where it did not.
    series = {isin: s for isin, s in snapshot.histories.items() if s is not None
              and len(s) >= MIN_OBSERVATIONS}
    if len(series) < 2:
        raise ValueError(
            f"only {len(series)} instrument(s) have at least "
            f"{MIN_OBSERVATIONS} price observations; a covariance matrix needs "
            f"two. Check the provider and the recorded symbols.")
    closes = pd.DataFrame(series).sort_index()
    closes = closes.iloc[-(lookback + 1):] if lookback else closes
    if not isinstance(closes.index, pd.DatetimeIndex):
        closes.index = pd.to_datetime(closes.index)

    positions = derive_positions(transactions, instruments,
                                 rates=snapshot.fx, quotes=snapshot.quotes,
                                 strict=False)
    held = {k: float(v) for k, v in
            position_weights(positions, snapshot.fx, on=dt.date.today()).items()
            if k in closes.columns}
    total = sum(held.values())
    held = {k: v / total for k, v in held.items()} if total > 0 else held

    value = account_value
    if value is None:
        value = sum(float(p.market_value(snapshot.fx, dt.date.today()).amount)
                    for p in positions.values()
                    if p.is_open and p.is_derivable
                    and p.market_value(snapshot.fx, dt.date.today()) is not None)
    costs = CostModel(account_value=value or 15_000.0,
                      per_instrument=cost_table(instruments))

    # Two separate reasons a holding's weight cannot be moved, kept apart
    # because they are undone by different things and the report should say
    # which one applies. A second broker is a fact about the account and will
    # not change; a missing tax band is a gap in the record that one contract
    # note closes.
    #
    # An unpriceable holding is frozen rather than fatal. The alternative is
    # what the previous build did: raise the moment the optimiser touched it,
    # so a single unrecorded rate meant no result at all. Refusing to *price*
    # it is still absolute -- nothing here invents a cost -- but a holding
    # whose cost is unknown is exactly a holding no policy may trade, which is
    # a constraint the machinery already models.
    unpriceable = {isin for isin in closes.columns
                   if costs.facts(isin).tob_rate is None}
    tradeable = frozenset(isin for isin in closes.columns
                          if instruments[isin].tradeable and isin not in unpriceable)
    frozen = {isin: held.get(isin, 0.0) for isin in closes.columns
              if isin not in tradeable}

    notes = []
    for isin in sorted(frozen):
        inst = instruments[isin]
        if not inst.tradeable:
            notes.append(
                f"{inst.display_name} ({isin}) is frozen at {frozen[isin]:.1%}: "
                f"held at {inst.broker or 'a different broker'}, so its weight "
                f"cannot be traded against the rest of the book without a "
                f"multi-day cash transfer between institutions. It stays in the "
                f"risk model.")
        else:
            notes.append(
                f"{inst.display_name} ({isin}) is frozen at {frozen[isin]:.1%} "
                f"because no transaction tax rate is recorded for it, so no "
                f"trade in it can be priced. This is a gap in the record rather "
                f"than a fact about the account: read the rate off a contract "
                f"note and run `portfolio instruments set {isin} --tob-rate "
                f"<rate> --observed` to free it.")
    if store.last_migration:
        notes.append(
            f"{store.instruments_path.name} predated the trading-cost columns "
            f"and was migrated on load; the derived values are assumptions "
            f"until confirmed against contract notes. The full notice was "
            f"printed above.")
    unpriced = [i for i in instruments if i not in closes.columns]
    if unpriced:
        notes.append(f"not in the panel (too little price history): "
                     f"{', '.join(sorted(unpriced))}")
    # Euro values and share prices, for the allocator. Values come from the
    # same positions the weights did; prices from the last close in the panel,
    # so a purchase is priced off the same series the risk model is estimated
    # from rather than off a quote that arrived at a different moment.
    values = {isin: float(w) * (value or 0.0) for isin, w in held.items()}
    prices = {str(c): float(closes[c].dropna().iloc[-1])
              for c in closes.columns if closes[c].notna().any()}
    buyable = frozenset(isin for isin in closes.columns
                        if instruments[isin].buyable)

    return Book(panel=Panel(closes=closes), weights=held, costs=costs,
                tradeable=tradeable, frozen=frozen, instruments=instruments,
                notes=tuple(notes), values=values, prices=prices,
                buyable=buyable)


def allocate_new_money(book: Book, cash: float, *, lookback: int = 252):
    """Where a purchase of `cash` should go, on this book.

    The covariance comes through the same estimator and window the equal risk
    policy uses, because this is the same claim about the same quantity: it
    forecasts nothing and reads only the covariance matrix. A second estimator
    for the same purpose would be a second answer to defend.
    """
    from .agents.allocate import allocate_buy_only
    from .core.returns import simple_returns
    from .core.risk import covariance_matrix

    window = simple_returns(book.panel.closes).dropna().iloc[-lookback:]
    cov = covariance_matrix(window)
    return allocate_buy_only(values=book.values, prices=book.prices, cov=cov,
                             cash=cash, costs=book.costs, buyable=book.buyable)


def run_equal_risk_contribution(book: Book, *, lookback: int = 252,
                                warmup: int | None = None,
                                rebalance_every: int = 21,
                                trials: int = 1,
                                trial_sharpe_sd: float = 0.0) -> Comparison:
    """Equal risk contribution against buy-and-hold, on this book.

    The benchmark is the book left alone from the same starting weights, so
    the comparison isolates the policy's trading rather than its choice of
    holdings. `overlap` is set to the rebalance interval, because successive
    decisions share all but a few days of their estimation window and a
    t-statistic computed on the raw count would be inflated by roughly the
    square root of that overlap.
    """
    warmup = warmup if warmup is not None else lookback + 1
    policy = EqualRiskContribution(lookback=lookback, fixed=dict(book.frozen))
    refereed = Refereed(policy, Referee(costs=book.costs,
                                        tradeable=book.tradeable))
    result = walk_forward(book.panel, refereed, warmup=warmup,
                          rebalance_every=rebalance_every,
                          cost_model=book.costs,
                          execution=Execution.NEXT_CLOSE,
                          initial_weights=book.weights,
                          frozen=frozenset(book.frozen))
    benchmark = buy_and_hold(book.panel, book.weights, warmup=warmup,
                             cost_model=book.costs,
                             execution=Execution.NEXT_CLOSE)
    notes = list(book.notes)

    # Did the policy do the thing it claims to do? Separate question from
    # whether that was worth doing, and the one a reader should be able to
    # check first: a policy that failed to equalise risk has not been
    # evaluated at all, whatever its Sharpe ratio came out as.
    if result.decisions:
        from .agents.risk import risk_contribution_spread, risk_dispersion
        from .core.returns import simple_returns
        from .core.risk import covariance_matrix
        window = simple_returns(book.panel.closes).dropna().iloc[-lookback:]
        cov = covariance_matrix(window)
        among = [c for c in book.panel.closes.columns if c in book.tradeable]
        # Every column, not just the held ones. `risk_decomposition` requires a
        # weight for each name in the covariance matrix, and a panel routinely
        # carries instruments with no position -- a watchlist entry, or one
        # sold since. Their weight is zero, which is a fact rather than a gap,
        # and omitting it raised `no weight supplied for [...]` and took the
        # whole backtest down on any book with a watchlist.
        columns = [str(c) for c in cov.columns]
        start = {k: float(book.weights.get(k, 0.0)) for k in columns}
        ending = result.weights.iloc[-1].to_dict()
        final = {k: float(ending.get(k, 0.0)) for k in columns}
        # Both numbers, and the same two names the allocator uses, so
        # "dispersion" means one thing in this project. The coefficient of
        # variation uses every holding; the range is decided by two of them
        # and is here because it says whether one holding is the problem.
        notes.append(
            f"risk-share dispersion across the {len(among)} tradeable "
            f"holdings: {risk_dispersion(start, cov, among=among):.2f} "
            f"for the book as it stands, "
            f"{risk_dispersion(final, cov, among=among):.2f} after the policy "
            f"(coefficient of variation, zero when the contributions are "
            f"equal). The range over the same mean goes "
            f"{risk_contribution_spread(start, cov, among=among):.2f} "
            f"to {risk_contribution_spread(final, cov, among=among):.2f}. "
            f"Lower is more equal; this is what the policy claims to do, "
            f"measured separately from whether doing it paid.")

    dominant = dominant_holding(book, warmup=warmup)
    if dominant is not None:
        notes.append(dominant)

    if refereed.adjustments:
        notes.append(f"{len(refereed.adjustments)} proposed trades were skipped "
                     f"as too small to cover their broker's fee; the first was "
                     f"{refereed.adjustments[0]}")
    if result.decisions:
        notes.append("policy said: " + result.decisions[-1].reason)
    from .core.taxes import CapitalGainsTax
    return compare(result, benchmark, cost_model=book.costs, trials=trials,
                   trial_sharpe_sd=trial_sharpe_sd, overlap=rebalance_every,
                   weights=book.weights, capital_gains=CapitalGainsTax(),
                   constraint_notes=tuple(notes))


def dominant_holding(book: Book, *, warmup: int = 0,
                     multiple: float = 3.0) -> "str | None":
    """`names_the_window` on this book's panel. See it for the reasoning."""
    closes = book.panel.closes.iloc[warmup:] if warmup else book.panel.closes
    names = {i: inst.display_name for i, inst in book.instruments.items()}
    return names_the_window(closes, names, multiple=multiple)


def names_the_window(closes: pd.DataFrame, names: "dict[str, str]", *,
                     multiple: float = 3.0) -> "str | None":
    """Name the holding that made the window, when one did.

    A backtest over a window in which one holding nearly doubled is not a
    backtest of a policy; it is a measurement of what that policy did about
    that holding. Equal risk contribution trims the most volatile holding, so
    a window whose return came from the most volatile holding is one it was
    always going to lose, and the reader is entitled to know that before
    reading the verdict rather than after.

    Computed rather than asserted: the holding is named only when its total
    return over the measured window exceeds `multiple` times the RUNNER-UP's,
    so on a book where nothing dominated nothing is claimed.

    Against the runner-up rather than against the median of the others, which
    was the first rule and was wrong in a way a test caught. On a book where
    two holdings returned 80% and 75% and the other two returned 5%, the
    median of the others is 5%, so the leader cleared three times it easily
    and the report named one of a pair as though it alone made the window. If
    two ran, no single one explains the result and naming either is choosing a
    story. The runner-up is floored at the median absolute return of the rest,
    so a leader that ran while everything else fell still counts.

    None when no holding stands out, which is the case a note would only
    clutter.
    """
    if closes.shape[0] < 2 or closes.shape[1] < 3:
        return None
    first = closes.ffill().bfill().iloc[0]
    last = closes.ffill().iloc[-1]
    total = ((last / first) - 1.0).dropna()
    if total.empty:
        return None
    leader = total.idxmax()
    others = total.drop(index=leader)
    if others.empty:
        return None
    best = float(total[leader])
    runner_up = max(float(others.max()), float(others.abs().median()))
    if best <= 0 or runner_up <= 0 or best < multiple * runner_up:
        return None
    return (
        f"one holding made this window: {names.get(leader, leader)} "
        f"({leader}) returned {best:+.1%} while the rest of the book ran "
        f"between {others.min():+.1%} and {others.max():+.1%}. That is what "
        f"the benchmark's Sharpe ratio is mostly measuring, and it is most of "
        f"why a policy that trims the most volatile holding lost: the most "
        f"volatile holding was the one that ran. Read the verdict against "
        f"that rather than as a property of the policy.")


def replay_the_ledger(book: Book, *, mode: str = "user",
                      data_root: "pathlib.Path | None" = None,
                      lookback: int = 252, warmup: int = 60,
                      on_sale: str = "prorata"):
    """Re-run the real purchases with only the destination changed.

    The dates and the amounts are the ledger's, so both arms spend the same
    money on the same days and neither pays extra turnover. What differs is
    where it went, which is the only thing the allocator chooses.
    """
    from .core.models import TransactionType
    from .eval.replay import replay_purchases

    root = pathlib.Path(data_root) if data_root else None
    store = DataStore.open(DataMode(mode), root=root)
    ledger = [t for t in store.load_transactions()
              if t.isin in book.panel.closes.columns]
    buys = [(t.date, t.isin, float(t.quantity)) for t in ledger
            if t.type is TransactionType.BUY]
    # Disposals are passed rather than filtered out. The replay stops at the
    # first one and says so; dropping them silently made the "actual" arm a
    # book that never sold, which is not what was bought.
    sales = [(t.date, t.isin, float(t.quantity)) for t in ledger
             if t.type is TransactionType.SELL]
    if not buys:
        raise ValueError(
            "the ledger has no purchases in instruments the panel covers, so "
            "there is nothing to replay.")
    return replay_purchases(book.panel.closes, sorted(buys),
                            costs=book.costs, buyable=book.buyable,
                            lookback=lookback, warmup=warmup,
                            sales=sorted(sales), on_sale=on_sale)


# --------------------------------------------------------------------------
# The bid-ask spread, per instrument, off each instrument's own bars
# --------------------------------------------------------------------------


# Beyond this, the lag-1 autocorrelation of the per-bar series is called
# large: the standard error assumes zero, simulated bars give -0.003, and
# the first real book gave -0.14 to -0.40 on four of seven instruments.
AUTOCORRELATION_LARGE = 0.10

# How many recent bars the tick grid is read off. Long enough that the
# candidate grids are distinguishable, short enough to sit inside the period
# since the instrument's last split -- splits are applied by the provider
# whatever the adjustment flag says, and they divide older prices by a ratio
# that takes them off any grid.
TICK_WINDOW = 250


def tick_for(prices: "pd.DataFrame") -> "object":
    """The venue tick grid, read off RECENT bars only.

    Two reasons, and only one of them is about splits.

    yfinance applies split adjustments whatever `auto_adjust` says, so a
    series spanning a split has its older section divided by the ratio and
    sitting on no grid at all. Reading the whole series would then find no
    tick and the instrument would be refused for having had a split, which is
    not a reason to refuse anything.

    And the MiFID II tick regime bands by price as well as by liquidity, so
    the tick that applies to a trade today is the one today's prices are on.
    A ten-year series can span several tick bands, and the average of them is
    not a tick.

    The grid is also the check on whether the right series arrived at all:
    dividend adjustment multiplies prices by a factor that is not a multiple
    of the tick, so a recent tail that is off grid means an adjusted series
    reached this code, and an adjusted price is one that never traded.
    """
    from .core.spread import infer_tick_size
    return infer_tick_size(prices.tail(TICK_WINDOW).to_numpy().ravel())


# A cached history whose last bar is older than this is topped up from the
# provider before it is used. Days, not trading days: a week covers any
# holiday run, and a survey that reads bars ending a fortnight ago is
# measuring a fortnight-old market without saying so.
STALE_BARS_DAYS = 5
# The incremental fetch starts this far before the last cached bar, so that
# a provider that revises its most recent rows overwrites them.
BARS_OVERLAP = dt.timedelta(days=7)


def _bars_for(symbol: str, provider, cache, *,
              today: "dt.date | None" = None) -> "pd.DataFrame":
    """Unadjusted OHLCV for `symbol`, through the cache, whole history.

    "max", not the two years the price loader takes. A spread is a property
    of the instrument and its market makers rather than of the holding
    period, and the noise floor thins as the fourth root of the sample: on
    250 bars nothing under about 9 bps resolves, which is most of a European
    ETF book. Raises whatever the provider raises on a fresh fetch; the
    caller decides whether that is a refusal or a failure.

    Three things a cache must not do here, each found by a reviewer probing
    the previous version rather than by a test:

    * Serve a short history for ever. The first wiring fetched two years and
      wrote every flag a later version checked, so "max" was never fetched
      and the survey believed it was reading the whole history. The period
      the rows were fetched with is now recorded, and anything not recorded
      as "max" is refetched as "max".
    * Never advance. A cached history with no refresh reads the same bars in
      March as in January, and the run log's purpose -- comparing two runs
      months apart -- would compare a feed against itself. The tail is
      topped up when the last bar is more than `STALE_BARS_DAYS` old. If the
      top-up fails, the cached rows are used and the survey prints the dates
      they span, so the staleness is visible rather than silent.
    * Return the provider's frame on the first run and the cache's copy on
      every later one. The cache drops rows with a missing price, so the
      two frames pair different days and the first run's estimate differs
      from every subsequent one. The cache's view is returned every time.
    """
    today = today or dt.date.today()
    cached = cache.get_bars(symbol)
    whole = (cached is not None and not cache.bars_predate_volume(symbol)
             and cache.bars_period(symbol) == "max")
    if not whole:
        cache.put_bars(symbol, provider.bars(symbol, period="max"),
                       period="max")
    else:
        last = cached[0].index[-1].date()
        if (today - last).days > STALE_BARS_DAYS:
            try:
                tail = provider.bars(symbol, start=last - BARS_OVERLAP)
            except Exception:                    # noqa: BLE001 - stale is visible, absent is not
                tail = None
            if tail is not None and not tail.empty:
                cache.put_bars(symbol, tail)
    stored = cache.get_bars(symbol)
    if stored is None or stored[0].empty:
        raise ValueError(f"no bars for {symbol} survived the cache: the "
                         f"provider returned rows without all four prices")
    return stored[0]


def _liquidity_proxy(frame: "pd.DataFrame") -> tuple[float | None, int]:
    """Median daily traded value, and how many bars it rests on.

    Median rather than mean because one index-rebalance day can carry a tenth
    of a small fund's annual volume, and a mean would rank the book by
    whether each fund happened to have had one. In the quote currency of the
    line, unconverted: it ranks instruments against each other and a book
    quoted in one currency loses nothing, but a line quoted in pence next to
    one in euros would rank a hundred times too high, and that is said where
    the number is printed rather than silently fixed with a rate this code
    does not have.

    None when the provider reported no volume, or fewer than 20 bars of it,
    or a median of zero. A zero median means the proxy is missing rather
    than that nothing trades; how often a provider reports it for a line
    that does trade is not measured here.
    """
    if "volume" not in frame.columns:
        return None, 0
    traded = (frame["volume"] * frame["close"]).dropna()
    if len(traded) < 20 or float(traded.median()) <= 0:
        return None, int(len(traded))
    return float(traded.median()), int(len(traded))


@dataclasses.dataclass(frozen=True)
class SpreadSurvey:
    """What every instrument's spread is, and how much of it is evidence.

    Two estimates per instrument, not one. `decisions` is the estimator on
    every bar the provider sent; `clean` is the same estimator with the bars
    `core.spread.classify_bars` set aside -- carried closes, carried whole
    bars, impossible bars, zero-volume bars -- and `quality` says how many of
    each there were. The two are printed side by side and the ranking check
    is run on both.

    The decision that would be written is still the one on every bar. The
    exclusion is a hypothesis about the feed, measured on simulated bars and
    not yet on this provider's, and the survey is where that hypothesis is
    tested rather than assumed: if the two columns disagree the difference
    is attributable to a named count of named bars. If they agree, the
    bars that are exact repeats did not matter -- and no more than that. A
    close that is stale without being a copy (the last print hours before
    the close, or a carried close nudged by a rounding) inflates the
    estimate by the same mechanism and is not counted, because from four
    prices alone it cannot be told from a genuine close.

    No significance is attached to the difference between the two columns.
    They are nested samples -- the clean bars are a subset of all of them --
    so the sum of their variances overstates the variance of the difference,
    and a single error bar can understate it by several times once the bars
    set aside are the ones that carried the bias. Neither is a z-score, so
    none is printed. Both estimates carry their own error bars; read the
    difference against those.
    """
    decisions: dict                      # isin -> agents.spreads.SpreadDecision
    ranking: object                      # agents.spreads.RankingCheck
    names: dict
    fallback_bps: float
    refused: dict                        # isin -> why no bars at all
    sweeps: dict = dataclasses.field(default_factory=dict)  # isin -> WindowSweep
    quality: dict = dataclasses.field(default_factory=dict)  # isin -> BarQuality
    clean: dict = dataclasses.field(default_factory=dict)    # isin -> SpreadDecision
    clean_sweeps: dict = dataclasses.field(default_factory=dict)
    clean_ranking: object = None         # RankingCheck on the clean estimates
    liquidity: dict = dataclasses.field(default_factory=dict)  # isin -> proxy
    liquidity_bars: dict = dataclasses.field(default_factory=dict)
    symbols: dict = dataclasses.field(default_factory=dict)  # isin -> symbol
    provider: str = ""
    # isin -> (ceiling a clean sample of the same length and volatility
    # would report at zero spread, the daily volatility used). Only for
    # instruments whose estimate did not resolve, because for those the
    # ceiling IS the claim and this is what says whether it is plausible.
    null_ceiling: dict = dataclasses.field(default_factory=dict)
    # isin -> (first date, last date, rows): which bars the estimate rests
    # on, printed so that a stale or short history is visible.
    spans: dict = dataclasses.field(default_factory=dict)

    @property
    def bars_set_aside(self) -> int:
        return sum(q.excluded_count for q in self.quality.values())

    def lines(self) -> list[str]:
        from .agents.spreads import ASSUMED, ESTIMATED

        out = ["Bid-ask spread by instrument", "=" * 74, "",
               "The largest component of the cost model and the only one that "
               "appears on no",
               "document. Estimated from each instrument's own open, high, "
               "low and close,",
               "twice: on every bar the provider sent, and again with these "
               "set aside: bars",
               "identical to the previous day's, closes identical to the "
               "previous close,",
               "bars whose open or close lies outside their own range, and "
               "bars with zero",
               "volume. Flat bars are counted and kept; the estimator already "
               "treats them",
               "as untraded, and setting them aside was measured to change "
               "nothing.", ""]
        for isin, d in sorted(self.decisions.items(),
                              key=lambda kv: -kv[1].half_spread_bps):
            label = self.names.get(isin, "")
            symbol = self.symbols.get(isin, "")
            out.append(f"{isin}  {label}{'  (' + symbol + ')' if symbol else ''}")
            span = self.spans.get(isin)
            if span is not None:
                first, last, rows = span
                out.append(f"    {rows} bars, {first} to {last}")
            out.append(f"    {d.half_spread_bps:6.1f} bps  [{d.source}]"
                       f"{'  CLAMPED TO TICK' if d.clamped_to_tick else ''}"
                       f"   on every bar")
            for line in d.reason.splitlines():
                out.append(f"           {line}")
            sweep = self.sweeps.get(isin)
            if sweep is not None and len(sweep.rungs) > 1:
                out.extend(f"      {line}" for line in sweep.lines())
            rho = None if d.estimate is None else d.estimate.autocorrelation
            if rho is not None:
                # The standard error treats the per-bar series as independent,
                # which was measured to hold on simulated bars. Real bars have
                # volatility clustering the simulation does not, so the number
                # is printed for every instrument and called out when it is
                # large enough to make the error bar optimistic. Reported
                # rather than corrected: raising `lags` on this evidence would
                # be fitting the standard error to one sample of one series.
                threshold = rho_threshold(d.estimate.usable_bars)
                loud = (f" -- LARGE, beyond {threshold:.2f} (three standard "
                        f"errors off {d.estimate.usable_bars} bars); the "
                        f"error bar above is optimistic"
                        if abs(rho) > threshold else "")
                out.append(f"           per-bar autocorrelation {rho:+.3f} "
                           f"(the error bar assumes ~0){loud}")
            allowed = self.null_ceiling.get(isin)
            if allowed is not None and d.estimate is not None \
                    and d.estimate.spread is not None:
                ceiling_bps, sigma = allowed
                from .agents.spreads import _upper_bound_bps
                upper = _upper_bound_bps(d.estimate, 2.0)
                if upper is not None and ceiling_bps > 0:
                    times = upper / ceiling_bps
                    # The ratio, and no cause. Carried bars were measured
                    # to inflate the estimate and RESOLVE it, not to widen
                    # the error bar; what widens it this much has not been
                    # simulated, so nothing here says what did.
                    loud = (" -- the error bar, not the spread, is what is "
                            "large: the per-bar series has\n           "
                            "variance the model does not produce, and this "
                            "survey has not measured what did"
                            if times > CEILING_SLACK else "")
                    out.append(f"           the ceiling is {times:.1f}x the "
                               f"{ceiling_bps:.1f} bps a clean sample of "
                               f"{d.estimate.usable_bars} bars at "
                               f"{sigma:.1%} a day\n           would put on "
                               f"a spread of zero{loud}")
            quality = self.quality.get(isin)
            if quality is not None:
                out.extend(f"    {line}" for line in quality.lines())
            c = self.clean.get(isin)
            if c is not None and quality is not None and quality.excluded_count:
                out.append(f"    {c.half_spread_bps:6.1f} bps  [{c.source}]"
                           f"{'  CLAMPED TO TICK' if c.clamped_to_tick else ''}"
                           f"   with those bars set aside")
                for line in c.reason.splitlines():
                    out.append(f"           {line}")
                if d.estimate is not None and c.estimate is not None \
                        and d.estimate.spread and c.estimate.spread:
                    ratio = d.half_spread_bps / c.half_spread_bps \
                        if c.half_spread_bps else float("inf")
                    out.append(f"           the bars set aside move the number "
                               f"charged by a factor of {ratio:.2f}; the two "
                               f"samples are nested,")
                    out.append(f"           so no significance is attached -- "
                               f"read it against the two error bars")
            elif c is not None and quality is not None:
                out.append("    nothing set aside, so the second estimate is "
                           "the first")
            proxy = self.liquidity.get(isin)
            with_figure = self.liquidity_bars.get(isin, 0)
            if proxy is None:
                out.append(f"    liquidity proxy: none -- the provider "
                           f"reported no usable volume for this line "
                           f"({with_figure} bars carry a figure), so it is "
                           f"not in the ranking check")
            else:
                from .agents.spreads import _money
                out.append(f"    liquidity proxy: median daily traded value "
                           f"{_money(proxy)} over {with_figure} bars, from "
                           f"the provider's volume, in the line's quote "
                           f"currency")
            out.append("")

        for isin, why in sorted(self.refused.items()):
            out.append(f"{isin}  {self.names.get(isin, '')}")
            out.append(f"    no bars: {why}")
            out.append("")

        from .agents.spreads import BOUNDED, OBSERVED

        by_tier = {tier: [i for i, d in self.decisions.items()
                          if d.source == tier]
                   for tier in (OBSERVED, ESTIMATED, BOUNDED, ASSUMED)}
        total = len(self.decisions) + len(self.refused)
        measured = by_tier[OBSERVED] + by_tier[ESTIMATED]
        bounded = by_tier[BOUNDED]
        constant = by_tier[ASSUMED] + list(self.refused)

        out.append("-" * 74)
        out.append(f"Of {total} instruments: {len(measured)} carry a measured "
                   f"spread, {len(bounded)} carry an upper")
        out.append(f"bound from their own data, and {len(constant)} were not "
                   f"measurable at all and keep")
        out.append(f"the declared {self.fallback_bps:.0f} bps.")

        own = measured + bounded
        if own:
            got = [self.decisions[i].half_spread_bps for i in own]
            if min(got) > 0:
                out.append(f"The numbers coming from the instruments' own "
                           f"data run {min(got):.1f} to {max(got):.1f} bps, a "
                           f"factor of")
                out.append(f"{max(got) / min(got):.1f} that a single constant "
                           f"could not express -- which is the whole reason "
                           f"to measure.")
        if bounded:
            over = [i for i in bounded
                    if self.decisions[i].half_spread_bps > self.fallback_bps]
            if over:
                # Not "which is the safe direction". A ceiling is a statement
                # about the bars under it and nothing else: on bars that
                # carry a close forward it is wrong in the direction the carry
                # sends it, and that direction is up. A number that is wrong
                # in a direction one likes is still wrong.
                out.append(f"{len(over)} of the {len(bounded)} ceilings "
                           f"{'sits' if len(over) == 1 else 'sit'} above the "
                           f"{self.fallback_bps:.0f} bps constant. That is "
                           f"not reassurance: a")
                out.append("ceiling is only as good as the bars under it, and "
                           "a carried close pushes it")
                out.append("up. Whether these bars can be believed is what "
                           "the bars set aside and the")
                out.append("ranking check below are for.")
        if bounded and not measured:
            out.append("Nothing resolved. Every number above is a ceiling "
                       "rather than a reading,")
            out.append("which is still per instrument and still beats one "
                       "constant for all of them,")
            out.append("but no spread here has been measured and none should "
                       "be quoted as one.")
        out.append("")
        out.append("The liquidity proxy is the provider's own volume times "
                   "the close, median over")
        out.append("the history, unconverted between currencies. A "
                   "provider's volume for one")
        out.append("listing of a fund may be another listing's, or absent; "
                   "nothing here checks it,")
        out.append("which is why it is printed per instrument above, so that "
                   "a failed ranking can")
        out.append("be laid at the right number.")
        out.append("")
        out.append("On every bar the provider sent:")
        out.extend(self.ranking.line(self.names).splitlines())
        if self.clean_ranking is not None:
            out.append("")
            if self.bars_set_aside:
                out.append(f"With the {self.bars_set_aside} bars set aside "
                           f"above excluded:")
                out.extend(self.clean_ranking.line(self.names).splitlines())
            else:
                out.append("No bars were set aside on any instrument, so the "
                           "second ranking check is the first.")
        return out

    def record(self) -> dict:
        """Everything above as data, for the run log.

        A failed check with its numbers preserved is a result; a failed check
        whose numbers scrolled off a terminal is an anecdote.
        """
        def estimate(e) -> dict | None:
            if e is None:
                return None
            return {"half_spread_bps": e.half_spread_bps,
                    "error_bps": e.half_spread_error_bps,
                    "t": e.t_statistic, "signed_square": e.signed_square,
                    "square_standard_error": e.square_standard_error,
                    "usable_bars": e.usable_bars,
                    "autocorrelation": e.autocorrelation,
                    "refusal": e.refusal or None}

        def decision(d, sweep) -> dict | None:
            if d is None:
                return None
            return {"charged_bps": d.half_spread_bps, "source": d.source,
                    "clamped_to_tick": d.clamped_to_tick,
                    "estimate": estimate(d.estimate),
                    "window_bars": None if sweep is None else sweep.chosen_bars,
                    "drifted_at": None if sweep is None else sweep.drifted_at}

        def ranking(r) -> dict | None:
            if r is None:
                return None
            return {"rho": r.rho, "instruments": r.instruments,
                    "passed": r.passed,
                    "pairs": [list(p) for p in r.pairs]}

        instruments = {}
        for isin, d in self.decisions.items():
            q = self.quality.get(isin)
            allowed = self.null_ceiling.get(isin)
            span = self.spans.get(isin)
            instruments[isin] = {
                "name": self.names.get(isin, ""),
                "symbol": self.symbols.get(isin),
                "bars": None if span is None else {
                    "first": str(span[0]), "last": str(span[1]),
                    "rows": span[2]},
                "every_bar": decision(d, self.sweeps.get(isin)),
                "bars_set_aside": None if q is None else q.counts(),
                "clean": decision(self.clean.get(isin),
                                  self.clean_sweeps.get(isin)),
                "null_ceiling_bps": None if allowed is None else allowed[0],
                "daily_volatility": None if allowed is None else allowed[1],
                "liquidity": self.liquidity.get(isin),
                "liquidity_bars": self.liquidity_bars.get(isin, 0)}
        return {"provider": self.provider,
                "fallback_bps": self.fallback_bps,
                "instruments": instruments,
                "refused": dict(self.refused),
                "ranking": ranking(self.ranking),
                "clean_ranking": ranking(self.clean_ranking)}


def record_survey(survey: SpreadSurvey, path: "pathlib.Path") -> int:
    """Append the survey to a JSON-lines log and return its line number.

    Every run, passed or failed, and the failed ones are the point: the
    survey that found the most liquid holding estimated widest refused to
    write and then existed only in a terminal. Appended rather than
    overwritten so that two runs a month apart can be compared, which is the
    only way a slow change in a feed would ever be seen.
    """
    import datetime as _dt
    import json
    entry = {"recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
        timespec="seconds")}
    entry.update(survey.record())
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        # allow_nan=False: json would otherwise write a bare `NaN` token,
        # which is not JSON and which a reader in another language rejects.
        # `_json_ready` has already turned every non-finite float into null
        # and every numpy scalar into a Python one, so this is a tripwire.
        fh.write(json.dumps(_json_ready(entry), allow_nan=False) + "\n")
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _json_ready(value):
    """The record with numpy scalars unwrapped and non-finite floats as null."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def survey_spreads(book: Book, *, mode: str = "user",
                   data_root: "pathlib.Path | None" = None,
                   provider="yfinance",
                   fallback_bps: float = 8.0) -> SpreadSurvey:
    """Estimate the spread for every instrument in the book.

    Fetches unadjusted OHLC through the cache, runs EDGE on each instrument
    separately, infers each one's tick from its own prices, and applies the
    three-tier rule. Nothing is written to the instrument records here; that
    is `portfolio spreads --write`, so that looking is not the same action as
    committing.

    Every instrument is estimated on its own bars and nothing is pooled. Two
    funds tracking the same index on two venues have different spreads, and
    the venue is the reason.
    """
    from .agents.spreads import decide_spread, ranking_is_plausible
    from .core.spread import classify_bars, exclude_bars, sweep_windows
    from .data.cache import PriceCache
    from .data.market import MarketData

    root = pathlib.Path(data_root) if data_root else None
    store = DataStore.open(DataMode(mode), root=root)
    market_provider = _provider_named(provider)

    cache_dir = store.root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = PriceCache(cache_dir / f"bars-{market_provider.name}.sqlite")
    # Only to resolve the fetch handle: the same ISIN-to-symbol rule the price
    # loader uses, rather than a second one that could drift from it.
    resolver = MarketData(market_provider, cache)
    decisions, refused, spreads, liquidity, names = {}, {}, {}, {}, {}
    sweeps: dict = {}
    quality: dict = {}
    clean: dict = {}
    clean_sweeps: dict = {}
    clean_spreads: dict = {}
    liquidity_bars: dict = {}
    symbols: dict = {}
    null_ceiling: dict = {}
    spans: dict = {}

    for isin, inst in book.instruments.items():
        names[isin] = getattr(inst, "name", "") or ""
        symbol = resolver.symbol_for(inst)
        if not symbol:
            refused[isin] = "no provider symbol is recorded for it"
            continue
        symbols[isin] = symbol
        try:
            frame = _bars_for(symbol, market_provider, cache)
        except Exception as exc:
            refused[isin] = f"{type(exc).__name__}: {exc}"
            continue

        prices = frame[["open", "high", "low", "close"]]
        o, h, l, c = (prices[k].to_numpy() for k in ("open", "high", "low",
                                                     "close"))
        spans[isin] = (frame.index[0].date(), frame.index[-1].date(),
                       int(len(frame)))
        sweep = sweep_windows(o, h, l, c)
        estimate = sweep.chosen
        tick = tick_for(prices)
        price = float(frame["close"].iloc[-1])
        if not tick.usable:
            # Off-grid recent prices mean the series has been adjusted, and an
            # adjusted series is the wrong input to the estimator too, not
            # merely to the tick inference. Refuse rather than charge a number
            # derived from prices that never traded.
            # The reason is whatever the residuals support, and no more. The
            # previous version asserted "the series has been adjusted for
            # distributions" on every miss. On a real book that fired on four
            # accumulating ETFs, which have never made a distribution, while
            # the one dividend-paying holding had the HIGHEST fit of the seven
            # -- the ordering was backwards and it sent the reader to look at
            # dividends. Same defect as the refusal that explained a property
            # fund as a debt security: a confident, specific, untested cause
            # attached to a real refusal.
            refused[isin] = (
                f"the last {min(len(prices), TICK_WINDOW)} bars do not sit on "
                f"any tick grid this can establish. {tick.why_not_a_grid()}")
            continue
        # An already-recorded observed spread wins, and the estimate is
        # reported beside it rather than discarded: two independent readings
        # of the same quantity are worth comparing, and `--write` must never
        # replace something somebody watched with something inferred.
        watched = (float(inst.half_spread_bps)
                   if getattr(inst, "spread_observed", False)
                   and inst.half_spread_bps is not None else None)
        decision = decide_spread(estimate, price=price, tick=tick,
                                 fallback_bps=fallback_bps,
                                 observed_bps=watched)
        decisions[isin] = decision
        sweeps[isin] = sweep
        if decision.is_evidence:
            spreads[isin] = decision.half_spread_bps

        # For an estimate that did not resolve, the ceiling is the claim,
        # and a ceiling can be wrong: a sample of this length and
        # volatility can only report so wide an error bar under the model.
        # What it can report is simulated and printed beside it. The
        # volatility is measured on the window the sweep chose, not on the
        # whole history: the sweep truncates exactly when the tail differs
        # from the history, and a reference at the history's volatility
        # would then be a sample of the right length from the wrong market.
        if (estimate.spread is not None and not estimate.resolved()
                and estimate.usable_bars > 0):
            sigma = daily_volatility(c[-sweep.chosen_bars:])
            if sigma:
                null_ceiling[isin] = (
                    null_ceiling_bps(estimate.usable_bars, sigma), sigma)

        # The same estimator again, with the bars a feed manufactures set
        # aside. Which bars, and what each kind was measured to do to the
        # estimate, is in core/spread.py above `classify_bars`; the survey's
        # job is to run both and print both, so that a disagreement between
        # them is attributable to a count rather than to a suspicion.
        volume = frame["volume"].to_numpy() if "volume" in frame.columns else None
        bars = classify_bars(o, h, l, c, volume)
        quality[isin] = bars
        if bars.excluded_count:
            kept = exclude_bars(o, h, l, c, bars.excluded)
            clean_sweep = sweep_windows(*kept)
        else:
            clean_sweep = sweep
        clean_sweeps[isin] = clean_sweep
        clean[isin] = decide_spread(clean_sweep.chosen, price=price, tick=tick,
                                    fallback_bps=fallback_bps,
                                    observed_bps=watched)
        if clean[isin].is_evidence:
            clean_spreads[isin] = clean[isin].half_spread_bps

        # Liquidity for the ranking check: median daily traded value, which
        # is independent of anything the estimator saw. Kept per instrument
        # and printed, so that a failed ranking is attributable to the
        # number it was ranked by and not only to the number it ranked.
        proxy, on_bars = _liquidity_proxy(frame)
        liquidity_bars[isin] = on_bars
        if proxy is not None:
            liquidity[isin] = proxy

    cache.close()
    return SpreadSurvey(decisions=decisions,
                        ranking=ranking_is_plausible(spreads, liquidity),
                        names=names, fallback_bps=fallback_bps,
                        refused=refused, sweeps=sweeps, quality=quality,
                        clean=clean, clean_sweeps=clean_sweeps,
                        clean_ranking=ranking_is_plausible(clean_spreads,
                                                           liquidity),
                        liquidity=liquidity, liquidity_bars=liquidity_bars,
                        symbols=symbols, provider=market_provider.name,
                        null_ceiling=null_ceiling, spans=spans)


def _provider_named(provider):
    """A provider by name, or the provider object handed in.

    Accepting an object is what lets a test hand the survey a provider that
    contaminates its bars on purpose, which is the only way to see the
    two-column survey and the ranking check disagree without a network.
    """
    if not isinstance(provider, str):
        return provider
    if provider == "fixture":
        from .data.providers.fixture import FixtureProvider
        return FixtureProvider()
    from .data.providers.yahoo import YahooProvider
    return YahooProvider()


# --------------------------------------------------------------------------
# The ladder: the estimator on instruments whose spread is not in doubt
# --------------------------------------------------------------------------
#
# Why this exists
# ---------------
# Six controls establish that the estimator recovers a spread it was not told
# from simulated bars. None of them establishes that the bars this provider
# sends are the bars the estimator's derivation assumes, and the first real
# book it was run on came back with the most liquid holding estimated widest.
# The estimator-plus-provider path had never been controlled, only the
# estimator.
#
# So: run the whole path -- provider, cache, window sweep, tick, decision --
# on instruments whose half-spread is known independently to within a factor
# of two, and see which end of the path the fault is at.
#
#   * SPY and AAPL on their home venues: about one basis point, and their
#     daily bars are the most scrutinised data in finance. If the path reads
#     them at 40 bps, the pipeline is wrong and nothing about Europe is
#     learned.
#   * SU.PA and MEUD.PA on Euronext Paris: a CAC 40 mega-cap at one or two
#     bps and a large core tracker at a few. If the US lines read right and
#     these read at 40, the fault is in this provider's European bars, and
#     the counts of bars set aside per rung say which kind.
#   * A thin European ETF at tens of bps: the rung that checks the estimator
#     is not reading *every* European line at the same number. Without it a
#     path that returned 25 bps for everything would pass the other four.
#
# The bands are the figures the user stated, with room either side so that
# only a gross disagreement fails: the real-book failure was a factor of ten
# to twenty, and a band that failed on a factor of 1.5 would be measuring the
# band's author. They are printed next to every result so a disagreement is
# with a visible number.
#
# What this cannot do is run offline. It is a control on a data path, and a
# data path with the data replaced by a fixture is a different path.
# `--provider fixture` runs it on invented bars whose spreads bear no relation
# to the bands, is expected to fail, and is kept because a control that
# cannot be seen failing has not been shown to check anything.


@dataclasses.dataclass(frozen=True)
class Rung:
    """One instrument whose half-spread is known well enough to check against.

    `low_bps` and `high_bps` are the band the verdict tests against, wider
    than the figure stated for the instrument so that only a gross
    disagreement fails. `stated` is that figure, as given, printed beside
    the band so that a reading inside the band but outside what was stated
    is seen for what it is.
    """
    symbol: str
    label: str
    low_bps: float
    high_bps: float
    region: str                          # "US", "EU" or "thin"
    stated: str = ""

    @property
    def band(self) -> str:
        return f"{self.low_bps:g} to {self.high_bps:g} bps"


LADDER: tuple[Rung, ...] = (
    Rung("SPY", "SPDR S&P 500, NYSE Arca", 0.1, 2.0, "US", "about 1 bp"),
    Rung("AAPL", "Apple, Nasdaq", 0.2, 3.0, "US", "1 to 2 bps"),
    Rung("SU.PA", "Schneider Electric, Euronext Paris", 0.3, 4.0, "EU",
         "1 to 2 bps"),
    Rung("MEUD.PA", "Amundi Stoxx Europe 600, Euronext Paris", 0.5, 8.0, "EU",
         "a few bps"),
    # The fund the book holds, on its Amsterdam line. The band is the user's
    # own figure for it, 15 to 30 bps, with room either side. The symbol is
    # the one the book records for the fund and has not been verified from
    # the machine this was written on; `--rung` overrides it.
    Rung("IPRP.AS", "iShares European Property Yield, Euronext Amsterdam",
         8.0, 40.0, "thin", "15 to 30 bps"),
)

# The same fund on two venues: IE00BMC38736, the VanEck Semiconductor UCITS
# ETF, on Xetra and on the London Stock Exchange's USD line. The estimate is
# per line and the venue is a real reason for two lines to differ --
# different market makers, different tick, different session -- so
# agreement within error bars is not required. Two resolved lines more
# than `PAIR_RATIO` apart at more than `PAIR_SIGMA` are called inconsistent;
# both thresholds are choices, printed with the result, and what two venues
# legitimately differ by for this fund is not measured here.
PAIRS: tuple[tuple[str, str], ...] = (("VVSM.DE", "SMH.L"),)

# Both, because a ratio of 1.3 at ten sigma is a venue difference measured
# precisely, and a ratio of 4 at one sigma is noise.
PAIR_SIGMA = 3.0
PAIR_RATIO = 3.0


def null_ceiling_bps(bars: int, sigma_day: float, spread_bps: float = 0.0, *,
                     runs: int = 60, seed: int = 0,
                     significance: float = 2.0) -> float:
    """The ceiling a sample of this length and volatility puts on a spread.

    Simulated rather than looked up: the noise floor was measured at 3.97
    bps for 500 bars at 1% daily volatility, and it scales with both, but a
    formula fitted to that table would be one more constant to keep right.
    `runs` draws of the same market the controls use, at `spread_bps` true
    half-spread, and the median of the ceilings they report. Sixty draws
    rather than three because the reference gates a verdict at a factor of
    three: at three draws it sat up to a third from its converged value on
    a fixed seed, which is enough to turn a two-fold ratio into the
    three-fold one that fails a rung. Sixty draws cost half a second at
    5000 bars.

    What it is for: an estimate that fails to resolve is a ceiling, and a
    ceiling can be wrong too. Off 5000 bars at 1.5% a day, a one-bp
    instrument reports a ceiling of about 7 bps. The real book reported 60
    on such an instrument, eight times that, which is not a wide spread but
    a per-bar series with variance the model does not produce. What
    produces such a series has not been simulated -- carried bars were
    tried and inflate the estimate without widening its error bar -- so
    this number says how far the ceiling is from what the sample allows,
    and nothing about why.

    >>> round(null_ceiling_bps(500, 0.01), 0) in (5.0, 6.0, 7.0)
    True
    >>> null_ceiling_bps(4000, 0.01) < null_ceiling_bps(500, 0.01)
    True
    """
    from .core.spread import edge
    from .eval.spread_controls import simulate_bars
    ceilings = []
    for i in range(runs):
        o, h, l, c = simulate_bars(2.0 * spread_bps / 10_000.0,
                                   bars=max(int(bars), 3), sigma=sigma_day,
                                   seed=seed + i)
        e = edge(o, h, l, c, minimum_bars=0)
        if e.signed_square is None or not e.square_standard_error:
            continue
        top = e.signed_square + significance * e.square_standard_error
        ceilings.append(math.sqrt(max(top, 0.0)) * 10_000.0 / 2.0)
    return float(np.median(ceilings)) if ceilings else float("nan")


# An unresolved rung's ceiling may exceed the ceiling its sample allows by
# this factor before it is called wide. Real bars have volatility clustering
# the reference simulation does not, and a clean clustered sample's ceiling
# sits above the reference: measured with a log-AR(1) volatility path at
# 0.97 persistence and a log-volatility spread of 0.35 to 0.7, the median
# ratio runs 1.1 to 1.7 and the largest of a dozen draws 2.4 at the
# heaviest clustering, once the volatility is measured as a winsorised RMS.
# With a median-based volatility the reference sat lower and the largest
# ratio reached 3.7, which is why the RMS. The failure this exists to catch
# was a factor of eight on the ceiling.
CEILING_SLACK = 3.0


def rung_verdict(estimate, rung: Rung, *, sigma_day: float | None = None,
                 significance: float = 2.0
                 ) -> tuple[str, str, float | None, float | None]:
    """Is this estimate consistent with the rung's band?

    On the square, where the sampling distribution is symmetric: the interval
    is ``sqrt(s^2 -+ k SE(s^2))`` rooted, and the band is missed when the
    whole interval lies outside it.

    That alone is a dead check for the case it exists for, and the test that
    found it is `test_a_wide_reading_on_a_tight_instrument_fails_on_its_floor`.
    The real book's 41.85 bps on a one-bp instrument came with t = 1.93:
    unresolved, so its interval reaches down to zero and "consistent" with
    any band. What is wrong with it is not the point estimate but the
    **ceiling**: off thousands of bars, a sample cannot put a 42 bps ceiling
    on a tight spread unless its per-bar series has variance the model does
    not produce. So when the estimate is unresolved and `sigma_day` is
    given, the ceiling is compared against the one a sample of the same
    length and volatility reports at the top of the band, and a ceiling
    more than `CEILING_SLACK` times that is called wide.

    Returns ``(status, detail, lower_bps, upper_bps)``.

    Without `sigma_day` an unresolved estimate cannot have its ceiling
    checked, and the status says so -- `unchecked` -- rather than passing,
    because a pass that was never able to fail is the thing this project
    keeps finding.

    >>> from portfolio.core.spread import SpreadEstimate
    >>> spy = Rung("SPY", "", 0.1, 2.0, "US")
    >>> tight = SpreadEstimate(0.00012, 1.4e-8, 0.00004, 5000, 4999,
    ...                        square_standard_error=1.0e-8)
    >>> rung_verdict(tight, spy, sigma_day=0.01)[0]
    'consistent'
    >>> rung_verdict(tight, spy)[0]
    'unchecked'
    >>> wide = SpreadEstimate(0.0080, 6.4e-5, 0.0004, 5000, 4999,
    ...                       square_standard_error=6.0e-6)
    >>> rung_verdict(wide, spy)[0]
    'too wide'
    >>> thin = Rung("X", "", 8.0, 40.0, "thin")
    >>> rung_verdict(tight, thin)[0]
    'too narrow'
    >>> rung_verdict(SpreadEstimate(None, None, None, 10, 9,
    ...                             refusal="only 9 usable bars"), spy)[0]
    'no estimate'
    """
    if estimate is None or estimate.spread is None:
        why = "no bars" if estimate is None else estimate.refusal
        return "no estimate", f"no estimate: {why}", None, None
    s2 = estimate.signed_square
    se = estimate.square_standard_error or 0.0
    lower = math.sqrt(max(s2 - significance * se, 0.0)) * 10_000.0 / 2.0
    top = s2 + significance * se
    upper = math.sqrt(top) * 10_000.0 / 2.0 if top > 0 else None
    point = estimate.half_spread_bps
    if lower > rung.high_bps:
        return ("too wide",
                f"{point:.1f} bps, and at least {lower:.1f} at "
                f"{significance:.0f} standard errors; the band is "
                f"{rung.band}", lower, upper)
    if upper is None:
        return ("too narrow",
                f"the squared spread is negative beyond {significance:.0f} "
                f"standard errors, so no positive spread fits, against a "
                f"band of {rung.band}", lower, upper)
    if upper < rung.low_bps:
        return ("too narrow",
                f"at most {upper:.1f} bps at {significance:.0f} standard "
                f"errors; the band is {rung.band}", lower, upper)
    if estimate.resolved(significance):
        return ("consistent", f"{point:.1f} bps, resolved, inside the band "
                              f"{rung.band}", lower, upper)
    if sigma_day is not None and sigma_day > 0 and estimate.usable_bars > 0:
        allowed = null_ceiling_bps(estimate.usable_bars, sigma_day,
                                   rung.high_bps, significance=significance)
        if allowed == allowed and upper > CEILING_SLACK * allowed:
            return ("too wide",
                    f"not resolved, and the ceiling of {upper:.1f} bps is "
                    f"{upper / allowed:.0f}x the {allowed:.1f} that "
                    f"{estimate.usable_bars} bars at {sigma_day:.1%} a day "
                    f"put on a spread at the top of the band ({rung.band}). "
                    f"The error bar, not the estimate, is what is wrong: "
                    f"the per-bar series has variance the model does not "
                    f"produce", lower, upper)
        return ("consistent",
                f"not resolved; at most {upper:.1f} bps, against the "
                f"{allowed:.1f} a clean sample this size would report at "
                f"the top of the band {rung.band}", lower, upper)
    return ("unchecked",
            f"not resolved; at most {upper:.1f} bps, inside the band "
            f"{rung.band}, but the ceiling could not be checked against "
            f"what the sample allows because the daily volatility is not "
            f"estimable", lower, upper)


# Returns beyond this many scale units are clipped before the volatility is
# taken, so that one impossible bar does not set it for the whole series.
VOLATILITY_CLIP = 5.0


def daily_volatility(close: np.ndarray) -> float | None:
    """Close-to-close log volatility, robust to the odd impossible bar.

    A winsorised root mean square: returns are clipped at `VOLATILITY_CLIP`
    times a median-based scale and the RMS of what remains is taken. The
    clip is what keeps a feed that produced one bar at a hundredth of the
    price from setting the volatility for the whole series. The RMS, rather
    than the median-based scale on its own, is because the per-bar series
    the ceiling is built from has variance scaling with the fourth moment
    of returns: under volatility clustering the median-based scale sits at
    0.6 to 0.8 of the RMS, the reference ceiling built from it sits as far
    below where it should, and every ratio printed against it is inflated
    in the direction that fails a rung.

    Where the median scale is zero -- more than half the closes repeat the
    previous close, which a thin line on a coarse grid does -- the RMS of
    all returns is the scale, so the check is not silently disabled on the
    instruments it most exists for. None only when there are too few closes
    or no movement at all.

    >>> rng = np.random.default_rng(0)
    >>> closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1000)))
    >>> 0.009 < daily_volatility(closes) < 0.011
    True
    >>> spiked = closes.copy(); spiked[500] = 1.0     # one impossible bar
    >>> 0.009 < daily_volatility(spiked) < 0.012
    True
    >>> daily_volatility(np.full(50, 7.5)) is None
    True
    """
    c = np.asarray(close, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size < 20:
        return None
    returns = np.diff(np.log(c))
    mad = float(np.median(np.abs(returns - np.median(returns))))
    scale = 1.4826 * mad if mad > 0 else float(np.sqrt(np.mean(returns ** 2)))
    if scale <= 0:
        return None
    clipped = np.clip(returns, -VOLATILITY_CLIP * scale, VOLATILITY_CLIP * scale)
    rms = float(np.sqrt(np.mean(clipped ** 2)))
    return rms if rms > 0 else None


@dataclasses.dataclass(frozen=True)
class RungResult:
    rung: Rung
    status: str                          # consistent / too wide / too narrow / no estimate / unavailable
    detail: str
    sweep: object = None                 # WindowSweep
    quality: object = None               # BarQuality
    clean: object = None                 # SpreadEstimate with bars set aside
    clean_status: str = ""
    clean_detail: str = ""
    lower_bps: float | None = None
    upper_bps: float | None = None

    @property
    def estimate(self):
        """The estimate the verdict was reached on, or None."""
        return None if self.sweep is None else self.sweep.chosen

    @property
    def autocorrelation(self) -> float | None:
        e = self.estimate
        return None if e is None else e.autocorrelation

    def lines(self) -> list[str]:
        mark = {"consistent": "ok  ", "too wide": "WIDE", "too narrow": "NARR",
                "no estimate": "----", "unavailable": "----",
                "unchecked": "?   "}[self.status]
        stated = f" (stated: {self.rung.stated})" if self.rung.stated else ""
        out = [f"[{mark}] {self.rung.symbol:<9} {self.rung.label}",
               f"       band {self.rung.band}{stated}: {self.detail}"]
        e = self.estimate
        if e is not None and self.sweep.rungs:
            out.append(f"       {e.usable_bars} usable bars in the window "
                       f"taken, of {self.sweep.available_bars} available")
        if e is not None and e.autocorrelation is not None:
            out.append(f"       per-bar autocorrelation "
                       f"{_rho_text(e.autocorrelation, e.usable_bars)}")
        if self.sweep is not None and len(self.sweep.rungs) > 1:
            out.extend(f"       {line}" for line in self.sweep.lines())
        if self.quality is not None:
            out.extend(f"       {line}" for line in self.quality.lines())
            if self.quality.excluded_count and self.clean is not None:
                rho = self.clean.autocorrelation
                out.append(f"       with those set aside: {self.clean_detail} "
                           f"-> {self.clean_status}"
                           + (f"; per-bar autocorrelation "
                              f"{_rho_text(rho, self.clean.usable_bars)}"
                              if rho is not None else ""))
        return out


def rho_threshold(usable_bars: int) -> float:
    """Beyond this, a per-bar autocorrelation is more than sampling noise.

    A lag-1 autocorrelation estimated from ``n`` terms has standard error
    about ``1/sqrt(n)`` under the null, so the flat `AUTOCORRELATION_LARGE`
    is raised to three standard errors where the sample is short. Off 500
    bars that is 0.13: the fixture's AAPL read -0.104 there, two sigma of
    nothing, and a flat threshold would have called the path guilty on it.
    The book's -0.40 on a similar count is nine.

    >>> round(rho_threshold(5000), 3), round(rho_threshold(500), 3)
    (0.1, 0.134)
    """
    if usable_bars <= 0:
        return AUTOCORRELATION_LARGE
    return max(AUTOCORRELATION_LARGE, 3.0 / math.sqrt(usable_bars))


def _rho_text(rho: float, usable_bars: int) -> str:
    """The autocorrelation with its verdict, the same words the survey uses."""
    threshold = rho_threshold(usable_bars)
    loud = (f" -- LARGE, beyond {threshold:.2f} (three standard errors off "
            f"{usable_bars} bars); the error bar assumes ~0"
            if abs(rho) > threshold else "")
    return f"{rho:+.3f}{loud}"


@dataclasses.dataclass(frozen=True)
class PairResult:
    symbols: tuple[str, str]
    estimates: tuple                     # (SpreadEstimate | None, SpreadEstimate | None)
    qualities: tuple = (None, None)
    z: float | None = None
    ratio: float | None = None
    detail: str = ""
    consistent: bool | None = None       # None when not estimable on both

    def lines(self) -> list[str]:
        a, b = self.symbols
        out = [f"{a} against {b}, the same fund on two venues:"]
        for symbol, e, q in zip(self.symbols, self.estimates, self.qualities):
            described = "no bars" if e is None else e.describe()
            if e is not None and e.autocorrelation is not None:
                described += (f"; per-bar autocorrelation "
                              f"{_rho_text(e.autocorrelation, e.usable_bars)}")
            out.append(f"       {symbol:<9} {described}")
            if q is not None:
                out.append(f"                 {q.lines()[0]}")
        out.append(f"       {self.detail}")
        return out


@dataclasses.dataclass(frozen=True)
class LadderReport:
    rungs: tuple[RungResult, ...]
    pairs: tuple[PairResult, ...]
    verdict: str
    passed: bool
    provider: str

    def lines(self) -> list[str]:
        out = ["Spread ladder: the estimator on instruments whose spread is "
               "not in doubt",
               "=" * 74, "",
               f"Provider: {self.provider}. Every rung runs the path the "
               f"survey estimates by --",
               "bars through the cache, the window sweep, the bars set aside, "
               "the ceiling",
               "against what the sample allows -- on an instrument whose "
               "half-spread is known",
               "to within a factor of two. The tick floor and the tier "
               "decision are not run;",
               "they act on the estimate and cannot make a wrong one right. "
               "The band is",
               "printed beside each result, so a disagreement is with a "
               "number.", ""]
        for r in self.rungs:
            out.extend(r.lines())
            out.append("")
        for p in self.pairs:
            out.extend(p.lines())
            out.append("")
        out.extend(self.table())
        out.append("")
        out.append("-" * 74)
        out.extend(self.verdict.splitlines())
        note = _autocorrelation_note(self.rungs)
        if note:
            out.append("")
            out.extend(note.splitlines())
        return out

    def table(self) -> list[str]:
        """One row per rung: the estimate and, beside it, what the survey
        could not reproduce on any simulated bars -- the per-bar
        autocorrelation. SPY and AAPL are the rows that decide whether the
        path manufactures it."""
        out = [f"  {'rung':<9} {'estimate':>9} {'t':>6} {'ceiling':>8} "
               f"{'per-bar rho':>11} {'set aside':>9}  status"]
        for r in self.rungs:
            e = r.estimate
            if e is None or e.spread is None:
                out.append(f"  {r.rung.symbol:<9} {'-':>9} {'-':>6} {'-':>8} "
                           f"{'-':>11} {'-':>9}  {r.status}")
                continue
            ceiling = "-" if r.upper_bps is None else f"{r.upper_bps:.1f}"
            rho = "-" if e.autocorrelation is None else f"{e.autocorrelation:+.3f}"
            aside = ("-" if r.quality is None
                     else f"{r.quality.excluded_count / max(r.quality.bars, 1):.1%}")
            out.append(f"  {r.rung.symbol:<9} {e.half_spread_bps:7.1f} bps "
                       f"{e.t_statistic:+6.2f} {ceiling:>8} {rho:>11} "
                       f"{aside:>9}  {r.status}")
        return out

    def record(self) -> dict:
        def estimate(e) -> dict | None:
            if e is None or e.spread is None:
                return None
            return {"half_spread_bps": e.half_spread_bps,
                    "error_bps": e.half_spread_error_bps, "t": e.t_statistic,
                    "usable_bars": e.usable_bars,
                    "autocorrelation": e.autocorrelation}

        return {"provider": self.provider, "passed": self.passed,
                "verdict": self.verdict,
                "autocorrelation_note": _autocorrelation_note(self.rungs),
                "rungs": [{"symbol": r.rung.symbol, "band": [r.rung.low_bps,
                                                             r.rung.high_bps],
                           "region": r.rung.region, "status": r.status,
                           "detail": r.detail,
                           "estimate": estimate(r.estimate),
                           "clean_estimate": estimate(r.clean),
                           "lower_bps": r.lower_bps, "upper_bps": r.upper_bps,
                           "bars_set_aside": (None if r.quality is None
                                              else r.quality.counts()),
                           "clean_status": r.clean_status}
                          for r in self.rungs],
                "pairs": [{"symbols": list(p.symbols), "z": p.z,
                           "ratio": p.ratio, "consistent": p.consistent,
                           "estimates": [estimate(e) for e in p.estimates],
                           "detail": p.detail} for p in self.pairs]}


def _autocorrelation_note(rungs) -> str:
    """What the per-bar autocorrelation on the ladder says, US rungs first.

    The one number from the real book that no simulated contamination
    reproduces is a per-bar autocorrelation of -0.40. The ladder runs the
    same estimator on SPY and AAPL, whose bars are not in doubt, so their
    rows decide where it comes from: large there means the path
    manufactures it and it has nothing to do with European listings; near
    zero there and large on the European rungs means it is a property of
    those lines, and the next question is thin trading on the specific
    listing rather than the fund. Neither branch changes the ladder's
    verdict; this is a measurement printed beside it.

    "Large" is judged against the sampling error of the number, three
    standard errors at ``1/sqrt(n)``, not against a flat 0.10: the fixture's
    AAPL read -0.104 off 505 bars, which is two sigma of nothing, and a flat
    threshold called the path guilty on it.

    >>> from portfolio.core.spread import SpreadEstimate, WindowSweep
    >>> def rung(symbol, region, rho, bars=999):
    ...     e = SpreadEstimate(0.0004, 1.6e-7, 0.0001, bars + 1, bars,
    ...                        square_standard_error=1e-7, autocorrelation=rho)
    ...     return RungResult(Rung(symbol, "", 0.1, 2.0, region), "consistent",
    ...                       "", sweep=WindowSweep((), (), e, bars + 1, "all"))
    >>> note = _autocorrelation_note((rung("SPY", "US", -0.31),
    ...                               rung("SU.PA", "EU", -0.28)))
    >>> "SPY -0.310" in note and "the path manufactures it" in note
    True

    The real ladder run: SPY +0.101 and a European line -0.459. Nothing
    manufactures both signs, and the first version of this said it did:

    >>> note = _autocorrelation_note((rung("SPY", "US", 0.101, bars=8000),
    ...                               rung("IPRE.DE", "EU", -0.459)))
    >>> "manufactures" in note, "not one mechanism" in note.replace("\\n", " ")
    (False, True)
    >>> note = _autocorrelation_note((rung("SPY", "US", -0.004),
    ...                               rung("SU.PA", "EU", -0.28)))
    >>> "a property of those lines" in note and "SU.PA" in note.split("beyond")[1]
    True

    Two sigma off 500 bars is not a finding about the path:

    >>> note = _autocorrelation_note((rung("AAPL", "US", -0.104, bars=504),
    ...                               rung("SU.PA", "EU", -0.02, bars=504)))
    >>> "Near zero on every rung" in note
    True
    """
    def each(items) -> str:
        return ", ".join(f"{r.rung.symbol} {r.autocorrelation:+.3f}"
                         for r in items)

    def large(r) -> bool:
        return abs(r.autocorrelation) > rho_threshold(r.estimate.usable_bars)

    measured = [r for r in rungs if r.autocorrelation is not None]
    if not measured:
        return ""
    us = [r for r in measured if r.rung.region == "US"]
    eu = [r for r in measured if r.rung.region != "US"]
    head = "Per-bar autocorrelation: "
    parts = []
    if us:
        parts.append(f"{each(us)} on the US rungs")
    if eu:
        parts.append(f"{each(eu)} on the European ones")
    head += "; ".join(parts) + "."
    large_us = [r for r in us if large(r)]
    large_eu = [r for r in eu if large(r)]

    def limits(items) -> str:
        return ", ".join(f"{rho_threshold(r.estimate.usable_bars):.2f} for "
                         f"{r.rung.symbol}" for r in items)

    def extreme(items):
        return max(items, key=lambda r: abs(r.autocorrelation))

    if large_us and large_eu:
        # Both sides beyond sampling error. One mechanism in the path would
        # leave the same sign at comparable size on every rung; the first
        # version of this branch did not look, and called SPY at +0.10 and
        # a European line at -0.46 "manufactured". Opposite signs, or a
        # European value several times the US one, is not one mechanism.
        top_us, top_eu = extreme(large_us), extreme(large_eu)
        same_sign = (top_us.autocorrelation > 0) == (top_eu.autocorrelation > 0)
        comparable = abs(top_eu.autocorrelation) <= 2.0 * abs(top_us.autocorrelation)
        if same_sign and comparable:
            tail = (f"Beyond three standard errors of zero on both "
                    f"({limits(large_us + large_eu)}), the same sign and of "
                    f"comparable size, so one mechanism in the path "
                    f"manufactures it and it says nothing about European "
                    f"listings.")
        else:
            why = ("opposite signs" if not same_sign
                   else f"{top_eu.rung.symbol} is "
                        f"{abs(top_eu.autocorrelation) / abs(top_us.autocorrelation):.0f}x "
                        f"the size of {top_us.rung.symbol}")
            tail = (f"Beyond three standard errors of zero on both sides, but "
                    f"{why}: not one mechanism. The path contributes at most "
                    f"what the US rungs show ({each(large_us)}); what "
                    f"{', '.join(r.rung.symbol for r in large_eu)} shows "
                    f"beyond that is a property of those lines, and the next "
                    f"question is thin trading on the specific listing, not "
                    f"the fund.")
    elif large_us:
        tail = (f"Beyond three standard errors of zero ({limits(large_us)}) "
                f"on the US rungs only; the European rungs are within "
                f"sampling error. The number is on the lines whose data is "
                f"not in doubt and absent from the European ones, the reverse "
                f"of the book, and nothing here explains the book's -0.40.")
    elif us and large_eu:
        tail = (f"Within sampling error of zero where the data is not in "
                f"doubt and beyond three standard errors ({limits(large_eu)}) "
                f"on the European rungs, so it is a property of those lines "
                f"rather than of the path. The next question is thin trading "
                f"on the specific listing, not the fund.")
    elif us:
        tail = ("Near zero on every rung, within three standard errors of "
                "zero at each sample; the book's -0.40 is not reproduced on "
                "any of these lines.")
    else:
        tail = ("No US rung was available to say whether the path "
                "manufactures it.")
    return _wrap(head + " " + tail)


def _wrap(text: str, width: int = 74) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


def _ladder_verdict(rungs: "tuple[RungResult, ...]",
                    pairs: "tuple[PairResult, ...]") -> tuple[str, bool]:
    """Which end of the path the fault is at, from where the ladder broke.

    Written as a decision table rather than a sentence per case, so that
    every combination the control can produce has a line here and none is
    answered by a guess.
    """
    us = [r for r in rungs if r.rung.region == "US"]
    eu = [r for r in rungs if r.rung.region != "US"]
    wide_us = [r for r in us if r.status == "too wide"]
    wide_eu = [r for r in eu if r.status == "too wide"]
    narrow = [r for r in rungs if r.status == "too narrow"]
    missing = [r for r in rungs
               if r.status in ("no estimate", "unavailable", "unchecked")]
    fine = [r for r in rungs if r.status == "consistent"]
    bad_pairs = [p for p in pairs if p.consistent is False]

    def names(items) -> str:
        return ", ".join(r.rung.symbol for r in items)

    if wide_us:
        return (f"THE PIPELINE. {names(wide_us)} read wide on bars whose "
                f"open, high, low and close\nare not in doubt, so the path "
                f"from provider to estimate is producing spreads\nthe market "
                f"does not have, and nothing it says about a European line "
                f"can be\nread until this is found. The bars set aside per "
                f"rung above are the first\nplace to look.", False)
    if wide_eu and us and not [r for r in us if r.status != "consistent"]:
        return (f"THIS PROVIDER'S EUROPEAN BARS. The same path reads the US "
                f"rungs inside their\nbands and {names(wide_eu)} wide, so "
                f"whatever differs is in the European bars or\nin how the "
                f"model fits them, not in the path. The counts of bars set "
                f"aside above\nsay whether it is bars that repeat, and the "
                f"estimate with them set aside says\nwhether that is the "
                f"whole of it.", False)
    if wide_eu:
        return (f"{names(wide_eu)} read wide, and no US rung was available "
                f"to say whether the\npipeline reads a known-tight instrument "
                f"right. Inconclusive between the\npipeline and the "
                f"provider's European bars; run again with the US rungs.",
                False)
    if narrow:
        return (f"THE PIPELINE, UNDER-READING. {names(narrow)} came back "
                f"below their bands.\nA path that reads a wide spread narrow "
                f"is as wrong as one that reads a tight\none wide, and would "
                f"make the allocator keener to trade a thin line than\nit "
                f"should be.", False)
    if bad_pairs:
        listed = "; ".join(f"{p.symbols[0]}/{p.symbols[1]} {p.detail}"
                           for p in bad_pairs)
        return (f"THE DATA FOR ONE LISTING. Every rung is inside its band, "
                f"but the same fund\non two venues disagrees by more than "
                f"this control allows two venues: {listed}.\nWhich line is "
                f"wrong, and whether that factor is a venue difference for "
                f"this fund,\nis not measured here.", False)
    if missing and not fine:
        return (f"NOTHING TO SAY. No rung produced an estimate "
                f"({names(missing)}); the\nprovider is unreachable or the "
                f"symbols are not its. Nothing about the path\nhas been "
                f"checked.", False)
    if missing:
        return (f"CONSISTENT WHERE IT COULD BE CHECKED: {names(fine)} inside "
                f"their bands;\n{names(missing)} produced no estimate and "
                f"checked nothing. If the book's\nranking check still fails, "
                f"the fault is not one this ladder reproduces on\nthese "
                f"instruments: look at the book's own symbols and its "
                f"liquidity proxy.", False)
    return ("CONSISTENT. Every rung reads inside its band, the tight ones "
            "tight and the\nthin one wide, and the paired listings agree. "
            "The pipeline and this provider's\nbars pass on these "
            "instruments. If the book's ranking check still fails, the\n"
            "fault is not one this ladder reproduces: look at the book's own "
            "symbols and\nits liquidity proxy, both printed by `portfolio "
            "spreads`.", True)


# --------------------------------------------------------------------------
# The reference check: one symbol, three questions, the exact bars we read
# --------------------------------------------------------------------------
#
# The first real ladder run put SPY at about 19 bps against a true half-
# spread near one, and every line of eleven between 8 and 34 whatever its
# truth. Three questions separate the halves, in this order, and none of
# them is answered by changing the estimator:
#
#   T5  The authors' own package on the identical arrays. Same number: the
#       transcription is faithful and the fault is in the input or in the
#       method's fit to daily bars. A different number: the implementation
#       diverges on real data in a way three hundred synthetic panels could
#       not show.
#   T6  The adjusted series beside the unadjusted one. A factor constant
#       within a bar cancels in every log ratio the estimator takes, so the
#       two should agree except across ex-dates; a large difference is a
#       finding about the adjustment path.
#   T7  Blocks by date. SPY's history starts in 1993 and US markets
#       decimalised in 2001; if the early block reads far wider, part of a
#       long-window number is market history, and the drift test's silence
#       on it is measured rather than argued.
#
# And the four moment conditions on their own, because a floor common to
# every instrument is one additive term in s^2, and the four products say
# which prices carry it.


@dataclasses.dataclass(frozen=True)
class DateBlock:
    label: str
    first: object                        # date
    last: object
    estimate: object                     # SpreadEstimate
    components: object                   # SpreadComponents | None
    quality: object                      # BarQuality


@dataclasses.dataclass(frozen=True)
class ReferenceReport:
    symbol: str
    provider: str
    first: object
    last: object
    rows: int
    quality: object                      # BarQuality
    ours: object                         # SpreadEstimate on every bar
    components: object                   # SpreadComponents | None
    theirs_half_bps: float | None        # bidask.edge, signed root, half bps
    theirs_note: str
    adjusted: object = None              # SpreadEstimate on the adjusted series
    adjusted_note: str = ""
    blocks: tuple = ()                   # DateBlock, in date order
    sigma_day: float | None = None

    @property
    def ours_half_bps(self) -> float | None:
        e = self.ours
        if e.signed_square is None:
            return None
        return math.copysign(math.sqrt(abs(e.signed_square)),
                             e.signed_square) * 10_000.0 / 2.0

    def lines(self) -> list[str]:
        out = [f"Reference check: {self.symbol} on {self.provider}",
               "=" * 74, "",
               f"{self.rows} bars, {self.first} to {self.last}"]
        out.extend(f"  {line}" for line in self.quality.lines())
        out.append("")
        out.append("T5  the authors' package on the identical arrays")
        mine = self.ours_half_bps
        out.append(f"  ours    {'-' if mine is None else f'{mine:+8.2f}'} bps "
                   f"(signed root of s^2; {self.ours.describe()})")
        if self.theirs_half_bps is None:
            out.append(f"  bidask  not run: {self.theirs_note}")
        else:
            gap = (None if mine is None else self.theirs_half_bps - mine)
            out.append(f"  bidask  {self.theirs_half_bps:+8.2f} bps"
                       + ("" if gap is None else
                          f"   difference {gap:+.4f} bps -- "
                          + ("the two agree: the transcription is faithful "
                             "and the fault is in the input or in the "
                             "method's fit to daily bars"
                             if abs(gap) < 0.01 else
                             "THE TWO DISAGREE on real bars, which three "
                             "hundred synthetic panels did not show")))
        if self.components is not None:
            out.append("")
            out.append("  the four moment conditions, separately:")
            out.extend(f"  {line}" for line in self.components.lines())
        rho = self.ours.autocorrelation
        if rho is not None:
            out.append(f"  per-bar autocorrelation "
                       f"{_rho_text(rho, self.ours.usable_bars)}")
        out.append("")
        out.append("T6  adjusted against unadjusted prices")
        if self.adjusted is None:
            out.append(f"  not run: {self.adjusted_note}")
        else:
            adj = self.adjusted
            adj_bps = (None if adj.signed_square is None else
                       math.copysign(math.sqrt(abs(adj.signed_square)),
                                     adj.signed_square) * 10_000.0 / 2.0)
            out.append(f"  unadjusted {'-' if mine is None else f'{mine:+8.2f}'} bps"
                       f"   adjusted {'-' if adj_bps is None else f'{adj_bps:+8.2f}'} bps"
                       f"   ({adj.describe()})")
            out.append("  a factor constant within a bar cancels in every log "
                       "ratio; the two should differ only by what the steps at "
                       "each ex-date contribute")
        out.append("")
        out.append("T7  blocks by date")
        if not self.blocks:
            out.append("  no --split-at given; the window sweep's own blocks "
                       "are in the ladder output")
        for b in self.blocks:
            e = b.estimate
            out.append(f"  {b.label}: {b.first} to {b.last}, "
                       f"{b.quality.bars} bars")
            out.append(f"      {e.describe()}"
                       + (f"; per-bar autocorrelation "
                          f"{_rho_text(e.autocorrelation, e.usable_bars)}"
                          if e.autocorrelation is not None else ""))
            if b.components is not None:
                parts = b.components
                out.append(f"      moment conditions, signed roots: "
                           f"r1r2 {parts.half_bps(parts.open_previous_mid):+.1f}  "
                           f"r3r4 {parts.half_bps(parts.close_previous_mid):+.1f}  "
                           f"r1r5 {parts.half_bps(parts.open_previous_close):+.1f}  "
                           f"r5r4 {parts.half_bps(parts.close_open):+.1f}")
            out.append(f"      {b.quality.lines()[0]}")
        return out

    def record(self) -> dict:
        def estimate(e):
            if e is None or e.spread is None:
                return None
            return {"half_spread_bps": e.half_spread_bps,
                    "signed_square": e.signed_square,
                    "error_bps": e.half_spread_error_bps, "t": e.t_statistic,
                    "usable_bars": e.usable_bars,
                    "autocorrelation": e.autocorrelation}

        def components(p):
            return None if p is None else dataclasses.asdict(p)

        return {"symbol": self.symbol, "provider": self.provider,
                "bars": {"first": str(self.first), "last": str(self.last),
                         "rows": self.rows},
                "quality": self.quality.counts(),
                "ours": estimate(self.ours), "ours_half_bps": self.ours_half_bps,
                "components": components(self.components),
                "bidask_half_bps": self.theirs_half_bps,
                "bidask_note": self.theirs_note,
                "adjusted": estimate(self.adjusted),
                "adjusted_note": self.adjusted_note,
                "blocks": [{"label": b.label, "first": str(b.first),
                            "last": str(b.last), "estimate": estimate(b.estimate),
                            "components": components(b.components),
                            "quality": b.quality.counts()} for b in self.blocks]}


def reference_check(symbol: str, *, provider="yfinance",
                    data_root: "pathlib.Path | None" = None,
                    splits: tuple = ()) -> ReferenceReport:
    """T5, T6 and T7 on one symbol, on the exact bars the survey reads.

    `splits` are dates; the history is cut into blocks at each. Nothing here
    changes what the survey does: it is a diagnostic, run before any fix,
    because a correction tuned to one instrument without knowing the
    mechanism is how a wrong number acquires a plausible face.
    """
    from .core.spread import classify_bars, edge, edge_components
    from .data.cache import PriceCache

    market_provider = _provider_named(provider)
    root = pathlib.Path(data_root) if data_root else DataStore.open("user").root
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = PriceCache(cache_dir / f"bars-{market_provider.name}.sqlite")
    try:
        frame = _bars_for(symbol, market_provider, cache)
    finally:
        cache.close()

    o, h, l, c = (frame[k].to_numpy() for k in ("open", "high", "low", "close"))
    volume = frame["volume"].to_numpy() if "volume" in frame.columns else None
    quality = classify_bars(o, h, l, c, volume)
    ours = edge(o, h, l, c)
    parts = edge_components(o, h, l, c)

    theirs, note = None, ""
    try:
        import bidask
    except ImportError:
        note = "the bidask package is not installed (pip install bidask)"
    else:
        try:
            signed = float(bidask.edge(o, h, l, c, sign=True))
            theirs = math.copysign(abs(signed), signed) * 10_000.0 / 2.0
        except Exception as exc:                 # noqa: BLE001 - reported, not raised
            note = f"bidask.edge raised {type(exc).__name__}: {exc}"

    adjusted, adjusted_note = None, ""
    fetch = getattr(market_provider, "adjusted_bars", None)
    if fetch is None:
        adjusted_note = f"{market_provider.name} does not supply adjusted bars"
    else:
        try:
            adj = fetch(symbol, period="max")
            adjusted = edge(*(adj[k].to_numpy()
                              for k in ("open", "high", "low", "close")))
        except Exception as exc:                 # noqa: BLE001 - reported, not raised
            adjusted_note = f"{type(exc).__name__}: {exc}"

    blocks = []
    if splits:
        edges = [None] + sorted(pd.Timestamp(s) for s in splits) + [None]
        for start, stop in zip(edges, edges[1:]):
            mask = np.ones(len(frame), dtype=bool)
            if start is not None:
                mask &= frame.index >= start
            if stop is not None:
                mask &= frame.index < stop
            if mask.sum() < 3:
                continue
            piece = frame[mask]
            po, ph, pl, pc = (piece[k].to_numpy()
                              for k in ("open", "high", "low", "close"))
            pv = piece["volume"].to_numpy() if "volume" in piece.columns else None
            label = (f"{'start' if start is None else start.date()} to "
                     f"{'end' if stop is None else stop.date()}")
            blocks.append(DateBlock(label, piece.index[0].date(),
                                    piece.index[-1].date(), edge(po, ph, pl, pc),
                                    edge_components(po, ph, pl, pc),
                                    classify_bars(po, ph, pl, pc, pv)))

    return ReferenceReport(
        symbol=symbol, provider=market_provider.name,
        first=frame.index[0].date(), last=frame.index[-1].date(),
        rows=int(len(frame)), quality=quality, ours=ours, components=parts,
        theirs_half_bps=theirs, theirs_note=note, adjusted=adjusted,
        adjusted_note=adjusted_note, blocks=tuple(blocks),
        sigma_day=daily_volatility(c))


def run_ladder(*, provider="yfinance",
               data_root: "pathlib.Path | None" = None,
               rungs: tuple[Rung, ...] = LADDER,
               pairs: tuple[tuple[str, str], ...] = PAIRS) -> LadderReport:
    """Run the survey's whole path on the ladder and say where it broke.

    Bars go through the same cache file the survey uses, so a rung fetched
    once is free thereafter and so that the ladder reads exactly the rows
    the survey would.
    """
    from .core.spread import classify_bars, exclude_bars, sweep_windows
    from .data.cache import PriceCache

    market_provider = _provider_named(provider)
    root = pathlib.Path(data_root) if data_root else DataStore.open("user").root
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = PriceCache(cache_dir / f"bars-{market_provider.name}.sqlite")

    def measure(symbol: str):
        """(sweep, quality, clean estimate, daily volatility) or an error."""
        try:
            frame = _bars_for(symbol, market_provider, cache)
        except Exception as exc:                 # noqa: BLE001 - reported, not raised
            return f"{type(exc).__name__}: {exc}"
        o, h, l, c = (frame[k].to_numpy() for k in ("open", "high", "low",
                                                    "close"))
        volume = frame["volume"].to_numpy() if "volume" in frame.columns else None
        sweep = sweep_windows(o, h, l, c)
        bars = classify_bars(o, h, l, c, volume)
        if bars.excluded_count:
            clean = sweep_windows(*exclude_bars(o, h, l, c, bars.excluded)).chosen
        else:
            clean = sweep.chosen
        # Volatility on the window the sweep chose, for the reason given in
        # `survey_spreads`.
        return sweep, bars, clean, daily_volatility(c[-sweep.chosen_bars:])

    results = []
    for rung in rungs:
        got = measure(rung.symbol)
        if isinstance(got, str):
            results.append(RungResult(rung, "unavailable", got))
            continue
        sweep, bars, clean, sigma = got
        status, detail, lower, upper = rung_verdict(sweep.chosen, rung,
                                                    sigma_day=sigma)
        clean_status, clean_detail, _, _ = rung_verdict(clean, rung,
                                                        sigma_day=sigma)
        results.append(RungResult(rung, status, detail, sweep=sweep,
                                  quality=bars, clean=clean,
                                  clean_status=clean_status,
                                  clean_detail=clean_detail,
                                  lower_bps=lower, upper_bps=upper))

    paired = []
    for a, b in pairs:
        got_a, got_b = measure(a), measure(b)
        est_a = None if isinstance(got_a, str) else got_a[0].chosen
        est_b = None if isinstance(got_b, str) else got_b[0].chosen
        q_a = None if isinstance(got_a, str) else got_a[1]
        q_b = None if isinstance(got_b, str) else got_b[1]
        s_a = None if isinstance(got_a, str) else got_a[3]
        s_b = None if isinstance(got_b, str) else got_b[3]
        paired.append(_compare_pair((a, b), (est_a, est_b), (q_a, q_b),
                                    (got_a if isinstance(got_a, str) else "",
                                     got_b if isinstance(got_b, str) else ""),
                                    sigmas=(s_a, s_b)))

    cache.close()
    verdict, passed = _ladder_verdict(tuple(results), tuple(paired))
    return LadderReport(tuple(results), tuple(paired), verdict, passed,
                        market_provider.name)


def _interval_bps(estimate, significance: float = 2.0
                  ) -> tuple[float, float | None]:
    """(lower, upper) bound on the half-spread in bps, on s^2 then rooted."""
    s2 = estimate.signed_square
    se = estimate.square_standard_error or 0.0
    lower = math.sqrt(max(s2 - significance * se, 0.0)) * 10_000.0 / 2.0
    top = s2 + significance * se
    upper = math.sqrt(top) * 10_000.0 / 2.0 if top > 0 else None
    return lower, upper


def _compare_pair(symbols, estimates, qualities, errors,
                  sigmas=(None, None)) -> PairResult:
    """Two listings of one fund: how far apart, in sigma and in ratio.

    Two resolved lines are compared on ``s^2``, the disjoint-sample
    arithmetic the drift test uses, and called inconsistent only when both
    the distance in standard errors and the ratio exceed the thresholds.

    An unresolved line cannot be compared that way, and the first version
    of this did anyway, which made it a dead check for exactly the case it
    exists for: when the wider line is unresolved, ``|z| <= t < 2`` whatever
    the ratio, so two listings a factor of thirteen apart were reported as
    agreeing within error. The real book resolved nothing. So for an
    unresolved line the ceiling is what is judged, the way `rung_verdict`
    judges it: against the ceiling a clean sample of the same length and
    volatility would put on the OTHER line's spread. A ceiling more than
    `CEILING_SLACK` times that is a line whose error bar its sample cannot
    explain, and that line is the inconsistent one. An unresolved ceiling
    that sits BELOW the other line's lower bound is inconsistent the other
    way. Two unresolved lines with ceilings both inside what their samples
    allow are reported as not comparable, which is the truth.
    """
    from .core.spread import _drift_z
    a, b = estimates
    if a is None or b is None or a.spread is None or b.spread is None:
        why = "; ".join(
            f"{s}: {err or (e.refusal if e is not None else 'no bars')}"
            for s, e, err in zip(symbols, estimates, errors)
            if e is None or e.spread is None)
        return PairResult(symbols, estimates, qualities, detail=f"not "
                          f"comparable -- {why}", consistent=None)

    ratio = (b.half_spread_bps / a.half_spread_bps
             if a.half_spread_bps else float("inf"))
    wider = symbols[1] if ratio >= 1 else symbols[0]
    factor = max(ratio, 1.0 / ratio) if ratio > 0 else float("inf")
    resolved = (a.resolved(), b.resolved())

    if all(resolved):
        z = _drift_z(a, b)
        far = z is not None and abs(z) > PAIR_SIGMA and factor > PAIR_RATIO
        if far:
            detail = (f"{wider} is {factor:.1f}x wider at {abs(z):.1f} "
                      f"standard errors on s^2: over the factor of "
                      f"{PAIR_RATIO:g} this control allows two venues")
        elif z is not None and abs(z) > PAIR_SIGMA:
            detail = (f"{wider} is {factor:.2f}x wider at {abs(z):.1f} "
                      f"standard errors: a real difference between the two "
                      f"lines, under the factor of {PAIR_RATIO:g} this "
                      f"control allows two venues")
        else:
            detail = (f"{factor:.2f}x apart at "
                      f"{'?' if z is None else f'{abs(z):.1f}'} standard "
                      f"errors: the two venues agree within error")
        return PairResult(symbols, estimates, qualities, z=z, ratio=ratio,
                          detail=detail, consistent=not far)

    # At least one line unresolved: judge each unresolved ceiling against
    # what its sample allows at the other line's spread, and against the
    # other line's lower bound.
    notes, blamed, unchecked = [], [], []
    for i, (symbol, mine, other, sigma) in enumerate(
            zip(symbols, (a, b), (b, a), sigmas)):
        if resolved[i]:
            continue
        lower_other, _ = _interval_bps(other)
        _, upper_mine = _interval_bps(mine)
        if upper_mine is None:
            notes.append(f"{symbol}: the squared spread is negative beyond "
                         f"two standard errors")
            continue
        if sigma is None or mine.usable_bars <= 0:
            unchecked.append(symbol)
            notes.append(f"{symbol}: not resolved, at most {upper_mine:.1f} "
                         f"bps, and its ceiling could not be checked (no "
                         f"volatility)")
            continue
        reference = other.half_spread_bps if resolved[1 - i] else 0.0
        allowed = null_ceiling_bps(mine.usable_bars, sigma, reference)
        if allowed == allowed and upper_mine > CEILING_SLACK * allowed:
            blamed.append(symbol)
            notes.append(f"{symbol}: not resolved, and its ceiling of "
                         f"{upper_mine:.1f} bps is {upper_mine / allowed:.0f}x "
                         f"the {allowed:.1f} that {mine.usable_bars} bars at "
                         f"{sigma:.1%} a day put on a spread of "
                         f"{reference:.1f} bps -- an error bar its sample "
                         f"cannot explain")
        elif resolved[1 - i] and upper_mine < lower_other:
            blamed.append(symbol)
            notes.append(f"{symbol}: not resolved, and its ceiling of "
                         f"{upper_mine:.1f} bps sits below the other line's "
                         f"floor of {lower_other:.1f}")
        else:
            notes.append(f"{symbol}: not resolved, at most {upper_mine:.1f} "
                         f"bps, inside what {mine.usable_bars} bars at "
                         f"{sigma:.1%} a day allow ({allowed:.1f})")
    if blamed:
        detail = ("; ".join(notes) + f". {', '.join(blamed)} is the "
                  f"inconsistent line")
        return PairResult(symbols, estimates, qualities, ratio=ratio,
                          detail=detail, consistent=False)
    if unchecked or not any(resolved):
        return PairResult(symbols, estimates, qualities, ratio=ratio,
                          detail="not comparable -- " + "; ".join(notes),
                          consistent=None)
    return PairResult(symbols, estimates, qualities, ratio=ratio,
                      detail="; ".join(notes) + ": the unresolved line does "
                      "not contradict the resolved one", consistent=True)
