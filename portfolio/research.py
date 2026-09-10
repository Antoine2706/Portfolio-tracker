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
           "PairResult", "LadderReport", "run_ladder", "rung_verdict"]


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


def _bars_for(symbol: str, provider, cache) -> "pd.DataFrame":
    """Unadjusted OHLCV for `symbol`, through the cache, whole history.

    "max", not the two years the price loader takes. A spread is a property
    of the instrument and its market makers rather than of the holding
    period, and the noise floor thins as the fourth root of the sample: on
    250 bars nothing under about 9 bps resolves, which is most of a European
    ETF book. Raises whatever the provider raises; the caller decides whether
    that is a refusal or a failure.
    """
    cached = cache.get_bars(symbol)
    if cached is not None and not cache.bars_predate_volume(symbol):
        frame = cached[0]
        if frame is not None and not frame.empty:
            return frame
    frame = provider.bars(symbol, period="max")
    cache.put_bars(symbol, frame)
    return frame


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
    or a median of zero -- which for a European ETF on a consolidated feed is
    common, and means the proxy is missing rather than that nothing trades.
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
    tested rather than assumed: if the two columns agree the bars set aside
    did not matter, and if they disagree the difference is attributable to a
    named count of named bars.

    No significance is attached to the difference between the two columns.
    They are nested samples -- the clean bars are a subset of all of them --
    so neither the sum of their variances nor either alone is the variance
    of the difference, and a z-score printed from either would be wrong in
    a known direction. Both estimates carry their own error bars; read the
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
               "twice: on every bar the provider sent, and again with the "
               "bars a feed",
               "manufactures set aside. Where the two disagree, the bars set "
               "aside are why.", ""]
        for isin, d in sorted(self.decisions.items(),
                              key=lambda kv: -kv[1].half_spread_bps):
            label = self.names.get(isin, "")
            symbol = self.symbols.get(isin, "")
            out.append(f"{isin}  {label}{'  (' + symbol + ')' if symbol else ''}")
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
                loud = " -- LARGE; the error bar above is optimistic" \
                    if abs(rho) > 0.10 else ""
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
                    loud = (" -- the error bar, not the spread, is what is "
                            "large; the per-bar series has\n           "
                            "variance the model does not produce, which is "
                            "what bars that are not trades do"
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
        out.append("the history, unconverted between currencies. For a "
                   "European ETF the volume")
        out.append("Yahoo reports is often another listing's, or none; it is "
                   "printed per instrument")
        out.append("above so that a failed ranking can be laid at the right "
                   "number.")
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
            instruments[isin] = {
                "name": self.names.get(isin, ""),
                "symbol": self.symbols.get(isin),
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
        fh.write(json.dumps(entry, default=_json_default) + "\n")
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _json_default(value):
    """numpy scalars and anything else json cannot serialise on its own."""
    import numpy as np
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and value != value:
        return None
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
        # What it can report is simulated and printed beside it.
        if (estimate.spread is not None and not estimate.resolved()
                and estimate.usable_bars > 0):
            sigma = daily_volatility(c)
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
                        null_ceiling=null_ceiling)


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
    """One instrument whose half-spread is known well enough to check against."""
    symbol: str
    label: str
    low_bps: float
    high_bps: float
    region: str                          # "US", "EU" or "thin"

    @property
    def band(self) -> str:
        return f"{self.low_bps:g} to {self.high_bps:g} bps"


LADDER: tuple[Rung, ...] = (
    Rung("SPY", "SPDR S&P 500, NYSE Arca", 0.1, 2.0, "US"),
    Rung("AAPL", "Apple, Nasdaq", 0.2, 3.0, "US"),
    Rung("SU.PA", "Schneider Electric, Euronext Paris", 0.3, 4.0, "EU"),
    Rung("MEUD.PA", "Amundi Stoxx Europe 600, Euronext Paris", 0.5, 8.0, "EU"),
    # The fund the book holds, on its Amsterdam line. The band is the user's
    # own figure for it, 15 to 30 bps, with room either side. The symbol is
    # the one the book records for the fund and has not been verified from
    # the machine this was written on; `--rung` overrides it.
    Rung("IPRP.AS", "iShares European Property Yield, Euronext Amsterdam",
         8.0, 40.0, "thin"),
)

# The same fund on two venues. The estimate is per line and the venue is a
# real reason for two lines to differ by tens of per cent -- different market
# makers, different tick, different session -- so agreement within error
# bars is not required. A factor of several is not a venue difference.
PAIRS: tuple[tuple[str, str], ...] = (("VVSM.DE", "SMH.L"),)

# Beyond this many standard errors apart, AND this ratio between them, two
# listings of the same fund are called inconsistent. Both, because a ratio
# of 1.3 at ten sigma is a venue difference measured precisely, and a ratio
# of 4 at one sigma is noise.
PAIR_SIGMA = 3.0
PAIR_RATIO = 3.0


def null_ceiling_bps(bars: int, sigma_day: float, spread_bps: float = 0.0, *,
                     runs: int = 3, seed: int = 0,
                     significance: float = 2.0) -> float:
    """The ceiling a sample of this length and volatility puts on a spread.

    Simulated rather than looked up: the noise floor was measured at 3.97
    bps for 500 bars at 1% daily volatility, and it scales with both, but a
    formula fitted to that table would be one more constant to keep right.
    Three runs of the same market the controls use, at `spread_bps` true
    half-spread, and the median of the ceilings they report.

    What it is for: an estimate that fails to resolve is a ceiling, and a
    ceiling can be wrong too. Off 5000 bars at 1.5% a day, a one-bp
    instrument reports a ceiling of a few bps. One that reports 42 has an
    error bar sixty times what its sample allows, which is not a wide
    spread but a per-bar series with variance the model does not produce
    -- the signature of bars that are not trades. This is the number that
    makes that visible.

    >>> round(null_ceiling_bps(500, 0.01), 0) in (5.0, 6.0)
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
# the simulation does not, which widens a real error bar by some tens of
# per cent; the failure this exists to catch was a factor of ten to sixty.
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

    >>> from portfolio.core.spread import SpreadEstimate
    >>> spy = Rung("SPY", "", 0.1, 2.0, "US")
    >>> tight = SpreadEstimate(0.00012, 1.4e-8, 0.00004, 5000, 4999,
    ...                        square_standard_error=1.0e-8)
    >>> rung_verdict(tight, spy)[0]
    'consistent'
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
    return ("consistent",
            f"not resolved; at most {upper:.1f} bps, inside the band "
            f"{rung.band}", lower, upper)


def daily_volatility(close: np.ndarray) -> float | None:
    """Close-to-close log volatility, robust to the odd impossible bar.

    Median absolute deviation scaled to a standard deviation, because a
    feed that produced one bar at a hundredth of the price would otherwise
    set the volatility for the whole series from that one bar, and the
    whole point of measuring it is to judge a feed that may do that.
    """
    c = np.asarray(close, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size < 20:
        return None
    returns = np.diff(np.log(c))
    mad = float(np.median(np.abs(returns - np.median(returns))))
    return 1.4826 * mad if mad > 0 else None


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

    def lines(self) -> list[str]:
        mark = {"consistent": "ok  ", "too wide": "WIDE", "too narrow": "NARR",
                "no estimate": "----", "unavailable": "----"}[self.status]
        out = [f"[{mark}] {self.rung.symbol:<9} {self.rung.label}",
               f"       expected {self.rung.band}: {self.detail}"]
        if self.sweep is not None and len(self.sweep.rungs) > 1:
            out.extend(f"       {line}" for line in self.sweep.lines())
        if self.quality is not None:
            out.extend(f"       {line}" for line in self.quality.lines())
            if self.quality.excluded_count and self.clean is not None:
                out.append(f"       with those set aside: {self.clean_detail} "
                           f"-> {self.clean_status}")
        return out


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
               f"Provider: {self.provider}. Every rung runs the whole path "
               f"the survey runs --",
               "bars through the cache, the window sweep, the bars set aside, "
               "the tick -- on",
               "an instrument whose half-spread is known to within a factor "
               "of two. The",
               "band is printed beside each result, so a disagreement is with "
               "a number.", ""]
        for r in self.rungs:
            out.extend(r.lines())
            out.append("")
        for p in self.pairs:
            out.extend(p.lines())
            out.append("")
        out.append("-" * 74)
        out.extend(self.verdict.splitlines())
        return out

    def record(self) -> dict:
        return {"provider": self.provider, "passed": self.passed,
                "verdict": self.verdict,
                "rungs": [{"symbol": r.rung.symbol, "band": [r.rung.low_bps,
                                                             r.rung.high_bps],
                           "region": r.rung.region, "status": r.status,
                           "detail": r.detail,
                           "lower_bps": r.lower_bps, "upper_bps": r.upper_bps,
                           "bars_set_aside": (None if r.quality is None
                                              else r.quality.counts()),
                           "clean_status": r.clean_status}
                          for r in self.rungs],
                "pairs": [{"symbols": list(p.symbols), "z": p.z,
                           "ratio": p.ratio, "consistent": p.consistent,
                           "detail": p.detail} for p in self.pairs]}


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
    missing = [r for r in rungs if r.status in ("no estimate", "unavailable")]
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
        return (f"THIS PROVIDER'S EUROPEAN BARS. The US rungs read inside "
                f"their bands and\n{names(wide_eu)} read wide, on the same "
                f"path. The estimator is not at fault; the\nbars it was "
                f"handed for the European lines are. The counts of bars set "
                f"aside\nabove say which kind, and the estimate with them set "
                f"aside says whether\nthat kind is the whole of it.", False)
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
                f"but the same fund\non two venues disagrees beyond what a "
                f"venue difference explains: {listed}.\nOne of those two "
                f"lines is not the fund's trading.", False)
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
        return sweep, bars, clean, daily_volatility(c)

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
        paired.append(_compare_pair((a, b), (est_a, est_b), (q_a, q_b),
                                    (got_a if isinstance(got_a, str) else "",
                                     got_b if isinstance(got_b, str) else "")))

    cache.close()
    verdict, passed = _ladder_verdict(tuple(results), tuple(paired))
    return LadderReport(tuple(results), tuple(paired), verdict, passed,
                        market_provider.name)


def _compare_pair(symbols, estimates, qualities, errors) -> PairResult:
    """Two listings of one fund: how far apart, in sigma and in ratio."""
    from .core.spread import _drift_z
    a, b = estimates
    if a is None or b is None or a.spread is None or b.spread is None:
        why = "; ".join(
            f"{s}: {err or (e.refusal if e is not None else 'no bars')}"
            for s, e, err in zip(symbols, estimates, errors)
            if e is None or e.spread is None)
        return PairResult(symbols, estimates, qualities, detail=f"not "
                          f"comparable -- {why}", consistent=None)
    z = _drift_z(a, b)
    ratio = (b.half_spread_bps / a.half_spread_bps
             if a.half_spread_bps else float("inf"))
    wider = symbols[1] if ratio >= 1 else symbols[0]
    factor = max(ratio, 1.0 / ratio) if ratio > 0 else float("inf")
    far = z is not None and abs(z) > PAIR_SIGMA and factor > PAIR_RATIO
    if far:
        detail = (f"{wider} is {factor:.1f}x wider at {abs(z):.1f} standard "
                  f"errors on s^2: over the factor of {PAIR_RATIO:g} this "
                  f"control allows two venues")
    elif z is not None and abs(z) > PAIR_SIGMA:
        detail = (f"{wider} is {factor:.2f}x wider at {abs(z):.1f} standard "
                  f"errors: a real difference between the two lines, under "
                  f"the factor of {PAIR_RATIO:g} this control allows two "
                  f"venues")
    else:
        detail = (f"{factor:.2f}x apart at "
                  f"{'?' if z is None else f'{abs(z):.1f}'} standard errors: "
                  f"the two venues agree within error")
    return PairResult(symbols, estimates, qualities, z=z, ratio=ratio,
                      detail=detail, consistent=not far)
