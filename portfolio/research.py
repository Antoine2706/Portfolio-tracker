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
import pathlib

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
           "dominant_holding"]


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
