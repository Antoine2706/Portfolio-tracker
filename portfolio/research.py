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

__all__ = ["Book", "load_book", "run_equal_risk_contribution"]


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

    tradeable = frozenset(isin for isin in closes.columns
                          if instruments[isin].tradeable)
    frozen = {isin: held.get(isin, 0.0) for isin in closes.columns
              if isin not in tradeable}

    value = account_value
    if value is None:
        value = sum(float(p.market_value(snapshot.fx, dt.date.today()).amount)
                    for p in positions.values()
                    if p.is_open and p.is_derivable
                    and p.market_value(snapshot.fx, dt.date.today()) is not None)
    costs = CostModel(account_value=value or 15_000.0,
                      per_instrument=cost_table(instruments))

    notes = []
    for isin in sorted(frozen):
        inst = instruments[isin]
        notes.append(
            f"{inst.display_name} ({isin}) is frozen at {frozen[isin]:.1%}: held "
            f"at {inst.broker or 'a different broker'}, so its weight cannot be "
            f"traded against the rest of the book without a multi-day cash "
            f"transfer between institutions. It stays in the risk model.")
    unpriced = [i for i in instruments if i not in closes.columns]
    if unpriced:
        notes.append(f"not in the panel (too little price history): "
                     f"{', '.join(sorted(unpriced))}")
    return Book(panel=Panel(closes=closes), weights=held, costs=costs,
                tradeable=tradeable, frozen=frozen, instruments=instruments,
                notes=tuple(notes))


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
        from .agents.risk import risk_contribution_spread
        from .core.returns import simple_returns
        from .core.risk import covariance_matrix
        window = simple_returns(book.panel.closes).dropna().iloc[-lookback:]
        cov = covariance_matrix(window)
        among = [c for c in book.panel.closes.columns if c in book.tradeable]
        before = risk_contribution_spread(book.weights, cov, among=among)
        after = risk_contribution_spread(result.weights.iloc[-1].to_dict(),
                                         cov, among=among)
        notes.append(
            f"risk-share dispersion across the {len(among)} tradeable "
            f"holdings, as a fraction of their mean: {before:.2f} for the book "
            f"as it stands, {after:.2f} after the policy. Lower is more equal; "
            f"this is what the policy claims to do, measured separately from "
            f"whether doing it paid.")

    if refereed.adjustments:
        notes.append(f"{len(refereed.adjustments)} proposed trades were skipped "
                     f"as too small to cover their broker's fee; the first was "
                     f"{refereed.adjustments[0]}")
    if result.decisions:
        notes.append("policy said: " + result.decisions[-1].reason)
    return compare(result, benchmark, cost_model=book.costs, trials=trials,
                   trial_sharpe_sd=trial_sharpe_sd, overlap=rebalance_every,
                   constraint_notes=tuple(notes))
