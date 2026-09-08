"""Walk-forward evaluation: decide on a close, trade on the next bar, pay for it.

The harness is the product. A strategy is a hypothesis; this is the instrument
that measures it, and an instrument that has not been calibrated produces
confident numbers whether or not it is measuring anything. So the design here
is driven by the ways a backtest lies, not by the ways a strategy works.

How the future is kept out
--------------------------
Not by a check. Checks get forgotten, and a forgotten check leaves a leak that
looks exactly like a discovery. Instead the policy is handed a `MarketView`
built by *slicing* the price panel at the decision date, so the rest of the
panel is not in the object. To see tomorrow, a policy would have to reach
outside its only argument -- which is still possible for a closure that
captured the full frame, and is exactly what `leakage.py` perturbs the future
to detect.

The execution model
-------------------
A decision is taken on the close of day t, using data up to and including that
close, and executed on day t+1. Deciding and executing on the same close is
the most common look-ahead in amateur backtests: it assumes you can trade at a
price you only knew once trading had finished.

With `Execution.NEXT_OPEN`, day t+1 is split in two:

    overnight   open(t+1) / close(t) - 1     earned on the OLD weights
    intraday    close(t+1) / open(t+1) - 1   earned on the NEW weights

and the trade happens at the open, against weights that have already drifted
through the overnight move. That drift is why turnover is measured against the
drifted weights and not against the previous target: a portfolio that never
trades must show zero turnover, and undoing drift is precisely what a
rebalance is.

With `Execution.NEXT_CLOSE`, used when the panel has no open prices, the whole
of day t+1 accrues to the old weights and the trade happens at that day's
close. It is strictly more conservative -- a full extra day of delay -- which
is the right direction to be wrong in. The harness refuses to invent opens.

What is deliberately not here
-----------------------------
No strategy. The controls in `controls.py` are the only policies this module
is ever run against until the harness has been shown to discriminate.
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np
import pandas as pd

from ..agents.base import MarketView, Policy, Proposal
from ..core.returns import TRADING_DAYS_PER_YEAR
from .metrics import TrackRecord, track_record

__all__ = ["Execution", "Panel", "Decision", "BacktestResult", "walk_forward",
           "buy_and_hold"]


class Execution(enum.Enum):
    """When an order decided on a close actually trades."""
    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"


@dataclasses.dataclass(frozen=True)
class Panel:
    """Aligned price history for the instruments under evaluation.

    `closes` are adjusted closes: rows are dates, columns are ISINs. A NaN
    means the instrument was not trading on that date -- not listed yet, or
    delisted -- and the harness treats it that way rather than dropping the
    column, which is what makes the evaluation survivorship-honest.

    `opens` is optional and must match `closes` exactly in shape and labels
    when present. Without it, only `Execution.NEXT_CLOSE` is available.
    """
    closes: pd.DataFrame
    opens: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.closes.index, pd.DatetimeIndex):
            raise ValueError("panel closes must be indexed by date")
        if not self.closes.index.is_monotonic_increasing:
            raise ValueError(
                "panel dates are not sorted; every index calculation in this "
                "module assumes they are, and an out-of-order panel would "
                "produce a plausible backtest of a shuffled history")
        if self.closes.index.has_duplicates:
            raise ValueError("panel has duplicate dates")
        if self.opens is not None:
            if list(self.opens.columns) != list(self.closes.columns):
                raise ValueError("opens and closes must have the same columns")
            if not self.opens.index.equals(self.closes.index):
                raise ValueError("opens and closes must have the same dates")

    def __len__(self) -> int:
        return int(len(self.closes))


@dataclasses.dataclass(frozen=True)
class Decision:
    """One rebalance, with everything needed to audit it afterwards."""
    decided_on: pd.Timestamp
    executed_on: pd.Timestamp
    view_ends: pd.Timestamp              # last date the policy could see
    weights_before: dict[str, float]     # drifted, immediately before trading
    weights_after: dict[str, float]
    turnover: float                      # one-way
    cost: float                          # as a fraction of portfolio value
    reason: str
    confidence: float


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    """The out-of-sample record, gross and net, with the decisions behind it."""
    returns: pd.Series                   # net of costs
    gross_returns: pd.Series
    weights: pd.DataFrame                # held at each day's close
    turnover: pd.Series                  # non-zero only on execution days
    costs: pd.Series
    decisions: tuple[Decision, ...]
    warnings: tuple[str, ...]
    policy: str

    @property
    def total_cost(self) -> float:
        """Total cost paid, as a fraction of capital."""
        return float(self.costs.sum())

    @property
    def mean_turnover(self) -> float:
        """Mean one-way turnover per rebalance, not per day."""
        executed = self.turnover[self.turnover.index.isin(
            [d.executed_on for d in self.decisions])]
        return float(executed.mean()) if len(executed) else 0.0

    def track(self, *, trials: int = 1, trial_sharpe_sd: float = 0.0,
              overlap: int = 1,
              periods_per_year: int = TRADING_DAYS_PER_YEAR) -> TrackRecord:
        """Summarise the net record with its uncertainty attached."""
        rebalances_per_year = (
            len(self.decisions) / (len(self.returns) / periods_per_year)
            if len(self.returns) else 0.0)
        return track_record(
            self.returns, trials=trials, trial_sharpe_sd=trial_sharpe_sd,
            periods_per_year=periods_per_year, overlap=overlap,
            costs=self.total_cost, turnover=self.mean_turnover,
            rebalances_per_year=rebalances_per_year)


# --------------------------------------------------------------------------
# Weight arithmetic
# --------------------------------------------------------------------------


def _drift(weights: dict[str, float], returns: pd.Series
           ) -> tuple[dict[str, float], float, list[str]]:
    """Carry weights forward through one period's returns.

        g      = sum_i w_i r_i                     the portfolio's return
        w_i'   = w_i (1 + r_i) / (1 + g)           its new weights

    Cash is the unallocated remainder and earns nothing, which is why the
    denominator is 1 + g rather than a renormalisation over the risky assets
    alone: scaling the risky weights back to sum to 1 would silently spend the
    cash.

    A held instrument with no price today is carried at a zero return and
    named in the third return value. That is the honest treatment of a
    trading halt or a delisting mid-period: the position has not vanished, and
    valuing it at zero would fabricate a total loss.
    """
    if not weights:
        return {}, 0.0, []
    missing: list[str] = []
    growth: dict[str, float] = {}
    g = 0.0
    for isin, w in weights.items():
        r = returns.get(isin, np.nan)
        if r is None or pd.isna(r):
            missing.append(isin)
            r = 0.0
        g += w * float(r)
        growth[isin] = w * (1.0 + float(r))
    scale = 1.0 + g
    if abs(scale) < 1e-12:
        # The portfolio has been wiped out. Returning the pre-drift weights
        # avoids dividing by zero; there is nothing left to allocate anyway.
        return dict(weights), float(g), missing
    return {k: v / scale for k, v in growth.items()}, float(g), missing


def _turnover(target: dict[str, float], drifted: dict[str, float]) -> float:
    """One-way turnover: half the sum of absolute weight changes, cash included.

    Cash is counted because moving out of the market is a trade with a cost.
    Leaving it out would make a volatility-targeting policy that de-risks into
    cash look free.
    """
    keys = set(target) | set(drifted)
    gross = sum(abs(target.get(k, 0.0) - drifted.get(k, 0.0)) for k in keys)
    cash = abs((1.0 - sum(target.values())) - (1.0 - sum(drifted.values())))
    return float(0.5 * (gross + cash))


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


def walk_forward(panel: Panel, policy: Policy, *,
                 warmup: int,
                 rebalance_every: int = 21,
                 cost_model=None,
                 execution: Execution = Execution.NEXT_CLOSE,
                 purge: int = 0,
                 min_observations: int = 60) -> BacktestResult:
    """Roll a policy through the panel, one decision at a time.

    `warmup` is the number of rows reserved before the first decision, so that
    the policy's estimator has a full window. The first decision is taken on
    the close of row `warmup - 1` and executed on row `warmup`; nothing before
    that is part of the track record, because a policy cannot be judged on
    days it was not allowed to trade.

    `purge` shortens each view by that many rows, so the policy cannot see the
    `purge` bars immediately before its decision. In this setting it is
    normally unnecessary and defaults to 0: the estimator already reads only
    returns strictly before the decision, and the record is built only from
    returns strictly after execution, so there is no overlapping label to leak
    the way there is in the classification setting the technique comes from.
    It exists because that argument stops holding the moment a policy is added
    whose signal spans several bars, and discovering that later is worse than
    carrying an unused parameter now.

    `cost_model` is anything with `.cost(turnover, weights_before,
    weights_after) -> float` returning a fraction of portfolio value. Passing
    None means a cost-free run, which is only ever appropriate for calibrating
    the harness itself against a known effect size -- never for judging a
    strategy.
    """
    closes = panel.closes
    idx = closes.index
    n = len(idx)
    if warmup < 2:
        raise ValueError(f"warmup must be at least 2 rows, got {warmup}")
    if warmup >= n:
        raise ValueError(
            f"warmup of {warmup} rows leaves nothing to evaluate in a panel of "
            f"{n}; the track record would be empty rather than short")
    if rebalance_every < 1:
        raise ValueError(f"rebalance_every must be at least 1, got {rebalance_every}")
    if purge < 0:
        raise ValueError(f"purge cannot be negative, got {purge}")
    if execution is Execution.NEXT_OPEN and panel.opens is None:
        raise ValueError(
            "next-open execution needs open prices, and this panel has only "
            "closes. Supply opens, or use Execution.NEXT_CLOSE, which delays "
            "execution by a further day rather than inventing a price.")

    weights: dict[str, float] = {}
    pending: Proposal | None = None
    pending_from: pd.Timestamp | None = None
    pending_view_end: pd.Timestamp | None = None

    dates: list[pd.Timestamp] = []
    gross: list[float] = []
    net: list[float] = []
    turns: list[float] = []
    costs: list[float] = []
    held_rows: list[dict[str, float]] = []
    decisions: list[Decision] = []
    warnings: list[str] = []

    # The first decision is taken at the close of `warmup - 1`, so the first
    # day that can earn a return under a policy's weights is `warmup`.
    first_decision = warmup - 1
    decision_days = set(range(first_decision, n - 1, rebalance_every))

    def decide(i: int) -> None:
        nonlocal pending, pending_from, pending_view_end
        end = i - purge
        if end < 1:
            return
        view = MarketView(as_of=idx[end], _closes=closes.iloc[:end + 1],
                          held=dict(weights), min_observations=min_observations)
        proposal = policy.observe(view)
        if not isinstance(proposal, Proposal):
            raise TypeError(
                f"policy {getattr(policy, 'name', policy)!r} returned "
                f"{type(proposal).__name__}, not a Proposal")
        pending, pending_from, pending_view_end = proposal, idx[i], idx[end]

    decide(first_decision)

    for i in range(warmup, n):
        date = idx[i]
        prev_close = closes.iloc[i - 1]
        today_close = closes.iloc[i]
        cost = 0.0
        turn = 0.0

        if execution is Execution.NEXT_OPEN and pending is not None:
            today_open = panel.opens.iloc[i]
            r_overnight = today_open / prev_close - 1.0
            drifted, g1, missing = _drift(weights, r_overnight)
            target = dict(pending.weights)
            turn = _turnover(target, drifted)
            cost = cost_model.cost(turn, drifted, target) if cost_model else 0.0
            r_intraday = today_close / today_open - 1.0
            weights, g2, missing2 = _drift(target, r_intraday)
            gross_r = (1.0 + g1) * (1.0 + g2) - 1.0
            decisions.append(Decision(
                decided_on=pending_from, executed_on=date,
                view_ends=pending_view_end, weights_before=drifted,
                weights_after=target, turnover=turn, cost=cost,
                reason=pending.reason, confidence=pending.confidence))
            pending = None
            for isin in set(missing) | set(missing2):
                warnings.append(f"{isin} had no price on {date:%Y-%m-%d} "
                                f"while held; carried at a zero return")
        else:
            r_day = today_close / prev_close - 1.0
            weights, gross_r, missing = _drift(weights, r_day)
            for isin in missing:
                warnings.append(f"{isin} had no price on {date:%Y-%m-%d} "
                                f"while held; carried at a zero return")
            if pending is not None:
                # Next-close execution: the day has already accrued to the old
                # weights, and the trade happens at tonight's close.
                target = dict(pending.weights)
                turn = _turnover(target, weights)
                cost = cost_model.cost(turn, weights, target) if cost_model else 0.0
                decisions.append(Decision(
                    decided_on=pending_from, executed_on=date,
                    view_ends=pending_view_end, weights_before=dict(weights),
                    weights_after=target, turnover=turn, cost=cost,
                    reason=pending.reason, confidence=pending.confidence))
                weights = target
                pending = None

        dates.append(date)
        gross.append(float(gross_r))
        net.append(float(gross_r - cost))
        turns.append(float(turn))
        costs.append(float(cost))
        held_rows.append(dict(weights))

        if i in decision_days:
            decide(i)

    index = pd.DatetimeIndex(dates)
    frame = pd.DataFrame(held_rows, index=index).reindex(
        columns=[str(c) for c in closes.columns]).fillna(0.0)
    return BacktestResult(
        returns=pd.Series(net, index=index, name="net"),
        gross_returns=pd.Series(gross, index=index, name="gross"),
        weights=frame,
        turnover=pd.Series(turns, index=index, name="turnover"),
        costs=pd.Series(costs, index=index, name="cost"),
        decisions=tuple(decisions),
        warnings=tuple(dict.fromkeys(warnings)),
        policy=getattr(policy, "name", type(policy).__name__),
    )


def buy_and_hold(panel: Panel, weights: dict[str, float], *, warmup: int,
                 cost_model=None,
                 execution: Execution = Execution.NEXT_CLOSE) -> BacktestResult:
    """The benchmark that matters: the portfolio you already own, left alone.

    Beating an index you do not hold is irrelevant -- the real alternative to
    any policy is doing nothing, which means buying once and never trading
    again while the weights drift wherever the market takes them. That drift
    is the point of the comparison: a policy's turnover has to buy something
    better than free.
    """
    class _Once:
        name = "buy-and-hold"

        def __init__(self) -> None:
            self.done = False

        def observe(self, view: MarketView) -> Proposal:
            self.done = True
            return Proposal(dict(weights), 1.0,
                            "buy once and never trade again; the alternative "
                            "to every policy under test", self.name)

    # rebalance_every larger than the panel means the one initial decision is
    # the only one, which is what "never trade again" means mechanically.
    return walk_forward(panel, _Once(), warmup=warmup,
                        rebalance_every=len(panel) + 1, cost_model=cost_model,
                        execution=execution)
