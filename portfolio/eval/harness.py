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
           "buy_and_hold", "Hold"]


class Execution(enum.Enum):
    """When an order decided on a close actually trades."""
    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"


# Below this a holding is not really held, and its missing price cannot move
# the portfolio's return.
WEIGHT_EPSILON = 1e-9


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
    # What the POLICY proposed, and what was actually EXECUTED. They differ
    # whenever a frozen holding is present: its weight is taken from the
    # drifted book at execution rather than from the proposal, and the drift
    # happens on the bar after the decision. Keeping them apart matters for
    # the leak detector, which must test the policy's output -- a pure
    # function of data up to the decision -- and not the executed target,
    # which legitimately depends on the execution bar.
    proposed: dict[str, float]
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
    # True where the day is a clean one-period observation: every holding that
    # mattered printed a price on this bar and the one before it. False days
    # are still real money and stay in the compounded return; they are kept
    # out of every variance, because a holding that could not be marked did
    # not return 0.00%, and the day after a gap carries two days of move.
    measured: pd.Series
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
    def estimable(self) -> pd.Series:
        """The subset of the record that may enter a variance."""
        return self.returns[self.measured.to_numpy(dtype=bool)]

    @property
    def excluded_days(self) -> int:
        return int((~self.measured.to_numpy(dtype=bool)).sum())

    @property
    def mean_turnover(self) -> float:
        """Mean one-way turnover per rebalance, not per day."""
        executed = self.turnover[self.turnover.index.isin(
            [d.executed_on for d in self.decisions])]
        return float(executed.mean()) if len(executed) else 0.0

    def track(self, *, trials: int = 1, trial_sharpe_sd: float = 0.0,
              overlap: int = 1,
              periods_per_year: int = TRADING_DAYS_PER_YEAR) -> TrackRecord:
        """Summarise the net record with its uncertainty attached.

        The estimator sees only the clean one-period observations; the total
        return and the drawdown see everything that happened. A day a holding
        could not be marked is real money and a fictional variance datum, and
        the two facts have to go to different places.
        """
        rebalances_per_year = (
            len(self.decisions) / (len(self.returns) / periods_per_year)
            if len(self.returns) else 0.0)
        return track_record(
            self.estimable, trials=trials, trial_sharpe_sd=trial_sharpe_sd,
            periods_per_year=periods_per_year, overlap=overlap,
            costs=self.total_cost, turnover=self.mean_turnover,
            rebalances_per_year=rebalances_per_year,
            realised_returns=self.returns)


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

    A zero is right for the *valuation* and wrong for the *estimator*: the
    position genuinely could not be marked today, but 0.00% is not an observed
    daily return. The caller reads the third value to keep such days out of
    every variance -- see the `measured` mask on `BacktestResult`.

    Only holdings with a non-trivial weight are reported. A zero-weight
    instrument's missing price cannot affect the portfolio's return, and
    flagging it would throw away days for no reason.
    """
    if not weights:
        return {}, 0.0, []
    missing: list[str] = []
    growth: dict[str, float] = {}
    g = 0.0
    for isin, w in weights.items():
        r = returns.get(isin, np.nan)
        if r is None or pd.isna(r):
            if abs(w) > WEIGHT_EPSILON:
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


def _apply_frozen(target: dict[str, float], drifted: dict[str, float],
                  frozen) -> dict[str, float]:
    """Frozen holdings keep the weight they drifted to; the rest share what is left.

    A holding that cannot be traded does not hold a target weight -- it holds
    whatever the market left it at. Applying a target to it would imply a
    trade, which is the thing that cannot happen, and the amount would be
    small enough to look like a rounding error while being a real order.

    So the frozen weights are taken from the drifted book and the tradeable
    block is rescaled into what remains. That is also what a real rebalance
    does: you rebalance the sleeve you can trade, among itself.
    """
    if not frozen:
        return target
    out = dict(target)
    reserved = 0.0
    for isin in frozen:
        out[isin] = float(drifted.get(isin, 0.0))
        reserved += out[isin]
    free = {k: v for k, v in out.items() if k not in frozen}
    total = sum(free.values())
    budget = max(0.0, 1.0 - reserved)
    if total > 0:
        for k in free:
            out[k] = free[k] * budget / total
    return out


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
                 min_observations: int = 60,
                 initial_weights: "dict[str, float] | None" = None,
                 frozen=frozenset()) -> BacktestResult:
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

    `initial_weights` is the book as it already stands at the start of the
    evaluation. Without it the portfolio begins in cash and the first decision
    has to buy everything, which charges the policy for building a position it
    was never asked to build and -- worse -- makes any holding the policy is
    forbidden to trade permanently unbuyable. Both the policy and its
    benchmark should be given the same starting weights, so that the
    comparison isolates what the policy does rather than how it got in.
    Establishing the position is not itself a decision and is not charged.

    `frozen` names holdings that must not be traded at all. Their weight is
    taken from the drifted book at execution rather than from the proposal,
    and the tradeable block is rescaled into what is left. Without this a
    frozen holding still gets traded: its weight drifts between the decision
    and the execution one bar later, so even proposing the weight it had at
    decision time implies an order by the time the order is placed.
    """
    closes = panel.closes
    idx = closes.index
    n = len(idx)

    # Two views of the same prices, and the distinction is the whole of the
    # missing-data policy.
    #
    # `carried` forward-fills, and is what a position is VALUED against. Using
    # the previous row instead means that when an instrument does not trade on
    # day t, day t+1's return is computed against a NaN and is also lost -- so
    # a single venue holiday destroys the two-day move rather than deferring
    # it. That is money vanishing from the track record, and it was doing so
    # silently.
    #
    # `observed` records where a price really printed, and is what decides
    # whether a day is ESTIMATED from. A return is admitted to a variance only
    # if both of its endpoint prices were observed on consecutive panel dates:
    # the gap day itself is not a daily return (the holding could not be
    # marked), and the day after it is not either (it carries two days of
    # move). Keeping the money and dropping the observation is the only
    # combination that is honest about both.
    carried = closes.ffill()
    observed = closes.notna()
    column_at = {str(c): k for k, c in enumerate(closes.columns)}
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

    weights: dict[str, float] = {str(k): float(v)
                                for k, v in (initial_weights or {}).items()}
    if weights:
        unknown = set(weights) - {str(c) for c in closes.columns}
        if unknown:
            raise ValueError(
                f"initial weights name instruments not in the panel: "
                f"{sorted(unknown)}")
        total = sum(weights.values())
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"initial weights sum to {total:.6g}; this portfolio is "
                f"long-only and unlevered")
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
    measured: list[bool] = []

    def is_clean(i: int, used) -> bool:
        """Did every instrument that mattered today print on both bars?"""
        return all(bool(observed.iat[i, column_at[k]])
                   and bool(observed.iat[i - 1, column_at[k]])
                   for k in used if k in column_at)

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
        prev_close = carried.iloc[i - 1]
        today_close = closes.iloc[i]
        cost = 0.0
        turn = 0.0
        used = {k for k, v in weights.items() if abs(v) > WEIGHT_EPSILON}

        if execution is Execution.NEXT_OPEN and pending is not None:
            today_open = panel.opens.iloc[i]
            r_overnight = today_open / prev_close - 1.0
            drifted, g1, missing = _drift(weights, r_overnight)
            target = _apply_frozen(dict(pending.weights), drifted, frozen)
            turn = _turnover(target, drifted)
            cost = cost_model.cost(turn, drifted, target) if cost_model else 0.0
            r_intraday = today_close / today_open - 1.0
            weights, g2, missing2 = _drift(target, r_intraday)
            gross_r = (1.0 + g1) * (1.0 + g2) - 1.0
            decisions.append(Decision(
                decided_on=pending_from, executed_on=date,
                view_ends=pending_view_end, weights_before=drifted,
                proposed=dict(pending.weights), weights_after=target,
                turnover=turn, cost=cost,
                reason=pending.reason, confidence=pending.confidence))
            pending = None
            used |= {k for k, v in target.items() if abs(v) > WEIGHT_EPSILON}
            for isin in set(missing) | set(missing2):
                warnings.append(f"{isin} had no price on {date:%Y-%m-%d} "
                                f"while held; valued at its last print and "
                                f"the day excluded from every variance")
        else:
            r_day = today_close / prev_close - 1.0
            weights, gross_r, missing = _drift(weights, r_day)
            for isin in missing:
                warnings.append(f"{isin} had no price on {date:%Y-%m-%d} "
                                f"while held; valued at its last print and "
                                f"the day excluded from every variance")
            if pending is not None:
                # Next-close execution: the day has already accrued to the old
                # weights, and the trade happens at tonight's close.
                target = _apply_frozen(dict(pending.weights), weights, frozen)
                turn = _turnover(target, weights)
                cost = cost_model.cost(turn, weights, target) if cost_model else 0.0
                decisions.append(Decision(
                    decided_on=pending_from, executed_on=date,
                    view_ends=pending_view_end, weights_before=dict(weights),
                    proposed=dict(pending.weights), weights_after=target,
                    turnover=turn, cost=cost,
                    reason=pending.reason, confidence=pending.confidence))
                weights = target
                pending = None

        dates.append(date)
        gross.append(float(gross_r))
        net.append(float(gross_r - cost))
        turns.append(float(turn))
        costs.append(float(cost))
        held_rows.append(dict(weights))
        measured.append(is_clean(i, used))

        if i in decision_days:
            decide(i)

    index = pd.DatetimeIndex(dates)
    frame = pd.DataFrame(held_rows, index=index).reindex(
        columns=[str(c) for c in closes.columns]).fillna(0.0)
    return BacktestResult(
        returns=pd.Series(net, index=index, name="net"),
        gross_returns=pd.Series(gross, index=index, name="gross"),
        weights=frame,
        measured=pd.Series(measured, index=index, name="measured"),
        turnover=pd.Series(turns, index=index, name="turnover"),
        costs=pd.Series(costs, index=index, name="cost"),
        decisions=tuple(decisions),
        warnings=tuple(dict.fromkeys(warnings)),
        policy=getattr(policy, "name", type(policy).__name__),
    )


class Hold:
    """Propose exactly what is already held. Turnover zero, cost zero."""
    name = "buy-and-hold"

    def observe(self, view: MarketView) -> Proposal:
        return Proposal(dict(view.held), 1.0,
                        "hold the book and let the weights drift; the "
                        "alternative to every policy under test", self.name)


def buy_and_hold(panel: Panel, weights: dict[str, float], *, warmup: int,
                 cost_model=None,
                 execution: Execution = Execution.NEXT_CLOSE) -> BacktestResult:
    """The benchmark that matters: the portfolio you already own, left alone.

    Beating an index you do not hold is irrelevant -- the real alternative to
    any policy is doing nothing, which means holding what you have while the
    weights drift wherever the market takes them. That drift is the point of
    the comparison: a policy's turnover has to buy something better than free.

    It starts from `weights` and never trades, so it pays nothing at all. The
    policy it is compared against starts from the same weights, which is what
    makes the difference between them attributable to the policy's decisions
    rather than to the cost of getting invested.

    Every holding is passed as frozen, which is precisely what "hold" means:
    each keeps whatever weight the market drifts it to. Without that, the one
    decision taken at the start would execute a bar later against weights that
    had already moved, and buy-and-hold would pay for a rebalance it never
    made -- small, plausible, and enough to bias the benchmark it is the whole
    point of this function to state honestly.
    """
    return walk_forward(panel, Hold(), warmup=warmup,
                        rebalance_every=len(panel) + 1, cost_model=cost_model,
                        execution=execution, initial_weights=weights,
                        frozen=frozenset(str(c) for c in panel.closes.columns))
