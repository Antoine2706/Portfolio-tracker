"""The constraints no policy may talk its way past.

A policy proposes; the referee decides what is actually executable. The
separation matters because the constraints are facts about the *account* --
which broker holds what, what a trade costs, how small is too small -- and a
policy that had to know all of them would be a policy that could get them
wrong. Every policy would then have to get them right independently, and one
of them eventually would not.

The constraint that motivated this file
---------------------------------------
The gold ETC is held at a different broker from the other six holdings.
Moving weight from gold into an ETF is not a rebalance: it is a sale at one
institution, a wait for settlement, a cash transfer between two banks, and a
purchase at the other -- roughly a week, out of position throughout, and two
sets of costs. A rebalancing policy assumes weight can move between holdings.
Across brokers it cannot.

So gold's weight is exogenous. It stays in the risk model, because it is
genuinely part of the portfolio and affects every covariance, correlation and
risk contribution; what it cannot do is be traded against the others. An
optimiser that is not told this will propose gold trades, and they will be
unexecutable.

This is enforced here rather than filtered afterwards. A filter applied to
the output is a filter someone forgets to apply to the next policy; a referee
that rejects the proposal makes the failure loud at the point it is made.

What it enforces
----------------
1.  No weight change in a non-tradeable holding. Rejected outright rather
    than clipped, because silently altering a proposal hides a policy that is
    solving the wrong problem.
2.  No trade below the minimum economic size *for that instrument's broker*.
    A flat fee makes a 200 EUR trade cost 1.2% at one broker and nothing at
    the other, so this is per broker and derived from the fee structure.
3.  Long-only and unlevered, which `Proposal` already checks at construction;
    re-checked here because the referee is the last thing between a proposal
    and an order.
"""

from __future__ import annotations

import dataclasses

from .base import Proposal

__all__ = ["Verdict", "Referee", "Refereed"]

# Below this the change is rounding rather than a decision: floating-point
# noise in an optimiser's output should not be reported as an attempt to
# trade a frozen holding.
WEIGHT_EPSILON = 1e-9


@dataclasses.dataclass(frozen=True)
class Verdict:
    """What the referee decided, and why."""
    allowed: bool
    weights: dict[str, float]
    rejections: tuple[str, ...] = ()
    adjustments: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.allowed

    def explain(self) -> str:
        if self.allowed and not self.adjustments:
            return "accepted as proposed"
        parts = list(self.adjustments) + list(self.rejections)
        return ("accepted, " if self.allowed else "REJECTED: ") + "; ".join(parts)


@dataclasses.dataclass(frozen=True)
class Referee:
    """Applies the account's hard constraints to a proposal.

    `held` is the weight vector actually in force, needed because a
    constraint is about the *change* a proposal implies, not its level: a
    proposal that leaves a frozen holding exactly where it is fine, and one
    that moves it by a basis point is not.

    >>> from .execution import CostModel, InstrumentCost
    >>> costs = CostModel(account_value=15_000.0, per_instrument={
    ...     "GOLD": InstrumentCost(broker="Keytrade", tob_rate=None),
    ...     "ETF1": InstrumentCost(broker="MeDirect"),
    ...     "ETF2": InstrumentCost(broker="MeDirect")})
    >>> ref = Referee(costs=costs, tradeable={"ETF1", "ETF2"})
    >>> held = {"GOLD": 0.2, "ETF1": 0.4, "ETF2": 0.4}

    Leaving the frozen holding alone is fine:

    >>> ok = ref.review(Proposal({"GOLD": 0.2, "ETF1": 0.5, "ETF2": 0.3},
    ...                          0.5, "shift between the two ETFs", "p"), held)
    >>> ok.allowed
    True

    Touching it is not, however small the move:

    >>> bad = ref.review(Proposal({"GOLD": 0.25, "ETF1": 0.45, "ETF2": 0.3},
    ...                           0.5, "trim gold", "p"), held)
    >>> bad.allowed
    False
    >>> print(bad.explain())
    REJECTED: GOLD is not tradeable (held at a different broker from the rest of the book, so its weight cannot be moved against them without a multi-day cash transfer), but the proposal moves it from 20.00% to 25.00%
    """
    costs: object
    tradeable: frozenset | set
    enforce_minimum_trade: bool = True

    def review(self, proposal: Proposal, held: dict[str, float]) -> Verdict:
        weights = dict(proposal.weights)
        rejections: list[str] = []
        adjustments: list[str] = []

        for isin in sorted(set(weights) | set(held)):
            before = float(held.get(isin, 0.0))
            after = float(weights.get(isin, 0.0))
            change = after - before
            if abs(change) <= WEIGHT_EPSILON:
                continue

            if isin not in self.tradeable:
                rejections.append(
                    f"{isin} is not tradeable (held at a different broker from "
                    f"the rest of the book, so its weight cannot be moved "
                    f"against them without a multi-day cash transfer), but the "
                    f"proposal moves it from {before:.2%} to {after:.2%}")
                continue

            if self.enforce_minimum_trade:
                traded = abs(change) * getattr(self.costs, "account_value", 0.0)
                floor = self.costs.minimum_trade_value(isin)
                if floor > 0 and 0 < traded < floor:
                    # Not a rejection: the sensible response to "this trade is
                    # too small to be worth its fee" is not to trade, which
                    # means holding the drifted weight rather than the target.
                    weights[isin] = before
                    adjustments.append(
                        f"{isin} left at {before:.2%}: the proposed "
                        f"{traded:.0f} EUR trade is below the {floor:.0f} EUR "
                        f"its broker's fee structure makes worthwhile")

        total = sum(weights.values())
        if total > 1.0 + 1e-9:
            rejections.append(
                f"weights sum to {total:.6g}; this portfolio is long-only and "
                f"unlevered")
        if any(v < 0 for v in weights.values()):
            rejections.append("negative weights proposed; this portfolio is long-only")

        return Verdict(allowed=not rejections, weights=weights,
                       rejections=tuple(rejections), adjustments=tuple(adjustments))


@dataclasses.dataclass
class Refereed:
    """A policy with the referee's constraints applied to everything it says.

    Wrapping rather than trusting. The policy is told which holdings are
    frozen and normally respects it; this is the independent check that it
    did, and it is what the harness actually runs.

    An adjustment -- a trade too small to be worth its fee -- is applied
    silently, because "do not make this trade" is a sensible answer to
    "this trade costs more than it can earn". A rejection is not: proposing a
    weight change in a holding that cannot be traded means the policy is
    solving a different problem from the one the account poses, and every
    number downstream would be measuring that other problem. So it raises.
    """
    policy: object
    referee: Referee
    name: str = ""
    rejected: int = 0

    def __post_init__(self) -> None:
        self.name = self.name or getattr(self.policy, "name", "policy")
        self.adjustments: list[str] = []

    def observe(self, view):
        proposal = self.policy.observe(view)
        verdict = self.referee.review(proposal, dict(view.held))
        if not verdict.allowed:
            self.rejected += 1
            raise ValueError(
                f"the referee rejected {self.name}'s proposal on "
                f"{view.as_of:%Y-%m-%d}: {verdict.explain()}")
        for note in verdict.adjustments:
            self.adjustments.append(f"{view.as_of:%Y-%m-%d}: {note}")
        if verdict.weights == proposal.weights:
            return proposal
        return Proposal(verdict.weights, proposal.confidence,
                        f"{proposal.reason} [{verdict.explain()}]", self.name)
