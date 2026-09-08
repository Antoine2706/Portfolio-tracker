"""Comparing a policy against doing nothing, and saying what its trading cost.

The benchmark is buy-and-hold of the portfolio already owned. Beating an
index nobody holds is irrelevant; the real alternative to any policy is
leaving the book alone and letting the weights drift wherever the market
takes them. That drift is the point of the comparison: a policy's turnover
has to buy something better than free.

Breakeven turnover
------------------
The headline this module exists for. Every result is reported gross and net,
and beside them the turnover at which the policy's gross edge is entirely
consumed by its own trading:

    edge        = annualised gross return, policy minus benchmark
    unit cost   = total cost paid / total one-way turnover
    breakeven   = edge / unit cost                       annual one-way turnover

Read against the turnover the policy actually runs at, that is the decision
criterion. A policy running at half its breakeven has room; one running at
twice it is paying for the privilege of being right. It also separates the
two ways a policy can fail -- no edge at all, or an edge too small to trade
-- which a net Sharpe ratio alone conflates.

The figure is only as good as the cost model, and the cost model's largest
component here is an estimate: bid-ask spreads are paid inside the execution
price and never appear on a contract note. So `Comparison.lines()` prints the
cost model's own assumptions underneath, every time.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..core.returns import TRADING_DAYS_PER_YEAR
from .harness import BacktestResult
from .metrics import (SUSPICIOUS_ACTIVE_SHARPE, SUSPICIOUS_ANNUAL_SHARPE,
                      TrackRecord, track_record)

__all__ = ["Breakeven", "Comparison", "compare"]

# Below this the two records are not distinguishable at conventional
# significance and must be reported as such rather than ranked.
DECISIVE_T = 2.0


def _annualised(returns, periods_per_year: int) -> float:
    """Geometric annual growth rate of a return series."""
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    total = float(np.prod(1.0 + arr))
    years = arr.size / periods_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


@dataclasses.dataclass(frozen=True)
class Breakeven:
    """The turnover at which a policy's gross edge pays for its own trading."""
    edge_gross: float                    # annualised, policy minus benchmark
    unit_cost: float                     # cost per unit of one-way turnover
    actual_turnover: float               # annual, one-way
    breakeven_turnover: float | None     # annual, one-way; None if no edge

    @property
    def headroom(self) -> float | None:
        """Breakeven turnover divided by actual. Above 1 the edge survives."""
        if self.breakeven_turnover is None or self.actual_turnover <= 0:
            return None
        return float(self.breakeven_turnover / self.actual_turnover)

    def sentence(self) -> str:
        if self.edge_gross <= 0:
            return (f"No gross edge to trade on: the policy returned "
                    f"{self.edge_gross:+.2%} a year less than doing nothing, "
                    f"before any costs. Turnover cannot rescue that.")
        if self.breakeven_turnover is None:
            return (f"Gross edge {self.edge_gross:+.2%} a year, but the policy "
                    f"never traded, so there is no cost to compare it against.")
        head = (f"Gross edge {self.edge_gross:+.2%} a year is fully consumed at "
                f"{self.breakeven_turnover:.0%} annual one-way turnover; the "
                f"policy runs at {self.actual_turnover:.0%}")
        room = self.headroom
        if room is None:
            return head + "."
        if room >= 1.0:
            return head + f", so it keeps about {1 - 1 / room:.0%} of the edge."
        return (head + f", so it spends about {1 / room:.1f} times the edge on "
                f"trading and the net result is negative by construction.")


@dataclasses.dataclass(frozen=True)
class Comparison:
    """A policy against buy-and-hold, gross and net, with its costs named."""
    policy_name: str
    net: TrackRecord
    gross: TrackRecord
    benchmark: TrackRecord
    breakeven: Breakeven
    total_cost: float
    mean_turnover: float
    rebalances: int
    warnings: tuple[str, ...]
    cost_assumptions: tuple[str, ...]
    constraint_notes: tuple[str, ...] = ()
    cost_provenance: tuple[str, ...] = ()
    # The policy's return minus the benchmark's, day by day. Its Sharpe is the
    # information ratio, and it is the quantity the whole comparison turns on.
    active: TrackRecord | None = None

    # -- the comparison ----------------------------------------------------

    def paired_lines(self) -> list[str]:
        """Why the two marginal Sharpe ratios are not the test.

        Both legs hold the same book on the same days, so the great majority
        of each one's estimation error is the same error, and it cancels in
        the difference. Comparing the marginal ratios against their own
        standard errors therefore asks a much harder question than the one
        being posed, and answers "cannot tell" long after the paired series
        could have answered it.
        """
        if self.active is None or self.active.sharpe is None:
            return ["No paired comparison: the two records do not share "
                    "enough dates to difference."]
        gap = ((self.net.sharpe - self.benchmark.sharpe)
               if (self.net.sharpe is not None
                   and self.benchmark.sharpe is not None) else None)
        out = [f"Active return, policy net minus buy-and-hold day by day: "
               f"information ratio {self.active.band()}, "
               f"t = {self.active.t_statistic:+.2f} over "
               f"{self.active.observations} observations."]
        if gap is not None:
            out.append(
                f"The gap between the two annualised Sharpe ratios is "
                f"{gap:+.2f}. Do not test that against either ratio's own "
                f"standard error: both legs hold the same book on the same "
                f"days, so most of that error is common to them and cancels. "
                f"The paired series above is the test with power.")
        t = self.active.t_statistic
        if t is None:
            return out
        if abs(t) < DECISIVE_T:
            out.append(
                f"|t| = {abs(t):.2f} is below {DECISIVE_T:.0f}, so on this "
                f"sample the policy and doing nothing are INDISTINGUISHABLE. "
                f"The ordering in the table above is not a ranking, and "
                f"reporting it as one would be reading noise.")
        else:
            better = "beats" if t > 0 else "loses to"
            out.append(
                f"|t| = {abs(t):.2f}: the policy {better} doing nothing by "
                f"more than this sample can attribute to chance.")
        return out

    def verdict(self) -> str:
        """The headline, with the benchmark in it.

        The level test on its own sent a reader hunting for a leak on a book
        whose benchmark scored higher still -- and a leak inside a rebalancing
        policy cannot lift a benchmark that never trades. What a shared high
        level does mean is that the cause is shared: the window, or the price
        data feeding both legs. Neither is evidence about the policy.
        """
        if self.net.sharpe is None or self.benchmark.sharpe is None:
            return self.net.verdict()
        mine, theirs = self.net.sharpe, self.benchmark.sharpe
        if (self.active is not None and self.active.sharpe is not None
                and self.active.sharpe > SUSPICIOUS_ACTIVE_SHARPE):
            return (f"Information ratio {self.active.sharpe:.2f} is above the "
                    f"{SUSPICIOUS_ACTIVE_SHARPE} threshold for an active "
                    f"return, which doing nothing cannot produce and a good "
                    f"window cannot explain. Find the leak before reporting "
                    f"this as a result.")
        if mine > SUSPICIOUS_ANNUAL_SHARPE and theirs > SUSPICIOUS_ANNUAL_SHARPE:
            return (f"Annualised Sharpe {self.net.band()} is above the "
                    f"{SUSPICIOUS_ANNUAL_SHARPE} threshold, and so is "
                    f"buy-and-hold at {theirs:.2f}. A leak inside a policy "
                    f"cannot lift a benchmark that never trades, so this level "
                    f"is a fact about the window or about the prices feeding "
                    f"both legs, not evidence about the policy. Read the "
                    f"active return instead.")
        if mine > SUSPICIOUS_ANNUAL_SHARPE:
            return (f"Annualised Sharpe {self.net.band()} is above the "
                    f"{SUSPICIOUS_ANNUAL_SHARPE} threshold while buy-and-hold "
                    f"is at {theirs:.2f}. A gap that size, on a benchmark that "
                    f"does no trading, is what a look-ahead leak looks like. "
                    f"Find it before reporting this as a result.")
        return self.net.verdict()

    def lines(self) -> list[str]:
        out = [f"{self.policy_name} against buy-and-hold", "=" * 64, ""]
        rows = [("", "policy net", "policy gross", "buy & hold"),
                ("annualised Sharpe", self.net.sharpe, self.gross.sharpe,
                 self.benchmark.sharpe),
                ("  1 standard error", self.net.annual_standard_error,
                 self.gross.annual_standard_error,
                 self.benchmark.annual_standard_error),
                ("annualised volatility", self.net.volatility,
                 self.gross.volatility, self.benchmark.volatility),
                ("total return", self.net.total_return, self.gross.total_return,
                 self.benchmark.total_return),
                ("maximum drawdown", self.net.max_drawdown,
                 self.gross.max_drawdown, self.benchmark.max_drawdown)]
        for label, *values in rows:
            if not label:
                out.append(f"{'':24}{values[0]:>14}{values[1]:>14}{values[2]:>14}")
                out.append("-" * 64)
                continue
            cells = "".join(
                f"{('n/a' if v is None else f'{v:.4f}'):>14}" for v in values)
            out.append(f"{label:24}{cells}")
        excluded = self.net.excluded_observations
        sample = (f"observations          {self.net.observations} measured "
                  f"({self.net.independent_observations} independent) over "
                  f"{self.net.years:.2f} years")
        if excluded:
            sample += (f"\n                      {excluded} further day"
                       f"{'' if excluded == 1 else 's'} are in the return but "
                       f"not in any variance: a holding did not print a price, "
                       f"so the day is real money and not a measurement")
        out += [
            "",
            *sample.split("\n"),
            f"rebalances            {self.rebalances}, mean one-way turnover "
            f"{self.mean_turnover:.2%}",
            f"cost paid             {self.total_cost:.2%} of capital over the "
            f"period, {self.net.cost_drag or 0:.2%} a year",
            "",
            "Breakeven",
            "-" * 64,
        ]
        out += ["  " + line for line in _wrap(self.breakeven.sentence())]
        out += ["", "Policy against benchmark", "-" * 64]
        for note in self.paired_lines():
            out += ["  " + line for line in _wrap(note)]
            out.append("")
        out += ["Verdict", "-" * 64]
        out += ["  " + line for line in _wrap(self.verdict())]
        if self.constraint_notes:
            out += ["", "Constraints", "-" * 64]
            for note in self.constraint_notes:
                wrapped = _wrap(note, 60)
                out.append("  - " + wrapped[0])
                out += ["    " + line for line in wrapped[1:]]
        out += ["", "How much of the cost figure rests on observed inputs",
                "-" * 64]
        for note in self.cost_provenance:
            out += ["  " + line for line in _wrap(note, 62)]
            out.append("")
        out.append("  Estimates, not observations:")
        for note in self.cost_assumptions:
            wrapped = _wrap(note, 58)
            out.append("    - " + wrapped[0])
            out += ["      " + line for line in wrapped[1:]]
        if self.warnings:
            out += ["", "Warnings", "-" * 64]
            for note in sorted(self.warnings)[:10]:
                out.append(f"  - {note}")
        return out


def _wrap(text: str, width: int = 62) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def compare(policy: BacktestResult, benchmark: BacktestResult, *,
            cost_model=None, trials: int = 1, trial_sharpe_sd: float = 0.0,
            overlap: int = 1, constraint_notes: "tuple[str, ...]" = (),
            weights: "dict[str, float] | None" = None,
            periods_per_year: int = TRADING_DAYS_PER_YEAR) -> Comparison:
    """Summarise a policy against doing nothing.

    `overlap` is passed to the track record so the independent-observation
    count reflects how much of the estimation window successive decisions
    share. Leaving it at 1 on a daily-rebalanced policy overstates the
    evidence, which is the direction that matters.

    `weights` are the book's actual weights, used only to weight the
    observed-against-assumed split of the cost inputs. Counting instruments
    treats a 2% holding and a 40% holding as the same evidence.
    """
    net = policy.track(trials=trials, trial_sharpe_sd=trial_sharpe_sd,
                       overlap=overlap, periods_per_year=periods_per_year)
    gross = track_record(policy.gross_returns[policy.measured.to_numpy(dtype=bool)],
                         trials=trials, trial_sharpe_sd=trial_sharpe_sd,
                         overlap=overlap, periods_per_year=periods_per_year,
                         realised_returns=policy.gross_returns)
    # The same overlap convention on every column, so the three standard
    # errors in the table are computed on the same basis and can be read side
    # by side. Buy-and-hold takes no decisions, so the adjustment is not
    # *about* it -- but a table whose columns use different conventions
    # invites exactly the comparison it cannot support.
    bench = benchmark.track(overlap=overlap, periods_per_year=periods_per_year)
    active = _active_record(policy, benchmark, overlap=overlap,
                            periods_per_year=periods_per_year)

    edge = (_annualised(policy.gross_returns, periods_per_year)
            - _annualised(benchmark.gross_returns, periods_per_year))
    turnover_total = float(policy.turnover.sum())
    years = len(policy.returns) / periods_per_year if len(policy.returns) else 0.0
    annual_turnover = turnover_total / years if years > 0 else 0.0
    unit_cost = (policy.total_cost / turnover_total) if turnover_total > 0 else 0.0
    breakeven = (edge / unit_cost) if (edge > 0 and unit_cost > 0) else None

    if cost_model is not None:
        assumptions = tuple(cost_model.assumptions())
        provenance = tuple(cost_model.provenance(weights).lines())
    else:
        assumptions = (
            "no cost model was supplied: this run is COST-FREE and cannot be "
            "used to judge a strategy, only to calibrate the harness",)
        provenance = ("None of it. There is no cost model in this run at all, "
                      "so every figure above is gross by construction.",)

    return Comparison(
        policy_name=policy.policy, net=net, gross=gross, benchmark=bench,
        breakeven=Breakeven(edge_gross=edge, unit_cost=unit_cost,
                            actual_turnover=annual_turnover,
                            breakeven_turnover=breakeven),
        total_cost=policy.total_cost, mean_turnover=policy.mean_turnover,
        rebalances=len(policy.decisions),
        warnings=policy.warnings, cost_assumptions=assumptions,
        constraint_notes=constraint_notes, cost_provenance=provenance,
        active=active)


def _active_record(policy: BacktestResult, benchmark: BacktestResult, *,
                   overlap: int, periods_per_year: int) -> "TrackRecord | None":
    """The paired difference series, which is the comparison with power.

    A policy and its buy-and-hold benchmark hold the same book on the same
    days, so their return series are correlated to something like 0.99. Almost
    all of each Sharpe estimate's error is therefore the *same* error, and it
    cancels when the series are differenced. Testing the gap between the two
    marginal ratios against either one's standard error asks a question about
    two independent samples, which these are not, and it answers "cannot tell"
    for years after the paired series could have answered it.

    Differencing is done day by day, on the dates both records measured, so
    the mask from the missing-price policy applies to the difference too.
    """
    a, b = policy.returns, benchmark.returns
    dates = a.index.intersection(b.index)
    if len(dates) < 2:
        return None
    active = (a.loc[dates] - b.loc[dates]).rename("active")
    clean = (policy.measured.loc[dates].to_numpy(dtype=bool)
             & benchmark.measured.loc[dates].to_numpy(dtype=bool))
    if int(clean.sum()) < 2:
        return None
    return track_record(active[clean], overlap=overlap,
                        periods_per_year=periods_per_year,
                        realised_returns=active)
