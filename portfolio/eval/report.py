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
import math

import numpy as np

from ..core.returns import TRADING_DAYS_PER_YEAR
from .harness import BacktestResult
from .metrics import (SUSPICIOUS_ACTIVE_SHARPE, SUSPICIOUS_ANNUAL_SHARPE,
                      TrackRecord, sharpe_difference_standard_error,
                      track_record)

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
class RiskAdjusted:
    """The comparison at matched risk, and the split of the raw one.

    Why the raw active return is the wrong thing to test
    ----------------------------------------------------
    Equal risk contribution's whole function is to hold less of the volatile
    assets. Run it in a rising year and it underperforms *by construction*:
    a book carrying 14% less risk gives up 14% of the benchmark's return
    before any question of selection arises. A t-statistic on the raw active
    return then reports, with confidence, that a de-risking policy de-risked.
    Run the identical policy through a falling year and the identical test
    calls it significantly better. The sign belongs to the window, not to the
    strategy, so the number does not carry out of sample.

    That is this project's recurring defect wearing new clothes: a confidently
    correct figure measuring something other than what it claims.

    The identity
    ------------
    With the risk-free rate at zero, an annualised arithmetic return is Sharpe
    times volatility, so the raw gap splits exactly:

        R_p - R_b = (SR_p - SR_b) x sigma_b   +   SR_p x (sigma_p - sigma_b)
                    \\_____ selection _____/       \\______ mandate ______/

    *Selection* is what the policy cost after being levered to the benchmark's
    risk: the part a different window could have reversed. *Mandate* is what
    running at lower risk cost at the policy's own risk-adjusted rate: the
    part the policy was asked to produce.

    Both numbers are true and they are true about different decisions. The raw
    gap is what the account actually lost; the selection term is what the
    strategy cost. Neither is the answer on its own, so both are printed.

    What matched risk assumes
    -------------------------
    Scaling the policy to the benchmark's volatility means levering it by
    sigma_b / sigma_p. That is exact at a zero risk-free rate, and it is not
    available in a cash account at a retail broker. The comparison is
    therefore about the strategy, not about an executable alternative, and the
    report says so in those words.
    """
    sharpe_policy: float                 # annualised
    sharpe_benchmark: float
    volatility_policy: float             # annualised
    volatility_benchmark: float
    correlation: float                   # measured, never assumed
    observations: int                    # effective, at the frequency used
    difference: float                    # SR_p - SR_b, annualised
    standard_error: float                # of that difference, annualised
    raw_gap: float                       # annualised arithmetic, policy - benchmark
    selection: float                     # per year
    mandate: float                       # per year

    @property
    def t_statistic(self) -> float | None:
        if self.standard_error <= 0:
            return None
        return float(self.difference / self.standard_error)

    @property
    def decisive(self) -> bool:
        t = self.t_statistic
        return t is not None and abs(t) >= DECISIVE_T

    def lines(self) -> list[str]:
        out = [
            f"At matched risk: the policy's Sharpe is {self.difference:+.4f} "
            f"against buy-and-hold, standard error {self.standard_error:.4f} "
            f"at a measured correlation of {self.correlation:.4f} over "
            f"{self.observations} effective observations.",
        ]
        t = self.t_statistic
        if t is None:
            out.append("The two legs are indistinguishable to the last "
                       "decimal, so there is no difference to test.")
        elif abs(t) < DECISIVE_T:
            out.append(
                f"t = {t:+.2f}, inside {DECISIVE_T:.0f}: return per unit of "
                f"risk is INDISTINGUISHABLE between the two. This is the "
                f"comparison that carries out of sample, and on this sample it "
                f"does not resolve.")
        else:
            out.append(
                f"t = {t:+.2f}: return per unit of risk really does differ, by "
                f"more than this sample can attribute to chance.")
        # Sign-neutral wording on the mandate term. It is a cost only while
        # the benchmark is rising: with a negative Sharpe, carrying less risk
        # is what saved money, and calling that a cost would be exactly the
        # window-dependent reading this whole section exists to prevent.
        direction = ("less" if self.volatility_policy < self.volatility_benchmark
                     else "more")
        out.append(
            f"Splitting the {self.raw_gap:+.2%} a year raw gap (annualised "
            f"arithmetic, so it will not match the compounded figures in the "
            f"table exactly): {self.selection:+.2%} is selection, what the "
            f"strategy did once levered to the benchmark's risk, and "
            f"{self.mandate:+.2%} is mandate, what carrying {direction} risk "
            f"contributed by construction ({self.volatility_policy:.2%} "
            f"against {self.volatility_benchmark:.2%} annualised).")
        out.append(
            "Matched risk means levering the policy by "
            f"{self.volatility_benchmark / self.volatility_policy:.2f}x, which "
            f"a cash account cannot do. So the raw gap is what the account "
            f"lost and the selection term is what the strategy cost; they "
            f"answer different questions and neither replaces the other.")
        return out


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
    # information ratio: the realised cost of the mandate in THIS window, and
    # window-dependent by construction. Reported, never tested against.
    active: TrackRecord | None = None
    # The same two legs compared at matched risk, which is the part that
    # generalises. This is what the verdict tests.
    risk_adjusted: RiskAdjusted | None = None

    # -- the comparison ----------------------------------------------------

    def paired_lines(self) -> list[str]:
        """Two statistics on the same two series, labelled by their question.

        They can disagree, and when they do the disagreement is the finding.
        The raw information ratio answers "did I end the window with less
        money than doing nothing". The Sharpe difference answers "did I get
        less return per unit of risk". For a policy whose job is to hold less
        risk, the first is largely a restatement of the window's direction and
        the second is the one that carries out of sample -- so the verdict is
        applied to the second, and the first is labelled as what it is.
        """
        if self.active is None or self.active.sharpe is None:
            return ["No paired comparison: the two records do not share "
                    "enough dates to difference."]
        out = [
            f"Raw active return, policy net minus buy-and-hold day by day: "
            f"information ratio {self.active.band()}. This is the realised "
            f"cost of the mandate in THIS window and its sign is the window's "
            f"direction: a policy holding less risk underperforms a rising "
            f"benchmark by construction and would outperform a falling one. "
            f"Not tested, because a significant result here would be a "
            f"tautology.",
        ]
        if self.risk_adjusted is None:
            out.append("No risk-adjusted comparison: one of the legs has no "
                       "volatility to match to.")
            return out
        out += self.risk_adjusted.lines()
        return out

    def verdict(self) -> str:
        """The headline, with the benchmark in it.

        The level test on its own sent a reader hunting for a leak on a book
        whose benchmark scored higher still -- and a leak inside a rebalancing
        policy cannot lift a benchmark that never trades. What a shared high
        level does mean is that the cause is shared: the window, or the price
        data feeding both legs. Neither is evidence about the policy.

        The implausibility alarm is on the RISK-ADJUSTED gap rather than on
        the raw information ratio, for the same reason the verdict is: a
        de-risking policy in a falling market produces a large positive raw
        information ratio out of arithmetic alone, and the alarm would have
        fired on it.
        """
        if self.net.sharpe is None or self.benchmark.sharpe is None:
            return self.net.verdict()
        mine, theirs = self.net.sharpe, self.benchmark.sharpe
        if (self.risk_adjusted is not None
                and self.risk_adjusted.difference > SUSPICIOUS_ACTIVE_SHARPE):
            return (f"At matched risk the policy beats buy-and-hold by "
                    f"{self.risk_adjusted.difference:.2f} of annualised "
                    f"Sharpe, above the {SUSPICIOUS_ACTIVE_SHARPE} threshold. "
                    f"Doing nothing cannot produce that, a good window cannot "
                    f"explain it, and neither can holding more or less risk. "
                    f"Find the leak before reporting this as a result.")
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
        # The one-line summary a reader is entitled to: what the policy did to
        # risk, and whether the return per unit of it moved at all. Stating
        # only the first would flatter the policy; only the second would hide
        # the thing it was actually asked to do.
        own = self.net.verdict()
        ra = self.risk_adjusted
        if ra is None:
            return own
        change = ra.volatility_policy / ra.volatility_benchmark - 1.0
        moved = ("lower" if change < 0 else "higher")
        if ra.decisive:
            tail = (f"return per unit of risk differs by "
                    f"{ra.difference:+.2f} of Sharpe, t = {ra.t_statistic:+.2f}")
        else:
            tail = ("return per unit of risk is indistinguishable from doing "
                    "nothing on this sample")
        return (f"{own} Against the benchmark: volatility {abs(change):.0%} "
                f"{moved} ({ra.volatility_policy:.2%} against "
                f"{ra.volatility_benchmark:.2%}), and {tail}.")

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
    rho = _correlation(policy, benchmark)
    risk_adjusted = (None if rho is None
                     else _risk_adjusted(net, bench, rho))

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
        active=active, risk_adjusted=risk_adjusted)


def _correlation(policy: BacktestResult, benchmark: BacktestResult) -> float | None:
    """Measured, never assumed: every number in the paired test depends on it.

    Computed on the dates both records measured, so a day excluded from one
    leg's variance is excluded from the correlation too.
    """
    dates = policy.returns.index.intersection(benchmark.returns.index)
    if len(dates) < 3:
        return None
    clean = (policy.measured.loc[dates].to_numpy(dtype=bool)
             & benchmark.measured.loc[dates].to_numpy(dtype=bool))
    a = policy.returns.loc[dates].to_numpy()[clean]
    b = benchmark.returns.loc[dates].to_numpy()[clean]
    if len(a) < 3 or a.std(ddof=1) <= 0 or b.std(ddof=1) <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _risk_adjusted(policy: TrackRecord, benchmark: TrackRecord,
                   correlation: float) -> "RiskAdjusted | None":
    """The matched-risk comparison, and the split of the raw gap.

    Uses the two records' own effective observation counts, taking the smaller
    when they differ, so the paired standard error is never computed on more
    evidence than the shorter leg carries.
    """
    if (policy.sharpe is None or benchmark.sharpe is None
            or policy.volatility <= 0 or benchmark.volatility <= 0):
        return None
    n = min(policy.independent_observations, benchmark.independent_observations)
    if n < 2 or policy.years <= 0:
        return None
    # The formula wants both ratios and the count at ONE sampling interval.
    # `k` converts between the effective frequency and annual: an annualised
    # Sharpe is the per-observation one times sqrt(observations per year).
    # Feeding it annualised ratios with T in years is the natural mistake and
    # gives an answer about twice too large, because an asymptotic variance at
    # T = 1 is not an approximation of anything.
    k = math.sqrt(n / policy.years)
    se = sharpe_difference_standard_error(policy.sharpe / k,
                                          benchmark.sharpe / k, correlation, n)
    return RiskAdjusted(
        sharpe_policy=policy.sharpe, sharpe_benchmark=benchmark.sharpe,
        volatility_policy=policy.volatility,
        volatility_benchmark=benchmark.volatility,
        correlation=correlation, observations=n,
        difference=policy.sharpe - benchmark.sharpe,
        standard_error=se * k,
        raw_gap=(policy.sharpe * policy.volatility
                 - benchmark.sharpe * benchmark.volatility),
        selection=(policy.sharpe - benchmark.sharpe) * benchmark.volatility,
        mandate=policy.sharpe * (policy.volatility - benchmark.volatility))


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
