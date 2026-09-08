"""Which spread the cost model charges, and on what evidence.

`core/spread.py` estimates a spread and says how uncertain it is. This decides
what to do with that: whether the estimate is good enough to charge, what to
charge when it is not, and how to say which of the two happened.

Three tiers
-----------
Every instrument's half-spread carries a `spread_source`, and the cost model's
provenance report totals by it:

    observed   read off a document or a quote screen. Nothing is at this tier
               today. It is here because the tier system would otherwise be a
               two-way split with an aspirational third name, and because a
               half-spread that someone actually watched should not have to
               share a label with one this file computed.

    estimated  EDGE on the instrument's own bars, resolved at two standard
               errors, above the tick floor, and off enough bars. All four
               conditions, and any one of them failing drops the instrument to
               the tier below rather than shading the number.

    assumed    the declared constant. What every instrument used before this,
               and what an instrument still gets when its own data cannot
               support anything better.

The point of the split is not decoration. The cost model already reports what
fraction of its figure rests on documents; until now the spread -- the largest
single component -- was entirely in the assumed half by construction, and the
provenance line said so in a sentence that could never change. Now it can
change, and it changes per instrument, which is the whole reason to estimate
rather than assume: a 4 bps tracker and a 60 bps thematic fund are the same
number under a constant and 15 times apart under a measurement.

Why an unresolved estimate is discarded rather than used
--------------------------------------------------------
The estimator has a noise floor of roughly 4 bps off two years of daily bars,
which thins only as the fourth root of the sample. Below that it returns a
number that looks exactly like a measurement and is its own sampling error.
An instrument whose true spread is 2 bps and one whose data is simply too
short both come back near 4, and nothing in the number distinguishes them.

Charging 4 bps to such an instrument would not be a small error. It would be
an error that *reads as evidence*, and the cost model would report it in the
observed-or-estimated column, which is the one a reader uses to decide how
much of the answer to believe. The declared constant is worse as a number and
better as a claim.

Why the tick floor is a separate gate
-------------------------------------
Resolution is about the sample. The tick is about the venue. An instrument at
4 EUR on a one-cent grid cannot quote a half-spread under 12.5 bps, so an
estimate of 6 bps there is not a tight spread -- it is a sign that the
estimator found no bounce, whatever its standard error says about it. Both
gates can fail independently and each has instruments only it would catch.

When the clamp binds, the estimate is raised to the floor and *recorded as
clamped*, which is the third thing that can happen to an estimate and is
neither of the other two. A silent clamp would be a number the data does not
support wearing the estimated label.
"""

from __future__ import annotations

import dataclasses

from ..core.spread import SpreadEstimate, TickSize

__all__ = ["OBSERVED", "ESTIMATED", "ASSUMED", "SpreadDecision",
           "decide_spread", "ranking_is_plausible", "RankingCheck"]

OBSERVED = "observed"
ESTIMATED = "estimated"
ASSUMED = "assumed"

# The significance the estimate must clear, on s^2 -- see
# `SpreadEstimate.resolved` for why it is not on s.
SIGNIFICANCE = 2.0


@dataclasses.dataclass(frozen=True)
class SpreadDecision:
    """What the cost model will charge, and the sentence explaining it."""
    half_spread_bps: float
    source: str                          # one of the three tiers
    reason: str
    clamped_to_tick: bool = False
    estimate: SpreadEstimate | None = None
    tick: TickSize | None = None

    @property
    def is_evidence(self) -> bool:
        return self.source in (OBSERVED, ESTIMATED)

    def line(self, isin: str) -> str:
        mark = "*" if self.clamped_to_tick else " "
        return (f"{isin:<14}{mark}{self.half_spread_bps:7.1f} bps  "
                f"{self.source:<10}{self.reason}")


def decide_spread(estimate: SpreadEstimate, *, price: float,
                  tick: TickSize | None = None,
                  fallback_bps: float = 8.0,
                  significance: float = SIGNIFICANCE,
                  observed_bps: float | None = None) -> SpreadDecision:
    """Turn an estimate into a charge, or decline to.

    `fallback_bps` is the declared constant, used whenever the estimate does
    not clear every gate. `price` is a recent price for the instrument, needed
    only to express the tick as a fraction.

    `observed_bps` is a half-spread somebody watched and recorded. It wins
    outright -- a quote screen beats an inference from daily bars, and this
    file exists to rank evidence, not to prefer its own arithmetic. The
    estimate is still computed and still reported beside it, because two
    independent readings of the same quantity disagreeing by a factor is a
    finding about one of them:

    >>> from portfolio.core.spread import SpreadEstimate
    >>> wide = SpreadEstimate(0.0090, 8.1e-5, 0.0006, 500, 499,
    ...                       square_standard_error=1.1e-5)
    >>> d = decide_spread(wide, price=50.0, observed_bps=6.0)
    >>> d.source, d.half_spread_bps
    ('observed', 6.0)
    >>> print(d.reason)
    recorded from a quote, which outranks an inference from daily bars
    (EDGE on the same instrument's bars says 45.0 bps, 7.5x apart -- worth
    checking which is stale)

    A refusal from the estimator itself passes straight through:

    >>> from portfolio.core.spread import SpreadEstimate
    >>> nothing = SpreadEstimate(None, None, None, 30, 29,
    ...                          refusal="only 29 usable bars")
    >>> d = decide_spread(nothing, price=50.0)
    >>> d.source, d.half_spread_bps
    ('assumed', 8.0)
    >>> d.reason
    'no estimate: only 29 usable bars'

    An estimate that stands clear of zero is charged:

    >>> good = SpreadEstimate(0.0040, 1.6e-5, 0.0004, 500, 499,
    ...                       square_standard_error=3.2e-6)
    >>> d = decide_spread(good, price=50.0)
    >>> d.source, round(d.half_spread_bps, 1)
    ('estimated', 20.0)

    One that does not is not a narrow spread, it is no reading:

    >>> weak = SpreadEstimate(0.0008, 6.4e-7, 0.0006, 500, 499,
    ...                       square_standard_error=9.6e-7)
    >>> d = decide_spread(weak, price=50.0)
    >>> d.source, d.half_spread_bps
    ('assumed', 8.0)
    >>> print(d.reason)
    4.0 bps is 0.7 standard errors from zero, under 2.0; not distinguishable
    from no spread at all, which the estimator's own noise floor also is
    """
    if observed_bps is not None:
        note = ("recorded from a quote, which outranks an inference from "
                "daily bars")
        if estimate.spread is not None and observed_bps > 0:
            ratio = estimate.half_spread_bps / observed_bps
            if ratio > 2.0 or ratio < 0.5:
                note += (f"\n(EDGE on the same instrument's bars says "
                         f"{estimate.half_spread_bps:.1f} bps, "
                         f"{max(ratio, 1 / ratio):.1f}x apart -- worth\n"
                         f"checking which is stale)")
            else:
                note += (f"\n(EDGE on the same instrument's bars says "
                         f"{estimate.half_spread_bps:.1f} bps, which agrees)")
        return SpreadDecision(float(observed_bps), OBSERVED, note,
                              estimate=estimate, tick=tick)

    if estimate.spread is None:
        return SpreadDecision(fallback_bps, ASSUMED,
                              f"no estimate: {estimate.refusal}",
                              estimate=estimate)

    got = estimate.half_spread_bps
    assert got is not None
    t = estimate.t_statistic
    if t is None or t < significance:
        shown = "unknown" if t is None else f"{t:.1f}"
        return SpreadDecision(
            fallback_bps, ASSUMED,
            f"{got:.1f} bps is {shown} standard errors from zero, under "
            f"{significance:.1f}; not distinguishable\nfrom no spread at all, "
            f"which the estimator's own noise floor also is",
            estimate=estimate, tick=tick)

    floor = tick.floor_bps(price) if tick is not None else None
    if floor is not None and got < floor:
        return SpreadDecision(
            floor, ESTIMATED,
            f"{got:.1f} bps is under half a tick ({floor:.1f} bps at "
            f"{price:.2f}); raised to the floor, because a\nspread cannot be "
            f"finer than the grid the venue quotes on",
            clamped_to_tick=True, estimate=estimate, tick=tick)

    error = estimate.half_spread_error_bps
    return SpreadDecision(
        got, ESTIMATED,
        f"EDGE on {estimate.usable_bars} bars, +/- "
        f"{0.0 if error is None else error:.1f} at {t:.1f} standard errors",
        estimate=estimate, tick=tick)


@dataclasses.dataclass(frozen=True)
class RankingCheck:
    """Do the estimated spreads rank the way liquidity says they should?

    The controls establish that the estimator works on simulated markets. They
    cannot establish that it is being fed the right bars for the right
    instruments -- a symbol mapped to the wrong listing, a panel column
    misaligned by one, an adjusted series where an unadjusted one was meant.
    Every one of those produces spreads that are individually plausible and
    collectively nonsense.

    So: rank the instruments by estimated spread, rank them by a liquidity
    proxy, and check the two disagree the way they should. Bigger and more
    heavily traded means tighter, reliably enough that a *positive*
    correlation is evidence something is wired wrong, not evidence about the
    market.

    Deliberately weak. Spearman's rho over six or seven instruments has almost
    no power, so this is a check on gross wiring errors and is honest about
    being one. A near-zero correlation says nothing either way and reports
    itself as saying nothing.
    """
    rho: float | None
    instruments: int
    verdict: str
    passed: bool

    def line(self) -> str:
        if self.rho is None:
            return f"Ranking check: {self.verdict}"
        return (f"Ranking check: rho = {self.rho:+.2f} between estimated "
                f"spread and liquidity\nacross {self.instruments} "
                f"instruments. {self.verdict}")


def ranking_is_plausible(spreads: dict[str, float],
                         liquidity: dict[str, float], *,
                         minimum: int = 4) -> RankingCheck:
    """Spearman rank correlation between spread and a liquidity proxy.

    `liquidity` is anything where larger means more liquid -- average daily
    traded value is the natural one. Only instruments present in both are used.

    >>> tight = {"A": 4.0, "B": 8.0, "C": 20.0, "D": 55.0}
    >>> big = {"A": 900.0, "B": 400.0, "C": 90.0, "D": 5.0}
    >>> check = ranking_is_plausible(tight, big)
    >>> check.rho, check.passed
    (-1.0, True)

    Reversed, which is what a mis-mapped symbol would look like:

    >>> check = ranking_is_plausible(tight, {k: 1 / v for k, v in big.items()})
    >>> check.rho, check.passed
    (1.0, False)
    >>> print(check.verdict)
    The most liquid instruments are estimated to have the WIDEST spreads.
    That ordering is backwards; suspect the symbol mapping or the panel
    alignment before believing it.

    >>> ranking_is_plausible({"A": 4.0}, {"A": 900.0}).passed
    True
    >>> print(ranking_is_plausible({"A": 4.0}, {"A": 900.0}).verdict)
    only 1 instrument has both a spread and a liquidity figure; too few to rank
    """
    shared = sorted(set(spreads) & set(liquidity))
    n = len(shared)
    if n < minimum:
        return RankingCheck(
            None, n,
            f"only {n} instrument{'' if n == 1 else 's'} {'has' if n == 1 else 'have'} "
            f"both a spread and a liquidity figure; too few to rank", True)

    rho = _spearman([spreads[k] for k in shared], [liquidity[k] for k in shared])
    if rho > 0.5:
        return RankingCheck(rho, n, (
            "The most liquid instruments are estimated to have the WIDEST "
            "spreads.\nThat ordering is backwards; suspect the symbol mapping "
            "or the panel\nalignment before believing it."), False)
    if rho > -0.2:
        return RankingCheck(rho, n, (
            "Spread and liquidity are close to unranked here. Over this few "
            "instruments\nthat is weak evidence either way, and is reported "
            "rather than read as a pass."), True)
    return RankingCheck(rho, n, (
        "More liquid instruments are estimated tighter, which is the "
        "ordering\nliquidity predicts."), True)


def _spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, ties averaged. Written out; core/ has no scipy.

    >>> round(_spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 6)
    1.0
    >>> round(_spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]), 6)
    -1.0
    >>> round(_spearman([1.0, 1.0, 3.0], [5.0, 5.0, 9.0]), 6)
    1.0
    """
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    da = [x - mean_a for x in ra]
    db = [x - mean_b for x in rb]
    top = sum(x * y for x, y in zip(da, db))
    bottom = (sum(x * x for x in da) * sum(y * y for y in db)) ** 0.5
    return float(top / bottom) if bottom > 0 else 0.0


def _ranks(values: list[float]) -> list[float]:
    """Ranks, with ties given their average rank.

    >>> _ranks([10.0, 30.0, 20.0])
    [1.0, 3.0, 2.0]
    >>> _ranks([5.0, 5.0, 9.0])
    [1.5, 1.5, 3.0]
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = average
        i = j + 1
    return out
