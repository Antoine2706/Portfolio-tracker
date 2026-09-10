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

    bounded    not distinguishable from zero, so the *upper confidence
               bound* is charged and labelled as a bound rather than an
               estimate. See below: failing the significance test does not
               mean nothing was learned.

    assumed    the declared constant. What every instrument used before this,
               and what an instrument still gets when it was not measurable
               at all -- too few bars, or prices that are not on any venue's
               tick grid, which means the series has been adjusted and is the
               wrong input entirely.

The point of the split is not decoration. The cost model already reports what
fraction of its figure rests on documents; until now the spread -- the largest
single component -- was entirely in the assumed half by construction, and the
provenance line said so in a sentence that could never change. Now it can
change, and it changes per instrument, which is the whole reason to estimate
rather than assume: a 4 bps tracker and a 60 bps thematic fund are the same
number under a constant and 15 times apart under a measurement.

Why an unresolved estimate becomes a bound rather than a number
---------------------------------------------------------------
The estimator has a noise floor of roughly 4 bps off two years of daily bars,
which thins only as the fourth root of the sample. Below that it returns a
number that looks exactly like a measurement and is its own sampling error.
An instrument whose true spread is 2 bps and one whose data is simply too
short both come back near 4, and nothing in the *point estimate*
distinguishes them.

Charging that 4 bps as an estimate would be an error that *reads as
evidence*. But falling back to the declared constant, which is what this file
did first, throws away something real. ``|s^2| >= k SE(s^2)`` failing says the
spread is not distinguishable from zero. It does not say nothing was learned:
it puts a **ceiling** on the spread, and the ceiling is per instrument.

So the upper confidence bound is charged instead:

    s_upper = sqrt(max(s^2 + k SE(s^2), 0))

which beats the constant on two counts. It is per instrument, because the
standard error depends on that instrument's own volatility and bar count, so
the differentiation the whole exercise exists for survives even where nothing
resolves. And it is falsifiable: a bound that sits below a spread later
observed on a quote screen is a bug report, which a constant never could be.

An earlier version of this paragraph offered a third count: that a ceiling
errs in the direction that costs least, since overstating the spread makes
the allocator too reluctant rather than too eager. That was withdrawn after
the first real book, where seven ceilings came back with the most liquid
holding widest. A ceiling is only as good as the bars under it; a carried
close pushes it up, and a number that is wrong in a direction one likes is
still wrong. The direction of an error is not evidence about its size.

There is a step at the threshold -- just below it the bound is charged, just
above it the point estimate -- and at exactly ``s^2 = k SE`` the bound is
``sqrt(2)`` times the estimate. That discontinuity is real and is the price of
a hard threshold. It is not smoothed over, because the two sides answer
different questions and the label says which is being answered.

If ``s^2 + k SE(s^2)`` is still negative, the squared spread is significantly
*negative*, which the model does not permit. That is not a tight spread; it is
the data contradicting the estimator, and it falls through to the constant
with that said.

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
import math

from ..core.spread import SpreadEstimate, TickSize

__all__ = ["OBSERVED", "ESTIMATED", "BOUNDED", "ASSUMED", "SpreadDecision",
           "decide_spread", "ranking_is_plausible", "RankingCheck"]

OBSERVED = "observed"
ESTIMATED = "estimated"
BOUNDED = "bounded"
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
        """Did this number come from the instrument's own data at all?"""
        return self.source in (OBSERVED, ESTIMATED, BOUNDED)

    @property
    def is_measurement(self) -> bool:
        """Is it a measurement of the spread, as opposed to a ceiling on it?

        Separate from `is_evidence` because the two answer different
        questions and conflating them is how a bound would come to be counted
        as a measurement in a provenance report.
        """
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

    One that does not is not a narrow spread; it is a ceiling, and the
    ceiling is charged:

    >>> weak = SpreadEstimate(0.0008, 6.4e-7, 0.0006, 500, 499,
    ...                       square_standard_error=9.6e-7)
    >>> d = decide_spread(weak, price=50.0)
    >>> d.source, round(d.half_spread_bps, 1)
    ('bounded', 8.0)
    >>> print(d.reason)
    4.0 bps is 0.7 standard errors from zero, under 2.0, so it is a ceiling
    and not a measurement: the spread is at most 8.0 bps off 499 bars

    A squared spread that is significantly negative is the data contradicting
    the estimator, not a tight market:

    >>> impossible = SpreadEstimate(0.0008, -6.4e-6, 0.0006, 500, 499,
    ...                             square_standard_error=9.6e-7)
    >>> d = decide_spread(impossible, price=50.0)
    >>> d.source, d.half_spread_bps
    ('assumed', 8.0)
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
    floor = tick.floor_bps(price) if tick is not None else None
    t = estimate.t_statistic

    if t is None or t < significance:
        shown = "unknown" if t is None else f"{t:.1f}"
        ceiling = _upper_bound_bps(estimate, significance)
        if ceiling is None:
            return SpreadDecision(
                fallback_bps, ASSUMED,
                f"the squared spread is {shown} standard errors from zero, "
                f"so its upper bound is\nnegative too. That is the data "
                f"contradicting the estimator rather than a\ntight market, "
                f"and it is not something to charge",
                estimate=estimate, tick=tick)
        clamped = floor is not None and ceiling < floor
        charged = max(ceiling, floor) if clamped else ceiling
        # How the point estimate is described depends on the sign of the
        # square, and the distinction is not pedantic. Where s^2 is positive
        # the root is a spread estimate that merely fails a significance test.
        # Where s^2 is NEGATIVE the root is sqrt(|s^2|), which is not an
        # estimate of anything -- printing it as "10.8 bps" next to a ceiling
        # of 6.5 reads as a contradiction, when in fact a negative square is
        # exactly what a spread too small to detect looks like and the low
        # ceiling is the informative half.
        if (estimate.signed_square or 0.0) < 0:
            head = (f"the squared spread came out negative "
                    f"({shown} standard errors below zero), which is what a "
                    f"spread too\nsmall for {estimate.usable_bars} bars to "
                    f"detect looks like. So there is no reading, only a "
                    f"ceiling:\nthe spread is at most {charged:.1f} bps")
        else:
            head = (f"{got:.1f} bps is {shown} standard errors from zero, "
                    f"under {significance:.1f}, so it is a ceiling\nand not "
                    f"a measurement: the spread is at most {charged:.1f} bps "
                    f"off {estimate.usable_bars} bars")
        return SpreadDecision(
            charged, BOUNDED,
            head + ("\n(raised to half a tick, which is a firmer floor than "
                    "the data gives)" if clamped else ""),
            clamped_to_tick=clamped, estimate=estimate, tick=tick)

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
    instruments -- a feed that carries a close forward on days a venue was
    shut, a symbol resolving to the wrong listing, a volume figure that
    belongs to another line of the same fund. Every one of those produces
    spreads that are individually plausible and collectively nonsense.

    So: rank the instruments by estimated spread, rank them by a liquidity
    proxy, and check the two disagree the way they should. Bigger and more
    heavily traded means tighter, reliably enough that a *positive*
    correlation is evidence something is wired wrong, not evidence about the
    market.

    Deliberately weak. Spearman's rho over six or seven instruments has almost
    no power, so this is a check on gross wiring errors and is honest about
    being one. A near-zero correlation says nothing either way and reports
    itself as saying nothing.

    What it does not do is say *why* it failed. It compares two numbers per
    instrument and either can be the wrong one; the first version of this
    verdict said "suspect the symbol mapping or the panel alignment", which
    was a guess dressed as a diagnosis and sent the reader to the wrong
    place. It now carries the pairs it ranked, so that the reader can see
    which instruments drove the correlation, and leaves the cause to the
    checks that can measure it: the per-instrument bar counts in the survey,
    and the ladder control on instruments whose spread is not in doubt.
    """
    rho: float | None
    instruments: int
    verdict: str
    passed: bool
    # (key, half-spread bps, liquidity), most liquid first. The evidence the
    # verdict was reached on, printed with it so a failure is attributable.
    pairs: tuple[tuple[str, float, float], ...] = ()

    def table(self, names: dict[str, str] | None = None) -> list[str]:
        """The pairs, most liquid first, with the rank each number holds."""
        if not self.pairs:
            return []
        by_spread = sorted(self.pairs, key=lambda p: p[1])
        spread_rank = {p[0]: i + 1 for i, p in enumerate(by_spread)}
        out = [f"  {'liquidity rank':>14}  {'spread rank':>11}  "
               f"{'half-spread':>11}  {'median daily traded':>19}  instrument"]
        for i, (key, spread, traded) in enumerate(self.pairs):
            label = (names or {}).get(key, "")
            out.append(f"  {i + 1:>14}  {spread_rank[key]:>11}  "
                       f"{spread:8.1f} bps  {_money(traded):>19}  {key}"
                       f"{'  ' + label if label else ''}")
        return out

    def line(self, names: dict[str, str] | None = None) -> str:
        if self.rho is None:
            return f"Ranking check: {self.verdict}"
        head = (f"Ranking check: rho = {self.rho:+.2f} between estimated "
                f"spread and liquidity\nacross {self.instruments} "
                f"instruments. {self.verdict}")
        table = self.table(names)
        return head + ("\n" + "\n".join(table) if table else "")


def _money(value: float) -> str:
    """A traded value, in a unit a reader can rank at a glance."""
    if value >= 1e9:
        return f"{value / 1e9:.1f}bn"
    if value >= 1e6:
        return f"{value / 1e6:.1f}m"
    if value >= 1e3:
        return f"{value / 1e3:.0f}k"
    return f"{value:.0f}"


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

    Reversed, which is what a mis-mapped symbol, a carried close on the
    liquid lines, or a volume figure from the wrong listing would all look
    like. The verdict names what was measured and which instruments sit at
    the extremes; it does not pick among the causes, because it cannot:

    >>> check = ranking_is_plausible(tight, {k: 1 / v for k, v in big.items()})
    >>> check.rho, check.passed
    (1.0, False)
    >>> print(check.verdict)
    The most liquid instruments are estimated to have the WIDEST spreads: the
    most traded by the proxy, D, is estimated at 55.0 bps and the least
    traded, A, at 4.0. That ordering is backwards. This check compares two
    numbers per instrument and cannot say which is wrong: the bars (a feed
    that carries a close forward inflates the estimate -- see the bars set
    aside per instrument above), the liquidity proxy (a volume figure from
    another listing, or none), or the symbol (the wrong line of the fund).
    The pairs it ranked follow.

    >>> ranking_is_plausible({"A": 4.0}, {"A": 900.0}).passed
    True
    >>> print(ranking_is_plausible({"A": 4.0}, {"A": 900.0}).verdict)
    only 1 instrument has both a spread and a liquidity figure; too few to rank
    """
    shared = sorted(set(spreads) & set(liquidity))
    n = len(shared)
    pairs = tuple(sorted(((k, float(spreads[k]), float(liquidity[k]))
                          for k in shared), key=lambda p: -p[2]))
    if n < minimum:
        return RankingCheck(
            None, n,
            f"only {n} instrument{'' if n == 1 else 's'} {'has' if n == 1 else 'have'} "
            f"both a spread and a liquidity figure; too few to rank", True,
            pairs=pairs)

    rho = _spearman([spreads[k] for k in shared], [liquidity[k] for k in shared])
    if rho > 0.5:
        most, least = pairs[0], pairs[-1]
        return RankingCheck(rho, n, (
            f"The most liquid instruments are estimated to have the WIDEST "
            f"spreads: the\nmost traded by the proxy, {most[0]}, is estimated "
            f"at {most[1]:.1f} bps and the least\ntraded, {least[0]}, at "
            f"{least[1]:.1f}. That ordering is backwards. This check compares "
            f"two\nnumbers per instrument and cannot say which is wrong: the "
            f"bars (a feed\nthat carries a close forward inflates the estimate "
            f"-- see the bars set\naside per instrument above), the liquidity "
            f"proxy (a volume figure from\nanother listing, or none), or the "
            f"symbol (the wrong line of the fund).\nThe pairs it ranked "
            f"follow."), False, pairs=pairs)
    if rho > -0.2:
        return RankingCheck(rho, n, (
            "Spread and liquidity are close to unranked here. Over this few "
            "instruments\nthat is weak evidence either way, and is reported "
            "rather than read as a pass."), True, pairs=pairs)
    return RankingCheck(rho, n, (
        "More liquid instruments are estimated tighter, which is the "
        "ordering\nliquidity predicts."), True, pairs=pairs)


def _upper_bound_bps(estimate: SpreadEstimate,
                     significance: float) -> float | None:
    """The upper confidence bound on the half-spread, in bps, or None.

    Formed on ``s^2`` and then rooted, because that is where the sampling
    distribution is symmetric and where the standard error was validated.
    None when the bound is itself negative, which means the estimate is
    significantly below zero and the model is contradicted rather than the
    spread being small.

    >>> from portfolio.core.spread import SpreadEstimate
    >>> e = SpreadEstimate(0.0008, 6.4e-7, 0.0006, 500, 499,
    ...                    square_standard_error=9.6e-7)
    >>> round(_upper_bound_bps(e, 2.0), 2)      # sqrt(6.4e-7 + 2*9.6e-7)
    8.0
    >>> _upper_bound_bps(SpreadEstimate(0.001, -1e-5, 0.0005, 500, 499,
    ...                                 square_standard_error=1e-6), 2.0) is None
    True
    """
    if estimate.signed_square is None or not estimate.square_standard_error:
        return None
    upper = estimate.signed_square + significance * estimate.square_standard_error
    if upper <= 0:
        return None
    return float(math.sqrt(upper) * 10_000.0 / 2.0)


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
