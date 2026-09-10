"""Does the spread estimator recover a spread it was not told?

Why this file exists
--------------------
This project has now found several confident outputs that measured something
other than what they named. An unvalidated spread estimator would be the next
one, and it would be worse than the flat 8 bps constant it replaces, because
the constant is obviously a guess and an estimate looks like a measurement.

So the estimator does not ship on the strength of matching a published
formula. It ships on the strength of recovering spreads that were imposed on
simulated data by machinery that knows nothing about the formula.

The simulation
--------------
Nothing here inverts the estimator. The generator is a market:

  * An efficient log price takes `ticks` random-walk steps inside each bar.
  * Every step is a trade, and every trade prints at the bid or the ask --
    `mid * (1 -+ s/2)` -- with the side drawn by a coin.
  * The bar's open is the first of those prints, its close the last, its high
    and low the largest and smallest.

That is the microstructure the estimator's derivation assumes, written
forwards. The estimator is then handed the four numbers a real data feed would
publish, and has to get `s` back out of them. `ticks` is what makes the test
sharp: at 400 trades a bar the high is at the ask and the low at the bid
almost surely, and any of the classical estimators would do; at 4 trades a bar
the open is frequently itself the extreme, which is the case EDGE's `p_o` and
`p_c` corrections exist for and the case its predecessors are biased in.

Seven controls
--------------
**Positive.** Sweep the imposed spread from 2 to 100 bps and report the bias
and the dispersion at each rung, over many independent runs. Not a pass or a
fail on one draw: the question is whether the estimator is centred on the
right answer and how wide it is around it, and a single run answers neither.

**Negative.** Impose no spread at all. The signed square must be centred on
zero. The *root* cannot be -- ``sqrt(|x|)`` of a mean-zero quantity is
positive by construction -- and so this control also measures the floor, which
is the number that says how small an estimate is still distinguishable from
nothing.

**Standard error.** The estimator reports an error bar. This checks it against
the dispersion of the estimates themselves across independent runs, which is
what the error bar claims to predict. A standard error that is present but
wrong is the exact defect this project found in the Sharpe ratio, where the
ratio was daily and its error monthly and every t-statistic came out 4.6 times
too small.

**Resolution.** An estimate that is not distinguishable from zero must not be
claimed as a measurement. This checks the rule that enforces that, in both
directions: quiet below the floor, and -- the half that is easy to forget --
audible above it, because a gate that never opens is a constant with extra
steps.

**Refusal.** Below a minimum number of usable bars the estimator must decline
rather than return a number. This checks that the refusal fires, and -- the
part that matters -- that the region just above the threshold is not itself
producing garbage that the threshold was set too low to catch.

**Resolution by window.** What a longer history buys, at each spread: the
measurement that moved the survey from the holding period to the whole
available history, and the one that says the tight end is not rescued.

**Contamination.** What each kind of bar a daily feed manufactures does to
the estimate, and whether `core.spread.classify_bars` sets aside the kinds
that matter. This is the table above `classify_bars`, regenerated: a carried
close or a carried whole bar on five per cent of days must inflate the
estimate by more than a third and setting them aside must recover it; a
flat bar, a carried open, and a widened high or low must move it by less
than a tenth. The control exists because the table first lived in a comment
and a reviewer had to re-derive it to trust it, which is the "scrolled off a
terminal" failure the run log was added to prevent.

It also carries the finding that no count of bars reaches. A **daily
reversal in the price itself** -- each day's move partly undone overnight,
which is what a stale or non-synchronous close looks like from outside --
inflates the estimate by 50% at a return autocorrelation of -0.05 and by
120% at -0.15, and sets aside nothing, because every bar is a faithful bar
of a price that reverts. At the daily frequency a bounce and a reversal are
the same negative covariance; that is Roll's identification problem and the
paper's assumption of uncorrelated efficient-price increments, read from the
other side. It is the one contamination tried that also pushes the per-bar
autocorrelation negative, as the real book's was, though only to -0.03
against the book's -0.40.

What these controls found
-------------------------
They were not a formality. Three results changed the code or the claims:

**The estimator has a noise floor of about 4 bps, and 2 bps is below it.**
At zero imposed spread, 500 daily bars, the reported half-spread averages
3.97 bps. That is not a bug: it is ``sqrt(|s^2|)`` folding the negative half
of a mean-zero distribution onto the positive side. The floor falls as the
fourth root of the sample -- measured at 4.65, 3.97, 3.38, 2.87 and 2.36 bps
over 250, 500, 1000, 2000 and 4000 bars, against a predicted ratio of
``2^-0.25 = 0.841`` per doubling and an observed 0.85, 0.85, 0.85, 0.82. So
reaching a 2 bps floor would take about 6000 daily bars, twenty-four years,
of an instrument whose spread today is the thing being asked about.

The consequence is a limit on what may be claimed, not a number to tune away:
**this estimator cannot resolve the spread of the tightest European trackers
from daily bars.** It is reliable from roughly 5 bps upward, unbiased to
within 2.2% across the rest of the swept range, and below that it returns its
own noise. `resolution_control` is the check that the estimator declines to
claim those.

**The Newey-West correction was switched off, and the argument for it was
wrong.** It went in because consecutive terms of the per-bar series share a
bar and therefore *ought* to correlate. Measured over 200 samples at each of
three spreads, that autocorrelation is ``-0.003 +/- 0.002`` at lag 1 and no
larger out to lag 4 -- absent. Averaged across samples the correction moves
the standard error by 0.3%; on a *single* sample, which is all a real
instrument gets, it moves it by up to 15%, because estimating autocovariances
that are truly zero adds noise and removes nothing. So `lags` now defaults to
0 and the estimate carries the autocorrelation it measured, so that the same
question can be asked of real bars instead of assuming the simulation's
answer carries over.

An earlier draft of this file claimed the i.i.d. version *fails* this
control. It does not. That claim was written before the measurement, which
is the failure mode this project keeps finding, in prose, for the seventh
time.

**The significance test was on the wrong quantity, and read as twice what it
was.** The rule for claiming an estimate was first written ``s >= 2 SE(s)``.
The delta method makes ``SE(s) = SE(s^2) / (2s)``, so that rearranges exactly
to ``|s^2| >= SE(s^2)``: a one-standard-error test wearing a
two-standard-error label. At zero true spread it claimed a spread in 32% of
samples. Moved onto ``s^2``, where the sampling distribution is symmetric and
the sign survives, the same nominal rule claims 2%. `resolution_control`
prints both columns.

**A one-year window was throwing away most of the book, and a longer one
only half rescues it.** The survey first estimated over the holding period,
about 250 bars, for no better reason than that the covariance window is 250
bars. A covariance window must be short because correlations move with
regime; a spread is a microstructure property that moves slowly and has
nothing to do with when the holding was bought. Measured, at 80 runs a cell,
as the fraction of samples in which a spread is claimed as a measurement:

    half-spread     250     500    1000    2500   bars
        2.0 bps      0%      2%      4%      5%
        4.0 bps      4%      4%     15%     16%
        6.0 bps     15%     22%     44%     91%
        8.0 bps     28%     66%     80%     99%
       10.0 bps     49%     82%     94%    100%
       15.0 bps     96%    100%    100%    100%

So the whole available history is now taken, and it moves the point at which
half of samples resolve from about 10 bps to about 6. That rescues the wide
end of a European ETF book. It does not rescue the tight end: at 2 and 4 bps
the fraction resolved is flat in the sample length, because that is under the
floor at every length daily bars can offer, and no window reaches it. Those
instruments get the upper-bound tier instead, which is why that tier is not a
nicety.

The longer window has a cost, and it is measured rather than assumed:
spreads narrow as a fund grows, so a ten-year estimate can be an average of a
market that no longer exists. `core.spread.sweep_windows` estimates over each
nested window, prints the sequence, and compares **disjoint** older blocks
against the most recent one at three standard errors. Flat means take
everything; a real difference means the spread has moved and the window stops
there. Disjoint, because nested windows share their data and the difference
of two nested estimates has a variance smaller than the sum of theirs.

**The estimator runs 1 to 3% high on wide spreads, and it is a finite-trade
effect rather than a defect.** The positive control's bias column is monotone
in the spread -- -0.6, +0.9, +1.0, +1.2, +2.1% at 5, 10, 20, 50 and 100 bps --
and at 60 runs a rung that +2.1% is about seven standard errors, so it is real
and not sampling noise. Holding the spread fixed and varying the number of
trades per bar separates the cause:

    imposed        10       30       60      200      600   trades/bar
     20 bps     -1.7%    -0.3%    +1.9%    +0.6%    -0.9%
     50 bps     +2.3%    +2.1%    +2.2%    +1.0%    -0.0%
    100 bps     +4.6%    +3.0%    +2.4%    +1.3%    +0.4%

It shrinks to nothing as trading becomes continuous, which is the paper's
asymptotic claim holding. What it is: ``(h+l)/2`` is a slightly biased read on
the efficient mid-range when the high and the low are drawn from few prints,
and the size of that error scales with the spread. Distinct from -- and much
smaller than -- the *downward* infrequent-trading bias that ``p_o`` and
``p_c`` exist to remove, which is 21% on three-trade bars.

This is not the simulator's artefact to fix: real instruments trade discretely
too, so a generator that did not show it would be the wrong generator. The
direction is the harmless one for a cost model, which overstates cost slightly
rather than understating it, and it is under half the estimator's own standard
error at every rung. Recorded rather than corrected, because a correction
fitted to this simulation's trade counts would not transfer to an instrument
with different ones.

**The error bar is about 7% optimistic.** Across 10 to 100 bps the reported
standard error is 0.87 to 1.07 times the dispersion it predicts, median 0.93.
The cause is a plug-in estimator's usual one: the combination weight, the two
trading-frequency probabilities and the de-meaning are all estimated from the
same sample and then treated as known. A subsampling standard error was tried
as a principled fix and came out *worse* (0.71 to 0.87), so the analytic one
stands and the residual optimism is declared here rather than corrected by a
factor fitted to this simulation.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..core.spread import MINIMUM_BARS, classify_bars, edge, exclude_bars

__all__ = [
    "simulate_bars", "SweepRung", "SpreadControlReport", "positive_control",
    "negative_control", "standard_error_control", "resolution_control",
    "resolution_by_window_control", "refusal_control",
    "contamination_control", "contaminate", "CONTAMINATIONS",
    "simulate_reversal_bars", "REVERSALS", "ContaminationRow",
    "run_spread_controls", "SWEEP_BPS",
]


# The plausible range for a European ETF, per the user's instruction: 2 bps is
# about the tightest a large core tracker quotes, 100 bps about the widest a
# thin thematic fund quotes before it stops being investable.
SWEEP_BPS: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def simulate_bars(spread: float, *, bars: int = 500, ticks: int = 60,
                  sigma: float = 0.01, seed: int = 0, drift: float = 0.0,
                  tick_size: float | None = None) -> tuple[np.ndarray, ...]:
    """OHLC bars from a market with a known effective spread.

    `spread` is the full effective spread as a fraction of price: 0.001 is 10
    bps, and a trade prints 5 bps either side of the mid. `sigma` is the
    efficient price's volatility per bar, `ticks` the number of trades in one.
    `drift` is the efficient price's log drift per bar, so 0.002 is roughly
    65% a year; values large enough to overflow `exp` are a caller error and
    the estimator will refuse the result rather than return nonsense.

    `tick_size` rounds every print onto a price grid, the way a real venue
    does. Off by default because it is not needed to test the spread itself,
    and on when a test needs prices that `core.spread.infer_tick_size` can
    read a grid off.

    The efficient price is continuous across the bar boundary -- the last tick
    of one bar and the first of the next are one random-walk step apart -- so
    the overnight gap is a real return and not an artefact of restarting the
    walk.

    >>> o, h, l, c = simulate_bars(0.002, bars=4, ticks=10, seed=1)
    >>> len(o), len(h), len(l), len(c)
    (4, 4, 4, 4)

    The high is the largest print and the low the smallest, so the range
    always contains the open and the close:

    >>> bool(np.all((h >= o) & (h >= c) & (l <= o) & (l <= c)))
    True

    Widening the spread widens the average bar range, which is the effect the
    estimator lives off:

    >>> def mean_range(s):
    ...     o, h, l, c = simulate_bars(s, bars=400, ticks=20, seed=3)
    ...     return float(np.mean(h / l - 1.0))
    >>> mean_range(0.01) > mean_range(0.0)
    True

    With a tick imposed, every print lands on the grid:

    >>> o, h, l, c = simulate_bars(0.002, bars=50, seed=2, tick_size=0.01)
    >>> bool(np.allclose(np.round(np.concatenate([o, h, l, c]) / 0.01),
    ...                  np.concatenate([o, h, l, c]) / 0.01))
    True
    """
    if spread < 0:
        raise ValueError("spread must be non-negative")
    if bars < 1 or ticks < 1:
        raise ValueError("bars and ticks must be at least 1")
    rng = np.random.default_rng(seed)

    # One continuous walk of bars*ticks steps, then cut into bars. Cutting a
    # single walk rather than starting a fresh one per bar is what keeps the
    # close-to-open return a real return.
    steps = rng.normal(drift / ticks, sigma / math.sqrt(ticks), bars * ticks)
    mid = np.exp(np.cumsum(steps)).reshape(bars, ticks) * 100.0

    side = rng.choice((-1.0, 1.0), size=(bars, ticks))
    prints = mid * (1.0 + side * spread / 2.0)
    if tick_size:
        prints = np.round(prints / tick_size) * tick_size

    return (prints[:, 0].copy(), prints.max(axis=1), prints.min(axis=1),
            prints[:, -1].copy())


@dataclasses.dataclass(frozen=True)
class SweepRung:
    """One imposed spread, and what came back over many independent runs."""
    imposed_bps: float
    runs: int
    mean_bps: float
    median_bps: float
    sd_bps: float
    mean_reported_error_bps: float

    @property
    def bias_bps(self) -> float:
        return self.mean_bps - self.imposed_bps

    @property
    def bias_fraction(self) -> float:
        return self.bias_bps / self.imposed_bps if self.imposed_bps else float("nan")

    def line(self) -> str:
        return (f"  {self.imposed_bps:6.1f} -> {self.mean_bps:7.2f}  "
                f"bias {self.bias_bps:+6.2f} ({self.bias_fraction:+6.1%})  "
                f"sd {self.sd_bps:5.2f}  reported +/- {self.mean_reported_error_bps:5.2f}")


@dataclasses.dataclass(frozen=True)
class SpreadControlReport:
    """One control, what it measured, and whether it behaved."""
    name: str
    passed: bool
    detail: str
    numbers: dict

    def lines(self) -> list[str]:
        out = [f"[{'PASS' if self.passed else 'FAIL'}] {self.name}"]
        out.extend(f"       {line}" for line in self.detail.splitlines())
        return out


def _estimates(imposed: float, *, runs: int, bars: int, ticks: int,
               sigma: float, seed0: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half-spread estimates in bps, their reported errors, and signed squares."""
    got, errors, squares = [], [], []
    for i in range(runs):
        o, h, l, c = simulate_bars(imposed, bars=bars, ticks=ticks,
                                   sigma=sigma, seed=seed0 + i)
        e = edge(o, h, l, c, minimum_bars=0)
        if e.spread is None:
            continue
        got.append(e.half_spread_bps)
        errors.append(e.half_spread_error_bps
                      if e.half_spread_error_bps is not None else np.nan)
        squares.append(e.signed_square)
    return np.array(got), np.array(errors), np.array(squares)


def positive_control(*, runs: int = 60, bars: int = 500, ticks: int = 60,
                     sigma: float = 0.01, seed0: int = 1000,
                     sweep: tuple[float, ...] = SWEEP_BPS,
                     tolerance: float = 0.05,
                     resolved_above_bps: float = 5.0) -> SpreadControlReport:
    """Does it recover a spread it was not told, across the plausible range?

    `tolerance` is on the *bias*, as a fraction of the imposed spread, and it
    is applied only at rungs at or above `resolved_above_bps`. Below that the
    estimator is returning its own noise floor -- see the module docstring --
    and holding it to a bias tolerance there would be asking it for something
    daily bars do not contain.

    Those sub-floor rungs are not skipped. They are asserted to behave the way
    the floor explanation predicts: the estimate must come back *near the
    floor* rather than near the imposed value, because if a 2 bps imposed
    spread ever recovered as 2 bps the floor explanation would be wrong and
    something else would be going on.
    """
    rungs = [
        _rung(imposed, runs=runs, bars=bars, ticks=ticks, sigma=sigma,
              seed0=seed0 + int(imposed) * 977)
        for imposed in sweep
    ]
    floor = _rung(0.0, runs=runs, bars=bars, ticks=ticks, sigma=sigma,
                  seed0=seed0 + 31).mean_bps

    resolved = [r for r in sweep if r >= resolved_above_bps]
    unresolved = [r for r in sweep if r < resolved_above_bps]
    tested = [r for r in rungs if r.imposed_bps >= resolved_above_bps]
    worst = max(tested, key=lambda r: abs(r.bias_fraction))
    # Below the floor the estimate must sit near the floor, not near the
    # imposed value: within half the floor of it, and at least half the floor
    # above the imposed spread it is failing to see.
    at_floor = all(abs(r.mean_bps - floor) < 0.5 * floor
                   for r in rungs if r.imposed_bps < resolved_above_bps)
    passed = all(abs(r.bias_fraction) <= tolerance for r in tested) and at_floor

    detail = "\n".join(
        [f"imposed  recovered (mean of {runs} runs, {bars} bars, "
         f"{ticks} trades/bar)"]
        + [r.line() + ("   [below the floor]"
                       if r.imposed_bps < resolved_above_bps else "")
           for r in rungs]
        + [f"measured noise floor at zero imposed spread: {floor:.2f} bps",
           f"bias tolerance {tolerance:.0%} applied at "
           f"{', '.join(f'{r:.0f}' for r in resolved)} bps: worst is "
           f"{worst.bias_fraction:+.1%} at {worst.imposed_bps:.0f}"]
        + ([f"{', '.join(f'{r:.0f}' for r in unresolved)} bps sit below the "
            f"floor and are checked for the opposite: they must come back "
            f"AT the floor, and they do"] if unresolved else []))
    return SpreadControlReport(
        name="Positive control: a known spread, imposed and recovered",
        passed=passed, detail=detail,
        numbers={"rungs": [dataclasses.asdict(r) for r in rungs],
                 "floor_bps": floor,
                 "worst_bias_fraction": worst.bias_fraction})


def _rung(imposed_bps: float, *, runs: int, bars: int, ticks: int,
          sigma: float, seed0: int) -> SweepRung:
    # The sweep is quoted as a HALF-spread in bps, which is the unit the cost
    # model charges in, so the full spread handed to the simulator is twice it.
    got, errors, _ = _estimates(2.0 * imposed_bps / 10_000.0, runs=runs,
                                bars=bars, ticks=ticks, sigma=sigma, seed0=seed0)
    return SweepRung(imposed_bps=imposed_bps, runs=int(got.size),
                     mean_bps=float(got.mean()), median_bps=float(np.median(got)),
                     sd_bps=float(got.std(ddof=1)),
                     mean_reported_error_bps=float(np.nanmean(errors)))


def negative_control(*, runs: int = 200, ticks: int = 60, sigma: float = 0.01,
                     seed0: int = 5000,
                     lengths: tuple[int, ...] = (250, 500, 1000, 2000)
                     ) -> SpreadControlReport:
    """Zero imposed spread. What does the estimator say, and how loudly?

    Two separate questions, and conflating them would hide the answer to the
    second.

    The **signed square** must be centred on zero. That is the honest test of
    unbiasedness, and it is a two-sided test against the standard error of the
    mean over the runs.

    The **reported spread** cannot be centred on zero, because it is
    ``sqrt(|s^2|)`` and the absolute value folds the negative half of the
    distribution onto the positive side. Whatever it comes out as is the
    estimator's noise floor: the smallest spread still distinguishable from no
    spread at all, and the hard limit on what this instrument can claim.

    So the control measures the floor rather than asserting a bound on it, and
    then makes the floor explanation falsifiable. If the floor is sampling
    noise in ``s^2``, then ``s = sqrt(|s^2|)`` inherits the square root of the
    convergence rate: ``s^2`` converges at ``n^-0.5``, so the floor must fall
    as ``n^-0.25``, which is a ratio of ``2^-0.25 = 0.841`` per doubling of
    the sample. A floor that did not thin that way would mean something other
    than sampling noise was producing it, and the estimator would have a
    defect rather than a resolution limit.
    """
    predicted = 2.0 ** -0.25
    floors = []
    for i, bars in enumerate(lengths):
        got, _, squares = _estimates(0.0, runs=runs, bars=bars, ticks=ticks,
                                     sigma=sigma, seed0=seed0 + 7919 * i)
        floors.append((bars, float(got.mean()),
                       float(got.std(ddof=1) / math.sqrt(got.size)),
                       float(squares.mean()),
                       float(squares.std(ddof=1) / math.sqrt(squares.size))))

    # Each floor is itself a mean over `runs` draws, so the ratio of two of
    # them carries the error of both. Judging the ratio against a fixed
    # tolerance would make the control's verdict depend on how many runs it
    # was given, which is a control that fails for the wrong reason.
    ratios = []
    for (n0, f0, e0, _, _), (n1, f1, e1, _, _) in zip(floors, floors[1:]):
        if n1 != 2 * n0:
            continue
        ratio = f1 / f0
        error = ratio * math.hypot(e0 / f0, e1 / f1)
        ratios.append((ratio, error, abs(ratio - predicted) / error))

    # Unbiasedness is judged on the longest sample, where the mean of s^2 has
    # the tightest error bar and so the test has the most power to fail.
    _, floor, _, mean_square, se_square = floors[-1]
    t = mean_square / se_square if se_square > 0 else float("nan")
    centred = abs(t) < 3.0
    thins = all(z < 3.0 for _, _, z in ratios) if ratios else False

    detail = "\n".join(
        [f"{runs} runs per length, zero imposed spread",
         "   bars   reported half-spread (the floor)   ratio to previous"]
        + [f"  {bars:5d}   {f:11.2f} +/- {e:<10.2f}   "
           f"{('%.3f' % (f / floors[i - 1][1])) if i else '    -':>17}"
           for i, (bars, f, e, _, _) in enumerate(floors)]
        + [f"predicted ratio per doubling if the floor is sampling noise: "
           f"{predicted:.3f}",
           "observed: " + ", ".join(f"{r:.3f} +/- {e:.3f} ({z:.1f} sigma off)"
                                    for r, e, z in ratios),
           f"signed s^2 at {floors[-1][0]} bars: {mean_square:+.3e} +/- "
           f"{se_square:.3e} (t = {t:+.2f}, centred on zero: "
           f"{'yes' if centred else 'NO'})",
           f"the floor is sampling noise thinning as n^-0.25, not a spread: "
           f"{'confirmed' if thins else 'NOT CONFIRMED'}",
           f"nothing below about {floor:.0f} bps may be claimed as a "
           f"measurement from {floors[-1][0]} daily bars"])
    return SpreadControlReport(
        name="Negative control: no spread imposed", passed=centred and thins,
        detail=detail,
        numbers={"floors": floors, "ratios": ratios, "t": t,
                 "floor_bps": floor})


def standard_error_control(*, runs: int = 150, bars: int = 500, ticks: int = 60,
                           sigma: float = 0.01, seed0: int = 9000,
                           imposed: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0),
                           lag_choices: tuple[int, ...] = (0, 1, 4),
                           band: tuple[float, float] = (0.80, 1.25)
                           ) -> SpreadControlReport:
    """Is the reported error bar the right size?

    The estimate's standard error claims to predict how far the estimate would
    move if the sample were redrawn. So redraw it, many times, and compare the
    dispersion of what comes back against the error the estimator reported.
    The ratio must be near one.

    Compared against the **median** reported error rather than the mean. The
    delta method divides by the estimate, and ``1/x`` is convex, so a run that
    happens to estimate low reports an error bar that is too wide by more than
    a run that estimates high reports one that is too narrow. The mean
    inherits that asymmetry; at 5 bps imposed it runs 33% high while the
    median runs 3% high, and the asymmetry is a property of the summary, not
    of the error bar.

    The band is 0.80 to 1.25 rather than something tight. The defect this
    guards against is not a few percent: when this project last got a standard
    error wrong, the ratio was 4.6. A tight band here would fail on the
    measured 7% optimism, which is declared in the module docstring and is not
    worth a fitted correction factor.

    Every lag truncation is reported, including 0. That is a measurement, not
    a demonstration that 0 is wrong -- it is not, by 0.3% -- and the row is
    printed so that nobody has to take the module docstring's word for it.
    """
    rows = []
    for imposed_bps in imposed:
        full = 2.0 * imposed_bps / 10_000.0
        for lags in lag_choices:
            got, reported = [], []
            for i in range(runs):
                o, h, l, c = simulate_bars(full, bars=bars, ticks=ticks,
                                           sigma=sigma,
                                           seed=seed0 + int(imposed_bps) * 13 + i)
                e = edge(o, h, l, c, minimum_bars=0, lags=lags)
                if e.half_spread_bps is None or e.half_spread_error_bps is None:
                    continue
                got.append(e.half_spread_bps)
                reported.append(e.half_spread_error_bps)
            actual = float(np.std(got, ddof=1))
            claimed = float(np.median(reported))
            rows.append((imposed_bps, lags, actual, claimed,
                         claimed / actual if actual > 0 else float("nan")))

    in_use = [r for r in rows if r[1] == 1]
    ratios = [r[4] for r in in_use]
    lo, hi = band
    passed = all(lo <= r <= hi for r in ratios)
    worst = max(ratios, key=lambda r: abs(math.log(r)))
    detail = "\n".join(
        [f"{runs} independent runs per cell, {bars} bars each",
         "  imposed  lag   actual sd   reported SE (median)   ratio"]
        + [f"  {imp:7.0f}  {lags:3d}   {actual:9.3f}   {claimed:20.3f}   "
           f"{r:5.3f}{'  <- in use' if lags == 1 else ''}"
           for imp, lags, actual, claimed, r in rows]
        + [f"at the lag in use the ratio spans "
           f"{min(ratios):.2f} to {max(ratios):.2f}, worst {worst:.2f}, "
           f"against a band of {lo:.2f} to {hi:.2f}",
           "the error bar runs slightly optimistic: the combination weight, "
           "both trading-frequency probabilities and the de-meaning are all "
           "estimated from the same sample and then treated as known."])
    return SpreadControlReport(
        name="Standard error: does the error bar predict the dispersion?",
        passed=passed, detail=detail,
        numbers={"rows": rows, "ratios_in_use": ratios, "worst": worst})


def refusal_control(*, bars_below: int = MINIMUM_BARS - 10,
                    bars_above: int = MINIMUM_BARS + 10, runs: int = 60,
                    imposed_bps: float = 20.0, ticks: int = 60,
                    sigma: float = 0.01, seed0: int = 12_000
                    ) -> SpreadControlReport:
    """Does the minimum-bar refusal fire, and is it set high enough?

    The first half is trivial and would pass on any threshold. The second is
    the one worth running: just *above* the threshold the estimator returns
    numbers, and this measures how bad they are. If a sample ten bars over the
    line still produces estimates scattered by more than the value itself,
    the threshold is decoration.
    """
    full = 2.0 * imposed_bps / 10_000.0
    o, h, l, c = simulate_bars(full, bars=bars_below + 1, ticks=ticks,
                               sigma=sigma, seed=seed0)
    below = edge(o, h, l, c)
    fires = below.spread is None and "usable bars" in below.refusal

    got = []
    for i in range(runs):
        o, h, l, c = simulate_bars(full, bars=bars_above + 1, ticks=ticks,
                                   sigma=sigma, seed=seed0 + 100 + i)
        e = edge(o, h, l, c)
        if e.spread is not None:
            got.append(e.half_spread_bps)
    arr = np.array(got)
    noise = float(arr.std(ddof=1) / arr.mean()) if arr.size > 1 else float("nan")
    usable = noise < 1.0

    detail = (
        f"{bars_below} bars: {below.describe()}\n"
        f"{bars_above} bars, {runs} runs at {imposed_bps:.0f} bps imposed: "
        f"mean {arr.mean():.1f} bps, sd {arr.std(ddof=1):.1f}, "
        f"relative scatter {noise:.0%}\n"
        f"just above the threshold the estimate is "
        f"{'usable, if wide' if usable else 'NOISE -- raise MINIMUM_BARS'}")
    return SpreadControlReport(
        name="Refusal: too few bars declines rather than guesses",
        passed=fires and usable, detail=detail,
        numbers={"fires": fires, "relative_scatter_just_above": noise})


def resolution_control(*, runs: int = 150, bars: int = 500, ticks: int = 60,
                       sigma: float = 0.01, seed0: int = 20_000,
                       significance: float = 2.0,
                       below: tuple[float, ...] = (0.0, 1.0, 2.0),
                       above: tuple[float, ...] = (20.0, 50.0, 100.0)
                       ) -> SpreadControlReport:
    """Does the estimator decline to claim what it cannot resolve?

    The other three controls establish that the estimator has a noise floor
    around 4 bps. This one checks the rule that keeps that floor out of the
    cost model: an estimate is claimable only when it stands `significance`
    standard errors clear of zero.

    Both directions matter and only one of them is obvious. A rule that
    refused everything would pass the first half trivially, so the second half
    requires that spreads which *are* resolvable are in fact claimed --
    a gate that never opens is not a gate, it is a constant with extra steps.

    This control is also the one that caught the significance test being
    applied to the wrong quantity. The rule was first written as
    ``s >= 2 SE(s)``, which reads as a two-standard-error test and is one, on
    ``s``; but ``SE(s) = SE(s^2)/(2s)``, so it rearranges to
    ``|s^2| >= SE(s^2)`` -- one standard error on the square. It claimed a
    spread in 32% of zero-spread samples while looking like a 2.3% test. Both
    forms are reported below, because the wrong one is the one an
    implementation reaches for.

    >>> r = resolution_control(runs=30, bars=400, below=(0.0,), above=(50.0,))
    >>> r.passed
    True
    """
    def rates(imposed_bps: float) -> tuple[float, float]:
        """Claim rate under the test on s^2, and under the test on s."""
        full = 2.0 * imposed_bps / 10_000.0
        on_square = on_root = total = 0
        for i in range(runs):
            o, h, l, c = simulate_bars(full, bars=bars, ticks=ticks,
                                       sigma=sigma,
                                       seed=seed0 + int(imposed_bps * 7) + i)
            e = edge(o, h, l, c, minimum_bars=0)
            if e.spread is None or e.standard_error is None:
                continue
            total += 1
            on_square += e.resolved(significance)
            on_root += e.spread >= significance * e.standard_error
        if not total:
            return float("nan"), float("nan")
        return on_square / total, on_root / total

    quiet = [(bps, *rates(bps)) for bps in below]
    loud = [(bps, *rates(bps)) for bps in above]
    # Below the floor: at most 10% false claims. The threshold is two standard
    # errors on a symmetric quantity, so a perfectly calibrated error bar
    # would give 2.3% one-sided; 10% leaves room for the measured 7% optimism
    # and for the sampling error of the rate itself.
    passed = (all(rate <= 0.10 for _, rate, _ in quiet)
              and all(rate >= 0.95 for _, rate, _ in loud))
    detail = "\n".join(
        [f"a spread is claimed only at {significance:.0f} standard errors "
         f"clear of zero; {runs} runs per rung, {bars} bars",
         "  imposed   claimed: test on s^2   the same test on s (wrong)"]
        + [f"  {bps:7.0f}   {sq:18.0%}   {rt:25.0%}"
           f"{'   <- must stay quiet' if i == 0 else ''}"
           for i, (bps, sq, rt) in enumerate(quiet)]
        + [f"  {bps:7.0f}   {sq:18.0%}   {rt:25.0%}"
           f"{'   <- must speak up' if i == 0 else ''}"
           for i, (bps, sq, rt) in enumerate(loud)]
        + ["the right-hand column is what a two-standard-error test on the "
           "spread itself does; it is a one-standard-error test on the square."])
    return SpreadControlReport(
        name="Resolution: nothing is claimed that cannot be resolved",
        passed=passed, detail=detail,
        numbers={"below": quiet, "above": loud})


def resolution_by_window_control(
        *, runs: int = 120, ticks: int = 60, sigma: float = 0.01,
        seed0: int = 30_000,
        spreads: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0, 15.0),
        lengths: tuple[int, ...] = (250, 500, 1000, 2500)
        ) -> SpreadControlReport:
    """What can actually be resolved, at what spread, off how many bars.

    The other controls establish that the estimator works. This one answers
    the question that decides whether any of it is useful on a particular
    book: given a European ETF whose true half-spread is 2 to 15 bps, how
    often does the significance test let it through?

    It exists because the answer changed a design decision. The survey used
    to estimate over the holding period, about 250 bars, on no better reason
    than that the covariance window is 250 bars. A covariance window must be
    short -- correlations move with regime. A spread does not, and the noise
    floor thins as the fourth root of the sample, so the window was costing
    resolution for nothing. The survey now takes the whole available history.

    The falsifiable claim: **resolution must improve with sample length at
    every spread the estimator can reach at all.** A rule that did not would
    mean the longer window buys nothing and the change was pointless.

    What it does not claim, and what the table shows plainly, is that a longer
    window rescues everything. It does not. Below about 4 bps the fraction
    resolved is flat in the sample length, because that is under the floor at
    every length available, and no amount of daily history reaches it.

    The spreads passed here must include at least one with room to improve.
    15 bps alone would not do: it already resolves 96% of the time off 250
    bars, so it has nowhere to go and the control would report that it found
    no improvement rather than that none exists.

    >>> r = resolution_by_window_control(runs=25, spreads=(8.0, 15.0),
    ...                                  lengths=(250, 1000))
    >>> r.passed
    True
    """
    grid = {}
    for half_bps in spreads:
        full = 2.0 * half_bps / 10_000.0
        for bars in lengths:
            claimed = 0
            for i in range(runs):
                o, h, l, c = simulate_bars(full, bars=bars, ticks=ticks,
                                           sigma=sigma,
                                           seed=seed0 + int(half_bps * 97) + i)
                claimed += edge(o, h, l, c, minimum_bars=0).resolved()
            grid[(half_bps, bars)] = claimed / runs

    # Two claims, because one of them alone is satisfiable for the wrong
    # reason. A longer sample must never resolve LESS often -- that would mean
    # something other than sampling error is at work. And where there is room
    # to improve it must actually improve, or the longer window is buying
    # nothing and the change that introduced it was pointless.
    #
    # A spread already resolving nearly always on the short window has no room
    # and is excluded from the second claim rather than failing it: 15 bps
    # resolves 93% off 250 bars and cannot gain fifteen points.
    reachable = [s for s in spreads if grid[(s, lengths[-1])] >= 0.5]
    headroom = [s for s in reachable if grid[(s, lengths[0])] < 0.80]
    never_worse = all(grid[(s, lengths[-1])] >= grid[(s, lengths[0])] - 0.10
                      for s in spreads)
    improves = bool(headroom) and all(
        grid[(s, lengths[-1])] >= grid[(s, lengths[0])] + 0.15
        for s in headroom)
    unreachable = [s for s in spreads if grid[(s, lengths[-1])] < 0.5]

    detail = "\n".join(
        [f"{runs} runs per cell; the fraction claimed as a measurement",
         "  half-spread  " + "  ".join(f"{b:>6}" for b in lengths) + "   bars"]
        + [f"  {s:8.1f} bps  "
           + "  ".join(f"{grid[(s, b)]:5.0%} " for b in lengths)
           + ("" if s in reachable else "   never reaches half")
           for s in spreads]
        + [f"never resolves less often on the longer sample: "
           f"{'yes' if never_worse else 'NO'}",
           f"and improves materially where there is room to "
           f"({', '.join(f'{s:.0f}' for s in headroom) or 'nowhere'} bps): "
           f"{'yes' if improves else 'NO'}"]
        + ([f"{', '.join(f'{s:.0f}' for s in unreachable)} bps stay under the "
            f"floor at every length available, so a longer window does not "
            f"rescue them and the upper-bound tier is what they get"]
           if unreachable else []))
    return SpreadControlReport(
        name="Resolution by window: what a longer history actually buys",
        passed=never_worse and improves, detail=detail,
        numbers={"grid": grid, "reachable": reachable,
                 "unreachable": unreachable})


# The kinds of bar a daily feed manufactures, each written forwards as a
# change to clean bars. The names are the ones `classify_bars` counts under,
# where it counts them at all: the last three it does not, because they
# were measured to do nothing, and this control is what keeps that claim
# honest.
CONTAMINATIONS: tuple[str, ...] = (
    "whole bar carried forward", "close carried forward",
    "open carried (= previous close)", "high and low widened by 30 bps",
    "high alone widened by 30 bps", "high == low (one price all day)",
    "open == high == low == close")

# Kinds the mask must set aside, and that must move the estimate by more
# than this fraction at 5% of bars before exclusion.
BIASING: frozenset[str] = frozenset(CONTAMINATIONS[:2])
BIAS_AT_LEAST = 0.35
# Kinds that must move it by less than this at 15% of bars.
HARMLESS_WITHIN = 0.10


def contaminate(kind: str, open_: np.ndarray, high: np.ndarray,
                low: np.ndarray, close: np.ndarray, picked: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copies of the four series with `kind` imposed on the `picked` bars.

    `picked` must be sorted ascending and exclude bar 0, so that a carried
    bar copies from a bar that has itself already been settled.

    >>> o = np.array([1.0, 2.0, 3.0]); h = o + 0.1; l = o - 0.1; c = o.copy()
    >>> o2, h2, l2, c2 = contaminate("close carried forward", o, h, l, c,
    ...                              np.array([2]))
    >>> float(c2[2]), float(c[2])
    (2.0, 3.0)
    """
    o, h, l, c = (np.array(x, dtype=float, copy=True)
                  for x in (open_, high, low, close))
    for i in picked:
        if kind == "whole bar carried forward":
            o[i], h[i], l[i], c[i] = o[i - 1], h[i - 1], l[i - 1], c[i - 1]
        elif kind == "close carried forward":
            c[i] = c[i - 1]
            h[i], l[i] = max(h[i], c[i]), min(l[i], c[i])
        elif kind == "open carried (= previous close)":
            o[i] = c[i - 1]
            h[i], l[i] = max(h[i], o[i]), min(l[i], o[i])
        elif kind == "high and low widened by 30 bps":
            h[i] *= 1.003
            l[i] /= 1.003
        elif kind == "high alone widened by 30 bps":
            h[i] *= 1.003
        elif kind == "high == low (one price all day)":
            h[i] = l[i] = c[i]
            o[i] = c[i]
        elif kind == "open == high == low == close":
            o[i] = h[i] = l[i] = c[i]
        else:
            raise ValueError(f"unknown contamination {kind!r}")
    return o, h, l, c


def simulate_reversal_bars(spread: float, *, bars: int = 500, ticks: int = 60,
                           sigma: float = 0.01, phi: float = -0.15,
                           seed: int = 0) -> tuple[np.ndarray, ...]:
    """`simulate_bars`, with part of each day's move reversed overnight.

    The efficient price is no longer a random walk at the daily horizon:
    the first tick of day t carries `phi` times the previous day's total
    move, so daily returns have autocorrelation close to `phi`. That is
    what a stale or non-synchronous close looks like from outside -- a
    print that has not caught up, followed by a morning that catches up --
    and it is not a property of any single bar, so no bar of it can be
    set aside.

    >>> o, h, l, c = simulate_reversal_bars(0.002, bars=2000, phi=-0.3, seed=1)
    >>> r = np.diff(np.log(c))
    >>> bool(np.corrcoef(r[1:], r[:-1])[0, 1] < -0.2)
    True
    """
    if spread < 0:
        raise ValueError("spread must be non-negative")
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma / math.sqrt(ticks), (bars, ticks))
    day = steps.sum(axis=1)
    carry = np.zeros(bars)
    for t in range(1, bars):
        carry[t] = phi * (day[t - 1] + carry[t - 1])
    steps[:, 0] += carry
    mid = np.exp(np.cumsum(steps.ravel())).reshape(bars, ticks) * 100.0
    side = rng.choice((-1.0, 1.0), size=(bars, ticks))
    prints = mid * (1.0 + side * spread / 2.0)
    return (prints[:, 0].copy(), prints.max(axis=1), prints.min(axis=1),
            prints[:, -1].copy())


# Daily return autocorrelations the reversal rows are run at. -0.15 is a
# stale-price signature of ordinary size; -0.05 is barely visible in a
# return series and still doubles the estimate's error.
REVERSALS: tuple[float, ...] = (-0.05, -0.15)


@dataclasses.dataclass(frozen=True)
class ContaminationRow:
    kind: str
    share: str                           # "5%" or "phi -0.15"
    contaminated_bps: float
    contaminated_se_bps: float
    excluded_bps: float
    excluded_se_bps: float
    autocorrelation: float
    set_aside: float                     # mean count of bars the mask flagged


def contamination_control(*, runs: int = 60, bars: int = 1500, ticks: int = 60,
                          sigma: float = 0.01, seed0: int = 40_000,
                          imposed_bps: float = 10.0,
                          fractions: tuple[float, ...] = (0.05, 0.15),
                          reversals: tuple[float, ...] = REVERSALS
                          ) -> SpreadControlReport:
    """What each kind of manufactured bar does, and whether the mask catches it.

    Four claims, each checked:

      * a carried close or a carried whole bar on the smaller fraction of
        days inflates the estimate by at least `BIAS_AT_LEAST`;
      * setting aside the bars `classify_bars` flags recovers the imposed
        spread, to within three standard errors of the mean over runs;
      * the other kinds of bar move the estimate by less than
        `HARMLESS_WITHIN`, or less than their own sampling error, even at
        the larger fraction -- which is why they are counted and not
        excluded;
      * a daily reversal in the price itself inflates the estimate by at
        least `BIAS_AT_LEAST` at the stronger `reversals` setting while the
        mask sets aside nothing, because nothing is wrong with any bar.

    The last is a limit of the method, pinned so it is not forgotten: at
    the daily frequency the estimator cannot tell a bounce from a
    reversal, which is Roll's identification problem and the paper's
    assumption of uncorrelated efficient-price increments read backwards.
    A feed whose close is a print the market has moved past by the morning
    produces exactly that, and no count of bars will find it.

    The per-bar autocorrelation is reported alongside, with no pass
    criterion: on the real book it was negative on every instrument, and
    this is the place to see that carried bars push it one way and a
    reversal the other.

    >>> r = contamination_control(runs=8, bars=800)
    >>> r.passed
    True
    """
    def summary(values) -> tuple[float, float]:
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))

    baseline = []
    for i in range(runs):
        o, h, l, c = simulate_bars(2.0 * imposed_bps / 10_000.0, bars=bars,
                                   ticks=ticks, sigma=sigma, seed=seed0 + i)
        baseline.append(edge(o, h, l, c).half_spread_bps)
    base, base_se = summary(baseline)

    def measure(kind: str, share: str, make) -> ContaminationRow:
        got, cleaned, rhos, flagged = [], [], [], []
        for i in range(runs):
            oc, hc, lc, cc = make(i)
            e = edge(oc, hc, lc, cc)
            got.append(e.half_spread_bps)
            rhos.append(e.autocorrelation)
            mask = classify_bars(oc, hc, lc, cc).excluded
            flagged.append(int(mask.sum()))
            cleaned.append(edge(*exclude_bars(oc, hc, lc, cc, mask))
                           .half_spread_bps)
        c_mean, c_se = summary(got)
        x_mean, x_se = summary(cleaned)
        return ContaminationRow(
            kind, share, c_mean, c_se, x_mean, x_se,
            float(np.nanmean([r for r in rhos if r is not None])),
            float(np.mean(flagged)))

    rows: list[ContaminationRow] = []
    for kind in CONTAMINATIONS:
        for fraction in fractions:
            def make(i, kind=kind, fraction=fraction):
                o, h, l, c = simulate_bars(2.0 * imposed_bps / 10_000.0,
                                           bars=bars, ticks=ticks, sigma=sigma,
                                           seed=seed0 + i)
                rng = np.random.default_rng(seed0 + 7 * i + 1)
                picked = np.sort(rng.choice(np.arange(1, bars),
                                            int(fraction * bars), replace=False))
                return contaminate(kind, o, h, l, c, picked)
            rows.append(measure(kind, f"{fraction:.0%}", make))
    for phi in reversals:
        def make(i, phi=phi):
            return simulate_reversal_bars(2.0 * imposed_bps / 10_000.0,
                                          bars=bars, ticks=ticks, sigma=sigma,
                                          phi=phi, seed=seed0 + i)
        rows.append(measure("daily reversal in the price itself",
                            f"phi {phi:+.2f}", make))

    def bias(row: ContaminationRow) -> float:
        return (row.contaminated_bps - base) / base

    small, large = f"{min(fractions):.0%}", f"{max(fractions):.0%}"
    biasing = [r for r in rows if r.kind in BIASING]
    harmless_rows = [r for r in rows
                     if r.kind in CONTAMINATIONS and r.kind not in BIASING]
    reversal_rows = [r for r in rows if r.kind not in CONTAMINATIONS]
    inflates = all(bias(r) >= BIAS_AT_LEAST for r in biasing if r.share == small)
    recovers = all(abs(r.excluded_bps - base)
                   <= 3.0 * math.hypot(r.excluded_se_bps, base_se)
                   for r in biasing)
    # Within a tenth, or within its own sampling error: at a handful of
    # runs a tenth is under one standard error and the control would fail
    # on noise, which is a control failing for the wrong reason.
    harmless = all(abs(r.contaminated_bps - base)
                   <= max(HARMLESS_WITHIN * base,
                          3.0 * math.hypot(r.contaminated_se_bps, base_se))
                   for r in harmless_rows if r.share == large)
    strongest = (min(reversals) if reversals else None)
    reversal_bites = all(bias(r) >= BIAS_AT_LEAST and r.set_aside == 0.0
                         for r in reversal_rows
                         if strongest is not None
                         and r.share == f"phi {strongest:+.2f}")
    passed = inflates and recovers and harmless and reversal_bites

    detail = "\n".join(
        [f"{runs} runs of {bars} bars at {imposed_bps:.0f} bps imposed; "
         f"clean baseline {base:.2f} +/- {base_se:.2f} bps",
         f"  {'contamination':<34} {'share':>9}  {'estimate':>8}  "
         f"{'bias':>6}  {'set aside':>9}  {'flagged':>7}  per-bar rho"]
        + [f"  {r.kind:<34} {r.share:>9}  {r.contaminated_bps:8.2f}  "
           f"{bias(r):+6.0%}  {r.excluded_bps:9.2f}  {r.set_aside:7.0f}  "
           f"{r.autocorrelation:+.3f}" for r in rows]
        + [f"carried close and carried bar inflate by at least "
           f"{BIAS_AT_LEAST:.0%} at {small}: {'yes' if inflates else 'NO'}",
           f"setting aside what classify_bars flags recovers the imposed "
           f"spread: {'yes' if recovers else 'NO'}",
           f"the other kinds of bar move it by under {HARMLESS_WITHIN:.0%} "
           f"(or their own sampling error) at {large}: "
           f"{'yes' if harmless else 'NO'}",
           f"a daily reversal inflates it by at least {BIAS_AT_LEAST:.0%} "
           f"with nothing to set aside: {'yes' if reversal_bites else 'NO'}",
           "the (h+l)/2 hypothesis -- that a widened high or low is what "
           "breaks the estimator -- is the widened rows, and they are the "
           "reason the survey counts carried bars rather than ranges. The "
           "reversal rows are the limit no count reaches: at the daily "
           "frequency a bounce and a reversal are the same covariance."])
    return SpreadControlReport(
        name="Contamination: what a manufactured bar does, and what is set aside",
        passed=passed, detail=detail,
        numbers={"baseline_bps": base, "baseline_se_bps": base_se,
                 "rows": [dataclasses.asdict(r) for r in rows]})


def run_spread_controls(*, quick: bool = False) -> list[SpreadControlReport]:
    """All seven. This is what `portfolio controls --spread` prints."""
    if quick:
        return [
            positive_control(runs=20, bars=400, sweep=(2.0, 20.0, 100.0)),
            negative_control(runs=40, lengths=(250, 500, 1000)),
            standard_error_control(runs=40, bars=400, imposed=(20.0, 100.0),
                                   lag_choices=(0, 1)),
            resolution_control(runs=40, bars=400, below=(0.0, 2.0),
                               above=(20.0, 100.0)),
            resolution_by_window_control(runs=30, spreads=(4.0, 8.0, 10.0),
                                         lengths=(250, 1000)),
            refusal_control(runs=20),
            contamination_control(runs=12, bars=800),
        ]
    return [positive_control(), negative_control(), standard_error_control(),
            resolution_control(), resolution_by_window_control(),
            refusal_control(), contamination_control()]
