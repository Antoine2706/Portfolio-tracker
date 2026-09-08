"""The effective bid-ask spread, estimated from open, high, low and close.

Why this file exists
--------------------
The bid-ask spread is the largest single component of this account's trading
cost and the only one that appears on no document. Commission is printed on
the confirmation. The transaction tax is printed on the confirmation. The
spread is paid *inside* the execution price: the confirmation says the trade
went off at 14.7135 and says nothing about where the mid was, so there is no
arithmetic that recovers it from anything the broker sent.

Until now the model carried a flat 8 bps for every instrument, declared as an
estimate. That is honest but it is also the largest input to the cost figure
and it is the same number for a 12 billion euro core equity tracker and a
thinly traded thematic fund, which cannot both be right. Worse, it makes the
cost model unable to distinguish between instruments at all on the dimension
where they most differ.

What is estimated, and what it is not
------------------------------------
This estimates the **effective** spread -- twice the average distance between
the transaction price and the efficient price -- not the **quoted** spread.
They differ: a trade can execute inside the quotes, and a large order can
execute outside them. The effective spread is the one a cost model wants,
because it is what was actually paid.

It is estimated from daily bars, so it is an average over the day's trading
weighted the way that day's trades happened to be weighted. It is not the
spread facing a particular order at a particular moment, and nothing here
should be read as one.

The estimator
-------------
This implements the efficient estimator of Ardia, Guidotti and Kroencke,
"Efficient estimation of bid-ask spreads from open, high, low, and close
prices", Journal of Financial Economics 161 (2024) 103916, known as EDGE.
Written out here rather than imported, so that the mathematics is readable in
the same place as everything else in this project.

Let the efficient (unobserved) log price follow a process with uncorrelated
increments, and let an observed transaction log price be the efficient price
plus a bounce term ``+-s/2``, where ``s`` is the effective spread as a
fraction of price and the sign is the side the trade initiator took.

Within a bar write ``o``, ``h``, ``l``, ``c`` for the log open, high, low and
close, and

    eta = (h + l) / 2

for the **log mid-range**. The mid-range is the load-bearing quantity. If the
bar contains at least one buy and at least one sell, the high sits at the ask
and the low at the bid, so their average cancels the bounce and ``eta`` is a
clean read on the efficient price. Every single transaction price -- open,
close, high, low -- carries the bounce; their midpoint does not.

Now take five returns across a consecutive pair of bars ``t-1``, ``t``:

    r1 = eta_t   - o_t         contains  -(s/2) q_o
    r2 = o_t     - eta_{t-1}   contains  +(s/2) q_o
    r3 = eta_t   - c_{t-1}     contains  -(s/2) q_c
    r4 = c_{t-1} - eta_{t-1}   contains  +(s/2) q_c
    r5 = o_t     - c_{t-1}     contains  +(s/2) q_o - (s/2) q_c

where ``q_o`` and ``q_c`` are the +-1 signs of the open and the previous
close. The efficient-price increments in these five are increments over
disjoint intervals, so they are uncorrelated with each other and with the
signs. That leaves the bounce terms, and every one of these four products
isolates one squared bounce:

    E[r1 r2] = -(s^2/4) E[q_o^2] = -s^2/4
    E[r3 r4] = -s^2/4
    E[r1 r5] = -s^2/4        (the q_c term in r5 is independent of q_o)
    E[r5 r4] = -s^2/4

which is Roll's covariance argument, except that it is run against the
mid-range rather than against another transaction price, and so it does not
need the successive-trade-sign independence that makes Roll fail on real data.

Two estimators, and why there are two
-------------------------------------
Pair the products up two different ways:

    x1 = -(4/p_o) d1 r2  -  (4/p_c) d3 r4
    x2 = -(4/p_o) d1 r5  -  (4/p_c) d5 r4

Each has expectation ``s^2`` (two contributions of ``s^2/2`` under continuous
trading, where ``p_o = p_c = 2``). They use the same bars but combine them
differently, so they are two distinct unbiased estimators, and a variance-
weighted average of the two beats either. That combination is the "efficient"
in the estimator's name.

``d1``, ``d3``, ``d5`` are the de-meaned versions of ``r1``, ``r3``, ``r5``.
De-meaning one factor of each product turns ``E[XY]`` into ``Cov(X, Y)``,
which is what the argument above is about; without it a drifting price
contributes ``E[X]E[Y]`` and the spread comes out wrong in a direction that
depends on the window's drift.

The infrequent-trading correction
---------------------------------
``p_o`` and ``p_c`` are what this estimator adds over its predecessors, and
they are the reason the paper's title says *efficient* and its abstract says
its predecessors are *downward biased when trading is infrequent*.

The cancellation above assumed the high is at the ask and the low is at the
bid. When a bar has few trades that fails: the open may itself *be* the high,
in which case ``eta`` is contaminated by the same bounce sign as ``o`` and the
product ``r1 r2`` no longer isolates ``s^2/4``. So

    p_o = P(o != h) + P(o != l)
    p_c = P(c_{-1} != h_{-1}) + P(c_{-1} != l_{-1})

are estimated from the bars themselves, and each product is divided by the
probability that it was informative. Under continuous trading both are 2 and
the correction does nothing. On a fund that trades ten times a day it is the
difference between a spread estimate and a downward-biased one.

``tau`` marks the bars that moved at all: ``h != l`` or ``l != c_{-1}``. A bar
whose entire range is one price equal to the previous close contains no
information about the spread and enters as an exact zero, which would drag the
mean down. It is excluded, and the count of surviving bars is reported,
because an estimate off nine usable bars and an estimate off five hundred are
not the same claim.

What this file adds beyond the paper
------------------------------------
A standard error, and a refusal.

The paper's estimator returns a number. A number with no error bar is not
something this project is willing to print -- the last time it did, a Sharpe
ratio and its standard error were computed at different frequencies and every
t-statistic came out 4.6 times too small. So ``s^2`` is reported as the mean
of a per-bar series and the standard error of that mean is reported with it.

That error is the ordinary i.i.d. one, and it took a measurement to settle
that it should be. Consecutive terms of the per-bar series share a bar --
``x_t`` and ``x_{t+1}`` both use bar ``t`` -- so they ought to correlate, and
the first version of this file used a Newey-West correction on that argument.
The argument is wrong. Measured over 200 samples at each of three spreads, the
series' autocorrelation is ``-0.003 +/- 0.002`` at lag 1 and no larger at
lags 2 to 4: not distinguishable from zero. Each product pairs a bounce at one
bar edge against the same bounce at the same edge, and neighbouring terms use
different edges, so the shared bar carries no shared bounce.

Switching the correction on therefore absorbs nothing and adds the estimation
noise of the autocovariances themselves -- up to 15% on a single sample at
lag 8. It is left in as `lags`, defaulting to 0, because real bars have
volatility clustering the simulation does not, and `SpreadEstimate` reports
the autocorrelation it measured so that the choice can be checked against real
data rather than assumed to carry over. What it must not be is switched on by
an argument that sounds right and was never tested.

The delta method carries the error from ``s^2`` to ``s``:

    SE(s) = SE(s^2) / (2 s)

which diverges as the estimate approaches zero. That is not a defect of the
formula, it is the truth about the estimate: near zero the sign of ``s^2`` is
not established and the root is not identified.

**Significance is tested on the square, never on the root.** Rearranging the
delta method, ``s >= k SE(s)`` is ``|s^2| >= (k/2) SE(s^2)``, so a test that
reads as two standard errors on the spread is one standard error on the
quantity that actually has a symmetric sampling distribution. Measured at
zero true spread, the two-standard-error form of that rule claimed a spread
in 32% of samples. `SpreadEstimate.t_statistic` and `.resolved()` do it on
the square, where the same rule claims one in 3%.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

__all__ = ["SpreadEstimate", "edge", "MINIMUM_BARS", "TickSize",
           "infer_tick_size", "CANDIDATE_TICKS"]


# Below this many usable bars the estimator returns a refusal rather than a
# number. Not a statistical threshold -- the standard error is the statistical
# statement -- but a floor under which the error bar is itself unreliable,
# since it is estimated from the same few terms. Sixty bars is roughly a
# quarter of trading days.
MINIMUM_BARS = 60


@dataclasses.dataclass(frozen=True)
class SpreadEstimate:
    """An effective-spread estimate, or a statement of why there isn't one.

    `spread` is the full effective spread as a fraction of price: 0.001 is
    10 bps. `half_spread_bps` is what the cost model charges on one side.

    `signed_square` is the estimate of ``s^2`` *before* the absolute value and
    the square root. It matters: ``s^2`` is a mean of a noisy series and can
    come out negative, and reporting only ``sqrt(|s^2|)`` turns a negative
    estimate into a positive spread. A book of instruments whose true spreads
    are all near zero would then show a floor of pure sampling noise that
    reads exactly like a measurement. The negative control in
    `eval/spread_controls.py` exists to measure that floor.
    """
    spread: float | None                 # effective spread, fraction of price
    signed_square: float | None          # the estimate of s^2, sign kept
    standard_error: float | None         # SE of `spread`, delta method
    bars: int                            # consecutive pairs supplied
    usable_bars: int                     # pairs that survived tau and NaN
    square_standard_error: float | None = None   # SE of `signed_square`
    # Lag-1 autocorrelation of the per-bar series the estimate is the mean of.
    # Reported rather than assumed: the standard error treats the series as
    # i.i.d., and this is the number that says whether it may. On simulated
    # bars it sits at -0.003; a real instrument reading materially away from
    # zero is a reason to raise `lags`, and a reason to say so out loud.
    autocorrelation: float | None = None
    refusal: str = ""                    # non-empty means no estimate

    @property
    def half_spread_bps(self) -> float | None:
        """One side, in basis points -- the unit the cost model charges in."""
        if self.spread is None:
            return None
        return float(self.spread * 10_000.0 / 2.0)

    @property
    def half_spread_error_bps(self) -> float | None:
        if self.standard_error is None:
            return None
        return float(self.standard_error * 10_000.0 / 2.0)

    @property
    def t_statistic(self) -> float | None:
        """How many standard errors the estimate stands clear of zero.

        Computed on ``s^2``, not on ``s``, and the difference is not cosmetic.
        The delta method gives ``SE(s) = SE(s^2) / (2s)``, so the apparently
        natural test ``s >= 2 SE(s)`` rearranges to ``|s^2| >= SE(s^2)``:
        asking for two standard errors on the root silently asks for *one* on
        the square. Measured at zero true spread, that rule claimed a spread
        in 32% of samples while looking like a 2.3% test.

        The square is also the quantity with the symmetric sampling
        distribution. ``s`` is its absolute value's root, which cannot go
        negative and so cannot be centred on zero even when the truth is.

        >>> SpreadEstimate(0.002, 4e-6, 0.0005, 500, 499,
        ...                square_standard_error=1e-6).t_statistic
        4.0
        """
        if self.signed_square is None or not self.square_standard_error:
            return None
        return float(self.signed_square / self.square_standard_error)

    def resolved(self, significance: float = 2.0) -> bool:
        """Is this estimate distinguishable from no spread at all?

        An unresolved estimate is not a small spread. It is the estimator's
        own noise, which stands at roughly 4 bps off two years of daily bars
        and thins only as the fourth root of the sample. Treating it as a
        measurement would put a number in the cost model that says "this
        instrument trades at 4 bps" about an instrument the data cannot speak
        to at all.
        """
        t = self.t_statistic
        return t is not None and t >= significance

    def describe(self) -> str:
        """One line, with the error bar and the sample it rests on.

        >>> SpreadEstimate(None, None, None, 12, 0,
        ...                refusal="only 12 bars").describe()
        'no estimate: only 12 bars'
        """
        if self.refusal:
            return f"no estimate: {self.refusal}"
        assert self.half_spread_bps is not None
        line = f"{self.half_spread_bps:.1f} bps half-spread"
        if self.half_spread_error_bps is not None:
            line += f" +/- {self.half_spread_error_bps:.1f}"
        return line + f", off {self.usable_bars} usable bars"


def _newey_west(x: np.ndarray, lags: int) -> float:
    """Long-run variance of a series whose terms overlap.

    The Bartlett-weighted sum

        gamma_0 + 2 sum_{j=1..L} (1 - j/(L+1)) gamma_j

    with `gamma_j` the sample autocovariance at lag j. The weights taper to
    zero so the estimate is guaranteed non-negative, which a raw truncated sum
    is not.

    Lag 0 gives the ordinary variance:

    >>> a = np.array([1.0, 2.0, 3.0, 4.0])
    >>> round(_newey_west(a, 0), 6) == round(float(np.var(a)), 6)
    True

    A perfectly persistent series has more long-run variance than short-run:

    >>> b = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    >>> _newey_west(b, 2) > _newey_west(b, 0)
    True
    """
    n = x.size
    if n == 0:
        return float("nan")
    centred = x - x.mean()
    total = float(np.mean(centred * centred))
    for j in range(1, min(lags, n - 1) + 1):
        gamma = float(np.mean(centred[j:] * centred[:-j])) * (n - j) / n
        total += 2.0 * (1.0 - j / (lags + 1.0)) * gamma
    # The taper makes this non-negative in exact arithmetic; floating point at
    # tiny magnitudes can still cross zero, and a negative variance must not
    # reach a square root.
    return max(total, 0.0)


# A missing bar is supported input here, not an anomaly: a venue holiday on
# one exchange and not another is the normal shape of this project's panels.
# NaN arithmetic warns by default, so a gap would print a stack of
# RuntimeWarnings through every caller. Silenced for the whole function rather
# than at each line, because every line of it is NaN-aware by design.
@np.errstate(divide="ignore", invalid="ignore")
def edge(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
         close: np.ndarray, *, minimum_bars: int = MINIMUM_BARS,
         lags: int = 0) -> SpreadEstimate:
    """Estimate the effective spread from a series of OHLC bars.

    Prices, not log prices; in any currency, since the estimate is a fraction.
    Bars must be in ascending time order. A missing bar may be passed as NaN.

    A worked case where the answer is known in advance. Put the high and low
    a fixed distance either side of a drifting efficient price, so the
    mid-range is that price exactly, and displace the open and the close from
    it by independent noise of standard deviation ``sigma``. That noise *is* a
    bounce: the argument above needs only ``E[q^2]``, not that ``q`` is
    binary, so a displacement of standard deviation ``sigma`` is a half-spread
    of ``sigma``. At 10 bps imposed:

    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    >>> hi, lo = mid * 1.005, mid * 0.995
    >>> op = mid * (1 + rng.normal(0, 0.0010, n))
    >>> cl = mid * (1 + rng.normal(0, 0.0010, n))
    >>> print(edge(op, hi, lo, cl).describe())
    9.9 bps half-spread +/- 1.6, off 499 usable bars

    The error bar is not decoration: a fifth of the imposed value, off two
    years of daily bars. `eval/spread_controls.py` checks that it is the right
    size rather than merely present.

    Too few bars is a refusal, not a number:

    >>> print(edge(op[:20], hi[:20], lo[:20], cl[:20]).describe())
    no estimate: only 19 usable bars, fewer than the 60 required

    A price that never moved carries no bounce, and says so rather than
    returning a confident zero:

    >>> edge(*[np.full(200, 7.5)] * 4).refusal
    'no bar pair moved; the price never changed'

    Mismatched lengths are a programming error and say so:

    >>> edge(np.ones(5), np.ones(5), np.ones(5), np.ones(4))
    Traceback (most recent call last):
        ...
    ValueError: open, high, low and close must be the same length
    """
    o_raw = np.asarray(open_, dtype=float)
    h_raw = np.asarray(high, dtype=float)
    l_raw = np.asarray(low, dtype=float)
    c_raw = np.asarray(close, dtype=float)
    n = o_raw.size
    if h_raw.size != n or l_raw.size != n or c_raw.size != n:
        raise ValueError("open, high, low and close must be the same length")

    pairs = max(n - 1, 0)
    if n < 3:
        return SpreadEstimate(None, None, None, pairs, 0,
                              refusal=f"only {pairs} bar pairs, at least 2 are needed")

    # Log prices. A non-positive price is not a price; it becomes NaN here
    # rather than -inf, so that it is excluded rather than poisoning a mean.
    o_all = np.where(o_raw > 0, np.log(np.where(o_raw > 0, o_raw, 1.0)), np.nan)
    h_all = np.where(h_raw > 0, np.log(np.where(h_raw > 0, h_raw, 1.0)), np.nan)
    l_all = np.where(l_raw > 0, np.log(np.where(l_raw > 0, l_raw, 1.0)), np.nan)
    c_all = np.where(c_raw > 0, np.log(np.where(c_raw > 0, c_raw, 1.0)), np.nan)
    m_all = (h_all + l_all) / 2.0

    # Split into the current bar and the one before it.
    h1, l1, c1, m1 = h_all[:-1], l_all[:-1], c_all[:-1], m_all[:-1]
    o, h, l, c, m = (o_all[1:], h_all[1:], l_all[1:], c_all[1:], m_all[1:])

    # The five returns of the docstring.
    r1 = m - o
    r2 = o - m1
    r3 = m - c1
    r4 = c1 - m1
    r5 = o - c1

    # tau: did this bar carry any price information at all?
    tau = np.where(np.isnan(h) | np.isnan(l) | np.isnan(c1), np.nan,
                   ((h != l) | (l != c1)).astype(float))
    # The four indicators behind p_o and p_c. Each is conditioned on tau, so a
    # bar that did not move contributes to neither the numerator nor the
    # denominator of the probability.
    po1 = tau * np.where(np.isnan(o) | np.isnan(h), np.nan, (o != h).astype(float))
    po2 = tau * np.where(np.isnan(o) | np.isnan(l), np.nan, (o != l).astype(float))
    pc1 = tau * np.where(np.isnan(c1) | np.isnan(h1), np.nan, (c1 != h1).astype(float))
    pc2 = tau * np.where(np.isnan(c1) | np.isnan(l1), np.nan, (c1 != l1).astype(float))

    if not np.any(np.isfinite(tau)):
        return SpreadEstimate(None, None, None, pairs, 0,
                              refusal="no bar pair had prices on both sides")

    p_tau = float(np.nanmean(tau))
    p_o = float(np.nanmean(po1) + np.nanmean(po2))
    p_c = float(np.nanmean(pc1) + np.nanmean(pc2))
    active = int(np.nansum(tau))

    if p_tau <= 0 or active < 2:
        return SpreadEstimate(None, None, None, pairs, active,
                              refusal="no bar pair moved; the price never changed")
    if p_o <= 0 or p_c <= 0:
        # Every open was simultaneously the high and the low, or every close
        # was. One price a day is not a bar and carries no bounce to measure.
        return SpreadEstimate(None, None, None, pairs, active,
                              refusal="every open (or every close) was the "
                                      "whole range; there is no bounce to "
                                      "measure")

    # De-mean the leading factor of each product, putting the whole drift onto
    # the bars that moved.
    d1 = r1 - np.nanmean(r1) / p_tau * tau
    d3 = r3 - np.nanmean(r3) / p_tau * tau
    d5 = r5 - np.nanmean(r5) / p_tau * tau

    x1 = -4.0 / p_o * d1 * r2 + -4.0 / p_c * d3 * r4
    x2 = -4.0 / p_o * d1 * r5 + -4.0 / p_c * d5 * r4

    # One mask for both series, so the point estimate and the standard error
    # are computed off exactly the same rows. The two series depend on the
    # same five inputs, so this discards nothing either of them could use.
    usable = np.isfinite(x1) & np.isfinite(x2)
    k = int(usable.sum())
    if k < minimum_bars:
        return SpreadEstimate(None, None, None, pairs, k,
                              refusal=f"only {k} usable bars, fewer than the "
                                      f"{minimum_bars} required")
    a = x1[usable]
    b = x2[usable]

    e1, e2 = float(a.mean()), float(b.mean())
    v1 = float(a.var())
    v2 = float(b.var())
    vt = v1 + v2

    # The variance-weighted combination: the lower-variance estimator gets the
    # larger weight. With no variance to go on, weight them equally.
    weight = v2 / vt if vt > 0 else 0.5
    s2 = weight * e1 + (1.0 - weight) * e2

    # The combined per-bar series, whose mean IS s2. Its long-run variance is
    # the honest input to a standard error: consecutive terms share bar t, so
    # they are correlated at lag 1 by construction.
    combined = weight * a + (1.0 - weight) * b
    se_s2 = math.sqrt(_newey_west(combined, lags) / k)

    # Reported so the i.i.d. assumption above is checkable on real data
    # rather than carried over from the simulation that justified it.
    centred = combined - combined.mean()
    denominator = float(np.mean(centred * centred))
    rho = (float(np.mean(centred[1:] * centred[:-1])) / denominator
           if denominator > 0 else None)

    spread = math.sqrt(abs(s2))
    # Delta method. sqrt is not differentiable at zero and the error bar is
    # correspondingly undefined there; report no error rather than infinity.
    se = se_s2 / (2.0 * spread) if spread > 0 else None

    return SpreadEstimate(spread=spread, signed_square=float(s2),
                          standard_error=se, bars=pairs, usable_bars=k,
                          square_standard_error=se_s2, autocorrelation=rho)


# --------------------------------------------------------------------------
# The tick, measured rather than looked up
# --------------------------------------------------------------------------

# The ticks a European venue actually quotes in. Used as candidates to test
# the observed prices against, not as an answer keyed on the venue: the MiFID
# II tick regime bands by price *and* by liquidity, so "Euronext Amsterdam"
# does not determine a tick, and a table claiming it did would be exactly the
# kind of confident wrong input this cost model was rebuilt to get away from.
CANDIDATE_TICKS: tuple[float, ...] = (
    0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0)


@dataclasses.dataclass(frozen=True)
class TickSize:
    """A tick inferred from the prices themselves, and how well it fits.

    `agreement` is the fraction of observed prices that are exact multiples of
    `size`. A real tick fits essentially every print; anything below about 0.98
    means the series has been adjusted for distributions, or comes from a venue
    quoting finer than the candidate list, and the inference should not be
    used. `usable` folds that judgement into one boolean.
    """
    size: float | None
    agreement: float
    prices: int

    @property
    def usable(self) -> bool:
        return self.size is not None and self.agreement >= 0.98

    def floor_bps(self, price: float) -> float | None:
        """The narrowest half-spread this tick permits, at `price`, in bps.

        The quoted spread cannot be finer than one tick, so the half-spread
        cannot be finer than half a tick. An *effective* half-spread can in
        principle come in below that, because a trade can execute at the
        midpoint -- but on a lit continuous order book with no midpoint
        matching, an estimate materially under half a tick means the estimator
        found no bounce rather than that the bounce is small.
        """
        if not self.usable or price <= 0:
            return None
        return float(self.size / 2.0 * 10_000.0 / price)


def infer_tick_size(prices: np.ndarray, *,
                    candidates: tuple[float, ...] = CANDIDATE_TICKS) -> TickSize:
    """The coarsest tick that essentially every observed price is a multiple of.

    Prices on a lit venue land on a grid. That grid is observable, so this
    measures it rather than reading it off a table of venue rules that would
    have to be kept correct by hand and could not be checked.

    Coarsest, not finest, because every price on a 0.01 grid is also on the
    0.005 and 0.001 grids -- the finest candidate always fits and would tell
    you nothing. The coarsest fitting grid is the one the venue is quoting on.

    >>> grid = np.round(np.arange(100.0, 110.0, 0.37) / 0.01) * 0.01
    >>> t = infer_tick_size(grid)
    >>> t.size, round(t.agreement, 3), t.usable
    (0.01, 1.0, True)

    Prices that have been adjusted for a distribution are off the grid, and
    the inference declines rather than reporting the finest candidate:

    >>> t = infer_tick_size(grid * 0.98317)
    >>> t.usable
    False

    >>> infer_tick_size(np.array([np.nan, np.nan])).size is None
    True
    """
    clean = np.asarray(prices, dtype=float)
    clean = clean[np.isfinite(clean) & (clean > 0)]
    if clean.size < 10:
        return TickSize(None, 0.0, int(clean.size))

    best: tuple[float, float] | None = None
    for tick in sorted(candidates, reverse=True):
        # A price is on the grid if it is within a millionth of a multiple.
        # The slack absorbs binary floating point, not a genuine mismatch:
        # 0.1 + 0.2 is not 0.3 in any of this.
        residual = np.abs(clean / tick - np.round(clean / tick))
        agreement = float(np.mean(residual < 1e-6))
        if agreement >= 0.98:
            best = (tick, agreement)
            break
        if best is None or agreement > best[1]:
            best = (tick, agreement)

    assert best is not None
    size, agreement = best
    return TickSize(size if agreement >= 0.98 else None, agreement,
                    int(clean.size))
