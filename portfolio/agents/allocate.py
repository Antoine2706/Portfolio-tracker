"""Directing new money, when selling is the expensive half.

Why this exists rather than a rebalancing schedule
--------------------------------------------------
This account has no regular contribution and no leverage. Both facts point
away from continuous rebalancing and towards this:

*Every scheduled rebalance is a sell plus a buy.* Transaction tax is charged
each way and the spread is paid each way, so a rebalancing policy pays twice
for every unit of drift it removes. Step 3 measured what that costs.

*New money is a rebalancing channel that costs nothing extra.* The purchase
was going to happen anyway. Choosing its destination is free, and it is the
only way to move toward equal risk contribution without paying on the sell
side.

So: given an amount of new cash C and the book as it stands, choose how much
of C goes into each holding so that the resulting risk shares are as equal as
possible.

    minimise    dispersion of the risk shares of x = v + b
    over        b_i >= 0,  sum(b_i) = C

Risk shares are homogeneous of degree zero in the weight vector, so the
normalisation by V + C never has to appear: the shares of `v + b` are the
shares of `(v + b) / (V + C)`.

This forecasts nothing. Its only input is the covariance matrix, the same one
`EqualRiskContribution` uses, over the same window and through the same
estimator. It is tier 1 for the same reason.

What it cannot do, which the output has to say
----------------------------------------------
Buy-only reduces an *overweight* risk share only by dilution. If one holding
is 28% of the risk on 18% of the capital, no purchase of the others brings it
to equal unless the purchase is large relative to the book. The reachable set
at cash C is

    { w : w_i >= v_i / (V + C),  sum(w) = 1 }

which is a strict subset of the simplex, shrinking to the current book as
C -> 0 and growing to the whole simplex as C -> infinity. Every honest report
from here therefore carries three numbers, not one: where the book is now,
the best reachable *with this much money*, and the best reachable if selling
were allowed. A 2% improvement presented on its own is a result; presented
against a floor it cannot pass, it is a fact about the account.

Does more money always help? Only if every holding can receive it
----------------------------------------------------------------
The tempting claim is that the reachable sets nest, so the floor is
non-increasing in C and the inverse question ("how much would I need?") is
answerable by bisection from zero. The proof is a rescaling. Given a
reachable x1 = v + b1 at C1, put lambda = (V + C2) / (V + C1) for C2 > C1.
Then x2 = lambda x1 has *identical* risk shares, since shares are homogeneous
of degree zero, and it is reachable at C2:

    b2 = x2 - v = (lambda - 1) v + lambda b1 >= 0,   sum(b2) = C2.

That is correct, and it is what this module claimed. It is also conditional,
and the condition was not stated: b2_i must be zero for a holding that cannot
receive new money, and (lambda - 1) v_i is strictly positive whenever v_i is.
A pinned holding's weight is an equality v_i / (V + C) that *moves* with C,
not an inequality that relaxes. So the configurations do not nest, and the
floor can rise.

On the seed book it does. Three holdings there cannot receive; as C grows
their weights are diluted towards nothing, so their risk shares go to zero
and the best the other seven can manage is equality among themselves. Seven
equal shares and three zeros disperse by (1/7) / (1/10) = 10/7, and the
measured floor climbs towards that: 1.4189 at four times the book, 1.4259 at
sixteen, 1.4277 at fifty, against a limit of 1.4286. Nothing is broken. Money
cannot fix a structure it is not allowed to touch, and past a point it makes
the gap worse by diluting what it cannot buy.

So the guarantee has two halves, and `cash_for_dispersion` states both: the
amount it names really does reach the target, always; that no smaller amount
does holds only when nothing is pinned. `test_allocate.py` checks the
monotonicity where it is real and checks the counterexample where it is not.

What "dispersion" means here, and why it is not a range
-------------------------------------------------------
The coefficient of variation of the risk shares: their standard deviation
over their mean. Zero exactly when the contributions are equal, and bounded
above by sqrt(m - 1) for m holdings, so it reads against a known scale.

It was a range -- maximum minus minimum, over the mean -- and that was the
mistake underneath everything below. A range is a rank statistic: on six
holdings it is decided by two of them and discards the other four, it is
non-differentiable wherever the argmax or argmin changes hands, and a policy
minimising it chases the single worst laggard while ignoring the shape of the
rest. Worse, nothing actually minimised it: the descent minimised a sum of
squared deviations and the report printed a range, so the tool graded answers
by a rule it had not used to produce them. That mismatch is what a
least-squares descent reaching 1.205 against a coarse grid's 1.136 was
measuring. It read as a solver failure and it was an objective failure.

The range is still printed, beside the dispersion and never instead of it,
because it answers something a coefficient of variation does not: how far
apart the extremes are, which is what says whether one holding is the
problem.

How it is solved
----------------
The feasible set is convex. The objective is smooth but *not* convex -- risk
shares are ratios of quadratics -- and the specification's claim that a convex
feasible set makes the problem convex does not follow. So there is no
certificate from the search itself, and every part below exists because
something measurable went wrong without it.

1.  Projected gradient descent on the objective itself, which is the reported
    dispersion squared. Not a surrogate: `dispersion_of` and
    `objective_and_gradient` are the same function, so the descent and the
    report cannot disagree about which of two allocations is better. Squared
    only because a standard deviation has a square-root kink at zero, exactly
    where the optimum is. The gradient is derived in full below rather than
    taken from a library, and checked against central differences.

2.  `aim_at_equal_risk` hands the search the equal-risk portfolio mapped into
    b-space. Past the amount at which that portfolio becomes reachable it *is*
    the answer, and no agnostic start finds it: without this the floor on a
    hedged book found equal risk at four times the book and lost it again at
    sixteen.

3.  A multi-resolution pattern search over pairwise exchanges, from every
    candidate. Smooth is not convex, and this is not a tidy-up: on one
    eight-holding book the descent converged to a point with a
    projected-gradient residual of 1e-16 -- a genuine constrained stationary
    point, not a stall, and more iterations do not move it -- from which the
    exchange search escaped by moving a third of the money, taking the
    dispersion from 1.00 to 0.56.

The result is labelled "best found", not "the minimum", because on a
non-convex objective those are different claims. Two independent things check
it: a dense brute-force grid on small books, and `spinu_sweep`, which solves
the convex Maillard-Roncalli-Teiletche/Spinu form to global optimality over
this same polytope for a family of barrier parameters. The search has never
lost to either, and by a margin as small as 6e-4 in one place, which is what
makes it a check rather than a formality.

Buy-only is a constraint of the problem, not a preference. A negative
purchase is a sale, so the search asserts non-negativity and the order
asserts it fits the money. Both of those caught real bugs on their first run:
a hoisted affordability check went stale inside the search and drove a
holding to -2,500 EUR, which reached the order as a plausible 148 shares of
something the purchase could not pay for.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ..core.risk import risk_decomposition
from .risk import (equal_risk_weights, risk_contribution_spread,
                   risk_dispersion)

__all__ = ["project_onto_simplex", "risk_shares", "dispersion_of",
           "spread_of", "among_indices", "metric_on_arrays",
           "objective_and_gradient", "solve_continuous",
           "pattern_search", "spinu_sweep", "aim_at_equal_risk",
           "reachable_floor",
           "unconstrained_floor",
           "Purchase", "Refusal", "Destination", "BuyOnlyAllocation",
           "allocate_buy_only", "cash_for_dispersion", "pinned_holdings",
           "best_reachable_dispersion", "smallest_meaningful_cash"]


# Below this, a difference of two dispersions is the solver's residual rather
# than a fact about the book. Dispersion is a dimensionless ratio and a real
# one runs from about 0.01 to 25 on the books here, so 1e-6 is far under
# anything meaningful and far over the 1e-17 that float arithmetic leaves
# behind. It exists because a gap tested against exact zero answered
# differently on two machines.
RESOLVED = 1e-6


def project_onto_simplex(y: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto { b : b >= 0, sum(b) = total }.

    The exact algorithm of Duchi, Shalev-Shwartz, Singer and Chandra (2008):
    sort descending, find how many coordinates survive, subtract the common
    threshold and clip. Exact and O(n log n), where a naive "clip then
    renormalise" is neither -- it does not return the nearest feasible point,
    and on a book with one holding pushed to zero it converges somewhere else
    entirely.

    Already feasible points are returned unchanged:

    >>> project_onto_simplex(np.array([0.25, 0.75]), 1.0).round(10)
    array([0.25, 0.75])

    Negative coordinates are removed and their mass redistributed, not
    clipped in place:

    >>> project_onto_simplex(np.array([1.0, -0.5]), 1.0).round(10)
    array([1., 0.])
    >>> project_onto_simplex(np.array([0.6, 0.2, -0.4]), 1.0).round(10)
    array([0.7, 0.3, 0. ])

    All the cash goes somewhere, whatever came in, and an input that ranks
    every destination as bad still has to choose:

    >>> out = project_onto_simplex(np.array([-3.0, -1.0, -2.0]), 500.0)
    >>> float(out.sum()), out.round(4).tolist()
    (500.0, [165.6667, 167.6667, 166.6667])
    """
    if total < 0:
        raise ValueError(f"cannot allocate a negative amount: {total}")
    n = int(y.size)
    if n == 0:
        return y.astype(float)
    if total == 0:
        return np.zeros(n, dtype=float)
    ordered = np.sort(np.asarray(y, dtype=float))[::-1]
    cumulative = np.cumsum(ordered) - total
    counts = np.arange(1, n + 1, dtype=float)
    survives = ordered - cumulative / counts > 0
    rho = int(np.nonzero(survives)[0][-1]) + 1 if survives.any() else 1
    theta = float(cumulative[rho - 1] / rho)
    return np.maximum(np.asarray(y, dtype=float) - theta, 0.0)


def risk_shares(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Each holding's share of portfolio volatility, from Euler's theorem.

        RC_i = x_i (Sigma x)_i / (x' Sigma x)      and sum_i RC_i = 1

    Written here on raw arrays because the optimiser calls it thousands of
    times; `core.risk.risk_decomposition` is the same identity on labelled
    frames and is what every reported figure goes through.
    """
    total = float(x @ sigma @ x)
    if total <= 0:
        return np.zeros_like(x)
    return (x * (sigma @ x)) / total


def dispersion_of(x, cov: pd.DataFrame, among=None) -> float:
    """The reported metric: coefficient of variation of the risk shares.

    Delegates to `agents.risk` so the allocator and the equal risk policy
    cannot drift into reporting two different numbers under one name.

    It is also, squared, exactly what the solver minimises. That identity is
    the point of it and it was not true before: the report was a range and the
    descent minimised a sum of squares, so the two disagreed about which of
    two allocations was better and the tool was grading answers by a rule it
    had not used to produce them.
    """
    keys = [str(c) for c in cov.columns]
    weights = {k: float(v) for k, v in zip(keys, np.asarray(x, dtype=float))}
    return risk_dispersion(weights, cov, among=among)


def spread_of(x, cov: pd.DataFrame, among=None) -> float:
    """The range of the risk shares, over their mean. Descriptive only.

    Printed beside the dispersion because it answers something the coefficient
    of variation does not -- how far apart the extremes are, which is what
    tells you whether one holding is the problem. Nothing minimises it.
    """
    keys = [str(c) for c in cov.columns]
    weights = {k: float(v) for k, v in zip(keys, np.asarray(x, dtype=float))}
    return risk_contribution_spread(weights, cov, among=among)


def metric_on_arrays(cov: pd.DataFrame, among=None):
    """The same metric as `dispersion_of`, as a fast closure over raw arrays.

    The search evaluates it tens of thousands of times, and the labelled path
    rebuilds a `RiskDecomposition` on every call. This is the identical
    arithmetic without the frames -- `test_allocate.py` asserts the two agree
    exactly, because a fast path that is only nearly the reported one is a
    tool optimising something it does not print.
    """
    sigma = cov.to_numpy(dtype=float)
    idx = among_indices(cov, among)

    def metric(x: np.ndarray) -> float:
        if idx.size == 0:
            return 0.0
        chosen = risk_shares(x, sigma)[idx]
        mean = float(chosen.mean())
        if mean == 0:
            return 0.0
        return float(chosen.std(ddof=0) / abs(mean))

    return metric


def among_indices(cov: pd.DataFrame, among=None) -> np.ndarray:
    """Positions in the covariance matrix of the holdings the metric covers."""
    keys = [str(c) for c in cov.columns]
    return np.array([keys.index(k) for k in
                     (among if among is not None else keys)], dtype=int)


def objective_and_gradient(b: np.ndarray, held: np.ndarray, sigma: np.ndarray,
                           among: np.ndarray) -> "tuple[float, np.ndarray]":
    """The objective -- the reported metric, squared -- and its exact gradient.

    Not a surrogate. `dispersion_of` is the coefficient of variation of the
    risk shares and this is its square, so the descent and the report are the
    same function and cannot disagree about which of two allocations is
    better. Squared because the square is smooth everywhere including at the
    optimum, where a standard deviation has a square-root kink, and because
    minimising a non-negative quantity and minimising its square are the same
    problem.

    Write A for the holdings the metric covers, m = |A|, and

        RC_i = x_i s_i / q,   s = Sigma x,   q = x' Sigma x,   x = held + b
        mu   = (1/m) sum_{i in A} RC_i
        P    = (1/m) sum_{i in A} RC_i^2
        g    = P / mu^2 - 1                      = CV^2

    Differentiating a risk share,

        d RC_i / d x_k = ( [i = k] s_i + x_i Sigma_ik ) / q  -  2 RC_i s_k / q

    so for any coefficient vector c that is zero outside A the weighted sum
    collapses to one expression, which is the only piece of arithmetic here:

        G(c)_k = ( c_k s_k + (Sigma (c * x))_k - 2 s_k (c . RC) ) / q

    Then dP/dx = (2/m) G(RC_A) and d mu/dx = (1/m) G(1_A), giving

        dg/dx = ( 2 / (m mu^2) ) [ G(RC_A) - (P / mu) G(1_A) ]

    The formula checks itself when A is every holding: the shares sum to one
    whatever x is, so mu is the constant 1/m and its gradient must vanish.
    Substituting c = 1 gives G(1)_k = (s_k + s_k - 2 s_k) / q = 0, which it
    does. `test_allocate.py` checks the whole thing against central
    differences anyway, because a hand-derived gradient with a sign error
    still descends -- just to the wrong place.
    """
    x = held + b
    s = sigma @ x
    q = float(x @ s)
    if q <= 0 or among.size == 0:
        return 0.0, np.zeros_like(b)
    rc = (x * s) / q
    chosen = rc[among]
    m = float(among.size)
    mu = float(chosen.mean())
    if mu == 0:
        return 0.0, np.zeros_like(b)
    p = float((chosen * chosen).mean())

    def weighted(c: np.ndarray) -> np.ndarray:
        return (c * s + sigma @ (c * x) - 2.0 * s * float(c @ rc)) / q

    carrier = np.zeros_like(rc)
    carrier[among] = chosen
    ones = np.zeros_like(rc)
    ones[among] = 1.0
    grad = (2.0 / (m * mu * mu)) * (weighted(carrier) - (p / mu) * weighted(ones))
    # Clamped because p / mu^2 - 1 is a difference of two numbers that agree to
    # every digit once the shares are equal, so rounding can put it just below
    # zero exactly where the answer is right.
    return max(p / (mu * mu) - 1.0, 0.0), grad


def _descend(start: np.ndarray, held: np.ndarray, sigma: np.ndarray,
             cash: float, free: np.ndarray, among: np.ndarray, *,
             steps: int, tolerance: float) -> np.ndarray:
    """One projected-gradient descent on the objective, from one start."""
    n = held.size
    b = start.copy()
    full = np.zeros(n)
    full[free] = b
    value, grad = objective_and_gradient(full, held, sigma, among)
    step = max(cash, 1.0) / max(float(np.abs(grad[free]).max()), 1e-30)
    for _ in range(steps):
        moved = project_onto_simplex(b - step * grad[free], cash)
        trial = np.zeros(n)
        trial[free] = moved
        trial_value, trial_grad = objective_and_gradient(trial, held, sigma, among)
        if trial_value < value:
            improvement = value - trial_value
            b, value, grad = moved, trial_value, trial_grad
            step *= 1.6
            if improvement <= tolerance * max(value, 1e-16):
                break
        else:
            step *= 0.4
            if step <= 1e-18:
                break
    return b


def starting_points(cash: float, free: int, seed: int = 20260908,
                    randoms: int = 12) -> "list[np.ndarray]":
    """Where to start the descent, given that the objective is not convex.

    The equal split, every single-destination vertex, and a spread of random
    points drawn uniformly from the simplex. The vertices matter because the
    answer at small cash often *is* a vertex -- one holding is short and the
    whole purchase belongs there -- and the random draws matter because a
    descent from the equal split alone was measurably beaten by a coarse
    brute-force grid on a five-holding book. That is the experiment in
    `test_allocate.py::TestItFindsTheOptimum`, and it is the reason this
    function exists rather than one start.
    """
    rng = np.random.default_rng(seed)
    out = [np.full(free, cash / free)]
    for j in range(free):
        vertex = np.zeros(free)
        vertex[j] = cash
        out.append(vertex)
    for _ in range(randoms):
        draw = rng.exponential(size=free)          # uniform on the simplex
        out.append(cash * draw / draw.sum())
    return out


def solve_continuous(held: np.ndarray, sigma: np.ndarray, cash: float,
                     allowed: np.ndarray, among: np.ndarray, *,
                     steps: int = 400, randoms: int = 12,
                     tolerance: float = 1e-14) -> "list[np.ndarray]":
    """Candidate allocations: one descent per start, all of them returned.

    `among` is the index array the metric covers, from `among_indices`.

    `allowed` is a boolean mask of the holdings new money may go into.
    Everything else is pinned at zero purchase -- it still sits in the
    covariance matrix and still has a risk share, it simply cannot receive.

    Every candidate comes back rather than the best one, because "best" on
    the surrogate and "best" on the reported metric are different questions
    and the caller answers the second. Picking here on the surrogate and
    polishing only the winner is exactly how the solver came to be beaten by
    a grid: the surrogate's optimum sat in a valley the polish could not
    cross, while a start it had discarded was already on the right side.
    """
    n = held.size
    free = np.nonzero(allowed)[0]
    if cash <= 0 or free.size == 0:
        return [np.zeros(n, dtype=float)]
    out = []
    for start in starting_points(cash, int(free.size), randoms=randoms):
        b = _descend(start, held, sigma, cash, free, among,
                     steps=steps, tolerance=tolerance)
        full = np.zeros(n, dtype=float)
        full[free] = b
        out.append(full)
    return out


def pattern_search(b: np.ndarray, held: np.ndarray, metric,
                   allowed: np.ndarray, cash: float, *,
                   resolution: float = 1e-7) -> np.ndarray:
    """Descend on the same metric by moving cash between holdings.

    Not a tidy-up, and no longer a correction for the descent optimising
    something other than the report -- it now optimises exactly the report.
    This exists because smooth is not convex.

    The measurement. On the eight-holding hedged book in `test_allocate.py`
    the projected-gradient descent converges to a point whose
    projected-gradient residual is 1e-16 of the purchase: a genuine
    constrained stationary point, not an early stop, and raising the step
    budget from 400 to 10,000 does not move it. Its risk shares are

        0.237  0.191  0.172 -0.052  0.168  0.218 -0.121  0.185

    with dispersion 1.00. Moving a third of the money between holdings reaches

        0.165  0.148  0.131  0.176  0.132  0.161 -0.058  0.145

    at 0.56. No line search finds that from there, because it is not downhill
    in any single direction; a pairwise exchange finds it because it is a
    different basin, not a further step in the same one.

    The search is a multi-resolution pattern search: sweep every ordered pair
    of holdings moving a fixed quantum from one to the other, repeat while
    anything improves, then halve the quantum. Cash-conserving by
    construction, since every move takes from one holding and gives to
    another, so no rounding drift can creep into the total.
    """
    free = list(np.nonzero(allowed)[0])
    if len(free) < 2 or cash <= 0:
        return b
    best = b.copy()
    best_value = metric(held + best)
    quantum = cash / 2.0
    floor = cash * resolution
    while quantum > floor:
        improved = True
        while improved:
            improved = False
            for i in free:
                for j in free:
                    # The affordability check belongs HERE, against the
                    # current `best`, not once per source outside this loop.
                    # Hoisted, it goes stale the moment a move succeeds: the
                    # source has already given a quantum away and the next
                    # destination takes another, so a holding starting with
                    # 5,000 of a 5,000 purchase went to 0 and then to -2,500.
                    # A negative purchase is a SALE, which is the one thing
                    # this allocator must never propose, and it reached the
                    # order as a plausible-looking 148 shares of something the
                    # money could not buy.
                    if i == j or best[i] < quantum:
                        continue
                    trial = best.copy()
                    trial[i] -= quantum
                    trial[j] += quantum
                    value = metric(held + trial)
                    if value < best_value - 1e-15:
                        best, best_value, improved = trial, value, True
        quantum *= 0.5
    if best.min() < -1e-9:                                   # pragma: no cover
        raise AssertionError(
            f"the search produced a negative purchase of {best.min():,.2f}, "
            f"which is a sale. Buy-only is a constraint of the problem, not a "
            f"preference, so this is a bug rather than a result.")
    return best


def spinu_sweep(held: np.ndarray, cov: pd.DataFrame, cash: float,
                allowed: np.ndarray, among=None, *, rungs: int = 21,
                steps: int = 800) -> "list[np.ndarray]":
    """Certified points from the convex form of equal risk contribution.

    A control, not part of the answer, and deliberately kept out of
    `reachable_floor` so that it stays an independent check on it.

    The formulation is Maillard, Roncalli and Teiletche's, sharpened by Spinu
    (2013): over x > 0,

        f(x) = 0.5 x' Sigma x  -  lam * sum_{i in A} log x_i

    is convex -- a positive semi-definite quadratic plus a sum of convex
    terms -- and its stationarity condition is

        (Sigma x)_i = lam / x_i,   i.e.   x_i (Sigma x)_i = lam  for all i,

    which is equal risk contributions exactly. Since x = v + b is affine in b
    and the buy-only set { b >= 0, sum b = C, b_i = 0 where pinned } is a
    polytope, the whole problem stays convex over it: projected gradient with
    a backtracking step converges to the *global* optimum, and any point it
    returns has a certificate the multi-start search does not.

    The catch, which is the reason this is a sweep rather than one solve. In
    the unconstrained problem `lam` is irrelevant: the objective is scale
    invariant up to an additive constant, so any `lam` gives the same weights
    once normalised. Here the budget is pinned and the weights carry lower
    bounds `w_i >= v_i / (V + C)`, that invariance is gone, and `lam` becomes
    a real parameter whose value changes the answer. One `lam` is not a
    substitute for a search over the polytope. What the family gives is a
    one-parameter set of globally certified points, cheap and reproducible,
    and `test_allocate.py` requires that the search never loses to it. If it
    ever does, the search is under-converged -- which is the failure mode that
    took a hand-run brute-force grid to notice last time.

    Empty when the barrier is not defined: a holding in the metric's set that
    is worth nothing and cannot receive has x_i = 0, so log x_i is not finite
    and this family has nothing to say about that book.
    """
    n = held.size
    free = np.nonzero(allowed)[0]
    idx = among_indices(cov, among)
    if cash <= 0 or free.size == 0 or idx.size == 0:
        return []
    sigma = cov.to_numpy(dtype=float)
    barrier = np.zeros(n, dtype=bool)
    barrier[idx] = True
    start = np.zeros(n, dtype=float)
    start[free] = cash / free.size
    if float((held + start)[barrier].min()) <= 0:
        return []

    def value(b: np.ndarray) -> float:
        x = held + b
        if float(x[barrier].min()) <= 0:
            return float("inf")
        return float(0.5 * x @ sigma @ x - lam * np.log(x[barrier]).sum())

    def gradient(b: np.ndarray) -> np.ndarray:
        x = held + b
        g = sigma @ x
        g[barrier] -= lam / x[barrier]
        return g

    # A scale for lam from the problem itself rather than a guess: at a point
    # where the contributions are equal, x' Sigma x = sum_i x_i (Sigma x)_i =
    # m * lam, so lam ~ (x' Sigma x) / m at the starting point is the right
    # order of magnitude whatever the covariance's units are.
    x0 = held + start
    anchor = float(x0 @ sigma @ x0) / float(idx.size)
    out: "list[np.ndarray]" = []
    for power in np.linspace(-3.0, 3.0, rungs):
        lam = anchor * float(10.0 ** power)
        b = start.copy()
        here = value(b)
        step = 1.0 / max(float(np.abs(gradient(b)[free]).max()), 1e-30)
        for _ in range(steps):
            moved = np.zeros(n, dtype=float)
            moved[free] = project_onto_simplex(
                b[free] - step * gradient(b)[free], cash)
            there = value(moved)
            if there < here - 1e-18:
                b, here, step = moved, there, step * 1.5
            else:
                step *= 0.4
                if step <= 1e-24:
                    break
        out.append(b)
    return out


def aim_at_equal_risk(held: np.ndarray, cov: pd.DataFrame, cash: float,
                      allowed: np.ndarray) -> "np.ndarray | None":
    """Start the search from the answer, when the answer is reachable.

    Every other start in this module is agnostic: an equal split, a vertex, a
    random simplex point. None of them knows what equal risk contribution
    looks like, and past the amount at which that portfolio becomes reachable
    it is the answer, so a start there is a start at the optimum.

    The construction. A holding new money cannot enter has weight exactly
    `v_i / (V + C)` at every reachable point, so it is *pinned*, not
    approximated -- which is precisely `equal_risk_weights`' `fixed`
    argument. Solve for the weights that equalise the rest against it, turn
    those weights back into euros, and subtract what is already held:

        b = w (V + C) - v

    That sums to C by construction, since `w` sums to one. It is feasible if
    and only if no component is negative, and a negative component is a
    holding already above its equal-risk weight -- a sale, which is the one
    thing not on offer. Projecting onto the simplex is what makes it a start
    rather than a proposal: it returns the nearest point that is buy-only and
    spends exactly C.

    Why it matters, and it is not a refinement. On the seed book with every
    holding allocatable -- the case where the module docstring's nesting
    argument does hold, so the floor provably cannot rise -- the floor
    measured without this start was

        0.25x     1x      4x     16x     50x
        2.3838  1.3556  2.2424  1.7013  1.2996

    which rises at four times the book and does not reach equal risk at
    fifty. That is forbidden, so it was the solver, not the account. With
    this start:

        2.3838  1.3556  0.0000  0.0000  0.0000

    Past the point where the equal risk portfolio becomes reachable it is
    found exactly, because it is handed over rather than searched for. Over
    forty-eight random books at ten amounts each, the floor now never rises.

    What makes the surface hard is worth naming, because it is not book size
    and it is not the number of holdings. It is *negative correlation*. When
    (Sigma x)_i is negative a holding's risk share is negative, so the minimum
    of the range is not bounded below by zero and the surface acquires local
    minima that no exchange between two holdings can leave. On books of
    independent assets the search never failed at all -- forty-eight of them,
    ten amounts each, monotone with the start and monotone without it, which
    is why `test_allocate.py` checks this on a fixture with a hedge in it
    instead. The real book is that fixture: its gold ETC runs at -0.98
    against one of the equity holdings, which is the entire reason it is
    held.

    Two uncorrelated assets, the second twice as volatile, so equal risk is
    two thirds and one third. A book that is 90% in the volatile one, and
    enough new money to fix it outright:

    >>> cov = pd.DataFrame([[0.01, 0.0], [0.0, 0.04]], index=["A", "B"],
    ...                    columns=["A", "B"])
    >>> held = np.array([100.0, 900.0])
    >>> b = aim_at_equal_risk(held, cov, 9000.0, np.ones(2, dtype=bool))
    >>> [round(float(v), 4) for v in b]
    [6566.6667, 2433.3333]
    >>> round(dispersion_of(held + b, cov), 10)
    0.0

    A tenth of that money cannot reach it: the aim asks for 733 in A and 367
    in B against 900 already held in B, so B's component is negative. The
    projection spends the lot on A rather than proposing the sale:

    >>> b = aim_at_equal_risk(held, cov, 100.0, np.ones(2, dtype=bool))
    >>> [round(float(v), 6) for v in b]
    [100.0, 0.0]
    """
    free = np.nonzero(allowed)[0]
    if cash <= 0 or free.size == 0:
        return None
    total = float(held.sum()) + float(cash)
    if total <= 0:
        return None

    keys = [str(c) for c in cov.columns]
    fixed = {keys[i]: float(held[i]) / total
             for i in range(len(keys)) if not allowed[i]}
    try:
        weights = equal_risk_weights(cov, fixed=fixed)
    except (RuntimeError, ValueError):
        # A degenerate covariance matrix has no equal-risk portfolio to aim
        # at. The other starts still stand; this one simply has nothing to
        # contribute, and a fabricated aim would be worse than none.
        return None

    raw = weights.to_numpy(dtype=float) * total - held
    out = np.zeros(held.size, dtype=float)
    out[free] = project_onto_simplex(raw[free], float(cash))
    return out


def reachable_floor(held: np.ndarray, cov: pd.DataFrame, cash: float,
                    allowed: np.ndarray, among=None, *, thorough: bool = True
                    ) -> "tuple[float, np.ndarray]":
    """The best dispersion buy-only money can reach at this amount, and how.

    "Best found" rather than "the minimum": the objective is not convex, so
    the honest claim is what a multi-start descent plus a direct polish
    achieved. On small books `test_allocate.py` compares it against a dense
    grid, which is the only check that distinguishes the two.
    """
    sigma = cov.to_numpy(dtype=float)
    metric = metric_on_arrays(cov, among)
    free = np.nonzero(allowed)[0]
    candidates = solve_continuous(held, sigma, cash, allowed,
                                  among_indices(cov, among),
                                  randoms=12 if thorough else 2)
    aim = aim_at_equal_risk(held, cov, cash, allowed)
    if aim is not None:
        candidates.append(aim)
    # The gradient descents supply good starts; raw simplex points supply
    # starts whose basin the surrogate never visits. Both are needed: the
    # surrogate's optimum sat in the wrong basin on the fixture above, and a
    # search seeded only from it stayed there however long it ran.
    if cash > 0 and free.size:
        for start in starting_points(cash, int(free.size), seed=8412,
                                     randoms=6 if thorough else 0):
            raw = np.zeros(held.size, dtype=float)
            raw[free] = start
            candidates.append(raw)

    best_value, best = np.inf, np.zeros(held.size, dtype=float)
    for candidate in candidates:
        searched = pattern_search(candidate, held, metric, allowed, cash)
        value = metric(held + searched)
        if value < best_value:
            best_value, best = value, searched
    return float(best_value), best


def unconstrained_floor(cov: pd.DataFrame, among=None) -> float:
    """What selling could reach: the equal risk contribution portfolio itself.

    Zero to solver tolerance whenever the covariance matrix is well behaved.
    Reported anyway rather than printed as a constant, because a near-singular
    matrix makes it not zero, and a reader comparing a buy-only floor against
    an assumed zero would be measuring the solver instead of the account.
    """
    try:
        weights = equal_risk_weights(cov)
    except (RuntimeError, ValueError):
        return float("nan")
    return dispersion_of(weights.to_numpy(dtype=float), cov, among)


# --------------------------------------------------------------------------
# From a solution to an order
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Purchase:
    """One line of the order: whole shares, and what it does to the book."""
    isin: str
    name: str
    broker: str
    shares: int
    price: float
    amount: float
    cost: float
    weight_before: float
    weight_after: float
    risk_before: float
    risk_after: float


@dataclasses.dataclass(frozen=True)
class Refusal:
    """A holding new money could not go into, and the reason in full."""
    isin: str
    name: str
    reason: str


@dataclasses.dataclass(frozen=True)
class Diluted:
    """A holding the purchase pushes further from where it should be.

    Every euro that goes somewhere else makes a holding that cannot receive
    money a smaller share of a larger book. That is a real consequence of the
    recommendation and it points the wrong way, so it is stated rather than
    left to be inferred from two weights on different lines.

    `target` is the weight unconstrained equal risk contribution wants, which
    is the standard the rest of the report is measured against. Printing the
    move without it would say the holding got smaller without saying whether
    smaller was wrong.
    """
    isin: str
    name: str
    weight_before: float
    weight_after: float
    target: float
    reason: str

    @property
    def drift(self) -> float:
        """How much further from the target the purchase leaves it.

        Positive means the gap widened. Negative would mean the dilution
        happened to help, which is possible when the holding is above its
        target to begin with.
        """
        return abs(self.weight_after - self.target) - abs(
            self.weight_before - self.target)


@dataclasses.dataclass(frozen=True)
class Destination:
    """What happens if the whole purchase goes to one holding.

    The table that answers "is directing this money worth what it costs".
    Improvement per euro of cost is the ratio a reader can act on: a
    wide-spread holding can buy a dispersion improvement that a cheaper one
    buys almost as much of for a third of the money.
    """
    isin: str
    name: str
    dispersion: float
    cost: "float | None"                 # None when it cannot be priced at this size
    improvement: float                   # dispersion now minus dispersion here

    @property
    def per_euro(self) -> "float | None":
        """Improvement bought per euro of trading cost."""
        if self.cost is None or self.cost <= 0:
            return None
        return float(self.improvement / self.cost)


def _priceable(costs, isin: str) -> "str | None":
    """None if a purchase can be priced, else the refusal in full.

    Asked by pricing a token amount rather than by inspecting the cost
    model's internals, so the allocator refuses exactly what the executor
    would refuse and cannot drift from it.
    """
    from .execution import UnknownCost
    try:
        costs.instrument_cost(isin, 100.0, "buy")
    except UnknownCost as refusal:
        return str(refusal)
    return None


def _allowed_mask(keys, buyable, costs, refusals: "list[Refusal]") -> np.ndarray:
    """Which holdings new money may go into, with every exclusion explained.

    Two separate reasons, kept separate because they are undone by different
    things. `buyable` is a fact about the account -- an instrument nobody can
    add to, whatever the price. Unpriceable is a gap in the record that one
    contract note closes, and the holding becomes allocatable the moment the
    rate is written down.

    Being untradeable is NOT one of them. A holding at a second broker cannot
    be rebalanced against the rest of the book, and can be bought with new
    money perfectly well: the two flags answer different questions and reading
    one as the other gets the constraint wrong in one direction or the other.
    """
    mask = np.zeros(len(keys), dtype=bool)
    for i, isin in enumerate(keys):
        facts = costs.facts(isin)
        name = facts.name or isin
        if isin not in buyable:
            refusals.append(Refusal(isin, name,
                                    "not marked buyable, so new money may not "
                                    "go into it"))
            continue
        refused = _priceable(costs, isin)
        if refused is not None:
            refusals.append(Refusal(isin, name, refused))
            continue
        mask[i] = True
    return mask


def _unaffordable(b: np.ndarray, keys, allowed: np.ndarray,
                  costs) -> "tuple[np.ndarray, list[tuple[str, str]]]":
    """Allocations a broker's fee schedule cannot support, and why.

    Two reasons, both derived per broker from its own structure rather than
    from one global number. MeDirect charges no commission, so no purchase
    there is too small on fee grounds; Keytrade's flat 2.45 EUR is 1.2% of a
    200 EUR order, which sets a floor.

    And a ceiling, which is the one that is easy to miss: Keytrade publishes
    only its first tier, so a 4,000 EUR order there has no recorded fee at
    all. The cost model refuses to price it -- correctly -- and an allocator
    that did not expect the refusal simply crashed on a perfectly ordinary
    book. Being priceable is a property of the trade, not only of the
    instrument.
    """
    from .execution import UnknownCost
    dropped = []
    out = allowed.copy()
    for i, isin in enumerate(keys):
        if not allowed[i] or b[i] <= 0:
            continue
        floor = costs.minimum_trade_value(isin)
        if floor > 0 and b[i] < floor:
            out[i] = False
            dropped.append((isin,
                            f"the allocator wanted {b[i]:,.0f} EUR, below the "
                            f"{floor:,.0f} EUR its broker's flat fee makes "
                            f"worthwhile, so nothing goes here"))
            continue
        try:
            costs.instrument_cost(isin, float(b[i]), "buy")
        except UnknownCost as refusal:
            out[i] = False
            dropped.append((isin, f"a {b[i]:,.0f} EUR purchase cannot be "
                                  f"priced: {refusal}"))
    return out, dropped


def _round_to_shares(b: np.ndarray, prices: np.ndarray, cash: float,
                     held: np.ndarray, metric, allowed: np.ndarray,
                     limit: int = 2000) -> np.ndarray:
    """Whole share counts, then spend the remainder where it helps most.

    Rounding down first and topping up greedily, rather than rounding to
    nearest: rounding to nearest can spend more cash than there is, and a
    proposal that does not fit the money is not a proposal.

    The greedy top-up stops when no affordable single share improves the
    metric, so leftover cash is reported as leftover rather than forced into
    a holding that does not want it.
    """
    shares = np.zeros(b.size, dtype=int)
    for i in np.nonzero(allowed)[0]:
        if prices[i] > 0:
            shares[i] = int(b[i] // prices[i])
    spent = float(shares @ prices)
    best_value = metric(held + shares * prices)
    for _ in range(limit):
        leftover = cash - spent
        pick, pick_value = None, best_value
        for i in np.nonzero(allowed)[0]:
            if prices[i] <= 0 or prices[i] > leftover:
                continue
            trial = shares.copy()
            trial[i] += 1
            value = metric(held + trial * prices)
            if value < pick_value - 1e-15:
                pick, pick_value = i, value
        if pick is None:
            break
        shares[pick] += 1
        spent += float(prices[pick])
        best_value = pick_value
    return shares


@dataclasses.dataclass(frozen=True)
class BuyOnlyAllocation:
    """Where a purchase should go, and how much of the gap it can close."""
    cash: float
    invested: float
    leftover: float
    purchases: tuple[Purchase, ...]
    refused: tuple[Refusal, ...]
    destinations: tuple[Destination, ...]
    diluted: tuple[Diluted, ...]
    dispersion_now: float
    dispersion_after: float              # on the EXECUTABLE, rounded order
    spread_now: float                    # the range, descriptive only
    spread_after: float
    floor_at_cash: float                 # continuous best found at this cash
    floor_unlimited: float               # what selling could reach
    cost_parts: "dict[str, float]"
    meaningful_cash: "float | None"
    best_worst_gap: float                # how much the choice of destination is worth
    assumptions: tuple[str, ...]

    @property
    def total_cost(self) -> float:
        return float(sum(self.cost_parts.values()))

    @property
    def improvement(self) -> float:
        return float(self.dispersion_now - self.dispersion_after)

    @property
    def rounding_penalty(self) -> float:
        """What whole shares cost against the continuous optimum.

        Reported rather than swallowed. On a small purchase of high-priced
        instruments it is not negligible, and a tool quoting the continuous
        figure is quoting a portfolio that cannot be bought.
        """
        return float(self.dispersion_after - self.floor_at_cash)

    @property
    def closable(self) -> float:
        """The share of the gap to equal risk this much money can close."""
        room = self.dispersion_now - self.floor_unlimited
        if room <= 0:
            return 0.0
        return float((self.dispersion_now - self.floor_at_cash) / room)

    @staticmethod
    def _labels(rows) -> "dict[str, str]":
        """Display names, with the ISIN attached where two holdings share one.

        The seed book carries two ETFs both shortening to "Europe Defence",
        and a table listing both under one name is a table nobody can act on.
        Only the colliding rows are lengthened, since an ISIN on every line
        would cost the width the names need -- but the collision is decided
        across the whole report rather than per table, so a name that is
        qualified in one place is not bare in another.
        """
        # By ISIN first: a holding appears in the order AND in the
        # destinations table, and counting it twice would qualify every
        # purchased name as though it collided with itself.
        named = {row.isin: row.name for row in rows}
        counts: "dict[str, int]" = {}
        for name in named.values():
            counts[name] = counts.get(name, 0) + 1
        return {isin: (f"{name[:14]} {isin}" if counts[name] > 1 else name)
                for isin, name in named.items()}

    def lines(self) -> list[str]:
        names = self._labels([*self.purchases, *self.destinations])
        out = [f"Directing {self.cash:,.0f} EUR of new money", "=" * 64, ""]
        # Two different questions, kept apart because merging them once
        # printed "where this goes matters" beside "closes 2.3% of the gap".
        # How much the purchase MOVES the book is the threshold below; how
        # much the CHOICE is worth is the best-worst spread, and a purchase
        # too small to matter can still have a destination worth avoiding.
        if self.meaningful_cash is None:
            out.append("No purchase of any size meaningfully changes the risk "
                       "shares of this book.")
        elif self.cash < self.meaningful_cash:
            out.append(
                f"At {self.cash:,.0f} EUR this hardly moves the book: the best "
                f"any purchase this size reaches is {self.floor_at_cash:.4f} "
                f"against {self.dispersion_now:.4f} now. The smallest purchase "
                f"that closes a worthwhile share of the gap is about "
                f"{self.meaningful_cash:,.0f} EUR.")
        else:
            out.append(
                f"This moves the book: {self.cash:,.0f} EUR closes "
                f"{self.closable:.1%} of the gap to equal risk contribution.")
        out.append(
            f"Best and worst destinations differ by "
            f"{self.best_worst_gap:.3f} of dispersion, so the choice is worth "
            f"{'making' if self.best_worst_gap > 0.01 else 'almost nothing'}.")
        out += ["", "Order", "-" * 64]
        if not self.purchases:
            out.append("  nothing to buy")
        for p in self.purchases:
            # 27 wide, because a qualified label is a 14-character name, a
            # space and a 12-character ISIN, and half an ISIN identifies
            # nothing -- it is worse than the collision it was meant to fix.
            out.append(f"  {p.shares:>6} x {names[p.isin][:27]:27} "
                       f"{p.amount:>9,.0f} EUR at {p.price:>8,.2f}  "
                       f"({p.broker})")
            out.append(f"         risk share {p.risk_before:>6.1%} -> "
                       f"{p.risk_after:>6.1%}   weight {p.weight_before:>6.1%} "
                       f"-> {p.weight_after:>6.1%}")
        if any(min(p.risk_before, p.risk_after) < 0 for p in self.purchases):
            out += ["",
                    "  A risk share below zero is not a misprint. A holding "
                    "that moves against the rest of the book has a negative "
                    "marginal contribution to its volatility, so it carries "
                    "risk away rather than adding it -- that is what a hedge "
                    "is for, and the shares still sum to 100%. It does mean "
                    "the dispersion figure below spans a wider range than an "
                    "unhedged book's would, so compare it against this book "
                    "over time rather than against a number from elsewhere."]
        out += [
            "",
            f"  invested   {self.invested:>10,.2f} EUR",
            f"  left over  {self.leftover:>10,.2f} EUR (whole shares only)",
            f"  cost       {self.total_cost:>10,.2f} EUR = "
            + " + ".join(f"{v:,.2f} {k}" for k, v in self.cost_parts.items() if v),
            "",
            "Dispersion of risk shares", "-" * 64,
            f"  {'':24}{'CV':>10}{'range':>10}",
            f"  now                      {self.dispersion_now:>10.4f}"
            f"{self.spread_now:>10.4f}",
            f"  after this purchase      {self.dispersion_after:>10.4f}"
            f"{self.spread_after:>10.4f}",
            f"  floor at this amount     {self.floor_at_cash:>10.4f}"
            f"{'':10}   best reachable buy-only",
            f"  floor with no limit      {self.floor_unlimited:>10.4f}"
            f"{'':10}   what selling could reach",
            "",
            f"  CV is the coefficient of variation of the risk shares -- their "
            f"standard deviation over their mean, zero when the contributions "
            f"are equal, at most sqrt(m - 1) for m holdings. It is what is "
            f"minimised and what is reported, and they are the same number. "
            f"The range is the largest gap between any two shares over the "
            f"same mean; it is descriptive, it is decided by two holdings out "
            f"of {len(self.destinations) or 'several'}, and nothing optimises "
            f"it.",
            f"  This purchase closes {self.closable:.1%} of the gap between the "
            f"book as it stands and equal risk contribution. Whole shares cost "
            f"{self.rounding_penalty:.4f} of that against the continuous "
            f"optimum.",
        ]
        if self.destinations:
            out += ["", "If it all went to one holding", "-" * 64,
                    f"  {'holding':28}{'dispersion':>11}{'cost':>9}"
                    f"{'gain/EUR':>11}"]
            for d in sorted(self.destinations, key=lambda x: x.dispersion):
                per = "n/a" if d.per_euro is None else f"{d.per_euro:.5f}"
                cost = "n/a" if d.cost is None else f"{d.cost:,.2f}"
                out.append(f"  {names[d.isin][:27]:28}{d.dispersion:>11.4f}"
                           f"{cost:>9}{per:>11}")
        if self.diluted:
            out += ["", "What this purchase does to what it cannot buy",
                    "-" * 64,
                    f"  {'holding':28}{'weight now':>12}{'after':>10}"
                    f"{'equal risk':>12}{'drift':>9}"]
            for d in self.diluted:
                out.append(f"  {d.name[:27]:28}{d.weight_before:>12.1%}"
                           f"{d.weight_after:>10.1%}{d.target:>12.1%}"
                           f"{d.drift:>+9.1%}")
            out.append(
                "  Every euro that goes somewhere else makes these a smaller "
                "share of a larger book. A positive drift means the purchase "
                "leaves them further from where equal risk contribution wants "
                "them, which is a cost of the recommendation and not a "
                "rounding artefact.")
        if self.refused:
            out += ["", "Refused", "-" * 64]
            for r in self.refused:
                # Already prefixed by its ISIN, so it needs no qualifying.
                out.append(f"  {r.isin} ({r.name}): {r.reason}")
        if self.assumptions:
            out += ["", "Cost inputs that are estimates", "-" * 64]
            out += [f"  - {a}" for a in self.assumptions]
        return out


def allocate_buy_only(*, values: "dict[str, float]", prices: "dict[str, float]",
                      cov: pd.DataFrame, cash: float, costs,
                      buyable: "set[str] | frozenset[str]",
                      among=None, meaningful: float = 0.05
                      ) -> BuyOnlyAllocation:
    """Choose where new money goes so risk shares come out as equal as possible.

    `values` is what each holding is currently worth, `prices` what one share
    costs, `buyable` which holdings new money may go into at all. Nothing is
    ever sold: `b >= 0` is a constraint of the problem, not a preference.

    The order of operations matters and is not arbitrary. Solve continuously;
    drop any holding whose share of the money falls below its own broker's
    minimum economic trade and solve again without it; only then round to
    whole shares and spend the remainder greedily. Rounding first would let a
    holding keep an allocation the fee makes pointless, and applying the
    minimum after rounding would leave the released cash unspent.

    Every reported figure is recomputed on the rounded, executable order. The
    continuous optimum appears once, as the floor, so the rounding penalty is
    visible rather than absorbed.
    """
    from .execution import UnknownCost

    keys = [str(c) for c in cov.columns]
    held = np.array([float(values.get(k, 0.0)) for k in keys], dtype=float)
    share_prices = np.array([float(prices.get(k, 0.0)) for k in keys], dtype=float)
    metric = metric_on_arrays(cov, among)

    refusals: "list[Refusal]" = []
    allowed = _allowed_mask(keys, set(buyable), costs, refusals)
    for i, k in enumerate(keys):
        if allowed[i] and share_prices[i] <= 0:
            allowed[i] = False
            refusals.append(Refusal(k, costs.facts(k).name or k,
                                    "no share price is available, so a whole "
                                    "number of shares cannot be worked out"))

    dispersion_now = dispersion_of(held, cov, among)
    spread_now = spread_of(held, cov, among)
    floor_unlimited = unconstrained_floor(cov, among)

    # Solve, drop what the fees make pointless, solve again. Bounded by the
    # number of holdings: each pass removes at least one or stops.
    continuous = np.zeros(len(keys), dtype=float)
    floor_at_cash = dispersion_now
    for _ in range(len(keys) + 1):
        if not allowed.any() or cash <= 0:
            break
        floor_at_cash, continuous = reachable_floor(held, cov, cash, allowed, among)
        allowed, dropped = _unaffordable(continuous, keys, allowed, costs)
        if not dropped:
            break
        for isin, why in dropped:
            refusals.append(Refusal(isin, costs.facts(isin).name or isin, why))

    shares = _round_to_shares(continuous, share_prices, cash, held, metric, allowed)
    amounts = shares * share_prices
    after = held + amounts
    invested = float(amounts.sum())
    if invested > cash + 1e-6:                               # pragma: no cover
        raise AssertionError(
            f"the order spends {invested:,.2f} of a {cash:,.2f} purchase. A "
            f"proposal that does not fit the money is not a proposal.")
    # The floor is an infimum over a set the integer order belongs to, so an
    # executable order that beats the continuous search is not a paradox: it
    # is proof the search did not reach the floor. Reporting the search's
    # answer anyway would print a floor the tool has already been under.
    dispersion_after = dispersion_of(after, cov, among)
    spread_after = spread_of(after, cov, among)
    floor_at_cash = min(floor_at_cash, dispersion_after)

    total_value = float(held.sum())
    shares_before = risk_shares(held, cov.to_numpy(dtype=float))
    shares_after = risk_shares(after, cov.to_numpy(dtype=float))
    parts: "dict[str, float]" = {}
    purchases = []
    for i, k in enumerate(keys):
        if shares[i] <= 0:
            continue
        for name, amount in costs.cost_breakdown(k, float(amounts[i]), "buy").items():
            parts[name] = parts.get(name, 0.0) + amount        # noqa: PERF401
        facts = costs.facts(k)
        purchases.append(Purchase(
            isin=k, name=facts.name or k, broker=facts.broker,
            shares=int(shares[i]), price=float(share_prices[i]),
            amount=float(amounts[i]),
            cost=float(costs.instrument_cost(k, float(amounts[i]), "buy")),
            weight_before=float(held[i] / total_value) if total_value else 0.0,
            weight_after=float(after[i] / after.sum()) if after.sum() else 0.0,
            risk_before=float(shares_before[i]), risk_after=float(shares_after[i])))

    destinations = []
    for i, k in enumerate(keys):
        if not allowed[i]:
            continue
        only = held.copy()
        only[i] += cash
        value = metric(only)
        try:
            priced = float(costs.instrument_cost(k, cash, "buy"))
        except UnknownCost:
            priced = None            # a tiered broker with no tier this large
        destinations.append(Destination(
            isin=k, name=costs.facts(k).name or k, dispersion=value,
            cost=priced, improvement=float(dispersion_now - value)))
    gap = (max(d.dispersion for d in destinations)
           - min(d.dispersion for d in destinations)) if destinations else 0.0

    # What the purchase does to the holdings it cannot enter. Every euro that
    # goes elsewhere makes them a smaller share of a larger book, which is a
    # consequence of the recommendation pointing the wrong way, so it is
    # printed rather than left to be inferred from two weights.
    diluted = []
    if total_value > 0 and after.sum() > 0:
        try:
            wanted = equal_risk_weights(cov)
        except (RuntimeError, ValueError):
            wanted = None
        excuses = {r.isin: r.reason for r in refusals}
        for i, k in enumerate(keys):
            if allowed[i] or held[i] <= 0:
                continue
            diluted.append(Diluted(
                isin=k, name=costs.facts(k).name or k,
                weight_before=float(held[i] / total_value),
                weight_after=float(after[i] / after.sum()),
                target=(float(wanted[k]) if wanted is not None else float("nan")),
                reason=excuses.get(k, "new money may not go into it")))

    return BuyOnlyAllocation(
        cash=float(cash), invested=invested, leftover=float(cash - invested),
        purchases=tuple(purchases), refused=tuple(refusals),
        destinations=tuple(destinations), diluted=tuple(diluted),
        dispersion_now=dispersion_now,
        dispersion_after=dispersion_after,
        spread_now=spread_now, spread_after=spread_after,
        floor_at_cash=floor_at_cash, floor_unlimited=floor_unlimited,
        cost_parts={k: v for k, v in parts.items() if v},
        meaningful_cash=smallest_meaningful_cash(
            values=values, cov=cov, costs=costs, buyable=buyable, among=among,
            improvement=meaningful),
        best_worst_gap=float(gap),
        assumptions=tuple(costs.assumptions()))


def cash_for_dispersion(target: float, *, values: "dict[str, float]",
                        cov: pd.DataFrame, costs,
                        buyable: "set[str] | frozenset[str]", among=None,
                        ceiling: float = 50.0, tolerance: float = 0.01
                        ) -> "float | None":
    """The smallest purchase that reaches `target` dispersion, buy-only.

    The question worth asking, and more decision-relevant than any single
    allocation: if the answer is 40,000 EUR on a 15,000 EUR book, contributions
    of the size this account actually makes will not fix the structure and the
    real choice is whether to sell.

    It scans a geometric ladder for the first amount that reaches the target
    and bisects inside the bracket, rather than bisecting from zero. Bisecting
    from zero is what a non-increasing floor would allow, and the module
    docstring shows the floor is non-increasing only when every holding can
    receive new money -- a pinned holding is diluted by the money it cannot
    take, and on the seed book the floor rises towards 10/7 because of it.

    What that costs is stated rather than hidden. Every step of both the scan
    and the bisection keeps `floor(high) <= target`, and the floor is
    evaluated by the same search the allocation uses at a lower start count,
    which can only find a worse floor than the thorough one. So the amount
    returned really does reach the target, always. What is guaranteed only in
    the unpinned case is that no *smaller* amount does: with a pinned holding
    the ladder can in principle step over a window where the target was
    briefly reachable, and the answer is then too high rather than wrong.

    None when no rung up to `ceiling` times the book reaches the target.
    Returning a very large number instead would suggest a plan; None says
    this search did not find one.
    """
    keys = [str(c) for c in cov.columns]
    held = np.array([float(values.get(k, 0.0)) for k in keys], dtype=float)
    refusals: "list[Refusal]" = []
    allowed = _allowed_mask(keys, set(buyable), costs, refusals)
    if not allowed.any():
        return None
    book = float(held.sum()) or 1.0

    # The bisection calls this a dozen times, so it runs the cheaper search.
    # The answer is a threshold to reason about, not an order to place, and a
    # slightly conservative one -- fewer starts can only find a worse floor,
    # so the cash requirement comes out if anything too high rather than too
    # low, which is the safe direction for a number that says "this cannot be
    # fixed by contributions".
    def floor(cash: float) -> float:
        return reachable_floor(held, cov, cash, allowed, among,
                               thorough=False)[0]

    if dispersion_of(held, cov, among) <= target:
        return 0.0

    # The ladder. It starts at the resolution rather than at the book, because
    # the amount that matters is often a small fraction of it -- that is what
    # `smallest_meaningful_cash` asks -- and climbs by halves so a window where
    # a pinned book briefly reaches the target is narrow enough to land in.
    # The ceiling is tested explicitly: a target reachable only at the very top
    # would otherwise be missed by a rung that overshot it.
    rungs = []
    rung = tolerance * book
    while rung < ceiling * book:
        rungs.append(rung)
        rung *= 1.5
    rungs.append(ceiling * book)

    low, high = 0.0, None
    for rung in rungs:
        if floor(rung) <= target:
            high = rung
            break
        low = rung
    if high is None:
        return None

    while (high - low) > tolerance * book:
        mid = 0.5 * (low + high)
        if floor(mid) <= target:
            high = mid
        else:
            low = mid
    return float(high)


def pinned_holdings(*, values: "dict[str, float]", cov: pd.DataFrame, costs,
                    buyable: "set[str] | frozenset[str]") -> "list[str]":
    """Holdings with money in them that new money cannot enter.

    The condition under which the floor stops being monotone in cash, so the
    thing a report has to name before it explains why more money made the
    number worse. A pinned holding worth nothing does not count: the rescaling
    in the module docstring needs `(lambda - 1) v_i` to be zero, and a zero
    value gives that without any purchase.
    """
    keys = [str(c) for c in cov.columns]
    allowed = _allowed_mask(keys, set(buyable), costs, [])
    return [k for i, k in enumerate(keys)
            if not allowed[i] and float(values.get(k, 0.0)) > 0]


def best_reachable_dispersion(*, values: "dict[str, float]", cov: pd.DataFrame,
                              costs, buyable: "set[str] | frozenset[str]",
                              among=None, ceiling: float = 50.0,
                              tolerance: float = 0.01
                              ) -> "tuple[float, float]":
    """The lowest dispersion any purchase reaches, and roughly what it takes.

    What to say when `cash_for_dispersion` returns None, because "no" on its
    own is not a decision. With a pinned holding the floor falls, bottoms out
    and rises again -- the money dilutes towards nothing a risk share it is
    not allowed to buy -- so there is a *best* amount rather than "as much as
    you can", and it is worth naming even when the target is out of reach.

    The amount is the best rung of the same ladder `cash_for_dispersion`
    climbs, so it is accurate to about a factor of a half. That is the right
    precision for a number whose use is deciding whether to sell instead.
    """
    keys = [str(c) for c in cov.columns]
    held = np.array([float(values.get(k, 0.0)) for k in keys], dtype=float)
    allowed = _allowed_mask(keys, set(buyable), costs, [])
    now = dispersion_of(held, cov, among)
    if not allowed.any():
        return now, 0.0
    book = float(held.sum()) or 1.0

    best, at = now, 0.0
    rung = tolerance * book
    while rung < ceiling * book * 1.5:
        value = reachable_floor(held, cov, min(rung, ceiling * book), allowed,
                                among, thorough=False)[0]
        if value < best:
            best, at = value, min(rung, ceiling * book)
        rung *= 1.5
    return float(best), float(at)


def smallest_meaningful_cash(*, values: "dict[str, float]", cov: pd.DataFrame,
                             costs, buyable: "set[str] | frozenset[str]",
                             among=None, improvement: float = 0.05
                             ) -> "float | None":
    """The purchase below which this hardly moves the book.

    The smallest amount whose best reachable dispersion closes `improvement`
    of the gap between the book as it stands and what selling could reach.
    Answering "if I type 200 EUR, does this do anything" with a number rather
    than a shrug.

    `improvement` is a *fraction of that gap*, not an absolute number of
    dispersion units, and the difference is not pedantry. As an absolute
    threshold of 0.05 it read as scale-free and was not: on a book sitting at
    a dispersion of 1.5 it asked for a third of the whole gap, while on the
    demo book at 15.07 it asked for 0.3% of it and duly reported that 197 EUR
    "meaningfully changes" a book that purchase moves by 2%. A threshold whose
    meaning depends on how bad the book already is answers a different
    question on every book it is applied to.

    None when there is no gap to close -- the book is already at equal risk
    contribution, so no purchase improves it and the honest answer is that
    the destination does not matter at any size.

    "No gap" is `RESOLVED`, not zero, and the difference is not cosmetic. Both
    ends of the gap come out of the same iterative solver, so on a book that
    *is* at equal risk they differ by whatever the last bisection left behind:
    exactly 0.0 here, and 1e-17 on the CI runner's BLAS. Tested against zero,
    that machine took a gap seventeen orders of magnitude below anything
    measurable as real structure and reported that 100 EUR would close a fifth
    of it. A guard that gives different answers on two machines is not
    measuring the book.
    """
    now = dispersion_of(
        np.array([float(values.get(str(c), 0.0)) for c in cov.columns]),
        cov, among)
    room = now - unconstrained_floor(cov, among)
    if not (room > RESOLVED):             # also catches a nan from a bad cov
        return None
    return cash_for_dispersion(now - improvement * room, values=values,
                               cov=cov, costs=costs, buyable=buyable,
                               among=among)
