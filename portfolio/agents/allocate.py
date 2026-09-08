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

The reachable sets nest -- C1 < C2 implies reachable(C1) is contained in
reachable(C2), since every lower bound v_i / (V + C) falls -- so the floor is
non-increasing in C and the inverse question ("how much would I need?") is
answerable by bisection. `test_allocate.py` checks that monotonicity rather
than assuming it.

How it is solved, and why not the way the specification assumed
--------------------------------------------------------------
The feasible set is convex. The objective is not, and it is worth being
explicit about that rather than inheriting a false comfort: the reported
dispersion is `risk_contribution_spread`, a maximum minus a minimum, which is
neither smooth nor convex in the weights. Nor is the smooth least-squares
alternative convex -- risk shares are ratios of quadratics.

So the solve is in two steps, and the second one is the optimiser.

1.  Projected gradient descent on the smooth surrogate

        F(b) = sum_i ( RC_i(v + b) - 1/n )^2

    with the gradient written out below rather than taken from a library,
    projected onto the simplex { b >= 0, sum b = C } at every step, from many
    starting points. This generates candidates; it does not choose between
    them.

2.  A multi-resolution pattern search on the reported metric itself, run from
    every candidate and from raw simplex points besides. Least squares and a
    range disagree about more than the last decimal -- the worked numbers are
    in `pattern_search` -- so descending on the surrogate and calling its
    answer the floor was measurably beaten by a coarse brute-force grid.

The result is labelled "best found", not "the minimum", because on a
non-convex objective those are different claims. `test_allocate.py` checks it
against a dense brute-force search on small books, which is the only way to
tell the two apart.

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
from .risk import equal_risk_weights, risk_contribution_spread

__all__ = ["project_onto_simplex", "risk_shares", "dispersion_of",
           "metric_on_arrays", "objective_and_gradient", "solve_continuous",
           "pattern_search", "reachable_floor", "unconstrained_floor",
           "Purchase", "Refusal", "Destination", "BuyOnlyAllocation",
           "allocate_buy_only", "cash_for_dispersion",
           "smallest_meaningful_cash"]


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
    """The reported metric: largest gap between risk shares, over their mean.

    Delegates to `agents.risk` so the allocator and the equal risk policy
    cannot drift into reporting two different numbers under one name.
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
    keys = [str(c) for c in cov.columns]
    sigma = cov.to_numpy(dtype=float)
    idx = np.array([keys.index(k) for k in (among if among is not None else keys)],
                   dtype=int)

    def metric(x: np.ndarray) -> float:
        if idx.size == 0:
            return 0.0
        chosen = risk_shares(x, sigma)[idx]
        mean = float(chosen.mean())
        if mean == 0:
            return 0.0
        return float((chosen.max() - chosen.min()) / abs(mean))

    return metric


def objective_and_gradient(b: np.ndarray, held: np.ndarray, sigma: np.ndarray,
                           target: np.ndarray) -> "tuple[float, np.ndarray]":
    """The smooth surrogate and its exact gradient.

        F(b) = sum_i ( RC_i(x) - t_i )^2,        x = held + b

    Differentiating RC_i = x_i s_i / q with s = Sigma x and q = x' Sigma x:

        d RC_i / d x_k = ( [i = k] s_i + x_i Sigma_ik ) / q  -  2 RC_i s_k / q

    and so, writing d_i = RC_i - t_i,

        dF / d x_k = (2/q) [ d_k s_k + (Sigma (d * x))_k - 2 s_k (d . RC) ]

    Written out rather than differentiated numerically: a finite-difference
    gradient on a ratio of quadratics loses most of its precision near the
    optimum, which is exactly where the optimiser needs it. `test_allocate.py`
    checks this against central differences anyway, because a hand-derived
    gradient with a sign error still descends -- just to the wrong place.
    """
    x = held + b
    s = sigma @ x
    q = float(x @ s)
    if q <= 0:
        return 0.0, np.zeros_like(b)
    rc = (x * s) / q
    d = rc - target
    grad = (2.0 / q) * (d * s + sigma @ (d * x) - 2.0 * s * float(d @ rc))
    return float(d @ d), grad


def _equal_target(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=float)


def _descend(start: np.ndarray, held: np.ndarray, sigma: np.ndarray,
             cash: float, free: np.ndarray, target: np.ndarray, *,
             steps: int, tolerance: float) -> np.ndarray:
    """One projected-gradient descent on the surrogate, from one start."""
    n = held.size
    b = start.copy()
    full = np.zeros(n)
    full[free] = b
    value, grad = objective_and_gradient(full, held, sigma, target)
    step = max(cash, 1.0) / max(float(np.abs(grad[free]).max()), 1e-30)
    for _ in range(steps):
        moved = project_onto_simplex(b - step * grad[free], cash)
        trial = np.zeros(n)
        trial[free] = moved
        trial_value, trial_grad = objective_and_gradient(trial, held, sigma, target)
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
                     allowed: np.ndarray, *, steps: int = 400,
                     randoms: int = 12,
                     tolerance: float = 1e-14) -> "list[np.ndarray]":
    """Candidate allocations: one descent per start, all of them returned.

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
    target = _equal_target(n)
    out = []
    for start in starting_points(cash, int(free.size), randoms=randoms):
        b = _descend(start, held, sigma, cash, free, target,
                     steps=steps, tolerance=tolerance)
        full = np.zeros(n, dtype=float)
        full[free] = b
        out.append(full)
    return out


def pattern_search(b: np.ndarray, held: np.ndarray, metric,
                   allowed: np.ndarray, cash: float, *,
                   resolution: float = 1e-7) -> np.ndarray:
    """Descend on the REPORTED metric by moving cash between holdings.

    This is the optimiser, not a tidy-up. The gradient descent above
    minimises a sum of squared deviations; the number printed and compared
    against is a maximum minus a minimum, and on a real book the two disagree
    about more than the last decimal. On the five-holding fixture in
    `test_allocate.py` least squares reaches risk shares

        0.011  0.252  0.252  0.232  0.252     range 0.241, squares 0.0448

    by equalising four holdings and abandoning the fifth, while the range
    metric prefers

        0.072  0.295  0.299  0.072  0.263     range 0.227, squares 0.0557

    which lifts the laggards at the cost of spreading the rest. A range is
    determined by two coordinates and is blind to everything between them, so
    no smooth surrogate is a substitute for descending on it directly.

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
                                  randoms=12 if thorough else 2)
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
    dispersion_now: float
    dispersion_after: float              # on the EXECUTABLE, rounded order
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

    def lines(self) -> list[str]:
        out = [f"Directing {self.cash:,.0f} EUR of new money", "=" * 64, ""]
        if self.meaningful_cash is None:
            out.append("No purchase of any size meaningfully changes the risk "
                       "shares of this book.")
        elif self.cash < self.meaningful_cash:
            out.append(
                f"At {self.cash:,.0f} EUR it does not much matter where this "
                f"goes: the best and worst destinations differ by "
                f"{self.best_worst_gap:.3f} of dispersion. The smallest "
                f"purchase that moves it meaningfully is about "
                f"{self.meaningful_cash:,.0f} EUR.")
        else:
            out.append(
                f"Where this goes matters: best and worst destinations differ "
                f"by {self.best_worst_gap:.3f} of dispersion.")
        out += ["", "Order", "-" * 64]
        if not self.purchases:
            out.append("  nothing to buy")
        for p in self.purchases:
            out.append(f"  {p.shares:>6} x {p.name[:26]:26} "
                       f"{p.amount:>9,.0f} EUR at {p.price:>8,.2f}  "
                       f"({p.broker})")
            out.append(f"         risk share {p.risk_before:>6.1%} -> "
                       f"{p.risk_after:>6.1%}   weight {p.weight_before:>6.1%} "
                       f"-> {p.weight_after:>6.1%}")
        out += [
            "",
            f"  invested   {self.invested:>10,.2f} EUR",
            f"  left over  {self.leftover:>10,.2f} EUR (whole shares only)",
            f"  cost       {self.total_cost:>10,.2f} EUR = "
            + " + ".join(f"{v:,.2f} {k}" for k, v in self.cost_parts.items() if v),
            "",
            "Dispersion of risk shares", "-" * 64,
            f"  now                      {self.dispersion_now:>8.4f}",
            f"  after this purchase      {self.dispersion_after:>8.4f}",
            f"  floor at this amount     {self.floor_at_cash:>8.4f}   "
            f"best reachable buy-only",
            f"  floor with no limit      {self.floor_unlimited:>8.4f}   "
            f"what selling could reach",
            "",
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
                out.append(f"  {d.name[:27]:28}{d.dispersion:>11.4f}"
                           f"{cost:>9}{per:>11}")
        if self.refused:
            out += ["", "Refused", "-" * 64]
            for r in self.refused:
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

    return BuyOnlyAllocation(
        cash=float(cash), invested=invested, leftover=float(cash - invested),
        purchases=tuple(purchases), refused=tuple(refusals),
        destinations=tuple(destinations),
        dispersion_now=dispersion_now,
        dispersion_after=dispersion_after,
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
    allocation: if the answer is 40,000 EUR on a 15,000 EUR book, the
    structure cannot be fixed by contributions and the real choice is whether
    to sell.

    Bisection is valid because the reachable set grows with the money -- every
    lower bound v_i / (V + C) falls as C rises, so reachable(C1) is contained
    in reachable(C2) for C1 < C2 and the floor is non-increasing. That
    monotonicity is a property of the problem, not of the solver, so
    `test_allocate.py` checks the solver actually exhibits it.

    None when no purchase up to `ceiling` times the book reaches the target.
    Returning a very large number instead would suggest a plan; None says
    there is not one.
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
    high = book
    while floor(high) > target:
        high *= 2.0
        if high > ceiling * book:
            return None
    low = 0.0
    while (high - low) > tolerance * book:
        mid = 0.5 * (low + high)
        if floor(mid) <= target:
            high = mid
        else:
            low = mid
    return float(high)


def smallest_meaningful_cash(*, values: "dict[str, float]", cov: pd.DataFrame,
                             costs, buyable: "set[str] | frozenset[str]",
                             among=None, improvement: float = 0.05
                             ) -> "float | None":
    """The purchase below which the destination hardly matters.

    Defined as the smallest amount whose best reachable dispersion is
    `improvement` better than the book as it stands, in the units the metric
    is reported in. Answering "if I type 200 EUR, does it matter where it
    goes" with a number rather than a shrug.
    """
    now = dispersion_of(
        np.array([float(values.get(str(c), 0.0)) for c in cov.columns]),
        cov, among)
    return cash_for_dispersion(now - improvement, values=values, cov=cov,
                               costs=costs, buyable=buyable, among=among)
