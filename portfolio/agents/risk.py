"""Equal risk contribution, with some weights held fixed.

The tracker's headline is the gap between a holding's share of the money and
its share of the risk. Schneider Electric at 18% of capital and 28% of risk is
the number no broker app shows. This is the policy that closes that gap
deliberately rather than pointing at it.

It forecasts nothing. It needs only the covariance matrix, which is the one
thing at this sample size that can actually be estimated -- volatility
clusters and is autocorrelated, while returns at these horizons are close to
unforecastable. That is why it is first in the build order: of the four
policies on the shortlist it is the only one whose inputs the data can
support.

The problem
-----------
Write sigma_p(w) = sqrt(w' Sigma w). Euler's theorem on a function homogeneous
of degree 1 gives the exact decomposition

    sigma_p = sum_i w_i (Sigma w)_i / sigma_p        each term is asset i's
                                                     contribution to risk

Equal risk contribution asks for those terms to be equal, so

    w_i (Sigma w)_i  =  k    for every i, some constant k

For a positive definite Sigma and long-only weights this has a unique
solution (Maillard, Roncalli and Teiletche, 2010).

With some weights fixed
-----------------------
The gold ETC is held at a different broker and cannot be traded against the
rest of the book, so its weight is exogenous. The problem becomes: equalise
risk contributions across the *free* assets, with the fixed assets present in
Sigma and contributing to every marginal risk, and the free weights summing
to whatever the fixed ones leave over.

That is not a different problem, it is the same optimisation over a
restricted simplex, and it is still well posed. It matters that gold stays in
Sigma: it is genuinely part of the portfolio, it moves the correlations, and
excluding it would equalise contributions to a portfolio that does not exist.

How it is solved
----------------
Coordinate descent on the first-order condition, with an outer bisection that
makes the budget constraint hold by construction rather than by hope.

Setting the derivative of the standard convex reformulation to zero for one
coordinate gives a quadratic with a closed-form positive root:

    Sigma_ii w_i^2 + c_i w_i - lambda = 0,   c_i = sum_{j != i} Sigma_ij w_j

    w_i = ( -c_i + sqrt(c_i^2 + 4 Sigma_ii lambda) ) / (2 Sigma_ii)

The positive root is always positive for lambda > 0 whatever the sign of
c_i, which is what makes this robust where the naive iteration
w_i <- b_i / (Sigma w)_i is not: a holding that hedges the rest of the book
has a negative covariance with it, and the naive update sends its weight
negative. Gold is exactly such a holding, so this is not a hypothetical.

Sweeping the coordinates to convergence gives free weights that rise
monotonically in lambda, so bisecting lambda until they sum to the budget
converges. The result is then *verified* rather than assumed: the risk
contributions are recomputed from `core.risk` and the function refuses to
return if they are not equal to tolerance.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from ..core.risk import risk_decomposition
from .base import MarketView, Proposal

__all__ = ["equal_risk_weights", "risk_contribution_spread",
           "EqualRiskContribution"]


def risk_contribution_spread(weights, cov: pd.DataFrame,
                             among=None) -> float:
    """Largest gap between any two risk contributions, as a fraction of the mean.

    The acceptance test for the solver: zero means the contributions really
    are equal. Reported rather than asserted silently, so a policy can say
    how well it solved its own problem.
    """
    decomposition = risk_decomposition(weights, cov)
    keys = [str(c) for c in cov.columns]
    percent = dict(zip(keys, decomposition.percent))
    chosen = [percent[k] for k in (among if among is not None else keys)]
    if not chosen:
        return 0.0
    mean = float(np.mean(chosen))
    if mean == 0:
        return 0.0
    return float((max(chosen) - min(chosen)) / abs(mean))


def _sweep(sigma: np.ndarray, w: np.ndarray, free: list[int], lam: float,
           sweeps: int, tolerance: float) -> np.ndarray:
    """Cyclic coordinate descent at a fixed lambda. Fixed entries never move."""
    w = w.copy()
    for _ in range(sweeps):
        before = w[free].copy()
        for i in free:
            a = float(sigma[i, i])
            if a <= 0:
                raise ValueError(
                    f"asset {i} has non-positive variance {a}; the covariance "
                    f"matrix is degenerate and equal risk contribution is "
                    f"undefined on it")
            c = float(sigma[i] @ w) - a * w[i]
            w[i] = (-c + math.sqrt(c * c + 4.0 * a * lam)) / (2.0 * a)
        if float(np.max(np.abs(w[free] - before))) < tolerance:
            break
    return w


def equal_risk_weights(cov: pd.DataFrame, fixed: "dict[str, float] | None" = None,
                       tolerance: float = 1e-12, max_sweeps: int = 500,
                       max_bisections: int = 200) -> pd.Series:
    """Weights whose risk contributions are equal across the tradeable assets.

    `fixed` maps instrument to a weight that must not change. Those assets
    stay in the covariance matrix and affect every other holding's marginal
    risk; they are simply not solved for.

    Two uncorrelated assets, one twice as volatile as the other. Equal risk
    means weight inversely proportional to volatility, so two thirds and one
    third -- a result available in closed form, which is what makes it a
    check on the solver rather than a restatement of it:

    >>> cov = pd.DataFrame([[0.01, 0.0], [0.0, 0.04]], index=["A", "B"],
    ...                    columns=["A", "B"])
    >>> [round(w, 8) for w in equal_risk_weights(cov)]
    [0.66666667, 0.33333333]

    Identical assets get identical weights, correlated or not:

    >>> same = pd.DataFrame([[0.04, 0.02], [0.02, 0.04]], index=["A", "B"],
    ...                     columns=["A", "B"])
    >>> [round(w, 8) for w in equal_risk_weights(same)]
    [0.5, 0.5]

    With one weight pinned, the rest share what is left and their own
    contributions are equalised among themselves:

    >>> three = pd.DataFrame(
    ...     [[0.04, 0.01, 0.00], [0.01, 0.09, 0.00], [0.00, 0.00, 0.02]],
    ...     index=["A", "B", "G"], columns=["A", "B", "G"])
    >>> w = equal_risk_weights(three, fixed={"G": 0.20})
    >>> round(float(w["G"]), 10)
    0.2
    >>> round(float(w.sum()), 10)
    1.0
    >>> round(risk_contribution_spread(w.to_dict(), three, among=["A", "B"]), 8)
    0.0

    Pinning every weight leaves nothing to solve, and the input comes back:

    >>> list(equal_risk_weights(three, fixed={"A": 0.3, "B": 0.5, "G": 0.2}))
    [0.3, 0.5, 0.2]
    """
    keys = [str(c) for c in cov.columns]
    sigma = cov.to_numpy(dtype=float)
    fixed = {k: float(v) for k, v in (fixed or {}).items()}

    unknown = set(fixed) - set(keys)
    if unknown:
        raise ValueError(f"fixed weights name instruments not in the covariance "
                         f"matrix: {sorted(unknown)}")
    pinned = sum(fixed.values())
    if pinned > 1.0 + 1e-9:
        raise ValueError(f"fixed weights already sum to {pinned:.6g}; there is "
                         f"nothing left to allocate")

    free = [i for i, k in enumerate(keys) if k not in fixed]
    w = np.array([fixed.get(k, 0.0) for k in keys], dtype=float)
    if not free:
        return pd.Series(w, index=keys)

    budget = 1.0 - pinned
    if budget <= 0:
        return pd.Series(w, index=keys)

    # Seed the free block at equal weights within its budget.
    w[free] = budget / len(free)

    # Free weights rise monotonically in lambda, so bracket it and bisect.
    # The bracket is found by doubling rather than guessed, because the right
    # scale depends on the covariance's units and a fixed guess would work
    # for daily returns and fail for monthly ones.
    lo, hi = 1e-18, 1e-12
    for _ in range(max_bisections):
        if float(_sweep(sigma, w, free, hi, max_sweeps, tolerance)[free].sum()) >= budget:
            break
        hi *= 4.0
    else:                                                    # pragma: no cover
        raise RuntimeError("could not bracket the risk-parity solution")

    solved = w
    for _ in range(max_bisections):
        mid = 0.5 * (lo + hi)
        candidate = _sweep(sigma, w, free, mid, max_sweeps, tolerance)
        total = float(candidate[free].sum())
        solved = candidate
        if abs(total - budget) <= tolerance * max(1.0, budget):
            break
        if total < budget:
            lo = mid
        else:
            hi = mid

    out = pd.Series(solved, index=keys)
    # Renormalise the free block: bisection lands within tolerance, and the
    # weights must sum to exactly 1 for the harness's accounting.
    free_sum = float(out.iloc[free].sum())
    if free_sum > 0:
        for i in free:
            out.iloc[i] = out.iloc[i] * budget / free_sum

    # Verify rather than trust. A solver that quietly returns a non-solution
    # is worse than one that fails, because everything downstream would then
    # be measuring a policy nobody wrote.
    spread = risk_contribution_spread(out.to_dict(), cov,
                                      among=[keys[i] for i in free])
    if spread > 1e-6:
        raise RuntimeError(
            f"equal risk contribution did not converge: the free assets' risk "
            f"shares still differ by {spread:.3%} of their mean. The "
            f"covariance matrix may be near-singular.")
    return out


@dataclasses.dataclass
class EqualRiskContribution:
    """Size positions so each contributes equally to portfolio volatility.

    Tier 1 on the shortlist: it forecasts nothing. The only input is the
    covariance matrix, and its whole claim is that capital weights and risk
    weights should be the same thing.

    `fixed` names holdings whose weight the policy may not change, and it is
    passed in rather than discovered, because whether a holding is tradeable
    is a fact about the account -- which broker holds it -- and not about the
    market. The referee enforces the same constraint independently; this is
    the policy declining to propose what it knows cannot be executed.
    """
    lookback: int = 252
    fixed: "dict[str, float] | None" = None
    name: str = "equal-risk-contribution"

    def observe(self, view: MarketView) -> Proposal:
        available = list(view.available)
        if len(available) < 2:
            return Proposal(dict(view.held), 0.0,
                            f"only {len(available)} instrument(s) have enough "
                            f"history to model; holding", self.name)

        held = {k: float(v) for k, v in view.held.items()}
        # A frozen holding is pinned at the weight actually held right now,
        # never at a target. Only the keys of `fixed` are used: its weight
        # drifts with the market and the policy has no way to correct it, so
        # pinning it anywhere other than where it actually is would propose a
        # trade that cannot be made.
        pinned = {k: held.get(k, 0.0) for k in (self.fixed or {})
                  if k in available}

        cov = view.covariance(self.lookback, available)
        weights = equal_risk_weights(cov, fixed=pinned)

        reason = self._reason(cov, weights, pinned, available)
        return Proposal({k: float(v) for k, v in weights.items()}, 0.5,
                        reason, self.name)

    def _reason(self, cov, weights, pinned, available) -> str:
        """What the policy did, and what the constraint cost.

        The second half matters more than the first. If equal risk
        contribution wants a different gold weight and cannot have it, that is
        a real finding about the account's structure -- two brokers -- rather
        than a defect, and it should be visible rather than inferred from a
        weight that never moves.
        """
        free = [k for k in available if k not in pinned]
        spread = risk_contribution_spread(weights.to_dict(), cov, among=free)
        head = (f"equal risk contribution across {len(free)} tradeable "
                f"holdings over a {self.lookback}-day window "
                f"(risk shares equal to {spread:.1e} of their mean)")
        if not pinned:
            return head
        unconstrained = equal_risk_weights(cov)
        gaps = []
        for k, w in sorted(pinned.items()):
            want = float(unconstrained[k])
            gaps.append(f"{k} is frozen at {w:.1%}; unconstrained equal risk "
                        f"would hold {want:.1%} ({want - w:+.1%})")
        return head + ". " + "; ".join(gaps)
