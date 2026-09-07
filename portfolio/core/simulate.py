"""What-if: how the risk picture changes if the book changes.

The question the simulator answers is not "what is the risk of this
portfolio" -- `risk.py` does that -- but "what happens to it if I add 2,000
to this holding or sell half of that one". Everything here is therefore a
before/after pair computed by the same functions, so the diff cannot be an
artefact of two methods disagreeing.

Why this reuses `risk.py` rather than re-deriving anything: MCTR is the
answer to "if I add a euro here, how does portfolio volatility move", and
that is exactly the derivative the simulator is showing. Computing it a
second way would only create a way for the two views to differ.

Rebalancing targets are provided as weight generators rather than a single
"optimise" button, because each answers a different question:

    equal weights      the naive baseline everything else is measured against
    risk parity        every holding contributes the same share of the risk;
                       what the divergence table is implicitly asking for
    minimum variance   the long-only portfolio with the least volatility,
                       whatever it takes -- usually it takes concentrating in
                       the calmest holdings, which is why it is shown and not
                       recommended

Both optimisers are iterative and written out: risk parity as the convex
formulation solved by cyclical coordinate descent, minimum variance as
projected gradient descent onto the simplex. A QP solver would be shorter
and would be scipy, which `core` does not import.

No cash account, so a set of target weights is rescaled to sum to 1: a
target book summing to 90% would put 10% of the money nowhere.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .risk import (annualise_volatility, concentration, diversification_ratio,
                   risk_decomposition)

__all__ = [
    "SimulatedHolding", "SimulationResult", "Trade",
    "simulate_cash_changes", "simulate_target_weights", "equal_weights",
    "risk_parity_weights", "min_variance_weights", "trades_to_target",
]


@dataclasses.dataclass(frozen=True)
class SimulatedHolding:
    isin: str
    value_before: float
    value_after: float
    weight_before: float
    weight_after: float
    risk_before: float          # share of portfolio risk (CCTR / sigma_p)
    risk_after: float
    marginal_after: float       # MCTR after, annualised


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    holdings: list[SimulatedHolding]
    total_before: float
    total_after: float
    volatility_before: float    # annualised
    volatility_after: float
    effective_before: float
    effective_after: float
    diversification_before: float
    diversification_after: float
    max_risk_share_before: float
    max_risk_share_after: float


def _universe(cov: pd.DataFrame, *key_sets) -> list[str]:
    """Every ISIN named by the caller, in covariance column order.

    An ISIN with no covariance column has no place in the risk model -- it
    has too little history, or none -- and is refused by name rather than
    silently carried at zero risk, which would make adding it look free.
    """
    wanted = {k for keys in key_sets for k in keys}
    unknown = sorted(k for k in wanted if k not in cov.columns)
    if unknown:
        raise ValueError(f"{unknown} not in the risk model (no covariance column); a "
                         f"holding needs enough price history before it can be simulated")
    return [str(c) for c in cov.columns if c in wanted]


def _weights(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    return values / total if total > 0 else np.zeros_like(values)


def _analyse(before: dict[str, float], after: dict[str, float],
             cov: pd.DataFrame) -> SimulationResult:
    keys = _universe(cov, before, after)
    for label, values in (("values", before), ("values after", after)):
        negative = sorted(k for k, v in values.items() if v < 0)
        if negative:
            raise ValueError(f"{label} has negative entries for {negative}; short "
                             f"positions are not modelled")
    sub = cov.loc[keys, keys]
    v_before = np.array([float(before.get(k, 0.0)) for k in keys])
    v_after = np.array([float(after.get(k, 0.0)) for k in keys])
    w_before, w_after = _weights(v_before), _weights(v_after)
    d_before = risk_decomposition(w_before, sub)
    d_after = risk_decomposition(w_after, sub)

    holdings = [
        SimulatedHolding(
            isin=k,
            value_before=float(v_before[i]), value_after=float(v_after[i]),
            weight_before=float(w_before[i]), weight_after=float(w_after[i]),
            risk_before=float(d_before.percent[i]), risk_after=float(d_after.percent[i]),
            marginal_after=annualise_volatility(float(d_after.marginal[i])),
        )
        for i, k in enumerate(keys)
    ]
    holdings.sort(key=lambda h: (-h.value_after, -h.value_before, h.isin))
    return SimulationResult(
        holdings=holdings,
        total_before=float(v_before.sum()), total_after=float(v_after.sum()),
        volatility_before=annualise_volatility(d_before.portfolio_volatility),
        volatility_after=annualise_volatility(d_after.portfolio_volatility),
        effective_before=concentration(w_before).effective_holdings,
        effective_after=concentration(w_after).effective_holdings,
        diversification_before=diversification_ratio(w_before, sub),
        diversification_after=diversification_ratio(w_after, sub),
        max_risk_share_before=float(d_before.percent.max()) if keys else 0.0,
        max_risk_share_after=float(d_after.percent.max()) if keys else 0.0,
    )


def simulate_cash_changes(values: dict[str, float], cov: pd.DataFrame,
                          changes: dict[str, float]) -> SimulationResult:
    """Add or remove money per holding and recompute the risk picture.

    `values` are current market values; `changes` are signed amounts in the
    base currency. An ISIN in `changes` but not in `values` is a watchlist
    entry being bought for the first time, and must be a covariance column.

    A change that sells more than is held clips the holding at zero rather
    than raising: on a slider, dragging past "sell everything" means sell
    everything. The clipped amount is visible as the difference between the
    requested change and value_after - value_before.
    """
    after = {k: float(v) for k, v in values.items()}
    for isin, change in changes.items():
        after[isin] = max(after.get(isin, 0.0) + float(change), 0.0)
    return _analyse({k: float(v) for k, v in values.items()}, after, cov)


def simulate_target_weights(values: dict[str, float], cov: pd.DataFrame,
                            targets: dict[str, float]) -> SimulationResult:
    """Redistribute the current total across target weights.

    The total is held constant, so this is a rebalance and not a deposit.
    Targets are rescaled to sum to 1 (see the module docstring); a holding
    absent from `targets` is sold out. Negative targets are refused.
    """
    total = float(sum(float(v) for v in values.values()))
    scaled = _normalised_targets(targets)
    after = {k: 0.0 for k in values}
    after.update({k: t * total for k, t in scaled.items()})
    return _analyse({k: float(v) for k, v in values.items()}, after, cov)


def _normalised_targets(targets: dict[str, float]) -> dict[str, float]:
    negative = sorted(k for k, t in targets.items() if float(t) < 0)
    if negative:
        raise ValueError(f"negative target weights for {negative}")
    total = float(sum(float(t) for t in targets.values()))
    if total <= 0:
        raise ValueError("target weights sum to zero; nothing to rebalance towards")
    return {k: float(t) / total for k, t in targets.items()}


def equal_weights(keys: list[str]) -> dict[str, float]:
    """1/n each; the baseline.

    >>> equal_weights(["A", "B", "C", "D"])
    {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
    >>> equal_weights([])
    {}
    """
    n = len(keys)
    return {k: 1.0 / n for k in keys} if n else {}


def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 10_000,
                        tol: float = 1e-10) -> dict[str, float]:
    """Weights at which every holding contributes an equal share of risk.

    Solves Spinu's convex formulation

        minimise  1/2 x' Sigma x - (1/n) sum_i ln x_i     over x > 0

    whose optimum satisfies x_i (Sigma x)_i = 1/n for every i: equal risk
    contributions, and w = x / sum(x) keeps that property because the
    contributions are homogeneous in scale. Solved by cyclical coordinate
    descent: with the other coordinates fixed, the optimal x_i is the
    positive root of the quadratic

        Sigma_ii x_i^2 + (sum_{j != i} Sigma_ij x_j) x_i - 1/n = 0

    which converges monotonically because the objective is strictly convex.
    Rejected alternative: the fixed-point iteration w_i <- w_i (target /
    RC_i), which is shorter and can oscillate when correlations are
    negative.

    Stops when every risk share is within `tol` of 1/n, or after `max_iter`
    cycles, in which case the best iterate is returned; on any real ten-asset
    matrix it converges in well under a hundred. A holding with zero variance
    would take an infinite weight and is refused by name.

    On a diagonal covariance the answer is w_i proportional to 1/sigma_i.
    """
    columns = [str(c) for c in cov.columns]
    sigma = cov.to_numpy(dtype=float)
    n = len(columns)
    if n == 0:
        return {}
    variances = np.diag(sigma)
    riskless = [c for c, v in zip(columns, variances) if v <= 0]
    if riskless:
        raise ValueError(f"{riskless} have zero variance; risk parity would give them an "
                         f"infinite weight")
    budget = np.full(n, 1.0 / n)
    x = 1.0 / np.sqrt(variances)             # inverse-volatility start, exact when diagonal
    x /= x.sum()
    for _ in range(max_iter):
        for i in range(n):
            cross = float(sigma[i] @ x - sigma[i, i] * x[i])
            x[i] = (-cross + np.sqrt(cross * cross + 4.0 * sigma[i, i] * budget[i])) \
                / (2.0 * sigma[i, i])
        w = x / x.sum()
        contributions = w * (sigma @ w)
        shares = contributions / contributions.sum()
        if np.max(np.abs(shares - budget)) <= tol:
            break
    w = x / x.sum()
    return dict(zip(columns, (float(v) for v in w)))


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum w = 1} (Duchi et al., 2008).

    Sort descending, find the largest k for which u_k - (cumsum_k - 1)/k is
    positive, subtract that threshold from everything and clip at zero.

    >>> _project_simplex(np.array([0.5, 0.5, 0.5])).round(6).tolist()
    [0.333333, 0.333333, 0.333333]
    >>> _project_simplex(np.array([2.0, -1.0])).tolist()
    [1.0, 0.0]
    """
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u)
    ranks = np.arange(1, v.size + 1)
    positive = np.nonzero(u - (cumulative - 1.0) / ranks > 0)[0]
    k = positive[-1] + 1
    theta = (cumulative[k - 1] - 1.0) / k
    return np.maximum(v - theta, 0.0)


def min_variance_weights(cov: pd.DataFrame, max_iter: int = 10_000) -> dict[str, float]:
    """The long-only, fully invested portfolio with the smallest variance.

        minimise  w' Sigma w     subject to  w >= 0, sum w = 1

    Projected gradient descent from equal weights: step against the
    gradient 2 Sigma w with step size 1 / lambda_max(Sigma) -- the largest
    step for which the descent is guaranteed on a quadratic -- and project
    back onto the simplex. The problem is convex, so this converges to the
    global optimum; it stops when the weights move by less than 1e-12 or
    after `max_iter` steps. Rejected alternative: the closed form
    Sigma^-1 1 / (1' Sigma^-1 1), which is the unconstrained answer and
    routinely shorts the most volatile holding.

    On a diagonal covariance the answer is w_i proportional to 1 / Sigma_ii.
    A covariance with no variance at all (all zeros) has nothing to minimise
    and returns equal weights.
    """
    columns = [str(c) for c in cov.columns]
    sigma = cov.to_numpy(dtype=float)
    n = len(columns)
    if n == 0:
        return {}
    lam_max = float(np.linalg.eigvalsh(sigma)[-1])
    if lam_max <= 0:
        return equal_weights(columns)
    step = 1.0 / lam_max
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        w_next = _project_simplex(w - step * (sigma @ w))
        moved = float(np.max(np.abs(w_next - w)))
        w = w_next
        if moved < 1e-12:
            break
    return dict(zip(columns, (float(v) for v in w)))


@dataclasses.dataclass(frozen=True)
class Trade:
    isin: str
    action: str             # "BUY" | "SELL"
    amount: float           # base currency, positive
    units: float | None     # amount / price when a price is known
    weight_before: float
    weight_after: float


def trades_to_target(values: dict[str, float], targets: dict[str, float],
                     prices: dict[str, float] | None = None,
                     total: float | None = None, min_amount: float = 0.0) -> list[Trade]:
    """The trades that move the book from `values` to `targets`.

        amount_i = target_i x total - value_i

    BUY when positive, SELL when negative; a holding absent from `targets`
    is sold out, an ISIN absent from `values` is bought from nothing. `total`
    defaults to the sum of `values`, so the default is a pure rebalance;
    pass a larger total to rebalance and deposit at once. Targets are
    rescaled to sum to 1. Trades below `min_amount` are dropped, because a
    3-euro rebalance costs more in fees than it moves; `weight_after` is
    still the target, not the weight after the dropped trades. Units are
    amount / price where a positive price is known. Largest trade first.
    """
    prices = prices or {}
    before_total = float(sum(float(v) for v in values.values()))
    total = before_total if total is None else float(total)
    scaled = _normalised_targets(targets)
    keys = list(dict.fromkeys([*values, *scaled]))
    trades: list[Trade] = []
    for isin in keys:
        value = float(values.get(isin, 0.0))
        target = scaled.get(isin, 0.0)
        amount = target * total - value
        if amount == 0 or abs(amount) < min_amount:
            continue
        price = prices.get(isin)
        units = abs(amount) / float(price) if price is not None and float(price) > 0 else None
        trades.append(Trade(
            isin=isin, action="BUY" if amount > 0 else "SELL", amount=abs(amount),
            units=units,
            weight_before=value / before_total if before_total > 0 else 0.0,
            weight_after=target,
        ))
    return sorted(trades, key=lambda t: (-t.amount, t.isin))
