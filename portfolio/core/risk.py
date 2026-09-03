"""Portfolio risk mathematics, written out rather than delegated.

Every formula here is implemented directly in numpy so it can be read against a
textbook and verified by hand. No library risk function is used, deliberately:
an opaque `.risk()` call cannot be checked, and the whole point of this tool is
that the numbers are inspectable.

Notation throughout:

    w   vector of portfolio weights, length n, summing to 1
    Σ   sample covariance matrix of daily simple returns, n x n
    σ_p portfolio daily volatility, a scalar
    σ_i standalone daily volatility of asset i (the square root of Σ_ii)

The invariant that anchors the whole module
-------------------------------------------
Component contributions to risk sum exactly to portfolio volatility:

    Σ_i CCTR_i = σ_p

This is Euler's theorem applied to σ_p(w), which is homogeneous of degree 1 in
w. It is not a modelling choice, it is an identity, so if the test asserting it
ever fails the code is wrong. That test is the single best check in the suite.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .returns import TRADING_DAYS_PER_YEAR

__all__ = [
    "RiskDecomposition", "ConcentrationStats", "DrawdownStats", "CorrelationPair",
    "covariance_matrix", "portfolio_volatility", "annualise_volatility",
    "risk_decomposition", "diversification_ratio", "concentration",
    "correlation_matrix", "high_correlation_pairs", "drawdown", "beta",
    "HIGH_CORRELATION_THRESHOLD",
]

# Above this, two holdings are likely the same exposure wearing two tickers.
HIGH_CORRELATION_THRESHOLD = 0.90


def _as_weight_vector(weights, columns) -> np.ndarray:
    """Weights as an array ordered to match the covariance matrix columns.

    Ordering is the classic silent bug here: a dict iterated in insertion order
    against a DataFrame in alphabetical order gives a plausible number for the
    wrong portfolio. So the mapping is always applied by name.
    """
    if isinstance(weights, dict):
        missing = [c for c in columns if c not in weights]
        if missing:
            raise ValueError(f"no weight supplied for {missing}")
        w = np.array([float(weights[c]) for c in columns], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(columns),):
            raise ValueError(
                f"weights has shape {w.shape}, expected ({len(columns)},) to match "
                f"the covariance matrix")
    return w


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance of daily returns, Bessel-corrected (ddof=1).

    ddof=1 because these are a sample of returns, not the population. With 252
    observations the difference is under half a percent, but the estimator
    should be the right one regardless.
    """
    if returns.shape[0] < 2:
        raise ValueError(
            f"covariance needs at least 2 return observations, got {returns.shape[0]}")
    if returns.shape[1] == 0:
        raise ValueError("covariance needs at least one instrument")
    return returns.astype(float).cov(ddof=1)


def portfolio_volatility(weights, cov: pd.DataFrame) -> float:
    """σ_p = sqrt(wᵀ Σ w), the daily volatility of the weighted portfolio."""
    w = _as_weight_vector(weights, cov.columns)
    variance = float(w @ cov.to_numpy() @ w)
    # Numerically, a tiny negative can appear for a near-singular Σ.
    return float(np.sqrt(max(variance, 0.0)))


def annualise_volatility(daily_vol: float,
                         trading_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """σ_annual = σ_daily x sqrt(252).

    The square-root-of-time rule assumes returns are serially uncorrelated. That
    is roughly true for daily equity returns and is the standard convention; it
    understates risk where returns trend and overstates it where they mean-revert.
    """
    return float(daily_vol * np.sqrt(trading_days))


@dataclasses.dataclass(frozen=True)
class RiskDecomposition:
    """Where the portfolio's risk actually comes from.

    The divergence between `weight` and `pct_contribution` is the most useful
    output of the whole application: a holding at 10% of capital contributing
    25% of risk is the thing a pie chart of weights cannot show you.
    """
    instruments: tuple[str, ...]
    weights: np.ndarray
    portfolio_volatility: float          # daily
    marginal: np.ndarray                 # MCTR, dσ_p/dw_i
    component: np.ndarray                # CCTR_i = w_i x MCTR_i
    percent: np.ndarray                  # CCTR_i / σ_p

    def as_frame(self) -> pd.DataFrame:
        """Table for the risk view: capital weight beside risk share."""
        return pd.DataFrame({
            "weight": self.weights,
            "marginal_ctr": self.marginal,
            "component_ctr": self.component,
            "pct_of_risk": self.percent,
        }, index=pd.Index(self.instruments, name="isin"))

    def check_invariant(self, tolerance: float = 1e-10) -> bool:
        """CCTR must sum to σ_p. An identity, not an approximation."""
        return bool(abs(self.component.sum() - self.portfolio_volatility) <= tolerance)


def risk_decomposition(weights, cov: pd.DataFrame) -> RiskDecomposition:
    """Marginal, component and percentage contributions to risk.

        MCTR = (Σ w) / σ_p          the partial derivative dσ_p/dw_i
        CCTR_i = w_i x MCTR_i       that asset's share of σ_p
        pct_i  = CCTR_i / σ_p       which sums to 1

    MCTR answers "if I add a euro to this holding, how does portfolio
    volatility move" -- which is the question the v2 what-if simulator asks,
    and the reason a volatile but weakly-correlated asset can *reduce* total
    risk: its MCTR can be lower than its own σ_i.
    """
    w = _as_weight_vector(weights, cov.columns)
    sigma = cov.to_numpy()
    sigma_p = portfolio_volatility(w, cov)

    if sigma_p == 0:
        # A zero-variance portfolio has no risk to attribute. Returning zeros
        # keeps the invariant true (0 == 0) rather than dividing by zero.
        zeros = np.zeros_like(w)
        return RiskDecomposition(tuple(cov.columns), w, 0.0, zeros, zeros, zeros)

    marginal = (sigma @ w) / sigma_p
    component = w * marginal
    percent = component / sigma_p
    return RiskDecomposition(tuple(cov.columns), w, sigma_p, marginal, component, percent)


def diversification_ratio(weights, cov: pd.DataFrame) -> float:
    """DR = (Σ_i w_i σ_i) / σ_p.

    The weighted average of standalone volatilities divided by the volatility
    actually realised. Equal to 1 for a single asset, or for a set of perfectly
    correlated ones; rises as holdings genuinely offset each other.

    Plain-language gloss for the UI: "your holdings are individually X times as
    volatile as the portfolio they make up -- the difference is diversification."
    """
    w = _as_weight_vector(weights, cov.columns)
    sigma_i = np.sqrt(np.diag(cov.to_numpy()))
    sigma_p = portfolio_volatility(w, cov)
    if sigma_p == 0:
        return 1.0
    return float((w @ sigma_i) / sigma_p)


@dataclasses.dataclass(frozen=True)
class ConcentrationStats:
    herfindahl: float
    effective_holdings: float
    actual_holdings: int

    def summary(self) -> str:
        return (f"{self.actual_holdings} holdings, but an effective number of "
                f"{self.effective_holdings:.1f}")


def concentration(weights) -> ConcentrationStats:
    """Herfindahl index and the effective number of holdings.

        HHI = Σ w_i²
        N_eff = 1 / HHI

    N_eff is the headline: ten holdings with an effective number of three says
    the portfolio behaves like a three-stock portfolio, which is the insight a
    weights pie chart hides.

    >>> round(concentration([0.25, 0.25, 0.25, 0.25]).effective_holdings, 6)
    4.0
    >>> concentration([0.9, 0.05, 0.05]).herfindahl        # 0.81 + 0.0025 + 0.0025
    0.815
    >>> round(concentration([0.9, 0.05, 0.05]).effective_holdings, 3)
    1.227
    """
    w = np.asarray(list(weights.values()) if isinstance(weights, dict) else weights,
                   dtype=float)
    if w.size == 0:
        return ConcentrationStats(0.0, 0.0, 0)
    hhi = float(np.sum(w ** 2))
    return ConcentrationStats(hhi, float(1.0 / hhi) if hhi > 0 else 0.0, int(w.size))


def correlation_matrix(cov: pd.DataFrame) -> pd.DataFrame:
    """Normalise Σ to correlations: ρ_ij = Σ_ij / (σ_i σ_j).

    Derived from the covariance already computed rather than recomputed from
    returns, so the heatmap and the risk numbers cannot disagree.
    """
    sigma_i = np.sqrt(np.diag(cov.to_numpy()))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov.to_numpy() / np.outer(sigma_i, sigma_i)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


@dataclasses.dataclass(frozen=True)
class CorrelationPair:
    a: str
    b: str
    correlation: float

    def sentence(self, names: dict[str, str] | None = None) -> str:
        """Plain sentence naming both funds, for the risk view."""
        names = names or {}
        na, nb = names.get(self.a, self.a), names.get(self.b, self.b)
        return (f"{na} and {nb} move together {self.correlation:.0%} of the time. "
                f"They are likely holding much the same underlying companies, so "
                f"owning both diversifies less than owning two tickers suggests.")


def high_correlation_pairs(corr: pd.DataFrame,
                           threshold: float = HIGH_CORRELATION_THRESHOLD,
                           ) -> list[CorrelationPair]:
    """Pairs above the threshold, most correlated first.

    The motivating case: two European defence ETFs that plausibly hold nearly
    identical underlying stocks. The tool should say so rather than let two
    tickers read as diversification.
    """
    out: list[CorrelationPair] = []
    cols = list(corr.columns)
    values = corr.to_numpy()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = float(values[i, j])
            if rho >= threshold:
                out.append(CorrelationPair(cols[i], cols[j], rho))
    return sorted(out, key=lambda p: p.correlation, reverse=True)


@dataclasses.dataclass(frozen=True)
class DrawdownStats:
    max_drawdown: float                  # negative, e.g. -0.23 for -23%
    max_drawdown_peak: pd.Timestamp | None
    max_drawdown_trough: pd.Timestamp | None
    current_drawdown: float              # negative or zero
    current_peak: pd.Timestamp | None


def drawdown(values: pd.Series) -> DrawdownStats:
    """Maximum and current drawdown of a portfolio value series.

    Drawdown at t is value_t / running_max_t - 1, so it is zero at a new high
    and negative below one. Both figures matter: the maximum says how bad it has
    been, the current says where you are now.
    """
    v = values.dropna().astype(float)
    if v.empty:
        return DrawdownStats(0.0, None, None, 0.0, None)
    running_max = v.cummax()
    dd = v / running_max - 1.0
    trough = dd.idxmin()
    max_dd = float(dd.loc[trough])
    # The peak is the last date at or before the trough where a new high was set.
    peak = v.loc[:trough].idxmax() if max_dd < 0 else None
    return DrawdownStats(
        max_drawdown=max_dd,
        max_drawdown_peak=peak,
        max_drawdown_trough=trough if max_dd < 0 else None,
        current_drawdown=float(dd.iloc[-1]),
        current_peak=v.idxmax(),
    )


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """β = Cov(r_p, r_b) / Var(r_b), on the dates the two series share.

    The caller is responsible for stating which benchmark this is; a beta
    without a named benchmark is meaningless, so the UI must display the
    benchmark alongside the number.
    """
    # sort=True is explicit: pandas is deprecating the implicit default, and a
    # beta computed over out-of-order dates would be wrong rather than merely noisy.
    joined = pd.concat([portfolio_returns.rename("p"),
                        benchmark_returns.rename("b")], axis=1, sort=True).dropna()
    if len(joined) < 2:
        raise ValueError(
            f"beta needs at least 2 overlapping observations, got {len(joined)}")
    var_b = float(joined["b"].var(ddof=1))
    if var_b == 0:
        raise ValueError("benchmark has zero variance; beta is undefined")
    return float(joined["p"].cov(joined["b"], ddof=1) / var_b)
