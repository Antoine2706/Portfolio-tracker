"""The what-if simulator: a small POST, answered from the cached analysis.

Every number comes from `core.simulate`. This route selects the covariance
columns to use (held instruments plus any watchlist instruments the client
asks to include) and projects the result.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...core.returns import InsufficientHistory, align_returns
from ...core.risk import covariance_matrix
from ...core.simulate import (equal_weights, min_variance_weights, risk_parity_weights,
                              simulate_cash_changes, simulate_target_weights,
                              trades_to_target)
from .. import schemas as S
from ..services import PortfolioService
from ._deps import service

router = APIRouter(tags=["simulate"])


@router.post("/simulate", response_model=S.SimulationOut)
def simulate(req: S.SimulateRequest, svc: PortfolioService = Depends(service)) -> S.SimulationOut:
    analysis = svc.analysis()
    if analysis.cov is None or analysis.returns is None:
        raise HTTPException(status_code=409, detail=(
            analysis.snapshot.risk.reason or "No risk model is available to simulate against."))

    modelled = [c for c in analysis.cov.columns]
    extra = [i.strip().upper() for i in req.include if i.strip().upper() not in modelled]
    cov = analysis.cov
    if extra:
        series = {isin: analysis.market.histories[isin] for isin in modelled + extra
                  if isin in analysis.market.histories}
        missing = [i for i in extra if i not in series]
        if missing:
            raise HTTPException(status_code=400, detail=(
                f"No price history for {', '.join(missing)}; it cannot enter the model."))
        try:
            returns, _ = align_returns(series, lookback=analysis.lookback)
        except InsufficientHistory as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        cov = covariance_matrix(returns)

    values = {isin: analysis.values.get(isin, 0.0) for isin in cov.columns}
    if sum(values.values()) <= 0:
        raise HTTPException(status_code=409, detail="nothing is held; nothing to simulate")

    keys = list(cov.columns)
    if req.preset == "equal":
        result = simulate_target_weights(values, cov, equal_weights(keys))
    elif req.preset == "risk_parity":
        result = simulate_target_weights(values, cov, risk_parity_weights(cov))
    elif req.preset == "min_variance":
        result = simulate_target_weights(values, cov, min_variance_weights(cov))
    elif req.targets:
        targets = {k.strip().upper(): float(v) for k, v in req.targets.items()}
        unknown = [k for k in targets if k not in values]
        if unknown:
            raise HTTPException(status_code=400, detail=(
                f"{', '.join(unknown)} not in the model; pass it in `include` first."))
        if any(v < 0 for v in targets.values()):
            raise HTTPException(status_code=400, detail="target weights cannot be negative")
        total = sum(targets.values())
        if total <= 0:
            raise HTTPException(status_code=400, detail="target weights sum to zero")
        targets = {k: v / total for k, v in targets.items()}
        for k in values:
            targets.setdefault(k, 0.0)
        result = simulate_target_weights(values, cov, targets)
    elif req.changes:
        changes = {k.strip().upper(): float(v) for k, v in req.changes.items()}
        unknown = [k for k in changes if k not in values]
        if unknown:
            raise HTTPException(status_code=400, detail=(
                f"{', '.join(unknown)} not in the model; pass it in `include` first."))
        result = simulate_cash_changes(values, cov, changes)
    else:
        raise HTTPException(status_code=400, detail="give changes, targets or a preset")

    after_values = {h.isin: h.value_after for h in result.holdings}
    after_total = sum(after_values.values())
    targets_after = ({k: v / after_total for k, v in after_values.items()}
                     if after_total > 0 else {k: 0.0 for k in after_values})
    trades = trades_to_target(values, targets_after, prices=analysis.prices_base,
                              total=after_total, min_amount=req.min_trade)
    names = analysis.names
    return S.SimulationOut(
        holdings=[S.SimulatedHolding(
            isin=h.isin, name=names.get(h.isin, h.isin), value_before=h.value_before,
            value_after=h.value_after, weight_before=h.weight_before,
            weight_after=h.weight_after, risk_before=h.risk_before,
            risk_after=h.risk_after, marginal_after=h.marginal_after)
            for h in result.holdings],
        total_before=result.total_before, total_after=result.total_after,
        volatility_before=result.volatility_before, volatility_after=result.volatility_after,
        effective_before=result.effective_before, effective_after=result.effective_after,
        diversification_before=result.diversification_before,
        diversification_after=result.diversification_after,
        max_risk_share_before=result.max_risk_share_before,
        max_risk_share_after=result.max_risk_share_after,
        trades=[S.Trade(isin=t.isin, name=names.get(t.isin, t.isin), action=t.action,
                        amount=t.amount, units=t.units,
                        price=analysis.prices_base.get(t.isin),
                        weight_before=t.weight_before, weight_after=t.weight_after)
                for t in trades],
        weights_after=targets_after)
