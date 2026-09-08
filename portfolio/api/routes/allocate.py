"""Where new money should go: a small POST, answered from the cached analysis.

The route computes nothing. It selects the holdings new money may enter,
builds the cost model from the instrument records, and hands both to
`agents.allocate`, which is the same code the CLI runs and the same code the
replay evaluates. A second implementation for the screen would be a second
answer to defend.

Two things it does decide, because they are questions about *this* account
rather than about the optimisation:

`buyable` comes from the instrument record and is deliberately not
`tradeable`. A holding at a second broker cannot be rebalanced against the
rest of the book and can be bought with new money perfectly well.

The covariance is the analysis's own -- the same matrix the risk page and the
simulator show -- so the allocator cannot quietly disagree with the rest of
the screen about what the book's risk looks like.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...agents.allocate import (allocate_buy_only, best_reachable_dispersion,
                                cash_for_dispersion, pinned_holdings)
from ...agents.execution import CostModel, cost_table
from .. import schemas as S
from ..services import PortfolioService
from ._deps import service

router = APIRouter(tags=["allocate"])

MAX_AMOUNT = 100_000_000.0


@router.post("/allocate", response_model=S.AllocationOut)
def allocate(req: S.AllocateRequest,
             svc: PortfolioService = Depends(service)) -> S.AllocationOut:
    analysis = svc.analysis()
    if analysis.cov is None:
        raise HTTPException(status_code=409, detail=(
            analysis.snapshot.risk.reason
            or "No risk model is available to allocate against."))
    if not (0 < req.amount <= MAX_AMOUNT):
        raise HTTPException(status_code=400, detail=(
            f"An amount between 0 and {MAX_AMOUNT:,.0f} is needed."))

    keys = [str(c) for c in analysis.cov.columns]
    values = {k: float(analysis.values.get(k, 0.0)) for k in keys}
    prices = {k: float(analysis.prices_base[k]) for k in keys
              if analysis.prices_base.get(k)}
    if sum(values.values()) <= 0:
        raise HTTPException(status_code=409, detail=(
            "Nothing is held, so there is no risk structure to even out. Any "
            "first purchase is as good as any other."))

    instruments = {k: analysis.instruments[k] for k in keys
                   if k in analysis.instruments}
    costs = CostModel(account_value=sum(values.values()) or 1.0,
                      per_instrument=cost_table(instruments))
    buyable = {k for k, inst in instruments.items() if inst.buyable}

    result = allocate_buy_only(values=values, prices=prices, cov=analysis.cov,
                               cash=float(req.amount), costs=costs,
                               buyable=buyable)

    shared = dict(values=values, cov=analysis.cov, costs=costs,
                  buyable=buyable)
    pinned = pinned_holdings(**shared)
    needed = best = best_at = None
    if req.target is not None:
        needed = cash_for_dispersion(float(req.target), **shared)
        if needed is None:
            # With a pinned holding the floor turns and climbs, so there is a
            # best amount rather than "as much as possible". A screen that
            # only said "unreachable" would leave the reader to guess it.
            best, best_at = best_reachable_dispersion(**shared)

    def label(isin: str) -> str:
        return analysis.short_names.get(isin) or analysis.names.get(isin) or isin

    return S.AllocationOut(
        cash=result.cash, invested=result.invested, leftover=result.leftover,
        book_value=float(sum(values.values())),
        purchases=[S.AllocationPurchase(
            isin=p.isin, name=label(p.isin), broker=p.broker, shares=p.shares,
            price=p.price, amount=p.amount, cost=p.cost,
            weight_before=p.weight_before, weight_after=p.weight_after,
            risk_before=p.risk_before, risk_after=p.risk_after)
            for p in result.purchases],
        destinations=[S.AllocationDestination(
            isin=d.isin, name=label(d.isin), dispersion=d.dispersion,
            cost=d.cost, improvement=d.improvement, per_euro=d.per_euro)
            for d in result.destinations],
        refused=[S.AllocationRefusal(isin=r.isin, name=label(r.isin),
                                     reason=r.reason)
                 for r in result.refused],
        dispersion_now=result.dispersion_now,
        dispersion_after=result.dispersion_after,
        spread_now=result.spread_now, spread_after=result.spread_after,
        floor_at_cash=result.floor_at_cash,
        floor_unlimited=result.floor_unlimited,
        rounding_penalty=result.rounding_penalty,
        closable=result.closable,
        cost_parts=result.cost_parts, total_cost=result.total_cost,
        meaningful_cash=result.meaningful_cash,
        best_worst_gap=result.best_worst_gap,
        assumptions=list(result.assumptions),
        target=req.target, cash_for_target=needed,
        best_reachable=best, best_reachable_cash=best_at,
        pinned=[label(k) for k in pinned])
