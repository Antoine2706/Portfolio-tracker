"""The snapshot, the holding detail, the refresh, the benchmarks."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...data.benchmarks import BENCHMARKS
from .. import builder
from .. import schemas as S
from ..services import PortfolioService
from ._deps import service

router = APIRouter(tags=["snapshot"])


@router.get("/snapshot", response_model=S.Snapshot)
def snapshot(benchmark: str | None = Query(default=None),
             lookback: int | None = Query(default=None, ge=2, le=2000),
             svc: PortfolioService = Depends(service)) -> S.Snapshot:
    return svc.analysis(benchmark, lookback).snapshot


@router.post("/refresh", response_model=S.Snapshot)
def refresh(benchmark: str | None = Query(default=None),
            lookback: int | None = Query(default=None, ge=2, le=2000),
            svc: PortfolioService = Depends(service)) -> S.Snapshot:
    svc.refresh()
    return svc.analysis(benchmark, lookback).snapshot


@router.get("/holdings/{isin}", response_model=S.HoldingDetail)
def holding(isin: str, benchmark: str | None = Query(default=None),
            lookback: int | None = Query(default=None, ge=2, le=2000),
            svc: PortfolioService = Depends(service)) -> S.HoldingDetail:
    analysis = svc.analysis(benchmark, lookback)
    detail = builder.holding_detail(analysis, isin.strip().upper(),
                                    svc.store().load_amendments())
    if detail is None:
        raise HTTPException(status_code=404, detail=f"{isin} is not in the universe")
    return detail


@router.get("/benchmarks", response_model=list[S.Benchmark])
def benchmarks() -> list[S.Benchmark]:
    return [S.Benchmark(symbol=b.symbol, name=b.name, index=b.index, label=b.label,
                        currency=b.currency, note=b.note, is_default=b is BENCHMARKS[0])
            for b in BENCHMARKS]
