"""Health, settings and the demo/live switch."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...data.store import DataMode
from .. import schemas as S
from ..config import VERSION
from ..services import PortfolioService
from ._deps import service

router = APIRouter(tags=["settings"])


@router.get("/health", response_model=S.Health)
def health(svc: PortfolioService = Depends(service)) -> S.Health:
    return S.Health(status="ok", version=VERSION, mode=svc.mode.value,
                    provider=svc.provider.name)


def _settings(svc: PortfolioService) -> S.Settings:
    stats = svc.cache.stats()
    return S.Settings(
        mode=svc.mode.value, data_dir=str(svc.store().directory),
        provider=svc.provider.name, base_currency=svc.config.base_currency,
        cache=S.CacheStats(path=str(svc.cache.path), symbols=stats.symbols,
                           rows=stats.rows, size_bytes=stats.size_bytes,
                           oldest=stats.oldest.isoformat() if stats.oldest else None,
                           newest=stats.newest.isoformat() if stats.newest else None),
        last_refresh=svc.last_refresh.isoformat() if svc.last_refresh else None,
        version=VERSION,
        quote_ttl_minutes=int(svc.config.quote_ttl.total_seconds() // 60),
        lookback=svc.config.lookback)


@router.get("/settings", response_model=S.Settings)
def get_settings(svc: PortfolioService = Depends(service)) -> S.Settings:
    return _settings(svc)


@router.put("/settings", response_model=S.Settings)
def put_settings(patch: S.SettingsPatch,
                 svc: PortfolioService = Depends(service)) -> S.Settings:
    if patch.mode is not None:
        svc.set_mode(DataMode(patch.mode))
    return _settings(svc)


@router.post("/cache/clear", response_model=S.Settings)
def clear_cache(svc: PortfolioService = Depends(service)) -> S.Settings:
    svc.cache.clear()
    svc.invalidate()
    return _settings(svc)
