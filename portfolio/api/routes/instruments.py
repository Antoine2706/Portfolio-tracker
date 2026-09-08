"""The instrument universe: resolve, confirm, save; edit; deactivate; delete.

The add flow is resolve -> confirm -> save, never a text box that writes.
Nothing is stored until the client has posted the exact listing the user
saw and chose.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...core.models import AssetClass, ValidationError, is_valid_isin
from ...core.universe import check_deletable, deactivate, reactivate
from ...data.resolve import Candidate, Resolution, Verdict, instrument_from_candidate
from ...data.venues import yahoo_suffix_to_mic
from .. import builder
from .. import schemas as S
from ..services import PortfolioService
from ._deps import service

router = APIRouter(prefix="/instruments", tags=["instruments"])


def _candidate_out(c: Candidate) -> S.CandidateOut:
    return S.CandidateOut(
        isin=c.isin, ticker=c.ticker, yahoo_symbol=c.yahoo_symbol,
        venue_label=c.venue_label, mic=c.mic, verdict=c.verdict.value,
        selectable=c.verdict is not Verdict.FAILED and c.verdict is not Verdict.REFUSED,
        observations=c.observations,
        first_date=c.first_date.isoformat() if c.first_date else None,
        last_date=c.last_date.isoformat() if c.last_date else None,
        currency=c.currency, reported_exchange=c.reported_exchange, name=c.name,
        reasons=list(c.reasons), description=c.describe())


def _resolution_out(r: Resolution) -> S.ResolutionOut:
    return S.ResolutionOut(
        isin=r.isin, candidates=[_candidate_out(c) for c in r.candidates],
        refused=[_candidate_out(c) for c in r.refused],
        recommended=r.recommended.yahoo_symbol if r.recommended else None,
        blocked=r.blocked, block_reason=r.block_reason(), summary=r.summary(),
        listings_seen=r.listings_seen, filtered_by_class=dict(r.filtered_by_class),
        errors=list(r.errors))


@router.get("", response_model=list[S.InstrumentOut])
def list_instruments(svc: PortfolioService = Depends(service)) -> list[S.InstrumentOut]:
    store = svc.store()
    transactions = store.load_transactions()
    return [builder.instrument_out(inst, transactions)
            for inst in sorted(store.load_instruments().values(), key=lambda i: i.name)]


@router.post("/resolve", response_model=S.ResolutionOut)
def resolve(req: S.ResolveRequest, svc: PortfolioService = Depends(service)) -> S.ResolutionOut:
    isin = req.isin.strip().upper()
    if not is_valid_isin(isin):
        raise HTTPException(status_code=400, detail=(
            f"{isin!r} is not a valid ISIN: twelve characters with a correct check "
            f"digit. A typo here would create a second, empty instrument rather than "
            f"failing loudly."))
    existing = svc.store().load_instruments()
    if isin in existing:
        raise HTTPException(status_code=409, detail=(
            f"{isin} is already in your universe as {existing[isin].name}."))
    return _resolution_out(svc.resolve(isin, req.lookback))


@router.post("", response_model=S.InstrumentOut, status_code=201)
def save(req: S.SaveInstrumentRequest, svc: PortfolioService = Depends(service)) -> S.InstrumentOut:
    isin = req.isin.strip().upper()
    if not is_valid_isin(isin):
        raise HTTPException(status_code=400, detail=f"{isin!r} is not a valid ISIN")
    store = svc.store()
    existing = store.load_instruments()
    if isin in existing:
        raise HTTPException(status_code=409, detail=f"{isin} already exists")

    # The posted listing must be one the resolver offered for this ISIN. A
    # client cannot save a symbol the user never saw on the confirm screen.
    resolution = svc.resolve(isin)
    offered = {c.yahoo_symbol: c for c in resolution.candidates}
    chosen = offered.get(req.yahoo_symbol)
    if chosen is None:
        raise HTTPException(status_code=400, detail=(
            f"{req.yahoo_symbol} was not among the listings resolved for {isin}. "
            f"Resolve again and choose one of the offered listings."))
    if chosen.verdict in (Verdict.FAILED, Verdict.REFUSED):
        raise HTTPException(status_code=400, detail=(
            f"{req.yahoo_symbol} is {chosen.verdict.value} and cannot be saved: "
            + " ".join(chosen.reasons)))
    try:
        inst = instrument_from_candidate(
            chosen, svc.provider.name, base_currency=req.base_currency.strip().upper(),
            name=req.name.strip(), issuer=req.issuer.strip(),
            asset_class=AssetClass(req.asset_class).value)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    existing[inst.isin] = inst
    store.save_instruments(existing)
    svc.invalidate()
    return builder.instrument_out(inst, store.load_transactions())


@router.patch("/{isin}", response_model=S.InstrumentOut)
def patch(isin: str, req: S.InstrumentPatch,
          svc: PortfolioService = Depends(service)) -> S.InstrumentOut:
    store = svc.store()
    instruments = store.load_instruments()
    inst = instruments.get(isin.strip().upper())
    if inst is None:
        raise HTTPException(status_code=404, detail=f"{isin} is not in the universe")
    try:
        if req.name is not None:
            if not req.name.strip():
                raise ValidationError("an instrument needs a name")
            inst.override("name", req.name.strip())
        if req.short_name is not None:
            short = req.short_name.strip()
            if short:
                inst.override("short_name", short)
            else:
                # Clearing returns the label to its derived form. Storing a
                # blank as a manual override would pin the instrument to
                # having no label at all, which is never what clearing a
                # field means.
                inst.short_name = ""
                inst.manual_overrides.discard("short_name")
        if req.issuer is not None:
            inst.override("issuer", req.issuer.strip())
        if req.asset_class is not None:
            inst.override("asset_class", AssetClass(req.asset_class))
        if req.base_currency is not None:
            inst.override("base_currency", req.base_currency.strip().upper())
        if req.note is not None:
            inst.note = req.note
        if req.provider_symbols:
            for provider, symbol in req.provider_symbols.items():
                symbol = symbol.strip()
                if not symbol:
                    raise ValidationError(f"empty symbol for {provider}")
                inst.override(f"provider_symbols.{provider.strip()}", symbol)
                if provider.strip() == svc.provider.name:
                    inst.exchange = yahoo_suffix_to_mic(symbol) or inst.exchange
        if req.active is not None:
            if req.active:
                reactivate(inst)
            else:
                deactivate(inst, "hidden from holdings")
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.save_instruments(instruments)
    svc.invalidate()
    return builder.instrument_out(inst, store.load_transactions())


@router.delete("/{isin}", status_code=204)
def delete(isin: str, svc: PortfolioService = Depends(service)) -> None:
    store = svc.store()
    instruments = store.load_instruments()
    key = isin.strip().upper()
    if key not in instruments:
        raise HTTPException(status_code=404, detail=f"{isin} is not in the universe")
    check = check_deletable(key, store.load_transactions())
    if not check.allowed:
        raise HTTPException(status_code=409, detail=f"{check.reason} {check.alternative}")
    del instruments[key]
    store.save_instruments(instruments)
    svc.invalidate()
