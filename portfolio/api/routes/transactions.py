"""The ledger: append, void, import, export. Nothing here rewrites a row."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal, InvalidOperation

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from ...core.models import Transaction, TransactionType, ValidationError
from ...core.positions import InsufficientUnits, derive_positions
from ...data.export import holdings_csv, transactions_csv, workbook
from ...data.importers import ColumnMapping, mark_duplicates, preview_import
from .. import builder
from .. import schemas as S
from ..services import PortfolioService
from ._deps import service

router = APIRouter(tags=["transactions"])


def _names(svc: PortfolioService) -> dict[str, str]:
    return {isin: inst.name for isin, inst in svc.store().load_instruments().items()}


@router.get("/transactions", response_model=list[S.TransactionOut])
def list_transactions(include_voided: bool = Query(default=False),
                      svc: PortfolioService = Depends(service)) -> list[S.TransactionOut]:
    store = svc.store()
    names = _names(svc)
    voided_by = {a.target_id: a for a in store.load_amendments()}
    rows = store.load_transactions(include_voided=include_voided)
    return [builder.transaction_out(t, names, voided_by)
            for t in sorted(rows, key=lambda x: (x.date, x.id), reverse=True)]


def _parse(req: S.TransactionIn) -> Transaction:
    try:
        return Transaction(
            date=dt.date.fromisoformat(req.date), isin=req.isin.strip().upper(),
            type=TransactionType(req.type.strip().upper()),
            quantity=Decimal(str(req.quantity or "0")),
            price_per_unit=Decimal(str(req.price_per_unit or "0")),
            currency=(req.currency or "EUR").strip(),
            fees=Decimal(str(req.fees or "0")), note=req.note or "")
    except (ValidationError, InvalidOperation, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/transactions", response_model=S.TransactionOut, status_code=201)
def add_transaction(req: S.TransactionIn,
                    svc: PortfolioService = Depends(service)) -> S.TransactionOut:
    store = svc.store()
    instruments = store.load_instruments()
    txn = _parse(req)
    if txn.isin not in instruments:
        raise HTTPException(status_code=400, detail=(
            f"{txn.isin} is not in your universe. Add the instrument first so the "
            f"ledger is never keyed on an unresolved ISIN."))
    # Replaying the ledger with the new row catches a SELL of more units than
    # are held before it is written, rather than after it has corrupted every
    # later position.
    try:
        derive_positions(store.load_transactions() + [txn], strict=False)
    except InsufficientUnits as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.append_transaction(txn)
    svc.invalidate()
    return builder.transaction_out(txn, _names(svc))


@router.post("/transactions/{txn_id}/void", response_model=S.TransactionOut)
def void_transaction(txn_id: str, req: S.VoidRequest,
                     svc: PortfolioService = Depends(service)) -> S.TransactionOut:
    store = svc.store()
    everything = {t.id: t for t in store.load_transactions(include_voided=True)}
    if txn_id not in everything:
        raise HTTPException(status_code=404, detail=f"no transaction {txn_id}")
    live = {t.id for t in store.load_transactions()}
    if txn_id not in live:
        raise HTTPException(status_code=409, detail=f"{txn_id} is already voided")
    remaining = [t for t in store.load_transactions() if t.id != txn_id]
    try:
        derive_positions(remaining, strict=False)
    except InsufficientUnits as exc:
        raise HTTPException(status_code=409, detail=(
            f"Voiding this entry would leave a sell without the units it sold: "
            f"{exc} Void the later sell first.")) from exc
    amendment = store.void_transaction(txn_id, reason=req.reason)
    svc.invalidate()
    return builder.transaction_out(everything[txn_id], _names(svc),
                                   {amendment.target_id: amendment})


def _mapping(m: S.ColumnMappingIn | None) -> ColumnMapping | None:
    if m is None:
        return None
    return ColumnMapping(date=m.date, isin=m.isin, type=m.type, quantity=m.quantity,
                         price=m.price, currency=m.currency, fees=m.fees, note=m.note,
                         date_format=m.date_format, decimal_comma=m.decimal_comma)


def _preview(req: S.ImportRequest, svc: PortfolioService):
    store = svc.store()
    known = set(store.load_instruments())
    try:
        preview = preview_import(req.text, _mapping(req.mapping), known=known)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return mark_duplicates(preview, store.load_transactions())


def _preview_out(preview, names: dict[str, str]) -> S.ImportPreviewOut:
    m = preview.mapping
    return S.ImportPreviewOut(
        headers=list(preview.headers),
        mapping=S.ColumnMappingIn(date=m.date, isin=m.isin, type=m.type,
                                  quantity=m.quantity, price=m.price,
                                  currency=m.currency, fees=m.fees, note=m.note,
                                  date_format=m.date_format,
                                  decimal_comma=m.decimal_comma),
        rows=[S.ImportRowOut(
            line=r.line, ok=r.ok, error=r.error,
            raw={str(k): ("" if v is None else str(v)) for k, v in r.raw.items()},
            transaction=builder.transaction_out(r.transaction, names)
            if r.transaction is not None else None) for r in preview.rows],
        valid=preview.valid, invalid=preview.invalid,
        unknown_isins=list(preview.unknown_isins))


@router.post("/transactions/import/preview", response_model=S.ImportPreviewOut)
def import_preview(req: S.ImportRequest,
                   svc: PortfolioService = Depends(service)) -> S.ImportPreviewOut:
    return _preview_out(_preview(req, svc), _names(svc))


@router.post("/transactions/import")
def import_transactions(req: S.ImportRequest,
                        svc: PortfolioService = Depends(service)) -> dict:
    preview = _preview(req, svc)
    store = svc.store()
    rows = [r.transaction for r in preview.rows if r.ok and r.transaction is not None]
    if not rows:
        raise HTTPException(status_code=400, detail="no valid rows to import")
    try:
        derive_positions(store.load_transactions() + rows, strict=False)
    except InsufficientUnits as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for txn in rows:
        store.append_transaction(txn)
    svc.invalidate()
    return {"imported": len(rows), "skipped": preview.invalid}


# -- export ------------------------------------------------------------------

@router.get("/export/transactions.csv")
def export_transactions(svc: PortfolioService = Depends(service)) -> Response:
    store = svc.store()
    text = transactions_csv(store.load_transactions(), store.load_instruments())
    return Response(text, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="transactions.csv"'})


@router.get("/export/holdings.csv")
def export_holdings(svc: PortfolioService = Depends(service)) -> Response:
    text = holdings_csv(svc.analysis().table)
    return Response(text, media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="holdings.csv"'})


@router.get("/export/workbook.xlsx")
def export_workbook(svc: PortfolioService = Depends(service)) -> Response:
    analysis = svc.analysis()
    snap = analysis.snapshot
    sheets = {
        "Holdings": pd.DataFrame([h.model_dump(exclude={"sparkline", "warnings"})
                                  for h in snap.holdings]),
        "Transactions": pd.DataFrame([t.model_dump() for t in list_transactions(True, svc)]),
        "Risk": pd.DataFrame([r.model_dump() for r in snap.risk.divergence]),
        "Performance": pd.DataFrame({"date": snap.performance.dates,
                                     "value": snap.performance.value,
                                     "invested": snap.performance.invested,
                                     "index": snap.performance.index,
                                     "benchmark_index": snap.performance.benchmark_index}),
    }
    try:
        data = workbook(sheets)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="portfolio.xlsx"'})
