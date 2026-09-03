"""On-disk store, with seed and user data kept strictly apart.

Two failure modes this exists to prevent, and they point in opposite
directions:

  - A public demo that silently reads real positions.
  - A real session that silently reads demo data and reports numbers that
    are not yours.

Both are bad, so the mode is explicit, always reported, and never inferred.
`DataStore.describe()` returns a banner string the UI is expected to display.

Layout
------
    data_store/seed/    ten reference instruments + a synthetic ledger.
                        Committed to git. What the tests run against and what a
                        public demo shows.
    data_store/user/    real instruments and transactions. Gitignored.

The ledger is append-only on disk as well as in the model: `transactions.csv`
is only ever appended to, and corrections go to `amendments.csv`. Nothing here
rewrites a transaction row.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import enum
import os
import pathlib
from decimal import Decimal

from ..core.models import (Amendment, AmendmentAction, AssetClass, Instrument,
                           Transaction, TransactionType, apply_amendments)

__all__ = ["DataMode", "DataStore", "resolve_mode", "DEFAULT_ROOT"]

DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data_store"
MODE_ENV_VAR = "PORTFOLIO_DATA_MODE"


class DataMode(str, enum.Enum):
    SEED = "seed"
    USER = "user"


def resolve_mode(explicit: str | DataMode | None = None) -> DataMode:
    """Pick the mode, defaulting to SEED.

    SEED is the default on purpose: the failure of showing demo data when you
    wanted real data is visible and annoying, while the reverse -- real
    positions on a demo screen -- is a privacy failure you might not notice
    until it is public.
    """
    if explicit is not None:
        return DataMode(explicit)
    return DataMode(os.environ.get(MODE_ENV_VAR, DataMode.SEED.value).strip().lower())


INSTRUMENT_COLUMNS = ["isin", "name", "issuer", "asset_class", "base_currency",
                      "primary_symbol", "exchange", "quote_currency",
                      "provider_symbols", "active", "manual_overrides", "note"]
TRANSACTION_COLUMNS = ["id", "date", "isin", "type", "quantity", "price_per_unit",
                       "currency", "fees", "note"]
AMENDMENT_COLUMNS = ["id", "target_id", "action", "at", "reason"]


def _pack_symbols(d: dict[str, str]) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(d.items()) if v)


def _unpack_symbols(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (s or "").split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


@dataclasses.dataclass
class DataStore:
    """CSV-backed store. Deliberately boring: the data must outlive the tool."""
    mode: DataMode = DataMode.SEED
    root: pathlib.Path = DEFAULT_ROOT

    @classmethod
    def open(cls, mode: str | DataMode | None = None,
             root: pathlib.Path | None = None) -> "DataStore":
        store = cls(mode=resolve_mode(mode), root=root or DEFAULT_ROOT)
        store.directory.mkdir(parents=True, exist_ok=True)
        return store

    # -- paths -------------------------------------------------------------

    @property
    def directory(self) -> pathlib.Path:
        return self.root / self.mode.value

    @property
    def instruments_path(self) -> pathlib.Path:
        return self.directory / "instruments.csv"

    @property
    def transactions_path(self) -> pathlib.Path:
        return self.directory / "transactions.csv"

    @property
    def amendments_path(self) -> pathlib.Path:
        return self.directory / "amendments.csv"

    def describe(self) -> str:
        """The banner the UI must show. Never let the mode be a guess."""
        if self.mode is DataMode.SEED:
            return ("DEMO DATA - showing the seed instrument set and a synthetic "
                    "ledger. These are not your positions.")
        return f"LIVE DATA - your instruments and ledger from {self.directory}"

    # -- instruments -------------------------------------------------------

    def load_instruments(self) -> dict[str, Instrument]:
        if not self.instruments_path.exists():
            return {}
        out: dict[str, Instrument] = {}
        with self.instruments_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not (row.get("isin") or "").strip():
                    continue
                inst = Instrument(
                    isin=row["isin"],
                    name=row["name"],
                    issuer=row.get("issuer", ""),
                    asset_class=AssetClass(row.get("asset_class") or "ETF"),
                    base_currency=row.get("base_currency") or "EUR",
                    primary_symbol=row.get("primary_symbol", ""),
                    exchange=row.get("exchange", ""),
                    quote_currency=row.get("quote_currency", "") or "",
                    provider_symbols=_unpack_symbols(row.get("provider_symbols", "")),
                    active=(row.get("active", "true").strip().lower()
                            not in {"false", "0", "no"}),
                    manual_overrides={f for f in (row.get("manual_overrides") or "").split("|") if f},
                    note=row.get("note", ""),
                )
                out[inst.isin] = inst
        return out

    def save_instruments(self, instruments: dict[str, Instrument]) -> None:
        """Reference data is rewritten wholesale -- unlike the ledger, it is a
        current-state table, not a history."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.instruments_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=INSTRUMENT_COLUMNS)
            w.writeheader()
            for inst in sorted(instruments.values(), key=lambda i: i.isin):
                w.writerow({
                    "isin": inst.isin, "name": inst.name, "issuer": inst.issuer,
                    "asset_class": inst.asset_class.value,
                    "base_currency": inst.base_currency,
                    "primary_symbol": inst.primary_symbol,
                    "exchange": inst.exchange,
                    "quote_currency": inst.quote_currency,
                    "provider_symbols": _pack_symbols(inst.provider_symbols),
                    "active": "true" if inst.active else "false",
                    "manual_overrides": "|".join(sorted(inst.manual_overrides)),
                    "note": inst.note.replace("\n", "; "),
                })

    # -- ledger ------------------------------------------------------------

    def load_transactions(self, include_voided: bool = False) -> list[Transaction]:
        rows: list[Transaction] = []
        if self.transactions_path.exists():
            with self.transactions_path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if not (row.get("isin") or "").strip():
                        continue
                    kwargs = dict(
                        date=dt.date.fromisoformat(row["date"]),
                        isin=row["isin"],
                        type=TransactionType(row["type"]),
                        quantity=Decimal(row.get("quantity") or "0"),
                        price_per_unit=Decimal(row.get("price_per_unit") or "0"),
                        currency=row.get("currency") or "EUR",
                        fees=Decimal(row.get("fees") or "0"),
                        note=row.get("note", ""),
                    )
                    # A hand-edited CSV may omit the id column; let the model
                    # mint one rather than failing to load the whole ledger.
                    if (row.get("id") or "").strip():
                        kwargs["id"] = row["id"].strip()
                    rows.append(Transaction(**kwargs))
        if include_voided:
            return rows
        return apply_amendments(rows, self.load_amendments())

    def append_transaction(self, txn: Transaction) -> Transaction:
        """Append only. This function never rewrites an existing row."""
        self.directory.mkdir(parents=True, exist_ok=True)
        new_file = not self.transactions_path.exists()
        with self.transactions_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=TRANSACTION_COLUMNS)
            if new_file:
                w.writeheader()
            w.writerow({
                "id": txn.id, "date": txn.date.isoformat(), "isin": txn.isin,
                "type": txn.type.value, "quantity": str(txn.quantity),
                "price_per_unit": str(txn.price_per_unit), "currency": txn.currency,
                "fees": str(txn.fees), "note": txn.note,
            })
        return txn

    def load_amendments(self) -> list[Amendment]:
        if not self.amendments_path.exists():
            return []
        out: list[Amendment] = []
        with self.amendments_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not (row.get("target_id") or "").strip():
                    continue
                kwargs = dict(
                    target_id=row["target_id"],
                    action=AmendmentAction(row.get("action") or "VOID"),
                    at=dt.datetime.fromisoformat(row["at"]) if row.get("at")
                       else dt.datetime.now(dt.timezone.utc),
                    reason=row.get("reason", ""),
                )
                if (row.get("id") or "").strip():
                    kwargs["id"] = row["id"].strip()
                out.append(Amendment(**kwargs))
        return out

    def append_amendment(self, amendment: Amendment) -> Amendment:
        self.directory.mkdir(parents=True, exist_ok=True)
        new_file = not self.amendments_path.exists()
        with self.amendments_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=AMENDMENT_COLUMNS)
            if new_file:
                w.writeheader()
            w.writerow({"id": amendment.id, "target_id": amendment.target_id,
                        "action": amendment.action.value,
                        "at": amendment.at.isoformat(), "reason": amendment.reason})
        return amendment

    def void_transaction(self, txn_id: str, reason: str = "") -> Amendment:
        """Delete, expressed as an append. The original row stays on disk."""
        return self.append_amendment(Amendment(target_id=txn_id, reason=reason))
