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
import sys
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


# short_name is appended rather than inserted, and read with a default, so a
# CSV written by an older build still loads: a blank short name means "derive
# it from the legal name", which is what an absent column should mean too.
INSTRUMENT_COLUMNS = ["isin", "name", "short_name", "issuer", "asset_class",
                      "base_currency", "primary_symbol", "exchange",
                      "quote_currency", "provider_symbols", "active",
                      "manual_overrides", "note",
                      # where it is held and what trading it costs; appended
                      # so a CSV written by an older build still loads
                      "broker", "tradeable", "tob_rate", "tob_observed",
                      "half_spread_bps", "spread_observed", "buy_tax_rate",
                      # the venue the trade executes on, the commission the
                      # broker actually charged, and which of the three
                      # evidence tiers the spread came from
                      "venue", "commission", "commission_observed",
                      "spread_source",
                      # whether NEW money may go in, which is not the same
                      # question as whether the weight can be rebalanced
                      "buyable"]


def _opt_float(raw: str | None) -> float | None:
    """A blank cell means NOT RECORDED, which is not the same as zero.

    A transaction tax read as 0.0 when the cell was empty would price every
    trade in that instrument as tax-free -- a plausible number, silently
    wrong, which is the failure mode this column exists to prevent.
    """
    text = (raw or "").strip()
    return float(text) if text else None


def _bool(raw: str | None, default: bool = False) -> bool:
    text = (raw or "").strip().lower()
    if not text:
        return default
    return text not in {"false", "0", "no"}


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

    # Set by `load_instruments` when it had to migrate the file it read, so a
    # caller can print what changed. Not a constructor argument and not part
    # of the store's identity: it is a fact about the last read.
    last_migration: object = dataclasses.field(default=None, init=False,
                                               repr=False, compare=False)

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
                    short_name=row.get("short_name", "") or "",
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
                    broker=row.get("broker", "") or "",
                    tradeable=_bool(row.get("tradeable"), default=True),
                    buyable=_bool(row.get("buyable"), default=True),
                    tob_rate=_opt_float(row.get("tob_rate")),
                    tob_observed=_bool(row.get("tob_observed")),
                    half_spread_bps=_opt_float(row.get("half_spread_bps")),
                    spread_observed=_bool(row.get("spread_observed")),
                    spread_source=row.get("spread_source", "") or "",
                    buy_tax_rate=float(row.get("buy_tax_rate") or 0.0),
                    venue=row.get("venue", "") or "",
                    commission=_opt_float(row.get("commission")),
                    commission_observed=_bool(row.get("commission_observed")),
                )
                out[inst.isin] = inst
        if out:
            self._migrate_instruments(out)
        return out

    # The file this reads may predate the trading-cost columns. Left alone, a
    # book like that loads with every tax rate NOT RECORDED, which makes every
    # instrument unpriceable and every backtest impossible -- graceful loading
    # and refusing to guess, each correct alone, composing into a tool that
    # cannot run. See `migrations.py` for why the trigger is the header rather
    # than a blank cell.
    BACKUP_SUFFIX = ".before-trading-columns"

    def _migrate_instruments(self, instruments: dict[str, Instrument]) -> None:
        from .migrations import backfill_trading_facts, missing_columns

        missing = missing_columns(self.instruments_path)
        if not missing:
            return
        report = backfill_trading_facts(instruments, self.instruments_path,
                                        fillable=set(missing))
        backup = self.instruments_path.parent / (
            self.instruments_path.name + self.BACKUP_SUFFIX)
        try:
            if not backup.exists():
                backup.write_bytes(self.instruments_path.read_bytes())
            self.save_instruments(instruments)
        except OSError as exc:
            # A read-only install directory must not stop the application
            # starting. The derived values still apply for this run; they are
            # simply derived again next time.
            report = dataclasses.replace(report, persisted=False, backup=None,
                                         error=f"{type(exc).__name__}: {exc}.")
        else:
            report = dataclasses.replace(report, backup=backup)
        self.last_migration = report
        print("\n".join(report.lines()), file=sys.stderr)

    def save_instruments(self, instruments: dict[str, Instrument]) -> None:
        """Reference data is rewritten wholesale -- unlike the ledger, it is a
        current-state table, not a history."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.instruments_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=INSTRUMENT_COLUMNS)
            w.writeheader()
            for inst in sorted(instruments.values(), key=lambda i: i.isin):
                w.writerow({
                    "isin": inst.isin, "name": inst.name,
                    "short_name": inst.short_name, "issuer": inst.issuer,
                    "asset_class": inst.asset_class.value,
                    "base_currency": inst.base_currency,
                    "primary_symbol": inst.primary_symbol,
                    "exchange": inst.exchange,
                    "quote_currency": inst.quote_currency,
                    "provider_symbols": _pack_symbols(inst.provider_symbols),
                    "active": "true" if inst.active else "false",
                    "manual_overrides": "|".join(sorted(inst.manual_overrides)),
                    "note": inst.note.replace("\n", "; "),
                    "broker": inst.broker,
                    "tradeable": "true" if inst.tradeable else "false",
                    "buyable": "true" if inst.buyable else "false",
                    "tob_rate": "" if inst.tob_rate is None else inst.tob_rate,
                    "tob_observed": "true" if inst.tob_observed else "false",
                    "half_spread_bps": ("" if inst.half_spread_bps is None
                                        else inst.half_spread_bps),
                    "spread_observed": "true" if inst.spread_observed else "false",
                    "spread_source": inst.spread_source,
                    "buy_tax_rate": inst.buy_tax_rate,
                    "venue": inst.venue,
                    "commission": ("" if inst.commission is None
                                   else inst.commission),
                    "commission_observed": ("true" if inst.commission_observed
                                            else "false"),
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
