"""CSV import of transactions from broker exports. Preview first, always.

A broker export is never quite the ledger's shape: the columns have other
names, the dates are dd.mm.yyyy, the decimals are commas, a sell is a
negative quantity of "Trade", and one row in forty is broken. The import
therefore has two halves, and the split is the design:

    detect_mapping   headers -> which column is which (fuzzy, overridable)
    preview_import   text -> one ImportRow per line, each ok or carrying a
                     sentence that says what is wrong with it

Nothing is written here. The API imports the rows a preview marked ok, so the
user sees every rejection -- and every duplicate -- before the ledger changes.
The alternative, importing what parses and logging the rest, is how a ledger
ends up with thirty-nine rows and a position nobody can reconcile.

Every row is built through `core.models.Transaction`, so the validation the
ledger applies to a hand-entered row applies to an imported one, and the
message a user sees for a bad ISIN is the same sentence in both places.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import io
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Iterable

from ..core.models import Transaction, TransactionType, ValidationError

__all__ = ["ColumnMapping", "ImportRow", "ImportPreview", "detect_mapping",
           "preview_import", "mark_duplicates", "duplicate_key", "normalise_type",
           "parse_number", "parse_date"]


# --------------------------------------------------------------------------
# Column mapping
# --------------------------------------------------------------------------

@dataclasses.dataclass
class ColumnMapping:
    """Which header holds which field. None means "not present"."""
    date: str | None = None
    isin: str | None = None
    type: str | None = None
    quantity: str | None = None
    price: str | None = None
    currency: str | None = None
    fees: str | None = None
    note: str | None = None
    date_format: str | None = None      # strptime format, when ISO and the usual fail
    decimal_comma: bool = False         # "1.234,56" rather than "1,234.56"

    def missing_required(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if getattr(self, f) is None]


FIELDS = ("date", "isin", "type", "quantity", "price", "currency", "fees", "note")
# Without these a row cannot become a transaction at all; the rest default.
REQUIRED_FIELDS = ("date", "isin", "type", "quantity", "price")

# Header synonyms per field, in priority order. Matched after folding case,
# accents and punctuation, so "Währung", "waehrung" and "WAHRUNG" all land.
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "date": ("date", "trade date", "datum", "executed", "execution date",
             "transaction date", "handelstag", "ausfuhrungsdatum", "fecha"),
    "isin": ("isin", "isin code", "wertpapier isin"),
    "type": ("type", "side", "action", "transaction type", "buy sell", "buy/sell",
             "typ", "art", "transaktionstyp", "operation", "direction"),
    "quantity": ("quantity", "qty", "units", "shares", "amount of shares",
                 "anzahl", "stuck", "stk", "number of shares", "quantite", "nominal"),
    "price": ("price", "price per unit", "unit price", "kurs", "preis", "prix",
              "price per share", "execution price", "ausfuhrungskurs"),
    "currency": ("currency", "ccy", "wahrung", "devise", "curr"),
    "fees": ("fee", "fees", "commission", "costs", "gebuhren", "gebuhr", "kosten",
             "frais", "provision", "charges"),
    "note": ("note", "notes", "description", "comment", "comments", "bemerkung",
             "notiz", "kommentar", "memo", "remarks"),
}


def _fold(text: str) -> str:
    """Lower-case ASCII with punctuation collapsed to single spaces."""
    stripped = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    stripped = stripped.replace("ß", "ss").lower()
    return re.sub(r"[^a-z0-9/]+", " ", stripped).strip()


def detect_mapping(headers: list[str]) -> ColumnMapping:
    """Guess the mapping from the header row. Unmatched fields stay None.

    Exact synonym matches are claimed first for every field, then substring
    matches, so "Trade Date" beats "Settlement Date" for `date` even though
    both contain the word, and "Price" is not stolen by "Price Currency"
    when a plain "Currency" column exists.
    """
    folded = {h: _fold(h) for h in headers}
    claimed: set[str] = set()
    mapping = ColumnMapping()

    for field in FIELDS:
        for synonym in _SYNONYMS[field]:
            hit = next((h for h in headers if h not in claimed and folded[h] == synonym), None)
            if hit is not None:
                setattr(mapping, field, hit)
                claimed.add(hit)
                break

    for field in FIELDS:
        if getattr(mapping, field) is not None:
            continue
        for synonym in _SYNONYMS[field]:
            hit = next((h for h in headers if h not in claimed
                        and re.search(rf"\b{re.escape(synonym)}\b", folded[h])), None)
            if hit is not None:
                setattr(mapping, field, hit)
                claimed.add(hit)
                break
    return mapping


# --------------------------------------------------------------------------
# Cell parsing
# --------------------------------------------------------------------------

_TYPE_SYNONYMS: tuple[tuple[TransactionType, tuple[str, ...]], ...] = (
    # SELL before BUY on purpose: "Verkauf" contains "kauf".
    (TransactionType.SELL, ("sell", "sold", "verkauf", "vente", "sale", "vendita",
                            "venta", "s")),
    (TransactionType.BUY, ("buy", "bought", "kauf", "achat", "purchase", "acquisto",
                           "compra", "b")),
    (TransactionType.DIVIDEND, ("dividend", "div", "distribution", "ausschuttung",
                                "dividende", "dividendo", "ertrag", "income")),
    (TransactionType.FEE, ("fee", "fees", "charge", "custody", "gebuhr", "gebuhren",
                           "frais", "commission", "cost", "costs")),
)


def normalise_type(text: str) -> TransactionType:
    """BUY/SELL/DIVIDEND/FEE from whatever the broker calls it."""
    folded = _fold(text)
    if not folded:
        raise ValueError("the type is blank; expected buy, sell, dividend or fee")
    for kind, synonyms in _TYPE_SYNONYMS:
        if folded in synonyms or folded == kind.value.lower():
            return kind
    for kind, synonyms in _TYPE_SYNONYMS:
        if any(re.search(rf"\b{re.escape(s)}\b", folded) for s in synonyms if len(s) > 1):
            return kind
    raise ValueError(
        f"{text.strip()!r} is not a recognised transaction type; expected buy, "
        f"sell, dividend or fee (or a synonym such as Kauf, Verkauf, Ausschüttung)")


_NUMERIC_KEEP = re.compile(r"[^0-9,.\-+]")


def parse_number(text: str, decimal_comma: bool = False) -> Decimal:
    """A Decimal from "1,234.56", "1.234,56", "1234,56" or "-12".

    Both separators present: the last one is the decimal point. Only a
    comma: a decimal comma unless the file is known not to use them and the
    digits after it are a thousands group. `decimal_comma` therefore only
    settles the genuinely ambiguous "1,234".
    """
    raw = (text or "").strip()
    if not raw:
        raise InvalidOperation("blank")
    s = _NUMERIC_KEEP.sub("", raw.replace(" ", "").replace(" ", ""))
    if not s or s in {"-", "+"}:
        raise InvalidOperation(raw)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(",") == 1:
        tail = s.rpartition(",")[2]
        s = s.replace(",", ".") if decimal_comma or len(tail) != 3 else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "")
    elif decimal_comma and s.count(".") >= 1:
        # In a decimal-comma file a dot is a thousands separator -- "3.900"
        # is three thousand nine hundred -- unless it plainly is not.
        head, _, tail = s.rpartition(".")
        if s.count(".") > 1 or len(tail) == 3:
            s = s.replace(".", "")
    return Decimal(s)


_ISO_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")
_EUROPEAN_FORMATS = ("%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y", "%d.%m.%y",
                     "%d-%m-%y", "%Y/%m/%d", "%Y%m%d")


def parse_date(text: str, date_format: str | None = None) -> dt.date:
    """ISO first, then the stated format, then the common European ones.

    US month-first dates are deliberately not guessed: "03/04/2025" would
    parse either way and be wrong half the time without a word of warning.
    Give `date_format` for those.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("the date is blank")
    if _ISO_PREFIX.match(raw):
        return dt.date.fromisoformat(raw[:10])
    if date_format:
        try:
            return dt.datetime.strptime(raw, date_format).date()
        except ValueError:
            pass
    for fmt in _EUROPEAN_FORMATS:
        try:
            return dt.datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"{raw!r} is not a recognised date; use ISO 8601 (2025-01-15) or "
        f"dd.mm.yyyy, or set date_format in the mapping")


# --------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------

@dataclasses.dataclass
class ImportRow:
    line: int                          # 1-based line in the file; the header is line 1
    ok: bool
    transaction: Transaction | None
    error: str | None
    raw: dict[str, str]


@dataclasses.dataclass
class ImportPreview:
    rows: list[ImportRow]
    mapping: ColumnMapping
    valid: int
    invalid: int
    unknown_isins: list[str]
    headers: list[str] = dataclasses.field(default_factory=list)

    def transactions(self) -> list[Transaction]:
        """The rows that may be imported, in file order."""
        return [r.transaction for r in self.rows if r.ok and r.transaction is not None]


_DELIMITERS = ",;\t|"


def _delimiter(sample: str, header_line: str) -> str:
    """csv.Sniffer's answer, checked against the header row.

    The sniffer is right on clean files and wrong on short or decimal-comma
    ones, where a comma inside a number can outvote the semicolons. The
    delimiter that actually separates the header is the one to trust.
    """
    counts = {d: header_line.count(d) for d in _DELIMITERS}
    best = max(counts, key=lambda d: (counts[d], d == ","))
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter
    except csv.Error:
        sniffed = ","
    if counts.get(sniffed, 0) >= counts[best] and counts.get(sniffed, 0) > 0:
        return sniffed
    return best if counts[best] > 0 else ","


# Only the unambiguous shapes: a dot-grouped thousands part before the comma
# ("1.234,56"), or one or two digits after it ("3,90"). "1,234" is left
# alone -- in an English export it is a thousand, and flagging the whole file
# as decimal-comma on its evidence would turn every such quantity into 1.234.
_DECIMAL_COMMA = re.compile(r"^-?\d{1,3}(\.\d{3})+,\d+$|^-?\d+,\d{1,2}$")


def _looks_decimal_comma(rows: list[dict[str, str]], mapping: ColumnMapping) -> bool:
    columns = [c for c in (mapping.quantity, mapping.price, mapping.fees) if c]
    for row in rows:
        for col in columns:
            if _DECIMAL_COMMA.match((row.get(col) or "").strip()):
                return True
    return False


def _fill(mapping: ColumnMapping, detected: ColumnMapping) -> ColumnMapping:
    """A user mapping is authoritative where it speaks; detection fills the rest."""
    merged = dataclasses.replace(mapping)
    for field in FIELDS:
        if getattr(merged, field) is None:
            setattr(merged, field, getattr(detected, field))
    return merged


def preview_import(text: str, mapping: ColumnMapping | None = None,
                   known: set[str] | None = None) -> ImportPreview:
    """Parse a CSV into rows that are either transactions or sentences.

    `known` is the set of ISINs the store already has; ISINs outside it are
    listed in `unknown_isins` but their rows stay valid, because adding the
    instrument is a separate step the UI can offer. Never raises on bad
    input: an unusable file is a preview whose every row says why.
    """
    body = (text or "").lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in body.split("\n") if ln.strip()]
    if not lines:
        return ImportPreview([], mapping or ColumnMapping(), 0, 0, [], [])

    delimiter = _delimiter("\n".join(lines[:20]), lines[0])
    reader = csv.DictReader(io.StringIO(body), delimiter=delimiter, skipinitialspace=True)
    headers = [(h or "").strip() for h in (reader.fieldnames or [])]
    reader.fieldnames = headers

    raw_rows: list[tuple[int, dict[str, str]]] = []
    for record in reader:
        cells = {k: (v if isinstance(v, str) else "") for k, v in record.items()
                 if isinstance(k, str)}
        if not any(v.strip() for v in cells.values()):
            continue                    # a blank line is not a broken row
        raw_rows.append((reader.line_num, cells))

    detected = detect_mapping(headers)
    if mapping is None:
        resolved = detected
        resolved.decimal_comma = _looks_decimal_comma([r for _, r in raw_rows], resolved)
    else:
        resolved = _fill(mapping, detected)

    rows = [_build_row(line, cells, resolved) for line, cells in raw_rows]
    valid = sum(1 for r in rows if r.ok)
    unknown: list[str] = []
    if known is not None:
        unknown = sorted({r.transaction.isin for r in rows
                          if r.ok and r.transaction is not None
                          and r.transaction.isin not in known})
    return ImportPreview(rows=rows, mapping=resolved, valid=valid,
                         invalid=len(rows) - valid, unknown_isins=unknown,
                         headers=headers)


def _cell(cells: dict[str, str], column: str | None) -> str:
    return (cells.get(column) or "").strip() if column else ""


def _build_row(line: int, cells: dict[str, str], mapping: ColumnMapping) -> ImportRow:
    missing = mapping.missing_required()
    if missing:
        return ImportRow(line, False, None,
                         f"no column identified for {', '.join(missing)}; map "
                         f"the column by hand", cells)
    try:
        date = parse_date(_cell(cells, mapping.date), mapping.date_format)
        kind = normalise_type(_cell(cells, mapping.type))
        quantity = _decimal_field(cells, mapping.quantity, "quantity", mapping.decimal_comma)
        price = _decimal_field(cells, mapping.price, "price", mapping.decimal_comma)
        fees = _decimal_field(cells, mapping.fees, "fees", mapping.decimal_comma)
        if kind is TransactionType.SELL and quantity < 0:
            # Brokers that export a sell as a negative quantity mean the same
            # thing the ledger does; sign and type are one fact, not two.
            quantity = abs(quantity)
        txn = Transaction(
            date=date, isin=_cell(cells, mapping.isin), type=kind,
            quantity=quantity, price_per_unit=price,
            currency=_cell(cells, mapping.currency) or "EUR", fees=fees,
            note=_cell(cells, mapping.note))
    except (ValidationError, ValueError, InvalidOperation) as exc:
        return ImportRow(line, False, None, str(exc), cells)
    return ImportRow(line, True, txn, None, cells)


def _decimal_field(cells: dict[str, str], column: str | None, field: str,
                   decimal_comma: bool) -> Decimal:
    raw = _cell(cells, column)
    if not raw:
        return Decimal("0")
    try:
        return parse_number(raw, decimal_comma)
    except InvalidOperation:
        raise ValueError(f"{field}: {raw!r} is not a number") from None


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------

def duplicate_key(txn: Transaction) -> tuple[dt.date, str, str, Decimal, Decimal]:
    """What makes two rows the same trade. Fees and notes do not: a broker
    export and a hand-entered row often disagree on both for one trade."""
    return (txn.date, txn.isin, txn.type.value, txn.quantity, txn.price_per_unit)


def mark_duplicates(preview: ImportPreview, existing: Iterable[Transaction]
                    ) -> ImportPreview:
    """Flag rows already in the ledger, so a re-imported export adds nothing.

    The row keeps its parsed transaction so the UI can show what matched,
    but is no longer ok; the error names the ledger id it duplicates.
    """
    seen: dict[tuple, str] = {}
    for txn in existing:
        seen.setdefault(duplicate_key(txn), txn.id)
    rows: list[ImportRow] = []
    for row in preview.rows:
        if row.ok and row.transaction is not None:
            match = seen.get(duplicate_key(row.transaction))
            if match is not None:
                rows.append(dataclasses.replace(
                    row, ok=False,
                    error=f"duplicate of {match}: the ledger already has this "
                          f"{row.transaction.type.value} of {row.transaction.quantity} "
                          f"on {row.transaction.date.isoformat()}"))
                continue
        rows.append(row)
    valid = sum(1 for r in rows if r.ok)
    return dataclasses.replace(preview, rows=rows, valid=valid, invalid=len(rows) - valid)
