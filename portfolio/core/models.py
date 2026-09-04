"""The data model: instruments (reference data) and transactions (the ledger).

Two things are kept strictly apart, because conflating them is how a portfolio
tool ends up unable to answer "what did I actually pay":

  Instrument  Reference data. Exists whether or not you own any. An instrument
              with no transactions is a watchlist entry -- a legitimate state,
              not an error, and what the v2 what-if simulator reads from. There
              is deliberately no `is_watchlist` flag: an empty position derives
              it, and a flag would be a second source of truth to keep in sync.

  Transaction The ledger. Append-only. Positions are derived from it and never
              stored, because a position table cannot express a partial sell,
              realised P&L, or a time-weighted return.

Append-only, concretely
-----------------------
The UI needs edit and delete; the ledger must never be rewritten. Those are
reconciled with a second append-only log of `Amendment` rows: a delete appends
a VOID amendment naming the transaction id, and an edit appends a VOID plus a
new transaction. `transactions.csv` is therefore only ever appended to, and the
full history of what you corrected and when survives.

Rejected alternative: a `status` column mutated in place. Simpler to read, but
it destroys the audit trail and makes "why did last month's value change"
unanswerable.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import uuid
from decimal import Decimal

from .money import BASE_CURRENCY, Money, normalise_currency

__all__ = [
    "AssetClass", "TransactionType", "AmendmentAction", "Transaction",
    "Amendment", "Instrument", "isin_check_digit", "is_valid_isin",
    "ValidationError",
]


class ValidationError(ValueError):
    """A record that would corrupt the ledger if written."""


class AssetClass(str, enum.Enum):
    """ETC is separated from ETF on purpose.

    A commodity ETC is a collateralised debt security, not a UCITS fund. It
    carries issuer credit risk an ETF does not, several data providers classify
    it differently, and three of the ten reference instruments are ETCs. Folding
    them into "ETF" would hide all of that.
    """
    ETF = "ETF"
    ETC = "ETC"
    EQUITY = "EQUITY"
    FUND = "FUND"
    OTHER = "OTHER"


class TransactionType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    DIVIDEND = "DIVIDEND"
    FEE = "FEE"


class AmendmentAction(str, enum.Enum):
    VOID = "VOID"


# --------------------------------------------------------------------------
# ISIN validation
# --------------------------------------------------------------------------
# ISIN carries a check digit (ISO 6166): letters expand to two digits with
# A=10..Z=35, then a Luhn checksum over the resulting digit string. Validating
# it catches a mistyped character at the point of entry, which matters a great
# deal when ISIN is the primary key -- a typo there does not fail loudly, it
# creates a second, empty instrument that quietly splits a position in two.

def isin_check_digit(body: str) -> int:
    """Check digit for the first 11 characters of an ISIN.

    >>> isin_check_digit("IE0002Y8CX9")
    8
    """
    digits = "".join(str(int(c, 36)) for c in body.upper())
    total = 0
    # Luhn: double every second digit counting from the right.
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def is_valid_isin(isin: str) -> bool:
    """True if `isin` is 12 chars, correctly shaped, and passes its check digit.

    >>> is_valid_isin("IE0002Y8CX98")     # WisdomTree Europe Defence
    True
    >>> is_valid_isin("IE0002Y8CX97")     # one digit wrong
    False
    """
    s = (isin or "").strip().upper()
    if len(s) != 12 or not s[:2].isalpha() or not s[:11].isalnum() or not s[11].isdigit():
        return False
    return isin_check_digit(s[:11]) == int(s[11])


def _require_isin(isin: str) -> str:
    s = (isin or "").strip().upper()
    if not is_valid_isin(s):
        raise ValidationError(
            f"{isin!r} is not a valid ISIN (12 characters, correct check digit). "
            f"ISIN is the primary key -- a typo here creates a second, empty "
            f"instrument rather than failing loudly.")
    return s


def _dec(value: Decimal | int | float | str, field: str) -> Decimal:
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValidationError(f"{field}: {value!r} is not a number") from exc


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Transaction:
    """One entry in the append-only ledger.

    `currency` is the currency of `price_per_unit` and `fees` -- the currency
    actually paid, which is the listing's quote currency and often not the
    instrument's base currency. Both are kept because the cost basis converts
    at the transaction date and the two answers differ.
    """
    date: dt.date
    isin: str
    type: TransactionType
    quantity: Decimal = Decimal("0")
    price_per_unit: Decimal = Decimal("0")
    currency: str = BASE_CURRENCY
    fees: Decimal = Decimal("0")
    note: str = ""
    id: str = dataclasses.field(default_factory=lambda: _new_id("txn"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "isin", _require_isin(self.isin))
        object.__setattr__(self, "type", TransactionType(self.type))
        if isinstance(self.date, str):
            object.__setattr__(self, "date", dt.date.fromisoformat(self.date))

        qty = _dec(self.quantity, "quantity")
        price = _dec(self.price_per_unit, "price_per_unit")
        fees = _dec(self.fees, "fees")

        # Normalise a pence-quoted trade to pounds at construction, so nothing
        # downstream has to remember that LSE quotes some of these in GBX.
        code, price = normalise_currency(self.currency, price)
        _, fees = normalise_currency(self.currency, fees)
        object.__setattr__(self, "currency", code)
        object.__setattr__(self, "price_per_unit", price)
        object.__setattr__(self, "fees", fees)
        object.__setattr__(self, "quantity", qty)

        if self.type in (TransactionType.BUY, TransactionType.SELL):
            if qty <= 0:
                raise ValidationError(
                    f"{self.type.value} needs a positive quantity, got {qty}. "
                    f"A sell is recorded as a positive quantity of type SELL, "
                    f"not a negative BUY.")
            if price < 0:
                raise ValidationError(f"price_per_unit cannot be negative, got {price}")
        if fees < 0:
            raise ValidationError(f"fees cannot be negative, got {fees}")
        if self.type is TransactionType.FEE and fees == 0 and price == 0:
            raise ValidationError("a FEE transaction needs a non-zero fee or price")

    @property
    def gross(self) -> Money:
        """Quantity x price, before fees, in the transaction currency."""
        return Money(self.quantity * self.price_per_unit, self.currency)

    @property
    def fee_money(self) -> Money:
        return Money(self.fees, self.currency)

    def cash_flow(self) -> Money:
        """Signed cash effect in the transaction currency: negative is money out."""
        if self.type is TransactionType.BUY:
            return Money(-(self.quantity * self.price_per_unit) - self.fees, self.currency)
        if self.type is TransactionType.SELL:
            return Money((self.quantity * self.price_per_unit) - self.fees, self.currency)
        if self.type is TransactionType.DIVIDEND:
            gross = self.quantity * self.price_per_unit
            if gross == 0:
                gross = self.price_per_unit
            return Money(gross - self.fees, self.currency)
        return Money(-self.fees - (self.quantity * self.price_per_unit), self.currency)


@dataclasses.dataclass(frozen=True)
class Amendment:
    """A correction to the ledger, itself appended rather than applied in place."""
    target_id: str
    action: AmendmentAction = AmendmentAction.VOID
    at: dt.datetime = dataclasses.field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    reason: str = ""
    id: str = dataclasses.field(default_factory=lambda: _new_id("amd"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", AmendmentAction(self.action))
        if not self.target_id:
            raise ValidationError("an amendment must name the transaction it amends")


def apply_amendments(transactions: list[Transaction],
                     amendments: list[Amendment]) -> list[Transaction]:
    """The ledger as it currently stands: appended rows minus voided ones."""
    voided = {a.target_id for a in amendments if a.action is AmendmentAction.VOID}
    return [t for t in transactions if t.id not in voided]


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Instrument:
    """Reference data for one instrument, keyed by ISIN.

    `provider_symbols` is the mechanism that keeps ISIN as the key while still
    letting each provider be queried with whatever symbol it expects. The same
    UCITS ETF trades as EUDF on Xetra and WDEF on four other venues; ticker plus
    exchange is a fetch handle, never an identity.

    `manual_overrides` names fields the user corrected by hand, including
    dotted paths into `provider_symbols` such as "provider_symbols.eodhd".
    Re-resolution must not overwrite those: an automatic resolver that silently
    reverts a correction is worse than one that never runs.
    """
    isin: str
    name: str
    asset_class: AssetClass = AssetClass.ETF
    base_currency: str = BASE_CURRENCY
    issuer: str = ""
    primary_symbol: str = ""
    exchange: str = ""                       # MIC of the primary listing
    quote_currency: str = ""                 # currency of the primary listing
    provider_symbols: dict[str, str] = dataclasses.field(default_factory=dict)
    active: bool = True
    manual_overrides: set[str] = dataclasses.field(default_factory=set)
    note: str = ""
    added_at: dt.datetime = dataclasses.field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at: dt.datetime = dataclasses.field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    def __post_init__(self) -> None:
        self.isin = _require_isin(self.isin)
        self.asset_class = AssetClass(self.asset_class)
        self.base_currency, _ = normalise_currency(self.base_currency)
        if self.quote_currency:
            self.quote_currency, _ = normalise_currency(self.quote_currency)
        if not self.name.strip():
            raise ValidationError("an instrument needs a name")

    # -- manual override bookkeeping ---------------------------------------

    def override(self, field: str, value: object) -> "Instrument":
        """Set a field by hand and mark it protected from re-resolution."""
        if field.startswith("provider_symbols."):
            provider = field.split(".", 1)[1]
            self.provider_symbols[provider] = str(value)
        elif hasattr(self, field):
            setattr(self, field, value)
        else:
            raise ValidationError(f"no such instrument field: {field!r}")
        self.manual_overrides.add(field)
        self.updated_at = dt.datetime.now(dt.timezone.utc)
        return self

    def is_overridden(self, field: str) -> bool:
        return field in self.manual_overrides

    def apply_resolution(self, resolved: dict[str, object]) -> list[str]:
        """Merge fresh resolver output, skipping anything set by hand.

        Returns the field names it declined to touch, so the UI can say
        "3 fields kept your correction" rather than silently doing nothing.
        """
        skipped: list[str] = []
        for field, value in resolved.items():
            if field == "provider_symbols" and isinstance(value, dict):
                for provider, symbol in value.items():
                    key = f"provider_symbols.{provider}"
                    if key in self.manual_overrides:
                        skipped.append(key)
                        continue
                    self.provider_symbols[provider] = str(symbol)
                continue
            if field in self.manual_overrides:
                skipped.append(field)
                continue
            if hasattr(self, field):
                setattr(self, field, value)
        self.updated_at = dt.datetime.now(dt.timezone.utc)
        return skipped
