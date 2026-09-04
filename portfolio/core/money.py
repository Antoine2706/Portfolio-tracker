"""Currency: amounts, conversion, and the two traps that corrupt portfolios.

Six of the ten reference instruments are USD-base, three of those trade in EUR
on continental venues, and four LSE lines quote in pence. Currency is therefore
the normal case here, not an error path, which is why this module sits in
`core` as a first-class concept rather than in `data/fx.py` as a utility.

`core` never touches the network. This module defines what a rate *is* and how
conversion works; fetching rates is `data/fx.py`'s job and arrives here as an
injected `FxRates`. That split is what lets the whole risk stack be tested
offline.

Two design decisions worth stating:

1. Decimal, not float, for money and quantities. Cost basis accumulates across
   many transactions and float drift is real; more practically, the spec asks
   for a fixture portfolio with hand-computable values, and Decimal makes hand
   verification exact. Floats appear only at the boundary into numpy, where the
   risk maths needs them. Rejected alternative: floats everywhere, simpler but
   it makes "did I compute this right" unanswerable by hand.

2. GBX is normalised to GBP at 1/100. This is not pedantry. Four listings in
   the reference set quote in pence -- WDEP, NAVY, NATP, SPAG -- and GBX is not
   an ISO 4217 code, so a provider may return "GBX", "GBp" or "GBP" for the
   same line depending on the day. Treating pence as pounds overstates a
   position by 100x, which is the kind of error that looks like a data glitch
   rather than a bug.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Protocol, runtime_checkable

__all__ = [
    "CurrencyMismatch", "Money", "FxRates", "FxTable", "MissingRate",
    "normalise_currency", "convert", "BASE_CURRENCY",
]

# The reporting currency for everything the user sees.
BASE_CURRENCY = "EUR"


class CurrencyMismatch(ValueError):
    """Raised when two amounts in different currencies would be combined."""


class MissingRate(LookupError):
    """Raised when a conversion is asked for and no rate is available.

    Deliberately an error rather than a fallback to 1.0. A silent identity
    rate would turn a USD position into a EUR one at par and be invisible.
    """


# Currencies quoted in a minor unit, mapped to (major code, multiplier).
#
# Case matters here and only here. By convention "GBp" with a lowercase p means
# pence while "GBP" means pounds, so these codes MUST be matched before the
# string is upper-cased -- fold the case first and "GBp" becomes "GBP", pence
# get treated as pounds, and the position is overstated by 100x. That reads as
# a data glitch rather than a bug, which is what makes it dangerous.
_MINOR_UNIT_EXACT: dict[str, tuple[str, Decimal]] = {
    "GBp": ("GBP", Decimal("0.01")),
    "ZAc": ("ZAR", Decimal("0.01")),
    "ILa": ("ILS", Decimal("0.01")),
}
# These are unambiguous in any case, so they fold safely.
_MINOR_UNIT_UPPER: dict[str, tuple[str, Decimal]] = {
    "GBX": ("GBP", Decimal("0.01")),
    "ZAC": ("ZAR", Decimal("0.01")),
    "ILS_AGOROT": ("ILS", Decimal("0.01")),
}


def normalise_currency(code: str, amount: Decimal | None = None
                       ) -> tuple[str, Decimal | None]:
    """Return an ISO 4217 code, rescaling a minor-unit amount if given.

    >>> normalise_currency("GBp", Decimal("250"))
    ('GBP', Decimal('2.50'))
    >>> normalise_currency("GBX", Decimal("250"))
    ('GBP', Decimal('2.50'))
    >>> normalise_currency("GBP", Decimal("250"))
    ('GBP', Decimal('250'))
    >>> normalise_currency("eur")
    ('EUR', None)
    """
    if not code:
        raise ValueError("currency code is required")
    raw = code.strip()

    # Case-sensitive minor units first: see the note on _MINOR_UNIT_EXACT.
    if raw in _MINOR_UNIT_EXACT:
        major, factor = _MINOR_UNIT_EXACT[raw]
        return major, (None if amount is None else amount * factor)

    key = raw.upper()
    if key in _MINOR_UNIT_UPPER:
        major, factor = _MINOR_UNIT_UPPER[key]
        return major, (None if amount is None else amount * factor)

    if len(key) != 3 or not key.isalpha():
        raise ValueError(f"not an ISO 4217 currency code: {code!r}")
    return key, amount


@dataclasses.dataclass(frozen=True, order=False)
class Money:
    """An amount in a stated currency. Arithmetic across currencies raises."""
    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", Decimal(str(self.amount)))
        code, rescaled = normalise_currency(self.currency, self.amount)
        object.__setattr__(self, "currency", code)
        if rescaled is not None:
            object.__setattr__(self, "amount", rescaled)

    @classmethod
    def zero(cls, currency: str = BASE_CURRENCY) -> "Money":
        return cls(Decimal("0"), currency)

    def _check(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise CurrencyMismatch(
                f"cannot combine {self.currency} and {other.currency} without "
                f"converting first")

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: Decimal | int) -> "Money":
        return Money(self.amount * Decimal(str(factor)), self.currency)

    __rmul__ = __mul__

    def __neg__(self) -> "Money":
        return Money(-self.amount, self.currency)

    def __str__(self) -> str:
        return f"{self.amount:,.2f} {self.currency}"


@runtime_checkable
class FxRates(Protocol):
    """A source of FX rates. `core` depends on this shape, never on a fetcher."""

    def rate(self, frm: str, to: str, on: dt.date) -> Decimal:
        """Units of `to` per one unit of `frm`, on the given date."""
        ...


@dataclasses.dataclass
class FxTable:
    """An in-memory FxRates. Used by tests and as the cache behind data/fx.py.

    Rates are keyed (from, to, date). A missing exact date falls back to the
    most recent earlier date, which is what you want for a weekend or holiday
    trade -- and it records nothing silently, because `MissingRate` is raised
    when there is no earlier rate at all.
    """
    rates: dict[tuple[str, str, dt.date], Decimal] = dataclasses.field(default_factory=dict)

    def add(self, frm: str, to: str, on: dt.date, rate: Decimal | str) -> "FxTable":
        f, _ = normalise_currency(frm)
        t, _ = normalise_currency(to)
        self.rates[(f, t, on)] = Decimal(str(rate))
        return self

    def rate(self, frm: str, to: str, on: dt.date) -> Decimal:
        f, _ = normalise_currency(frm)
        t, _ = normalise_currency(to)
        if f == t:
            return Decimal("1")
        exact = self.rates.get((f, t, on))
        if exact is not None:
            return exact
        inverse = self.rates.get((t, f, on))
        if inverse is not None and inverse != 0:
            return Decimal("1") / inverse
        # Most recent earlier rate for this pair.
        earlier = [d for (a, b, d) in self.rates if a == f and b == t and d <= on]
        if earlier:
            return self.rates[(f, t, max(earlier))]
        earlier_inv = [d for (a, b, d) in self.rates if a == t and b == f and d <= on]
        if earlier_inv:
            got = self.rates[(t, f, max(earlier_inv))]
            if got != 0:
                return Decimal("1") / got
        raise MissingRate(f"no FX rate for {f}->{t} on or before {on}")


def convert(money: Money, to: str, on: dt.date, rates: FxRates) -> Money:
    """Convert at the rate for a stated date.

    The date matters and is never defaulted: cost basis converts at the
    transaction date, current market value at today's rate. Using one where the
    other belongs silently rewrites history.
    """
    target, _ = normalise_currency(to)
    if money.currency == target:
        return money
    return Money(money.amount * rates.rate(money.currency, target, on), target)
