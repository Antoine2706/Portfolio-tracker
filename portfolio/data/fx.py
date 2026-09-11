"""FX rate history: which pairs to fetch, and a table that answers by date.

`core.money.FxTable` is the reference implementation of `FxRates` and is
right for tests: a handful of hand-entered rates, looked up a few times. It
finds the most recent earlier date by scanning every key, which is O(n) per
lookup. Value history replays the ledger over the union of price calendars --
five hundred dates times six USD-base instruments times a lookup each -- and
that scan turns a sub-second page into a multi-second one. `DailyFxTable`
stores one sorted numpy array of dates per pair and bisects, so a lookup is
O(log n) whatever the history length.

Direction convention, stated once: a series added as (frm, to) holds units of
`to` per one unit of `frm`, which is how Yahoo names its pairs ("USDEUR=X" is
EUR per USD). `FX_PAIRS` maps a currency to the Yahoo symbol that converts it
*into* EUR, because EUR is the reporting currency and every conversion the
tool makes ends there.

Rates are returned as `Decimal` built from the float's 10 significant digits.
Ten is more than any FX quote carries and few enough that the Decimal does not
inherit float's binary noise ("0.92" rather than "0.92000000000000003996").
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Iterable

import numpy as np
import pandas as pd

from ..core.models import Instrument, Transaction
from ..core.money import BASE_CURRENCY, MissingRate, normalise_currency

__all__ = ["FX_PAIRS", "DailyFxTable", "currencies_needed", "pair_symbol"]

# Currency -> the Yahoo symbol quoting EUR per one unit of it.
FX_PAIRS: dict[str, str] = {
    "USD": "USDEUR=X",
    "GBP": "GBPEUR=X",
    "CHF": "CHFEUR=X",
    "SEK": "SEKEUR=X",
    "DKK": "DKKEUR=X",
    "NOK": "NOKEUR=X",
    "JPY": "JPYEUR=X",
    "CAD": "CADEUR=X",
    "AUD": "AUDEUR=X",
}

_EPOCH = dt.date(1970, 1, 1)


def pair_symbol(currency: str, base: str = BASE_CURRENCY) -> str | None:
    """The provider symbol converting `currency` into `base`, if one is known.

    Only EUR-based pairs are tabulated; a different base is a different
    product and returns None rather than a guessed symbol.
    """
    code, _ = normalise_currency(currency)
    target, _ = normalise_currency(base)
    if target != BASE_CURRENCY:
        return None
    return FX_PAIRS.get(code)


def _days(on: dt.date) -> int:
    # Days since the epoch: the same integer numpy's datetime64[D] uses, so a
    # series index converts vectorised and a lookup date converts by hand to
    # the identical scale.
    if isinstance(on, dt.datetime):
        on = on.date()
    return (on - _EPOCH).days


def _decimal(value: float) -> Decimal:
    return Decimal(f"{value:.10g}")


@dataclasses.dataclass
class _Pair:
    """Sorted parallel arrays. Immutable in use; replaced on every add."""
    days: np.ndarray        # int64, ascending, unique
    rates: np.ndarray       # float64

    def at_or_before(self, day: int) -> float | None:
        # side="right" then step back: the last stored date <= `day`, which
        # covers the exact date and the weekend-before-Monday case alike.
        idx = int(np.searchsorted(self.days, day, side="right")) - 1
        return float(self.rates[idx]) if idx >= 0 else None

    def merged(self, days: np.ndarray, rates: np.ndarray) -> "_Pair":
        all_days = np.concatenate([self.days, days])
        all_rates = np.concatenate([self.rates, rates])
        # Stable sort keeps insertion order within a date, so the newest value
        # for a repeated date is the last one and `unique` keeps it below.
        order = np.argsort(all_days, kind="stable")
        all_days, all_rates = all_days[order], all_rates[order]
        # np.unique returns the first index of each value; flip so "first"
        # means "latest added".
        _, first = np.unique(all_days[::-1], return_index=True)
        keep = len(all_days) - 1 - first
        keep.sort()
        return _Pair(all_days[keep], all_rates[keep])


class DailyFxTable:
    """An `FxRates` backed by daily series, with O(log n) lookups.

    Drop-in for `core.money.FxTable`: `add` accepts single spot rates and the
    fallback order -- exact date, most recent earlier date, the inverse pair,
    then `MissingRate` -- reads the same. It never returns a rate it does not
    have: a missing pair raises, because a silent 1.0 would value a USD
    position at par in EUR and nothing on screen would look wrong.
    """

    def __init__(self) -> None:
        self._pairs: dict[tuple[str, str], _Pair] = {}

    # -- loading -----------------------------------------------------------

    def add_series(self, frm: str, to: str, series: pd.Series) -> None:
        """Add a daily series of units of `to` per one `frm`, indexed by date."""
        f, _ = normalise_currency(frm)
        t, _ = normalise_currency(to)
        clean = series.dropna()
        if clean.empty:
            return
        index = pd.DatetimeIndex(clean.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        days = index.normalize().to_numpy().astype("datetime64[D]").astype("int64")
        rates = clean.to_numpy(dtype=float)
        self._store(f, t, days, rates)

    def add(self, frm: str, to: str, on: dt.date, rate: Decimal | str | float
            ) -> "DailyFxTable":
        """A single spot rate; the `FxTable.add` shape so tests can swap them."""
        f, _ = normalise_currency(frm)
        t, _ = normalise_currency(to)
        self._store(f, t, np.array([_days(on)], dtype="int64"),
                    np.array([float(Decimal(str(rate)))], dtype=float))
        return self

    def _store(self, f: str, t: str, days: np.ndarray, rates: np.ndarray) -> None:
        fresh = _Pair(np.asarray(days, dtype="int64"), np.asarray(rates, dtype=float))
        if len(fresh.days) > 1:
            fresh = _Pair(np.array([], dtype="int64"), np.array([], dtype=float)).merged(
                fresh.days, fresh.rates)
        existing = self._pairs.get((f, t))
        self._pairs[(f, t)] = fresh if existing is None else existing.merged(
            fresh.days, fresh.rates)

    # -- FxRates -----------------------------------------------------------

    def rate(self, frm: str, to: str, on: dt.date) -> Decimal:
        f, _ = normalise_currency(frm)
        t, _ = normalise_currency(to)
        if f == t:
            return Decimal("1")
        day = _days(on)
        direct = self._pairs.get((f, t))
        if direct is not None:
            value = direct.at_or_before(day)
            if value is not None:
                return _decimal(value)
        inverse = self._pairs.get((t, f))
        if inverse is not None:
            value = inverse.at_or_before(day)
            if value:
                return _decimal(1.0 / value)
        if direct is None and inverse is None:
            raise MissingRate(
                f"no FX rate for {f}->{t}: the pair was never loaded, so nothing "
                f"can be converted on {on}")
        raise MissingRate(
            f"no FX rate for {f}->{t} on or before {on}: the loaded history "
            f"starts later than that date")

    # -- introspection -----------------------------------------------------

    def latest(self, frm: str, to: str) -> tuple[dt.date, Decimal] | None:
        """The most recent rate held for the pair, direct or inverted."""
        f, _ = normalise_currency(frm)
        t, _ = normalise_currency(to)
        if f == t:
            return None
        direct = self._pairs.get((f, t))
        if direct is not None and len(direct.days):
            return (_EPOCH + dt.timedelta(days=int(direct.days[-1])),
                    _decimal(float(direct.rates[-1])))
        inverse = self._pairs.get((t, f))
        if inverse is not None and len(inverse.days) and inverse.rates[-1]:
            return (_EPOCH + dt.timedelta(days=int(inverse.days[-1])),
                    _decimal(1.0 / float(inverse.rates[-1])))
        return None

    def pairs(self) -> list[tuple[str, str]]:
        return sorted(self._pairs)

    def observations(self, frm: str, to: str) -> int:
        f, _ = normalise_currency(frm)
        t, _ = normalise_currency(to)
        pair = self._pairs.get((f, t))
        return 0 if pair is None else int(len(pair.days))

    def __len__(self) -> int:
        return len(self._pairs)


def currencies_needed(instruments: dict[str, Instrument] | Iterable[Instrument],
                      transactions: Iterable[Transaction],
                      base: str = BASE_CURRENCY) -> set[str]:
    """Every currency a conversion into `base` will be asked for.

    Quote currencies of the listings and currencies of the ledger rows, and
    nothing else. An instrument's *base* currency is deliberately excluded:
    a USD-base fund quoted in EUR on Xetra is valued in EUR, and a rate for
    USD would be fetched and never used.
    """
    target, _ = normalise_currency(base)
    insts = instruments.values() if isinstance(instruments, dict) else instruments
    codes: set[str] = set()
    for inst in insts:
        if inst.quote_currency:
            codes.add(normalise_currency(inst.quote_currency)[0])
    for txn in transactions:
        codes.add(normalise_currency(txn.currency)[0])
    codes.discard(target)
    return codes
