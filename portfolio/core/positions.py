"""Derive positions from the transaction log.

Positions are never stored. They are recomputed from the ledger every time,
which is what makes a partial sell, a realised P&L and a corrected transaction
all work without a migration.

Cost basis method: weighted average cost.
--------------------------------------------------
When you sell part of a holding, something has to decide which units left.
This module uses weighted average cost: the realised gain is the sale proceeds
minus the average cost of the units sold, and the remaining basis shrinks
proportionally.

Rejected alternative: FIFO, which most European tax regimes actually require
for capital gains reporting. FIFO needs a lot per purchase and turns this
module into a queue simulation. Average cost is the right default for *risk and
performance* reporting, which is what this tool does; tax reporting is
explicitly out of scope for v1. The choice is isolated in `_apply_sell` so
swapping in FIFO later is one function, not a rewrite.

Currency
--------
Cost basis converts at the FX rate of the *transaction date* and market value
at today's rate. Using one where the other belongs silently rewrites history.
Both are held: `Position` carries the base-currency figures the UI totals, and
the original transaction currency is preserved on the ledger.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections import defaultdict
from decimal import Decimal

from .models import Instrument, Transaction, TransactionType
from .money import BASE_CURRENCY, FxRates, Money, convert

__all__ = ["Position", "PriceQuote", "derive_positions", "InsufficientUnits",
           "DEFAULT_FRESHNESS_BUSINESS_DAYS", "business_days_since", "is_outdated"]

# A price older than this many trading days is not a current price.
#
# Five is a week of trading, which tolerates a long weekend, a public holiday
# and a provider that lags a day, without tolerating a series that has stopped
# updating. It is deliberately generous: the failure this catches is a listing
# frozen months ago, not one that missed yesterday.
#
# Business days, not calendar days, because a Monday check against a Friday
# close must not read as three days stale. Public holidays are not modelled --
# a market closed for a week would need the threshold raised rather than the
# rule weakened.
DEFAULT_FRESHNESS_BUSINESS_DAYS = 5


def business_days_since(last: dt.date, as_of: dt.date | None = None) -> int:
    """Trading days between `last` and `as_of`, weekends excluded."""
    import numpy as np
    end = as_of or dt.date.today()
    if last > end:
        return 0
    return int(np.busday_count(last, end))


def is_outdated(last: dt.date | dt.datetime | None,
                as_of: dt.date | None = None,
                max_business_days: int = DEFAULT_FRESHNESS_BUSINESS_DAYS) -> bool:
    """True if a price this old must not be shown as current.

    The motivating case: IS0C.DE resolves on a European venue, quotes EUR, and
    returns exactly 252 rows -- a full lookback window, so no row-count check
    fires. Its last observation is a year old. Every existing check passes it.
    Only the date betrays it.
    """
    if last is None:
        return False
    if isinstance(last, dt.datetime):
        last = last.date()
    return business_days_since(last, as_of) > max_business_days


class InsufficientUnits(ValueError):
    """A SELL for more units than the ledger says are held."""


@dataclasses.dataclass(frozen=True)
class PriceQuote:
    """A price with the provenance the UI is required to display.

    `as_of` and `is_stale` are not optional decoration. The spec forbids ever
    presenting delayed data as real time, so a quote that cannot say when it was
    taken cannot be shown as live.
    """
    price: Money
    as_of: dt.datetime | dt.date
    source: str = ""
    delay_minutes: int | None = None
    is_stale: bool = False          # True when this is a fallback last close

    def is_outdated(self, as_of: dt.date | None = None,
                    max_business_days: int = DEFAULT_FRESHNESS_BUSINESS_DAYS) -> bool:
        """Too old to present as a current price, whatever its provenance.

        Checked at fetch time as well as at resolution, because a listing that
        was fresh when it was chosen can stop updating afterwards.
        """
        return is_outdated(self.as_of, as_of, max_business_days)

    def staleness_note(self, as_of: dt.date | None = None,
                       max_business_days: int = DEFAULT_FRESHNESS_BUSINESS_DAYS
                       ) -> str | None:
        """The sentence the Holdings row must carry when the price is old."""
        if not self.is_outdated(as_of, max_business_days):
            return None
        when = self.as_of.date() if isinstance(self.as_of, dt.datetime) else self.as_of
        days = business_days_since(when, as_of)
        return (f"last observation {when} is {days} trading days old - "
                f"this is NOT a current price")


@dataclasses.dataclass
class Position:
    """A derived holding. Money fields are in the reporting currency (EUR)."""
    isin: str
    quantity: Decimal = Decimal("0")
    cost_basis: Money = dataclasses.field(default_factory=lambda: Money.zero(BASE_CURRENCY))
    realised_pnl: Money = dataclasses.field(default_factory=lambda: Money.zero(BASE_CURRENCY))
    dividends: Money = dataclasses.field(default_factory=lambda: Money.zero(BASE_CURRENCY))
    fees_paid: Money = dataclasses.field(default_factory=lambda: Money.zero(BASE_CURRENCY))
    quote: PriceQuote | None = None
    instrument: Instrument | None = None
    transaction_count: int = 0
    # Set when this position could not be fully derived -- currently only when
    # an FX rate was unavailable. Quantity is still correct; the money fields
    # are not, and must not be displayed as though they were.
    error: str | None = None

    @property
    def is_derivable(self) -> bool:
        return self.error is None

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    @property
    def is_watchlist(self) -> bool:
        """No transactions ever. A legitimate state, and what v2's simulator reads.

        Derived, never stored: a boolean flag on Instrument would be a second
        source of truth that drifts the first time a transaction is voided.
        """
        return self.transaction_count == 0

    @property
    def average_cost(self) -> Money:
        """Average cost per unit in the reporting currency."""
        if self.quantity == 0:
            return Money.zero(self.cost_basis.currency)
        return Money(self.cost_basis.amount / self.quantity, self.cost_basis.currency)

    def market_value(self, rates: FxRates | None = None,
                     on: dt.date | None = None) -> Money | None:
        """Current value in the reporting currency, or None without a quote."""
        if self.quote is None or self.quantity == 0:
            return None
        gross = Money(self.quote.price.amount * self.quantity, self.quote.price.currency)
        if gross.currency == self.cost_basis.currency:
            return gross
        if rates is None:
            # Refusing is the point: returning the unconverted number would add
            # USD to EUR one row later and nothing would look wrong.
            return None
        return convert(gross, self.cost_basis.currency, on or dt.date.today(), rates)

    def unrealised_pnl(self, rates: FxRates | None = None,
                       on: dt.date | None = None) -> Money | None:
        mv = self.market_value(rates, on)
        return None if mv is None else mv - self.cost_basis

    def unrealised_pnl_pct(self, rates: FxRates | None = None,
                           on: dt.date | None = None) -> Decimal | None:
        pnl = self.unrealised_pnl(rates, on)
        if pnl is None or self.cost_basis.amount == 0:
            return None
        return pnl.amount / self.cost_basis.amount


def _to_base(money: Money, on: dt.date, rates: FxRates | None,
             base: str) -> Money:
    if money.currency == base:
        return money
    if rates is None:
        raise ValueError(
            f"a {money.currency} transaction on {on} needs an FX rate to report "
            f"in {base}; refusing to add unconverted currencies")
    return convert(money, base, on, rates)


def _apply_sell(pos: Position, units: Decimal, proceeds: Money,
                fees: Money) -> None:
    """Weighted-average-cost sell. Swap this one function for FIFO if needed."""
    if units > pos.quantity:
        raise InsufficientUnits(
            f"{pos.isin}: SELL of {units} units but only {pos.quantity} held. "
            f"Check for a missing BUY or a duplicated SELL.")

    if units == pos.quantity:
        # Selling out entirely: the basis relieved is, by definition, all of it.
        # Computing units * average instead would divide and re-multiply, and
        # for a basis that does not divide evenly (31.00 over 3 units) the
        # residual lands in realised P&L -- a cent of phantom profit on a
        # closed position, which is exactly the kind of error nobody chases.
        basis_sold = pos.cost_basis
    else:
        basis_sold = Money(pos.average_cost.amount * units, pos.cost_basis.currency)

    pos.realised_pnl = pos.realised_pnl + (proceeds - basis_sold) - fees
    pos.cost_basis = pos.cost_basis - basis_sold
    pos.quantity -= units


def derive_positions(transactions: list[Transaction],
                     instruments: dict[str, Instrument] | None = None,
                     rates: FxRates | None = None,
                     quotes: dict[str, PriceQuote] | None = None,
                     base: str = BASE_CURRENCY,
                     strict: bool = True) -> dict[str, Position]:
    """Replay the ledger into positions, oldest first.

    Instruments with no transactions are returned as empty positions rather
    than omitted -- that is the watchlist, and dropping them would hide exactly
    the instruments the what-if simulator needs.

    `strict` controls what happens when one instrument cannot be derived, which
    in practice means a missing FX rate. Strict raises, which is right for
    tests and for any caller that needs a complete answer. Non-strict isolates
    the failure: that instrument gets its `error` set and its money fields left
    at zero, and every other holding still derives.

    The distinction matters because an FX provider being unreachable should not
    blank a portfolio that is mostly EUR. What it must never do is guess a rate
    -- `core.money` raises rather than defaulting to parity, and this only
    decides how far that refusal propagates.
    """
    instruments = instruments or {}
    quotes = quotes or {}
    positions: dict[str, Position] = {
        isin: Position(isin=isin, instrument=inst)
        for isin, inst in instruments.items()
    }

    by_isin: dict[str, list[Transaction]] = defaultdict(list)
    for t in transactions:
        by_isin[t.isin].append(t)

    for isin, txns in by_isin.items():
        pos = positions.setdefault(isin, Position(isin=isin,
                                                  instrument=instruments.get(isin)))
        try:
            _replay(pos, txns, rates, base)
        except (ValueError, LookupError) as exc:
            if strict or isinstance(exc, InsufficientUnits):
                raise
            # Quantity needs no conversion, so derive it separately rather
            # than keeping whatever the aborted replay left behind. Without
            # this the position reads as closed and disappears from the table,
            # which is a worse failure than the one being handled.
            positions[isin] = Position(
                isin=isin, quantity=_quantity_only(txns),
                instrument=instruments.get(isin),
                transaction_count=len(txns), error=str(exc))
        q = quotes.get(isin)
        if q is not None:
            positions[isin].quote = q

    for isin, q in quotes.items():
        if isin in positions and positions[isin].quote is None:
            positions[isin].quote = q
    return positions


def _quantity_only(txns: list[Transaction]) -> Decimal:
    """Units held, ignoring money entirely.

    Quantity is FX-free, so it survives a missing rate. Deriving it separately
    is what keeps an unconvertible holding visible instead of reading as closed.
    """
    quantity = Decimal("0")
    for t in sorted(txns, key=lambda x: (x.date, x.id)):
        if t.type is TransactionType.BUY:
            quantity += t.quantity
        elif t.type is TransactionType.SELL:
            quantity -= t.quantity
    return quantity


def _replay(pos: Position, txns: list[Transaction], rates: FxRates | None,
            base: str) -> None:
    """Apply one instrument's transactions to its position, in date order."""
    for t in sorted(txns, key=lambda x: (x.date, x.id)):
        pos.transaction_count += 1
        fees_base = _to_base(t.fee_money, t.date, rates, base)
        pos.fees_paid = pos.fees_paid + fees_base

        if t.type is TransactionType.BUY:
            gross_base = _to_base(t.gross, t.date, rates, base)
            # Fees are capitalised into the basis: they are part of what the
            # units cost you, and excluding them overstates every gain.
            pos.cost_basis = pos.cost_basis + gross_base + fees_base
            pos.quantity += t.quantity

        elif t.type is TransactionType.SELL:
            proceeds_base = _to_base(t.gross, t.date, rates, base)
            _apply_sell(pos, t.quantity, proceeds_base, fees_base)

        elif t.type is TransactionType.DIVIDEND:
            gross = t.gross
            if gross.amount == 0:
                gross = Money(t.price_per_unit, t.currency)
            pos.dividends = pos.dividends + _to_base(gross, t.date, rates, base) - fees_base

        elif t.type is TransactionType.FEE:
            charge = _to_base(t.gross, t.date, rates, base)
            pos.realised_pnl = pos.realised_pnl - fees_base - charge


def total_market_value(positions: dict[str, Position], rates: FxRates | None = None,
                       on: dt.date | None = None,
                       base: str = BASE_CURRENCY) -> tuple[Money, list[str]]:
    """Portfolio value, plus the ISINs that could not be valued.

    Returning the unpriced list rather than skipping quietly is deliberate: a
    total that silently omits a holding is wrong in a way nobody notices.
    """
    total = Money.zero(base)
    missing: list[str] = []
    for isin, pos in positions.items():
        if not pos.is_open:
            continue
        mv = pos.market_value(rates, on)
        if mv is None:
            missing.append(isin)
            continue
        total = total + mv
    return total, missing


def weights(positions: dict[str, Position], rates: FxRates | None = None,
            on: dt.date | None = None,
            base: str = BASE_CURRENCY) -> dict[str, Decimal]:
    """Capital weights of open, priced positions. Sums to exactly 1, or is empty."""
    total, _ = total_market_value(positions, rates, on, base)
    if total.amount == 0:
        return {}
    out: dict[str, Decimal] = {}
    for isin, pos in positions.items():
        if not pos.is_open:
            continue
        mv = pos.market_value(rates, on)
        if mv is None:
            continue
        out[isin] = mv.amount / total.amount
    return out
