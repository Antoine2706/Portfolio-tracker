"""The real MeDirect documents, transcribed, as the first external check.

Every number this project has produced until now was computed by the tracker
from a ledger the tracker also holds, and checked against nothing outside
itself. A test suite in that position can only catch inconsistency, never
error: if the ledger and the analytics agree on a wrong share count, both
halves pass.

These are documents from the account. Seven trade confirmations and one
holdings statement, dated four months apart, produced by a bank that has
never seen this code. Reconstructing the statement from the confirmations is
the first thing here that can fail for a reason outside the repository.

Two facts on the statement that matter beyond the arithmetic:

*Gold is absent.* IE00B579F325 is held at Keytrade, not MeDirect, so it does
not appear. That is a check on the broker split rather than an omission: a
tracker that put every holding at one institution would reconcile to a total
that does not match any statement the account actually receives.

*The bank's acquisition values are pre-tax on every line.* Its cost basis
excludes the transaction tax paid on acquisition, so its realised gain is
overstated by that amount. The tracker includes acquisition costs in the
basis, which is the correct treatment and disagrees with the bank by design.
`test_reconciliation.py` prints the difference rather than adopting either
figure silently.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal

from ..core.models import Transaction, TransactionType

__all__ = ["CONFIRMATIONS", "STATEMENT", "STATEMENT_DATE", "STATEMENT_TOTAL",
           "ACCOUNT_TYPE", "Confirmation", "StatementLine", "ledger"]


@dataclasses.dataclass(frozen=True)
class Confirmation:
    """One trade confirmation, transcribed field by field."""
    date: dt.date
    isin: str
    venue: str
    side: str
    quantity: int
    price: Decimal
    notional: Decimal
    commission: Decimal
    tob: Decimal
    tob_rate: Decimal
    other: Decimal = Decimal("0")
    other_note: str = ""

    @property
    def implied_tob_rate(self) -> Decimal:
        return self.tob / self.notional

    @property
    def exact_price(self) -> Decimal:
        """The notional per share, which is what was actually paid.

        The confirmation prints a price rounded to two decimals and a notional
        that is not their product: 207 shares at a printed 9.67 is 2,001.69,
        and the note says 2,001.89. The rounded figure is a display of the
        average execution price; the notional is the money that moved. Using
        the printed price for the cost basis would be off by 20 cents on that
        line and by 23 on another, which is small until it is subtracted from
        a sale price to compute a taxable gain.
        """
        return self.notional / Decimal(self.quantity)


# MeDirect trade confirmations, February to June 2026. `notional` is the
# figure the confirmation prints, not quantity x price recomputed here, so
# that a transcription error shows up as a failed reconciliation rather than
# being smoothed over.
CONFIRMATIONS: tuple[Confirmation, ...] = (
    Confirmation(dt.date(2026, 2, 2), "DE000A2QP372", "XAMS", "BUY",
                 114, Decimal("17.76"), Decimal("2024.87"), Decimal("0.00"),
                 Decimal("2.43"), Decimal("0.001200")),
    Confirmation(dt.date(2026, 2, 20), "IE00BKM4GZ66", "GSEI", "BUY",
                 46, Decimal("43.17"), Decimal("1985.86"), Decimal("0.00"),
                 Decimal("2.38"), Decimal("0.001198")),
    Confirmation(dt.date(2026, 2, 20), "IE00BMW42520", "GSEI", "BUY",
                 207, Decimal("9.67"), Decimal("2001.89"), Decimal("0.00"),
                 Decimal("2.40"), Decimal("0.001199")),
    Confirmation(dt.date(2026, 2, 23), "IE00BMC38736", "XETA", "BUY",
                 32, Decimal("61.12"), Decimal("1955.84"), Decimal("0.00"),
                 Decimal("2.35"), Decimal("0.001202")),
    # The one that falsified the derived-band table: same issuer, same
    # domicile, same asset class, same broker, same week as the four above.
    Confirmation(dt.date(2026, 2, 23), "IE00BGDQ0L74", "XETA", "BUY",
                 380, Decimal("5.21"), Decimal("1979.80"), Decimal("0.00"),
                 Decimal("26.13"), Decimal("0.013198"),
                 other_note="corrected confirmation"),
    Confirmation(dt.date(2026, 3, 2), "FR0000121972", "XPAR", "BUY",
                 9, Decimal("268.70"), Decimal("2418.30"), Decimal("7.00"),
                 Decimal("8.46"), Decimal("0.003500"),
                 other=Decimal("9.67"), other_note="French FTT, 0.4000%"),
    Confirmation(dt.date(2026, 6, 18), "IE00BMC38736", "JPEU", "SELL",
                 16, Decimal("107.14"), Decimal("1714.28"), Decimal("0.00"),
                 Decimal("2.06"), Decimal("0.001202"),
                 other=Decimal("73.64"),
                 other_note="capital gains tax, 10.00% of a 736.36 gain"),
)


@dataclasses.dataclass(frozen=True)
class StatementLine:
    isin: str
    name: str
    quantity: int
    price: Decimal
    value: Decimal


STATEMENT_DATE = dt.date(2026, 6, 30)
ACCOUNT_TYPE = "Execution Only"

# MeDirect holdings statement, 30 June 2026. Prices are the bank's own closing
# marks and are not expected to equal a data provider's to the cent.
STATEMENT: tuple[StatementLine, ...] = (
    StatementLine("FR0000121972", "Schneider Electric", 9,
                  Decimal("285.40"), Decimal("2568.60")),
    StatementLine("IE00BKM4GZ66", "iShares Core MSCI EM IMI", 46,
                  Decimal("48.80"), Decimal("2244.80")),
    StatementLine("DE000A2QP372", "iShares EURO STOXX Banks 30-15", 114,
                  Decimal("19.55"), Decimal("2228.70")),
    StatementLine("IE00BGDQ0L74", "iShares European Property Yield", 380,
                  Decimal("4.89"), Decimal("1858.20")),
    StatementLine("IE00BMW42520", "iShares MSCI Europe Industrials", 207,
                  Decimal("9.77"), Decimal("2022.39")),
    StatementLine("IE00BMC38736", "VanEck Semiconductor", 16,
                  Decimal("109.16"), Decimal("1746.56")),
)

STATEMENT_TOTAL = Decimal("12669.25")


def ledger() -> "list[Transaction]":
    """The seven confirmations as ledger entries.

    `fees` carries the commission and every tax the confirmation charges on
    that trade EXCEPT the capital gains tax, which is not a cost of the
    transaction: it is a tax on a gain that accrued over the holding period
    and belongs in its own line, charged once the gain is realised. Folding it
    into the trade's fees would make it look proportional to the notional,
    which is exactly the thing it is not.
    """
    out = []
    for c in CONFIRMATIONS:
        fees = c.commission + c.tob
        if c.side == "BUY":
            fees += c.other                     # the French FTT is a trade cost
        out.append(Transaction(
            date=c.date, isin=c.isin, type=TransactionType(c.side),
            quantity=Decimal(c.quantity), price_per_unit=c.exact_price,
            currency="EUR", fees=fees,
            note=f"MeDirect {c.venue} confirmation"))
    return out
