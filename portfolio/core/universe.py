"""Instrument lifecycle: what may be deleted, what may only be deactivated.

The rule that matters: deleting an instrument that has transactions against it
would orphan ledger rows and silently change historical portfolio values. Last
month's number would quietly become a different number, with nothing on screen
to say why. So it is refused, and deactivation is offered instead.

Pure logic, no I/O, so the rules are testable and the UI cannot route around
them by writing to the store directly -- the store calls these too.
"""

from __future__ import annotations

import dataclasses

from .models import Instrument, Transaction

__all__ = ["DeletionCheck", "check_deletable", "deactivate", "reactivate",
           "InstrumentInUse"]


class InstrumentInUse(ValueError):
    """A delete that would orphan ledger rows."""


@dataclasses.dataclass(frozen=True)
class DeletionCheck:
    """Answer to 'can this be deleted', with the numbers the dialog must show."""
    isin: str
    allowed: bool
    transaction_count: int
    reason: str
    alternative: str = ""

    def __bool__(self) -> bool:
        return self.allowed


def check_deletable(isin: str, transactions: list[Transaction]) -> DeletionCheck:
    """Decide whether `isin` can be hard deleted.

    The confirmation dialog is required to show `transaction_count`. For a
    deletable instrument it is always zero, which is precisely the reassurance
    the user needs before clicking through.
    """
    n = sum(1 for t in transactions if t.isin == isin)
    if n:
        return DeletionCheck(
            isin=isin, allowed=False, transaction_count=n,
            reason=(f"{n} transaction{'s' if n != 1 else ''} reference this "
                    f"instrument. Deleting it would orphan those ledger rows and "
                    f"change historical portfolio values."),
            alternative=("Deactivate instead: it disappears from the holdings "
                         "view but stays in the history and stays available for "
                         "the transaction log. Deactivation is reversible."))
    return DeletionCheck(
        isin=isin, allowed=True, transaction_count=0,
        reason="No transactions reference this instrument; it is a watchlist "
               "entry and deleting it is just tidying.")


def deactivate(instrument: Instrument, reason: str = "") -> Instrument:
    """Hide from holdings, keep in history. Reversible, and never cascades."""
    instrument.active = False
    if reason:
        instrument.note = (instrument.note + "\n" if instrument.note else "") + \
            f"deactivated: {reason}"
    return instrument


def reactivate(instrument: Instrument) -> Instrument:
    instrument.active = True
    return instrument
