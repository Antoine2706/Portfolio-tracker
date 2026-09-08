"""Filling in columns that did not exist when the file was written.

The bug this exists to fix
--------------------------
`INSTRUMENT_COLUMNS` gained seven fields -- broker, tradeable, tob_rate,
tob_observed, half_spread_bps, spread_observed, buy_tax_rate -- appended so
that "a CSV written by an older build still loads". It did load. Every one of
those fields took its default, and the default for `tob_rate` is None,
meaning NOT RECORDED. The cost model refuses to price a trade in an
instrument whose tax band is not recorded, by design. So on a real book
written by the previous build, *every* instrument was unpriceable and the
backtest could not run at all.

Both halves of that behaved exactly as specified. Graceful loading did what
graceful loading is for, and refusing to guess a tax rate is the convention
this project deliberately adopted after a guessed tax rate was found to be
wrong by a factor of eleven. The two composed into a tool that cannot run.

That is the shape of the defect worth naming: neither piece was wrong, and
no test failed, because no test loaded an old file and then tried to price a
trade from it.

What the migration does, and what it refuses to do
--------------------------------------------------
It fires on exactly one condition: **the CSV header is missing these
columns**. Not on a blank cell. That distinction is the whole design.

A blank `tob_rate` in a file that *has* the column is a recorded fact -- it
says "this rate is not established, refuse to price it", which is precisely
what the gold ETC needs. A migration triggered by blankness would overwrite
that with a plausible number every time the file was loaded, which is the
failure the column was added to prevent. Triggering on the header instead
makes the migration one-shot by construction: after it writes, the columns
exist, and it never looks at that file again.

What it fills is what a document says, and the list is short on purpose:

    tob_rate       from a contract note, per ISIN, or not at all. See
                   OBSERVED_CONTRACT_NOTES.
    tob_observed   True where a note exists, absent where none does.
    buy_tax_rate   the French FTT, for French-domiciled shares -- the one
                   thing still derived, because it follows from domicile and
                   asset class rather than from a registration nobody can see.

It used to derive `tob_rate` from the asset class as well, on the reasoning
that a UCITS fund and a debt security do not sit in the same band. That was
true and insufficient. IE00BGDQ0L74 is an accumulating iShares UCITS ETF at
the same broker, in the same week, as four others that paid 0.1200%; its
corrected note says 1.3198%. The band turns on per-compartment Belgian
registration, which is on no field of the row. The table went, and an unread
rate is refused rather than guessed.

It does **not** invent a broker, because which institution holds a position
is not a property of the instrument and cannot be derived from anything on
the row. It does not invent a spread either: that is now estimated per
instrument from its own price history and reported as a third tier of
evidence, neither observed nor assumed.

Everything it writes names the document it came from, and
`CostModel.provenance()` keeps reporting the observed-against-estimated split
underneath every result afterwards. A derived number that presents itself as
a measurement would be worse than the crash it replaces.
"""

from __future__ import annotations

import csv
import dataclasses
import pathlib

from ..core.models import AssetClass, Instrument
from ..core.taxes import FRENCH_FTT_RATE

__all__ = ["TRADING_COLUMNS", "OBSERVED_CONTRACT_NOTES", "Backfill",
           "MigrationReport", "derived_facts", "backfill_trading_facts",
           "header_of", "missing_columns"]

# The columns added after the first release. A file whose header lacks any of
# them predates them and needs backfilling.
TRADING_COLUMNS = ("broker", "tradeable", "tob_rate", "tob_observed",
                   "half_spread_bps", "spread_observed", "buy_tax_rate",
                   "buyable")


@dataclasses.dataclass(frozen=True)
class ObservedRate:
    """A rate read off an actual document, with the document named."""
    tob_rate: float
    source: str


# Every transaction tax rate read off an actual MeDirect contract note. Each
# reconciles to the cent against the notional on the same document, and
# `tests/test_migrations.py` checks that arithmetic rather than trusting the
# transcription.
#
# Belgian TOB is a property of the instrument and the investor's regime
# rather than of the broker, which is why this is keyed by ISIN alone.
OBSERVED_CONTRACT_NOTES: dict[str, ObservedRate] = {
    "DE000A2QP372": ObservedRate(
        tob_rate=0.0012,
        source=("MeDirect contract note, 2 February 2026, XAMS: 2.43 EUR on a "
                "2,024.87 EUR notional, 0.1200%")),
    "IE00BKM4GZ66": ObservedRate(
        tob_rate=0.0012,
        source=("MeDirect contract note, 20 February 2026, GSEI: 2.38 EUR on "
                "a 1,985.86 EUR notional, 0.1198%")),
    "IE00BMW42520": ObservedRate(
        tob_rate=0.0012,
        source=("MeDirect contract note, 20 February 2026, GSEI: 2.40 EUR on "
                "a 2,001.89 EUR notional, 0.1199%")),
    "IE00BMC38736": ObservedRate(
        tob_rate=0.0012,
        source=("MeDirect contract note, 23 February 2026, XETA: 2.35 EUR on "
                "a 1,955.84 EUR notional, 0.1202%")),
    "IE00BGDQ0L74": ObservedRate(
        tob_rate=0.0132,
        source=("MeDirect corrected contract note, 23 February 2026, XETA: "
                "26.13 EUR on a 1,979.80 EUR notional, 1.3198% -- ELEVEN "
                "TIMES the band four other iShares accumulating ETFs on the "
                "same account paid on the same schedule")),
    "FR0000121972": ObservedRate(
        tob_rate=0.0035,
        source=("MeDirect contract note, 2 March 2026, XPAR: 8.46 EUR on a "
                "2,418.30 EUR notional, 0.3500%")),
}

# There is no table here on purpose, and the empty space is the point.
#
# There used to be one: ETF and FUND took the fund band, EQUITY took the
# equity band, ETC and OTHER stayed unpriced. It was reasoned from the only
# thing an ISIN can answer -- whether the instrument is a fund or a debt
# security -- and every rate it produced was marked assumed and flagged for
# confirmation. It was still wrong, and not at the margin.
#
# IE00BGDQ0L74 is an accumulating iShares UCITS ETF at the same broker, on the
# same schedule, in the same week as four others that paid 0.1200%. It paid
# 1.3198%. The difference is not the share class, not the issuer, not the
# asset class and not the venue: it is per-compartment Belgian registration,
# which appears nowhere on the instrument's label and cannot be derived from
# any field in the row. A table that reproduced the observed rate for four
# instruments and was out by a factor of eleven on the fifth is not a
# conservative default; it is a confident number about something it could not
# see, which is the failure mode this project keeps finding.
#
# So an unread rate is refused, not guessed, and the migration fills in only
# what a document says. The cost is that a new instrument is unpriceable until
# its first contract note arrives, which is the correct cost.


@dataclasses.dataclass(frozen=True)
class Backfill:
    """One field filled in, and the reason it was given that value."""
    isin: str
    field: str
    value: object
    because: str

    def line(self) -> str:
        shown = "not recorded" if self.value is None else self.value
        return f"{self.isin} {self.field} = {shown} ({self.because})"


def derived_facts(inst: Instrument) -> list[Backfill]:
    """What a document says about one instrument, and nothing else.

    An instrument with a contract note takes the rate off it, marked observed:

    >>> banks = Instrument("DE000A2QP372", "iShares EURO STOXX Banks 30-15")
    >>> [(b.field, b.value) for b in derived_facts(banks)]
    [('tob_rate', 0.0012), ('tob_observed', True)]

    Including the one that falsified the table this function used to have. It
    is an accumulating iShares UCITS ETF like the four that paid 0.1200%, and
    it paid eleven times that:

    >>> yield_ = Instrument("IE00BGDQ0L74", "iShares European Property Yield")
    >>> [(b.field, b.value) for b in derived_facts(yield_)]
    [('tob_rate', 0.0132), ('tob_observed', True)]

    An instrument with no note gets no rate. It stays unpriceable, which is the
    honest state and the one the cost model is built to refuse on -- and it is
    now the answer for an unknown ETF as much as for an ETC, because the label
    turned out not to predict the band:

    >>> etc = Instrument("IE00B579F325", "Invesco Physical Gold ETC",
    ...                  AssetClass.ETC)
    >>> derived_facts(etc)
    []
    >>> derived_facts(Instrument("IE00B4L5Y983", "iShares Core MSCI World"))
    []

    The French FTT is the one thing still derived, because it follows from the
    instrument's domicile and asset class rather than from a registration
    nobody can see:

    >>> share = Instrument("FR0000121972", "Schneider Electric SE",
    ...                    AssetClass.EQUITY)
    >>> [(b.field, b.value) for b in derived_facts(share)]
    [('tob_rate', 0.0035), ('tob_observed', True), ('buy_tax_rate', 0.004)]
    """
    out: list[Backfill] = []
    note = OBSERVED_CONTRACT_NOTES.get(inst.isin)
    if note is not None:
        out.append(Backfill(inst.isin, "tob_rate", note.tob_rate, note.source))
        out.append(Backfill(inst.isin, "tob_observed", True, note.source))

    # France taxes acquisitions of shares in French-headquartered companies
    # above 1bn EUR of market capitalisation. The threshold is not derivable
    # from the row, so this errs towards charging it: overstating a cost makes
    # a strategy look worse than it is, and that is the safe direction.
    if inst.asset_class is AssetClass.EQUITY and inst.isin.startswith("FR"):
        out.append(Backfill(
            inst.isin, "buy_tax_rate", FRENCH_FTT_RATE,
            "assumed: a French share, so the FTT applies on purchases if its "
            "market capitalisation is above 1bn EUR"))
    return out


@dataclasses.dataclass(frozen=True)
class MigrationReport:
    """What a migration filled in, ready to print beside the data it changed."""
    path: pathlib.Path
    columns_added: tuple[str, ...]       # what the header did not have
    filled: tuple[Backfill, ...]
    unresolved: tuple[str, ...]          # instruments still unpriceable
    brokers_missing: tuple[str, ...]     # instruments with no broker recorded
    backup: pathlib.Path | None = None
    persisted: bool = True
    error: str = ""

    def __bool__(self) -> bool:
        """True when the migration actually did something."""
        return bool(self.columns_added)

    def lines(self) -> list[str]:
        # Claim the consequence only where it applies. A file missing just
        # `broker` was inconvenient; a file missing `tob_rate` could not price
        # a single trade, and the two should not be described in the same
        # words merely because one code path handles both.
        if "tob_rate" in self.columns_added:
            headline = (f"Migrated {self.path}: it was written before the "
                        f"trading-cost columns existed, so every instrument in "
                        f"it was unpriceable and no backtest could run.")
        else:
            headline = (f"Migrated {self.path}: it was missing the "
                        f"{', '.join(self.columns_added)} column(s), now added.")
        out = [headline, ""]
        if self.filled:
            out.append("Filled in, each from the document named:")
            out += [f"  - {b.line()}" for b in self.filled]
        if self.unresolved:
            out += ["", "Still not priceable, deliberately:"]
            out += [f"  - {isin}: no contract note has been read for it. The "
                    f"band cannot be derived from the asset class -- one "
                    f"holding here pays eleven times what an identical-looking "
                    f"one pays -- so it stays refused rather than guessed."
                    for isin in self.unresolved]
        if self.brokers_missing:
            out += ["", "No broker recorded (which institution holds a position "
                    "is not a property of the instrument, so it cannot be "
                    "derived):"]
            out += [f"  - {isin}" for isin in self.brokers_missing]
        out += ["", "Record what you know with `portfolio instruments set`; "
                "until then the report under every result says which inputs "
                "are evidence and which are assumptions."]
        if not self.persisted:
            out += ["", f"NOT WRITTEN BACK: {self.error} The values above are "
                    f"in memory for this run only and will be derived again "
                    f"next time."]
        elif self.backup is not None:
            out += ["", f"The file as it was is kept at {self.backup.name}."]
        return out


def header_of(path: pathlib.Path) -> list[str]:
    """The CSV header, or an empty list if there is no readable file."""
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return next(csv.reader(fh), [])
    except (OSError, UnicodeDecodeError):
        return []


def missing_columns(path: pathlib.Path) -> list[str]:
    """Trading columns absent from the file's header.

    The trigger for the migration, and the reason it cannot run twice or
    trample a deliberately blank cell: after it writes, the header carries
    every column and this returns nothing forever after.
    """
    header = header_of(path)
    if not header:
        return []
    present = {c.strip() for c in header}
    return [c for c in TRADING_COLUMNS if c not in present]


def backfill_trading_facts(instruments: dict[str, Instrument],
                           path: pathlib.Path,
                           fillable: "set[str] | None" = None) -> MigrationReport:
    """Apply `derived_facts` across a whole book, in place.

    `fillable` is the set of columns the file does not have, and it is a hard
    limit rather than a hint: **a column that is present in the header is
    authoritative, including when its cell is blank.** A blank `tob_rate`
    under a `tob_rate` header is a recorded statement that the rate is not
    established, which is exactly what the gold ETC needs and exactly what a
    derived default would destroy. Only a column that is not there at all can
    be said to have no answer in it.

    That distinction was not free. The first version of this took the whole
    row as fillable whenever any trading column was missing, which overwrote a
    deliberately blank rate in a file that had that column -- reintroducing
    the plausible-wrong-number failure the column was added to prevent, inside
    the migration meant to preserve it.

    Two further guards: a value already recorded is never replaced, and
    `tob_observed` travels with the rate it describes, because a flag saying
    "read off a contract note" with no rate behind it is an estimate wearing
    the label of evidence.
    """
    allowed = set(TRADING_COLUMNS) if fillable is None else set(fillable)
    filled: list[Backfill] = []
    unresolved: list[str] = []
    brokers: list[str] = []
    for isin in sorted(instruments):
        inst = instruments[isin]
        entries = derived_facts(inst)
        set_rate = any(e.field == "tob_rate" for e in entries) and (
            "tob_rate" in allowed and inst.tob_rate is None)
        for entry in entries:
            if entry.field not in allowed:
                continue
            if entry.field in ("tob_rate", "tob_observed") and not set_rate:
                continue
            if entry.field == "buy_tax_rate" and inst.buy_tax_rate:
                continue
            setattr(inst, entry.field, entry.value)
            filled.append(entry)
        if inst.tob_rate is None:
            unresolved.append(isin)
        if not inst.broker:
            brokers.append(isin)
    return MigrationReport(path=path,
                           columns_added=tuple(c for c in TRADING_COLUMNS
                                               if c in allowed),
                           filled=tuple(filled), unresolved=tuple(unresolved),
                           brokers_missing=tuple(brokers))
