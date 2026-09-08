"""Transaction tax rates, as facts rather than as policy.

These live in `core/` because two layers need them and neither should depend
on the other: `agents/execution.py` prices a trade with them, and
`data/migrations.py` writes them into an instrument record that predates the
column. Putting them in either would make the data layer import the decision
layer or the reverse, for three numbers.

Nothing here is computed. It is reference data about jurisdictions, in the
same category as the issuer prefixes in `naming.py`, and it carries the
provenance of each number with it: which band a given instrument sits in
turns on facts -- fund or debt security, accumulating or distributing,
registered for public distribution in Belgium or not -- that cannot be
derived from an ISIN, so a rate applied without a contract note behind it is
an assumption and has to travel labelled as one.

Why that matters more than it looks
-----------------------------------
The first version of this project's cost model assumed 1.32% and concluded
that turnover-based strategies were not viable in the account it was built
for. A contract note said 0.12%. The conclusion inverted, and no line of code
had been wrong: the input had been guessed. Every rate below is therefore
either observed, and says where, or is marked as a band that must be
confirmed.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

__all__ = ["OBSERVED_TOB_RATE", "BELGIAN_TOB_BANDS", "FRENCH_FTT_RATE",
           "BELGIAN_TOB_CAP", "BELGIAN_CAPITAL_GAINS_RATE",
           "BELGIAN_CAPITAL_GAINS_ALLOWANCE", "CapitalGainsTax",
           "GainsLedger", "Disposal"]

# Read off a MeDirect contract note, 2 February 2026: 2.43 EUR of tax on a
# 2,024.87 EUR notional in DE000A2QP372, an accumulating UCITS ETF domiciled
# in Germany. Charged on both purchase and sale.
OBSERVED_TOB_RATE = 0.0012

# The bands, for reference when confirming the other holdings. Read each off
# its own contract note; an ISIN does not determine which one applies.
#
# All three are now confirmed against MeDirect contract notes, and the third
# line of this table is the one that matters. IE00BGDQ0L74 is an accumulating
# iShares UCITS ETF traded at the same broker in the same week as four others
# that paid 0.1200%; it paid 1.3198%. The band turns on per-compartment
# Belgian registration, which is on no field of an instrument record, so this
# table is for *checking* a note against, never for deriving a rate from a
# label. `data/migrations.py` used to do the latter and was out by eleven
# times on exactly this instrument.
BELGIAN_TOB_BANDS = {
    "observed_medirect_etf": 0.0012,      # 4 ETFs, Feb 2026
    "accumulating_registered": 0.0132,    # IE00BGDQ0L74, corrected note
    "equity": 0.0035,                     # FR0000121972, Mar 2026
    "unknown": None,
}

# Per transaction, in the 0.12% band.
BELGIAN_TOB_CAP = 1_300.0

# France taxes acquisitions of shares in French-headquartered companies above
# 1bn EUR of market capitalisation. Buy side only, regardless of the buyer's
# residence or the venue used, which is why anything pricing a trade with it
# has to know which side the trade is.
#
# Confirmed: MeDirect contract note, 2 March 2026, FR0000121972 -- 9.67 EUR on
# a 2,418.30 EUR notional, 0.4000%, reconciling to the cent.
FRENCH_FTT_RATE = 0.004

# Belgian capital gains tax on realised gains. Read off the MeDirect sale
# confirmation of 18 June 2026: 73.64 EUR on a realised gain of 736.36 EUR in
# IE00BMC38736, exactly 10.00%.
#
# This is structurally unlike every other rate in this file and the difference
# is the reason `agents/execution.py` cannot treat it as one more line item:
#
#   - It is proportional to the embedded GAIN, not to the notional. Selling
#     2,000 EUR of a position that has doubled costs 100 EUR; selling 2,000
#     EUR of a flat position costs nothing. Turnover in euros stops predicting
#     cost, which is what invalidates a breakeven-turnover figure computed
#     without it.
#   - It is one-sided. Buys never pay it, which is a structural argument for
#     directing new money over rebalancing that no spread or commission makes.
#   - It is annual and path dependent: an exempt tranche per calendar year
#     means the marginal rate is zero until cumulative realised gains cross it
#     and 10% after, so a flat rate is wrong in both directions.
#   - It is lot dependent. Two holdings of equal size and equal risk are not
#     equally cheap to trim if one is up 78% and the other down 6%.
BELGIAN_CAPITAL_GAINS_RATE = 0.10

# The annual exempt tranche, below which the marginal rate is zero. A
# PARAMETER, not a constant of nature: the figure is indexed and has to be
# confirmed for the filing year in question. It is here, named, rather than a
# literal inside a function, so that changing it is one edit in one place and
# so that a reader can see it is an assumption about a threshold rather than a
# rate read off a document.
BELGIAN_CAPITAL_GAINS_ALLOWANCE = 10_000.0


@dataclasses.dataclass(frozen=True)
class CapitalGainsTax:
    """A tax on realised gains, with an annual exempt tranche.

    The marginal formula, which is the whole of it:

        tax(g | c) = rate * [ max(c + g - allowance, 0) - max(c - allowance, 0) ]

    where `c` is the gains already realised this calendar year and `g` is the
    new one. Written this way rather than as `rate * max(total - allowance, 0)`
    computed at year end, because the tool has to answer "what does selling
    this cost me *now*", and the answer depends on where in the year it falls.

    The three regimes fall out of it rather than being special-cased:

    Wholly inside the tranche, nothing is due.

    >>> tax = CapitalGainsTax(rate=0.10, allowance=10_000.0)
    >>> tax.charge(4_000.0, realised=0.0)
    0.0

    Wholly above it, the full rate applies.

    >>> round(tax.charge(4_000.0, realised=12_000.0), 2)
    400.0

    And a gain that straddles the threshold is taxed only on the part above
    it, which is the case a flat rate gets wrong in both directions and the
    reason this is not one multiplication.

    >>> round(tax.charge(4_000.0, realised=8_000.0), 2)
    200.0

    A loss reduces the year's cumulative gain, and the refund is limited to
    what was actually paid rather than being the rate times the loss. A 3,000
    loss against 12,000 realised takes the year to 9,000, back below the
    tranche: only the 2,000 that was above it is refunded, so 200 rather than
    300. Symmetry with the straddle case above, and it falls out of the same
    expression instead of needing a rule.

    >>> round(tax.charge(-3_000.0, realised=12_000.0), 2)
    -200.0

    Below the tranche a loss changes nothing, because nothing was due.

    >>> tax.charge(-3_000.0, realised=2_000.0)
    0.0

    What this ASSUMES, and it is an assumption rather than a reading of a
    document: that losses offset gains within the same calendar year, and that
    nothing carries across years. Neither is modelled from a source. The rate
    is observed -- 73.64 EUR on a 736.36 EUR gain, exactly 10.00% -- and the
    tranche is a parameter that has to be confirmed for the filing year. Where
    a figure here is not evidence, the report says so.
    """
    rate: float = BELGIAN_CAPITAL_GAINS_RATE
    allowance: float = BELGIAN_CAPITAL_GAINS_ALLOWANCE
    rate_observed: bool = True           # 10.00%, MeDirect, 18 June 2026
    allowance_observed: bool = False     # indexed; confirm for the filing year

    def charge(self, gain: float, realised: float = 0.0) -> float:
        """Tax on `gain`, given `realised` already realised this year."""
        before = max(float(realised) - self.allowance, 0.0)
        after = max(float(realised) + float(gain) - self.allowance, 0.0)
        return float(self.rate * (after - before))

    def marginal_rate(self, realised: float) -> float:
        """The rate the next euro of gain would pay. Zero or `rate`.

        Reported because it is the number a decision is made against: whether
        trimming a holding costs anything depends on where the year stands,
        not on the average rate paid so far.

        >>> tax = CapitalGainsTax(rate=0.10, allowance=10_000.0)
        >>> tax.marginal_rate(0.0), tax.marginal_rate(12_000.0)
        (0.0, 0.1)
        """
        return self.rate if float(realised) >= self.allowance else 0.0

    def headroom(self, realised: float) -> float:
        """Gain that can still be realised this year before the rate bites.

        >>> CapitalGainsTax(allowance=10_000.0).headroom(6_500.0)
        3500.0
        """
        return max(self.allowance - float(realised), 0.0)


@dataclasses.dataclass(frozen=True)
class Disposal:
    """One sale, and what it realised."""
    date: dt.date
    isin: str
    proceeds: float
    basis: float                         # INCLUDING acquisition costs
    tax: float = 0.0

    @property
    def gain(self) -> float:
        return self.proceeds - self.basis


class GainsLedger:
    """Realised gains per calendar year, and the tax each disposal incurred.

    Stateful on purpose. The tax on a sale is not a property of the sale: it
    depends on every sale before it in the same year, so a function that took
    one disposal and returned one number would be answering a question that
    does not have an answer. The order in which disposals arrive is the order
    they happened, and this refuses to accept one dated before the last it saw
    rather than silently producing a figure for a history that did not occur.
    """

    def __init__(self, tax: CapitalGainsTax | None = None) -> None:
        self.tax = tax or CapitalGainsTax()
        self._by_year: dict[int, float] = {}
        self._last: dt.date | None = None
        self.disposals: list[Disposal] = []

    def realised(self, year: int) -> float:
        return self._by_year.get(int(year), 0.0)

    def record(self, when: dt.date, isin: str, proceeds: float,
               basis: float) -> Disposal:
        """Charge one sale and remember it.

        >>> from datetime import date
        >>> ledger = GainsLedger(CapitalGainsTax(0.10, 10_000.0))
        >>> a = ledger.record(date(2026, 3, 1), "X", 12_000.0, 4_000.0)
        >>> round(a.gain, 2), round(a.tax, 2)
        (8000.0, 0.0)
        >>> b = ledger.record(date(2026, 9, 1), "X", 6_000.0, 2_000.0)
        >>> round(b.gain, 2), round(b.tax, 2)
        (4000.0, 200.0)

        The tranche resets with the calendar year, so the same sale in January
        costs nothing again:

        >>> c = ledger.record(date(2027, 1, 5), "X", 6_000.0, 2_000.0)
        >>> round(c.tax, 2)
        0.0
        """
        if self._last is not None and when < self._last:
            raise ValueError(
                f"disposals must arrive in date order: {when} follows "
                f"{self._last}. The tax on a sale depends on the sales before "
                f"it in the same year, so out-of-order input produces a figure "
                f"for a history that did not happen.")
        gain = float(proceeds) - float(basis)
        charged = self.tax.charge(gain, self.realised(when.year))
        self._by_year[when.year] = self.realised(when.year) + gain
        self._last = when
        out = Disposal(when, isin, float(proceeds), float(basis), charged)
        self.disposals.append(out)
        return out

    @property
    def total_gain(self) -> float:
        return float(sum(d.gain for d in self.disposals))

    @property
    def total_tax(self) -> float:
        return float(sum(d.tax for d in self.disposals))

    def lines(self) -> "list[str]":
        """Realised gains and the tax on them, as their own section.

        Separate from transaction costs throughout, because they behave
        differently: a transaction cost is proportional to what is traded and
        a capital gains tax is proportional to what was *gained*, so the same
        turnover costs nothing on a flat position and 10% of the move on one
        that has doubled.
        """
        if not self.disposals:
            return ["No disposals, so no realised gains and no tax on them.",
                    "Buys never pay this, which is a structural reason to "
                    "direct new money rather than to rebalance."]
        out = ["Realised gains and the tax on them", "-" * 64,
               f"  {'date':12}{'holding':14}{'proceeds':>12}{'basis':>12}"
               f"{'gain':>12}{'tax':>10}"]
        for d in self.disposals:
            out.append(f"  {d.date:%Y-%m-%d}  {d.isin[:12]:12}"
                       f"{d.proceeds:>12,.2f}{d.basis:>12,.2f}"
                       f"{d.gain:>12,.2f}{d.tax:>10,.2f}")
        out.append(f"  {'':12}{'':14}{'':12}{'':12}"
                   f"{self.total_gain:>12,.2f}{self.total_tax:>10,.2f}")
        out.append("")
        for year in sorted(self._by_year):
            realised = self._by_year[year]
            out.append(
                f"  {year}: {realised:,.2f} realised, "
                f"{self.tax.headroom(realised):,.2f} of the "
                f"{self.tax.allowance:,.0f} tranche left, next euro of gain "
                f"taxed at {self.tax.marginal_rate(realised):.0%}.")
        out.append(
            "  The basis includes the transaction costs paid to acquire the "
            "shares, which is what was paid for them. A broker statement that "
            "uses the pre-tax acquisition value overstates the gain and the "
            "tax with it.")
        return out
