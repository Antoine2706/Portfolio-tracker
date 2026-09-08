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

__all__ = ["OBSERVED_TOB_RATE", "BELGIAN_TOB_BANDS", "FRENCH_FTT_RATE",
           "BELGIAN_TOB_CAP", "BELGIAN_CAPITAL_GAINS_RATE",
           "BELGIAN_CAPITAL_GAINS_ALLOWANCE"]

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
