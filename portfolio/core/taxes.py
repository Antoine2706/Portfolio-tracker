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
           "BELGIAN_TOB_CAP"]

# Read off a MeDirect contract note, 2 February 2026: 2.43 EUR of tax on a
# 2,024.87 EUR notional in DE000A2QP372, an accumulating UCITS ETF domiciled
# in Germany. Charged on both purchase and sale.
OBSERVED_TOB_RATE = 0.0012

# The bands, for reference when confirming the other holdings. Read each off
# its own contract note; an ISIN does not determine which one applies.
BELGIAN_TOB_BANDS = {
    "observed_medirect_etf": 0.0012,
    "accumulating_registered": 0.0132,
    "equity": 0.0035,
    "unknown": None,
}

# Per transaction, in the 0.12% band.
BELGIAN_TOB_CAP = 1_300.0

# France taxes acquisitions of shares in French-headquartered companies above
# 1bn EUR of market capitalisation. Buy side only, regardless of the buyer's
# residence or the venue used, which is why anything pricing a trade with it
# has to know which side the trade is.
FRENCH_FTT_RATE = 0.004
