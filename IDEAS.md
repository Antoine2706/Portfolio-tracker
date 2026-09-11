# Parked ideas

Written down rather than built, per the agreement to stop and use the tool.
Nothing here is scheduled. Each entry records what prompted it, so the reason
survives even if the idea turns out to be wrong.

---

## Correlation clusters instead of pairs — BUILT

**Prompted by:** the real portfolio, 300 aligned days to 2 September 2026. Five
defence pairs sit between 0.77 and 0.99 and only one crosses the flag. A book
holding four defence ETFs would raise one warning while carrying essentially
one bet across four lines.

**What was built:** `core.risk.correlation_clusters` takes connected components
over the pairs at or above the threshold and reports each group with its mean
and minimum correlation and its combined capital weight. The Risk page shows
clusters as chip groups beside the pairwise grid; a cluster of three or more
raises a SERIOUS alert. Threshold-based components were chosen over
hierarchical clustering because they need no linkage choice and their failure
mode on small samples (a missed edge splits a group) is visible rather than
silent.

---

## Historical FX conversion of a foreign listing — BUILT, first half

**Prompted by:** the currency-over-history ranking rule in `data/resolve.py`. A
USD-quoted line of a EUR-base fund embeds EURUSD movement, so its volatility
measures fund variance plus currency variance. That is why the EUR line wins
even with less history.

**What was built:** `data/fx.py` now loads a daily FX history per currency
into `DailyFxTable`, so cost basis converts at the transaction date and the
value history converts every day at that day's rate.

**Still parked:** using that history to convert a long foreign price series
into a valid EUR series for the covariance matrix, which would let WDEF.L's
500 days beat EUDF.DE's 377 legitimately. The resolver still prefers the EUR
line, and that remains the right default until the converted series has its
own gap handling and tests.

---

## The 0.85 threshold does not catch the pair it was lowered for

**Prompted by:** the observed Grains/Wheat correlation of 0.848, against a
threshold specified as 0.85.

**The gap:** 0.848 < 0.85, so that pair does not fire. The value was left at
0.85 as specified rather than quietly tuned to 0.84, because a threshold chosen
to fit a single observation is a threshold that means nothing. Documented in
`TestCorrelationThreshold.test_grains_against_wheat_at_0848_falls_just_below`.

**Decision needed:** either accept the miss, or set 0.84. One character in
`core/risk.py`. The Settings page shows the threshold read-only for now.

---

## FIFO cost basis for tax reporting

**Prompted by:** `core/positions.py` uses weighted average cost, which is the
right basis for risk and performance reporting and the wrong one for most
European capital-gains regimes, which require FIFO lots.

**The idea:** a lot queue per instrument, with realised gains attributed to
the oldest lots first, and a tax-year report.

**Why not now:** it is isolated in `_apply_sell`, so it is one function to
swap, but a tax report needs jurisdiction rules (holding periods, allowances,
loss carry-forward) that this tool has no business guessing at.

---

## Multiple portfolios / accounts

**Prompted by:** a broker account and a pension wrapper hold overlapping
instruments with different tax treatment.

**The idea:** an `account` column on the ledger and an account filter on
every page, with the risk model always computed on the union.

**Why not now:** it touches every layer for a single-user tool that does not
need it yet, and the ledger format is the one thing that must not churn.
