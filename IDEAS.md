# Parked ideas

Written down rather than built, per the agreement to stop and use the tool.
Nothing here is scheduled. Each entry records what prompted it, so the reason
survives even if the idea turns out to be wrong.

---

## Correlation clusters instead of pairs (v2)

**Prompted by:** the real portfolio, 300 aligned days to 2 September 2026. Five
defence pairs sit between 0.77 and 0.99 and only one crosses the flag. A book
holding four defence ETFs would raise one warning while carrying essentially
one bet across four lines.

**The idea:** name a mutually correlated *group* as a cluster rather than
reporting its edges one at a time. Pairwise comparison is structurally unable
to see this: it reports edges, never the group.

**Why not now:** it needs a clustering choice (threshold-based connected
components, or hierarchical on a correlation distance) and each has failure
modes on small samples. Effective number of holdings already captures the same
concentration, and is what the interface now leads with.

---

## Historical FX conversion of a foreign listing (v2)

**Prompted by:** the currency-over-history ranking rule in `data/resolve.py`. A
USD-quoted line of a EUR-base fund embeds EURUSD movement, so its volatility
measures fund variance plus currency variance. That is why the EUR line wins
even with less history.

**The idea:** a long foreign listing could be converted to EUR historically
using daily FX, yielding a valid *and* longer EUR series -- which would let
WDEF.L's 500 days beat EUDF.DE's 377 legitimately.

**Why not now:** needs a full daily FX history with its own gap handling and
error surface. The current spot-rate table is deliberately small.

---

## The 0.85 threshold does not catch the pair it was lowered for

**Prompted by:** the observed Grains/Wheat correlation of 0.848, against a
threshold specified as 0.85.

**The gap:** 0.848 < 0.85, so that pair does not fire. The value was left at
0.85 as specified rather than quietly tuned to 0.84, because a threshold chosen
to fit a single observation is a threshold that means nothing. Documented in
`TestCorrelationThreshold.test_grains_against_wheat_at_0848_falls_just_below`.

**Decision needed:** either accept the miss, or set 0.84. One character in
`core/risk.py`.
