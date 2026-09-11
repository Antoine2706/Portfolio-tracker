# Pre-registration: volatility targeting, de-risking half only

**Status: written, built, not run against real prices.** The policy is
`agents/voltarget.py`, the run and its three criteria are
`research.run_volatility_target` and `portfolio backtest voltarget`, and
every parameter below is the default there. The target is not an option at
all; the window and the interval are the options ERC shares, and
`--register` is refused at any value but the pre-registered ones, because a
run at another value is another trial. It has been exercised on synthetic
prices only, where `--register` is refused outright. The real run is the
user's: `portfolio backtest voltarget --register` against the provider
consumes the one trial this document declares, and nothing here is in
`research/registry.jsonl` until then. This document was written before any of
that code existed, so that the expectation is on the record before the
result is, which is the only thing that makes the deflated Sharpe ratio's
trial count mean anything.

One departure from the text below, forced by the account rather than chosen:
"scale the whole book" cannot touch the holding at the second broker, so the
scalar applies to the tradeable part and the target to the whole book. That
turns `k = min(1, sigma_target / sigma_hat)` into the positive root of
`A k^2 + 2 B k + (C - V) = 0`, with `A`, `B`, `C` the tradeable, cross and
frozen variance terms and `V` the squared target, capped at one; with nothing
frozen it is the same formula. The frozen part alone above the target leaves
`k = 0`, and the policy says so.

## The policy

Scale the whole book by a factor `k_t` chosen so that its forecast volatility
equals a target:

    k_t = min(1, sigma_target / sigma_hat_t)

`sigma_hat_t` is the same estimator, window and covariance matrix the equal
risk contribution policy uses — the annualised volatility of the current
weights over the trailing 252 trading days, decided on data strictly before
the bar it acts on. The residual `1 - k_t` sits in cash at zero interest.

**The `min(1, .)` is the whole point of this document.** Volatility targeting
in the literature is symmetric: lever up when realised volatility is below
target, down when above. This account cannot lever. There is no margin
facility, and there will not be one. So only the de-risking half is available,
and what is being tested is not volatility targeting — it is *half* of it, the
half that sells into calm and buys back into calm, with the compensating half
removed.

## What I expect, before running it

**I expect it to fail after costs, and I expect it to fail before costs too.**

Three reasons, in the order I believe them:

1. **The de-risking half is the losing half in a rising market, and step 3
   measured exactly that.** The equal risk contribution policy underperformed
   buy-and-hold by 7.46 points a year gross in the same window, of which the
   risk-adjusted residual was 1.55 points and the rest was the mandate: it held
   less risk than the benchmark while the benchmark rose. Volatility targeting
   without leverage is a stronger version of the same mandate — it goes to cash
   rather than merely reweighting — so it should lose *more* of the same thing
   in the same window, not less.

2. **Turnover is higher than equal risk contribution's, on the same estimator.**
   ERC rebalanced 13 times over the window at 2.77% mean one-way turnover,
   costing 0.20% a year. A volatility scalar moves every time the trailing
   volatility estimate moves, which is every bar, so any implementation needs a
   band or a schedule to be affordable at all. Whatever band makes it
   affordable also makes it slower to react, which is the thing it is for.

3. **The asymmetry is a ratchet in a rising window.** `min(1, ·)` can only ever
   hold less than the book, never more. Over a period in which the book rose
   30%, any policy that spends part of the period partly in cash at zero
   underperforms by construction, and the only question is by how much.

**Quantified, so it can be wrong:** I expect the annualised return shortfall
against buy-and-hold to be **larger than equal risk contribution's 7.46
points**, and the risk-adjusted comparison — the Jobson–Korkie difference of
Sharpe ratios with Memmel's correction, at a measured ρ — to be
**indistinguishable from zero at |t| < 2**, as ERC's was. I expect realised
volatility to be **materially below** buy-and-hold's 19.79%, because that is
the one thing the policy actually controls and the only claim it can honestly
make.

## What would count as it working

All three, not any one:

1. **Realised volatility within 15% of the target**, relative. Missing this
   means the estimator is not forecasting what it claims and nothing else in
   the run is interpretable.
2. **The matched-risk gap indistinguishable from zero**, |t| < 2 on
   Jobson–Korkie/Memmel. Not "better" — indistinguishable. A policy whose
   entire purpose is to hold less risk should not be asked to earn more per
   unit of it, and if it appeared to, on this sample size, I would suspect the
   test before I believed the result.
3. **Cost under 0.50% a year**, against ERC's 0.20%. Above that the policy is
   paying more than a quarter of a typical equity risk premium's standard error
   for a volatility reduction the account could get by holding less.

If (1) fails the run is uninterpretable, not negative. If (1) passes and
(2) or (3) fails, the policy is measured and rejected.

## What would NOT count, and why it is written here

- **A higher Sharpe ratio.** In this window and at this sample size the Sharpe
  ratio cannot separate 1.60 from 1.69; step 3 established that the standard
  error on the difference is sixteen times the difference itself. A higher
  Sharpe here is noise with a favourable sign, and if the run produces one I
  will treat it as noise, because that is what I have written down in advance.
- **Lower drawdown.** It is the same claim as lower volatility, differently
  sliced, and counting it separately would be counting one result twice.
- **Anything measured after changing the target, the window, the band or the
  rebalance schedule.** Each of those is a new trial and consumes another unit
  of the deflation budget. The parameters below are fixed before the run.

## Parameters, fixed now

| Parameter | Value | Why this one |
|---|---|---|
| `sigma_target` | 15% annualised | Just under buy-and-hold's realised 19.79% in the window, so the policy binds often enough to be measured. Not tuned. |
| estimator | trailing 252-day sample covariance | Identical to ERC's. A second estimator would be a second thing to defend. |
| leverage cap | 1.0 | The account fact this whole document is about. |
| rebalance | every 21 trading days | ERC's, so turnover is comparable rather than a free parameter. |
| band | none at first | Measured without one, so the cost of *not* having one is on the record before a band is introduced to fix it. A band would be trial 2. |
| execution | next close | The harness's, unchanged. |

## Trial accounting

This is **one** trial against the deflation budget, whatever it returns. The
controls are excluded from the count, as they always have been. If it fails
and I want to try a band, that is trial 2 and it gets its own document; the
deflated Sharpe ratio for any later claim takes the running count as an
argument, and a count that omits the failures is not a count.

## The honest summary

I am running this because I said I would, and because "I expect it to fail" is
worth nothing unless it is recorded before the number arrives. If it fails as
expected, the value of the run is the pre-registration: it is one fewer thing
to wonder about, at a known cost. If it succeeds on all three criteria I will
have been wrong in a way I can point at.
