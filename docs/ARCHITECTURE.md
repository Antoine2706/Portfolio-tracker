# Architecture

Personal investment portfolio tracker for European-listed ETFs and ETCs, keyed
on ISIN, with risk analytics that can be read against a textbook. This document
is the contract between the layers. Every module named here is built to the
signatures below; if a signature has to change, change it here first.

## Layers

```
portfolio/
  core/      pure: numpy + pandas only. No network, no UI, no provider code.
  data/      storage and providers: CSV store, SQLite price cache, yfinance,
             OpenFIGI, FX history, CSV import/export. May use the network.
  api/       FastAPI application. Reads core + data, serves JSON and the SPA.
             Computes NOTHING: every number is produced by core.
  web/       Static single-page application (Preact + htm + ECharts, vendored,
             zero build step). Renders JSON from api/. Computes nothing beyond
             formatting and client-side sorting/filtering.
  agents/    Decision policies and the trading cost model. Pure: numpy +
             pandas + core. Defines both its input type (MarketView) and its
             output type (Proposal). May NOT import eval/.
  eval/      The walk-forward harness, track-record statistics, the
             pre-registration log, the controls that calibrate the harness and
             the look-ahead detector. Pure; imports core and agents.
  research.py  The composition root for backtests: the only module that
             touches the data layer, the policies and the harness at once. It
             exists so eval/ and agents/ can stay offline -- they are handed a
             panel and a cost model and never learn where either came from.
             Nothing may import it back; the layering test enforces that.
  broker/    Paper trading only, and does not exist yet -- see the build
             order. A test fails if a live endpoint string appears anywhere.
  tests/     No network (sockets are blocked), no UI framework required for
             core tests. API tests use FastAPI's TestClient + FixtureProvider.
```

Dependency arrows point inward: `web -> api -> data -> core` and
`eval -> agents -> core`. `core` never imports from `data` or `api`; `api`
never imports numpy or scipy; `agents` never imports `eval`; `data` never
imports `agents`, `eval`, `api` or `research` (the layering tests enforce all
of it with `ast`).

That last one has a concrete temptation behind it. The schema migration in
`data/migrations.py` needs the Belgian transaction tax bands, which are also
what `agents/execution.py` prices a trade with. Importing them across would
make storage depend on the decision layer for three constants, and the next
thing imported along that edge would not be three constants. They live in
`core/taxes.py` instead -- reference data about jurisdictions, computed by
nobody -- and both layers reach inward for them.

That last arrow is the one that is easy to get backwards and expensive to get
wrong. A policy must not be able to tell that it is being backtested, because
a policy that can tell could behave differently under evaluation than in
production -- and then the backtest measures something that will never
happen. So the evaluation layer knows about policies; policies know nothing
about evaluation.

Run it: `pip install -e ".[app,data]"` then `portfolio serve` (opens the
browser at http://127.0.0.1:8765). `portfolio serve --provider fixture` runs a
fully offline demo with deterministic synthetic prices. `PORTFOLIO_DATA_MODE`
selects `seed` (demo) or `user` (your data), default `seed`.

## The evaluation contract

`eval/` exists to answer a different question from the rest of the codebase.
The tracker answers "what is my portfolio worth and how risky is it". The
harness answers "would I have been able to tell whether a strategy worked",
which has different failure modes: it produces a confident number whether or
not it is measuring anything.

Three rules follow, and each is enforced rather than documented.

**The future is absent, not filtered.** A policy is handed a `MarketView`
built by slicing the price panel at the decision date. The rest of the panel
is not in the object, so there is no check to forget. `eval/leakage.py`
rewrites every price after a date and re-runs; everything decided before it
must come back bit-identical. It also raises rather than passing when it is
wired so that it could not fail.

**Costs are an argument, not an option.** `walk_forward` takes a cost model.
`agents/execution.py` makes every parameter a named field and resolves it per
broker and per instrument, because at a real account they differ by more than
an order of magnitude between holdings. It takes a trade side, since some
taxes are charged on purchases only. Where a rate has not been read off a
document it either refuses to price the trade or reports itself as an
estimate; `CostModel.assumptions()` lists everything unobserved and the
backtest prints it beneath every result. A strategy that is profitable gross
and unprofitable net is the most common false positive in this field, and a
harness that cannot produce that finding is not measuring.

`CostModel.provenance(weights)` goes further and quantifies it: how many tax
rates were read off a document, what share of the book those holdings are,
and what fraction of the modelled round-trip cost rests on evidence rather
than judgement. A count alone is not the answer, because one observed rate on
a 2% position and one on a 40% position are not the same claim.

**A holding whose cost cannot be established cannot be traded.** Refusing to
price it stays absolute -- nothing invents a rate. But refusing to *run* was
too strong: `research.load_book` now freezes such a holding and says why,
which is what "no policy may trade this" already means elsewhere in the
system. It also keeps the two reasons for a freeze apart, since a second
broker is a permanent fact about the account and a missing tax band is a gap
one contract note closes.

**Some weights are exogenous.** A holding at a second broker cannot be traded
against the rest of the book. It is marked non-tradeable on the instrument
record, the referee rejects any proposal that moves it, and the harness takes
its weight from the drifted book at execution rather than from the proposal --
because a frozen holding drifts between the decision and the order, so even
proposing the weight it had a moment ago implies a trade.

**Nothing is reported that the sample cannot support.** `eval/metrics.py`
carries the standard error, the probabilistic and deflated Sharpe ratios and
the minimum track record length, all with the third and fourth moments rather
than a normality assumption. Every Sharpe is printed with its standard error
beside it, because a ratio on its own invites a ranking. Where the window is
too short, the verdict says so and gives the window that would suffice. The
deflation takes the number of attempts from `research/registry.jsonl`, which
is append-only for the same reason the ledger is.

A ratio and an observation count must describe the same sampling interval.
`effective_observations` changes the count when decisions overlap, so
`rescale_sharpe` restates the ratio at that frequency before either is used;
without it the standard error was computed from a daily Sharpe against a
monthly count and every t-statistic came out roughly sqrt(overlap) too small.
The negative control now measures the t-spread at both overlap settings,
because it previously ran only at 1 while the real backtest ran at 21.

**A day an instrument did not trade is not a 0.00% return.** Positions are
*valued* against the last price that printed; estimates admit a return only
when both of its endpoint prices were observed on consecutive panel dates.
That keeps the money -- the missed move lands on the next print, so the
compounded return is exactly what a buy-and-hold investor experienced -- and
throws away the observation, since neither the gap day nor the day after it
is a one-day move. `BacktestResult.measured` carries the mask and the report
prints the excluded count beside the observation count.

**The alarm fires on the policy, not on the window.** An annualised Sharpe
above 1.5 is the calibration rule, but a level is a property of the window,
and the benchmark shares the window: on a good year both legs clear it and
the alarm means nothing about either. `Comparison.verdict()` reports the
benchmark's own figure alongside and says plainly when a shared high level
points at the window or at the data feeding both legs rather than at a leak.

**And it fires on the risk-adjusted gap, not on the raw active return.** The
first attempt at the above put the threshold and the |t| = 2 verdict on the
active return -- policy minus benchmark, day by day. For a de-risking policy
that is a tautology: equal risk contribution holds less of the volatile
assets, so in a rising window it underperforms by construction, and the
identical policy in a falling window comes out significantly *better*. The
sign belongs to the window. `test_risk_adjusted.py` demonstrates it with a
policy that holds a fixed fraction of the benchmark's book and nothing else:
t = -2.6 in one window, +2.7 in the other, zero skill in both.

The quantity that generalises is the comparison at matched risk. With the
risk-free rate at zero an annualised arithmetic return is Sharpe times
volatility, so the raw gap splits exactly:

```
R_p - R_b = (SR_p - SR_b) x sigma_b   +   SR_p x (sigma_p - sigma_b)
            \_____ selection _____/       \______ mandate ______/
```

Both terms are printed. *Selection* is what the strategy cost once levered to
the benchmark's risk -- the part a different window could have reversed --
and it is what the verdict tests. *Mandate* is what carrying different risk
contributed by construction, and it is what the policy was asked to produce.
The report states that matching risk means levering by sigma_b / sigma_p,
which a cash account cannot do: the raw gap is what the account lost and the
selection term is what the strategy cost, and neither replaces the other.

The difference of the two Sharpe ratios is tested with Jobson-Korkie plus
Memmel's correction (`sharpe_difference_standard_error`), at a correlation
that is **measured and printed**, never assumed -- every figure depends on it.
Two legs holding the same book correlate at about 0.99, the `2(1 - rho)` term
collapses, and the paired error comes out a fraction of either marginal one:
that is what makes the test possible at all. Normality is assumed there and
declared, since the robust version needs a HAC estimator; the formula is
checked against 20,000 Monte Carlo draws per correlation in the suite.

## The buy-only allocator

`agents/allocate.py`. Given new cash C and the book, choose `b ≥ 0` with
`sum(b) = C` minimising the dispersion of the risk shares of `v + b`. Risk
shares are homogeneous of degree zero, so the normalisation by `V + C` never
has to appear.

**It never proposes a sale.** `b ≥ 0` is a constraint of the problem, not a
preference, and both the search and the order assert it. Both assertions
caught real bugs on their first run: an affordability check hoisted out of the
inner loop went stale the moment a move succeeded, drove one holding to
−2,500 EUR of a 5,000 EUR purchase, and reached the order as a plausible 148
shares of something the money could not buy.

**Dispersion is the coefficient of variation of the risk shares** — standard
deviation over mean, zero when the contributions are equal, at most `√(m−1)`
for `m` holdings — and it is *both* the reported metric and, squared, the
objective. That identity is the point.

It was a range, and that was the mistake underneath a year of symptoms. A
range is a rank statistic: on six holdings it is decided by two and discards
the other four, it is non-differentiable wherever the argmax or argmin changes
hands, and a policy minimising it chases the worst laggard while ignoring the
shape of everything else. Worse, nothing minimised it — the descent minimised
a sum of squared deviations while the report printed a range, so the tool
graded answers by a rule it had not used to produce them. A least-squares
descent reaching 1.205 against a coarse grid's 1.136 was measuring that
mismatch, and it was diagnosed as a solver failure and fixed by adding a
search. The range is still printed, beside the dispersion and never instead of
it, because it says how far apart the extremes are, which a CV does not.

**The objective is smooth but not convex.** Risk shares are ratios of
quadratics, so a convex feasible set does not make the problem convex — that
inference was in the specification and does not follow. Three parts, each
because something measurable went wrong without it:

1. *Projected gradient descent on the objective itself*, with the gradient
   derived in full (`objective_and_gradient`) and checked against central
   differences, including over a subset where the mean of the chosen shares is
   not constant — the half of the derivation a full-set check cannot see.
2. *`aim_at_equal_risk`*, which hands the search the equal-risk portfolio in
   b-space. Past the amount at which it becomes reachable it *is* the answer.
   Without it the floor on a hedged eight-holding book reaches equal risk at
   twice the book and loses it again at fifty times, a rise of 0.38.
3. *A multi-resolution pattern search over pairwise exchanges.* Not a tidy-up:
   the descent converges to points with a projected-gradient residual of 1e-16
   — genuine constrained stationary points, unmoved by 25× the iterations —
   from which an exchange of a third of the money reaches 0.56 where the
   stationary point sat at 1.00.

Two independent things check the result, which is labelled "best found": a
dense brute-force grid on small books, and **`spinu_sweep`**, which solves the
convex Maillard–Roncalli–Teiletche form sharpened by Spinu — minimise
`0.5·w'Σw − λ·Σ log wᵢ`, whose stationarity condition *is* equal risk
contribution — to global optimality over this same polytope, for a family of
`λ`. One `λ` is not the answer: the budget and the lower bounds destroy the
scale invariance that makes `λ` irrelevant unconstrained. What the family
gives is certified points, and the requirement that the search never lose to
one. It never has, and in one place by only 6e-4, which is what makes it a
check rather than a formality. It is deliberately *not* a candidate inside
`reachable_floor`; feeding it in would make the check vacuous.

**The floor is monotone in the cash — only if every holding can receive it.**
The proof is a rescaling: for `λ = (V+C₂)/(V+C₁)`, the vector `λx` has
identical risk shares (they are homogeneous of degree zero) and is affordable
at `C₂`, since `b₂ = (λ−1)v + λb₁ ≥ 0`. This module asserted that
unconditionally and bisected on it. It is false the moment a holding cannot
receive: `b₂ᵢ` must then be zero and `(λ−1)vᵢ` is strictly positive. The
pinned weight is an equality that *moves* with the money, not a bound that
relaxes, so the configurations do not nest and the floor can rise. On the demo
book it does, climbing towards `10/7` — seven equal risk shares and three
diluted to zero. `cash_for_dispersion` therefore scans a geometric ladder and
bisects inside the bracket: every step keeps `floor(high) ≤ target`, so the
amount named always reaches it; minimality is what pinning costs.

Where monotonicity *is* real it is checked on the solver, not assumed of it —
and the check has to be able to fail. Two findings shaped it. Books of
independent assets are monotone whether or not the fix is in, so a check on
them checks nothing; the failure needs **negative correlation**, where a
holding's `(Σx)ᵢ` goes negative, its risk share with it, and the range
acquires local minima no two-holding exchange can leave. And the fix is
`aim_at_equal_risk`: start from `b = w_erc·(V+C) − v` projected onto the
simplex, which is the optimum itself once the money makes it reachable.
Without it the floor on a hedged eight-holding book *finds* equal risk at four
times the book and loses it at sixteen, reporting 1.4254 for strictly more
money on a strictly larger feasible set.

**Whole shares, then the report.** Solve continuously; drop any holding whose
allocation falls below its own broker's minimum economic trade, or above the
largest order that broker has a published fee for, and solve again without it;
only then round down and spend the remainder greedily. Every reported figure
is recomputed on the executable order, and the continuous optimum appears once
as the floor so the rounding penalty is visible rather than absorbed.

`Instrument.buyable` is separate from `Instrument.tradeable`. The gold ETC
forces them apart: it cannot be rebalanced against the rest of the book, and a
fresh purchase at its broker is an ordinary order.

`eval/replay.py` evaluates it by re-running the real ledger with only the
destination changed — same dates, same amounts, so neither arm pays extra
turnover. Dispersion is computed and carries no sampling error; volatility is
estimated and carries its own, and the correlation between the two arms is
measured and printed rather than asserted, because ρ decides how much of each
marginal error cancels. The whole-share remainder is carried to the next
purchase, because over a ledger it compounded to 6.5% of the money and would
otherwise leave one arm quietly part in cash. A contribution is not a return
and neither is a withdrawal: both are backed out before the value series is
divided. The replay is checked for look-ahead the same way the harness is:
rewrite every price after a date and require every earlier decision back
unchanged — which is what rejected a single whole-panel covariance for the
measurement and forced it to be point-in-time.

**Disposals** are the one thing with no neutral answer, so both are offered
and the report names which ran. `on_sale="prorata"` (default) applies each
sale to *both* arms as the same proportional withdrawal: risk shares are
homogeneous of degree zero, so it changes none of them in either arm, and the
whole ledger is covered. The cost is that the actual arm is then "the
purchases that were made, with disposals taken pro-rata" rather than the book
that was held — deliberately, because selling a particular holding is a
decision the allocator never makes and crediting one arm for it would measure
that instead of the destination. `on_sale="stop"` ends at the first sale
instead: every figure true of a real book, over a shorter window. Ignoring
them, which this did, is the only wrong answer — the demo's "actual" arm went
on holding a position the ledger had sold.

## The pattern this project keeps finding

Every defect of consequence here has had the same shape: **a confident output
measuring something other than what it names.** Not a crash, not a wrong
formula — a number or a sentence that is precise, plausible, and about a
different quantity than its label.

They fall into two groups, and the split matters more than the list, because
the remedy differs. A wrong **result** is caught by a control that bites. A
dead **check** is caught only by deliberately breaking the code and confirming
the check screams — a different practice, and the one worth institutionalising.

The dead-check group comes first because a wrong result is one number, while a
dead check is a licence for every future number it was supposed to guard.
Group A's fourth entry is the argument in one line: the moment it was made
honest it found two live defects that had been shipping.

### Group A — the artefact *is* the check, and the check could not fail

Found only by sabotage. Nothing in a passing suite distinguishes these from a
codebase that is simply correct.

| # | Claimed to check | Could not have failed because |
|---|---|---|
| A1 | that the i.i.d. standard error fails a control | the sentence was written from the structural argument, before the control was run. It passes at every lag; the autocorrelation it corrects for is −0.003 ± 0.002 |
| A2 | that the tick grid is read off recent bars, not the whole series | nothing exercised it. Reading the whole series instead broke no test until the guard was pulled into `research.tick_for` and given its own |
| A3 | that the allocate page loads | it was run in a container with an *editable* install, where `WEB_DIR` is the clone and the file is always present. The environment could not exhibit the reported bug |
| A4 | that there was one clone, not two — `python -c "import portfolio, os; print(os.path.dirname(portfolio.__file__))"` | `sys.path[0]` is `""` for `python -c`, so run from a checkout it reports the checkout whatever is installed. Its output was read as disproof of a hypothesis it could not test |
| A5 | the same question, asked via `importlib.metadata` instead | `importlib.metadata` also walks `sys.path`, so from a checkout it finds the `*.egg-info` build artefact and answers with relative paths and no `direct_url.json` — an editable install reads as non-editable. The replacement inherited the defect it replaced |

A3 is why the allocate page took two hours: a false pass reported as a fact
about someone else's machine. A4 and A5 are the same question asked twice and
answered wrongly both times, the second time while fixing the first — which is
the strongest available argument for ranking this group above the other.

A4 and A5 look like a note about Python packaging and are not. `sys.path` is
one instance of a wider family: **a diagnostic whose answer is a function of
the context it is asked from.** The current directory, `PATH`, the active
virtualenv, git's upward search for a repo root, any environment variable —
each of them silently parameterises an answer that reads as a fact. Asking
"which copy is running" from inside the copy is the same error as asking
"which branch is this" from inside a submodule, or "which python is this" with
a virtualenv active.

So the operative test is not "run it from two directories" — that is the
`sys.path`-shaped instance and it generalises badly. It is: **find a context
where you know the answer should differ, ask from there, and check that it
does.** For A4 that was a directory containing a decoy `portfolio` package;
for A5, a directory containing a decoy `.egg-info`. Both are in
`tests/test_diagnostics.py`, and both say what to delete if the interpreter
ever stops behaving that way, so the exclusion is not carried forward on a
belief nobody has re-checked.

`portfolio doctor` now answers it properly: what `import portfolio` resolves to
*here*, whether the current directory is shadowing an install, what the
distribution metadata says **with the current directory excluded**, whether the
install is editable, and which files in the checkout are missing from the
directory actually being served. `portfolio serve` prints the last of those
before rendering anything.

### Group B — a result measuring something other than its name

| # | Named as | Actually measured |
|---|---|---|
| 1 | a leak alarm on the policy | the level of the window, which the benchmark shared |
| 2 | a day's return | a non-trading day, entered as 0.00% |
| 3 | the standard error of a Sharpe ratio | a daily figure against a monthly count — wrong by √21 |
| 4 | evidence the policy underperforms | the sign of the window, for a de-risking mandate; the same policy is "significant" the other way in a falling one |
| 5 | why a trade could not be priced | *"an ETC is a debt security"* — of a property fund, in prose |
| 6 | the risk-share dispersion of a book | a rank statistic on two holdings of six, while the solver minimised a different function entirely |
| 7 | the volatility of two portfolios | the cash flowing into them, and a refusal to rank them justified by *"nearly the same portfolio"* for series correlating at 0.363 |
| 8 | a gap worth closing | 1e-17 of solver residual, because the threshold was compared against exact zero and answered differently on two machines |
| 9 | a two-standard-error test that a spread is real | one standard error, because `SE(s) = SE(s²)/2s` makes `s ≥ 2·SE(s)` identical to `\|s²\| ≥ SE(s²)`; it claimed a spread in 32% of samples that had none |
| 10 | a Newey-West correction absorbing the per-bar series' overlap | nothing — the autocorrelation is −0.003 ± 0.002 — while adding up to 15% of noise to a single sample's error bar |
| 11 | an instrument's bid-ask spread, on the fixture | a discontinuity the fixture's own bar generator left at the close-to-open boundary, which is exactly where the estimator reads its bounce |
| 12 | an amount of new money, "10.000" | ten euros. The English convention hard-coded in a client-side parser, in a book kept in Belgium, while `data/importers.py` had the rule right all along |
| 13 | *"the series has been adjusted for distributions"*, refusing all 7 holdings | float32. Yahoo sends prices as float32 and `100.06` arrives as `100.05999755859375`, so an exact-divisibility test against a tick fails on nearly every price. Four of the seven are accumulating ETFs that have never made a distribution, and the one that pays a dividend had the *highest* fit rate — the explanation was not merely unproven, its ordering was backwards |
| 14 | *"suspect the symbol mapping or the panel alignment"*, on a ranking check that failed at ρ = +0.89 | a guess. The check compares two numbers per instrument — an estimated spread and a volume-based liquidity proxy — and cannot tell which is wrong; the sentence picked one. The refusal was right. The verdict now prints the pairs it ranked, names the extremes, and says what it cannot tell apart |
| 15 | a verdict of "consistent" from the ladder control on a 42 bps reading of a one-bp instrument | the interval on s² at t = 1.9 reaches below zero, so its floor is "inside any band". What was wrong was the *ceiling* — 60 bps off 5000 bars, an error bar sixty times what the sample allows — and the first version of the verdict did not look at it. Found by the test written for the real-book case before the control had been run on anything |

Three of Group B were false sentences rather than false numbers (5, 7, and the
"the structure cannot be fixed by contributions" that overclaimed what a search
had established). Prose is not exempt from the standard and gets no review by
default, which is why it is where they survive.

#12 and #13 are the entries a user found rather than a control, and both are
in Group B rather than Group A because neither artefact was claimed to be a
check. #12 shipped because nothing compared the client's arithmetic against
the CSV importer's.

#13 is #5 again — *"an ETC is a debt security"* said of a property fund — and
worth noting as a repeat: a confident, specific, untested cause attached to a
**correct** refusal. The refusal was right; the reason sent the reader to look
at dividends for an hour. The fix is not a better guess but a measurement: the
residual distance from the grid separates the causes by three orders of
magnitude (float32 leaves 2e-4 of a tick, a rescaling leaves 0.25), so
`TickSize.why_not_a_grid` reports that number and names candidates rather than
picking one. It also distinguishes a whole rescaled series from a partly
rescaled one, which is what a split looks like.

The near-miss is worth recording too. The first fix rounded prices to
float32's seven significant figures, which produced values like `98.37599`
that sit on the 0.0001 grid by construction: a dividend-adjusted series then
"fitted" at 93.5% and the inference manufactured the grid it claimed to find.
The real constraint is epistemic — **a grid finer than the data's own
precision cannot be established from it** — so those candidates are now
excluded rather than fitted.

#14 is #13 and #5 for the third time, and it is the entry that finally made
the shape a rule (rule 8 below): a correct refusal carrying a cause it never
tested. The three sentences were all plausible, all specific, and all sent the
reader somewhere the data had not pointed. What replaced the guess was not a
better guess but three measurements the check could not make on its own: the
bars each instrument's estimate rests on, classified and counted
(`core.spread.classify_bars`); the same estimator with the suspect bars set
aside, printed beside the first; and the ladder — the whole data path run on
instruments whose spread is not in doubt (`portfolio controls
--spread-ladder`).

The classification was preceded by a measurement of what each kind of bar
does, and the measurement corrected the hypothesis it was testing. The
working theory was that EDGE, reading (h+l)/2, is *maximally sensitive to
high/low contamination*. It is not: widening the high and low by 30 bps on 15%
of bars, symmetrically or on one side, moves the estimate by 0%. What it
cannot survive is a **carried price** — a close copied from the previous day,
or a whole bar copied — which turns a day's real price move into a squared
spread. Five per cent of such bars inflates a 10 bps spread to 18; fifteen per
cent, to 27. Setting them aside recovers 10.0. Flat bars do nothing, which is
what the estimator's own `tau` was designed for. The full table is above
`classify_bars`, and the tests in `test_spread.py` pin all three findings,
including the negative one.

#15 is the same family as Group A, found in the same way — by writing the
test for the case the control existed for before believing the control — and
is in Group B only because it *did* fail once the ceiling was looked at. It is
worth its own line because the fix is a measurement rather than a threshold:
the ceiling a clean sample of the same length and volatility would report is
simulated per rung, and a ceiling more than three times that is called wide.
The survey prints the same ratio for every instrument that did not resolve,
because for those the ceiling is the claim.

#9 and #10 are the same lesson from opposite sides. Both came from arguments
that were sound in form: consecutive terms share a bar, *therefore* correlate;
two standard errors *is* two standard errors. Both were wrong, and neither was
findable by reading — only by generating data where the answer was known and
looking at how often the rule fired.

The working rules that fall out of it, in the order they pay off:

1. **Optimise and report the same function.** #6 existed only because they
   differed; the solver was blamed for a year of symptoms that were the
   objective's.
2. **Compare a threshold against a scale, never against zero or an absolute.**
   #8 and the scale-dependent "meaningful purchase" threshold are the same
   error at different magnitudes.
3. **Measure what you are about to assert.** #7's ρ, not "nearly the same";
   #10's autocorrelation, not "they share a bar so they must correlate";
   A1's control, not the argument for it.
4. **Prove the check bites, by breaking the code on purpose.** This is the
   whole of Group A and it is not the same activity as writing a test. A
   fixture that passes whether or not the fix is in is not a test; a
   verification run in an environment that cannot exhibit the bug is not a
   verification. The question to ask of every check is not "does it pass" but
   "what would make it fail, and have I seen that happen". Every entry in
   Group A was found the first time somebody asked the second half.
5. **A diagnostic's answer may be a function of where it was asked.** The
   current directory, `sys.path`, `PATH`, the active virtualenv, git's search
   for a repo root, any environment variable. Rule 4 applied to diagnostics
   rather than to tests, and the same discipline: find a context where the
   answer *should* differ, ask from there, and confirm it does. A4 and A5 are
   both this rule, and A5 is what happens when the fix for one instance is
   written without applying the rule to the fix.
6. **Apply a significance test on the scale where the sampling distribution is
   symmetric.** #9 survived because `s` and `s²` carry the same information
   and only one of them can be centred on zero. A transformation that looks
   like relabelling can halve a threshold.
7. **A generator that produces data for a validated estimator is itself
   unvalidated.** #11 was found only because the estimator had already been
   calibrated against something else, so a disagreement pointed at the
   fixture. Had both been written together, they would have agreed on the
   wrong answer.
8. **A verdict may name only what it measured.** #5, #13 and #14 are one
   defect: a correct refusal with a confident cause attached that nothing had
   tested. The refusal is the result; the cause is a second claim and is held
   to the same standard. Where the check cannot tell causes apart, it says
   so and prints what it compared, and the discrimination is done by a check
   that can.
9. **Conservative is not a substitute for right.** The bounded tier was
   defended as "errs toward overstating cost, which is the safe direction",
   and that defence would have accepted seven ceilings wrong by a factor of
   ten. A number that is wrong in a direction one likes is still wrong, and
   the direction is not even stable: a ceiling built on carried bars is
   pushed *up*, and the same feed defect on a different estimator could push
   it down. The direction of an error is not evidence about its size.

## An open result: the first real-book spread survey

Recorded here because a failed control with its numbers preserved is a
result, and one whose numbers scrolled off a terminal is an anecdote. Every
subsequent run is appended to `spread-runs.jsonl` in the data root by
`portfolio spreads`, so this is the last one that has to be written by hand.

The survey was run on a real book of seven holdings against Yahoo's daily
bars, over each instrument's whole available history. **Nothing resolved.**
All seven came back `bounded`, with ceilings from **15.2 to 59.7 bps**. Two of
them are known to be wrong in opposite directions:

* Schneider Electric (FR0000121972), a CAC 40 mega-cap whose half-spread on
  Euronext Paris is one or two basis points, came back at **41.85 bps at
  t = 1.93** — the most nearly resolved of the seven, with a ceiling of 59.7.
* The European Property Yield fund, a small sector ETF whose half-spread is
  fifteen to thirty, came back at **3.18 bps**, the tightest of the book.

The ranking check compared the seven against median daily traded value and
found **ρ = +0.89**: the most liquid instruments estimated widest. It refused
to write, which is what it is for, and it was not weakened, given a tolerance,
or turned into a warning. The per-bar autocorrelation was negative on all
seven and beyond −0.10 on four — EURO STOXX Banks −0.403, VanEck
Semiconductors −0.197, Physical Gold −0.162, Europe Industrials −0.138.

What is and is not established:

* The estimator is not the fault on its own: it passes six controls on
  simulated bars, and #14's measurement shows that carried bars alone produce
  exactly this shape — an inflated ceiling with an error bar far wider than
  the sample allows — with the fault falling on whichever lines the feed
  carries most.
* Carried bars are **not** what produces the negative autocorrelation. On
  simulated bars they push it positive (+0.03 at 5%, up to +0.12 at 15%), and
  nothing tried — carried closes, carried opens, contaminated ranges, flat
  bars — pushes it negative. Seven negatives with four beyond −0.10 is a
  signature of something not yet simulated, and it is left as an observation
  rather than attributed.
* Whether Yahoo's European bars carry closes at a rate that explains a factor
  of twenty is not established from here. It is exactly what the two-column
  survey now counts and what the ladder decides: SPY and AAPL inside their
  bands with SU.PA and MEUD.PA outside theirs is the provider; SPY outside
  its band is the pipeline. That run needs the provider and is the user's.
* The book's symbol for Schneider is one thing the ladder and the survey now
  print side by side. SU.PA is Euronext Paris; a symbol resolving to a
  secondary listing would be a wide spread measured correctly on the wrong
  line, and the check's old verdict would, for once, have been right for the
  wrong reason.

The 8 bps constant stays, and stays described as a constant. None of the
ceilings was written. Every number above is under investigation and none of
them is a measurement of a spread.

## Why not Streamlit any more

Every widget interaction re-ran the whole script, re-derived every position and
re-serialised every chart. That is the performance ceiling, and the visual
ceiling: no drawers, no command palette, no client-side sorting, no live
what-if sliders. The maths in `core/` was never the problem and is untouched.

## Performance rules

1. One request per page load: `GET /api/snapshot` carries everything Overview,
   Holdings, Performance and Risk need. It is assembled once and held in memory
   by `api/services.py` until the ledger changes or prices are refreshed.
2. Network calls happen only inside `data/market.py`, concurrently (thread
   pool, at most 6 workers), through the SQLite cache in `data/cache.py`.
   History is fetched incrementally (from the last cached date), quotes have a
   15-minute TTL. A restart never refetches history that is already on disk.
3. Nothing polls. The refresh button (`POST /api/refresh`) is the only path that
   bypasses the quote TTL.
4. The SPA renders from one JSON object; sorting, filtering, range selection
   and the simulator's before/after diff are client-side or a single small POST.
5. Charts: ECharts with `useDirtyRect` / lazy update; every chart is disposed on
   unmount; window resize is debounced once for all charts.

## core contract (new modules)

Every function is pure, typed, documented, and tested in `portfolio/tests/`.
Money in `Decimal` at the ledger boundary, `float` inside numpy code. Series
are indexed by `pd.DatetimeIndex` (dates, no time). ISIN is the key everywhere.

### core/performance.py

```python
@dataclass(frozen=True)
class ValueHistory:
    value: pd.Series           # holdings market value in base, per date
    cost_basis: pd.Series      # cost basis of open positions, per date
    invested: pd.Series        # cumulative net external flow, per date
    flows: pd.Series           # external flow per date (+ = money in), base
    per_holding: pd.DataFrame  # value per ISIN per date (columns = ISIN)
    quantities: pd.DataFrame   # units per ISIN per date
    flows_per_holding: pd.DataFrame
    missing: tuple[str, ...]   # ISINs with no price history, excluded and named
    warnings: tuple[str, ...]

def value_history(transactions: list[Transaction],
                  histories: dict[str, pd.Series],        # ISIN -> adjusted close in quote ccy
                  quote_currencies: dict[str, str],       # ISIN -> quote currency
                  rates: FxRates | None = None,
                  base: str = BASE_CURRENCY,
                  start: dt.date | None = None) -> ValueHistory
```

Flow convention (holdings-only portfolio, no cash account; flows at the
start of the day): BUY = +gross+fees in; SELL = -(gross-fees) out;
DIVIDEND = -net out; FEE = +fee in (paid from outside, no value change, so it
depresses the return). Quantity per date is a step function of the ledger
replayed on the union of price calendars; price gaps are forward-filled and
counted in `warnings`.

```python
def time_weighted_return(value: pd.Series, flows: pd.Series) -> pd.Series
    # r_t = V_t / (V_{t-1} + F_t) - 1 ; 0 where the denominator is 0
def return_index(daily: pd.Series, start: float = 100.0) -> pd.Series
def annualised_return(daily: pd.Series, trading_days: int = 252) -> float | None
def xirr(cashflows: list[tuple[dt.date, float]]) -> float | None
    # Newton with bisection fallback; None if it cannot converge or < 2 flows
def money_weighted_return(flows: pd.Series, terminal_value: float,
                          terminal_date: dt.date) -> float | None
def period_returns(daily: pd.Series, freq: str) -> pd.Series   # "M" or "Y", compounded
def monthly_table(daily: pd.Series) -> pd.DataFrame            # index=year, columns=1..12, NaN gaps
def drawdown_series(index: pd.Series) -> pd.Series             # <= 0
def rolling_volatility(daily: pd.Series, window: int = 63, trading_days: int = 252) -> pd.Series
def rolling_beta(daily: pd.Series, benchmark: pd.Series, window: int = 63) -> pd.Series
def trailing_returns(index: pd.Series, as_of: dt.date | None = None) -> dict[str, float | None]
    # keys: "1w", "1m", "3m", "6m", "ytd", "1y", "all"

@dataclass(frozen=True)
class HoldingContribution:
    isin: str; pnl: float; contribution: float   # contribution = pnl / start value

def holding_contributions(history: ValueHistory,
                          start: dt.date | None = None,
                          end: dt.date | None = None) -> list[HoldingContribution]

@dataclass(frozen=True)
class PerformanceSummary:
    total_return: float | None; annualised_return: float | None
    money_weighted: float | None
    volatility: float | None; sharpe: float | None; sortino: float | None
    calmar: float | None; max_drawdown: float | None; current_drawdown: float | None
    best_day: float | None; worst_day: float | None
    best_month: float | None; worst_month: float | None
    positive_months: int; negative_months: int
    observations: int; first_date: dt.date | None; last_date: dt.date | None
    # against a benchmark, all None without one:
    benchmark_total_return: float | None; beta: float | None; alpha: float | None
    correlation: float | None; tracking_error: float | None; information_ratio: float | None

def performance_summary(daily: pd.Series, benchmark: pd.Series | None = None,
                        risk_free: float = 0.0, flows: pd.Series | None = None,
                        terminal_value: float | None = None) -> PerformanceSummary
```

### core/var.py

```python
@dataclass(frozen=True)
class VarEstimate:
    method: str            # "historical" | "parametric" | "cornish_fisher"
    confidence: float      # 0.95, 0.99
    horizon_days: int
    loss: float            # positive fraction: 0.031 means a 3.1% loss
    expected_shortfall: float
    observations: int

def historical_var(daily: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> VarEstimate
def parametric_var(daily: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> VarEstimate
def cornish_fisher_var(daily: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> VarEstimate
def var_report(daily: pd.Series, confidences=(0.95, 0.99), horizons=(1, 10)) -> list[VarEstimate]

@dataclass(frozen=True)
class WorstPeriod:
    start: dt.date; end: dt.date; length_days: int; ret: float

def worst_periods(daily: pd.Series, length_days: int = 1, n: int = 5) -> list[WorstPeriod]

@dataclass(frozen=True)
class Scenario:
    key: str; label: str; description: str
    portfolio_return: float
    per_holding: dict[str, float]     # ISIN -> return applied

def stress_scenarios(returns: pd.DataFrame, weights: dict[str, float]) -> list[Scenario]
    # historical: worst day / worst 5 days / worst 21 days of the window;
    # hypothetical: every holding repeats its own worst day; every holding -10%;
    # the largest holding -25% alone; correlation goes to 1 at current vols.
```

### core/simulate.py

```python
@dataclass(frozen=True)
class SimulatedHolding:
    isin: str
    value_before: float; value_after: float
    weight_before: float; weight_after: float
    risk_before: float; risk_after: float      # share of portfolio risk
    marginal_after: float                      # MCTR after, annualised

@dataclass(frozen=True)
class SimulationResult:
    holdings: list[SimulatedHolding]
    total_before: float; total_after: float
    volatility_before: float; volatility_after: float      # annualised
    effective_before: float; effective_after: float
    diversification_before: float; diversification_after: float
    max_risk_share_before: float; max_risk_share_after: float

def simulate_cash_changes(values: dict[str, float], cov: pd.DataFrame,
                          changes: dict[str, float]) -> SimulationResult
    # values: current market value per ISIN; changes: +/- base currency per ISIN.
    # An ISIN in `changes` but not in `values` must be a column of cov (watchlist).
def simulate_target_weights(values: dict[str, float], cov: pd.DataFrame,
                            targets: dict[str, float]) -> SimulationResult
def equal_weights(keys: list[str]) -> dict[str, float]
def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 10_000, tol: float = 1e-10) -> dict[str, float]
def min_variance_weights(cov: pd.DataFrame, max_iter: int = 10_000) -> dict[str, float]   # long-only

@dataclass(frozen=True)
class Trade:
    isin: str; action: str          # "BUY" | "SELL"
    amount: float                   # base currency, positive
    units: float | None             # amount / price when a price is known
    weight_before: float; weight_after: float

def trades_to_target(values: dict[str, float], targets: dict[str, float],
                     prices: dict[str, float] | None = None,
                     total: float | None = None, min_amount: float = 0.0) -> list[Trade]
```

### core/exposure.py

```python
@dataclass(frozen=True)
class ExposureSlice:
    key: str; label: str; value: float; weight: float; count: int; isins: tuple[str, ...]

DIMENSIONS = ("asset_class", "issuer", "base_currency", "quote_currency", "exchange")
def exposure_by(values: dict[str, float], instruments: dict[str, Instrument], dimension: str) -> list[ExposureSlice]
def exposures(values: dict[str, float], instruments: dict[str, Instrument]) -> dict[str, list[ExposureSlice]]
```

### core/alerts.py

```python
class Severity(str, enum.Enum): INFO, WARNING, SERIOUS, CRITICAL

@dataclass(frozen=True)
class Alert:
    code: str; severity: Severity; title: str; detail: str
    isins: tuple[str, ...] = (); route: str = ""   # SPA route to act on it

def build_alerts(holdings: HoldingsTable, instruments, *,
                 fetch_failures: list[str] = (),
                 concentration: ConcentrationStats | None = None,
                 pairs: list[CorrelationPair] = (),
                 clusters: list[CorrelationCluster] = (),
                 alignment: AlignmentReport | None = None,
                 max_weight: float = 0.35) -> list[Alert]
```
Sorted by severity then title. Rules: unpriced holding (SERIOUS), fetch failure
(SERIOUS), stale/outdated price (WARNING), a holding above `max_weight`
(WARNING), effective holdings below half of actual (WARNING), high-correlation
pair (WARNING), cluster of 3+ (SERIOUS), instrument excluded from the risk model
(INFO), window shortened (INFO), watchlist-only universe (INFO).

### core/risk.py additions

```python
@dataclass(frozen=True)
class CorrelationCluster:
    members: tuple[str, ...]; mean_correlation: float; min_correlation: float
    combined_weight: float | None

def correlation_clusters(corr: pd.DataFrame, threshold: float = HIGH_CORRELATION_THRESHOLD,
                         weights: dict[str, float] | None = None) -> list[CorrelationCluster]
    # connected components over edges >= threshold, size >= 2, largest weight first
```

## data contract (new modules)

### data/migrations.py

```python
TRADING_COLUMNS = ("broker", "tradeable", "tob_rate", "tob_observed",
                   "half_spread_bps", "spread_observed", "buy_tax_rate")

def missing_columns(path: Path) -> list[str]     # the trigger: absent from the header
def derived_facts(inst: Instrument) -> list[Backfill]
def backfill_trading_facts(instruments, path, fillable=None) -> MigrationReport
```

`DataStore.load_instruments` calls this when the file's header lacks any of
those columns, backs the file up, rewrites it and prints what it filled in.

The trigger is the **header**, never a blank cell, and `fillable` is a hard
limit rather than a hint. A blank `tob_rate` under a `tob_rate` header is a
recorded statement that the rate is not established -- exactly what the gold
ETC needs -- and a derived default would destroy it on every load. Only a
column that is not there at all can be said to have no answer in it. This
also makes the migration one-shot by construction.

It fills only what the instrument record determines: the tax band from the
asset class (a debt security does not take a fund's band, so an ETC gets
nothing and stays unpriceable), and the French FTT for French shares. It does
not invent a broker -- which institution holds a position is not a property of
the instrument, and the two brokers here have opposite fee shapes, so the
wrong schedule changes which trades are affordable rather than changing a cost
slightly. Everything derived is written with `tob_observed = False` and named
in the notice.

Populating what cannot be derived is `portfolio instruments set`, which
parses `0.0012`, `0,0012` and `0.12%` alike and refuses a bare `0.12` as
ambiguous, because a hundredfold error from one keystroke has nothing to
notice it by.

### data/providers/fixture.py  (exists)
`FixtureProvider(MarketDataProvider)`: deterministic synthetic prices for any
symbol, FX pairs and benchmarks included. Same interface as `YahooProvider`.
Used by every API test and by `portfolio serve --provider fixture`.

### data/cache.py
```python
class PriceCache:
    def __init__(self, path: pathlib.Path) -> None          # sqlite file, created
    def get_history(self, symbol: str) -> tuple[pd.Series, dt.datetime] | None   # series, fetched_at
    def put_history(self, symbol: str, series: pd.Series) -> None                 # upsert rows
    def get_quote(self, symbol: str, max_age: dt.timedelta) -> Quote | None
    def put_quote(self, symbol: str, quote: Quote) -> None
    def stats(self) -> CacheStats            # symbols, rows, size_bytes, oldest, newest
    def clear(self) -> None
```

### data/fx.py
```python
FX_PAIRS: dict[str, str]     # "USD" -> "USDEUR=X", GBP, CHF, SEK, DKK, NOK, JPY, CAD, AUD
class DailyFxTable:          # implements core.money.FxRates; O(log n) lookups
    def add_series(self, frm: str, to: str, series: pd.Series) -> None
    def rate(self, frm, to, on) -> Decimal      # exact, else most recent earlier, else inverse, else MissingRate
    def latest(self, frm, to) -> tuple[dt.date, Decimal] | None
def currencies_needed(instruments, transactions, base="EUR") -> set[str]
```

### data/market.py
```python
@dataclass
class Failure: key: str; name: str; message: str

@dataclass
class MarketSnapshot:
    quotes: dict[str, PriceQuote]              # by ISIN
    histories: dict[str, pd.Series]            # by ISIN, adjusted close in quote ccy
    benchmarks: dict[str, pd.Series]           # by symbol
    fx: DailyFxTable
    failures: list[Failure]
    fetched_at: dt.datetime; elapsed_seconds: float; provider: str
    cache_hits: int; cache_misses: int

class MarketData:
    def __init__(self, provider: MarketDataProvider, cache: PriceCache | None,
                 base: str = "EUR", max_workers: int = 6,
                 quote_ttl: dt.timedelta = 15 min, history_ttl: dt.timedelta = 6 h) -> None
    def load(self, instruments: dict[str, Instrument], benchmark_symbols=(),
             currencies=(), force: bool = False) -> MarketSnapshot
```
Never raises for a single symbol: failures are collected. `force=True` ignores
the quote TTL but still fetches history incrementally.

### data/importers.py
```python
@dataclass
class ColumnMapping: date, isin, type, quantity, price, currency, fees, note: str | None
                     date_format: str | None; decimal_comma: bool
def detect_mapping(headers: list[str]) -> ColumnMapping
def preview_import(text: str, mapping: ColumnMapping | None = None) -> ImportPreview
    # ImportPreview(rows: list[ImportRow], mapping, valid, invalid, unknown_isins)
    # ImportRow(line: int, ok: bool, transaction: Transaction | None, error: str | None, raw: dict)
```

### data/export.py
```python
def transactions_csv(transactions, instruments) -> str
def holdings_csv(table: HoldingsTable) -> str
def workbook(sheets: dict[str, pd.DataFrame]) -> bytes      # xlsx via openpyxl
```

## API contract

Prefix `/api`. JSON bodies are defined in `portfolio/api/schemas.py` (pydantic)
and that file is the source of truth for field names. Errors are
`{"error": {"code": str, "message": str}}` with 400/404/409/502.

| Method | Path | Purpose |
|---|---|---|
| GET | /health | `{status, version, mode, provider}` |
| GET | /settings | mode, data dir, provider, cache stats, last refresh |
| PUT | /settings | `{mode}` switch seed/user |
| POST | /refresh | reload prices ignoring the quote TTL |
| GET | /snapshot?benchmark=&lookback= | everything for Overview, Holdings, Performance, Risk |
| GET | /holdings/{isin} | detail: price series, trade markers, stats, transactions |
| GET | /instruments | universe |
| POST | /instruments/resolve | `{isin}` -> Resolution (candidates, refused, block reason) |
| POST | /instruments | save a confirmed candidate |
| PATCH | /instruments/{isin} | edit fields / symbols / note / active |
| DELETE | /instruments/{isin} | refused with 409 when transactions reference it |
| GET | /transactions?include_voided= | ledger |
| POST | /transactions | append |
| POST | /transactions/{id}/void | `{reason}` |
| POST | /transactions/import/preview | `{text, mapping?}` -> ImportPreview |
| POST | /transactions/import | `{text, mapping?}` -> `{imported}` |
| GET | /export/transactions.csv, /export/holdings.csv, /export/workbook.xlsx | downloads |
| POST | /simulate | `{changes}` or `{targets}` or `{preset}` -> SimulationResult + trades |
| POST | /allocate | `{amount, target?}` -> buy-only order, four floors, destinations, refusals |
| GET | /benchmarks | list |

`GET /` and any non-API path serve `web/index.html`; `/static/*` serves `web/`.

## web contract

Hash routes: `#/overview` (default), `#/holdings`, `#/holdings/{isin}`,
`#/performance`, `#/risk`, `#/simulator`, `#/allocate`, `#/instruments`,
`#/transactions`, `#/settings`. Files: `web/index.html`, `web/app.js` (shell + router),
`web/pages/*.js`, `web/components/*.js`, `web/lib/{api,format,charts,theme,store}.js`,
`web/styles/*.css`, `web/vendor/*` (ECharts 5.6, Preact+htm standalone, Inter).

`#/allocate` is the one page that answers a decision rather than describing a
state, and its order is the argument. The headline says whether the amount moves
the book at all — if 200 EUR does not, that sentence comes *before* any table of
destinations, since a table printed first implies the choice is worth making.
Beside it, and deliberately separate, is what the choice itself is worth: the two
were once merged and printed "where this goes matters" directly above "closes
2.3% of the gap". Then four floors rather than one, the whole-share order, what
each destination costs against what it buys, and what was refused. It computes
nothing: every figure is one POST to `/allocate`, which runs the same
`agents.allocate` the CLI runs and the replay evaluates.

Design principles carried over from the Streamlit version, because they were
right: colour is reserved for state (green = gain, red = loss, amber = warning)
and for polarity/magnitude in charts; a holding behaving as expected is visually
silent; tabular figures in every column; warnings sit next to the number they
qualify; demo mode is announced on every view; delayed prices are never shown
as live. Dark theme first, light theme selected (not inverted). The categorical
palette is the validated one in `web/styles/tokens.css`.
