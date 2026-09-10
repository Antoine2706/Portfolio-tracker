# Portfolio tracker

A personal investment portfolio tracker for European-listed ETFs and ETCs,
keyed on ISIN, with risk analytics you can read against a textbook.

The number no broker app shows you is the gap between a holding's share of
your money and its share of your risk. That gap is the headline of this tool;
everything else — performance attribution, value at risk, correlation
clusters, a what-if simulator, rebalancing — is supporting evidence for it.

![Overview](docs/screenshots/overview.png)

## Run it

```bash
pip install -e ".[app,data]"
portfolio serve                       # opens http://127.0.0.1:8765
```

It starts in **demo mode** on ten real reference instruments and a synthetic
ledger, clearly labelled on every page. Switch to your own data from the
settings page or with `portfolio serve --mode user`; your instruments and
ledger live in `portfolio/data_store/user/` (gitignored) as plain CSV that
outlives the tool.

No network? `portfolio serve --provider fixture` runs the full application on
deterministic synthetic prices.

## What it does

| Page | What you get |
|---|---|
| **Overview** | Portfolio value, day change, unrealised and realised P&L, value history against a benchmark, allocation, the risk headline, and an alert feed (stale prices, concentration, correlated holdings). |
| **Holdings** | Dense sortable table with sparklines, weight, risk share, day change and provenance for every price. Click a row for the holding's price chart with your trades marked, drawdown, beta and correlation to the rest of the book. |
| **Performance** | Time-weighted and money-weighted (XIRR) returns, benchmark comparison, drawdown, monthly and calendar-year returns, contribution by holding, rolling volatility and beta, Sharpe / Sortino / Calmar. |
| **Risk** | Capital share against risk share (component contribution to volatility), effective number of holdings, diversification ratio, beta against a named benchmark, value at risk (historical, parametric, Cornish-Fisher) and expected shortfall, worst periods, stress scenarios, the correlation grid, and correlation **clusters** — the group a pairwise grid cannot see. |
| **Simulator** | Add or remove money from any holding or watchlist instrument, set target weights, or apply equal-weight / risk-parity / minimum-variance presets, and see volatility, effective holdings and every risk share move before you trade. Produces the trade list. |
| **Allocate** | Where a purchase should go, buy-only. Type an amount; get whole shares per holding with the risk shares before and after, what it costs split into tax, spread and commission, and — first, before any table of destinations — whether the amount moves the book at all. |
| **Instruments** | Add by ISIN: resolved via OpenFIGI, every listing probed, US ticker collisions refused outright, thin and stale series flagged. Nothing is saved until you confirm the listing you saw. |
| **Transactions** | Append-only ledger. Corrections are voids, never edits. CSV import with column detection (European decimal commas and semicolons included) and duplicate detection. CSV and Excel export. |

Prices are delayed and are never presented as live. A holding that cannot be
valued is named and excluded from the total rather than valued at zero. A
missing FX rate raises rather than defaulting to parity.

| Risk | Simulator (risk parity) |
|---|---|
| ![Risk](docs/screenshots/risk.png) | ![Simulator](docs/screenshots/simulator.png) |
| **Performance** | **Holding detail** |
| ![Performance](docs/screenshots/performance.png) | ![Holding detail](docs/screenshots/holding-detail.png) |

Every chart has a table twin (the grid icon in its corner), the whole
application is reachable from the keyboard (`?` lists the shortcuts, `Ctrl K`
opens the command palette), and dark and light themes are designed
separately rather than inverted.

## Architecture

```
portfolio/
  core/    pure numpy + pandas: ledger, positions, returns, risk, performance,
           VaR, simulator. No network, no UI, importable and tested alone.
  data/    CSV store, SQLite price cache, yfinance + OpenFIGI providers,
           daily FX, CSV import/export, deterministic offline provider.
  api/     FastAPI: one cached snapshot per (mode, benchmark, lookback),
           invalidated by ledger writes or an explicit refresh. Projects only.
  web/     Zero-build single-page client: Preact + htm + ECharts, vendored.
  agents/  Decision policies. A policy is an object with one method, not a
           language model: it sees a point-in-time view and returns target
           weights with a written reason. Includes the trading cost model.
  eval/    The walk-forward harness, the statistics that say whether a result
           means anything, the pre-registration log, and the controls that
           calibrate all of it.
```

The layering is enforced by tests, not by convention: `core/` cannot import a
web framework or a network client, `api/` cannot import numpy, and `agents/`
cannot import `eval/` — a policy that could tell it was being backtested
could behave differently there. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the contract between layers.

## Evaluating a strategy

```bash
portfolio controls          # calibrate the harness before trusting a result
```

A backtest produces a confident Sharpe ratio whether or not it is measuring
anything, so the harness is calibrated before any strategy is run against it:

| Control | What it proves |
|---|---|
| **Negative** | A no-skill policy with matched turnover, over 200 seeds. The harness must show no edge, reject a true null at close to 5%, and produce t-statistics with standard deviation 1 — a standard error wrong by a factor shows up here as its reciprocal. Measured at both overlap settings, because it once ran only at the one the real backtest does not use. |
| **Positive** | A policy given a known probability of foreseeing the next bar, in a world where its true Sharpe is available in closed form. The measured figure must match the injected one. Without this, "we found no edge" and "we could not have found an edge" are the same sentence. |
| **Canary** | A policy that decides today using tomorrow's close. It must produce an absurd Sharpe. If it does not, the data feed leaks. |
| **Leak detector** | Rewrites every price after a date and re-runs. Everything decided before it must be bit-identical. It also refuses to run when wired so that it could not fail. |

Costs are mandatory, not optional: `walk_forward` takes a cost model, and the
one in `agents/execution.py` is per broker, per instrument and side-aware,
because at a real account all three vary. Every input is either read off a
contract note or marked as an estimate, and the model prints which — and how
much of the answer each is carrying:

> Transaction tax: 1 of 7 rates was read off a contract note, covering 17% of
> the book by value. Of the remaining 6: 5 (80%) assumed to sit in the same
> band, 1 (3%) not recorded at all, so trades in it are refused rather than
> priced.
>
> Weighted by what the model actually charges on a round trip, 9% of the cost
> figure rests on inputs read off a document and 91% on estimates, of which
> the bid-ask spread is 39%: it is paid inside the execution price and appears
> on no contract note.

Where those facts live is `portfolio instruments`:

```bash
portfolio instruments list                        # what is evidence, what is not
portfolio instruments set --all --broker MeDirect
portfolio instruments set IE00B579F325 --broker Keytrade --not-tradeable
portfolio instruments set DE000A2QP372 --tob-rate 0.12% --observed
```

Rates parse as `0.0012`, `0,0012` or `0.12%` interchangeably and a bare
`0.12` is refused as ambiguous, because a hundredfold error from one
keystroke has nothing to notice it by. `--observed` cannot be set without the
number it describes. A book written before these columns existed is migrated
on load, with a notice naming every value it derived and every one it
refused to.

```bash
portfolio backtest erc      # equal risk contribution against buy-and-hold
```

### The bid-ask spread, which appears on no document

```bash
portfolio spreads                  # each instrument's spread, from its own bars
portfolio spreads --write          # commit the ones that clear every gate
portfolio controls --spread        # prove the estimator recovers spreads it wasn't told
portfolio controls --spread-ladder # run the whole data path on SPY, AAPL, SU.PA...
portfolio controls --spread-reference SPY --split-at 2001-01-01 --split-at 2002-01-01
                                   # the authors' package on the same bars, adjusted
                                   # vs unadjusted, blocks by date, the four moments
```

Commission and transaction tax are printed on the confirmation. The spread is
paid *inside* the execution price, so there is no arithmetic that recovers it
from anything the broker sent — and it is the largest single component of the
cost model. Until now it was a flat 8 bps for every instrument, which cannot
be right for a €12bn core tracker and a thin thematic fund at once.

It is now estimated per instrument by **EDGE** (Ardia, Guidotti & Kroencke,
*JFE* 161, 2024), written out in `core/spread.py` rather than imported. The
mid-range (h+l)/2 is bounce-free when a bar holds both a buy and a sell, so
pairing it against the open and the previous close isolates s²/4 four separate
ways — Roll's covariance argument, run against a clean price rather than
another contaminated one.

**It has a noise floor of about 4 bps and says so.** At zero true spread it
reports 3.97 bps off 500 daily bars, because √|s²| folds a mean-zero
distribution onto the positive side. That floor thins only as n^(−¼) —
measured at 0.85 per doubling against a predicted 0.841 — so a 2 bps floor
would need twenty-four years of bars.

Which decides how much of a book this can speak to at all. Measured, as the
fraction of samples in which a spread is claimed as a measurement:

| true half-spread | 250 bars | 500 | 1000 | 2500 |
|---|---|---|---|---|
| 2 bps | 0% | 2% | 4% | 5% |
| 4 bps | 4% | 4% | 15% | 16% |
| 6 bps | 15% | 22% | 44% | 91% |
| 8 bps | 28% | 66% | 80% | 99% |
| 10 bps | 49% | 82% | 94% | 100% |
| 15 bps | 96% | 100% | 100% | 100% |

So the estimate is taken over the instrument's **whole available history**,
not over the holding period. A covariance window must be short because
correlations move with regime; a spread is a microstructure property that
moves slowly and has nothing to do with when the holding was bought. That
moves the point at which half of samples resolve from about 10 bps to about
6 — and it plainly does *not* rescue the tight end, where the fraction
resolved is flat in the sample length because no daily window reaches it.

The long window has a cost — spreads narrow as a fund grows, so a ten-year
estimate can average a market that no longer exists — so it is measured
rather than assumed. The estimate is computed over each nested window, the
sequence is printed, and **disjoint** older blocks are compared against the
most recent at three standard errors. Flat means take everything; a real
difference means the spread has moved and the window stops there.

Four tiers, and the third is why any of this is useful on a tight book:

| tier | meaning |
|---|---|
| `observed` | somebody watched it. Outranks everything below, and `--write` will not overwrite it |
| `estimated` | EDGE on that instrument's own bars, clearing **all** of: two standard errors clear of zero *on s²*, at or above half a tick, off at least 60 bars |
| `bounded` | not distinguishable from zero, so the **upper confidence bound** `√(s² + 2·SE)` is charged and labelled a ceiling |
| `assumed` | the declared constant, for an instrument that was not measurable at all |

Failing the significance test does not mean nothing was learned: it puts a
ceiling on the spread, and that ceiling is per instrument because the standard
error depends on that instrument's own volatility and bar count. So the
differentiation survives even where nothing resolves, and unlike a constant it
is falsifiable — a bound below a spread later seen on a quote screen is a bug
report. What a ceiling is *not* is safe: it is only as good as the bars under
it, and on the first real book this ran on, the two of seven ceilings that
could be checked were both wrong — one twenty-fold too wide on the holding
whose spread is least in doubt, one too tight on a thin fund. That run is
recorded in `docs/ARCHITECTURE.md`.

The tick is inferred from the prices themselves (the coarsest grid essentially
every print falls on) rather than from a table of venue rules nobody here
could check, and off **recent** bars only: a split divides older prices by its
ratio and takes them off any grid, and the MiFID II regime bands by price, so
the tick that applies today is the one today's prices are on. A recent tail
that is off grid means an adjusted series arrived, and an adjusted price never
traded — that instrument is refused rather than measured.

Seven controls back it, and they are the reason to believe any of the above:
a positive sweep from 2 to 100 bps reporting bias and dispersion at each rung,
a negative control measuring the floor and checking it thins as sampling noise
must, a standard error checked against the dispersion it claims to predict, a
resolution control, a resolution-by-window control that says what a longer
history actually buys, a refusal below a minimum bar count, and a
contamination control that regenerates the table of what each kind of
manufactured bar does to the estimate. Plus a ranking check on the result as
a whole: estimated spread against median daily traded value, because an
ordering that contradicts liquidity is more likely a wiring fault than a
market fact, and `--write` refuses when it fails.

Those seven validate the estimator on simulated bars. They say nothing about
whether a provider's bars are what the estimator assumes, and the first real
book failed the ranking check at ρ = +0.89 with the most liquid holding
estimated widest. So the survey now runs **twice** per instrument — on every
bar the provider sent, and again with bars identical to the previous day's,
closes identical to the previous close, impossible bars and zero-volume bars
set aside — and prints both with the counts. Measured on simulated bars, a
whole bar carried forward on five per cent of days inflates a 10 bps spread
to 18 and a carried close to 16, and setting those days aside recovers it;
widening the high and low does nothing. What no count of bars can see is a
**daily reversal in the price itself** — a stale close the market has moved
past by the morning — which inflates the same spread to 22 at a return
autocorrelation of −0.15: at the daily frequency a bounce and a reversal are
the same covariance, and that is the estimator's identification limit. The
liquidity proxy is printed per instrument so a failed ranking can be laid at
the right number, and every run is appended to `spread-runs.jsonl` in the
mode's data directory, whether it passed or not.

`portfolio controls --spread-ladder` is the eighth control and the only one
that needs the real provider. It decides which end of the path is wrong: the
survey's estimation path on SPY, AAPL, SU.PA, MEUD.PA and a thin European
fund, each against a band its spread is known to sit in, with the stated
figure printed beside the band. US tight and Europe wide is the provider's
European bars; SPY wide is the pipeline. An unresolved rung is judged on its
ceiling against the ceiling a clean sample of the same length and volatility
reports, because that is the case the real book presented and the case the
first version of the verdict passed. Every rung also prints its per-bar
autocorrelation beside the estimate — the one number from the real book that
no simulated contamination reproduces — so that SPY and AAPL say whether the
path manufactures it or the European lines own it. Against the fixture the
ladder is expected to fail — which is how the check is seen to bite.

The first real ladder run gave the pipeline verdict, correctly: SPY read about
19 bps, and all eleven lines collapsed into 8 to 34 bps whatever their true
spread — one additive term in s², not eleven faults. `portfolio controls
--spread-reference SYMBOL` is the tool for that question and changes nothing
else: on the exact bars the survey reads it runs the authors' own `bidask`
package (same number means the transcription is faithful and the fault is in
the input; a different one means the implementation diverges on real data),
the adjusted series beside the unadjusted one (a factor constant within a bar
cancels in every log ratio), blocks cut at dates you give (`--split-at`), and
the estimator's four moment conditions separately, because a contaminated
open inflates the two that use the open and a contaminated close the one that
uses the close. Nothing is corrected until that run is back.

### Directing new money

```bash
portfolio allocate 5000                 # where a purchase should go
portfolio allocate 5000 --target 1.0    # and: what would reaching 1.0 take?
portfolio backtest allocator            # replay your real purchases
```

In an account with no regular contribution and no leverage, a scheduled
rebalance is a sell plus a buy — tax and spread twice, for every unit of
drift removed. New money is the one rebalancing channel that costs nothing
extra, because the purchase was going to happen anyway. So the allocator
chooses **where** it goes, subject to `b ≥ 0`: it never proposes a sale.

**Dispersion** here is the coefficient of variation of the risk shares — their
standard deviation over their mean, zero when every holding carries the same
share of the volatility, at most √(m−1) for m holdings. It is what the solver
minimises *and* what it reports, which sounds like a tautology and is the fix
for a real defect: it used to report a range (max − min, over the mean) while
minimising a sum of squared deviations, so the tool graded answers by a rule it
had not used to produce them. A range is also a rank statistic — on six
holdings it is decided by two and discards the other four. It is still printed,
beside the dispersion and never instead of it, because it says how far apart
the extremes are.

Buy-only can only reduce an overweight risk share by dilution, so the output
carries three numbers rather than one — where the book is now, the best
reachable *with this much money*, and the best reachable if selling were
allowed. It answers the inverted question too, which is usually the
decision-relevant one — *reaching 1.00 would take about 40,000 EUR against a
book of 17,000* — and when the answer is that nothing reaches it, it says
what the best reachable is and at what amount, because "no" on its own is
not a decision. And it says when the destination hardly matters:

> At 200 EUR this hardly moves the book: the best any purchase this size
> reaches is 1.2126 against 1.2299 now. The smallest purchase that closes a
> worthwhile share of the gap is about 732 EUR.
>
> Best and worst destinations differ by 0.031 of dispersion, so the choice is
> worth making.

Two sentences, because they answer different questions and merging them once
printed *where this goes matters* directly above *closes 2.3% of the gap*: how
much the purchase moves the book, and how much the choice of destination is
worth. A purchase too small to matter can still have a destination worth
avoiding. The threshold is a **share of the gap** to equal risk contribution
rather than an absolute number of dispersion units — as an absolute 0.05 it
asked for a third of the whole gap on a book sitting at 1.5 and 0.3% of it on a
book at 15, which is a different question on every book it is applied to.

Whole shares only, with the rounding penalty shown against the continuous
optimum; per-broker minimum trade sizes derived from each fee schedule; and a
table of what each destination would cost against what it would buy, so a
wide-spread holding cannot quietly consume the improvement it delivers.

`buyable` is a separate flag from `tradeable`. A holding at a second broker
cannot be rebalanced against the rest of the book but can be bought with new
money perfectly well; reading one flag as the other gets the constraint wrong
in one direction or the other.

**More money does not always help, and the tool no longer pretends it does.**
Rescaling a reachable allocation leaves every risk share unchanged and stays
affordable, so the best reachable dispersion falls as the purchase grows —
*provided every holding can receive it*. One that cannot has its weight pinned
at `v/(V+C)`, an equality that moves with the money rather than a bound that
relaxes, so more cash dilutes it towards a zero risk share and past some
amount the floor turns and climbs. On the demo book it converges to 10/7,
which is what seven equal risk shares and three zeros disperse to. The inverse
question is therefore answered by a scan rather than a bisection: the amount
named always reaches the target, and what a pinned holding costs is the
guarantee that nothing smaller would.

The allocator is evaluated by **replaying the real ledger** — same dates,
same amounts, destination changed — because its claim is about risk
structure, not return. Dispersion is computed rather than estimated, so that
comparison carries no sampling error; realised volatility is estimated and
carries its standard error, and the correlation between the two arms is
measured and printed rather than described, because that is what decides how
much of each marginal error cancels. Below five purchases the report says
`READ THIS AS n POINTS, NOT AS A SERIES` and explains why.

Disposals have no neutral treatment, so both are offered and the report names
which ran. By default a sale is applied to *both* arms as the same
proportional withdrawal: risk shares are homogeneous of degree zero, so it
changes none of them in either arm, and the whole ledger is covered — at the
cost of the actual arm being "the purchases that were made, with disposals
taken pro-rata" rather than the book that was held. That is deliberate:
selling a particular holding is a decision the allocator never makes, and
crediting one arm for it would measure that instead of the destination.
`--stop-at-first-sale` gives the strict version over a shorter window.

reports the policy gross, net and against the benchmark, then the **breakeven
turnover** — the level at which the gross edge is entirely consumed by the
trading it takes to capture it. That is the decision criterion, and it
separates the two ways a policy fails: no edge at all, or an edge too small to
trade. The benchmark is always buy-and-hold of the portfolio you already own,
started from the same weights, since that is the actual alternative.

Holdings can be marked non-tradeable. A position at a second broker cannot be
rebalanced against the rest of the book — that is a cash transfer between
institutions taking about a week, not a trade — so its weight is exogenous.
It stays in the risk model, since it genuinely affects every correlation, and
the referee rejects any proposal that moves it. A holding whose trading cost
cannot be established is frozen the same way and for a stated reason: refusing
to price it is absolute, but refusing to produce any result at all was not the
same thing. Equal risk contribution then solves over the restricted simplex
and reports what the constraint cost.

Every statistic is reported with its uncertainty, and every Sharpe ratio is
printed with its standard error beside it. A ratio the sample cannot
establish is reported as undetermined, with the track record length that
*would* establish it — a true annual Sharpe of 0.5 needs sixteen years of
daily data to reach t = 2, and the tool says so rather than printing a
number. Attempts are pre-registered in `research/registry.jsonl`, because the
deflated Sharpe ratio takes the number of attempts as an argument and a count
that omits the failures is not a count.

The comparison against buy-and-hold is **paired and risk-adjusted**, and it
reports two statistics because they answer different questions.

The raw active return — policy minus benchmark, day by day — is the realised
cost of the mandate *in that window*, and it is reported rather than tested:
a policy that holds less risk underperforms a rising benchmark by
construction, so a significant result there is a tautology. The same policy
in a falling window comes out significantly better. The tool demonstrates
this on itself.

The comparison that carries out of sample is at matched risk: the difference
of the two Sharpe ratios, tested with Jobson-Korkie plus Memmel's correction
at a correlation that is measured and printed. Two legs holding one book
correlate at about 0.99, so almost all of each marginal error is the same
error and cancels — which is what makes the test possible. Below t = 2 the
tool says *indistinguishable* rather than ranking them. It also splits the
raw gap into the part explained by the volatility ratio and the residual, and
states that matching risk means leverage a cash account cannot apply: the raw
gap is what the account lost, the residual is what the strategy cost.

The 1.5 alarm is applied to the matched-risk gap as well as to the level. A
level above 1.5 is a property of the window and the benchmark shares it; a
leak inside a policy cannot lift a benchmark that never trades.

A day on which a held instrument did not trade is never a 0.00% return.
Positions are valued at the last price that printed, so the missed move lands
on the next one and the compounded return is exactly right; the day itself is
excluded from every variance, and the count of such days is printed beside
the observation count.

Paper trading only. Automatic execution on a real account is portfolio
management under MiFID II; `portfolio/tests/test_paper_only.py` fails if a
live broker endpoint appears anywhere in the repository.

Performance: every network call goes through a thread pool and an on-disk
cache; history is fetched incrementally and quotes expire after 15 minutes.
Nothing polls. After the first load, a page is a dictionary lookup.

## Tests

```bash
pip install -e ".[app,data,test]"
pytest                    # everything, including the API against the offline provider
pip install -e ".[test]" && pytest     # core only: proves core needs nothing else
```

## Data providers

yfinance for prices, OpenFIGI for identity. Both are free and keyless; an
`OPENFIGI_API_KEY` raises the identity rate limit. The provider spike that
chose them, with its evidence, is in `spike/`.
