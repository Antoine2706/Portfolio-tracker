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

The comparison against buy-and-hold is **paired**: the two legs hold the same
book on the same days, so their returns correlate to about 0.99 and almost
all of each Sharpe's error is the same error, cancelling in the difference.
The tool reports the active return — policy minus benchmark, day by day —
with its own error bar, and below t = 2 it says the two are
*indistinguishable* rather than ranking them. The 1.5 alarm is applied to
that active ratio as well as to the level, because a level above 1.5 is a
property of the window and the benchmark shares the window; a leak inside a
policy cannot lift a benchmark that never trades.

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
