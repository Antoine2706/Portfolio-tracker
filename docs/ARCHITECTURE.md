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

**The objective is not convex,** and saying so matters. The reported metric is
`risk_contribution_spread` — a maximum minus a minimum — which is neither
smooth nor convex; nor is the least-squares alternative, since risk shares are
ratios of quadratics. The solve is therefore projected gradient descent on the
smooth surrogate to *generate candidates*, and a multi-resolution pattern
search on the reported metric to *choose between them*. Descending on the
surrogate and calling its answer the floor was measurably beaten by a coarse
brute-force grid: least squares equalises four holdings and abandons the
fifth, while a range prefers lifting the laggards. The result is labelled
"best found", and `test_allocate.py` checks it against a dense grid.

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
estimated and carries its own. The whole-share remainder is carried to the
next purchase, because over a ledger it compounded to 6.5% of the money and
would otherwise leave one arm quietly part in cash. The replay is checked for
look-ahead the same way the harness is: rewrite every price after a date and
require every earlier decision back unchanged.

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
| GET | /benchmarks | list |

`GET /` and any non-API path serve `web/index.html`; `/static/*` serves `web/`.

## web contract

Hash routes: `#/overview` (default), `#/holdings`, `#/holdings/{isin}`,
`#/performance`, `#/risk`, `#/simulator`, `#/instruments`, `#/transactions`,
`#/settings`. Files: `web/index.html`, `web/app.js` (shell + router),
`web/pages/*.js`, `web/components/*.js`, `web/lib/{api,format,charts,theme,store}.js`,
`web/styles/*.css`, `web/vendor/*` (ECharts 5.6, Preact+htm standalone, Inter).

Design principles carried over from the Streamlit version, because they were
right: colour is reserved for state (green = gain, red = loss, amber = warning)
and for polarity/magnitude in charts; a holding behaving as expected is visually
silent; tabular figures in every column; warnings sit next to the number they
qualify; demo mode is announced on every view; delayed prices are never shown
as live. Dark theme first, light theme selected (not inverted). The categorical
palette is the validated one in `web/styles/tokens.css`.
