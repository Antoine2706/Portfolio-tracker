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
  tests/     No network (sockets are blocked), no UI framework required for
             core tests. API tests use FastAPI's TestClient + FixtureProvider.
```

Dependency arrows point inward: `web -> api -> data -> core`. `core` never
imports from `data` or `api`; `api` never imports numpy or scipy (the layering
tests enforce both with `ast`).

Run it: `pip install -e ".[app,data]"` then `portfolio serve` (opens the
browser at http://127.0.0.1:8765). `portfolio serve --provider fixture` runs a
fully offline demo with deterministic synthetic prices. `PORTFOLIO_DATA_MODE`
selects `seed` (demo) or `user` (your data), default `seed`.

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
