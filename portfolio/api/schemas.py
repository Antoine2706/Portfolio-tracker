"""The JSON contract between the API and the web application.

This file is the source of truth for every field name the SPA reads. Every
model here is a plain projection of a `core` structure: nothing is computed
in this layer, and nothing is renamed on the way to the browser.

Conventions
-----------
- Money and prices are floats in the reporting currency unless a `currency`
  field says otherwise. Decimal precision is a ledger concern; by the time a
  figure is on screen it is a float with a fixed display precision.
- Fractions are fractions: a weight of 12% is 0.12, a return of -3.1% is
  -0.031. The client formats.
- Dates are ISO strings (`YYYY-MM-DD`), timestamps ISO 8601 with offset.
- Anything that may be unavailable is `None`, never 0 and never a sentinel.
  The client is expected to render "not available" rather than a number.
- Series are parallel arrays (`dates`, `values`) rather than lists of objects,
  because that is what a chart wants and it is a third of the bytes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------


class Series(BaseModel):
    dates: list[str]
    values: list[float | None]


class Failure(BaseModel):
    key: str
    name: str
    message: str


class Meta(BaseModel):
    mode: Literal["seed", "user"]
    provider: str
    base_currency: str
    generated_at: str
    prices_as_of: str | None
    elapsed_ms: int
    cache_hits: int
    cache_misses: int
    failures: list[Failure]
    instrument_count: int
    transaction_count: int


# --------------------------------------------------------------------------
# Holdings
# --------------------------------------------------------------------------


class Totals(BaseModel):
    value: float
    cost_basis: float
    unrealised: float
    unrealised_pct: float | None
    realised: float
    dividends: float
    fees: float
    invested: float
    day_change: float | None
    day_change_pct: float | None
    priced_holdings: int
    unpriced_holdings: int


class Holding(BaseModel):
    isin: str
    name: str                           # the registered name, for detail and tooltips
    short_name: str                     # the label form, for charts and dense tables
    symbol: str | None
    asset_class: str
    issuer: str
    exchange: str
    quantity: float
    avg_cost: float
    cost_basis: float
    price: float | None
    price_currency: str | None
    price_as_of: str | None
    price_delay_minutes: int | None
    price_is_stale: bool
    price_note: str                     # provenance sentence, always present
    value: float | None
    unrealised: float | None
    unrealised_pct: float | None
    realised: float
    dividends: float
    fees: float
    weight: float | None
    day_change: float | None
    day_change_pct: float | None
    risk_share: float | None
    divergence: float | None
    volatility: float | None            # annualised standalone
    sparkline: list[float]              # last ~30 closes, quote currency
    fx_note: str | None
    warnings: list[str]
    transaction_count: int


class WatchlistItem(BaseModel):
    isin: str
    name: str
    short_name: str
    symbol: str | None
    price: float | None
    price_currency: str | None
    day_change_pct: float | None
    volatility: float | None
    sparkline: list[float]
    in_risk_model: bool                 # has enough history to simulate with


# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------


class Flow(BaseModel):
    date: str
    amount: float
    type: str
    isin: str
    name: str
    short_name: str


class PerformanceSummary(BaseModel):
    total_return: float | None
    annualised_return: float | None
    money_weighted: float | None
    volatility: float | None
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    max_drawdown: float | None
    current_drawdown: float | None
    best_day: float | None
    worst_day: float | None
    best_month: float | None
    worst_month: float | None
    positive_months: int
    negative_months: int
    observations: int
    first_date: str | None
    last_date: str | None
    benchmark_total_return: float | None
    beta: float | None
    alpha: float | None
    correlation: float | None
    tracking_error: float | None
    information_ratio: float | None


class MonthlyReturn(BaseModel):
    year: int
    month: int
    ret: float | None
    benchmark_ret: float | None


class YearlyReturn(BaseModel):
    year: int
    ret: float | None
    benchmark_ret: float | None


class Contribution(BaseModel):
    isin: str
    name: str
    short_name: str
    pnl: float
    contribution: float


class Performance(BaseModel):
    available: bool
    reason: str | None
    dates: list[str]
    value: list[float | None]
    invested: list[float | None]
    cost_basis: list[float | None]
    index: list[float | None]           # TWR index, 100 at the first date
    benchmark_index: list[float | None]
    drawdown: list[float | None]
    benchmark_drawdown: list[float | None]
    flows: list[Flow]
    summary: PerformanceSummary | None
    trailing: dict[str, float | None]
    benchmark_trailing: dict[str, float | None]
    monthly: list[MonthlyReturn]
    yearly: list[YearlyReturn]
    contributions: list[Contribution]
    rolling_volatility: Series
    rolling_beta: Series
    per_holding_value: dict[str, list[float | None]]
    holding_names: dict[str, str] = {}  # ISIN -> registered name, closed positions too
    holding_short_names: dict[str, str] = {}   # ISIN -> label form, same coverage
    missing: list[str]
    warnings: list[str]


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


class Window(BaseModel):
    requested: int
    effective: int
    first_date: str | None
    last_date: str | None
    binding_isin: str | None
    binding_name: str | None
    binding_observations: int | None
    excluded: list[dict]                # {isin, name, observations, reason}
    warnings: list[str]


class DivergenceRow(BaseModel):
    isin: str
    name: str
    short_name: str
    weight: float
    risk_share: float
    divergence: float
    marginal: float                     # MCTR, annualised
    sentence: str


class Metric(BaseModel):
    key: str
    label: str
    value: float | None
    display: str
    sentence: str
    warning: str | None


class Standalone(BaseModel):
    isin: str
    name: str
    short_name: str
    volatility: float
    multiple: float                     # times a broad European index
    weight: float


class Correlation(BaseModel):
    isins: list[str]
    names: list[str]                    # registered names, for the tooltip
    short_names: list[str]              # label form, when a ticker is unavailable
    matrix: list[list[float]]


class Pair(BaseModel):
    a: str
    b: str
    a_name: str
    b_name: str
    a_short: str
    b_short: str
    correlation: float
    sentence: str


class Cluster(BaseModel):
    members: list[str]
    names: list[str]
    short_names: list[str]
    mean_correlation: float
    min_correlation: float
    combined_weight: float | None


class VarRow(BaseModel):
    method: str
    confidence: float
    horizon_days: int
    loss: float
    loss_amount: float | None
    expected_shortfall: float
    expected_shortfall_amount: float | None
    observations: int


class WorstPeriod(BaseModel):
    start: str
    end: str
    length_days: int
    ret: float
    amount: float | None


class Scenario(BaseModel):
    key: str
    label: str
    description: str
    portfolio_return: float
    portfolio_amount: float | None
    per_holding: dict[str, float]


class Risk(BaseModel):
    available: bool
    reason: str | None
    window: Window | None
    headline: DivergenceRow | None
    divergence: list[DivergenceRow]
    metrics: list[Metric]
    volatility: float | None
    volatility_multiple: float | None
    diversification_ratio: float | None
    effective_holdings: float | None
    actual_holdings: int
    max_drawdown: float | None
    current_drawdown: float | None
    beta: float | None
    beta_benchmark: str | None
    standalone: list[Standalone]
    correlation: Correlation | None
    pairs: list[Pair]
    clusters: list[Cluster]
    threshold: float
    var: list[VarRow]
    worst_periods: list[WorstPeriod]
    scenarios: list[Scenario]
    portfolio_returns: Series           # daily portfolio return over the window


# --------------------------------------------------------------------------
# Exposure, alerts, benchmarks
# --------------------------------------------------------------------------


class ExposureSlice(BaseModel):
    key: str
    label: str
    value: float
    weight: float
    count: int
    isins: list[str]


class Alert(BaseModel):
    code: str
    severity: Literal["INFO", "WARNING", "SERIOUS", "CRITICAL"]
    title: str
    detail: str
    isins: list[str]
    route: str


class Benchmark(BaseModel):
    symbol: str
    name: str
    index: str
    label: str
    currency: str
    note: str
    is_default: bool


class Snapshot(BaseModel):
    meta: Meta
    totals: Totals
    holdings: list[Holding]
    watchlist: list[WatchlistItem]
    unpriced: list[str]
    performance: Performance
    risk: Risk
    exposure: dict[str, list[ExposureSlice]]
    alerts: list[Alert]
    benchmarks: list[Benchmark]
    selected_benchmark: str
    lookback: int


# --------------------------------------------------------------------------
# Holding detail
# --------------------------------------------------------------------------


class Marker(BaseModel):
    date: str
    type: str
    quantity: float
    price: float
    currency: str


class HoldingDetail(BaseModel):
    holding: Holding | WatchlistItem
    instrument: "InstrumentOut"
    prices: Series
    drawdown: Series
    markers: list[Marker]
    transactions: list["TransactionOut"]
    stats: dict[str, float | None]      # volatility, max_drawdown, beta_vs_portfolio,
                                        # correlation_vs_portfolio, contribution, trailing_*
    correlations: list[Pair]            # this holding against every other


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------


class InstrumentOut(BaseModel):
    isin: str
    name: str
    short_name: str                     # always filled: derived when not set by hand
    short_name_is_manual: bool          # so the form can say whether it is derived
    issuer: str
    asset_class: str
    base_currency: str
    primary_symbol: str
    exchange: str
    quote_currency: str
    provider_symbols: dict[str, str]
    active: bool
    manual_overrides: list[str]
    note: str
    held: bool
    transaction_count: int


class InstrumentPatch(BaseModel):
    name: str | None = None
    short_name: str | None = None       # "" clears it, returning to the derived form
    issuer: str | None = None
    asset_class: str | None = None
    base_currency: str | None = None
    provider_symbols: dict[str, str] | None = None
    note: str | None = None
    active: bool | None = None


class ResolveRequest(BaseModel):
    isin: str
    lookback: int = 252


class CandidateOut(BaseModel):
    isin: str
    ticker: str
    yahoo_symbol: str
    venue_label: str
    mic: str | None
    verdict: str
    selectable: bool
    observations: int
    first_date: str | None
    last_date: str | None
    currency: str | None
    reported_exchange: str | None
    name: str
    reasons: list[str]
    description: str


class ResolutionOut(BaseModel):
    isin: str
    candidates: list[CandidateOut]
    refused: list[CandidateOut]
    recommended: str | None             # yahoo_symbol
    blocked: bool
    block_reason: str | None
    summary: str
    listings_seen: int
    filtered_by_class: dict[str, int]
    errors: list[str]


class SaveInstrumentRequest(BaseModel):
    isin: str
    yahoo_symbol: str
    name: str
    issuer: str = ""
    asset_class: str = "ETF"
    base_currency: str = "EUR"
    quote_currency: str | None = None
    mic: str | None = None
    ticker: str | None = None


# --------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------


class TransactionOut(BaseModel):
    id: str
    date: str
    isin: str
    name: str
    short_name: str
    type: str
    quantity: float
    price_per_unit: float
    currency: str
    fees: float
    gross: float
    net: float                          # signed cash effect in the transaction currency
    note: str
    voided: bool
    void_reason: str | None


class TransactionIn(BaseModel):
    date: str
    isin: str
    type: str
    quantity: str | float = 0
    price_per_unit: str | float = 0
    currency: str = "EUR"
    fees: str | float = 0
    note: str = ""


class VoidRequest(BaseModel):
    reason: str = ""


class ColumnMappingIn(BaseModel):
    date: str | None = None
    isin: str | None = None
    type: str | None = None
    quantity: str | None = None
    price: str | None = None
    currency: str | None = None
    fees: str | None = None
    note: str | None = None
    date_format: str | None = None
    decimal_comma: bool = False


class ImportRequest(BaseModel):
    text: str
    mapping: ColumnMappingIn | None = None


class ImportRowOut(BaseModel):
    line: int
    ok: bool
    error: str | None
    raw: dict[str, str]
    transaction: TransactionOut | None


class ImportPreviewOut(BaseModel):
    headers: list[str]
    mapping: ColumnMappingIn
    rows: list[ImportRowOut]
    valid: int
    invalid: int
    unknown_isins: list[str]


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


class SimulateRequest(BaseModel):
    changes: dict[str, float] | None = None       # ISIN -> +/- base currency
    targets: dict[str, float] | None = None       # ISIN -> weight
    preset: Literal["equal", "risk_parity", "min_variance"] | None = None
    include: list[str] = Field(default_factory=list)   # extra ISINs (watchlist) to admit
    min_trade: float = 0.0


class SimulatedHolding(BaseModel):
    isin: str
    name: str
    value_before: float
    value_after: float
    weight_before: float
    weight_after: float
    risk_before: float
    risk_after: float
    marginal_after: float


class Trade(BaseModel):
    isin: str
    name: str
    short_name: str
    action: str
    amount: float
    units: float | None
    price: float | None
    weight_before: float
    weight_after: float


class SimulationOut(BaseModel):
    holdings: list[SimulatedHolding]
    total_before: float
    total_after: float
    volatility_before: float
    volatility_after: float
    effective_before: float
    effective_after: float
    diversification_before: float
    diversification_after: float
    max_risk_share_before: float
    max_risk_share_after: float
    trades: list[Trade]
    weights_after: dict[str, float]


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class CacheStats(BaseModel):
    path: str
    symbols: int
    rows: int
    size_bytes: int
    oldest: str | None
    newest: str | None


class Settings(BaseModel):
    mode: Literal["seed", "user"]
    data_dir: str
    provider: str
    base_currency: str
    cache: CacheStats | None
    last_refresh: str | None
    version: str
    quote_ttl_minutes: int
    lookback: int


class SettingsPatch(BaseModel):
    mode: Literal["seed", "user"] | None = None


class Health(BaseModel):
    status: str
    version: str
    mode: str
    provider: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorOut(BaseModel):
    error: ErrorBody


HoldingDetail.model_rebuild()


# --------------------------------------------------------------------------
# Directing new money
# --------------------------------------------------------------------------


class AllocateRequest(BaseModel):
    amount: float                                 # new cash, base currency
    target: float | None = None                   # also: what would this cost?


class AllocationPurchase(BaseModel):
    isin: str
    name: str
    broker: str
    shares: int
    price: float
    amount: float
    cost: float
    weight_before: float
    weight_after: float
    risk_before: float
    risk_after: float


class AllocationDestination(BaseModel):
    """What the whole purchase in one holding would do, and what it would cost.

    `cost` is null when the broker has no published fee for an order that
    size, which is a refusal rather than a zero.
    """
    isin: str
    name: str
    dispersion: float
    cost: float | None
    improvement: float
    per_euro: float | None


class AllocationRefusal(BaseModel):
    isin: str
    name: str
    reason: str


class AllocationDilution(BaseModel):
    """A holding the purchase pushes further from its equal-risk weight.

    New money that cannot enter a holding still makes it a smaller share of a
    larger book. `drift` is positive when the purchase widens the gap to
    `target`, which is a cost of the recommendation and belongs on the screen
    rather than in the reader's head.
    """
    isin: str
    name: str
    weight_before: float
    weight_after: float
    target: float
    drift: float
    reason: str


class AllocationOut(BaseModel):
    cash: float
    invested: float
    leftover: float
    book_value: float
    purchases: list[AllocationPurchase]
    destinations: list[AllocationDestination]
    refused: list[AllocationRefusal]
    diluted: list[AllocationDilution]
    # Three floors, not one: where the book is, the best reachable with this
    # much money, and the best reachable if selling were allowed.
    # `dispersion_*` is the coefficient of variation of the risk shares --
    # what the solver minimises and what the page leads with. `spread_*` is
    # the range over the same mean, descriptive only: it says how far apart
    # the extremes are, which a CV does not, and nothing optimises it.
    dispersion_now: float
    dispersion_after: float
    spread_now: float
    spread_after: float
    floor_at_cash: float
    floor_unlimited: float
    rounding_penalty: float
    closable: float
    cost_parts: dict[str, float]
    total_cost: float
    meaningful_cash: float | None
    best_worst_gap: float
    assumptions: list[str]
    target: float | None = None
    # `cash_for_target` is null when no purchase the search considered reaches
    # the target; `best_reachable` and `best_reachable_cash` then say what the
    # best is and roughly what it would take, because "no" on its own is not a
    # decision. `pinned` names the holdings that cannot receive new money --
    # the reason the floor stops falling and starts rising, and the reason the
    # amount above is the smallest found rather than provably the smallest.
    cash_for_target: float | None = None
    best_reachable: float | None = None
    best_reachable_cash: float | None = None
    pinned: list[str] = []
