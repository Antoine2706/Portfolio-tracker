"""Snapshot assembly: core structures projected into the JSON contract.

This module calls `core` and fills in `schemas`. It does not compute. The
layering test forbids numpy here; pandas is allowed only for reshaping frames
that core already produced into the parallel arrays the charts want. If a
number is needed that no core function returns, the right fix is a core
function with a test, not arithmetic in this file.

The `Analysis` object keeps the intermediate frames (aligned returns, the
covariance matrix, the value history) alongside the finished snapshot, so the
holding-detail and simulate endpoints can answer from them without building
anything twice.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal
from typing import Callable

import pandas as pd

from ..core.alerts import build_alerts
from ..core.exposure import exposures
from ..core.models import Amendment, Instrument, Transaction
from ..core.money import BASE_CURRENCY, FxRates, MissingRate, convert
from ..core.overview import day_change, position_totals, sparkline
from ..core.performance import (ValueHistory, drawdown_series, holding_contributions,
                                monthly_table, performance_summary, period_returns,
                                return_index, rolling_beta, rolling_volatility,
                                time_weighted_return, trailing_returns)
from ..core.positions import Position, derive_positions, weights
from ..core.report import (HoldingsTable, correlation_sentences, divergence_rows,
                           holdings_table, risk_metrics)
from ..core.returns import (MIN_OBSERVATIONS, AlignmentReport, InsufficientHistory,
                            align_returns, simple_returns)
from ..core.risk import (BROAD_EUROPEAN_EQUITY_VOLATILITY, HIGH_CORRELATION_THRESHOLD,
                         RiskDecomposition, annualise_volatility, beta, concentration,
                         correlation_clusters, correlation_matrix, covariance_matrix,
                         diversification_ratio, drawdown, high_correlation_pairs,
                         normalise_weights, portfolio_return_series,
                         portfolio_value_series, risk_decomposition,
                         standalone_volatilities)
from ..core.var import stress_scenarios, var_report, worst_periods
from ..data.benchmarks import BENCHMARKS, Benchmark
from ..data.market import MarketSnapshot
from ..data.store import DataMode
from . import schemas as S
from .config import Config

__all__ = ["Analysis", "build", "holding_detail", "transaction_out", "instrument_out"]

SPARKLINE_POINTS = 30
ROLLING_WINDOW = 63


@dataclasses.dataclass
class Analysis:
    """Everything derived for one (mode, benchmark, lookback)."""
    snapshot: S.Snapshot
    instruments: dict[str, Instrument]
    transactions: list[Transaction]
    positions: dict[str, Position]
    table: HoldingsTable
    market: MarketSnapshot
    values: dict[str, float]                   # base-currency value per priced holding
    prices_base: dict[str, float]              # unit price in base per priced holding
    returns: pd.DataFrame | None               # aligned returns, modelled ISINs
    cov: pd.DataFrame | None
    corr: pd.DataFrame | None
    decomposition: RiskDecomposition | None
    model_weights: dict[str, float]
    alignment: AlignmentReport | None
    portfolio_returns: pd.Series | None
    history: ValueHistory | None
    twr: pd.Series | None
    benchmark: Benchmark
    lookback: int
    names: dict[str, str]


# --------------------------------------------------------------------------
# Small projections
# --------------------------------------------------------------------------


def _f(value) -> float | None:
    """Decimal/number to float, None preserved. NaN becomes None."""
    if value is None:
        return None
    out = float(value)
    return None if out != out else out


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, dt.datetime):
        return value.isoformat(timespec="minutes")
    return value.isoformat()


def _dates(index) -> list[str]:
    return [pd.Timestamp(d).date().isoformat() for d in index]


def _series(s: pd.Series | None) -> S.Series:
    if s is None or len(s) == 0:
        return S.Series(dates=[], values=[])
    return S.Series(dates=_dates(s.index), values=[_f(v) for v in s.to_numpy()])


def _values(s: pd.Series | None, index) -> list[float | None]:
    if s is None:
        return [None] * len(index)
    aligned = s.reindex(index)
    return [_f(v) for v in aligned.to_numpy()]


def _when(v: dt.date | dt.datetime | None) -> str:
    """A quote timestamp as people read it: '4 Sep 2026, 17:30 UTC' or '4 Sep 2026'."""
    if v is None:
        return "?"
    if isinstance(v, dt.datetime):
        u = v.astimezone(dt.timezone.utc) if v.tzinfo else v
        return f"{u.day} {u:%b %Y}, {u:%H:%M} UTC"
    return f"{v.day} {v:%b %Y}"


def _provenance(pos: Position) -> str:
    q = pos.quote
    if q is None:
        return "no price available"
    when = _when(q.as_of)
    if q.is_stale:
        return f"last close {when} ({q.source}) - not a live price"
    delay = f", delayed ~{q.delay_minutes} min" if q.delay_minutes else ", delay unstated"
    return f"as of {when} ({q.source}{delay})"


def _to_base(amount: Decimal, currency: str, on: dt.date, rates: FxRates | None,
             base: str) -> float | None:
    if currency == base:
        return float(amount)
    if rates is None:
        return None
    try:
        return float(convert(_money(amount, currency), base, on, rates).amount)
    except MissingRate:
        return None


def _money(amount: Decimal, currency: str):
    from ..core.money import Money
    return Money(amount, currency)


def instrument_out(inst: Instrument, transactions: list[Transaction]) -> S.InstrumentOut:
    n = sum(1 for t in transactions if t.isin == inst.isin)
    return S.InstrumentOut(
        isin=inst.isin, name=inst.name, issuer=inst.issuer,
        asset_class=inst.asset_class.value, base_currency=inst.base_currency,
        primary_symbol=inst.primary_symbol, exchange=inst.exchange,
        quote_currency=inst.quote_currency, provider_symbols=dict(inst.provider_symbols),
        active=inst.active, manual_overrides=sorted(inst.manual_overrides),
        note=inst.note, held=n > 0, transaction_count=n)


def transaction_out(t: Transaction, names: dict[str, str],
                    voided_by: dict[str, Amendment] | None = None) -> S.TransactionOut:
    amendment = (voided_by or {}).get(t.id)
    return S.TransactionOut(
        id=t.id, date=t.date.isoformat(), isin=t.isin,
        name=names.get(t.isin, t.isin), type=t.type.value,
        quantity=float(t.quantity), price_per_unit=float(t.price_per_unit),
        currency=t.currency, fees=float(t.fees), gross=float(t.gross.amount),
        net=float(t.cash_flow().amount), note=t.note,
        voided=amendment is not None,
        void_reason=amendment.reason if amendment else None)


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------


def build(*, instruments: dict[str, Instrument], all_instruments: dict[str, Instrument],
          transactions: list[Transaction], amendments: list[Amendment],
          market: MarketSnapshot, benchmark: Benchmark, lookback: int,
          config: Config, mode: DataMode,
          symbol_for: Callable[[Instrument], str | None] | None = None) -> Analysis:
    started = dt.datetime.now(dt.timezone.utc)
    # The fetch handle each instrument was actually loaded with. The offline
    # provider borrows the yfinance symbols, so the market layer decides.
    symbol_for = symbol_for or (lambda inst: inst.provider_symbols.get(market.provider))
    base = config.base_currency
    today = dt.date.today()
    names = {isin: inst.name for isin, inst in all_instruments.items()}
    fx = market.fx

    # ---- positions and the holdings table --------------------------------
    positions = derive_positions(transactions, instruments, rates=fx,
                                 quotes=market.quotes, strict=False)
    table = holdings_table(positions, instruments, rates=fx, as_of=today, base=base)
    held = weights(positions, fx, on=today, base=base)
    values = {row.isin: float(row.market_value.amount)
              for row in table.rows if row.market_value is not None}
    prices_base = {}
    for row in table.rows:
        if row.price is not None and row.quantity:
            unit = _to_base(row.price.amount, row.price.currency, today, fx, base)
            if unit is not None:
                prices_base[row.isin] = unit

    # ---- risk --------------------------------------------------------------
    risk_state = _risk(held, market, instruments, names, benchmark, lookback, values)

    # ---- performance -----------------------------------------------------
    perf_state = _performance(transactions, instruments, market, benchmark, names,
                              positions, base, fx)

    # ---- holdings rows ---------------------------------------------------
    day_changes = {}
    for isin in positions:
        hist = market.histories.get(isin)
        if hist is not None:
            dc = day_change(hist)
            if dc is not None:
                day_changes[isin] = dc
    totals_core = position_totals(positions, day_changes, rates=fx, on=today, base=base)

    holdings = []
    for row in table.rows:
        pos = positions[row.isin]
        inst = instruments.get(row.isin)
        hist = market.histories.get(row.isin)
        dc = day_changes.get(row.isin)
        change_base = None
        if dc is not None and pos.quote is not None:
            change_base = _to_base(dc.amount * pos.quantity, pos.quote.price.currency,
                                   today, fx, base)
        holdings.append(S.Holding(
            isin=row.isin, name=row.name,
            symbol=symbol_for(inst) if inst else None,
            asset_class=inst.asset_class.value if inst else "OTHER",
            issuer=inst.issuer if inst else "", exchange=inst.exchange if inst else "",
            quantity=float(row.quantity), avg_cost=float(row.average_cost.amount),
            cost_basis=float(pos.cost_basis.amount),
            price=_f(row.price.amount) if row.price else None,
            price_currency=row.price.currency if row.price else None,
            price_as_of=_iso(pos.quote.as_of) if pos.quote else None,
            price_delay_minutes=pos.quote.delay_minutes if pos.quote else None,
            price_is_stale=bool(pos.quote.is_stale) if pos.quote else False,
            price_note=_provenance(pos),
            value=_f(row.market_value.amount) if row.market_value else None,
            unrealised=_f(row.unrealised.amount) if row.unrealised else None,
            unrealised_pct=_f(row.unrealised_pct),
            realised=float(pos.realised_pnl.amount), dividends=float(pos.dividends.amount),
            fees=float(pos.fees_paid.amount), weight=_f(row.weight),
            day_change=change_base, day_change_pct=dc.pct if dc else None,
            risk_share=risk_state.risk_share.get(row.isin),
            divergence=risk_state.divergence.get(row.isin),
            volatility=risk_state.standalone.get(row.isin),
            sparkline=sparkline(hist, SPARKLINE_POINTS) if hist is not None else [],
            fx_note=row.fx_note, warnings=list(row.warnings),
            transaction_count=pos.transaction_count))

    watchlist = []
    for isin, pos in positions.items():
        inst = instruments.get(isin)
        if not pos.is_watchlist or inst is None or not inst.active:
            continue
        hist = market.histories.get(isin)
        dc = day_changes.get(isin)
        vol = None
        if hist is not None and len(hist) > MIN_OBSERVATIONS:
            vol = float(annualise_volatility(float(simple_returns(hist).std(ddof=1))))
        watchlist.append(S.WatchlistItem(
            isin=isin, name=inst.name, symbol=symbol_for(inst),
            price=_f(pos.quote.price.amount) if pos.quote else None,
            price_currency=pos.quote.price.currency if pos.quote else None,
            day_change_pct=dc.pct if dc else None, volatility=vol,
            sparkline=sparkline(hist, SPARKLINE_POINTS) if hist is not None else [],
            in_risk_model=hist is not None and len(hist) >= MIN_OBSERVATIONS))

    invested = None
    if perf_state.history is not None and len(perf_state.history.invested):
        invested = _f(perf_state.history.invested.iloc[-1])
    totals = S.Totals(
        value=float(table.total_value.amount),
        cost_basis=float(totals_core.cost_basis.amount),
        unrealised=float(table.total_unrealised.amount),
        unrealised_pct=(float(table.total_unrealised.amount / totals_core.cost_basis.amount)
                        if totals_core.cost_basis.amount else None),
        realised=float(totals_core.realised.amount),
        dividends=float(totals_core.dividends.amount),
        fees=float(totals_core.fees.amount),
        invested=invested if invested is not None else float(totals_core.cost_basis.amount),
        day_change=_f(totals_core.day_change.amount) if totals_core.day_change else None,
        day_change_pct=totals_core.day_change_pct,
        priced_holdings=len(values), unpriced_holdings=len(table.unpriced))

    # ---- exposure, alerts --------------------------------------------------
    exposure = {dim: [S.ExposureSlice(key=s.key, label=s.label, value=s.value,
                                      weight=s.weight, count=s.count, isins=list(s.isins))
                      for s in slices]
                for dim, slices in exposures(values, instruments).items()}

    alerts = build_alerts(
        table, instruments,
        fetch_failures=[f"{f.name}: {f.message}" for f in market.failures],
        concentration=risk_state.concentration, pairs=risk_state.pairs,
        clusters=risk_state.clusters, alignment=risk_state.alignment)

    prices_as_of = None
    stamps = [pos.quote.as_of for pos in positions.values() if pos.quote is not None]
    if stamps:
        latest = max((s.date() if isinstance(s, dt.datetime) else s) for s in stamps)
        prices_as_of = latest.isoformat()

    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    meta = S.Meta(
        mode=mode.value, provider=market.provider, base_currency=base,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        prices_as_of=prices_as_of,
        elapsed_ms=int(round((market.elapsed_seconds + elapsed) * 1000)),
        cache_hits=market.cache_hits, cache_misses=market.cache_misses,
        failures=[S.Failure(key=f.key, name=f.name, message=f.message)
                  for f in market.failures],
        instrument_count=len(all_instruments), transaction_count=len(transactions))

    snapshot = S.Snapshot(
        meta=meta, totals=totals, holdings=holdings, watchlist=watchlist,
        unpriced=list(table.unpriced), performance=perf_state.out, risk=risk_state.out,
        exposure=exposure,
        alerts=[S.Alert(code=a.code, severity=a.severity.value, title=a.title,
                        detail=a.detail, isins=list(a.isins), route=a.route)
                for a in alerts],
        benchmarks=[S.Benchmark(symbol=b.symbol, name=b.name, index=b.index,
                                label=b.label, currency=b.currency, note=b.note,
                                is_default=b is BENCHMARKS[0]) for b in BENCHMARKS],
        selected_benchmark=benchmark.symbol, lookback=lookback)

    return Analysis(
        snapshot=snapshot, instruments=instruments, transactions=transactions,
        positions=positions, table=table, market=market, values=values,
        prices_base=prices_base, returns=risk_state.returns, cov=risk_state.cov,
        corr=risk_state.corr, decomposition=risk_state.decomposition,
        model_weights=risk_state.weights, alignment=risk_state.alignment,
        portfolio_returns=risk_state.portfolio_returns, history=perf_state.history,
        twr=perf_state.twr, benchmark=benchmark, lookback=lookback, names=names)


# --------------------------------------------------------------------------
# Risk block
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _RiskState:
    out: S.Risk
    returns: pd.DataFrame | None = None
    cov: pd.DataFrame | None = None
    corr: pd.DataFrame | None = None
    decomposition: RiskDecomposition | None = None
    weights: dict[str, float] = dataclasses.field(default_factory=dict)
    alignment: AlignmentReport | None = None
    portfolio_returns: pd.Series | None = None
    concentration: object = None
    pairs: list = dataclasses.field(default_factory=list)
    clusters: list = dataclasses.field(default_factory=list)
    risk_share: dict[str, float] = dataclasses.field(default_factory=dict)
    divergence: dict[str, float] = dataclasses.field(default_factory=dict)
    standalone: dict[str, float] = dataclasses.field(default_factory=dict)


def _unavailable(reason: str, actual: int) -> S.Risk:
    return S.Risk(
        available=False, reason=reason, window=None, headline=None, divergence=[],
        metrics=[], volatility=None, volatility_multiple=None,
        diversification_ratio=None, effective_holdings=None, actual_holdings=actual,
        max_drawdown=None, current_drawdown=None, beta=None, beta_benchmark=None,
        standalone=[], correlation=None, pairs=[], clusters=[],
        threshold=HIGH_CORRELATION_THRESHOLD, var=[], worst_periods=[], scenarios=[],
        portfolio_returns=S.Series(dates=[], values=[]))


def _risk(held: dict[str, Decimal], market: MarketSnapshot,
          instruments: dict[str, Instrument], names: dict[str, str],
          benchmark: Benchmark, lookback: int, values: dict[str, float]) -> _RiskState:
    if len(held) < 2:
        return _RiskState(_unavailable(
            f"Not enough holdings for a risk model. You have {len(held)} priced "
            f"position(s); covariance needs at least two. A single holding has no "
            f"diversification to measure, so these figures would be zeros rather "
            f"than answers.", len(held)))

    series = {isin: market.histories[isin] for isin in held if isin in market.histories}
    missing = [names.get(isin, isin) for isin in held if isin not in market.histories]
    if len(series) < 2:
        return _RiskState(_unavailable(
            "Could not load price history for at least two holdings. "
            + ("Missing: " + ", ".join(missing) + "." if missing else ""), len(held)))
    try:
        returns, alignment = align_returns(series, lookback=lookback)
    except InsufficientHistory as exc:
        return _RiskState(_unavailable(f"No covariance matrix can be built. {exc}",
                                       len(held)))

    used = [i for i in returns.columns if i in held]
    w = normalise_weights({i: float(held[i]) for i in held}, used)
    if not w:
        return _RiskState(_unavailable(
            "Excluded holdings carry all the weight; nothing left to model.", len(held)))

    cov = covariance_matrix(returns[used])
    decomposition = risk_decomposition(w, cov)
    rows = divergence_rows(decomposition, instruments)
    stats = concentration(list(w.values()))
    corr = correlation_matrix(cov)
    pairs = high_correlation_pairs(corr)
    clusters = correlation_clusters(corr, weights=w)
    sentences = correlation_sentences(pairs, instruments)
    portfolio_returns = portfolio_return_series(returns[used], w)
    dd = drawdown(portfolio_value_series(returns[used], w))
    standalone = standalone_volatilities(cov)

    beta_value = None
    bench_hist = market.benchmarks.get(benchmark.symbol)
    if bench_hist is not None:
        try:
            beta_value = beta(portfolio_returns, simple_returns(bench_hist))
        except ValueError:
            beta_value = None

    metrics_core = risk_metrics(
        decomposition, diversification_ratio(w, cov), stats, drawdown_stats=dd,
        beta_value=beta_value,
        benchmark_name=benchmark.label if beta_value is not None else None)
    annual = annualise_volatility(decomposition.portfolio_volatility)
    numeric = {
        "Annualised volatility": ("volatility", annual),
        "Diversification ratio": ("diversification", diversification_ratio(w, cov)),
        "Effective holdings": ("effective_holdings", stats.effective_holdings),
        "Maximum drawdown": ("max_drawdown", dd.max_drawdown),
    }
    metrics = []
    for m in metrics_core:
        key, value = numeric.get(m.label, ("beta", beta_value))
        metrics.append(S.Metric(key=key, label=m.label, value=_f(value),
                                display=m.value, sentence=m.sentence, warning=m.warning))

    marginal = {isin: float(annualise_volatility(float(mc)))
                for isin, mc in zip(decomposition.instruments, decomposition.marginal)}
    total_value = sum(values.get(i, 0.0) for i in used)
    div_rows = [S.DivergenceRow(isin=r.isin, name=r.name, weight=r.weight,
                                risk_share=r.risk_share, divergence=r.divergence,
                                marginal=marginal.get(r.isin, 0.0), sentence=r.sentence())
                for r in rows]

    var_rows = [S.VarRow(method=v.method, confidence=v.confidence,
                         horizon_days=v.horizon_days, loss=v.loss,
                         loss_amount=v.loss * total_value if total_value else None,
                         expected_shortfall=v.expected_shortfall,
                         expected_shortfall_amount=(v.expected_shortfall * total_value
                                                    if total_value else None),
                         observations=v.observations)
                for v in var_report(portfolio_returns)]
    worst = []
    for length in (1, 5, 21):
        for p in worst_periods(portfolio_returns, length_days=length, n=3):
            worst.append(S.WorstPeriod(start=p.start.isoformat(), end=p.end.isoformat(),
                                       length_days=p.length_days, ret=p.ret,
                                       amount=p.ret * total_value if total_value else None))
    scenarios = [S.Scenario(key=s.key, label=s.label, description=s.description,
                            portfolio_return=s.portfolio_return,
                            portfolio_amount=(s.portfolio_return * total_value
                                              if total_value else None),
                            per_holding=dict(s.per_holding))
                 for s in stress_scenarios(returns[used], w)]

    window = S.Window(
        requested=alignment.requested_lookback, effective=alignment.effective_lookback,
        first_date=_iso(alignment.first_date), last_date=_iso(alignment.last_date),
        binding_isin=alignment.binding_instrument,
        binding_name=names.get(alignment.binding_instrument or "", alignment.binding_instrument),
        binding_observations=alignment.binding_observations,
        excluded=[{"isin": e.isin, "name": names.get(e.isin, e.isin),
                   "observations": e.observations, "reason": e.reason}
                  for e in alignment.excluded],
        warnings=list(alignment.warnings) + [f"No price history: {m}" for m in missing])

    out = S.Risk(
        available=True, reason=None, window=window,
        headline=div_rows[0] if div_rows else None, divergence=div_rows, metrics=metrics,
        volatility=annual, volatility_multiple=annual / BROAD_EUROPEAN_EQUITY_VOLATILITY,
        diversification_ratio=diversification_ratio(w, cov),
        effective_holdings=stats.effective_holdings, actual_holdings=stats.actual_holdings,
        max_drawdown=dd.max_drawdown, current_drawdown=dd.current_drawdown,
        beta=beta_value, beta_benchmark=benchmark.label if beta_value is not None else None,
        standalone=[S.Standalone(isin=str(isin), name=names.get(str(isin), str(isin)),
                                 volatility=float(vol),
                                 multiple=float(vol) / BROAD_EUROPEAN_EQUITY_VOLATILITY,
                                 weight=w.get(str(isin), 0.0))
                    for isin, vol in standalone.items()],
        correlation=S.Correlation(isins=[str(c) for c in corr.columns],
                                  names=[names.get(str(c), str(c)) for c in corr.columns],
                                  matrix=[[float(x) for x in row] for row in corr.to_numpy()]),
        pairs=[S.Pair(a=p.a, b=p.b, a_name=names.get(p.a, p.a), b_name=names.get(p.b, p.b),
                      correlation=p.correlation, sentence=sentence)
               for p, sentence in zip(pairs, sentences)],
        clusters=[S.Cluster(members=list(c.members),
                            names=[names.get(m, m) for m in c.members],
                            mean_correlation=c.mean_correlation,
                            min_correlation=c.min_correlation,
                            combined_weight=c.combined_weight) for c in clusters],
        threshold=HIGH_CORRELATION_THRESHOLD, var=var_rows, worst_periods=worst,
        scenarios=scenarios, portfolio_returns=_series(portfolio_returns))

    return _RiskState(
        out=out, returns=returns, cov=cov, corr=corr, decomposition=decomposition,
        weights=w, alignment=alignment, portfolio_returns=portfolio_returns,
        concentration=stats, pairs=pairs, clusters=clusters,
        risk_share={r.isin: r.risk_share for r in rows},
        divergence={r.isin: r.divergence for r in rows},
        standalone={str(isin): float(vol) for isin, vol in standalone.items()})


# --------------------------------------------------------------------------
# Performance block
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _PerfState:
    out: S.Performance
    history: ValueHistory | None = None
    twr: pd.Series | None = None


def _empty_performance(reason: str) -> S.Performance:
    empty = S.Series(dates=[], values=[])
    return S.Performance(
        available=False, reason=reason, dates=[], value=[], invested=[], cost_basis=[],
        index=[], benchmark_index=[], drawdown=[], benchmark_drawdown=[], flows=[],
        summary=None, trailing={}, benchmark_trailing={}, monthly=[], yearly=[],
        contributions=[], rolling_volatility=empty, rolling_beta=empty,
        per_holding_value={}, missing=[], warnings=[])


def _performance(transactions: list[Transaction], instruments: dict[str, Instrument],
                 market: MarketSnapshot, benchmark: Benchmark, names: dict[str, str],
                 positions: dict[str, Position], base: str,
                 fx: FxRates | None) -> _PerfState:
    if not transactions:
        return _PerfState(_empty_performance(
            "No transactions yet, so there is no history to measure."))
    quote_ccy = {}
    for isin in {t.isin for t in transactions}:
        inst = instruments.get(isin)
        pos = positions.get(isin)
        if inst is not None and inst.quote_currency:
            quote_ccy[isin] = inst.quote_currency
        elif pos is not None and pos.quote is not None:
            quote_ccy[isin] = pos.quote.price.currency
        else:
            quote_ccy[isin] = base
    try:
        history = value_history_safe(transactions, market.histories, quote_ccy, fx, base)
    except Exception as exc:                     # a named refusal, never a blank page
        return _PerfState(_empty_performance(f"Could not build the value history: {exc}"))
    if history is None or len(history.value) < 2:
        return _PerfState(_empty_performance(
            "Not enough price history to build a value series."))

    twr = time_weighted_return(history.value, history.flows)
    index = return_index(twr)
    dates = index.index
    dd = drawdown_series(index)

    bench_index = None
    bench_daily = None
    bench_hist = market.benchmarks.get(benchmark.symbol)
    if bench_hist is not None and len(bench_hist):
        aligned = bench_hist.reindex(dates, method="ffill").dropna()
        if len(aligned) >= 2:
            bench_daily = simple_returns(aligned)
            bench_index = return_index(bench_daily).reindex(dates)
            bench_index.iloc[0] = 100.0 if pd.isna(bench_index.iloc[0]) else bench_index.iloc[0]

    terminal = float(history.value.iloc[-1])
    summary = performance_summary(twr, benchmark=bench_daily, flows=history.flows,
                                  terminal_value=terminal)
    monthly = monthly_table(twr)
    bench_monthly = monthly_table(bench_daily) if bench_daily is not None else None
    monthly_rows = []
    for year in monthly.index:
        for month in monthly.columns:
            ret = monthly.loc[year, month]
            if pd.isna(ret):
                continue
            b = None
            if bench_monthly is not None and year in bench_monthly.index \
                    and month in bench_monthly.columns:
                b = _f(bench_monthly.loc[year, month])
            monthly_rows.append(S.MonthlyReturn(year=int(year), month=int(month),
                                                ret=_f(ret), benchmark_ret=b))
    yearly = period_returns(twr, "Y")
    bench_yearly = period_returns(bench_daily, "Y") if bench_daily is not None else None
    yearly_rows = []
    for stamp, ret in yearly.items():
        year = pd.Timestamp(stamp).year
        b = None
        if bench_yearly is not None:
            match = [v for s, v in bench_yearly.items() if pd.Timestamp(s).year == year]
            b = _f(match[0]) if match else None
        yearly_rows.append(S.YearlyReturn(year=year, ret=_f(ret), benchmark_ret=b))

    contributions = [S.Contribution(isin=c.isin, name=names.get(c.isin, c.isin),
                                    pnl=c.pnl, contribution=c.contribution)
                     for c in holding_contributions(history)]

    flows = []
    for t in sorted(transactions, key=lambda x: (x.date, x.id)):
        amount = _to_base(-t.cash_flow().amount, t.currency, t.date, fx, base)
        if amount is None:
            continue
        flows.append(S.Flow(date=t.date.isoformat(), amount=amount, type=t.type.value,
                            isin=t.isin, name=names.get(t.isin, t.isin)))

    rolling_vol = rolling_volatility(twr, window=ROLLING_WINDOW)
    rolling_b = (rolling_beta(twr, bench_daily, window=ROLLING_WINDOW)
                 if bench_daily is not None else None)

    out = S.Performance(
        available=True, reason=None, dates=_dates(dates),
        value=_values(history.value, dates), invested=_values(history.invested, dates),
        cost_basis=_values(history.cost_basis, dates), index=_values(index, dates),
        benchmark_index=_values(bench_index, dates), drawdown=_values(dd, dates),
        benchmark_drawdown=_values(drawdown_series(bench_index.dropna()) if bench_index is not None else None, dates),
        flows=flows,
        summary=S.PerformanceSummary(
            total_return=summary.total_return, annualised_return=summary.annualised_return,
            money_weighted=summary.money_weighted, volatility=summary.volatility,
            sharpe=summary.sharpe, sortino=summary.sortino, calmar=summary.calmar,
            max_drawdown=summary.max_drawdown, current_drawdown=summary.current_drawdown,
            best_day=summary.best_day, worst_day=summary.worst_day,
            best_month=summary.best_month, worst_month=summary.worst_month,
            positive_months=summary.positive_months, negative_months=summary.negative_months,
            observations=summary.observations, first_date=_iso(summary.first_date),
            last_date=_iso(summary.last_date),
            benchmark_total_return=summary.benchmark_total_return, beta=summary.beta,
            alpha=summary.alpha, correlation=summary.correlation,
            tracking_error=summary.tracking_error,
            information_ratio=summary.information_ratio),
        trailing={k: _f(v) for k, v in trailing_returns(index).items()},
        benchmark_trailing=({k: _f(v) for k, v in trailing_returns(bench_index.dropna()).items()}
                            if bench_index is not None else {}),
        monthly=monthly_rows, yearly=yearly_rows, contributions=contributions,
        rolling_volatility=_series(rolling_vol.dropna()),
        rolling_beta=_series(rolling_b.dropna()) if rolling_b is not None else S.Series(dates=[], values=[]),
        per_holding_value={str(c): _values(history.per_holding[c], dates)
                           for c in history.per_holding.columns},
        holding_names={str(c): names.get(str(c), str(c)) for c in history.per_holding.columns},
        missing=[names.get(m, m) for m in history.missing],
        warnings=list(history.warnings))
    return _PerfState(out=out, history=history, twr=twr)


def value_history_safe(transactions, histories, quote_ccy, fx, base) -> ValueHistory | None:
    from ..core.performance import value_history
    return value_history(transactions, histories, quote_ccy, rates=fx, base=base)


# --------------------------------------------------------------------------
# Holding detail
# --------------------------------------------------------------------------


def holding_detail(analysis: Analysis, isin: str,
                   amendments: list[Amendment]) -> S.HoldingDetail | None:
    snap = analysis.snapshot
    inst = analysis.instruments.get(isin)
    if inst is None:
        return None
    holding = next((h for h in snap.holdings if h.isin == isin), None) or \
        next((w for w in snap.watchlist if w.isin == isin), None)
    if holding is None:
        return None

    hist = analysis.market.histories.get(isin)
    prices = _series(hist)
    dd = _series(drawdown_series(hist) if hist is not None and len(hist) else None)

    stats: dict[str, float | None] = {}
    if hist is not None and len(hist) > 2:
        daily = simple_returns(hist)
        stats["volatility"] = float(annualise_volatility(float(daily.std(ddof=1))))
        stats["max_drawdown"] = _f(drawdown(hist).max_drawdown)
        stats["current_drawdown"] = _f(drawdown(hist).current_drawdown)
        for k, v in trailing_returns(hist).items():
            stats[f"trailing_{k}"] = _f(v)
        if analysis.portfolio_returns is not None:
            try:
                stats["beta_vs_portfolio"] = float(beta(daily, analysis.portfolio_returns))
            except ValueError:
                stats["beta_vs_portfolio"] = None
            joined = pd.concat([daily.rename("a"), analysis.portfolio_returns.rename("p")],
                               axis=1, sort=True).dropna()
            stats["correlation_vs_portfolio"] = (_f(joined["a"].corr(joined["p"]))
                                                 if len(joined) > 2 else None)
    contribution = next((c for c in snap.performance.contributions if c.isin == isin), None)
    stats["contribution"] = contribution.contribution if contribution else None
    stats["pnl_contribution"] = contribution.pnl if contribution else None

    correlations = []
    if analysis.corr is not None and isin in analysis.corr.columns:
        for other in analysis.corr.columns:
            if other == isin:
                continue
            rho = float(analysis.corr.loc[isin, other])
            correlations.append(S.Pair(
                a=isin, b=str(other), a_name=analysis.names.get(isin, isin),
                b_name=analysis.names.get(str(other), str(other)), correlation=rho,
                sentence=""))
        correlations.sort(key=lambda p: p.correlation, reverse=True)

    voided_by = {a.target_id: a for a in amendments}
    own = [t for t in analysis.transactions if t.isin == isin]
    markers = [S.Marker(date=t.date.isoformat(), type=t.type.value,
                        quantity=float(t.quantity), price=float(t.price_per_unit),
                        currency=t.currency) for t in own if t.type.value in ("BUY", "SELL")]
    return S.HoldingDetail(
        holding=holding, instrument=instrument_out(inst, analysis.transactions),
        prices=prices, drawdown=dd, markers=markers,
        transactions=[transaction_out(t, analysis.names, voided_by)
                      for t in sorted(own, key=lambda x: (x.date, x.id), reverse=True)],
        stats=stats, correlations=correlations)
