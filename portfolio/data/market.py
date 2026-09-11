"""Everything the application fetches, in one call, through the cache.

This is the only module that talks to a price provider on the request path.
`api/services.py` calls `MarketData.load` once, gets a `MarketSnapshot`, and
every page renders from that. Three rules follow from the performance section
of the architecture document and are enforced here rather than remembered:

1. Symbols load concurrently. A thread pool (six workers by default, matching
   what Yahoo tolerates before it rate-limits) runs one task per symbol; the
   task fetches that symbol's history and then its quote, so one symbol's two
   calls are sequential while symbols overlap. Results are assembled on the
   calling thread, so nothing outside the pool needs a lock except the cache.

2. History is incremental. A cached series is refreshed from a week before
   its last date -- a week rather than a day because providers revise the
   last few closes (late prints, adjustment factors) and the overlap lets the
   fresh rows overwrite them. A restart never refetches two years of daily
   closes that are already on disk.

3. A single symbol never raises. Every failure becomes a `Failure` naming
   the instrument and saying what was served instead: the cached history, or
   the last close flagged as stale. A book that is mostly EUR must not go
   blank because one ETC's venue is unreachable, and the alerts view turns
   these into rows the user can act on.

`force=True` is the refresh button: it ignores the quote TTL and re-checks
history, still incrementally. Nothing here ever discards a cached series.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Iterable

import pandas as pd

from ..core.models import Instrument
from ..core.money import BASE_CURRENCY, Money, normalise_currency
from ..core.positions import PriceQuote
from .benchmarks import benchmark_by_symbol
from .cache import PriceCache
from .fx import DailyFxTable, pair_symbol
from .provider import MarketDataProvider, Quote

__all__ = ["Failure", "MarketSnapshot", "MarketData", "DEFAULT_QUOTE_TTL",
           "DEFAULT_HISTORY_TTL", "INCREMENTAL_OVERLAP"]

DEFAULT_QUOTE_TTL = dt.timedelta(minutes=15)
DEFAULT_HISTORY_TTL = dt.timedelta(hours=6)
# How far before the last cached date an incremental fetch starts.
INCREMENTAL_OVERLAP = dt.timedelta(days=7)

# Where to look for a symbol when the provider has none stored under its own
# name. The fixture provider answers for any symbol and its profile table is
# keyed on Yahoo symbols, so a book resolved against Yahoo runs as a demo
# without re-resolving every instrument. Listed here rather than inferred so
# that a real provider never silently borrows another provider's symbols.
_SYMBOL_FALLBACKS: dict[str, tuple[str, ...]] = {"fixture": ("yfinance",)}


@dataclasses.dataclass
class Failure:
    """One thing that could not be fetched, and what was done about it."""
    key: str            # ISIN, currency code, or benchmark symbol
    name: str
    message: str


@dataclasses.dataclass
class MarketSnapshot:
    quotes: dict[str, PriceQuote]              # by ISIN
    histories: dict[str, pd.Series]            # by ISIN, adjusted close in quote ccy
    benchmarks: dict[str, pd.Series]           # by symbol
    fx: DailyFxTable
    failures: list[Failure]
    fetched_at: dt.datetime
    elapsed_seconds: float
    provider: str
    cache_hits: int
    cache_misses: int

    def failure_messages(self) -> list[str]:
        """The sentences the alerts view needs, one per failure."""
        return [f"{f.name}: {f.message}" for f in self.failures]


@dataclasses.dataclass
class _SymbolResult:
    """What one pool task produced for one symbol."""
    symbol: str
    history: pd.Series | None = None
    quote: Quote | None = None
    failures: list[str] = dataclasses.field(default_factory=list)
    hits: int = 0
    misses: int = 0


def _clean_history(series: pd.Series, symbol: str) -> pd.Series:
    """Provider output into the one shape the rest of the code expects.

    Float values, a naive date-only DatetimeIndex, sorted, no duplicate
    dates, no NaN rows. Providers differ on every one of those and the
    covariance code downstream silently misaligns on any of them.
    """
    clean = pd.Series(series.to_numpy(dtype=float), index=pd.DatetimeIndex(series.index),
                      name=symbol)
    if clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)
    clean.index = clean.index.normalize()
    clean = clean.dropna()
    clean = clean[~clean.index.duplicated(keep="last")]
    return clean.sort_index()


def _merge(cached: pd.Series, fresh: pd.Series) -> pd.Series:
    """Union of dates; a fresh row wins over a cached one for the same date."""
    merged = pd.concat([cached, fresh])
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def _explain(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or f"{type(exc).__name__} with no message"


class MarketData:
    """Loads quotes, histories, FX and benchmarks for a universe, cached."""

    def __init__(self, provider: MarketDataProvider, cache: PriceCache | None,
                 base: str = BASE_CURRENCY, max_workers: int = 6,
                 quote_ttl: dt.timedelta = DEFAULT_QUOTE_TTL,
                 history_ttl: dt.timedelta = DEFAULT_HISTORY_TTL) -> None:
        self.provider = provider
        self.cache = cache
        self.base, _ = normalise_currency(base)
        self.max_workers = max(1, int(max_workers))
        self.quote_ttl = quote_ttl
        self.history_ttl = history_ttl

    # -- symbols -----------------------------------------------------------

    def symbol_for(self, inst: Instrument) -> str | None:
        """The fetch handle for this provider, or None if none is stored."""
        keys = (self.provider.name,) + _SYMBOL_FALLBACKS.get(self.provider.name, ())
        for key in keys:
            symbol = (inst.provider_symbols.get(key) or "").strip()
            if symbol:
                return symbol
        return None

    # -- the load ----------------------------------------------------------

    def load(self, instruments: dict[str, Instrument] | Iterable[Instrument],
             benchmark_symbols: Iterable[str] = (),
             currencies: Iterable[str] = (),
             force: bool = False) -> MarketSnapshot:
        started = time.perf_counter()
        fetched_at = dt.datetime.now(dt.timezone.utc)
        universe = (instruments if isinstance(instruments, dict)
                    else {i.isin: i for i in instruments})
        failures: list[Failure] = []

        # One task per distinct symbol. An instrument and a benchmark that
        # share a symbol (MEUD.PA held and used as the benchmark) fetch once.
        by_symbol: dict[str, list[Instrument]] = {}
        for inst in universe.values():
            symbol = self.symbol_for(inst)
            if symbol is None:
                failures.append(Failure(
                    inst.isin, inst.name,
                    f"no {self.provider.name} symbol stored; resolve the "
                    f"instrument to choose a listing"))
                continue
            by_symbol.setdefault(symbol, []).append(inst)

        fx_symbols: dict[str, str] = {}
        for raw in currencies:
            code, _ = normalise_currency(raw)
            if code == self.base or code in fx_symbols:
                continue
            symbol = pair_symbol(code, self.base)
            if symbol is None:
                failures.append(Failure(
                    code, f"{code}->{self.base}",
                    f"no FX pair is known for {code} into {self.base}; holdings "
                    f"quoted in {code} cannot be valued"))
                continue
            fx_symbols[code] = symbol

        bench_symbols = [s for s in dict.fromkeys(benchmark_symbols) if s]

        tasks: dict[str, tuple[bool, str]] = {}      # symbol -> (want_quote, ccy hint)
        for symbol, insts in by_symbol.items():
            hint = next((i.quote_currency for i in insts if i.quote_currency), self.base)
            tasks[symbol] = (True, hint)
        for symbol in list(fx_symbols.values()) + bench_symbols:
            tasks.setdefault(symbol, (False, self.base))

        results = self._run(tasks, force)

        # -- assemble on the calling thread ---------------------------------
        quotes: dict[str, PriceQuote] = {}
        histories: dict[str, pd.Series] = {}
        for symbol, insts in by_symbol.items():
            result = results[symbol]
            for inst in insts:
                for message in result.failures:
                    failures.append(Failure(inst.isin, inst.name, message))
                if result.history is not None:
                    histories[inst.isin] = result.history.rename(inst.isin)
                if result.quote is not None:
                    try:
                        quotes[inst.isin] = _to_price_quote(result.quote, inst, self.base)
                    except ValueError as exc:
                        failures.append(Failure(
                            inst.isin, inst.name,
                            f"the quote for {symbol} could not be used: {exc}"))

        fx = DailyFxTable()
        for code, symbol in fx_symbols.items():
            result = results[symbol]
            label = f"{code}->{self.base}"
            for message in result.failures:
                failures.append(Failure(code, label, message))
            if result.history is not None:
                fx.add_series(code, self.base, result.history)

        benchmarks: dict[str, pd.Series] = {}
        for symbol in bench_symbols:
            result = results[symbol]
            known = benchmark_by_symbol(symbol)
            label = known.label if known else symbol
            for message in result.failures:
                failures.append(Failure(symbol, label, message))
            if result.history is not None:
                benchmarks[symbol] = result.history.rename(symbol)

        return MarketSnapshot(
            quotes=quotes, histories=histories, benchmarks=benchmarks, fx=fx,
            failures=failures, fetched_at=fetched_at,
            elapsed_seconds=time.perf_counter() - started,
            provider=self.provider.name,
            cache_hits=sum(r.hits for r in results.values()),
            cache_misses=sum(r.misses for r in results.values()),
        )

    # -- the pool ----------------------------------------------------------

    def _run(self, tasks: dict[str, tuple[bool, str]], force: bool
             ) -> dict[str, _SymbolResult]:
        if not tasks:
            return {}
        results: dict[str, _SymbolResult] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as pool:
            futures = {symbol: pool.submit(self._load_symbol, symbol, want_quote,
                                           currency, force)
                       for symbol, (want_quote, currency) in tasks.items()}
            for symbol, future in futures.items():
                try:
                    results[symbol] = future.result()
                except Exception as exc:       # the task guards itself; belt and braces
                    results[symbol] = _SymbolResult(
                        symbol, failures=[f"loading {symbol} failed unexpectedly: "
                                          f"{_explain(exc)}"])
        return results

    def _load_symbol(self, symbol: str, want_quote: bool, currency: str,
                     force: bool) -> _SymbolResult:
        """History then quote for one symbol. Never raises."""
        result = _SymbolResult(symbol)
        result.history = self._history(symbol, force, result)
        if want_quote:
            result.quote = self._quote(symbol, currency, force, result)
        return result

    def _history(self, symbol: str, force: bool, result: _SymbolResult
                 ) -> pd.Series | None:
        cached: pd.Series | None = None
        if self.cache is not None:
            hit = self.cache.get_history(symbol)
            if hit is not None:
                cached, fetched_at = hit
                fresh_enough = dt.datetime.now(dt.timezone.utc) - fetched_at <= self.history_ttl
                if fresh_enough and not force:
                    result.hits += 1
                    return cached
        result.misses += 1

        try:
            if cached is not None:
                start = cached.index[-1].date() - INCREMENTAL_OVERLAP
                fresh = _clean_history(self.provider.history(symbol, start=start), symbol)
                if self.cache is not None:
                    self.cache.put_history(symbol, fresh)
                return _merge(cached, fresh)
            fresh = _clean_history(self.provider.history(symbol), symbol)
            if fresh.empty:
                raise ValueError(f"no history returned for {symbol}")
            if self.cache is not None:
                self.cache.put_history(symbol, fresh)
            return fresh
        except Exception as exc:
            if cached is not None:
                result.failures.append(
                    f"could not refresh history for {symbol}; serving cached history "
                    f"from {cached.index[-1].date().isoformat()}: {_explain(exc)}")
                return cached
            result.failures.append(
                f"could not load history for {symbol}: {_explain(exc)}")
            return None

    def _quote(self, symbol: str, currency: str, force: bool, result: _SymbolResult
               ) -> Quote | None:
        if self.cache is not None and not force:
            cached = self.cache.get_quote(symbol, self.quote_ttl)
            if cached is not None:
                result.hits += 1
                return cached
        result.misses += 1
        try:
            quote = self.provider.quote(symbol)
            if self.cache is not None:
                self.cache.put_quote(symbol, quote)
            return quote
        except Exception as exc:
            history = result.history
            if history is None or history.empty:
                result.failures.append(
                    f"no price available for {symbol}: {_explain(exc)}")
                return None
            last_date = history.index[-1].date()
            result.failures.append(
                f"no live quote for {symbol}; showing the last close from "
                f"{last_date.isoformat()} instead, marked as stale: {_explain(exc)}")
            # The fallback is built from history and says so: is_stale is what
            # makes the Holdings row carry "last known close, not a live quote".
            return Quote(symbol=symbol,
                         price=Decimal(str(round(float(history.iloc[-1]), 6))),
                         currency=currency or self.base, as_of=last_date,
                         source=self.provider.name, delay_minutes=None, is_stale=True)


def _to_price_quote(quote: Quote, inst: Instrument, base: str) -> PriceQuote:
    """Provider `Quote` into the `core` shape. Pence become pounds here."""
    currency = quote.currency or inst.quote_currency or base
    return PriceQuote(price=Money(quote.price, currency), as_of=quote.as_of,
                      source=quote.source, delay_minutes=quote.delay_minutes,
                      is_stale=quote.is_stale)
