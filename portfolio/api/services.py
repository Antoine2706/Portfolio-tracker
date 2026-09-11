"""The application service: one object that owns the store, the market data
and the cached analysis, shared by every request.

Caching model
-------------
Three layers, each invalidated by exactly one thing:

    PriceCache (sqlite)     survives restarts; quotes expire on a TTL, history
                            is fetched incrementally. Nothing here invalidates
                            it -- `refresh()` only bypasses the quote TTL.
    MarketSnapshot          prices for the current instrument set, held in
                            memory until the instrument set changes or a
                            refresh is requested.
    Analysis                every derived number for one (mode, benchmark,
                            lookback), held until the ledger, the instruments
                            or the market snapshot change.

The consequence is the performance rule in docs/ARCHITECTURE.md: a page load
after the first is a dictionary lookup, and a ledger write costs one rebuild
from cached prices rather than a single network call.

Thread safety: uvicorn runs synchronous handlers in a thread pool, so the
service is guarded by one re-entrant lock. Builds are short (tens of
milliseconds on cached prices), so a single lock is simpler and safer than
finer-grained locking around a dozen mutable caches.
"""

from __future__ import annotations

import datetime as dt
import threading

from ..core.models import Instrument
from ..data.benchmarks import BENCHMARKS, DEFAULT_BENCHMARK, benchmark_by_symbol
from ..data.cache import PriceCache
from ..data.fx import currencies_needed
from ..data.market import MarketData, MarketSnapshot
from ..data.provider import IdentityProvider, MarketDataProvider
from ..data.resolve import Resolution, resolve_isin
from ..data.store import DataMode, DataStore
from . import builder
from .config import Config

__all__ = ["PortfolioService"]


def _make_providers(name: str) -> tuple[MarketDataProvider, IdentityProvider]:
    if name == "fixture":
        from ..data.providers.fixture import FixtureIdentityProvider, FixtureProvider
        return FixtureProvider(), FixtureIdentityProvider()
    from ..data.providers.openfigi import OpenFIGIProvider
    from ..data.providers.yahoo import YahooProvider
    return YahooProvider(), OpenFIGIProvider()


class PortfolioService:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.mode: DataMode = config.mode
        self.provider, self.identity = _make_providers(config.provider)

        cache_dir = config.data_root / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = PriceCache(cache_dir / f"prices-{self.provider.name}.sqlite")
        self.market_data = MarketData(
            self.provider, self.cache, base=config.base_currency,
            max_workers=config.max_workers, quote_ttl=config.quote_ttl,
            history_ttl=config.history_ttl)

        self._lock = threading.RLock()
        self._market: MarketSnapshot | None = None
        self._market_key: tuple | None = None
        self._analyses: dict[tuple, builder.Analysis] = {}
        self._resolutions: dict[str, tuple[dt.datetime, Resolution]] = {}
        self.last_refresh: dt.datetime | None = None

    # -- store -------------------------------------------------------------

    def store(self) -> DataStore:
        return DataStore.open(self.mode, root=self.config.data_root)

    def set_mode(self, mode: DataMode) -> None:
        with self._lock:
            if mode is not self.mode:
                self.mode = mode
                self.invalidate()

    def invalidate(self) -> None:
        """The ledger or the universe changed: every derived number is stale.

        The market snapshot is dropped too, because the instrument set may
        have changed. Rebuilding it is cheap: every symbol already on disk is
        a cache hit.
        """
        with self._lock:
            self._analyses.clear()
            self._market = None
            self._market_key = None

    # -- market ------------------------------------------------------------

    def _universe(self, store: DataStore) -> tuple[dict[str, Instrument], list]:
        instruments = store.load_instruments()
        transactions = store.load_transactions()
        held = {t.isin for t in transactions}
        # Deactivated instruments are hidden from the watchlist but a
        # deactivated instrument with a position is still a position: hiding
        # it would make the total wrong without saying so.
        wanted = {isin: inst for isin, inst in instruments.items()
                  if inst.active or isin in held}
        return wanted, transactions

    def market(self, force: bool = False) -> MarketSnapshot:
        store = self.store()
        instruments, transactions = self._universe(store)
        currencies = sorted(currencies_needed(instruments, transactions,
                                              base=self.config.base_currency))
        symbols = tuple(sorted(s for inst in instruments.values()
                               for s in [self.market_data.symbol_for(inst)] if s))
        key = (self.mode.value, symbols, tuple(currencies))
        with self._lock:
            if not force and self._market is not None and self._market_key == key:
                return self._market
            snapshot = self.market_data.load(
                instruments, benchmark_symbols=[b.symbol for b in BENCHMARKS],
                currencies=currencies, force=force)
            self._market = snapshot
            self._market_key = key
            self._analyses.clear()
            if force:
                self.last_refresh = snapshot.fetched_at
            return snapshot

    def refresh(self) -> MarketSnapshot:
        return self.market(force=True)

    # -- analysis ----------------------------------------------------------

    def analysis(self, benchmark: str | None = None,
                 lookback: int | None = None) -> builder.Analysis:
        bench = benchmark_by_symbol(benchmark or "") or DEFAULT_BENCHMARK
        window = lookback or self.config.lookback
        if window < 2:
            raise ValueError("lookback must be at least 2 trading days")
        with self._lock:
            market = self.market()
            key = (self.mode.value, bench.symbol, window, market.fetched_at)
            found = self._analyses.get(key)
            if found is not None:
                return found
            store = self.store()
            instruments, transactions = self._universe(store)
            analysis = builder.build(
                instruments=instruments,
                all_instruments=store.load_instruments(),
                transactions=transactions,
                amendments=store.load_amendments(),
                market=market, benchmark=bench, lookback=window,
                config=self.config, mode=self.mode,
                symbol_for=self.market_data.symbol_for)
            self._analyses[key] = analysis
            return analysis

    # -- resolution --------------------------------------------------------

    def resolve(self, isin: str, lookback: int = 252) -> Resolution:
        isin = isin.strip().upper()
        now = dt.datetime.now(dt.timezone.utc)
        with self._lock:
            cached = self._resolutions.get(isin)
            if cached and now - cached[0] < self.config.resolution_ttl:
                return cached[1]
        resolution = resolve_isin(isin, self.identity, self.provider,
                                  known=None, lookback=lookback)
        with self._lock:
            self._resolutions[isin] = (now, resolution)
        return resolution
