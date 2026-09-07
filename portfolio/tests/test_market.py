"""The market loader: cache hits and misses, incremental history, and every
degraded path. All against the fixture provider -- a provider that can be
told to fail per symbol is what makes the degraded paths testable at all."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from portfolio.core.models import AssetClass, Instrument
from portfolio.core.positions import PriceQuote
from portfolio.data.cache import PriceCache
from portfolio.data.fx import DailyFxTable
from portfolio.data.market import (INCREMENTAL_OVERLAP, Failure, MarketData,
                                   MarketSnapshot)
from portfolio.data.provider import MarketDataProvider
from portfolio.data.providers.fixture import FIXTURE_END, FixtureProvider

EUDF, DFNC, WEAT, WDEP, NOSYM = ("IE0002Y8CX98", "IE000IAXNM41", "JE00BN7KB664",
                                 "IE000I7E6HL0", "LU1681048630")


class Counting(MarketDataProvider):
    """A provider that records every call so the tests can prove a cache hit."""
    name = "fixture"
    documented_delay_minutes = 15

    def __init__(self, inner: MarketDataProvider | None = None) -> None:
        self.inner = inner or FixtureProvider()
        self.history_calls: list[tuple[str, dt.date | None]] = []
        self.quote_calls: list[str] = []

    def probe(self, symbol, lookback_days=252):
        return self.inner.probe(symbol, lookback_days)

    def history(self, symbol, start=None):
        self.history_calls.append((symbol, start))
        return self.inner.history(symbol, start)

    def quote(self, symbol):
        self.quote_calls.append(symbol)
        return self.inner.quote(symbol)


def instrument(isin, name, symbols, quote_ccy="EUR", active=True) -> Instrument:
    return Instrument(isin, name, AssetClass.ETF, "EUR", quote_currency=quote_ccy,
                      provider_symbols=symbols, active=active)


@pytest.fixture
def instruments() -> dict[str, Instrument]:
    return {
        EUDF: instrument(EUDF, "WisdomTree Europe Defence", {"yfinance": "EUDF.DE"}),
        DFNC: instrument(DFNC, "iShares Europe Defence", {"yfinance": "DFNC.DE"}),
        WEAT: instrument(WEAT, "WisdomTree Wheat", {"yfinance": "WEAT.MI"}),
        # An LSE line: the fixture quotes .L symbols in pence.
        WDEP: instrument(WDEP, "Pence line", {"yfinance": "WDEP.L"}, quote_ccy="GBp"),
        NOSYM: instrument(NOSYM, "Never resolved", {}),
    }


@pytest.fixture
def cache(tmp_path) -> PriceCache:
    c = PriceCache(tmp_path / "prices.sqlite")
    yield c
    c.close()


@pytest.fixture
def provider() -> Counting:
    return Counting()


class TestFirstLoad:
    def test_everything_is_keyed_by_isin(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(instruments)
        assert isinstance(snap, MarketSnapshot)
        assert set(snap.quotes) == {EUDF, DFNC, WEAT, WDEP}
        assert set(snap.histories) == {EUDF, DFNC, WEAT, WDEP}
        assert snap.provider == "fixture"

    def test_histories_are_float_series_named_by_isin(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(instruments)
        s = snap.histories[EUDF]
        assert s.name == EUDF and s.dtype == float
        assert isinstance(s.index, pd.DatetimeIndex) and s.index.tz is None
        assert s.index.is_monotonic_increasing
        assert len(s) == 377 and s.index[-1].date() == FIXTURE_END

    def test_quotes_carry_provenance(self, instruments, cache, provider):
        q = MarketData(provider, cache).load(instruments).quotes[EUDF]
        assert isinstance(q, PriceQuote)
        assert q.price.currency == "EUR" and q.source == "fixture"
        assert q.delay_minutes == 15 and q.is_stale is False
        assert isinstance(q.as_of, dt.datetime)

    def test_pence_arrive_as_pounds(self, instruments, cache, provider):
        """The fixture says GBp for .L symbols. Money rescales by 1/100; a
        quote left in pence would overstate the position 100x."""
        raw = FixtureProvider().quote("WDEP.L")
        assert raw.currency == "GBp"
        q = MarketData(provider, cache).load(instruments).quotes[WDEP]
        assert q.price.currency == "GBP"
        assert q.price.amount == raw.price * Decimal("0.01")

    def test_cold_cache_counts_every_lookup_as_a_miss(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(instruments)
        assert snap.cache_hits == 0 and snap.cache_misses == 8   # 4 symbols x (history + quote)
        assert len(provider.history_calls) == 4 and len(provider.quote_calls) == 4
        assert all(start is None for _, start in provider.history_calls)

    def test_instrument_without_a_symbol_is_a_failure_not_a_crash(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(instruments)
        failure = next(f for f in snap.failures if f.key == NOSYM)
        assert failure.name == "Never resolved"
        assert "no fixture symbol stored" in failure.message
        assert NOSYM not in snap.quotes and NOSYM not in snap.histories

    def test_timing_and_timestamp(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(instruments)
        assert snap.elapsed_seconds >= 0
        assert snap.fetched_at.tzinfo is not None
        assert isinstance(snap.fx, DailyFxTable)

    def test_failure_messages_name_the_instrument(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(instruments)
        assert any(m.startswith("Never resolved: ") for m in snap.failure_messages())


class TestCacheBehaviour:
    def test_second_load_makes_no_provider_calls(self, instruments, cache, provider):
        md = MarketData(provider, cache)
        md.load(instruments)
        provider.history_calls.clear()
        provider.quote_calls.clear()
        snap = md.load(instruments)
        assert provider.history_calls == [] and provider.quote_calls == []
        assert snap.cache_hits == 8 and snap.cache_misses == 0
        assert len(snap.histories[EUDF]) == 377

    def test_force_refetches_quotes_but_history_only_incrementally(self, instruments, cache, provider):
        md = MarketData(provider, cache)
        md.load(instruments)
        provider.history_calls.clear()
        provider.quote_calls.clear()
        snap = md.load(instruments, force=True)
        assert sorted(provider.quote_calls) == ["DFNC.DE", "EUDF.DE", "WDEP.L", "WEAT.MI"]
        assert len(provider.history_calls) == 4
        for _, start in provider.history_calls:
            assert start == FIXTURE_END - INCREMENTAL_OVERLAP
        assert len(snap.histories[EUDF]) == 377, "the merge must not duplicate the overlap"
        assert snap.cache_misses == 8

    def test_expired_history_ttl_fetches_incrementally_and_keeps_the_quote(self, instruments, cache, provider):
        MarketData(provider, cache).load(instruments)
        provider.history_calls.clear()
        provider.quote_calls.clear()
        snap = MarketData(provider, cache, history_ttl=dt.timedelta(0)).load(instruments)
        assert provider.quote_calls == [], "the quote TTL has not expired"
        assert len(provider.history_calls) == 4
        assert all(start == FIXTURE_END - INCREMENTAL_OVERLAP for _, start in provider.history_calls)
        assert snap.cache_hits == 4 and snap.cache_misses == 4

    def test_incremental_rows_are_written_back(self, instruments, cache, provider):
        md = MarketData(provider, cache)
        md.load(instruments)
        # Simulate a revised close in the overlap window.
        class Revised(Counting):
            def history(self, symbol, start=None):
                s = super().history(symbol, start)
                return s + 1.0 if start is not None else s
        MarketData(Revised(), cache).load(instruments, force=True)
        cached, _ = cache.get_history("EUDF.DE")
        original = FixtureProvider().history("EUDF.DE")
        assert cached.iloc[-1] == pytest.approx(original.iloc[-1] + 1.0)
        assert cached.iloc[0] == pytest.approx(original.iloc[0])

    def test_without_a_cache_every_load_hits_the_provider(self, instruments, provider):
        md = MarketData(provider, None)
        first = md.load(instruments)
        second = md.load(instruments)
        assert len(provider.history_calls) == 8 and len(provider.quote_calls) == 8
        assert first.cache_hits == 0 and second.cache_hits == 0
        assert second.cache_misses == 8
        assert len(second.histories[EUDF]) == 377


class TestDegradedPaths:
    def test_unreachable_symbol_with_a_warm_cache_serves_history_and_a_stale_quote(
            self, instruments, cache):
        MarketData(FixtureProvider(), cache).load(instruments)
        snap = MarketData(FixtureProvider(fail={"WEAT.MI"}), cache).load(instruments, force=True)

        assert len(snap.histories[WEAT]) == 503, "the cached series is served"
        q = snap.quotes[WEAT]
        assert q.is_stale is True and q.delay_minutes is None and q.source == "fixture"
        assert q.as_of == FIXTURE_END
        assert q.price.amount == Decimal(str(round(float(snap.histories[WEAT].iloc[-1]), 6)))

        mine = [f for f in snap.failures if f.key == WEAT]
        assert len(mine) == 2 and all(f.name == "WisdomTree Wheat" for f in mine)
        assert any("serving cached history from 2026-09-04" in f.message for f in mine)
        assert any("last close" in f.message and "stale" in f.message for f in mine)
        # The other symbols are untouched by one symbol's failure.
        assert not snap.quotes[EUDF].is_stale

    def test_unreachable_symbol_with_a_cold_cache_is_omitted_and_reported(self, instruments, cache):
        snap = MarketData(FixtureProvider(fail={"WEAT.MI"}), cache).load(instruments)
        assert WEAT not in snap.histories and WEAT not in snap.quotes
        mine = [f for f in snap.failures if f.key == WEAT]
        assert mine and all("WEAT.MI" in f.message for f in mine)
        assert EUDF in snap.quotes

    def test_no_python_internals_in_failure_text(self, instruments, cache):
        class Exploding(FixtureProvider):
            def history(self, symbol, start=None):
                raise TypeError("argument of type 'NoneType' is not iterable")
            def quote(self, symbol):
                raise RuntimeError("")
        snap = MarketData(Exploding(), cache).load(instruments)
        mine = [f for f in snap.failures if f.key == EUDF]
        assert len(mine) == 2
        assert all("Traceback" not in f.message for f in mine)
        history, quote = mine
        assert history.message.startswith("could not load history for EUDF.DE: ")
        assert "NoneType" in history.message, "the text is kept; the type is what is diagnosable"
        assert quote.message.endswith("RuntimeError with no message")

    def test_a_provider_that_raises_on_construction_of_its_task_is_still_a_failure(
            self, instruments, cache, monkeypatch):
        md = MarketData(FixtureProvider(), cache)
        def boom(*args, **kwargs):
            raise RuntimeError("pool task exploded")
        monkeypatch.setattr(md, "_load_symbol", boom)
        snap = md.load(instruments)
        assert snap.quotes == {} and snap.histories == {}
        assert any("exploded" in f.message for f in snap.failures)


class TestFxAndBenchmarks:
    def test_fx_pairs_land_in_the_table(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(instruments, currencies=("USD", "GBp", "EUR"))
        assert snap.fx.pairs() == [("GBP", "EUR"), ("USD", "EUR")]
        assert snap.fx.rate("USD", "EUR", FIXTURE_END) == Decimal("0.92")
        assert snap.fx.rate("GBp", "EUR", FIXTURE_END) == Decimal("1.17")
        assert sorted(s for s, _ in provider.history_calls if s.endswith("=X")) == \
            ["GBPEUR=X", "USDEUR=X"]
        assert "USDEUR=X" not in provider.quote_calls, "FX pairs need history only"

    def test_fx_history_is_cached_like_any_symbol(self, instruments, cache, provider):
        md = MarketData(provider, cache)
        md.load(instruments, currencies=("USD",))
        provider.history_calls.clear()
        snap = md.load(instruments, currencies=("USD",))
        assert provider.history_calls == []
        assert snap.fx.observations("USD", "EUR") == 520

    def test_unknown_currency_is_a_failure(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(instruments, currencies=("XXX",))
        f = next(f for f in snap.failures if f.key == "XXX")
        assert "no FX pair" in f.message and "XXX" in f.name
        assert snap.fx.pairs() == []

    def test_fx_fetch_failure_is_reported_under_the_currency(self, instruments, cache):
        snap = MarketData(FixtureProvider(fail={"USDEUR=X"}), cache).load(
            instruments, currencies=("USD",))
        f = next(f for f in snap.failures if f.key == "USD")
        assert f.name == "USD->EUR" and "USDEUR=X" in f.message
        assert snap.fx.pairs() == []

    def test_benchmarks_load_by_symbol(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(instruments, benchmark_symbols=("MEUD.PA",))
        s = snap.benchmarks["MEUD.PA"]
        assert s.name == "MEUD.PA" and len(s) == 511
        assert "MEUD.PA" not in provider.quote_calls

    def test_benchmark_failure_carries_its_label(self, instruments, cache):
        snap = MarketData(FixtureProvider(fail={"MEUD.PA"}), cache).load(
            instruments, benchmark_symbols=("MEUD.PA",))
        f = next(f for f in snap.failures if f.key == "MEUD.PA")
        assert "STOXX Europe 600" in f.name
        assert snap.benchmarks == {}

    def test_a_symbol_shared_by_holding_and_benchmark_is_fetched_once(self, cache, provider):
        held = {EUDF: instrument(EUDF, "Held benchmark", {"yfinance": "MEUD.PA"})}
        snap = MarketData(provider, cache).load(held, benchmark_symbols=("MEUD.PA",))
        assert [s for s, _ in provider.history_calls] == ["MEUD.PA"]
        assert EUDF in snap.histories and "MEUD.PA" in snap.benchmarks
        assert snap.histories[EUDF].name == EUDF


class TestSymbolResolution:
    def test_explicit_provider_symbol_beats_the_fallback(self, cache, provider):
        inst = instrument(EUDF, "x", {"fixture": "ASWC.DE", "yfinance": "EUDF.DE"})
        MarketData(provider, cache).load({EUDF: inst})
        assert [s for s, _ in provider.history_calls] == ["ASWC.DE"]

    def test_fixture_falls_back_to_the_yahoo_symbol(self, cache, provider):
        """The seed universe stores Yahoo symbols only; the demo must still run."""
        md = MarketData(provider, cache)
        assert md.symbol_for(instrument(EUDF, "x", {"yfinance": "EUDF.DE"})) == "EUDF.DE"
        assert md.symbol_for(instrument(EUDF, "x", {"eodhd": "EUDF.XETRA"})) is None

    def test_a_real_provider_never_borrows_another_providers_symbols(self, cache):
        class Yahooish(Counting):
            name = "yfinance"
        md = MarketData(Yahooish(), cache)
        assert md.symbol_for(instrument(EUDF, "x", {"fixture": "EUDF.DE"})) is None

    def test_inactive_instruments_are_loaded_when_passed(self, cache, provider):
        inst = instrument(EUDF, "retired", {"yfinance": "EUDF.DE"}, active=False)
        snap = MarketData(provider, cache).load({EUDF: inst})
        assert EUDF in snap.quotes

    def test_accepts_a_list_of_instruments(self, instruments, cache, provider):
        snap = MarketData(provider, cache).load(list(instruments.values()))
        assert EUDF in snap.quotes

    def test_empty_universe_is_an_empty_snapshot(self, cache, provider):
        snap = MarketData(provider, cache).load({})
        assert snap.quotes == {} and snap.failures == []
        assert provider.history_calls == []


class TestFailureShape:
    def test_failure_is_a_plain_dataclass(self):
        f = Failure("IE0002Y8CX98", "name", "message")
        assert (f.key, f.name, f.message) == ("IE0002Y8CX98", "name", "message")
