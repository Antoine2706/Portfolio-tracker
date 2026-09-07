"""The SQLite price cache: round trips, merges, TTL edges, and shared use.

A cache that returns a series slightly different from what went in -- an
index that became object dtype, a float that lost a digit, a datetime that
came back as a date -- corrupts every number downstream while every test of
the maths still passes. So the round-trip tests are exact, not approximate.
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from portfolio.data import cache as cache_module
from portfolio.data.cache import CacheStats, PriceCache
from portfolio.data.provider import Quote
from portfolio.data.providers.fixture import FixtureProvider

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 9, 6, 12, 0, tzinfo=UTC)


@pytest.fixture
def cache(tmp_path) -> PriceCache:
    c = PriceCache(tmp_path / "prices.sqlite")
    yield c
    c.close()


@pytest.fixture
def series() -> pd.Series:
    return FixtureProvider().history("EUDF.DE")


def quote(price="12.345678", as_of=None, delay=15, stale=False) -> Quote:
    return Quote(symbol="EUDF.DE", price=Decimal(price), currency="EUR",
                 as_of=as_of or dt.datetime(2026, 9, 4, 17, 30, tzinfo=UTC),
                 source="fixture", delay_minutes=delay, is_stale=stale)


class TestHistory:
    def test_round_trip_is_exact(self, cache, series):
        cache.put_history("EUDF.DE", series)
        got = cache.get_history("EUDF.DE")
        assert got is not None
        back, fetched_at = got
        assert isinstance(back.index, pd.DatetimeIndex)
        assert back.index.tz is None
        assert back.index.is_monotonic_increasing
        assert back.index.equals(pd.DatetimeIndex(series.index))
        assert back.dtype == float and back.name == "EUDF.DE"
        assert np.abs(back.to_numpy() - series.to_numpy()).max() < 1e-9

    def test_fetched_at_is_a_tz_aware_utc_datetime(self, cache, series):
        cache.put_history("EUDF.DE", series)
        _, fetched_at = cache.get_history("EUDF.DE")
        assert fetched_at.tzinfo is not None
        assert fetched_at.utcoffset() == dt.timedelta(0)
        assert abs(dt.datetime.now(UTC) - fetched_at) < dt.timedelta(minutes=1)

    def test_unknown_symbol_is_none_not_an_error(self, cache):
        assert cache.get_history("NOPE.DE") is None

    def test_second_put_merges_and_overwrites_the_overlap(self, cache, series):
        """An incremental fetch hands over only the tail; earlier rows must
        survive and revised closes in the overlap must win."""
        cache.put_history("EUDF.DE", series.iloc[:300])
        tail = series.iloc[290:] + 1.0
        cache.put_history("EUDF.DE", tail)
        back, _ = cache.get_history("EUDF.DE")
        assert len(back) == len(series)
        overlap = series.index[290:300]
        assert np.allclose(back.loc[overlap], series.loc[overlap] + 1.0)
        assert np.allclose(back.iloc[:290], series.iloc[:290])

    def test_nan_rows_are_not_stored(self, cache, series):
        holed = series.copy()
        holed.iloc[5] = np.nan
        cache.put_history("EUDF.DE", holed)
        back, _ = cache.get_history("EUDF.DE")
        assert len(back) == len(series) - 1
        assert not back.isna().any()

    def test_a_nan_never_overwrites_a_good_close(self, cache, series):
        cache.put_history("EUDF.DE", series)
        holed = series.iloc[-3:].copy()
        holed.iloc[0] = np.nan
        cache.put_history("EUDF.DE", holed)
        back, _ = cache.get_history("EUDF.DE")
        assert len(back) == len(series)

    def test_empty_series_is_a_no_op(self, cache):
        cache.put_history("EUDF.DE", pd.Series([], dtype=float, index=pd.DatetimeIndex([])))
        assert cache.get_history("EUDF.DE") is None

    def test_tz_aware_index_is_stored_by_date(self, cache, series):
        aware = series.copy()
        aware.index = pd.DatetimeIndex(series.index).tz_localize("Europe/Berlin")
        cache.put_history("EUDF.DE", aware)
        back, _ = cache.get_history("EUDF.DE")
        assert back.index.tz is None
        assert back.index[0] == pd.Timestamp(series.index[0])

    def test_survives_reopening_the_file(self, tmp_path, series):
        path = tmp_path / "prices.sqlite"
        first = PriceCache(path)
        first.put_history("EUDF.DE", series)
        first.close()
        second = PriceCache(path)
        back, _ = second.get_history("EUDF.DE")
        assert len(back) == len(series)
        second.close()


class TestQuotes:
    def test_datetime_as_of_comes_back_as_a_datetime(self, cache):
        cache.put_quote("EUDF.DE", quote())
        back = cache.get_quote("EUDF.DE", dt.timedelta(minutes=15))
        assert back == quote()
        assert isinstance(back.as_of, dt.datetime) and back.as_of.tzinfo is not None

    def test_date_as_of_comes_back_as_a_date_not_a_midnight_datetime(self, cache):
        """The UI renders a close date and a quote time differently; the type
        is the signal."""
        cache.put_quote("EUDF.DE", quote(as_of=dt.date(2026, 9, 4), delay=None, stale=True))
        back = cache.get_quote("EUDF.DE", dt.timedelta(minutes=15))
        assert back.as_of == dt.date(2026, 9, 4)
        assert not isinstance(back.as_of, dt.datetime)
        assert back.delay_minutes is None and back.is_stale is True

    def test_price_is_decimal_exact(self, cache):
        cache.put_quote("EUDF.DE", quote(price="0.1"))
        assert cache.get_quote("EUDF.DE", dt.timedelta(hours=1)).price == Decimal("0.1")

    def test_missing_quote_is_none(self, cache):
        assert cache.get_quote("EUDF.DE", dt.timedelta(hours=1)) is None

    def test_a_second_put_replaces_the_first(self, cache):
        cache.put_quote("EUDF.DE", quote(price="1"))
        cache.put_quote("EUDF.DE", quote(price="2"))
        assert cache.get_quote("EUDF.DE", dt.timedelta(hours=1)).price == Decimal("2")

    def test_ttl_boundaries(self, cache, monkeypatch):
        monkeypatch.setattr(cache_module, "_now", lambda: T0)
        cache.put_quote("EUDF.DE", quote())
        ttl = dt.timedelta(minutes=15)

        monkeypatch.setattr(cache_module, "_now", lambda: T0 + dt.timedelta(minutes=14))
        assert cache.get_quote("EUDF.DE", ttl) is not None
        monkeypatch.setattr(cache_module, "_now", lambda: T0 + ttl)
        assert cache.get_quote("EUDF.DE", ttl) is not None, "the boundary is inclusive"
        monkeypatch.setattr(cache_module, "_now", lambda: T0 + ttl + dt.timedelta(seconds=1))
        assert cache.get_quote("EUDF.DE", ttl) is None

    def test_an_expired_quote_is_declined_not_deleted(self, cache, monkeypatch):
        monkeypatch.setattr(cache_module, "_now", lambda: T0)
        cache.put_quote("EUDF.DE", quote())
        monkeypatch.setattr(cache_module, "_now", lambda: T0 + dt.timedelta(days=1))
        assert cache.get_quote("EUDF.DE", dt.timedelta(minutes=15)) is None
        assert cache.get_quote("EUDF.DE", dt.timedelta(days=2)) is not None


class TestStatsAndClear:
    def test_counts_symbols_across_both_tables(self, cache, series):
        cache.put_history("EUDF.DE", series)
        cache.put_quote("EUDF.DE", quote())
        cache.put_quote("DFNC.DE", quote())
        stats = cache.stats()
        assert isinstance(stats, CacheStats)
        assert stats.symbols == 2
        assert stats.rows == len(series)
        assert stats.size_bytes > 0
        assert stats.oldest is not None and stats.newest is not None
        assert stats.oldest <= stats.newest
        assert stats.oldest.tzinfo is not None

    def test_empty_cache_reports_none_not_epoch(self, cache):
        stats = cache.stats()
        assert stats == CacheStats(symbols=0, rows=0, size_bytes=stats.size_bytes,
                                   oldest=None, newest=None)

    def test_clear_empties_everything_and_shrinks_the_file(self, cache, series):
        for symbol in ("EUDF.DE", "DFNC.DE", "WEAT.MI"):
            cache.put_history(symbol, series)
        cache.put_quote("EUDF.DE", quote())
        before = cache.stats().size_bytes
        cache.clear()
        after = cache.stats()
        assert after.symbols == 0 and after.rows == 0
        assert after.oldest is None and after.newest is None
        assert after.size_bytes < before
        assert cache.get_history("EUDF.DE") is None
        assert cache.get_quote("EUDF.DE", dt.timedelta(days=365)) is None

    def test_usable_after_clear(self, cache, series):
        cache.put_history("EUDF.DE", series)
        cache.clear()
        cache.put_history("EUDF.DE", series)
        assert cache.stats().rows == len(series)


class TestSharedAcrossThreads:
    def test_one_instance_serves_a_thread_pool(self, cache):
        """market.py loads symbols from a pool sharing one cache."""
        provider = FixtureProvider()
        symbols = [f"SYM{i}.DE" for i in range(12)]

        def work(symbol: str) -> int:
            s = provider.history(symbol)
            cache.put_history(symbol, s)
            cache.put_quote(symbol, provider.quote(symbol))
            back, _ = cache.get_history(symbol)
            return len(back)

        with ThreadPoolExecutor(max_workers=6) as pool:
            lengths = list(pool.map(work, symbols))
        assert all(n > 0 for n in lengths)
        assert cache.stats().symbols == 12
        for symbol in symbols:
            assert cache.get_history(symbol) is not None
            assert cache.get_quote(symbol, dt.timedelta(minutes=15)) is not None
