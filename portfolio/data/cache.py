"""SQLite price cache. A restart must never refetch history already on disk.

Why a cache at all
------------------
Ten instruments, one benchmark and a handful of FX pairs is a dozen history
calls and a dozen quote calls per page load. Yahoo rate-limits (HTTP 429 in the
2026-09-03 matrix run), and a two-year daily series does not change except at
its tail. So history is stored on disk and fetched incrementally from the last
cached date, and quotes are held for a short TTL so a browser refresh does not
become a provider round trip.

Why SQLite, and why the stdlib module
-------------------------------------
Rejected alternatives: one CSV per symbol (no atomic upsert, no cheap "give me
the last date"), a pickle of a dict (unreadable when it goes wrong, and it will
go wrong in someone's home directory), and an ORM (a dependency to carry for
three tables). `sqlite3` ships with Python, is transactional, and one file
under `data_store/` can be deleted to start over.

Concurrency
-----------
`data/market.py` loads symbols from a thread pool sharing one instance. SQLite
connections are not thread-safe, so the connection is opened with
`check_same_thread=False` and every public method takes a lock. WAL journal
mode is set so a reader on the main thread does not block a writer in the pool.

Nothing in this file interprets prices. It stores what the provider said and
returns it with the timestamp it was stored at; freshness decisions belong to
the caller, which knows the TTL it is working to.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import sqlite3
import threading
from contextlib import contextmanager
from decimal import Decimal
from typing import Iterator

import pandas as pd

from .provider import Quote

__all__ = ["PriceCache", "CacheStats"]


def _now() -> dt.datetime:
    """The cache's clock. One function so a test can move it."""
    return dt.datetime.now(dt.timezone.utc)


def _stamp(when: dt.datetime) -> str:
    # A fixed timespec keeps the stored text the same width for every row, so
    # the strings sort the way the instants do. isoformat() alone drops the
    # microsecond field when it is zero, and mixed widths are a sorting trap.
    return when.astimezone(dt.timezone.utc).isoformat(timespec="microseconds")


def _unstamp(text: str) -> dt.datetime:
    when = dt.datetime.fromisoformat(text)
    # A naive value can only have come from a hand-edited file; it was
    # written as UTC, so read it as UTC rather than as local time.
    return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    close  REAL NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS history_meta (
    symbol     TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    first_date TEXT,
    last_date  TEXT,
    rows       INTEGER
);
CREATE TABLE IF NOT EXISTS quotes (
    symbol        TEXT PRIMARY KEY,
    price         TEXT NOT NULL,
    currency      TEXT NOT NULL,
    as_of         TEXT NOT NULL,
    source        TEXT NOT NULL,
    delay_minutes INTEGER,
    is_stale      INTEGER NOT NULL,
    fetched_at    TEXT NOT NULL
);
"""


@dataclasses.dataclass(frozen=True)
class CacheStats:
    """What the Settings page shows about the cache."""
    symbols: int
    rows: int
    size_bytes: int
    oldest: dt.datetime | None      # earliest fetched_at across histories and quotes
    newest: dt.datetime | None


class PriceCache:
    """Histories and quotes by provider symbol, on disk.

    Keyed by symbol rather than ISIN on purpose: the cache stores what a
    provider returned for a fetch handle, and the ISIN-to-symbol map can be
    corrected by hand at any time. Keying on ISIN would make a corrected
    symbol serve the old symbol's prices until the TTL expired.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # isolation_level=None puts the connection in autocommit mode, which
        # is the only mode in which VACUUM is allowed to run; the bulk writes
        # open their own transaction explicitly in `_transaction`.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                     isolation_level=None)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)

    # -- plumbing ----------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "PriceCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- history -----------------------------------------------------------

    def get_history(self, symbol: str) -> tuple[pd.Series, dt.datetime] | None:
        """The cached series and when it was last fetched, or None.

        The series is sorted, float, indexed by a naive DatetimeIndex of
        dates, and named by the symbol -- the same shape a provider returns,
        so a caller cannot tell the two apart and never needs to.
        """
        with self._lock:
            meta = self._conn.execute(
                "SELECT fetched_at FROM history_meta WHERE symbol = ?", (symbol,)
            ).fetchone()
            if meta is None:
                return None
            rows = self._conn.execute(
                "SELECT date, close FROM history WHERE symbol = ? ORDER BY date",
                (symbol,)).fetchall()
        if not rows:
            return None
        dates = pd.to_datetime([r[0] for r in rows], format="%Y-%m-%d")
        series = pd.Series([r[1] for r in rows], index=pd.DatetimeIndex(dates),
                           name=symbol, dtype=float)
        return series, _unstamp(meta[0])

    def put_history(self, symbol: str, series: pd.Series) -> None:
        """Upsert rows; existing dates are overwritten, others are kept.

        That merge is what makes an incremental fetch safe: the caller can
        hand over only the tail it just fetched and the earlier rows survive.
        A NaN close is a gap, not a price, and is not stored -- storing it
        would let a provider's missing day overwrite a good one.
        """
        clean = series.dropna()
        if clean.empty:
            return
        index = pd.DatetimeIndex(clean.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        rows = [(symbol, ts.strftime("%Y-%m-%d"), float(v))
                for ts, v in zip(index, clean.to_numpy())]
        with self._transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO history (symbol, date, close) VALUES (?, ?, ?)",
                rows)
            first, last, count = conn.execute(
                "SELECT MIN(date), MAX(date), COUNT(*) FROM history WHERE symbol = ?",
                (symbol,)).fetchone()
            conn.execute(
                "INSERT OR REPLACE INTO history_meta "
                "(symbol, fetched_at, first_date, last_date, rows) VALUES (?, ?, ?, ?, ?)",
                (symbol, _stamp(_now()), first, last, count))

    # -- quotes ------------------------------------------------------------

    def get_quote(self, symbol: str, max_age: dt.timedelta) -> Quote | None:
        """The stored quote if it was fetched within `max_age`, else None.

        The boundary is inclusive: a quote exactly `max_age` old is served.
        Stale quotes are not deleted, only declined -- `data/market.py` may
        still prefer an expired quote's history over nothing at all.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT price, currency, as_of, source, delay_minutes, is_stale, fetched_at "
                "FROM quotes WHERE symbol = ?", (symbol,)).fetchone()
        if row is None:
            return None
        price, currency, as_of, source, delay, is_stale, fetched_at = row
        if _now() - _unstamp(fetched_at) > max_age:
            return None
        return Quote(symbol=symbol, price=Decimal(price), currency=currency,
                     as_of=_restore_as_of(as_of), source=source,
                     delay_minutes=None if delay is None else int(delay),
                     is_stale=bool(is_stale))

    def put_quote(self, symbol: str, quote: Quote) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO quotes (symbol, price, currency, as_of, source, "
                "delay_minutes, is_stale, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, str(quote.price), quote.currency, quote.as_of.isoformat(),
                 quote.source, quote.delay_minutes, int(bool(quote.is_stale)),
                 _stamp(_now())))

    # -- housekeeping ------------------------------------------------------

    def stats(self) -> CacheStats:
        with self._lock:
            symbols = self._conn.execute(
                "SELECT COUNT(*) FROM (SELECT symbol FROM history_meta "
                "UNION SELECT symbol FROM quotes)").fetchone()[0]
            rows = self._conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            stamps = [r[0] for r in self._conn.execute(
                "SELECT fetched_at FROM history_meta UNION ALL "
                "SELECT fetched_at FROM quotes")]
            # Logical size rather than the file's: with WAL the main file lags
            # behind until a checkpoint, and reporting that would show a cache
            # that appears not to grow.
            page_count = self._conn.execute("PRAGMA page_count").fetchone()[0]
            page_size = self._conn.execute("PRAGMA page_size").fetchone()[0]
        when = [_unstamp(s) for s in stamps]
        return CacheStats(symbols=int(symbols), rows=int(rows),
                          size_bytes=int(page_count) * int(page_size),
                          oldest=min(when) if when else None,
                          newest=max(when) if when else None)

    def clear(self) -> None:
        """Empty every table and give the space back to the filesystem."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM history")
            conn.execute("DELETE FROM history_meta")
            conn.execute("DELETE FROM quotes")
        with self._lock:
            self._conn.execute("VACUUM")


def _restore_as_of(text: str) -> dt.datetime | dt.date:
    # A datetime's ISO form always carries a "T"; a date's never does. That
    # one character is what tells a 17:30 quote from an end-of-day close, and
    # the UI renders the two differently, so the type must survive the trip.
    return dt.datetime.fromisoformat(text) if "T" in text else dt.date.fromisoformat(text)
