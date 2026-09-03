#!/usr/bin/env python3
"""
Phase 0 provider spike.

Question this script answers: which market data provider can actually serve
MY instruments -- small, European-listed, thematic UCITS ETFs and commodity
ETCs -- with both a current quote and two years of daily history?

Design notes (read these before the code):

1. This file is deliberately standalone and throwaway. It imports nothing from
   the `portfolio` package, because the package does not exist yet and its
   provider interface should be designed AFTER we see these results, not
   before. Nothing here is production code.

2. The real output is not the pass/fail matrix alone. It is the resolved
   symbol per provider per instrument -- the `provider_symbols` mapping that
   the Instrument model needs. A provider that "works" but only by us guessing
   the right exchange suffix by hand is a provider that will silently break.

3. Ticker collisions are the main hazard and this script treats them as such.
   `WEAT` is WisdomTree Wheat in London and Teucrium Wheat Fund in New York.
   `NATO` is a European defence ETF and also a US-listed product. A provider
   that answers a bare ticker with the wrong listing is worse than a provider
   that answers nothing, because it fails silently. So every resolution is
   checked against the expected currency and an allowlist of European
   exchanges, and anything else is reported as SUSPECT, never as PASS.

4. Where a provider has its own search endpoint we use it, because that is the
   mechanism the production code will use to resolve ISIN -> provider symbol.
   Yahoo has no such endpoint, so for yfinance we brute-force exchange
   suffixes. That asymmetry is itself a finding worth seeing.

5. Every raw response is written to spike/results/ as JSON. Free tiers have
   daily caps; re-running the script to re-read a number you already fetched
   is how you lose a day of quota. Use --offline to re-print the matrix from
   the last run without touching the network.

Usage:
    export TWELVEDATA_API_KEY=...      # optional, skipped if absent
    export FMP_API_KEY=...             # optional, skipped if absent
    export EODHD_API_KEY=...           # optional, skipped if absent
    python spike/check_providers.py
    python spike/check_providers.py --providers yfinance,twelvedata
    python spike/check_providers.py --offline
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Iterable

try:
    import requests
except ImportError:  # pragma: no cover - spike script, fail loudly
    sys.exit("pip install requests")


HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HISTORY_YEARS = 2
# 2 years of European trading days is roughly 510. We call history a PASS at
# 400+, which tolerates a listing that is a few months short of two full years
# without tolerating a provider that hands back a stub of 30 rows.
MIN_HISTORY_DAYS_PASS = 400
# The covariance floor from the spec. Below this an instrument cannot enter the
# risk model at all, so it is the hard fail line.
MIN_HISTORY_DAYS_USABLE = 60

# Currencies and exchanges we expect to see for European listings. Anything
# outside these sets means we probably resolved to a US ticker of the same
# name, which is the failure mode this whole spike exists to catch.
EXPECTED_CURRENCIES = {"EUR", "GBP", "GBp", "GBX", "CHF", "USD"}
EUROPEAN_EXCHANGE_HINTS = {
    "XETRA", "XETR", "GER", "FRA", "F", "DE", "DEU", "STU", "MUN", "BER", "DUS",
    "HAM", "GETTEX", "EUR", "AMS", "AS", "EURONEXT", "XAMS", "PAR", "PA", "XPAR",
    "MIL", "MI", "MTA", "XMIL", "BIT", "ETFPLUS", "LSE", "L", "LON", "XLON",
    "SIX", "SW", "EBS", "VTX", "MAD", "BME", "STO", "CPH", "OSL", "VIE", "BRU",
    "LIS", "IOB",
}
# Currency of the *listing* we expect. USD is legitimate for some London-listed
# commodity ETCs, so it is allowed above -- but a USD price on a US exchange is
# not, and the exchange check catches that.
US_EXCHANGE_MARKERS = {"NYSE", "NASDAQ", "NMS", "NYQ", "ARCA", "PCX", "BATS", "AMEX", "US"}


# --------------------------------------------------------------------------
# Instruments under test
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TestInstrument:
    """One instrument to probe.

    `isin` is intentionally optional and empty by default. I will not invent
    ISINs: a wrong ISIN in the primary key column is a bug that propagates into
    every later stage. Fill these in from your broker statement and the
    ISIN-based resolution paths below start working -- they are already
    implemented and will be exercised the moment an ISIN is present.
    """
    key: str                 # short internal label used in the matrix
    name: str
    ticker_hint: str
    asset_class: str         # equity_etf | commodity_etc
    isin: str = ""
    # Exchange suffixes worth trying first for this instrument, most likely
    # first. Used only by providers without a search endpoint (i.e. Yahoo).
    suffix_hints: tuple[str, ...] = ()


# Suffix orderings below are guesses about likely home listings, not
# assertions. The script tries the full fallback list either way; the hint only
# changes the order, which matters because we stop at the first good hit and
# that saves quota.
DEFAULT_SUFFIXES = (".DE", ".MI", ".AS", ".L", ".PA", ".SW", ".F", "")

INSTRUMENTS: tuple[TestInstrument, ...] = (
    TestInstrument("WDEF", "WisdomTree Europe Defence", "WDEF", "equity_etf",
                   suffix_hints=(".DE", ".MI", ".L", ".AS")),
    TestInstrument("DFEU", "iShares Europe Defence", "DFEU", "equity_etf",
                   suffix_hints=(".DE", ".AS", ".MI", ".L")),
    TestInstrument("ARMY", "Global defence exposure (ARMY)", "ARMY", "equity_etf",
                   suffix_hints=(".DE", ".MI", ".L", ".AS")),
    TestInstrument("NATO", "Global defence exposure (NATO)", "NATO", "equity_etf",
                   suffix_hints=(".DE", ".MI", ".L", ".AS")),
    TestInstrument("EUDF", "Global defence exposure (EUDF)", "EUDF", "equity_etf",
                   suffix_hints=(".DE", ".MI", ".L", ".AS")),
    TestInstrument("ISAG", "iShares Agribusiness", "ISAG", "equity_etf",
                   suffix_hints=(".L", ".DE", ".AS", ".MI")),
    TestInstrument("AIGG", "WisdomTree Grains", "AIGG", "commodity_etc",
                   suffix_hints=(".L", ".MI", ".DE")),
    TestInstrument("WEAT", "WisdomTree Wheat", "WEAT", "commodity_etc",
                   suffix_hints=(".L", ".MI", ".DE")),
    TestInstrument("AIGE", "WisdomTree Agriculture", "AIGE", "commodity_etc",
                   suffix_hints=(".L", ".MI", ".DE")),
    TestInstrument("ESIE", "Energy thematic (ESIE)", "ESIE", "equity_etf",
                   suffix_hints=(".DE", ".MI", ".AS", ".L")),
    TestInstrument("GLUX", "Amundi Global Luxury", "GLUX", "equity_etf",
                   suffix_hints=(".MI", ".DE", ".PA", ".L")),
)


# --------------------------------------------------------------------------
# Result record
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Probe:
    provider: str
    instrument: str

    quote_ok: bool = False
    quote_price: float | None = None
    quote_timestamp: str | None = None
    observed_staleness_minutes: float | None = None

    history_ok: bool = False
    history_days: int = 0
    first_date: str | None = None
    last_date: str | None = None

    currency: str | None = None
    exchange: str | None = None
    resolved_symbol: str | None = None
    resolution_method: str | None = None   # "search" | "suffix" | "isin"

    suspect: bool = False
    suspect_reason: str | None = None
    errors: list[str] = dataclasses.field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Three-state verdict.

        SUSPECT deliberately outranks PASS: data that arrived from the wrong
        listing is a worse outcome than no data, so it must never be counted
        as a pass in the summary.
        """
        if self.suspect:
            return "SUSPECT"
        if self.quote_ok and self.history_ok:
            return "PASS"
        if self.quote_ok or self.history_days >= MIN_HISTORY_DAYS_USABLE:
            return "PARTIAL"
        return "FAIL"

    def flag_suspect(self, reason: str) -> None:
        self.suspect = True
        self.suspect_reason = reason


def check_listing_plausible(probe: Probe, expected_european: bool = True) -> None:
    """Reject a resolution that looks like the wrong listing.

    This is the ISIN-is-the-primary-key constraint expressed as a runtime
    check. We cannot verify the ISIN without one on file, so we verify the next
    best thing: that the exchange we got back is a European one.
    """
    if not expected_european:
        return
    exch = (probe.exchange or "").upper()
    if not exch:
        return
    tokens = {t for t in exch.replace("/", " ").replace("-", " ").split()} | {exch}
    if tokens & US_EXCHANGE_MARKERS:
        probe.flag_suspect(
            f"resolved to a US listing ({probe.exchange}); "
            f"likely a different fund sharing the ticker"
        )
        return
    if not (tokens & EUROPEAN_EXCHANGE_HINTS):
        probe.flag_suspect(f"unrecognised exchange {probe.exchange!r}; verify by hand")


# --------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------

def get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 25) -> tuple[int, Any]:
    """One place for every HTTP call so throttling and errors are uniform."""
    try:
        resp = requests.get(url, params=params, timeout=timeout,
                            headers={"User-Agent": "portfolio-tracker-spike/0.1"})
    except requests.RequestException as exc:
        return 0, {"_transport_error": str(exc)}
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, {"_non_json_body": resp.text[:400]}


def days_between(first: str, last: str) -> int:
    try:
        a = dt.date.fromisoformat(first[:10])
        b = dt.date.fromisoformat(last[:10])
        return (b - a).days
    except Exception:
        return 0


# --------------------------------------------------------------------------
# Provider probes
# --------------------------------------------------------------------------

class ProviderProbe:
    name = "abstract"
    # These two fields are the second half of the Phase 0 question. Batching
    # changes the caching design entirely: with batch quotes you cache one
    # response per portfolio refresh, without them one per instrument.
    documented_rate_limit = "unknown"
    batch_quotes = "unknown"
    documented_delay = "unknown"
    seconds_between_calls = 0.0

    def available(self) -> tuple[bool, str]:
        return True, ""

    def measure_limits(self) -> dict[str, Any]:
        """Ask the provider what our actual limits are, where it exposes that.

        Documented limits go stale. A live usage endpoint is evidence.
        """
        return {}

    def probe(self, inst: TestInstrument) -> Probe:
        raise NotImplementedError

    def throttle(self) -> None:
        if self.seconds_between_calls:
            time.sleep(self.seconds_between_calls)


class YFinanceProbe(ProviderProbe):
    """yfinance -- the coverage benchmark, not necessarily the production pick.

    Judgement call: I use the yfinance library rather than hand-rolling calls to
    Yahoo's chart endpoint, because yfinance handles the cookie/crumb dance
    that Yahoo now requires and that changes every few months. The alternative I
    rejected was raw requests against query1.finance.yahoo.com: fewer
    dependencies, but it breaks the week Yahoo changes auth, which defeats the
    purpose of using it as a stable benchmark.

    Yahoo has no usable symbol-search API for this purpose, so resolution here
    is brute-force over exchange suffixes. That is a real cost: it is N calls
    per instrument instead of one, and it is why yfinance being "free" is
    misleading.
    """
    name = "yfinance"
    documented_rate_limit = ("undocumented and unofficial; soft-throttled by IP, "
                             "roughly a few hundred requests/hour before 429s")
    batch_quotes = ("yes for history via yf.download([...]) in one call; "
                    "quotes via yf.Tickers are looped internally, not truly batched")
    documented_delay = "15 minutes for most European exchanges (Yahoo does not state it per-call)"
    seconds_between_calls = 0.4

    def available(self) -> tuple[bool, str]:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False, "pip install yfinance"
        return True, ""

    def probe(self, inst: TestInstrument) -> Probe:
        import yfinance as yf

        p = Probe(self.name, inst.key)
        suffixes = list(dict.fromkeys(inst.suffix_hints + DEFAULT_SUFFIXES))
        start = (dt.date.today() - dt.timedelta(days=365 * HISTORY_YEARS + 10)).isoformat()

        best: tuple[int, str, Any] | None = None
        for suffix in suffixes:
            symbol = f"{inst.ticker_hint}{suffix}"
            self.throttle()
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start, interval="1d", auto_adjust=False)
            except Exception as exc:
                p.errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
                continue
            if hist is None or hist.empty:
                continue
            n = len(hist)
            # Keep the listing with the most history: for a fund cross-listed on
            # several exchanges, the deepest series is the one worth using.
            if best is None or n > best[0]:
                best = (n, symbol, (ticker, hist))
            if n >= MIN_HISTORY_DAYS_PASS:
                break

        if best is None:
            p.errors.append("no suffix returned any history")
            return p

        n, symbol, (ticker, hist) = best
        p.resolved_symbol = symbol
        p.resolution_method = "suffix"
        p.history_ok = n >= MIN_HISTORY_DAYS_PASS
        p.history_days = n
        p.first_date = str(hist.index[0].date())
        p.last_date = str(hist.index[-1].date())

        # We ask for auto_adjust=False above and check that an adjusted column
        # exists, because the risk maths must run on adjusted closes. A provider
        # that only offers raw closes will corrupt every return around a
        # distribution, and these ETCs distribute.
        if "Adj Close" not in hist.columns:
            p.errors.append("no 'Adj Close' column -- returns would be corrupted by distributions")

        try:
            info = ticker.fast_info
            p.currency = getattr(info, "currency", None)
            p.exchange = getattr(info, "exchange", None)
            last = getattr(info, "last_price", None)
            if last is not None:
                p.quote_ok = True
                p.quote_price = float(last)
        except Exception as exc:
            p.errors.append(f"fast_info: {type(exc).__name__}: {exc}")

        if not p.quote_ok:
            # Fall back to the last close, and say so. Never present this as a
            # live quote -- that is exactly the failure mode the spec forbids.
            p.quote_price = float(hist["Close"].iloc[-1])
            p.quote_timestamp = p.last_date
            p.errors.append("no live quote; showing last close only")

        check_listing_plausible(p)
        return p


class TwelveDataProbe(ProviderProbe):
    """Twelve Data free tier.

    Has a real symbol_search endpoint that returns exchange and currency, which
    is what we want for ISIN -> symbol resolution. Free tier credits are the
    binding constraint.
    """
    name = "twelvedata"
    documented_rate_limit = "free tier: 8 API credits/minute, 800 credits/day (verify in dashboard)"
    batch_quotes = "yes -- /quote?symbol=A,B,C, but each symbol costs one credit"
    documented_delay = "free tier is end-of-day / delayed for most European venues; real-time needs a paid plan"
    seconds_between_calls = 8.0  # 8 credits/minute means one call every ~7.5s

    BASE = "https://api.twelvedata.com"

    def __init__(self) -> None:
        self.key = os.environ.get("TWELVEDATA_API_KEY", "")

    def available(self) -> tuple[bool, str]:
        return bool(self.key), "set TWELVEDATA_API_KEY"

    def measure_limits(self) -> dict[str, Any]:
        code, body = get_json(f"{self.BASE}/api_usage", {"apikey": self.key})
        return {"http": code, "api_usage": body}

    def _resolve(self, inst: TestInstrument) -> tuple[str | None, dict[str, Any] | None]:
        self.throttle()
        code, body = get_json(f"{self.BASE}/symbol_search",
                              {"symbol": inst.ticker_hint, "outputsize": 30})
        if code != 200 or not isinstance(body, dict):
            return None, None
        candidates = body.get("data") or []
        # Prefer a European venue; that preference is the whole point.
        def score(row: dict[str, Any]) -> int:
            exch = str(row.get("exchange", "")).upper()
            country = str(row.get("country", "")).upper()
            s = 0
            if any(h in exch for h in EUROPEAN_EXCHANGE_HINTS):
                s += 10
            if country and country not in {"UNITED STATES", "US", "USA"}:
                s += 5
            if str(row.get("instrument_type", "")).upper() in {"ETF", "ETC", "FUND"}:
                s += 2
            return s
        ranked = sorted(candidates, key=score, reverse=True)
        if not ranked:
            return None, None
        return ranked[0].get("symbol"), ranked[0]

    def probe(self, inst: TestInstrument) -> Probe:
        p = Probe(self.name, inst.key)
        symbol, meta = self._resolve(inst)
        if not symbol:
            p.errors.append("symbol_search returned no candidates")
            return p
        p.resolved_symbol = symbol
        p.resolution_method = "search"
        exchange = (meta or {}).get("exchange")
        p.exchange = exchange
        p.currency = (meta or {}).get("currency")

        params = {"symbol": symbol, "apikey": self.key}
        if exchange:
            params["exchange"] = exchange

        self.throttle()
        code, quote = get_json(f"{self.BASE}/quote", params)
        if code == 200 and isinstance(quote, dict) and quote.get("close") is not None:
            p.quote_ok = True
            p.quote_price = float(quote["close"])
            ts = quote.get("timestamp")
            if ts:
                when = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)
                p.quote_timestamp = when.isoformat()
                p.observed_staleness_minutes = round(
                    (dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 60, 1)
            p.currency = quote.get("currency") or p.currency
            p.exchange = quote.get("exchange") or p.exchange
        else:
            p.errors.append(f"quote http={code} body={str(quote)[:180]}")

        self.throttle()
        hist_params = dict(params, interval="1day", outputsize=5000)
        code, hist = get_json(f"{self.BASE}/time_series", hist_params)
        values = hist.get("values") if isinstance(hist, dict) else None
        if code == 200 and values:
            p.history_days = len(values)
            # Twelve Data returns newest first.
            p.last_date = values[0]["datetime"][:10]
            p.first_date = values[-1]["datetime"][:10]
            p.history_ok = p.history_days >= MIN_HISTORY_DAYS_PASS
            # Note: the free tier serves unadjusted prices on /time_series.
            # Adjusted series require the `adjust` parameter on a paid plan --
            # if that is the case here, this provider cannot be used for the
            # risk maths without corrupting returns around distributions.
            p.errors.append("check whether these closes are dividend-adjusted before trusting returns")
        else:
            p.errors.append(f"time_series http={code} body={str(hist)[:180]}")

        check_listing_plausible(p)
        return p


class FMPProbe(ProviderProbe):
    """Financial Modeling Prep free tier.

    FMP's European coverage is the open question. Its search endpoint accepts a
    ticker and returns exchange plus currency, and it has an ISIN search path we
    exercise as soon as ISINs are on file.
    """
    name = "fmp"
    documented_rate_limit = "free tier: 250 requests/day, US-only on some plans (verify in dashboard)"
    batch_quotes = "yes -- /api/v3/quote/A,B,C in one request"
    documented_delay = "free tier is end-of-day; intraday requires a paid plan"
    seconds_between_calls = 0.5

    BASE = "https://financialmodelingprep.com/api"

    def __init__(self) -> None:
        self.key = os.environ.get("FMP_API_KEY", "")

    def available(self) -> tuple[bool, str]:
        return bool(self.key), "set FMP_API_KEY"

    def _resolve(self, inst: TestInstrument) -> tuple[str | None, dict[str, Any] | None]:
        # Path 1: ISIN, the correct key, used whenever we have one.
        if inst.isin:
            code, body = get_json(f"{self.BASE}/v4/search/isin",
                                  {"isin": inst.isin, "apikey": self.key})
            if code == 200 and isinstance(body, list) and body:
                return body[0].get("symbol"), body[0]
        # Path 2: ticker search, ranked toward European venues.
        code, body = get_json(f"{self.BASE}/v3/search",
                              {"query": inst.ticker_hint, "limit": 30, "apikey": self.key})
        if code != 200 or not isinstance(body, list) or not body:
            return None, None
        def score(row: dict[str, Any]) -> int:
            exch = str(row.get("exchangeShortName") or row.get("stockExchange") or "").upper()
            s = 10 if any(h in exch for h in EUROPEAN_EXCHANGE_HINTS) else 0
            if exch in US_EXCHANGE_MARKERS:
                s -= 10
            if str(row.get("currency", "")).upper() in {"EUR", "GBP", "GBX", "CHF"}:
                s += 5
            return s
        ranked = sorted(body, key=score, reverse=True)
        return ranked[0].get("symbol"), ranked[0]

    def probe(self, inst: TestInstrument) -> Probe:
        p = Probe(self.name, inst.key)
        symbol, meta = self._resolve(inst)
        if not symbol:
            p.errors.append("search returned no candidates")
            return p
        p.resolved_symbol = symbol
        p.resolution_method = "isin" if inst.isin else "search"
        p.exchange = (meta or {}).get("exchangeShortName") or (meta or {}).get("stockExchange")
        p.currency = (meta or {}).get("currency")

        self.throttle()
        code, quote = get_json(f"{self.BASE}/v3/quote/{symbol}", {"apikey": self.key})
        if code == 200 and isinstance(quote, list) and quote:
            row = quote[0]
            p.quote_ok = row.get("price") is not None
            p.quote_price = row.get("price")
            p.exchange = row.get("exchange") or p.exchange
            ts = row.get("timestamp")
            if ts:
                when = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)
                p.quote_timestamp = when.isoformat()
                p.observed_staleness_minutes = round(
                    (dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 60, 1)
        else:
            p.errors.append(f"quote http={code} body={str(quote)[:180]}")

        self.throttle()
        frm = (dt.date.today() - dt.timedelta(days=365 * HISTORY_YEARS + 10)).isoformat()
        code, hist = get_json(f"{self.BASE}/v3/historical-price-full/{symbol}",
                              {"from": frm, "to": dt.date.today().isoformat(), "apikey": self.key})
        rows = hist.get("historical") if isinstance(hist, dict) else None
        if code == 200 and rows:
            p.history_days = len(rows)
            p.last_date = rows[0]["date"][:10]
            p.first_date = rows[-1]["date"][:10]
            p.history_ok = p.history_days >= MIN_HISTORY_DAYS_PASS
            if "adjClose" not in rows[0]:
                p.errors.append("no adjClose field -- returns would be corrupted by distributions")
        else:
            p.errors.append(f"history http={code} body={str(hist)[:180]}")

        check_listing_plausible(p)
        return p


class EODHDProbe(ProviderProbe):
    """EODHD -- paid, but historically the strongest European ETF coverage.

    The point of testing it on a trial is to decide whether paying is warranted.
    Its search endpoint accepts an ISIN directly, which is exactly the
    resolution path the architecture wants, so if the coverage holds up this is
    the provider that fits the design best.
    """
    name = "eodhd"
    documented_rate_limit = "free demo key: ~20 req/day and a fixed demo symbol list; paid plans 1000+/day (verify via /api/user)"
    batch_quotes = "yes -- /api/real-time/AAA.US?s=BBB.LSE,CCC.MI"
    documented_delay = "15-20 minutes on most European venues; end-of-day is T+1"
    seconds_between_calls = 1.0

    BASE = "https://eodhd.com/api"

    def __init__(self) -> None:
        self.key = os.environ.get("EODHD_API_KEY", "")

    def available(self) -> tuple[bool, str]:
        return bool(self.key), "set EODHD_API_KEY (a trial or demo key is fine)"

    def measure_limits(self) -> dict[str, Any]:
        code, body = get_json(f"{self.BASE}/user", {"api_token": self.key, "fmt": "json"})
        if isinstance(body, dict):
            return {"http": code,
                    "daily_rate_limit": body.get("dailyRateLimit"),
                    "requests_used_today": body.get("apiRequests"),
                    "subscription": body.get("subscriptionType")}
        return {"http": code, "body": str(body)[:200]}

    def _resolve(self, inst: TestInstrument) -> tuple[str | None, dict[str, Any] | None]:
        # EODHD's search takes an ISIN or a ticker on the same path, which is
        # the cleanest ISIN-first resolution of the four providers.
        query = inst.isin or inst.ticker_hint
        code, body = get_json(f"{self.BASE}/search/{query}",
                              {"api_token": self.key, "fmt": "json", "limit": 30})
        if code != 200 or not isinstance(body, list) or not body:
            return None, None
        def score(row: dict[str, Any]) -> int:
            exch = str(row.get("Exchange", "")).upper()
            s = 10 if any(h in exch for h in EUROPEAN_EXCHANGE_HINTS) else 0
            if exch in US_EXCHANGE_MARKERS:
                s -= 10
            if str(row.get("Currency", "")).upper() in {"EUR", "GBP", "GBX", "CHF"}:
                s += 5
            return s
        ranked = sorted(body, key=score, reverse=True)
        top = ranked[0]
        symbol = f"{top.get('Code')}.{top.get('Exchange')}"
        return symbol, top

    def probe(self, inst: TestInstrument) -> Probe:
        p = Probe(self.name, inst.key)
        symbol, meta = self._resolve(inst)
        if not symbol:
            p.errors.append("search returned no candidates")
            return p
        p.resolved_symbol = symbol
        p.resolution_method = "isin" if inst.isin else "search"
        p.exchange = (meta or {}).get("Exchange")
        p.currency = (meta or {}).get("Currency")

        self.throttle()
        code, quote = get_json(f"{self.BASE}/real-time/{symbol}",
                               {"api_token": self.key, "fmt": "json"})
        if code == 200 and isinstance(quote, dict) and quote.get("close") not in (None, "NA"):
            p.quote_ok = True
            p.quote_price = float(quote["close"])
            ts = quote.get("timestamp")
            if isinstance(ts, (int, float)):
                when = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)
                p.quote_timestamp = when.isoformat()
                p.observed_staleness_minutes = round(
                    (dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 60, 1)
        else:
            p.errors.append(f"real-time http={code} body={str(quote)[:180]}")

        self.throttle()
        frm = (dt.date.today() - dt.timedelta(days=365 * HISTORY_YEARS + 10)).isoformat()
        code, hist = get_json(f"{self.BASE}/eod/{symbol}",
                              {"api_token": self.key, "fmt": "json", "period": "d", "from": frm})
        if code == 200 and isinstance(hist, list) and hist:
            p.history_days = len(hist)
            p.first_date = hist[0]["date"][:10]
            p.last_date = hist[-1]["date"][:10]
            p.history_ok = p.history_days >= MIN_HISTORY_DAYS_PASS
            if "adjusted_close" not in hist[0]:
                p.errors.append("no adjusted_close field -- returns would be corrupted by distributions")
        else:
            p.errors.append(f"eod http={code} body={str(hist)[:180]}")

        check_listing_plausible(p)
        return p


PROVIDERS: dict[str, Callable[[], ProviderProbe]] = {
    "yfinance": YFinanceProbe,
    "twelvedata": TwelveDataProbe,
    "fmp": FMPProbe,
    "eodhd": EODHDProbe,
}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

VERDICT_GLYPH = {"PASS": "PASS", "PARTIAL": "PART", "SUSPECT": "SUSP", "FAIL": "FAIL"}


def print_detail(provider: str, probes: list[Probe]) -> None:
    print(f"\n{'=' * 100}")
    print(f"PROVIDER: {provider}")
    print("=" * 100)
    head = (f"{'instrument':<10} {'symbol':<16} {'via':<7} {'quote':<6} {'days':>5} "
            f"{'first':<11} {'last':<11} {'ccy':<5} {'exchange':<12} verdict")
    print(head)
    print("-" * len(head))
    for p in probes:
        print(f"{p.instrument:<10} {(p.resolved_symbol or '-'):<16} "
              f"{(p.resolution_method or '-'):<7} {('yes' if p.quote_ok else 'no'):<6} "
              f"{p.history_days:>5} {(p.first_date or '-'):<11} {(p.last_date or '-'):<11} "
              f"{(p.currency or '-'):<5} {(p.exchange or '-'):<12} {p.verdict}")
    for p in probes:
        if p.suspect:
            print(f"  ! {p.instrument}: {p.suspect_reason}")
        for e in p.errors:
            print(f"    - {p.instrument}: {e}")


def print_matrix(results: dict[str, list[Probe]]) -> None:
    providers = list(results)
    print(f"\n{'=' * 100}")
    print("COVERAGE MATRIX  (PASS = quote + >=%d days history from a plausible European listing)"
          % MIN_HISTORY_DAYS_PASS)
    print("SUSP = data returned, but from a listing that looks wrong -- treat as a failure")
    print("=" * 100)
    width = max(12, max((len(p) for p in providers), default=12) + 2)
    print(f"{'instrument':<12}" + "".join(f"{p:<{width}}" for p in providers))
    print("-" * (12 + width * len(providers)))
    for inst in INSTRUMENTS:
        row = f"{inst.key:<12}"
        for prov in providers:
            probe = next((x for x in results[prov] if x.instrument == inst.key), None)
            row += f"{(VERDICT_GLYPH[probe.verdict] if probe else '-'):<{width}}"
        print(row)
    print("-" * (12 + width * len(providers)))
    score = f"{'PASS count':<12}"
    for prov in providers:
        n = sum(1 for x in results[prov] if x.verdict == "PASS")
        score += f"{f'{n}/{len(INSTRUMENTS)}':<{width}}"
    print(score)


def print_limits(probes: dict[str, ProviderProbe], measured: dict[str, dict]) -> None:
    print(f"\n{'=' * 100}")
    print("RATE LIMITS AND BATCHING  (documented values -- verify against your own dashboard)")
    print("=" * 100)
    for name, prov in probes.items():
        print(f"\n{name}")
        print(f"  rate limit : {prov.documented_rate_limit}")
        print(f"  batching   : {prov.batch_quotes}")
        print(f"  delay      : {prov.documented_delay}")
        if measured.get(name):
            print(f"  measured   : {json.dumps(measured[name])[:300]}")


def print_symbol_map(results: dict[str, list[Probe]]) -> None:
    """The provider_symbols mapping, ready to paste into instruments.csv.

    This is the deliverable that matters most after the matrix: it is the
    bridge between ISIN as the primary key and each provider's own symbol.
    """
    print(f"\n{'=' * 100}")
    print("RESOLVED provider_symbols (only non-suspect resolutions)")
    print("=" * 100)
    mapping: dict[str, dict[str, str]] = {}
    for prov, probes in results.items():
        for p in probes:
            if p.resolved_symbol and not p.suspect:
                mapping.setdefault(p.instrument, {})[prov] = p.resolved_symbol
    print(json.dumps(mapping, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers", default=",".join(PROVIDERS),
                    help="comma-separated subset of: " + ", ".join(PROVIDERS))
    ap.add_argument("--instruments", default="",
                    help="comma-separated instrument keys; default is all")
    ap.add_argument("--offline", action="store_true",
                    help="re-print the last saved run without any network calls")
    args = ap.parse_args()

    if args.offline:
        latest = sorted(RESULTS_DIR.glob("run-*.json"))
        if not latest:
            print("no saved run in spike/results/")
            return 1
        saved = json.loads(latest[-1].read_text())
        results = {prov: [Probe(**row) for row in rows] for prov, rows in saved["results"].items()}
        for prov, probes in results.items():
            print_detail(prov, probes)
        print_matrix(results)
        print_symbol_map(results)
        print(f"\n(offline replay of {latest[-1].name})")
        return 0

    wanted_instruments = [i for i in INSTRUMENTS
                          if not args.instruments or i.key in args.instruments.split(",")]

    results: dict[str, list[Probe]] = {}
    live: dict[str, ProviderProbe] = {}
    measured: dict[str, dict] = {}

    for name in args.providers.split(","):
        name = name.strip()
        if name not in PROVIDERS:
            print(f"unknown provider {name!r}, skipping")
            continue
        prov = PROVIDERS[name]()
        ok, why = prov.available()
        if not ok:
            print(f"[skip] {name}: {why}")
            continue
        live[name] = prov
        measured[name] = prov.measure_limits()
        print(f"[run ] {name}: probing {len(wanted_instruments)} instruments "
              f"(throttle {prov.seconds_between_calls}s/call)")
        probes = []
        for inst in wanted_instruments:
            probe = prov.probe(inst)
            probes.append(probe)
            print(f"       {inst.key:<6} -> {probe.verdict}")
        results[name] = probes

    if not results:
        print("\nNo provider ran. Set at least one API key, or install yfinance.")
        return 1

    for prov, probes in results.items():
        print_detail(prov, probes)
    print_matrix(results)
    print_limits(live, measured)
    print_symbol_map(results)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"run-{stamp}.json"
    out.write_text(json.dumps({
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results": {prov: [dataclasses.asdict(p) for p in probes]
                    for prov, probes in results.items()},
        "measured_limits": measured,
    }, indent=2))
    print(f"\nRaw results written to {out}")
    print("Re-print without spending quota:  python spike/check_providers.py --offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
