#!/usr/bin/env python3
"""
Phase 0 provider spike.

The question: which market data provider can serve MY instruments -- small,
European-listed, thematic UCITS ETFs and USD-denominated commodity ETCs -- with
a current quote and two years of daily history, keyed on ISIN?

This is not production code. It imports nothing from the `portfolio` package,
because that package does not exist yet and its provider interface should be
designed from these results rather than ahead of them.

Five checks, not one. Coverage is only the first.

  1. COVERAGE     Quote + >=400 trading days for each of the ten instruments,
                  resolved from its ISIN. Reported separately for ETFs and
                  ETCs, because ETCs are collateralised notes rather than UCITS
                  funds and several providers classify or omit them differently.

  2. IDENTITY     Every symbol of an instrument must resolve to that
                  instrument's ISIN. WDEF and EUDF are one fund. A provider
                  that answers them with two different instruments has exactly
                  the failure the ISIN-as-primary-key rule exists to prevent,
                  and it is invisible unless you test for it.

  3. COLLISION    Five of these tickers collide with real US-listed products.
                  We ask each provider for the bare ticker, the way a naive
                  lookup would, and report the ISIN and exchange it hands back.
                  WDEF is the dangerous case: same issuer, same theme, same
                  ticker, different fund.

  4. LIVENESS     Two of these funds have liquidated predecessors still present
                  in fund databases. Returning nothing is the correct answer. A
                  provider that returns recent-looking history for a dead ISIN
                  is serving stale data and is disqualified.

  5. MECHANICS    Rate limit, batch quote support, adjusted closes, and base
                  versus quote currency -- reported separately, because seven
                  of these ten quote in a currency other than their base.

Usage:
    export TWELVEDATA_API_KEY=...      # optional; unkeyed providers are skipped
    export FMP_API_KEY=...
    export EODHD_API_KEY=...

    python spike/check_providers.py
    python spike/check_providers.py --providers yfinance
    python spike/check_providers.py --checks coverage,collision
    python spike/check_providers.py --offline        # replay, spends no quota
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
from typing import Any, Callable

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install -r spike/requirements.txt")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from instruments import (  # noqa: E402
    BY_ISIN, COLLISIONS, DEAD_INSTRUMENTS, DEAD_LINE_STALE_DAYS, INSTRUMENTS,
    VENUES, Collision, DeadInstrument, Listing, TestInstrument, alias_groups,
)

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HISTORY_YEARS = 2
# ~510 European trading days in two years. 400 tolerates a listing a few months
# short of two years without tolerating a 30-row stub.
MIN_HISTORY_DAYS_PASS = 400
# The covariance floor from the spec: below this an instrument cannot enter the
# risk model at all.
MIN_HISTORY_DAYS_USABLE = 60

US_EXCHANGE_MARKERS = {
    "NYSE", "NASDAQ", "NMS", "NYQ", "NGM", "NCM", "ARCA", "PCX", "BATS",
    "AMEX", "ASE", "US", "USA", "NYSEARCA", "CBOE",
}
EUROPEAN_MICS = set(VENUES)
EUROPEAN_EXCHANGE_HINTS = {
    "XETRA", "XETR", "GER", "FRA", "F", "DE", "DEU", "STU", "MUN", "BER",
    "DUS", "HAM", "GETTEX", "AMS", "AS", "EURONEXT", "XAMS", "PAR", "PA",
    "XPAR", "MIL", "MI", "MTA", "XMIL", "BIT", "ETFPLUS", "LSE", "L", "LON",
    "XLON", "SIX", "SW", "EBS", "VTX", "SWX", "XSWX", "IOB", "MU", "XMUN",
} | EUROPEAN_MICS


# ==========================================================================
# Shared records
# ==========================================================================

@dataclasses.dataclass
class Resolution:
    """What a provider says an identifier refers to, before we fetch anything."""
    symbol: str | None = None
    exchange: str | None = None
    currency: str | None = None
    isin: str | None = None
    name: str | None = None
    method: str | None = None       # "isin" | "symbol-search" | "listing"
    note: str | None = None


@dataclasses.dataclass
class Probe:
    """One instrument, one provider, all five checks' worth of evidence."""
    provider: str
    isin: str
    name: str = ""
    asset_class: str = ""
    base_currency: str = ""

    quote_ok: bool = False
    quote_price: float | None = None
    quote_currency: str | None = None
    quote_timestamp: str | None = None
    observed_staleness_minutes: float | None = None

    history_ok: bool = False
    history_days: int = 0
    first_date: str | None = None
    last_date: str | None = None
    adjusted_close_available: bool | None = None

    resolved_symbol: str | None = None
    resolved_exchange: str | None = None
    resolved_isin: str | None = None
    resolution_method: str | None = None
    isin_check: str = "unavailable"      # match | mismatch | unavailable

    suspect: bool = False
    suspect_reason: str | None = None
    attempted: bool = True
    errors: list[str] = dataclasses.field(default_factory=list)

    @property
    def verdict(self) -> str:
        """SUSPECT deliberately outranks PASS.

        Data from the wrong listing is worse than no data: it is confidently
        wrong and it fails silently. It must never be counted as coverage.

        SKIPPED is kept distinct from FAIL because "the provider could not
        serve this" and "we ran out of quota before asking" support opposite
        conclusions about whether to pay for the provider.
        """
        if not self.attempted:
            return "SKIPPED"
        if self.suspect:
            return "SUSPECT"
        if self.quote_ok and self.history_ok:
            return "PASS"
        if self.quote_ok or self.history_days >= MIN_HISTORY_DAYS_USABLE:
            return "PARTIAL"
        return "FAIL"

    def flag(self, reason: str) -> None:
        self.suspect = True
        self.suspect_reason = reason


@dataclasses.dataclass
class AliasResult:
    provider: str
    isin: str
    symbol: str
    resolved_isin: str | None
    resolved_exchange: str | None
    agrees: bool
    note: str = ""


@dataclasses.dataclass
class CollisionResult:
    provider: str
    symbol: str
    expected_isin: str
    us_name: str
    naive_isin: str | None = None
    naive_symbol: str | None = None
    naive_exchange: str | None = None
    naive_currency: str | None = None
    outcome: str = "unknown"      # correct | wrong-fund | ambiguous | no-answer
    note: str = ""


@dataclasses.dataclass
class LivenessResult:
    provider: str
    dead_isin: str
    shadows_isin: str
    resolved_symbol: str | None = None
    history_days: int = 0
    last_date: str | None = None
    outcome: str = "unknown"      # correct-dead | serving-stale | ambiguous | no-answer
    note: str = ""


# ==========================================================================
# HTTP
# ==========================================================================

def get_json(url: str, params: dict[str, Any] | None = None,
             timeout: int = 25) -> tuple[int, Any]:
    try:
        resp = requests.get(url, params=params, timeout=timeout,
                            headers={"User-Agent": "portfolio-tracker-spike/0.2"})
    except requests.RequestException as exc:
        return 0, {"_transport_error": str(exc)}
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, {"_non_json_body": resp.text[:400]}


def looks_us(exchange: str | None) -> bool:
    if not exchange:
        return False
    e = exchange.upper()
    tokens = set(e.replace("/", " ").replace("-", " ").split()) | {e}
    return bool(tokens & US_EXCHANGE_MARKERS)


def looks_european(exchange: str | None) -> bool:
    if not exchange:
        return False
    e = exchange.upper()
    tokens = set(e.replace("/", " ").replace("-", " ").split()) | {e}
    return bool(tokens & EUROPEAN_EXCHANGE_HINTS)


def days_ago(iso_date: str | None) -> int | None:
    if not iso_date:
        return None
    try:
        return (dt.date.today() - dt.date.fromisoformat(iso_date[:10])).days
    except ValueError:
        return None


# ==========================================================================
# Provider probes
# ==========================================================================

class ProviderProbe:
    """Four primitives, and every check is built from them.

    Splitting resolution from fetching is what makes the identity, collision
    and liveness checks possible at all: they need to ask "what does this
    provider *think* this identifier is" without paying for a price series.

    Every call is counted against `daily_call_budget`. This is not bookkeeping
    for its own sake: FMP's free tier is 250 requests/day, and an unguarded run
    of all four checks over ten instruments with six listings each would spend
    that in a single pass and leave you locked out until tomorrow with a
    half-filled matrix. The spike stops at the budget and says so.
    """
    name = "abstract"
    documented_rate_limit = "unknown"
    batch_quotes = "unknown"
    documented_delay = "unknown"
    seconds_between_calls = 0.0
    daily_call_budget: int | None = None    # None = no hard documented cap
    calls_per_fetch = 2                     # for the pre-run cost estimate

    def __init__(self) -> None:
        self.calls = 0
        self.budget_hit = False

    def get(self, url: str, params: dict[str, Any] | None = None,
            timeout: int = 25) -> tuple[int, Any]:
        """Every HTTP call goes through here so the counter cannot drift."""
        self.calls += 1
        return get_json(url, params, timeout)

    def note_call(self, n: int = 1) -> None:
        """For yfinance, whose calls happen inside the library."""
        self.calls += n

    @property
    def over_budget(self) -> bool:
        if self.daily_call_budget is None:
            return False
        if self.calls >= self.daily_call_budget:
            self.budget_hit = True
        return self.calls >= self.daily_call_budget

    def available(self) -> tuple[bool, str]:
        return True, ""

    def measure_limits(self) -> dict[str, Any]:
        return {}

    def resolve_isin(self, isin: str) -> Resolution | None:
        """The production path: ISIN in, provider symbol out."""
        return None

    def resolve_symbol_naive(self, symbol: str) -> Resolution | None:
        """A bare ticker lookup with no venue hint, taking the provider's own
        top answer without our re-ranking. This is the collision test: it
        measures what you get if you trust the ticker, which is what a design
        keyed on ticker would do."""
        return None

    def fetch(self, probe: Probe, symbol: str, listing: Listing | None) -> None:
        """Populate quote and history fields on `probe` for a resolved symbol."""
        raise NotImplementedError

    def provider_symbol(self, symbol: str, listing: Listing) -> str:
        """Translate (symbol, venue) into this provider's symbol form."""
        return symbol

    def throttle(self) -> None:
        if self.seconds_between_calls:
            time.sleep(self.seconds_between_calls)


# --------------------------------------------------------------------------

class YFinanceProbe(ProviderProbe):
    """yfinance -- the coverage benchmark, not necessarily the production pick.

    Judgement call: the yfinance library rather than raw requests against
    Yahoo's chart endpoint. Rejected alternative: fewer dependencies, but Yahoo's
    cookie/crumb auth changes every few months, and a benchmark that breaks is
    not a benchmark. The cost is a dependency that is itself unofficial.

    Yahoo has no dependable ISIN search, so resolution here works the other way
    round: we build candidate symbols from the verified listing table and use
    `Ticker.isin` to *verify* rather than to discover. That asymmetry is a real
    finding -- it means yfinance can never resolve an instrument we have not
    already catalogued by hand.
    """
    name = "yfinance"
    documented_rate_limit = ("undocumented, unofficial; IP soft-throttled, "
                             "roughly a few hundred requests/hour before 429s")
    batch_quotes = ("history yes, via yf.download([...]) in one call; "
                    "quotes no -- yf.Tickers loops internally")
    documented_delay = "~15 min on European venues; Yahoo does not state it per-call"
    seconds_between_calls = 0.4
    daily_call_budget = 300      # self-imposed: Yahoo 429s an IP well before this
    calls_per_fetch = 1

    def available(self) -> tuple[bool, str]:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False, "pip install yfinance"
        return True, ""

    def provider_symbol(self, symbol: str, listing: Listing) -> str:
        return f"{symbol}{listing.venue.yahoo_suffix}"

    def _isin_of(self, ticker: Any) -> str | None:
        try:
            val = ticker.isin
        except Exception:
            return None
        if isinstance(val, str) and len(val) == 12 and val not in {"-", "N/A"}:
            return val.upper()
        return None

    def resolve_isin(self, isin: str) -> Resolution | None:
        import yfinance as yf
        search = getattr(yf, "Search", None)
        if search is None:
            return Resolution(method="isin", note="yfinance build has no Search API")
        self.throttle()
        self.note_call()
        try:
            quotes = search(isin, max_results=10).quotes or []
        except Exception as exc:
            return Resolution(method="isin", note=f"{type(exc).__name__}: {exc}")
        if not quotes:
            return None
        top = quotes[0]
        return Resolution(symbol=top.get("symbol"), exchange=top.get("exchange"),
                          currency=top.get("currency"), name=top.get("shortname"),
                          method="isin")

    def resolve_symbol_naive(self, symbol: str) -> Resolution | None:
        import yfinance as yf
        self.throttle()
        self.note_call()
        try:
            t = yf.Ticker(symbol)          # bare ticker: Yahoo picks the venue
            info = t.fast_info
            exch = getattr(info, "exchange", None)
            ccy = getattr(info, "currency", None)
            price = getattr(info, "last_price", None)
            if exch is None and price is None:
                return None
            return Resolution(symbol=symbol, exchange=exch, currency=ccy,
                              isin=self._isin_of(t), method="symbol-search")
        except Exception as exc:
            return Resolution(method="symbol-search", note=f"{type(exc).__name__}: {exc}")

    def fetch(self, probe: Probe, symbol: str, listing: Listing | None) -> None:
        import yfinance as yf
        start = (dt.date.today() - dt.timedelta(days=365 * HISTORY_YEARS + 10)).isoformat()
        self.throttle()
        self.note_call(2)                  # history + fast_info
        t = yf.Ticker(symbol)
        # auto_adjust=False so we can SEE whether an adjusted column exists.
        # Unadjusted prices read a distribution as a large negative return and
        # inflate measured volatility; these ETCs distribute, so this matters.
        hist = t.history(start=start, interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            probe.errors.append(f"{symbol}: no history")
            return
        probe.history_days = len(hist)
        probe.history_ok = len(hist) >= MIN_HISTORY_DAYS_PASS
        probe.first_date = str(hist.index[0].date())
        probe.last_date = str(hist.index[-1].date())
        probe.adjusted_close_available = "Adj Close" in hist.columns
        probe.resolved_isin = self._isin_of(t)
        try:
            info = t.fast_info
            probe.quote_currency = getattr(info, "currency", None)
            probe.resolved_exchange = getattr(info, "exchange", None)
            last = getattr(info, "last_price", None)
            if last is not None:
                probe.quote_ok = True
                probe.quote_price = float(last)
        except Exception as exc:
            probe.errors.append(f"fast_info: {type(exc).__name__}: {exc}")
        if not probe.quote_ok:
            # Last close, explicitly labelled. Never presented as a live quote.
            probe.quote_price = float(hist["Close"].iloc[-1])
            probe.quote_timestamp = probe.last_date
            probe.errors.append("no live quote; last close only")


# --------------------------------------------------------------------------

class TwelveDataProbe(ProviderProbe):
    """Twelve Data free tier.

    Uses MIC codes rather than the provider's own exchange labels wherever it
    accepts them, because MIC is the only venue identifier that is actually
    standardised and it removes a whole class of ambiguity.
    """
    name = "twelvedata"
    documented_rate_limit = "free: 8 credits/min, 800/day (verify via /api_usage)"
    batch_quotes = "yes -- /quote?symbol=A,B,C, but one credit per symbol"
    documented_delay = "free tier is delayed/EOD on European venues; real-time is paid"
    seconds_between_calls = 8.0     # 8 credits/minute
    daily_call_budget = 800         # free tier credits/day
    calls_per_fetch = 2

    BASE = "https://api.twelvedata.com"

    def __init__(self) -> None:
        super().__init__()
        self.key = os.environ.get("TWELVEDATA_API_KEY", "")

    def available(self) -> tuple[bool, str]:
        return bool(self.key), "set TWELVEDATA_API_KEY"

    def measure_limits(self) -> dict[str, Any]:
        code, body = self.get(f"{self.BASE}/api_usage", {"apikey": self.key})
        return {"http": code, "api_usage": body}

    def _search(self, query: str) -> list[dict[str, Any]]:
        self.throttle()
        code, body = self.get(f"{self.BASE}/symbol_search",
                              {"symbol": query, "outputsize": 30})
        if code != 200 or not isinstance(body, dict):
            return []
        return body.get("data") or []

    def resolve_isin(self, isin: str) -> Resolution | None:
        rows = self._search(isin)
        if not rows:
            return None
        # Prefer a European venue among the ISIN's listings.
        rows.sort(key=lambda r: 1 if looks_european(str(r.get("exchange", ""))) else 0,
                  reverse=True)
        top = rows[0]
        return Resolution(symbol=top.get("symbol"), exchange=top.get("exchange"),
                          currency=top.get("currency"), name=top.get("instrument_name"),
                          method="isin")

    def resolve_symbol_naive(self, symbol: str) -> Resolution | None:
        rows = self._search(symbol)
        if not rows:
            return None
        top = rows[0]                     # the provider's own ranking, unaided
        return Resolution(symbol=top.get("symbol"), exchange=top.get("exchange"),
                          currency=top.get("currency"), name=top.get("instrument_name"),
                          method="symbol-search")

    def fetch(self, probe: Probe, symbol: str, listing: Listing | None) -> None:
        params: dict[str, Any] = {"symbol": symbol, "apikey": self.key}
        if listing is not None:
            params["mic_code"] = listing.venue.twelvedata_mic

        self.throttle()
        code, quote = self.get(f"{self.BASE}/quote", params)
        if code == 200 and isinstance(quote, dict) and quote.get("close") is not None:
            probe.quote_ok = True
            probe.quote_price = float(quote["close"])
            probe.quote_currency = quote.get("currency")
            probe.resolved_exchange = quote.get("exchange") or probe.resolved_exchange
            ts = quote.get("timestamp")
            if ts:
                when = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)
                probe.quote_timestamp = when.isoformat()
                probe.observed_staleness_minutes = round(
                    (dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 60, 1)
        else:
            probe.errors.append(f"quote http={code} {str(quote)[:160]}")

        self.throttle()
        code, hist = self.get(f"{self.BASE}/time_series",
                              dict(params, interval="1day", outputsize=5000))
        values = hist.get("values") if isinstance(hist, dict) else None
        if code == 200 and values:
            probe.history_days = len(values)
            probe.last_date = values[0]["datetime"][:10]      # newest first
            probe.first_date = values[-1]["datetime"][:10]
            probe.history_ok = probe.history_days >= MIN_HISTORY_DAYS_PASS
            # The free tier's /time_series is unadjusted; `adjust` is a paid
            # parameter. If that holds, this provider cannot drive the risk
            # maths without corrupting returns around distributions.
            probe.adjusted_close_available = False
            probe.errors.append("free-tier /time_series is unadjusted -- confirm before trusting returns")
        else:
            probe.errors.append(f"time_series http={code} {str(hist)[:160]}")


# --------------------------------------------------------------------------

class FMPProbe(ProviderProbe):
    name = "fmp"
    documented_rate_limit = "free: 250 requests/day; some plans are US-only (verify in dashboard)"
    batch_quotes = "yes -- /v3/quote/A,B,C in one request"
    documented_delay = "free tier is end-of-day; intraday is paid"
    seconds_between_calls = 0.5
    daily_call_budget = 250         # the tightest cap of the four, by far
    calls_per_fetch = 3             # quote + history + profile(ISIN)

    BASE = "https://financialmodelingprep.com/api"

    def __init__(self) -> None:
        super().__init__()
        self.key = os.environ.get("FMP_API_KEY", "")

    def available(self) -> tuple[bool, str]:
        return bool(self.key), "set FMP_API_KEY"

    def provider_symbol(self, symbol: str, listing: Listing) -> str:
        return f"{symbol}{listing.venue.fmp_suffix}"

    def resolve_isin(self, isin: str) -> Resolution | None:
        code, body = self.get(f"{self.BASE}/v4/search/isin",
                              {"isin": isin, "apikey": self.key})
        if code != 200 or not isinstance(body, list) or not body:
            return None
        top = body[0]
        return Resolution(symbol=top.get("symbol"),
                          exchange=top.get("exchangeShortName") or top.get("stockExchange"),
                          currency=top.get("currency"), isin=isin,
                          name=top.get("name"), method="isin")

    def resolve_symbol_naive(self, symbol: str) -> Resolution | None:
        code, body = self.get(f"{self.BASE}/v3/search",
                              {"query": symbol, "limit": 30, "apikey": self.key})
        if code != 200 or not isinstance(body, list) or not body:
            return None
        top = body[0]
        return Resolution(symbol=top.get("symbol"),
                          exchange=top.get("exchangeShortName") or top.get("stockExchange"),
                          currency=top.get("currency"), name=top.get("name"),
                          method="symbol-search")

    def fetch(self, probe: Probe, symbol: str, listing: Listing | None) -> None:
        self.throttle()
        code, quote = self.get(f"{self.BASE}/v3/quote/{symbol}", {"apikey": self.key})
        if code == 200 and isinstance(quote, list) and quote and quote[0].get("price") is not None:
            row = quote[0]
            probe.quote_ok = True
            probe.quote_price = row.get("price")
            probe.resolved_exchange = row.get("exchange") or probe.resolved_exchange
            ts = row.get("timestamp")
            if ts:
                when = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)
                probe.quote_timestamp = when.isoformat()
                probe.observed_staleness_minutes = round(
                    (dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 60, 1)
        else:
            probe.errors.append(f"quote http={code} {str(quote)[:160]}")

        self.throttle()
        frm = (dt.date.today() - dt.timedelta(days=365 * HISTORY_YEARS + 10)).isoformat()
        code, hist = self.get(f"{self.BASE}/v3/historical-price-full/{symbol}",
                              {"from": frm, "to": dt.date.today().isoformat(),
                               "apikey": self.key})
        rows = hist.get("historical") if isinstance(hist, dict) else None
        if code == 200 and rows:
            probe.history_days = len(rows)
            probe.last_date = rows[0]["date"][:10]
            probe.first_date = rows[-1]["date"][:10]
            probe.history_ok = probe.history_days >= MIN_HISTORY_DAYS_PASS
            probe.adjusted_close_available = "adjClose" in rows[0]
        else:
            probe.errors.append(f"history http={code} {str(hist)[:160]}")

        # FMP exposes the ISIN on the profile endpoint, which is how we close
        # the loop and prove we fetched the fund we asked for. Guarded: on a
        # 250/day budget, spending a third call to identify a symbol that
        # returned no history at all is pure waste.
        if probe.history_days == 0 and not probe.quote_ok:
            return
        self.throttle()
        code, prof = self.get(f"{self.BASE}/v3/profile/{symbol}", {"apikey": self.key})
        if code == 200 and isinstance(prof, list) and prof:
            probe.resolved_isin = prof[0].get("isin")
            probe.quote_currency = prof[0].get("currency") or probe.quote_currency


# --------------------------------------------------------------------------

class EODHDProbe(ProviderProbe):
    """EODHD -- paid, but its search endpoint takes an ISIN directly, which is
    the only resolution model of the four that matches this architecture rather
    than fighting it. The point of the trial is to decide whether that plus its
    European coverage is worth paying for."""
    name = "eodhd"
    documented_rate_limit = "demo key: fixed symbol list; paid: 1000+/day (verify via /api/user)"
    batch_quotes = "yes -- /real-time/AAA.XETRA?s=BBB.LSE,CCC.MI"
    documented_delay = "15-20 min on European venues; EOD is T+1"
    seconds_between_calls = 1.0
    daily_call_budget = None        # plan-dependent; measured via /api/user
    calls_per_fetch = 2

    BASE = "https://eodhd.com/api"

    def __init__(self) -> None:
        super().__init__()
        self.key = os.environ.get("EODHD_API_KEY", "")

    def available(self) -> tuple[bool, str]:
        return bool(self.key), "set EODHD_API_KEY (trial or demo key is fine)"

    def measure_limits(self) -> dict[str, Any]:
        code, body = self.get(f"{self.BASE}/user", {"api_token": self.key, "fmt": "json"})
        if isinstance(body, dict):
            return {"http": code,
                    "daily_rate_limit": body.get("dailyRateLimit"),
                    "requests_used_today": body.get("apiRequests"),
                    "subscription": body.get("subscriptionType")}
        return {"http": code, "body": str(body)[:200]}

    def provider_symbol(self, symbol: str, listing: Listing) -> str:
        return f"{symbol}.{listing.venue.eodhd_code}"

    def _search(self, query: str) -> list[dict[str, Any]]:
        self.throttle()
        code, body = self.get(f"{self.BASE}/search/{query}",
                              {"api_token": self.key, "fmt": "json", "limit": 30})
        return body if code == 200 and isinstance(body, list) else []

    def resolve_isin(self, isin: str) -> Resolution | None:
        rows = self._search(isin)
        if not rows:
            return None
        rows.sort(key=lambda r: 1 if looks_european(str(r.get("Exchange", ""))) else 0,
                  reverse=True)
        top = rows[0]
        return Resolution(symbol=f"{top.get('Code')}.{top.get('Exchange')}",
                          exchange=top.get("Exchange"), currency=top.get("Currency"),
                          isin=top.get("ISIN") or isin, name=top.get("Name"),
                          method="isin")

    def resolve_symbol_naive(self, symbol: str) -> Resolution | None:
        rows = self._search(symbol)
        if not rows:
            return None
        top = rows[0]
        return Resolution(symbol=f"{top.get('Code')}.{top.get('Exchange')}",
                          exchange=top.get("Exchange"), currency=top.get("Currency"),
                          isin=top.get("ISIN"), name=top.get("Name"),
                          method="symbol-search")

    def fetch(self, probe: Probe, symbol: str, listing: Listing | None) -> None:
        self.throttle()
        code, quote = self.get(f"{self.BASE}/real-time/{symbol}",
                               {"api_token": self.key, "fmt": "json"})
        if code == 200 and isinstance(quote, dict) and quote.get("close") not in (None, "NA"):
            probe.quote_ok = True
            probe.quote_price = float(quote["close"])
            ts = quote.get("timestamp")
            if isinstance(ts, (int, float)):
                when = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc)
                probe.quote_timestamp = when.isoformat()
                probe.observed_staleness_minutes = round(
                    (dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 60, 1)
        else:
            probe.errors.append(f"real-time http={code} {str(quote)[:160]}")

        self.throttle()
        frm = (dt.date.today() - dt.timedelta(days=365 * HISTORY_YEARS + 10)).isoformat()
        code, hist = self.get(f"{self.BASE}/eod/{symbol}",
                              {"api_token": self.key, "fmt": "json", "period": "d", "from": frm})
        if code == 200 and isinstance(hist, list) and hist:
            probe.history_days = len(hist)
            probe.first_date = hist[0]["date"][:10]
            probe.last_date = hist[-1]["date"][:10]
            probe.history_ok = probe.history_days >= MIN_HISTORY_DAYS_PASS
            probe.adjusted_close_available = "adjusted_close" in hist[0]
        else:
            probe.errors.append(f"eod http={code} {str(hist)[:160]}")


PROVIDERS: dict[str, Callable[[], ProviderProbe]] = {
    "yfinance": YFinanceProbe,
    "twelvedata": TwelveDataProbe,
    "fmp": FMPProbe,
    "eodhd": EODHDProbe,
}


# ==========================================================================
# Check 1: coverage
# ==========================================================================

def verify(probe: Probe, inst: TestInstrument) -> None:
    """Decide whether we fetched the fund we asked for.

    Three tests, strongest first. The ISIN round-trip is the only one that
    proves identity; the other two are circumstantial and only used when the
    provider does not return an ISIN at all.
    """
    if probe.resolved_isin:
        if probe.resolved_isin.upper() == inst.isin.upper():
            probe.isin_check = "match"
        else:
            probe.isin_check = "mismatch"
            probe.flag(f"provider returned ISIN {probe.resolved_isin}, expected {inst.isin} "
                       f"-- this is a DIFFERENT INSTRUMENT")
            return
    else:
        probe.isin_check = "unavailable"

    if looks_us(probe.resolved_exchange):
        probe.flag(f"resolved to a US listing ({probe.resolved_exchange}); "
                   f"likely the colliding US product, not {inst.isin}")
        return

    if probe.isin_check == "unavailable":
        base = (probe.resolved_symbol or "").split(".")[0].upper()
        if base and base not in {s.upper() for s in inst.symbols}:
            probe.flag(f"symbol {probe.resolved_symbol!r} is not a known listing of "
                       f"{inst.isin}, and no ISIN was returned to check it against")
            return
        if probe.resolved_exchange and not looks_european(probe.resolved_exchange):
            probe.flag(f"unrecognised exchange {probe.resolved_exchange!r}, "
                       f"and no ISIN was returned to check it against")


def safe_resolve(fn, arg: str) -> Resolution | None:
    """A provider raising mid-run must not cost us the whole matrix."""
    try:
        return fn(arg)
    except Exception as exc:
        return Resolution(note=f"{type(exc).__name__}: {exc}")


def run_coverage(prov: ProviderProbe, instruments: tuple[TestInstrument, ...],
                 max_fallbacks: int = 4) -> list[Probe]:
    out: list[Probe] = []
    for inst in instruments:
        best: Probe | None = None

        # Candidate 1: the production path -- resolve the ISIN.
        candidates: list[tuple[str, str, Listing | None]] = []
        res = safe_resolve(prov.resolve_isin, inst.isin)
        if res and res.symbol:
            candidates.append(("isin", res.symbol, None))

        # Candidates 2..n: known listings, EUR primary first. This is a
        # fallback, and needing it is itself a finding: it means the provider
        # cannot resolve an instrument we have not catalogued by hand.
        #
        # Capped, because the fallback is where quota goes to die: ten
        # instruments times six listings times three calls is 180 requests
        # against a 250/day cap, for one check.
        for listing in inst.ordered_listings()[:max_fallbacks]:
            candidates.append(("listing", prov.provider_symbol(listing.symbol, listing), listing))

        for method, symbol, listing in candidates:
            if prov.over_budget:
                break
            p = Probe(prov.name, inst.isin, inst.name, inst.asset_class, inst.base_currency)
            p.resolution_method = method
            p.resolved_symbol = symbol
            if method == "isin" and res:
                p.resolved_exchange = res.exchange
                p.resolved_isin = res.isin
                p.quote_currency = res.currency
            elif listing is not None:
                p.resolved_exchange = listing.mic
            try:
                prov.fetch(p, symbol, listing)
            except Exception as exc:
                p.errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            verify(p, inst)
            # Keep the deepest usable series: for a cross-listed fund the
            # longest history is the one worth putting in the risk model.
            if best is None or (not best.history_ok and p.history_ok) or \
               (best.history_ok == p.history_ok and p.history_days > best.history_days):
                best = p
            if p.history_ok and p.quote_ok and not p.suspect:
                break

        if best is None:                 # budget ran out before any attempt
            best = Probe(prov.name, inst.isin, inst.name, inst.asset_class,
                         inst.base_currency)
            best.errors.append("not attempted: provider call budget exhausted")
        out.append(best)
        if prov.over_budget:
            # Record the rest explicitly. "-" in the matrix would be
            # indistinguishable from "this provider failed", and the two mean
            # very different things when you are choosing a provider.
            for skipped in instruments[len(out):]:
                stub = Probe(prov.name, skipped.isin, skipped.name,
                             skipped.asset_class, skipped.base_currency)
                stub.attempted = False
                stub.errors.append("NOT ATTEMPTED: provider call budget exhausted")
                out.append(stub)
            break
    return out


# ==========================================================================
# Check 2: symbol identity -- do all of an instrument's symbols agree on ISIN?
# ==========================================================================

def run_identity(prov: ProviderProbe, groups: list[tuple[str, tuple[str, ...]]]) -> list[AliasResult]:
    out: list[AliasResult] = []
    for isin, symbols in groups:
        for sym in symbols:
            if prov.over_budget:
                return out
            res = safe_resolve(prov.resolve_symbol_naive, sym)
            if res is None:
                out.append(AliasResult(prov.name, isin, sym, None, None, False,
                                       "no answer"))
                continue
            if not res.isin:
                out.append(AliasResult(prov.name, isin, sym, None, res.exchange, False,
                                       "provider returns no ISIN -- identity unverifiable"))
                continue
            out.append(AliasResult(prov.name, isin, sym, res.isin, res.exchange,
                                   res.isin.upper() == isin.upper()))
    return out


# ==========================================================================
# Check 3: ticker collisions with US-listed products
# ==========================================================================

def run_collision(prov: ProviderProbe) -> list[CollisionResult]:
    out: list[CollisionResult] = []
    for col in COLLISIONS:
        r = CollisionResult(prov.name, col.symbol, col.our_isin, col.us_name)
        if prov.over_budget:
            return out
        res = safe_resolve(prov.resolve_symbol_naive, col.symbol)
        if res is None:
            r.outcome = "no-answer"
            out.append(r)
            continue
        r.naive_symbol, r.naive_exchange = res.symbol, res.exchange
        r.naive_currency, r.naive_isin = res.currency, res.isin

        if res.isin:
            r.outcome = "correct" if res.isin.upper() == col.our_isin.upper() else "wrong-fund"
            if r.outcome == "wrong-fund":
                r.note = f"returned {res.isin} ({res.name or '?'})"
        elif looks_us(res.exchange):
            r.outcome = "wrong-fund"
            r.note = f"US venue {res.exchange} -- almost certainly {col.us_name}"
        elif looks_european(res.exchange):
            r.outcome = "ambiguous"
            r.note = "European venue but no ISIN returned; cannot confirm"
        else:
            r.outcome = "ambiguous"
            r.note = f"exchange {res.exchange!r}, no ISIN"
        out.append(r)
    return out


# ==========================================================================
# Check 4: liveness -- liquidated ISINs must return nothing
# ==========================================================================

def run_liveness(prov: ProviderProbe) -> list[LivenessResult]:
    out: list[LivenessResult] = []
    for dead in DEAD_INSTRUMENTS:
        r = LivenessResult(prov.name, dead.isin, dead.shadows_isin)
        if prov.over_budget:
            return out
        res = safe_resolve(prov.resolve_isin, dead.isin)
        if res is None or not res.symbol:
            r.outcome = "correct-dead"
            r.note = "no resolution -- provider does not carry the liquidated line"
            out.append(r)
            continue
        r.resolved_symbol = res.symbol
        scratch = Probe(prov.name, dead.isin, dead.name, "DEAD", "-")
        try:
            prov.fetch(scratch, res.symbol, None)
        except Exception as exc:
            scratch.errors.append(f"{type(exc).__name__}: {exc}")
        r.history_days, r.last_date = scratch.history_days, scratch.last_date
        age = days_ago(scratch.last_date)
        if scratch.history_days == 0:
            r.outcome = "correct-dead"
            r.note = "resolves, but returns no price history"
        elif age is not None and age <= DEAD_LINE_STALE_DAYS:
            r.outcome = "serving-stale"
            r.note = (f"quoting as of {scratch.last_date} ({age}d ago) for a liquidated "
                      f"line -- this would silently enter the covariance matrix")
        else:
            r.outcome = "correct-dead"
            r.note = f"history ends {scratch.last_date} ({age}d ago), consistent with liquidation"
        out.append(r)
    return out


# ==========================================================================
# Reporting
# ==========================================================================

GLYPH = {"PASS": "PASS", "PARTIAL": "PART", "SUSPECT": "SUSP", "FAIL": "FAIL",
         "SKIPPED": "skip"}
RULE = "=" * 108


def print_detail(provider: str, probes: list[Probe]) -> None:
    print(f"\n{RULE}\nPROVIDER: {provider}  --  coverage detail\n{RULE}")
    head = (f"{'ISIN':<14}{'cls':<5}{'symbol':<14}{'via':<9}{'isin?':<13}"
            f"{'quote':<7}{'days':>6} {'first':<11}{'last':<11}"
            f"{'base':<6}{'quote$':<7}{'exchange':<10}verdict")
    print(head + "\n" + "-" * len(head))
    for p in probes:
        print(f"{p.isin:<14}{p.asset_class:<5}{(p.resolved_symbol or '-'):<14}"
              f"{(p.resolution_method or '-'):<9}{p.isin_check:<13}"
              f"{('yes' if p.quote_ok else 'no'):<7}{p.history_days:>6} "
              f"{(p.first_date or '-'):<11}{(p.last_date or '-'):<11}"
              f"{p.base_currency:<6}{(p.quote_currency or '-'):<7}"
              f"{(p.resolved_exchange or '-'):<10}{p.verdict}")
    for p in probes:
        if p.suspect:
            print(f"  ! {p.isin} {p.name[:44]}: {p.suspect_reason}")
        if p.adjusted_close_available is False:
            print(f"  ~ {p.isin}: NO ADJUSTED CLOSE -- distributions would corrupt returns")
        for e in p.errors:
            print(f"    - {p.isin}: {e}")


def print_matrix(results: dict[str, list[Probe]]) -> None:
    providers = list(results)
    print(f"\n{RULE}")
    print(f"CHECK 1  COVERAGE MATRIX   PASS = quote + >={MIN_HISTORY_DAYS_PASS}d history, "
          f"identity confirmed")
    print("SUSP = data returned from a listing that looks wrong. Counted as a failure.")
    print("skip = not attempted, provider call budget exhausted. NOT a provider failure.")
    print(RULE)
    w = max(12, max((len(p) for p in providers), default=12) + 2)
    print(f"{'instrument':<34}{'cls':<5}" + "".join(f"{p:<{w}}" for p in providers))
    print("-" * (39 + w * len(providers)))

    for asset_class in ("ETF", "ETC"):
        group = [i for i in INSTRUMENTS if i.asset_class == asset_class]
        for inst in group:
            label = f"{inst.isin} {inst.name[:19]}"
            row = f"{label:<34}{inst.asset_class:<5}"
            for prov in providers:
                pr = next((x for x in results[prov] if x.isin == inst.isin), None)
                row += f"{(GLYPH[pr.verdict] if pr else '-'):<{w}}"
            print(row)
        # ETCs are collateralised notes, not UCITS funds. Several providers
        # classify them separately or omit them, so a subtotal split by class
        # is the difference between "works" and "works for the easy half".
        sub = f"{f'  {asset_class} subtotal':<34}{'':<5}"
        for prov in providers:
            n = sum(1 for x in results[prov]
                    if x.verdict == "PASS" and BY_ISIN[x.isin].asset_class == asset_class)
            sub += f"{f'{n}/{len(group)}':<{w}}"
        print(sub)
        print("-" * (39 + w * len(providers)))

    total = f"{'TOTAL PASS':<34}{'':<5}"
    for prov in providers:
        n = sum(1 for x in results[prov] if x.verdict == "PASS")
        total += f"{f'{n}/{len(INSTRUMENTS)}':<{w}}"
    print(total)


def print_identity(results: dict[str, list[AliasResult]]) -> None:
    print(f"\n{RULE}")
    print("CHECK 2  SYMBOL IDENTITY   every symbol of a fund must resolve to its ISIN")
    print("WDEF and EUDF are ONE fund. Two different answers is the failure ISIN keying prevents.")
    print(RULE)
    for prov, rows in results.items():
        if not rows:
            continue
        print(f"\n{prov}")
        by_isin: dict[str, list[AliasResult]] = {}
        for r in rows:
            by_isin.setdefault(r.isin, []).append(r)
        for isin, group in by_isin.items():
            inst = BY_ISIN.get(isin)
            agree = sum(1 for r in group if r.agrees)
            status = "OK" if agree == len(group) else "BROKEN"
            print(f"  {status:<7}{isin}  {inst.name[:42] if inst else ''}")
            for r in group:
                mark = "ok " if r.agrees else "BAD"
                print(f"      {mark} {r.symbol:<8}-> {(r.resolved_isin or '(no ISIN)'):<16}"
                      f"{(r.resolved_exchange or '-'):<12}{r.note}")


def print_collisions(results: dict[str, list[CollisionResult]]) -> None:
    print(f"\n{RULE}")
    print("CHECK 3  TICKER COLLISIONS   bare-ticker lookup, provider's own top answer")
    print("This measures what a ticker-keyed design would have given you.")
    print(RULE)
    danger = {c.symbol: c.danger for c in COLLISIONS if c.danger}
    for prov, rows in results.items():
        if not rows:
            continue
        print(f"\n{prov}")
        for r in rows:
            mark = {"correct": "ok  ", "wrong-fund": "WRONG", "ambiguous": "?   ",
                    "no-answer": "-   ", "unknown": "?   "}[r.outcome]
            print(f"  {mark} {r.symbol:<6}-> {(r.naive_symbol or '-'):<14}"
                  f"{(r.naive_exchange or '-'):<12}{(r.naive_currency or '-'):<5}"
                  f"{(r.naive_isin or '(no ISIN)'):<14}")
            print(f"        want {r.expected_isin}   collides with {r.us_name}")
            if r.note:
                print(f"        {r.note}")
            if r.symbol in danger and r.outcome != "correct":
                print(f"        >>> {danger[r.symbol]}")


def print_liveness(results: dict[str, list[LivenessResult]]) -> None:
    print(f"\n{RULE}")
    print("CHECK 4  LIQUIDATED LINES   returning NOTHING is the correct answer")
    print(RULE)
    for prov, rows in results.items():
        if not rows:
            continue
        print(f"\n{prov}")
        for r in rows:
            mark = {"correct-dead": "ok   ", "serving-stale": "STALE",
                    "ambiguous": "?    ", "no-answer": "-    ", "unknown": "?    "}[r.outcome]
            dead = next((d for d in DEAD_INSTRUMENTS if d.isin == r.dead_isin), None)
            print(f"  {mark} {r.dead_isin}  shadows {r.shadows_isin}  "
                  f"{r.history_days}d, last {r.last_date or '-'}")
            print(f"        {dead.name if dead else ''}")
            print(f"        {r.note}")


def print_mechanics(live: dict[str, ProviderProbe], measured: dict[str, dict],
                    coverage: dict[str, list[Probe]]) -> None:
    print(f"\n{RULE}")
    print("CHECK 5  MECHANICS   rate limits, batching, adjusted closes, currency")
    print(RULE)
    for name, prov in live.items():
        print(f"\n{name}")
        print(f"  rate limit : {prov.documented_rate_limit}")
        print(f"  batching   : {prov.batch_quotes}")
        print(f"  delay      : {prov.documented_delay}")
        cap = prov.daily_call_budget
        used = f"{prov.calls} calls made" + (f" of {cap}/day budget" if cap else "")
        print(f"  usage      : {used}"
              + ("   *** BUDGET EXHAUSTED -- results are incomplete ***"
                 if prov.budget_hit else ""))
        if measured.get(name):
            print(f"  measured   : {json.dumps(measured[name])[:300]}")
        probes = coverage.get(name, [])
        if probes:
            adj = [p for p in probes if p.adjusted_close_available is False]
            if adj:
                print(f"  ADJUSTED   : missing on {len(adj)}/{len(probes)} instruments "
                      f"-- returns would be corrupted by distributions")
            stale = [p.observed_staleness_minutes for p in probes
                     if p.observed_staleness_minutes is not None]
            if stale:
                print(f"  observed staleness: min {min(stale)}m, max {max(stale)}m "
                      f"(measured, not documented)")
            fx = [p for p in probes
                  if p.quote_currency and p.quote_currency.upper()[:3] != p.base_currency.upper()]
            print(f"  FX needed  : {len(fx)}/{len(probes)} instruments quote in a currency "
                  f"other than their base")


def print_symbol_map(results: dict[str, list[Probe]]) -> None:
    """The provider_symbols mapping, keyed by ISIN. This is the bridge between
    ISIN-as-primary-key and each provider's own symbol, and it is the artefact
    that gets pasted into instruments.csv."""
    print(f"\n{RULE}\nRESOLVED provider_symbols  (ISIN -> provider -> symbol; "
          f"suspect resolutions excluded)\n{RULE}")
    mapping: dict[str, dict[str, str]] = {}
    for prov, probes in results.items():
        for p in probes:
            if p.resolved_symbol and not p.suspect and p.history_days > 0:
                mapping.setdefault(p.isin, {})[prov] = p.resolved_symbol
    print(json.dumps(mapping, indent=2))


# ==========================================================================
# Entry point
# ==========================================================================

ALL_CHECKS = ("coverage", "identity", "collision", "liveness")


def estimate_calls(prov: ProviderProbe, n_instruments: int, n_alias_symbols: int,
                   checks: set[str], max_fallbacks: int) -> tuple[int, int]:
    """Worst- and best-case call count for a run, before spending anything.

    Worth printing up front because two of these free tiers are tight enough
    that a full run can exhaust a day's quota, and finding that out halfway
    through leaves you with a half-filled matrix and no way to finish it until
    tomorrow.
    """
    cpf = prov.calls_per_fetch
    worst = best = 0
    if "coverage" in checks:
        worst += n_instruments * (1 + (1 + max_fallbacks) * cpf)
        best += n_instruments * (1 + cpf)          # ISIN resolves first time
    if "identity" in checks:
        worst += n_alias_symbols
        best += n_alias_symbols
    if "collision" in checks:
        worst += len(COLLISIONS)
        best += len(COLLISIONS)
    if "liveness" in checks:
        worst += len(DEAD_INSTRUMENTS) * (1 + cpf)
        best += len(DEAD_INSTRUMENTS)              # dead ISIN does not resolve
    return best, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers", default=",".join(PROVIDERS))
    ap.add_argument("--checks", default=",".join(ALL_CHECKS),
                    help="subset of: " + ", ".join(ALL_CHECKS))
    ap.add_argument("--isins", default="", help="comma-separated ISINs; default all")
    ap.add_argument("--alias-scope", default="collisions", choices=("collisions", "all"),
                    help="identity check: only funds with colliding tickers (cheap), or every fund")
    ap.add_argument("--max-fallbacks", type=int, default=4,
                    help="max listings tried per instrument when ISIN resolution "
                         "fails; the main driver of quota use (default 4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the estimated call cost per provider and stop")
    ap.add_argument("--offline", action="store_true",
                    help="replay the last saved run; spends no quota")
    args = ap.parse_args()

    checks = {c.strip() for c in args.checks.split(",")}

    if args.offline:
        saved_files = sorted(RESULTS_DIR.glob("run-*.json"))
        if not saved_files:
            print("no saved run in spike/results/")
            return 1
        saved = json.loads(saved_files[-1].read_text())
        cov = {k: [Probe(**r) for r in v] for k, v in saved.get("coverage", {}).items()}
        idn = {k: [AliasResult(**r) for r in v] for k, v in saved.get("identity", {}).items()}
        col = {k: [CollisionResult(**r) for r in v] for k, v in saved.get("collision", {}).items()}
        liv = {k: [LivenessResult(**r) for r in v] for k, v in saved.get("liveness", {}).items()}
        for prov, probes in cov.items():
            print_detail(prov, probes)
        if cov:
            print_matrix(cov)
        if idn:
            print_identity(idn)
        if col:
            print_collisions(col)
        if liv:
            print_liveness(liv)
        if cov:
            print_symbol_map(cov)
        print(f"\n(offline replay of {saved_files[-1].name}, run at {saved.get('run_at')})")
        return 0

    wanted = tuple(i for i in INSTRUMENTS
                   if not args.isins or i.isin in args.isins.split(","))
    groups = alias_groups(collisions_only=(args.alias_scope == "collisions"))

    coverage: dict[str, list[Probe]] = {}
    identity: dict[str, list[AliasResult]] = {}
    collision: dict[str, list[CollisionResult]] = {}
    liveness: dict[str, list[LivenessResult]] = {}
    live: dict[str, ProviderProbe] = {}
    measured: dict[str, dict] = {}

    for name in (n.strip() for n in args.providers.split(",")):
        if name not in PROVIDERS:
            print(f"unknown provider {name!r}, skipping")
            continue
        prov = PROVIDERS[name]()
        ok, why = prov.available()
        if not ok:
            print(f"[skip] {name}: {why}")
            continue
        n_alias = sum(len(sy) for _, sy in groups)
        lo, hi = estimate_calls(prov, len(wanted), n_alias, checks, args.max_fallbacks)
        cap = prov.daily_call_budget
        budget_note = f", budget {cap}/day" if cap else ""
        warn = "  <-- MAY EXHAUST DAILY QUOTA" if cap and hi > cap else ""
        print(f"[cost] {name}: {lo}-{hi} calls{budget_note}{warn}")
        if args.dry_run:
            continue

        live[name] = prov
        measured[name] = prov.measure_limits()
        print(f"[run ] {name}  throttle={prov.seconds_between_calls}s/call, "
              f"est. {hi * prov.seconds_between_calls / 60:.0f} min worst case")

        if "coverage" in checks:
            print(f"       coverage: {len(wanted)} instruments")
            coverage[name] = run_coverage(prov, wanted, args.max_fallbacks)
            for p in coverage[name]:
                print(f"         {p.isin} {p.asset_class}  {p.verdict}")
        if "identity" in checks:
            print(f"       identity: {sum(len(s) for _, s in groups)} symbols")
            identity[name] = run_identity(prov, groups)
        if "collision" in checks:
            print(f"       collision: {len(COLLISIONS)} tickers")
            collision[name] = run_collision(prov)
        if "liveness" in checks:
            print(f"       liveness: {len(DEAD_INSTRUMENTS)} liquidated ISINs")
            liveness[name] = run_liveness(prov)

    if args.dry_run:
        print("\n--dry-run: nothing was fetched. Drop the flag to run for real.")
        return 0

    if not live:
        print("\nNo provider ran. Set at least one API key, or install yfinance.")
        return 1

    for prov, probes in coverage.items():
        print_detail(prov, probes)
    if coverage:
        print_matrix(coverage)
    if identity:
        print_identity(identity)
    if collision:
        print_collisions(collision)
    if liveness:
        print_liveness(liveness)
    print_mechanics(live, measured, coverage)
    if coverage:
        print_symbol_map(coverage)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"run-{stamp}.json"
    out.write_text(json.dumps({
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "coverage": {k: [dataclasses.asdict(p) for p in v] for k, v in coverage.items()},
        "identity": {k: [dataclasses.asdict(p) for p in v] for k, v in identity.items()},
        "collision": {k: [dataclasses.asdict(p) for p in v] for k, v in collision.items()},
        "liveness": {k: [dataclasses.asdict(p) for p in v] for k, v in liveness.items()},
        "measured_limits": measured,
    }, indent=2))
    print(f"\nRaw results -> {out}")
    print("Replay without spending quota:  python spike/check_providers.py --offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
