"""Session state, caching, and the data access every view shares.

Streamlit's fatal trap for this application
-------------------------------------------
Every widget interaction re-runs the whole script from the top. Uncached, that
means changing a dropdown refetches every price -- which burns the free-tier
rate limit within minutes and makes the interface feel broken for reasons the
user cannot see.

So every network call in this application goes through a `st.cache_data`
wrapper here, with a TTL. Nothing else fetches. The refresh button clears these
caches explicitly and is the only thing that does.

The resolver is cached too, and needs it more than prices do: probing candidate
listings is one call per candidate per instrument, so an uncached rerun of the
Instruments view could be dozens of calls from a single keystroke.

State lives in `st.session_state` so it survives reruns. If a change here seems
to need `st.rerun()`, the state model is wrong -- Streamlit reruns on its own
after a widget changes, and forcing another pass usually means two sources of
truth for the same value.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from decimal import Decimal

import pandas as pd
import streamlit as st

from ..core.money import BASE_CURRENCY, FxTable, Money
from ..core.positions import PriceQuote
from ..data.providers.yahoo import YahooProvider
from ..data.store import DataMode, DataStore

CACHE_TTL_SECONDS = 15 * 60          # matches the on-disk quote TTL
DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data_store"


# --------------------------------------------------------------------------
# Mode
# --------------------------------------------------------------------------

def current_mode() -> DataMode:
    """Seed or user, held in session state so it survives every rerun."""
    if "data_mode" not in st.session_state:
        st.session_state.data_mode = DataMode.SEED
    return st.session_state.data_mode


def set_mode(mode: DataMode) -> None:
    if st.session_state.get("data_mode") != mode:
        st.session_state.data_mode = mode
        # The stores differ, so cached values from the other one are wrong.
        st.cache_data.clear()


def store() -> DataStore:
    return DataStore.open(current_mode(), root=DATA_ROOT)


def mode_banner() -> None:
    """Unmissable, on every view.

    Demo data shown as real, and real data shown while the user believes they
    are in demo, are both bad enough to justify a coloured block rather than a
    caption. `st.error` is used for demo mode not because it is an error but
    because it is the loudest thing Streamlit offers.
    """
    if current_mode() is DataMode.SEED:
        st.error("**DEMO DATA** - seed instruments and a synthetic ledger. "
                 "These are not your positions.", icon=":material/science:")
    else:
        st.success(f"**LIVE DATA** - your instruments and ledger from "
                   f"`{store().directory}`", icon=":material/lock:")


# --------------------------------------------------------------------------
# Cached fetches. Nothing outside this module touches the network.
# --------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_history(symbol: str) -> pd.Series:
    """Adjusted close series. Cached: this is the expensive call."""
    return YahooProvider().history(symbol)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_quote(symbol: str) -> dict:
    """Latest quote as a plain dict, because Streamlit caches by value."""
    q = YahooProvider().quote(symbol)
    return {"symbol": q.symbol, "price": str(q.price), "currency": q.currency,
            "as_of": q.as_of.isoformat(), "source": q.source,
            "delay_minutes": q.delay_minutes, "is_stale": q.is_stale}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def probe_symbol(symbol: str, lookback: int = 252) -> dict:
    """Cached listing probe. Resolution fans out to one call per candidate."""
    p = YahooProvider().probe(symbol, lookback)
    return {"symbol": p.symbol, "ok": p.ok, "exchange": p.exchange,
            "currency": p.currency, "name": p.name,
            "observations": p.observations,
            "first_date": p.first_date.isoformat() if p.first_date else None,
            "last_date": p.last_date.isoformat() if p.last_date else None,
            "adjusted": p.adjusted, "error": p.error}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_figi_listings(isin: str) -> list[dict]:
    """Cached OpenFIGI lookup. Roughly 25 requests/minute unauthenticated."""
    from ..data.providers.openfigi import OpenFIGIProvider
    return [dataclass_to_dict(l) for l in OpenFIGIProvider().listings_for_isin(isin)]


def dataclass_to_dict(obj) -> dict:
    import dataclasses
    return dataclasses.asdict(obj)


def refresh_all() -> None:
    """The only thing that refetches. Wired to the one refresh button.

    Prices are never auto-polled: free-tier rate limits make polling fatal, and
    a page that silently refetches on every interaction is indistinguishable
    from one that is broken.
    """
    st.cache_data.clear()
    st.session_state.last_refresh = dt.datetime.now(dt.timezone.utc)


def last_refresh_text() -> str:
    when = st.session_state.get("last_refresh")
    return f"prices last fetched {when:%Y-%m-%d %H:%M UTC}" if when else \
        "prices not yet fetched this session"


# --------------------------------------------------------------------------
# FX
# --------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fx_rates_raw() -> dict[str, str]:
    """Spot rates into the reporting currency.

    Deliberately a small fixed set fetched once: this application converts a
    handful of currencies, and a full daily FX history is a v2 concern (it is
    what would let a long foreign listing be converted into a valid EUR series).
    """
    out: dict[str, str] = {}
    provider = YahooProvider()
    for code, pair in (("USD", "USDEUR=X"), ("GBP", "GBPEUR=X"), ("CHF", "CHFEUR=X")):
        try:
            out[code] = str(provider.quote(pair).price)
        except Exception:
            continue
    return out


def fx_table(as_of: dt.date | None = None) -> FxTable:
    """An FxTable built from the cached spot rates.

    A missing rate is left missing rather than defaulted: `core` raises
    `MissingRate` and the view reports the holding as unpriced, which is the
    correct outcome. Defaulting to parity would silently value USD as EUR.
    """
    day = as_of or dt.date.today()
    table = FxTable()
    for code, rate in fx_rates_raw().items():
        table.add(code, BASE_CURRENCY, day, rate)
    return table


def quote_from_cache(payload: dict) -> PriceQuote:
    """Rebuild a PriceQuote from the cached dict."""
    raw = payload["as_of"]
    as_of = (dt.datetime.fromisoformat(raw) if "T" in raw
             else dt.date.fromisoformat(raw))
    return PriceQuote(
        price=Money(Decimal(payload["price"]), payload["currency"]),
        as_of=as_of, source=payload["source"],
        delay_minutes=payload["delay_minutes"], is_stale=payload["is_stale"])


def load_quotes(instruments: dict, provider_name: str = "yfinance"
                ) -> tuple[dict[str, PriceQuote], list[str]]:
    """Quotes for every instrument with a stored symbol, plus failures.

    Failures are returned rather than raised: one unreachable symbol must not
    blank the whole Holdings view, and the affected row says so itself.
    """
    quotes: dict[str, PriceQuote] = {}
    failures: list[str] = []
    for isin, inst in instruments.items():
        symbol = inst.provider_symbols.get(provider_name)
        if not symbol:
            failures.append(f"{inst.name}: no {provider_name} symbol stored")
            continue
        try:
            quotes[isin] = quote_from_cache(fetch_quote(symbol))
        except Exception as exc:
            failures.append(f"{inst.name} ({symbol}): {exc}")
    return quotes, failures
