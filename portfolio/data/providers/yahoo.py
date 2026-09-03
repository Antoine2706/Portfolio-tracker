"""yfinance price provider. The v1 choice, per the 3 September 2026 matrix.

Judgement call recorded: the yfinance library rather than raw requests against
Yahoo's chart endpoint. The matrix run confirmed why -- a raw request returned
HTTP 429 without cookie and crumb handling, which yfinance manages internally.
That same machinery is the fragility that breaks Yahoo wrappers every few
months, which is why this sits behind `MarketDataProvider` and why the venue
map is stored rather than recomputed. If this breaks, EODHD's ISIN-native
search is the natural replacement and only this file changes.

Yahoo cannot resolve an ISIN. That is not worked around here: identity is
`OpenFIGIProvider`'s job.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd

from ..provider import ListingProbe, MarketDataProvider, ProviderError, Quote

__all__ = ["YahooProvider"]


class YahooProvider(MarketDataProvider):
    name = "yfinance"
    # Yahoo does not state a per-call delay. 15 minutes is the conventional
    # figure for European venues and is displayed as an estimate, never as fact.
    documented_delay_minutes = 15

    def __init__(self, session=None) -> None:
        self._session = session

    def _ticker(self, symbol: str):
        try:
            import yfinance as yf
        except ImportError as exc:                      # pragma: no cover
            raise ProviderError(self.name, "yfinance is not installed") from exc
        return yf.Ticker(symbol)

    def probe(self, symbol: str, lookback_days: int = 252) -> ListingProbe:
        """Cheap look at a symbol. Never raises: a failed probe is a result."""
        try:
            ticker = self._ticker(symbol)
            # auto_adjust=False so the presence of an adjusted column is
            # observable rather than assumed.
            hist = ticker.history(period="2y", interval="1d", auto_adjust=False)
        except Exception as exc:
            return ListingProbe(symbol=symbol, ok=False,
                                error=f"{type(exc).__name__}: {exc}")
        if hist is None or hist.empty:
            return ListingProbe(symbol=symbol, ok=False, error="no history returned")

        exchange = currency = None
        name = ""
        try:
            info = ticker.fast_info
            exchange = getattr(info, "exchange", None)
            currency = getattr(info, "currency", None)
        except Exception:
            pass
        try:
            name = (ticker.info or {}).get("longName") or ""
        except Exception:
            # quoteSummary 404s for some listings -- notably the ETCs on Xetra
            # in the matrix run. A missing name is not a failure.
            pass

        return ListingProbe(
            symbol=symbol, ok=True, exchange=exchange, currency=currency, name=name,
            observations=len(hist),
            first_date=hist.index[0].date(), last_date=hist.index[-1].date(),
            adjusted="Adj Close" in hist.columns,
        )

    def history(self, symbol: str, start: dt.date | None = None) -> pd.Series:
        ticker = self._ticker(symbol)
        # auto_adjust=True: the risk mathematics must run on adjusted closes.
        hist = ticker.history(start=start.isoformat() if start else None,
                              period=None if start else "2y",
                              interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            raise ProviderError(self.name, f"no history for {symbol}")
        series = hist["Close"].astype(float)
        series.index = pd.DatetimeIndex([d.date() for d in series.index])
        series.name = symbol
        return series

    def quote(self, symbol: str) -> Quote:
        ticker = self._ticker(symbol)
        try:
            info = ticker.fast_info
            price = getattr(info, "last_price", None)
            currency = getattr(info, "currency", None) or "EUR"
        except Exception:
            price, currency = None, "EUR"

        if price is not None:
            return Quote(symbol=symbol, price=Decimal(str(round(float(price), 6))),
                         currency=currency, as_of=dt.datetime.now(dt.timezone.utc),
                         source=self.name,
                         delay_minutes=self.documented_delay_minutes)

        # Fall back to the last close, explicitly marked. Never presented as live.
        hist = ticker.history(period="5d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            raise ProviderError(self.name, f"no quote or recent close for {symbol}")
        return Quote(symbol=symbol,
                     price=Decimal(str(round(float(hist["Close"].iloc[-1]), 6))),
                     currency=currency, as_of=hist.index[-1].date(),
                     source=self.name, delay_minutes=None, is_stale=True)
