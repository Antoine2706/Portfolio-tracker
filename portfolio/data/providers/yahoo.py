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


def _explain(exc: Exception) -> str:
    """A readable sentence for a library exception.

    yfinance surfaces network trouble as its own internal errors -- a
    `TypeError` from parsing a response that never arrived, for instance -- so
    the raw message is often about iteration rather than about the network.
    Naming the exception type keeps it diagnosable without pretending the
    underlying text is an explanation.
    """
    text = str(exc).strip()
    if not text or "NoneType" in text:
        return (f"the provider returned nothing usable ({type(exc).__name__}). "
                f"This is usually an unreachable network or an unknown symbol.")
    return f"{type(exc).__name__}: {text}"


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
        try:
            # auto_adjust=True: the risk mathematics must run on adjusted closes.
            hist = ticker.history(start=start.isoformat() if start else None,
                                  period=None if start else "2y",
                                  interval="1d", auto_adjust=True)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                self.name,
                f"could not load history for {symbol}: {_explain(exc)}") from exc
        if hist is None or hist.empty:
            raise ProviderError(self.name, f"no history returned for {symbol}")
        series = hist["Close"].astype(float)
        series.index = pd.DatetimeIndex([d.date() for d in series.index])
        series.name = symbol
        return series

    def quote(self, symbol: str) -> Quote:
        """Latest price, or a clearly-marked last close.

        Every failure leaves here as a `ProviderError` carrying a sentence. An
        unwrapped library exception -- yfinance raising `TypeError: argument of
        type 'NoneType' is not iterable` deep inside its own parsing, say --
        would surface a Python internal to someone reading their portfolio,
        which is the one thing every other failure path in this project avoids.
        """
        ticker = self._ticker(symbol)
        currency = "EUR"
        try:
            info = ticker.fast_info
            price = getattr(info, "last_price", None)
            currency = getattr(info, "currency", None) or "EUR"
        except Exception:
            # A missing live quote is ordinary; fall through to the last close.
            price = None

        if price is not None:
            return Quote(symbol=symbol, price=Decimal(str(round(float(price), 6))),
                         currency=currency, as_of=dt.datetime.now(dt.timezone.utc),
                         source=self.name,
                         delay_minutes=self.documented_delay_minutes)

        # Fall back to the last close, explicitly marked. Never presented as live.
        try:
            hist = ticker.history(period="5d", interval="1d", auto_adjust=True)
        except Exception as exc:
            raise ProviderError(
                self.name,
                f"no price available for {symbol}: {_explain(exc)}") from exc
        if hist is None or hist.empty:
            raise ProviderError(
                self.name,
                f"no quote and no recent close for {symbol}. The symbol may be "
                f"wrong, or the provider may be unreachable.")
        return Quote(symbol=symbol,
                     price=Decimal(str(round(float(hist["Close"].iloc[-1]), 6))),
                     currency=currency, as_of=hist.index[-1].date(),
                     source=self.name, delay_minutes=None, is_stale=True)
