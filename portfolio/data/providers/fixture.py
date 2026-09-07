"""Deterministic synthetic market data. The offline provider.

Two jobs, one implementation:

  1. Every API test runs against it, so the suite is fast and never touches
     the network (the test session blocks sockets outright).
  2. `portfolio serve --provider fixture` gives a complete, working demo with
     no network at all -- a new user sees every page populated within seconds
     of installing, and a developer can build the interface on a train.

The series are generated from a seeded random walk with a shared market
factor, so correlations, volatilities and drawdowns are plausible rather than
noise. The generator is keyed on the symbol string: the same symbol always
yields the same series, whatever order it is asked for in.

Nothing here is a claim about any real instrument's prices.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from decimal import Decimal

import numpy as np
import pandas as pd

from ..provider import ListingProbe, MarketDataProvider, ProviderError, Quote

__all__ = ["FixtureProvider", "FixtureIdentityProvider", "FIXTURE_END"]

# The last trading day of every synthetic series. Fixed so tests are
# reproducible and the demo does not drift day by day.
FIXTURE_END = dt.date(2026, 9, 4)

# Symbols the seed universe and the benchmarks use, with (annual drift, annual
# volatility, market beta, observations). Anything not listed gets sensible
# defaults derived from its hash. Row counts mirror what the real venues
# returned in the 2026-09-03 provider matrix, so the alignment report in the
# demo shows the same shortened window a real book would.
_PROFILES: dict[str, tuple[float, float, float, int]] = {
    "EUDF.DE": (0.35, 0.30, 1.3, 377),
    "DFNC.DE": (0.30, 0.29, 1.25, 320),
    "8RMY.DE": (0.28, 0.31, 1.2, 356),
    "ASWC.DE": (0.22, 0.27, 1.1, 505),
    "ISAE.AS": (0.04, 0.18, 0.8, 508),
    "AIGG.MI": (-0.05, 0.22, 0.2, 503),
    "WEAT.MI": (-0.08, 0.28, 0.15, 503),
    "AIGE.MI": (0.02, 0.33, 0.4, 503),
    "ESIE.DE": (0.08, 0.20, 0.9, 490),
    "GLUX.PA": (0.06, 0.19, 1.0, 508),
    "MEUD.PA": (0.09, 0.14, 1.0, 511),
    "IWDA.AS": (0.11, 0.15, 0.95, 510),
    "SMEA.MI": (0.08, 0.14, 1.0, 503),
}

# Spot levels for FX pairs, quoted as units of the second currency per one of
# the first, the way Yahoo names them ("USDEUR=X" is EUR per USD).
_FX_LEVELS: dict[str, float] = {
    "USDEUR=X": 0.92, "GBPEUR=X": 1.17, "CHFEUR=X": 1.05, "SEKEUR=X": 0.088,
    "DKKEUR=X": 0.134, "NOKEUR=X": 0.086, "JPYEUR=X": 0.0061, "CADEUR=X": 0.67,
    "AUDEUR=X": 0.60, "EURUSD=X": 1.087, "EURGBP=X": 0.855, "EURCHF=X": 0.952,
}

_EXCHANGES = {".DE": ("GER", "EUR"), ".MI": ("MIL", "EUR"), ".PA": ("PAR", "EUR"),
              ".AS": ("AMS", "EUR"), ".L": ("LSE", "GBp"), ".SW": ("EBS", "CHF")}


def _seed(symbol: str, salt: int) -> int:
    digest = hashlib.sha256(f"{salt}:{symbol}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


class FixtureProvider(MarketDataProvider):
    name = "fixture"
    documented_delay_minutes = 15

    def __init__(self, seed: int = 7, end: dt.date = FIXTURE_END,
                 fail: frozenset[str] = frozenset()) -> None:
        self.seed = seed
        self.end = end
        # Symbols that raise, so degraded states can be tested deliberately.
        self.fail = frozenset(fail)
        self._market = self._market_factor()

    # -- generation --------------------------------------------------------

    def _market_factor(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0004, 0.009, 600)

    def _profile(self, symbol: str) -> tuple[float, float, float, int]:
        if symbol in _PROFILES:
            return _PROFILES[symbol]
        h = _seed(symbol, 1)
        drift = -0.05 + (h % 1000) / 1000 * 0.30
        vol = 0.12 + ((h >> 10) % 1000) / 1000 * 0.25
        beta = 0.3 + ((h >> 20) % 1000) / 1000 * 1.0
        return drift, vol, beta, 505

    def _series(self, symbol: str) -> pd.Series:
        if symbol in self.fail:
            raise ProviderError(self.name, f"{symbol} is configured to fail")
        if symbol in _FX_LEVELS or symbol.endswith("=X"):
            return self._fx_series(symbol)
        drift, vol, beta, n = self._profile(symbol)
        rng = np.random.default_rng(_seed(symbol, self.seed))
        daily_vol = vol / np.sqrt(252)
        idio = rng.normal(0.0, daily_vol * 0.75, n)
        market = self._market[-n:] * beta
        steps = drift / 252 + market + idio
        level = 20.0 + (_seed(symbol, 2) % 9000) / 100.0
        prices = level * np.exp(np.cumsum(steps))
        index = pd.bdate_range(end=self.end, periods=n)
        return pd.Series(prices, index=index, name=symbol)

    def _fx_series(self, symbol: str) -> pd.Series:
        level = _FX_LEVELS.get(symbol, 1.0)
        rng = np.random.default_rng(_seed(symbol, self.seed))
        n = 520
        steps = rng.normal(0.0, 0.004, n)
        rates = level * np.exp(np.cumsum(steps) - steps.sum())
        index = pd.bdate_range(end=self.end, periods=n)
        return pd.Series(rates, index=index, name=symbol)

    def _exchange(self, symbol: str) -> tuple[str, str]:
        for suffix, meta in _EXCHANGES.items():
            if symbol.endswith(suffix):
                return meta
        if symbol.endswith("=X"):
            return "CCY", symbol[3:6]
        # A bare ticker is what a US listing looks like on Yahoo. Answering as
        # a US venue lets the demo exercise the resolver's hard gate: the
        # colliding WDEF/WEAT/GLUX tickers are the reason ISIN keying exists.
        return "PCX", "USD"

    # -- MarketDataProvider ------------------------------------------------

    def probe(self, symbol: str, lookback_days: int = 252) -> ListingProbe:
        try:
            series = self._series(symbol)
        except Exception as exc:
            return ListingProbe(symbol=symbol, ok=False, error=str(exc))
        exchange, currency = self._exchange(symbol)
        return ListingProbe(
            symbol=symbol, ok=True, exchange=exchange, currency=currency,
            name=f"Fixture {symbol.split('.')[0]}", observations=len(series),
            first_date=series.index[0].date(), last_date=series.index[-1].date(),
            adjusted=True)

    def history(self, symbol: str, start: dt.date | None = None) -> pd.Series:
        series = self._series(symbol)
        if start is not None:
            series = series[series.index >= pd.Timestamp(start)]
        if series.empty:
            raise ProviderError(self.name, f"no history returned for {symbol}")
        return series

    def quote(self, symbol: str) -> Quote:
        series = self._series(symbol)
        _, currency = self._exchange(symbol)
        return Quote(symbol=symbol,
                     price=Decimal(str(round(float(series.iloc[-1]), 6))),
                     currency=currency,
                     as_of=dt.datetime.combine(series.index[-1].date(),
                                               dt.time(17, 30),
                                               tzinfo=dt.timezone.utc),
                     source=self.name, delay_minutes=self.documented_delay_minutes)


# --------------------------------------------------------------------------
# Identity: the offline stand-in for OpenFIGI
# --------------------------------------------------------------------------

# Seed universe listings, mirroring what OpenFIGI returned on 2026-09-03 in
# miniature: the primary venue, one or two alternates, a Tradegate line and an
# MTF line so the filter counts are non-zero, and for the colliding tickers a
# US listing so the gate has something to refuse.
_SEED_LISTINGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "IE0002Y8CX98": ("EUDF", ("GR", "TH", "XE")),
    "IE000I7E6HL0": ("8RMY", ("GR", "TH")),
    "IE000IAXNM41": ("DFNC", ("GR", "TH", "XE")),
    "IE000OJ5TQP4": ("ASWC", ("GR", "TH")),
    "IE00B6R52143": ("ISAE", ("NA", "GS")),
    "IE00BMW42637": ("ESIE", ("GR", "TH")),
    "GB00B15KYB02": ("AIGE", ("IM", "LN")),
    "GB00B15KYL00": ("AIGG", ("IM", "LN")),
    "JE00BN7KB664": ("WEAT", ("IM", "LN", "UP")),
    "LU1681048630": ("GLUX", ("FP", "GR", "IM", "UP")),
}
_ALTERNATE_TICKERS: dict[str, dict[str, str]] = {
    "IE0002Y8CX98": {"LN": "WDEF", "SW": "WDEF", "UP": "WDEF"},
    "IE000I7E6HL0": {"LN": "ARMY"},
    "IE000OJ5TQP4": {"FP": "NATO", "UP": "NATO"},
    "IE00B6R52143": {"LN": "ISAG"},
}


class FixtureIdentityProvider:
    """Listings for any valid ISIN, without OpenFIGI.

    Seed instruments get their real venue shape. Any other ISIN gets a
    plausible set derived from its hash -- a Xetra primary, a London line, a
    Tradegate line and an MTF line -- so the add-instrument flow can be
    exercised end to end offline.
    """
    name = "fixture-identity"

    def listings_for_isin(self, isin: str) -> list[FigiListing]:
        from ..provider import FigiListing
        isin = isin.strip().upper()
        if isin in _SEED_LISTINGS:
            ticker, venues = _SEED_LISTINGS[isin]
            alternates = _ALTERNATE_TICKERS.get(isin, {})
            venues = tuple(venues) + tuple(v for v in alternates if v not in venues)
        else:
            h = _seed(isin, 3)
            letters = "".join(chr(65 + (h >> (5 * i)) % 26) for i in range(4))
            ticker, venues, alternates = letters, ("GR", "LN", "TH", "XE"), {}
        out = []
        for i, venue in enumerate(venues):
            out.append(FigiListing(
                figi=f"BBG{_seed(isin + venue, 4) % 10**9:09d}",
                ticker=alternates.get(venue, ticker), exchange_code=venue,
                security_type="ETP", name=f"Fixture fund {ticker}"))
        return out
