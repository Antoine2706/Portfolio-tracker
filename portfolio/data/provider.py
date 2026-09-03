"""Provider interfaces. Every market data call goes through one of these.

Two interfaces, because the 3 September 2026 matrix forced a split that turns
out to be the right architecture anyway:

    IdentityProvider    what is this ISIN, and where does it trade?
    MarketDataProvider  what does this symbol cost, and what is its history?

Yahoo has no ISIN search, so it cannot answer the first question at all.
OpenFIGI is ISIN-native and answers it precisely, but carries no prices. Rather
than treating that as a limitation to work around, the two jobs are separated:
OpenFIGI for identity, Yahoo for prices, both behind these interfaces.

The payoff is migration. If yfinance breaks -- and Yahoo wrappers break every
few months, which the matrix run confirmed by getting HTTP 429 from a raw chart
request without cookie and crumb handling -- only `MarketDataProvider` needs a
new implementation. The identity layer and the stored venue map are unaffected.

No provider-specific code may appear outside `data/providers/`.
"""

from __future__ import annotations

import abc
import dataclasses
import datetime as dt
from decimal import Decimal

import pandas as pd

__all__ = ["FigiListing", "ListingProbe", "Quote", "IdentityProvider",
           "MarketDataProvider", "ProviderError", "RateLimited"]


class ProviderError(RuntimeError):
    """A provider could not answer. Carries the provider name for the UI."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider


class RateLimited(ProviderError):
    """Rate limit hit. The caller must serve cached data and warn visibly,
    never blank the display and never crash."""


@dataclasses.dataclass(frozen=True)
class FigiListing:
    """One listing of an ISIN, as an identity provider reports it.

    `exchange_code` is in the provider's own vocabulary -- Bloomberg codes for
    OpenFIGI -- and is translated by `data.venues`, never interpreted here.
    """
    figi: str
    ticker: str
    exchange_code: str
    security_type: str = ""
    name: str = ""
    currency: str | None = None


@dataclasses.dataclass(frozen=True)
class ListingProbe:
    """What a price provider knows about a symbol, without committing to it.

    Used by the resolver to rank candidates before anything is saved. Every
    field is what the provider actually returned, not what we hoped for --
    `exchange` in particular, because that is what the US gate is applied to.
    """
    symbol: str
    ok: bool
    exchange: str | None = None
    currency: str | None = None
    name: str = ""
    observations: int = 0
    first_date: dt.date | None = None
    last_date: dt.date | None = None
    adjusted: bool | None = None
    error: str = ""


@dataclasses.dataclass(frozen=True)
class Quote:
    """A price with the provenance the UI is required to show.

    `as_of` and `delay_minutes` are mandatory, not decoration: delayed data must
    never be presented as real time. `is_stale` marks a last-known close served
    because the live quote failed -- that row must be visibly flagged.
    """
    symbol: str
    price: Decimal
    currency: str
    as_of: dt.datetime | dt.date
    source: str
    delay_minutes: int | None = None
    is_stale: bool = False

    def provenance(self) -> str:
        """The sentence the UI puts next to the price."""
        when = (self.as_of.isoformat(timespec="minutes")
                if isinstance(self.as_of, dt.datetime) else self.as_of.isoformat())
        if self.is_stale:
            return f"last close {when} ({self.source}) - NOT a live price"
        delay = (f", delayed ~{self.delay_minutes} min"
                 if self.delay_minutes else ", delay unstated")
        return f"as of {when} ({self.source}{delay})"


class IdentityProvider(abc.ABC):
    """Answers 'what is this ISIN and where does it trade'."""

    name: str = "abstract"

    @abc.abstractmethod
    def listings_for_isin(self, isin: str) -> list[FigiListing]:
        """Every listing the provider knows for this ISIN, unfiltered.

        Returning raw is deliberate: filtering belongs in `data.resolve`, where
        it is testable and where the counts can be reported to the user, rather
        than hidden inside a provider.
        """


class MarketDataProvider(abc.ABC):
    """Answers 'what does this symbol cost'."""

    name: str = "abstract"
    documented_delay_minutes: int | None = None

    @abc.abstractmethod
    def probe(self, symbol: str, lookback_days: int) -> ListingProbe:
        """Cheap look at a symbol: currency, exchange, row count, date range."""

    @abc.abstractmethod
    def history(self, symbol: str, start: dt.date | None = None) -> pd.Series:
        """Adjusted close series indexed by date.

        Must be adjusted. Unadjusted prices read a distribution as a large
        negative return and inflate measured volatility; several of these
        instruments distribute.
        """

    @abc.abstractmethod
    def quote(self, symbol: str) -> Quote:
        """Latest price, with its as-of timestamp and stated delay."""
