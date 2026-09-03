"""Venue codes: Bloomberg (OpenFIGI) to Yahoo suffix to MIC.

Three identifier systems name the same trading venue and none of them agree.
OpenFIGI answers in Bloomberg exchange codes, yfinance wants a filename-style
suffix, and MIC (ISO 10383) is the only standardised one, so the instrument
model stores MIC and this module translates.

The policy here is an ALLOWLIST, not a blocklist.
-------------------------------------------------
A single ISIN can return seventy-plus listings from OpenFIGI: primary venues,
German regionals, Tradegate, MTFs, dark venues, and separate lines per currency
and share class. Trying to enumerate the bad ones is a losing game -- a new MTF
code appears and it silently becomes a candidate.

So only codes in `PRIMARY_VENUES` generate candidates. Everything else is
excluded and *counted* by class, so the confirm screen can say "62 listings
filtered out: 8 German regional, 41 MTF or dark, 13 other" rather than
pretending the noise was not there. The class labels below exist purely to make
that sentence informative; they are not what does the filtering.
"""

from __future__ import annotations

import dataclasses
import enum

__all__ = [
    "VenueClass", "Venue", "PRIMARY_VENUES", "US_YAHOO_EXCHANGES",
    "NON_VENUE_EXCHANGES", "classify_bloomberg_code", "bloomberg_to_yahoo",
    "yahoo_suffix_to_mic", "is_us_yahoo_exchange", "refused_exchange_reason",
]


class VenueClass(str, enum.Enum):
    PRIMARY = "primary"                  # the only class that yields candidates
    GERMAN_REGIONAL = "german_regional"  # regional exchanges and Tradegate
    MTF = "mtf"                          # multilateral trading facilities, dark venues
    US = "us"                            # hard refusal, see the gate below
    OTHER = "other"


@dataclasses.dataclass(frozen=True)
class Venue:
    bloomberg: str
    label: str
    yahoo_suffix: str
    mic: str


# The allowlist. Every code here was confirmed against the stored venue map:
# the ten instruments resolved on 2026-09-03 all sit on one of these.
PRIMARY_VENUES: dict[str, Venue] = {
    "LN": Venue("LN", "London Stock Exchange", ".L", "XLON"),
    "IM": Venue("IM", "Borsa Italiana", ".MI", "XMIL"),
    "GR": Venue("GR", "Xetra", ".DE", "XETR"),
    "FP": Venue("FP", "Euronext Paris", ".PA", "XPAR"),
    "NA": Venue("NA", "Euronext Amsterdam", ".AS", "XAMS"),
    # SIX quotes under either code depending on the segment.
    "SW": Venue("SW", "SIX Swiss Exchange", ".SW", "XSWX"),
    "SE": Venue("SE", "SIX Swiss Exchange", ".SW", "XSWX"),
}

# Classified only so the report can explain what it discarded. German regional
# exchanges and Tradegate quote the same funds with thinner books; TH is
# Tradegate specifically. They are excluded by default because their history is
# patchier and their prices lag the primary listing.
_GERMAN_REGIONAL = {"GF", "GD", "GS", "GM", "GH", "GT", "GZ", "TH", "EU", "QT"}

# Bloomberg codes for US venues. Distinct from the Yahoo-side gate below: this
# one stops a US listing becoming a candidate at all.
_US_BLOOMBERG = {"US", "UN", "UQ", "UP", "UA", "UR", "UW", "UF", "UV", "UD", "PQ"}

# Yahoo's own exchange labels for US venues. This is the gate that matters most,
# because it is applied to what the price probe actually returned. The 2026-09-03
# matrix had all five bare colliding tickers resolving here with healthy series:
# WEAT gave 502 clean days of Teucrium Wheat Fund, GLUX 502 days of a
# pink-sheets airline. A check that only asks "did data come back" passes both.
US_YAHOO_EXCHANGES = frozenset({
    "PCX", "NGM", "NYQ", "NMS", "PNK", "NCM", "NIM", "ASE", "BTS", "OTC",
    "NYSE", "NASDAQ", "ARCA", "NYSEARCA", "AMEX", "BATS", "CBOE", "OPR",
})


# Yahoo labels a listing YHD when it has no real venue to name -- a generic
# marker rather than an exchange. IS0C.DE came back as YHD with 252 rows of
# EUR prices ending a year earlier: a full-looking window on a line Yahoo
# itself cannot place. The staleness check catches that case on the date, but
# the marker is reason enough on its own, because a price source that cannot
# name its venue cannot be verified against the ISIN.
NON_VENUE_EXCHANGES = frozenset({"YHD", "CCY", "CCC", "FGI"})


def refused_exchange_reason(exchange: str | None) -> str | None:
    """Why this exchange is refused outright, or None if it is acceptable.

    One place for the hard gate, so resolution and price-fetch cannot drift.
    """
    if is_us_yahoo_exchange(exchange):
        return (f"{exchange} is a US venue. The series may look complete and "
                f"still be a different fund sharing the ticker.")
    if exchange and exchange.strip().upper() in NON_VENUE_EXCHANGES:
        return (f"{exchange} is not a real exchange, it is a placeholder Yahoo "
                f"uses when it cannot name a venue. A price source that cannot "
                f"name its venue cannot be checked against the ISIN.")
    return None


def classify_bloomberg_code(code: str) -> VenueClass:
    """Label a Bloomberg exchange code, for reporting what was filtered.

    >>> classify_bloomberg_code("GR").value
    'primary'
    >>> classify_bloomberg_code("TH").value
    'german_regional'
    >>> classify_bloomberg_code("X2").value
    'mtf'
    >>> classify_bloomberg_code("UN").value
    'us'
    """
    c = (code or "").strip().upper()
    if c in PRIMARY_VENUES:
        return VenueClass.PRIMARY
    if c in _GERMAN_REGIONAL:
        return VenueClass.GERMAN_REGIONAL
    if c in _US_BLOOMBERG:
        return VenueClass.US
    # X* are MTFs and dark venues; E* are the Euronext/Cboe MTF family.
    # Checked after the allowlist so a primary code can never fall in here.
    if c.startswith("X") or c.startswith("E"):
        return VenueClass.MTF
    return VenueClass.OTHER


def bloomberg_to_yahoo(ticker: str, bloomberg_code: str) -> str | None:
    """Build the Yahoo symbol for a listing, or None if the venue is not primary.

    >>> bloomberg_to_yahoo("EUDF", "GR")
    'EUDF.DE'
    >>> bloomberg_to_yahoo("AIGG", "IM")
    'AIGG.MI'
    >>> bloomberg_to_yahoo("EUDF", "TH") is None      # Tradegate, excluded
    True
    """
    venue = PRIMARY_VENUES.get((bloomberg_code or "").strip().upper())
    if venue is None or not ticker:
        return None
    return f"{ticker.strip().upper()}{venue.yahoo_suffix}"


def yahoo_suffix_to_mic(symbol: str) -> str | None:
    """MIC for a Yahoo symbol, so the resolved listing stores a standard code.

    >>> yahoo_suffix_to_mic("ISAE.AS")
    'XAMS'
    >>> yahoo_suffix_to_mic("WEAT") is None           # bare ticker: no venue
    True
    """
    if "." not in (symbol or ""):
        return None
    suffix = "." + symbol.rsplit(".", 1)[1].upper()
    for venue in PRIMARY_VENUES.values():
        if venue.yahoo_suffix.upper() == suffix:
            return venue.mic
    return None


def is_us_yahoo_exchange(exchange: str | None) -> bool:
    """True if Yahoo's reported exchange is a US venue. The hard gate.

    >>> is_us_yahoo_exchange("PCX"), is_us_yahoo_exchange("GER")
    (True, False)
    """
    if not exchange:
        return False
    e = exchange.strip().upper()
    tokens = set(e.replace("-", " ").replace("/", " ").split()) | {e}
    return bool(tokens & US_YAHOO_EXCHANGES)
