"""
Verified instrument reference for the Phase 0 spike.

Separated from the probe logic because this table is *evidence*, not code. It
was compiled by hand from broker and issuer sources and every field in it is
load-bearing. Keeping it in its own module means a correction to the data is a
diff you can read without wading through HTTP handling.

Three things in here are not obvious and matter a great deal:

1. ISIN is the key. A `TestInstrument` has one ISIN and many `Listing`s. The
   same fund trades under different symbols on different venues, and two of
   those symbols (WDEF and EUDF) were mistaken for separate funds in the first
   draft of this test list. That mistake is now a test case, not a footnote.

2. `base_currency` and a listing's `currency` are different things. The
   iShares Agribusiness ETF has a USD base currency and trades in EUR in
   Amsterdam. Reporting only one of them hides where FX risk actually sits, and
   for this portfolio FX is the majority case, not an edge case.

3. Dead ISINs are listed deliberately. Two of these funds have liquidated
   predecessors with the same name, same TER and same index, still present in
   fund databases. They are negative test cases: a provider that returns
   plausible recent history for a liquidated line is serving stale data and
   must be disqualified.
"""

from __future__ import annotations

import dataclasses


# --------------------------------------------------------------------------
# Venues
# --------------------------------------------------------------------------
# Keyed by MIC (ISO 10383), because MIC is the only venue identifier that is
# actually standardised. Every provider invents its own exchange labels, so the
# per-provider columns here are the translation layer. The production code will
# need exactly this mapping; better to discover it now than mid-build.

@dataclasses.dataclass(frozen=True)
class Venue:
    mic: str
    label: str
    yahoo_suffix: str
    eodhd_code: str
    fmp_suffix: str
    twelvedata_mic: str


VENUES: dict[str, Venue] = {
    "XETR": Venue("XETR", "Xetra",             ".DE", "XETRA", ".DE", "XETR"),
    "XLON": Venue("XLON", "London",            ".L",  "LSE",   ".L",  "XLON"),
    "XMIL": Venue("XMIL", "Borsa Italiana",    ".MI", "MI",    ".MI", "XMIL"),
    "XPAR": Venue("XPAR", "Euronext Paris",    ".PA", "PA",    ".PA", "XPAR"),
    "XAMS": Venue("XAMS", "Euronext Amsterdam", ".AS", "AS",   ".AS", "XAMS"),
    "XSWX": Venue("XSWX", "SIX Swiss",         ".SW", "SW",    ".SW", "XSWX"),
    # gettex is a retail venue several providers do not carry at all. Included
    # because IS0C quotes there and its absence is itself a finding.
    "XMUN": Venue("XMUN", "gettex/Munich",     ".MU", "MU",    ".MU", "XMUN"),
}


@dataclasses.dataclass(frozen=True)
class Listing:
    """One (symbol, venue) pair for an instrument.

    `currency` is the *quote* currency of this listing, which is often not the
    fund's base currency. It is None where I could not verify it; the spike
    reports what the provider says rather than assuming.
    """
    symbol: str
    mic: str
    currency: str | None = None

    @property
    def venue(self) -> Venue:
        return VENUES[self.mic]


@dataclasses.dataclass(frozen=True)
class TestInstrument:
    isin: str                       # the primary key, everywhere
    name: str
    issuer: str
    asset_class: str                # "ETF" (UCITS fund) | "ETC" (collateralised note)
    base_currency: str              # the fund's own currency, not the listing's
    primary_symbol: str             # the EUR listing I would actually trade
    listings: tuple[Listing, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for l in self.listings:
            seen.setdefault(l.symbol, None)
        return tuple(seen)

    @property
    def primary_listing(self) -> Listing:
        for l in self.listings:
            if l.symbol == self.primary_symbol:
                return l
        return self.listings[0]

    def ordered_listings(self) -> tuple[Listing, ...]:
        """Primary EUR listing first, then the rest.

        Order matters for quota: we stop at the first listing that returns
        enough history, so trying the one most likely to work first is the
        difference between one call and six on a 250-request daily cap.
        """
        rest = tuple(l for l in self.listings if l is not self.primary_listing)
        return (self.primary_listing,) + rest


# --------------------------------------------------------------------------
# The coverage test set
# --------------------------------------------------------------------------
# Ten instruments. This is a *coverage test set*, not a portfolio: some of
# these are researched candidates rather than current holdings.

INSTRUMENTS: tuple[TestInstrument, ...] = (
    TestInstrument(
        isin="IE0002Y8CX98",
        name="WisdomTree Europe Defence UCITS ETF EUR Acc",
        issuer="WisdomTree", asset_class="ETF", base_currency="EUR",
        primary_symbol="EUDF",
        listings=(
            Listing("EUDF", "XETR", "EUR"),
            Listing("WDEF", "XLON"),
            Listing("WDEF", "XMIL"),
            Listing("WDEF", "XPAR"),
            Listing("WDEF", "XSWX"),
            Listing("WDEP", "XLON", "GBX"),
        ),
    ),
    TestInstrument(
        isin="IE000IAXNM41",
        name="iShares Europe Defence UCITS ETF EUR Acc",
        issuer="iShares", asset_class="ETF", base_currency="EUR",
        primary_symbol="DFNC",
        listings=(
            Listing("DFNC", "XETR", "EUR"),
            Listing("DFEU", "XAMS"),
            Listing("DFEU", "XLON"),
            Listing("DFEU", "XSWX"),
        ),
    ),
    TestInstrument(
        isin="IE000I7E6HL0",
        name="HANetf Future of European Defence Screened UCITS ETF Acc",
        issuer="HANetf", asset_class="ETF", base_currency="USD",
        primary_symbol="8RMY",
        listings=(
            Listing("8RMY", "XETR", "EUR"),
            Listing("ARMY", "XPAR"),
            Listing("ARMY", "XLON"),
            Listing("ARMY", "XSWX"),
            Listing("ARMI", "XMIL"),
            Listing("NAVY", "XLON", "GBX"),
        ),
    ),
    TestInstrument(
        isin="IE000OJ5TQP4",
        name="HANetf Future of Defence UCITS ETF",
        issuer="HANetf", asset_class="ETF", base_currency="USD",
        primary_symbol="ASWC",
        listings=(
            Listing("ASWC", "XETR", "EUR"),
            Listing("NATO", "XPAR"),
            Listing("NATO", "XMIL"),
            Listing("NATO", "XLON"),
            Listing("NATO", "XSWX"),
            Listing("NATP", "XLON", "GBX"),
        ),
    ),
    TestInstrument(
        isin="IE00B6R52143",
        name="iShares Agribusiness UCITS ETF",
        issuer="iShares", asset_class="ETF", base_currency="USD",
        primary_symbol="ISAE",
        listings=(
            Listing("ISAE", "XAMS", "EUR"),
            Listing("ISAG", "XMIL"),
            Listing("ISAG", "XLON"),
            Listing("SPAG", "XLON", "GBX"),
            Listing("IS0C", "XMUN"),
        ),
    ),
    TestInstrument(
        isin="GB00B15KYL00",
        name="WisdomTree Grains",
        issuer="WisdomTree", asset_class="ETC", base_currency="USD",
        primary_symbol="D7Y0",
        listings=(
            Listing("D7Y0", "XETR", "EUR"),
            Listing("AIGG", "XLON"),
            Listing("AIGG", "XMIL"),
            Listing("AGGP", "XLON", "GBP"),
        ),
    ),
    TestInstrument(
        isin="JE00BN7KB664",
        name="WisdomTree Wheat",
        issuer="WisdomTree", asset_class="ETC", base_currency="USD",
        primary_symbol="OD7S",
        listings=(
            Listing("OD7S", "XETR", "EUR"),
            Listing("WEAT", "XLON"),
            Listing("WEAT", "XMIL"),
            Listing("WEAP", "XLON", "GBP"),
            Listing("WEATP", "XPAR"),
        ),
    ),
    TestInstrument(
        isin="GB00B15KYB02",
        name="WisdomTree Energy",
        issuer="WisdomTree", asset_class="ETC", base_currency="USD",
        primary_symbol="OD7W",
        listings=(
            Listing("OD7W", "XETR", "EUR"),
            Listing("AIGE", "XLON"),
            Listing("AIGE", "XMIL"),
        ),
    ),
    TestInstrument(
        isin="IE00BMW42637",
        name="iShares MSCI Europe Energy Sector UCITS ETF EUR Acc",
        issuer="iShares", asset_class="ETF", base_currency="EUR",
        primary_symbol="ESIE",
        listings=(
            Listing("ESIE", "XETR", "EUR"),
            Listing("ESIE", "XLON", "GBP"),
        ),
    ),
    TestInstrument(
        isin="LU1681048630",
        name="Amundi Global Luxury UCITS ETF EUR Acc",
        issuer="Amundi", asset_class="ETF", base_currency="EUR",
        primary_symbol="GLUX",
        listings=(
            Listing("GLUX", "XETR", "EUR"),
            Listing("GLUX", "XPAR"),
            Listing("GLUX", "XMIL"),
            Listing("GLUX", "XSWX"),
        ),
    ),
)

BY_ISIN: dict[str, TestInstrument] = {i.isin: i for i in INSTRUMENTS}


# --------------------------------------------------------------------------
# Negative test cases: liquidated predecessors
# --------------------------------------------------------------------------
# Both of these are dead share classes that still appear in fund databases
# under the same name, TER and index as the live line. justETF still shows the
# WisdomTree Wheat predecessor quoting under WEAT and OD7S.
#
# The test is inverted: returning *nothing* is the correct answer. A provider
# that hands back recent-looking history for a liquidated ISIN is serving stale
# data, and would silently feed a dead price series into the covariance matrix.

@dataclasses.dataclass(frozen=True)
class DeadInstrument:
    isin: str
    name: str
    shadows_isin: str          # the live line it is confused with
    known_symbols: tuple[str, ...]
    note: str


DEAD_INSTRUMENTS: tuple[DeadInstrument, ...] = (
    DeadInstrument(
        isin="GB00B15KY765",
        name="WisdomTree Wheat (liquidated/merged predecessor)",
        shadows_isin="JE00BN7KB664",
        known_symbols=("WEAT", "OD7S"),
        note="justETF still shows this quoting under WEAT and OD7S",
    ),
    DeadInstrument(
        isin="DE000A1JS9B2",
        name="iShares Agribusiness (liquidated duplicate)",
        shadows_isin="IE00B6R52143",
        known_symbols=("ISAG", "ISAE"),
        note="same name, TER and index as the live Irish line",
    ),
)

# A liquidated line quoting within this many days of today means the provider
# is serving data it should not have. Generous, to avoid flagging a provider
# that simply carries a stale final print.
DEAD_LINE_STALE_DAYS = 21


# --------------------------------------------------------------------------
# Known ticker collisions with US-listed products
# --------------------------------------------------------------------------
# Every one of these is a real US product sharing a ticker with one of the
# instruments above. This is the table that justifies keying on ISIN.
#
# WDEF is the dangerous one and deserves reading twice: same issuer, same
# theme, same ticker, different fund. A provider returning the US listing
# passes every sanity check a human would apply -- right name, right sector,
# a sensible price series. Only the ISIN and the exchange tell them apart.

@dataclasses.dataclass(frozen=True)
class Collision:
    symbol: str
    our_isin: str
    us_identifier: str
    us_name: str
    danger: str = ""


COLLISIONS: tuple[Collision, ...] = (
    Collision("WDEF", "IE0002Y8CX98", "US97717Y3374",
              "WisdomTree Europe Defense Fund (US)",
              danger="SAME ISSUER, same theme, same ticker. Only ISIN and exchange differ."),
    Collision("NATO", "IE000OJ5TQP4", "US8829277677",
              "Themes Transatlantic Defense ETF (US)"),
    Collision("ARMY", "IE000I7E6HL0", "CUSIP 87975E784",
              "Tema International Defense ETF (US)"),
    Collision("WEAT", "JE00BN7KB664", "US88166A8707",
              "Teucrium Wheat Fund (US)"),
    Collision("GLUX", "LU1681048630", "-",
              "Great Lakes Aviation (US equity)"),
)


# --------------------------------------------------------------------------
# Symbol identity groups
# --------------------------------------------------------------------------
# Derived, not hand-written: every distinct symbol of an instrument must
# resolve to that instrument's ISIN. This is correction 1 turned into an
# assertion -- WDEF and EUDF are one fund, and a provider that answers them
# with two different instruments has the exact failure the ISIN rule prevents.

def alias_groups(collisions_only: bool = True) -> list[tuple[str, tuple[str, ...]]]:
    colliding = {c.symbol for c in COLLISIONS}
    out: list[tuple[str, tuple[str, ...]]] = []
    for inst in INSTRUMENTS:
        syms = inst.symbols
        if len(syms) < 2:
            continue
        if collisions_only and not (set(syms) & colliding):
            continue
        out.append((inst.isin, syms))
    return out


# Currency sanity: only four of the ten are EUR-based. FX is the majority case
# in this test set, which is why base and quote currency are reported apart.
EUR_BASED_ISINS = frozenset(
    i.isin for i in INSTRUMENTS if i.base_currency == "EUR"
)
