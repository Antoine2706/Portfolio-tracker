"""ISIN to a provider symbol: candidate generation, gating, ranking.

This is the critical path of the add-instrument screen, and it is where a
silent error would enter the system and never leave. So the design is
resolve, then confirm, then save -- never a text box that writes to the table.

The pipeline
------------
    1. Known?     If the ISIN is already in the stored venue map, return it.
                  Resolution only runs for instruments the map does not contain.
    2. Identity   OpenFIGI maps ISIN to every listing. For IE0002Y8CX98 that is
                  70+ rows: primary venues, German regionals, Tradegate, MTFs,
                  dark venues, and separate lines per currency and share class.
    3. Filter     Allowlist of primary venues only. Everything else is counted
                  by class and reported, never silently dropped.
    4. Translate  Bloomberg exchange code to Yahoo suffix.
    5. Probe      Ask the price provider what each candidate actually is:
                  currency, exchange, row count, date range.
    6. Gate       Hard refusal for a US venue, whatever the series looks like.
    7. Rank       PASS above THIN, EUR above foreign, longer history above short.
    8. Confirm    Present the top few. The user chooses. Nothing auto-saves.

Why the gate is hard rather than a ranking penalty
--------------------------------------------------
On 2026-09-03 all five bare colliding tickers returned healthy US securities:
WEAT gave 502 clean days of Teucrium Wheat Fund, GLUX 502 days of a pink-sheets
airline, NATO 474 days of a different defence ETF. Every one would outrank a
genuine European listing on history length alone. A penalty is a number that
can be outweighed; a gate cannot. US venues are refused and shown as refused.

Why THIN exists
---------------
AIGG.L and AIGE.L resolve on a European venue in the right currency and return
two rows. Nothing fails. Only the row count betrays them. THIN is never
auto-selected as primary -- but it is shown, because a thin listing may be the
only one an instrument has and the user is entitled to choose it knowingly.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
from typing import Callable

from ..core.models import Instrument
from ..core.positions import (DEFAULT_FRESHNESS_BUSINESS_DAYS,
                              business_days_since, is_outdated)
from ..core.returns import DEFAULT_LOOKBACK, MIN_OBSERVATIONS
from .provider import FigiListing, IdentityProvider, ListingProbe, MarketDataProvider
from .venues import (PRIMARY_VENUES, VenueClass, bloomberg_to_yahoo,
                     classify_bloomberg_code, refused_exchange_reason,
                     yahoo_suffix_to_mic)

__all__ = ["Verdict", "Candidate", "Resolution", "resolve_isin",
           "resolve_from_listings", "candidates_from_listings", "rank_candidates",
           "instrument_from_candidate", "PREFERRED_CURRENCY"]

PREFERRED_CURRENCY = "EUR"


class Verdict(str, enum.Enum):
    """Ordered by desirability; `REFUSED` is a wall, not a low score."""
    PASS = "PASS"        # primary European venue, current, history at/above lookback
    THIN = "THIN"        # resolves and is current, but too few rows
    STALE = "STALE"      # resolves with enough rows, but the series has stopped
    FAILED = "FAILED"    # no usable data returned
    REFUSED = "REFUSED"  # hard gate: US venue, or no nameable venue at all


@dataclasses.dataclass
class Candidate:
    """One possible listing for an ISIN, after probing."""
    isin: str
    ticker: str
    bloomberg_code: str
    yahoo_symbol: str
    venue_label: str
    mic: str | None
    verdict: Verdict
    observations: int = 0
    first_date: dt.date | None = None
    last_date: dt.date | None = None
    currency: str | None = None
    reported_exchange: str | None = None
    name: str = ""
    reasons: list[str] = dataclasses.field(default_factory=list)

    @property
    def selectable_as_primary(self) -> bool:
        """Only a PASS may be chosen automatically.

        A THIN or STALE candidate is still offered on the confirm screen, but
        the user has to pick it deliberately. Neither becomes the primary
        listing by default: a two-row series would poison the covariance matrix,
        and a year-old one would price the portfolio against a mark from last
        September while looking entirely healthy.
        """
        return self.verdict is Verdict.PASS

    @property
    def is_preferred_currency(self) -> bool:
        return (self.currency or "").upper() == PREFERRED_CURRENCY

    def describe(self) -> str:
        """The confirm-screen line. Every field the WDEF case needs to be
        distinguishable: ISIN, name, exchange, currency, history length."""
        span = (f"{self.observations} days, {self.first_date} to {self.last_date}"
                if self.observations else "no history")
        return (f"[{self.verdict.value}] {self.yahoo_symbol} - {self.name or '?'} - "
                f"{self.venue_label} ({self.reported_exchange or '?'}) - "
                f"{self.currency or '?'} - {span} - ISIN {self.isin}")


@dataclasses.dataclass
class Resolution:
    """Everything the confirm screen needs, including what was thrown away."""
    isin: str
    candidates: list[Candidate]          # ranked, selectable-first
    refused: list[Candidate]             # hard-gated; shown, never selectable
    listings_seen: int = 0
    filtered_by_class: dict[str, int] = dataclasses.field(default_factory=dict)
    from_stored_map: bool = False
    errors: list[str] = dataclasses.field(default_factory=list)

    @property
    def recommended(self) -> Candidate | None:
        """Top PASS candidate, or None. Never a THIN one."""
        for c in self.candidates:
            if c.selectable_as_primary:
                return c
        return None

    @property
    def blocked(self) -> bool:
        return self.recommended is None

    def block_reason(self) -> str | None:
        """Why the save is blocked, in a sentence the user can act on."""
        if not self.blocked:
            return None
        if not self.candidates and not self.refused:
            return (f"No listing found for {self.isin} on any allowlisted European "
                    f"venue. {self.listings_seen} listings were returned but all were "
                    f"filtered out: {self._filter_summary()}.")
        if self.refused and not self.candidates:
            venues = ", ".join(sorted({c.reported_exchange or "?" for c in self.refused}))
            return (f"Every listing found for {self.isin} is on a non-European venue "
                    f"({venues}). This is refused regardless of how complete the price "
                    f"history looks: a ticker collision returns a real security with a "
                    f"clean series for the wrong fund.")
        stale = [c for c in self.candidates if c.verdict is Verdict.STALE]
        thin = [c for c in self.candidates if c.verdict is Verdict.THIN]
        if thin:
            best = max(thin, key=lambda c: c.observations)
            extra = (f" {len(stale)} further listing(s) have enough rows but have "
                     f"stopped updating." if stale else "")
            return (f"{self.isin} resolves, but no listing has enough history to enter "
                    f"the risk model. The longest is {best.yahoo_symbol} with "
                    f"{best.observations} observations. You can select it deliberately, "
                    f"but it will not be chosen for you.{extra}")
        if stale:
            best = max(stale, key=lambda c: c.last_date or dt.date.min)
            return (f"{self.isin} resolves with enough history, but every listing has "
                    f"stopped updating. The most recent is {best.yahoo_symbol}, last "
                    f"observed {best.last_date}. Selecting it would price this holding "
                    f"against a stale mark and compute risk from a window that has "
                    f"already ended.")
        return f"No usable listing for {self.isin}."

    def _filter_summary(self) -> str:
        if not self.filtered_by_class:
            return "none"
        return ", ".join(f"{n} {cls.replace('_', ' ')}"
                         for cls, n in sorted(self.filtered_by_class.items()))

    def summary(self) -> str:
        return (f"{self.isin}: {self.listings_seen} listings seen, "
                f"{len(self.candidates)} candidates, {len(self.refused)} refused"
                + (f" ({self._filter_summary()} filtered)" if self.filtered_by_class else ""))


# --------------------------------------------------------------------------
# Steps 3-6, pure and independently testable
# --------------------------------------------------------------------------

def candidates_from_listings(isin: str, listings: list[FigiListing],
                             ) -> tuple[list[Candidate], dict[str, int]]:
    """Filter to primary venues and translate. Returns candidates and the
    per-class count of what was filtered out.

    Deduplicates on the resulting Yahoo symbol: OpenFIGI returns several rows
    per venue for different share and currency lines, and they collapse to the
    same symbol once translated.
    """
    filtered: dict[str, int] = {}
    out: list[Candidate] = []
    seen: set[str] = set()

    for listing in listings:
        cls = classify_bloomberg_code(listing.exchange_code)
        if cls is not VenueClass.PRIMARY:
            filtered[cls.value] = filtered.get(cls.value, 0) + 1
            continue
        symbol = bloomberg_to_yahoo(listing.ticker, listing.exchange_code)
        if symbol is None or symbol in seen:
            continue
        seen.add(symbol)
        venue = PRIMARY_VENUES[listing.exchange_code.strip().upper()]
        out.append(Candidate(
            isin=isin, ticker=listing.ticker,
            bloomberg_code=listing.exchange_code.strip().upper(),
            yahoo_symbol=symbol, venue_label=venue.label, mic=venue.mic,
            verdict=Verdict.FAILED, name=listing.name,
            currency=listing.currency,
        ))
    return out, filtered


def _apply_probe(candidate: Candidate, probe: ListingProbe,
                 lookback: int, min_observations: int,
                 as_of: dt.date | None = None,
                 max_stale_business_days: int = DEFAULT_FRESHNESS_BUSINESS_DAYS,
                 ) -> Candidate:
    """Turn a probe result into a verdict. The gate is applied first."""
    candidate.reported_exchange = probe.exchange
    candidate.currency = probe.currency or candidate.currency
    candidate.name = probe.name or candidate.name
    candidate.observations = probe.observations
    candidate.first_date = probe.first_date
    candidate.last_date = probe.last_date

    # The hard gate. Before any quality assessment, because quality is exactly
    # what makes a colliding US listing dangerous.
    refusal = refused_exchange_reason(probe.exchange)
    if refusal:
        candidate.verdict = Verdict.REFUSED
        candidate.reasons.append(f"refused: {refusal}")
        return candidate

    if not probe.ok or probe.observations == 0:
        candidate.verdict = Verdict.FAILED
        candidate.reasons.append(probe.error or "no data returned")
        return candidate

    # Freshness is checked BEFORE row count, because a stale series is the more
    # deceptive failure: IS0C.DE returned exactly 252 rows -- a full lookback --
    # ending a year earlier. Every row-count check passes it. Only the date does not.
    if is_outdated(probe.last_date, as_of, max_stale_business_days):
        candidate.verdict = Verdict.STALE
        days = business_days_since(probe.last_date, as_of)
        candidate.reasons.append(
            f"last observation {probe.last_date} is {days} trading days old; "
            f"the series has stopped updating. {probe.observations} rows is a "
            f"full-looking window that ended {probe.last_date}.")
        return candidate

    if probe.observations < min_observations:
        candidate.verdict = Verdict.THIN
        candidate.reasons.append(
            f"only {probe.observations} observations, below the "
            f"{min_observations} needed to enter a covariance matrix")
        return candidate

    if probe.observations < lookback:
        candidate.verdict = Verdict.THIN
        candidate.reasons.append(
            f"{probe.observations} observations, short of the {lookback}-day "
            f"lookback; usable but it would constrain the whole window")
        return candidate

    candidate.verdict = Verdict.PASS
    if probe.adjusted is False:
        candidate.reasons.append(
            "provider returned unadjusted prices; distributions would inflate "
            "measured volatility")
    if not candidate.is_preferred_currency:
        candidate.reasons.append(
            f"quotes in {candidate.currency}, not {PREFERRED_CURRENCY}; usable "
            f"but every value needs an FX conversion")
    return candidate


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """PASS above THIN above STALE; preferred currency above foreign; then length.

    Currency outranks history length by a wide margin, and the reason is
    analytical rather than convenience. A USD-quoted line of a EUR-base fund
    produces a *different return series*, because it embeds EURUSD movement.
    Volatility computed on it measures the fund's variance plus currency
    variance. For a EUR-reporting investor the EUR-quoted line is not merely
    tidier, it is the correct series: it is the return actually realised.

    So a 320-day EUR listing beats a 500-day USD one, and not as a tiebreak.

    Worth noting and explicitly not built for v1: a long foreign line can be
    converted to EUR historically using daily FX, yielding a valid and longer
    EUR series. That is a real enhancement, but it needs a full daily FX history
    and its own error handling, so it is not attempted here.
    """
    # THIN above STALE: a short but current series still prices the holding
    # correctly and merely drops out of the covariance. A stale one prices it
    # wrongly, which is worse.
    order = {Verdict.PASS: 0, Verdict.THIN: 1, Verdict.STALE: 2,
             Verdict.FAILED: 3, Verdict.REFUSED: 4}
    return sorted(candidates, key=lambda c: (order[c.verdict],
                                             0 if c.is_preferred_currency else 1,
                                             -c.observations,
                                             c.yahoo_symbol))


# --------------------------------------------------------------------------
# The whole pipeline
# --------------------------------------------------------------------------

def resolve_from_listings(isin: str,
                          listings: list[FigiListing],
                          probe: "Callable[[str], ListingProbe]",
                          lookback: int = DEFAULT_LOOKBACK,
                          min_observations: int = MIN_OBSERVATIONS,
                          max_candidates: int = 5,
                          as_of: dt.date | None = None,
                          max_stale_business_days: int = DEFAULT_FRESHNESS_BUSINESS_DAYS,
                          ) -> Resolution:
    """Steps 3-7 over listings already in hand, probing through `probe`.

    Split out so a caching layer can supply cached listings and a cached probe
    without importing private helpers. The filtering, gating and ranking stay
    here, where they are tested; a caller that reimplemented them would be
    exactly the logic leak this separation exists to prevent.
    """
    candidates, filtered = candidates_from_listings(isin, listings)
    errors: list[str] = []
    probed: list[Candidate] = []
    for candidate in candidates:
        try:
            result = probe(candidate.yahoo_symbol)
        except Exception as exc:
            result = ListingProbe(symbol=candidate.yahoo_symbol, ok=False,
                                  error=f"{type(exc).__name__}: {exc}")
            errors.append(f"{candidate.yahoo_symbol}: {exc}")
        probed.append(_apply_probe(candidate, result, lookback, min_observations,
                                   as_of, max_stale_business_days))

    refused = [c for c in probed if c.verdict is Verdict.REFUSED]
    usable = rank_candidates([c for c in probed if c.verdict is not Verdict.REFUSED])
    return Resolution(
        isin=isin, candidates=usable[:max_candidates], refused=refused,
        listings_seen=len(listings), filtered_by_class=filtered, errors=errors)


def resolve_isin(isin: str,
                 identity: IdentityProvider,
                 prices: MarketDataProvider,
                 known: dict[str, Instrument] | None = None,
                 lookback: int = DEFAULT_LOOKBACK,
                 min_observations: int = MIN_OBSERVATIONS,
                 max_candidates: int = 5,
                 as_of: dt.date | None = None,
                 max_stale_business_days: int = DEFAULT_FRESHNESS_BUSINESS_DAYS,
                 ) -> Resolution:
    """Resolve an ISIN to ranked candidate listings for the confirm screen.

    Nothing is written. The caller presents `Resolution.candidates`, the user
    picks, and only then is an Instrument saved.
    """
    isin = (isin or "").strip().upper()

    # Step 1. The stored venue map is the cheap path and the authoritative one:
    # a symbol the user already confirmed beats anything re-derived, and a
    # manual override must survive re-resolution.
    if known and isin in known:
        inst = known[isin]
        symbol = inst.provider_symbols.get(prices.name)
        if symbol:
            probe = prices.probe(symbol, lookback)
            candidate = Candidate(
                isin=isin, ticker=symbol.split(".")[0],
                bloomberg_code="", yahoo_symbol=symbol,
                venue_label=inst.exchange or "stored", mic=inst.exchange or None,
                verdict=Verdict.FAILED, name=inst.name,
                currency=inst.quote_currency or None,
            )
            candidate = _apply_probe(candidate, probe, lookback, min_observations,
                                     as_of, max_stale_business_days)
            if inst.is_overridden(f"provider_symbols.{prices.name}"):
                candidate.reasons.append(
                    "symbol was set by hand; re-resolution will not overwrite it")
            return Resolution(isin=isin, candidates=[candidate], refused=[],
                              listings_seen=1, from_stored_map=True)

    # Step 2. Identity lookup.
    try:
        listings = identity.listings_for_isin(isin)
    except Exception as exc:
        return Resolution(isin=isin, candidates=[], refused=[],
                          errors=[f"{identity.name}: {type(exc).__name__}: {exc}"])

    # Steps 3-7. Probing costs a network call each, so only the already-filtered
    # primary venues are probed -- typically a handful.
    return resolve_from_listings(
        isin, listings, lambda symbol: prices.probe(symbol, lookback),
        lookback=lookback, min_observations=min_observations,
        max_candidates=max_candidates, as_of=as_of,
        max_stale_business_days=max_stale_business_days)


def instrument_from_candidate(candidate: Candidate, provider_name: str,
                              base_currency: str, name: str = "",
                              issuer: str = "", asset_class: str = "ETF",
                              ) -> Instrument:
    """Build the Instrument a confirmed candidate implies.

    Called only after the user confirms. `base_currency` is not inferable from
    a listing -- ISAE.AS quotes EUR for a USD-base fund -- so the confirm screen
    must collect it rather than guess.
    """
    return Instrument(
        isin=candidate.isin,
        name=name or candidate.name,
        asset_class=asset_class,
        base_currency=base_currency,
        issuer=issuer,
        primary_symbol=candidate.ticker,
        exchange=candidate.mic or yahoo_suffix_to_mic(candidate.yahoo_symbol) or "",
        quote_currency=candidate.currency or "",
        provider_symbols={provider_name: candidate.yahoo_symbol},
    )
