"""Everything that needs the user's attention, as one ordered list.

The individual facts already exist: a row carries its stale-price sentence,
the alignment report names what it excluded, the correlation pairs are
listed on the risk view. What none of them can do is compete for attention.
An unpriced holding matters more than a shortened window, and a reader who
has to visit four pages to find that out will not. So the rules live here,
each with a severity, and the result is sorted so the first alert is the
one to act on.

Severity is deliberately coarse:

    CRITICAL    reserved; nothing in the current rules earns it, and having
                the level means a future rule (a ledger that does not
                replay, say) has somewhere to go above SERIOUS
    SERIOUS     the numbers on screen are incomplete or the portfolio is
                carrying a risk it does not show: an unpriced holding, a
                failed fetch, a cluster of holdings that are one bet
    WARNING     a figure that deserves a second look: a stale price, a
                position over the concentration limit, a correlated pair
    INFO        a fact about how the numbers were produced, no action implied

Every alert names the route where it can be acted on, so the UI can turn
the title into a link rather than describe the journey.
"""

from __future__ import annotations

import dataclasses
import enum

from .models import Instrument
from .naming import short_names
from .report import HoldingsTable
from .returns import AlignmentReport
from .risk import ConcentrationStats, CorrelationCluster, CorrelationPair

__all__ = ["Severity", "Alert", "build_alerts", "ROUTES"]

ROUTES = {
    "holdings": "#/holdings",
    "risk": "#/risk",
    "instruments": "#/instruments",
    "settings": "#/settings",
}


class Severity(str, enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    SERIOUS = "SERIOUS"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """Higher is more urgent.

        Explicit because a str enum sorts alphabetically, and alphabetically
        CRITICAL comes before INFO. Sorting on the value would put the most
        urgent alert first by accident and the least urgent second.

        >>> Severity.CRITICAL.rank > Severity.SERIOUS.rank > Severity.WARNING.rank > Severity.INFO.rank
        True
        """
        return _RANK[self]


_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.SERIOUS: 2, Severity.CRITICAL: 3}


@dataclasses.dataclass(frozen=True)
class Alert:
    code: str
    severity: Severity
    title: str
    detail: str
    isins: tuple[str, ...] = ()
    route: str = ""             # SPA route to act on it


def _name(instruments: dict[str, Instrument], isin: str) -> str:
    """The short display name. An alert title is a label, not a record.

    Derived across the whole universe rather than per instrument, so two funds
    tracking the same thing from different houses never both come back as the
    same string -- an alert naming two holdings identically is worse than one
    that spells them out in full.
    """
    if not instruments:
        return isin
    return short_names(instruments).get(isin, isin)


def _mentions_stale_price(warning: str) -> bool:
    text = warning.lower()
    return "not a current price" in text or "not a live quote" in text


def build_alerts(holdings: HoldingsTable, instruments, *,
                 fetch_failures: list[str] = (),
                 concentration: ConcentrationStats | None = None,
                 pairs: list[CorrelationPair] = (),
                 clusters: list[CorrelationCluster] = (),
                 alignment: AlignmentReport | None = None,
                 max_weight: float = 0.35) -> list[Alert]:
    """Apply every rule and return the alerts most urgent first.

    Rules, with their severity:

        unpriced holding             SERIOUS   one per holding in `holdings.unpriced`
        fetch failure                SERIOUS   one per string in `fetch_failures`
        stale or outdated price      WARNING   a row whose warnings say so
        holding above `max_weight`   WARNING
        effective holdings < half    WARNING   `concentration`, needs >= 2 holdings
        high-correlation pair        WARNING   unless both sit in a reported
                                               cluster of three or more, which
                                               already says it once
        cluster of 3+                SERIOUS   a two-member cluster is a pair
        excluded from risk model     INFO      one per `alignment.excluded`
        window shortened             INFO
        watchlist-only universe      INFO      nothing held, something watched

    Sorted by severity, then title, then detail, so the order is stable
    across runs and two alerts with the same title cannot swap places.
    """
    instruments = instruments or {}
    out: list[Alert] = []

    for isin in holdings.unpriced:
        row = next((r for r in holdings.rows if r.isin == isin), None)
        detail = (" ".join(row.warnings) if row is not None and row.warnings
                  else "No price could be obtained, so this holding is not in the total.")
        out.append(Alert("unpriced_holding", Severity.SERIOUS,
                         f"{_name(instruments, isin)} has no price", detail,
                         (isin,), ROUTES["holdings"]))

    for message in fetch_failures:
        out.append(Alert("fetch_failure", Severity.SERIOUS, "Price fetch failed",
                         str(message), (), ROUTES["settings"]))

    for row in holdings.rows:
        if row.isin in holdings.unpriced:
            continue
        stale = [w for w in row.warnings if _mentions_stale_price(w)]
        if stale:
            out.append(Alert("stale_price", Severity.WARNING,
                             f"Price for {_name(instruments, row.isin)} is not "
                             f"current", " ".join(stale),
                             (row.isin,), ROUTES["holdings"]))
        if row.weight is not None and float(row.weight) > max_weight:
            out.append(Alert(
                "concentrated_holding", Severity.WARNING,
                f"{_name(instruments, row.isin)} is {float(row.weight):.0%} "
                f"of the portfolio",
                f"Above the {max_weight:.0%} limit for a single holding. Whatever happens "
                f"to this one position happens to the portfolio.",
                (row.isin,), ROUTES["risk"]))

    if (concentration is not None and concentration.actual_holdings >= 2
            and concentration.effective_holdings < 0.5 * concentration.actual_holdings):
        out.append(Alert(
            "low_effective_holdings", Severity.WARNING,
            f"{concentration.actual_holdings} holdings behave like "
            f"{concentration.effective_holdings:.1f}",
            f"{concentration.summary()}. Fewer than half the positions are doing "
            f"independent work; the rest are weight, not diversification.",
            (), ROUTES["risk"]))

    big_clusters = [c for c in clusters if len(c.members) >= 3]
    clustered = [set(c.members) for c in big_clusters]
    for cluster in big_clusters:
        names = ", ".join(_name(instruments, m) for m in cluster.members)
        weight = (f" Together they are {cluster.combined_weight:.0%} of the portfolio."
                  if cluster.combined_weight is not None else "")
        out.append(Alert(
            "correlation_cluster", Severity.SERIOUS,
            f"{len(cluster.members)} holdings are one bet",
            f"{names} move together (mean correlation {cluster.mean_correlation:.0%}, "
            f"lowest pair {cluster.min_correlation:.0%}). Pairwise checks report edges; "
            f"this is the group.{weight}",
            tuple(cluster.members), ROUTES["risk"]))

    for pair in pairs:
        if any({pair.a, pair.b} <= members for members in clustered):
            continue
        out.append(Alert(
            "correlated_pair", Severity.WARNING,
            f"{_name(instruments, pair.a)} and {_name(instruments, pair.b)} move together",
            pair.sentence(short_names(instruments)),
            (pair.a, pair.b), ROUTES["risk"]))

    if alignment is not None:
        for excluded in alignment.excluded:
            out.append(Alert(
                "excluded_from_risk_model", Severity.INFO,
                f"{_name(instruments, excluded.isin)} is not in the risk model",
                excluded.reason, (excluded.isin,), ROUTES["instruments"]))
        if alignment.window_was_shortened:
            out.append(Alert(
                "window_shortened", Severity.INFO,
                f"Risk window shortened to {alignment.effective_lookback} days",
                alignment.summary(), tuple(
                    x for x in (alignment.binding_instrument,) if x), ROUTES["risk"]))

    if holdings.is_empty and holdings.watchlist:
        out.append(Alert(
            "watchlist_only", Severity.INFO,
            "Nothing held yet",
            f"{len(holdings.watchlist)} instrument(s) on the watchlist and no "
            f"transactions. Record a purchase to see holdings and risk.",
            tuple(holdings.watchlist), ROUTES["holdings"]))

    return sorted(out, key=lambda a: (-a.severity.rank, a.title, a.detail))
