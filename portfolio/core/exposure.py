"""Where the money is, sliced by instrument attribute.

A weights pie by ISIN answers "how much of each fund". This answers the
questions behind it: how much is commodity ETC rather than UCITS fund (they
carry different risks -- see `AssetClass`), how much sits with one issuer,
how much is priced in a currency other than the reporting one. None of it
is risk mathematics; it is grouping and summing, and it lives in `core`
because a grouping done in a view is a grouping nothing tests.

Only current market values are sliced, never cost, because exposure is a
statement about today. An attribute that is unset groups under "Unknown"
rather than being dropped or folded into the largest group: an issuer field
left blank on three instruments is a data-quality fact worth a slice.
"""

from __future__ import annotations

import dataclasses
import enum

from .models import AssetClass, Instrument

__all__ = ["ExposureSlice", "DIMENSIONS", "UNKNOWN", "exposure_by", "exposures"]

DIMENSIONS = ("asset_class", "issuer", "base_currency", "quote_currency", "exchange")
UNKNOWN = "Unknown"

# Labels where the raw value needs a word beside it. Everything else -- an
# issuer name, an ISO currency code, a MIC -- is its own label.
_ASSET_CLASS_LABELS = {
    AssetClass.ETF.value: "ETF",
    AssetClass.ETC.value: "ETC (commodity)",
    AssetClass.EQUITY.value: "Equity",
    AssetClass.FUND.value: "Fund",
    AssetClass.OTHER.value: "Other",
}


@dataclasses.dataclass(frozen=True)
class ExposureSlice:
    key: str
    label: str
    value: float
    weight: float
    count: int
    isins: tuple[str, ...]


def _key_for(instrument: Instrument | None, dimension: str) -> str:
    """The grouping key, with enums flattened and blanks made explicit.

    >>> from portfolio.core.models import Instrument
    >>> i = Instrument("IE0002Y8CX98", "WisdomTree Europe Defence", issuer="WisdomTree")
    >>> _key_for(i, "asset_class"), _key_for(i, "issuer"), _key_for(i, "exchange")
    ('ETF', 'WisdomTree', 'Unknown')
    >>> _key_for(None, "issuer")
    'Unknown'
    """
    if instrument is None:
        return UNKNOWN
    raw = getattr(instrument, dimension)
    if isinstance(raw, enum.Enum):
        raw = raw.value
    text = str(raw).strip() if raw is not None else ""
    return text or UNKNOWN


def exposure_by(values: dict[str, float], instruments: dict[str, Instrument],
                dimension: str) -> list[ExposureSlice]:
    """Group market values by one instrument attribute, largest slice first.

    `weight` is the slice's share of the values passed in, so weights sum to
    1 over the slices. Holdings with a zero or negative value are skipped:
    they have no exposure to report. Ties in value are broken by key so the
    order is deterministic.
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown exposure dimension {dimension!r}; one of {DIMENSIONS}")
    groups: dict[str, list[str]] = {}
    sums: dict[str, float] = {}
    total = 0.0
    for isin, value in values.items():
        value = float(value)
        if value <= 0:
            continue
        key = _key_for(instruments.get(isin), dimension)
        groups.setdefault(key, []).append(isin)
        sums[key] = sums.get(key, 0.0) + value
        total += value
    slices = [
        ExposureSlice(
            key=key,
            label=_ASSET_CLASS_LABELS.get(key, key) if dimension == "asset_class" else key,
            value=sums[key],
            weight=sums[key] / total if total > 0 else 0.0,
            count=len(members),
            isins=tuple(sorted(members)),
        )
        for key, members in groups.items()
    ]
    return sorted(slices, key=lambda s: (-s.value, s.key))


def exposures(values: dict[str, float], instruments: dict[str, Instrument]
              ) -> dict[str, list[ExposureSlice]]:
    """Every dimension at once, keyed by dimension name."""
    return {dimension: exposure_by(values, instruments, dimension) for dimension in DIMENSIONS}
