"""An OpenFIGI response shaped like the real one for IE0002Y8CX98.

Reconstructed from the reported structure of the live call (70+ rows across
primary venues, German regionals, Tradegate, MTFs and dark venues, plus
separate currency and share-class lines), not a verbatim capture. The exact
FIGIs are synthetic; the ticker/exchange-code combinations are the point and
they are the ones actually observed.

It exists so the filtering step can be tested against realistic noise. A
fixture with three tidy rows would prove nothing -- the whole design problem is
that seventy candidates arrive and only a handful are usable.
"""

from __future__ import annotations

from portfolio.data.provider import FigiListing

NAME = "WISDOMTREE EUROPE DEFENCE UCIT"


def _row(ticker: str, code: str, n: int) -> dict:
    return {"figi": f"BBG{n:09d}", "ticker": ticker, "exchCode": code,
            "securityType": "ETP", "name": NAME}


def _rows() -> list[dict]:
    out: list[dict] = []
    n = 0

    def add(ticker: str, codes: list[str]) -> None:
        nonlocal n
        for code in codes:
            n += 1
            out.append(_row(ticker, code, n))

    # Primary venues -- the ones that should survive filtering.
    add("WDEF", ["LN", "IM"])
    add("EUDF", ["GR"])
    # German regionals and Tradegate.
    add("EUDF", ["GF", "GD", "GS", "GM", "GH", "GT", "GZ", "TH"])
    # MTFs and dark venues, per currency line.
    add("WDEFEUR", ["EP", "EZ", "EO", "X2", "XH", "XF", "XJ", "XD", "XP", "XS",
                    "XB", "XL", "XV", "XN", "XT", "XA", "XC", "XE", "XG", "XK"])
    add("WDEFGBX", ["EP", "EZ", "EO", "X2", "XH", "XF", "XJ", "XD", "XP", "XS",
                    "XB", "XL", "XV", "XN", "XT", "XA", "XC", "XE"])
    add("WDEFUSD", ["EP", "EZ", "EO", "X2", "XH", "XF", "XJ", "XD", "XP", "XS",
                    "XB", "XL", "XV", "XN"])
    # Other share and currency lines.
    add("WDEFM", ["LN"])
    add("WDEFL", ["LN"])
    add("EUDFD", ["GR"])
    add("WDEPL", ["LN"])
    add("WDEP", ["LN"])
    return out


WDEF_MAPPING_RESPONSE: list[dict] = [{"data": _rows()}]


def wdef_listings() -> list[FigiListing]:
    from portfolio.data.providers.openfigi import parse_mapping_response
    return parse_mapping_response(WDEF_MAPPING_RESPONSE)


NO_MATCH_RESPONSE: list[dict] = [{"warning": "No identifier found."}]
