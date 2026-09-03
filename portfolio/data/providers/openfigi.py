"""OpenFIGI identity provider: ISIN to every listing, from Bloomberg.

Free, unauthenticated, ISIN-native. Roughly 25 requests/minute without a key
and more with one -- ample, because resolution happens once per instrument and
the answer is stored in `provider_symbols` rather than re-derived.

This module deliberately does no filtering. It returns everything OpenFIGI
says, including the seventy-odd rows a single ETF ISIN produces, because the
filtering has to be visible and testable in `data.resolve` rather than buried
here where nobody would see what was discarded.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ..provider import FigiListing, IdentityProvider, ProviderError, RateLimited

__all__ = ["OpenFIGIProvider", "parse_mapping_response"]

ENDPOINT = "https://api.openfigi.com/v3/mapping"


def parse_mapping_response(payload: list[dict]) -> list[FigiListing]:
    """Turn an OpenFIGI /v3/mapping response into listings.

    Split out from the HTTP call so the parsing is testable offline against
    recorded fixtures -- which matters, because the shape of this response is
    the whole reason the resolver needs a filtering step.
    """
    out: list[FigiListing] = []
    for block in payload or []:
        if not isinstance(block, dict):
            continue
        if block.get("warning") and not block.get("data"):
            continue
        for row in block.get("data") or []:
            ticker = (row.get("ticker") or "").strip()
            code = (row.get("exchCode") or "").strip()
            if not ticker or not code:
                continue
            out.append(FigiListing(
                figi=row.get("figi", ""),
                ticker=ticker,
                exchange_code=code,
                security_type=row.get("securityType") or row.get("securityType2") or "",
                name=(row.get("name") or "").strip(),
            ))
    return out


class OpenFIGIProvider(IdentityProvider):
    name = "openfigi"

    def __init__(self, api_key: str | None = None, timeout: int = 20) -> None:
        # A key raises the rate limit but is not required; the free
        # unauthenticated tier is enough for once-per-instrument resolution.
        self.api_key = api_key or os.environ.get("OPENFIGI_API_KEY") or ""
        self.timeout = timeout

    def listings_for_isin(self, isin: str) -> list[FigiListing]:
        body = json.dumps([{"idType": "ID_ISIN", "idValue": isin.strip().upper()}])
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-OPENFIGI-APIKEY"] = self.api_key

        request = urllib.request.Request(ENDPOINT, data=body.encode("utf-8"),
                                         headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimited(self.name, "rate limit hit; serve cached data "
                                             "and warn rather than blanking") from exc
            raise ProviderError(self.name, f"HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(self.name, f"unreachable: {exc.reason}") from exc
        return parse_mapping_response(payload)
