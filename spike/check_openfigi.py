#!/usr/bin/env python3
"""Validate the Bloomberg-to-Yahoo code map against live OpenFIGI.

The venue map stored on the seed instruments was confirmed by the 3 September
2026 yfinance matrix: ten ISINs with known-good Yahoo symbols and exchanges.
`portfolio/data/venues.py` claims it can reproduce every one of them from the
(ticker, Bloomberg exchange code) pair OpenFIGI returns.

A live batched run on 2026-09-03 confirmed this for all ten, and those observed
primary-venue listings are now recorded in `OBSERVED_PRIMARY_LISTINGS`. This
script re-checks that OpenFIGI still returns them: a mismatch means either the
code mapping drifted, the stored symbol drifted, or OpenFIGI changed a ticker,
and all three are worth knowing.

It also compares against the recorded listing set, so a venue disappearing from
OpenFIGI is visible even when the chosen symbol still resolves.

It cannot run in the build environment -- api.openfigi.com is blocked by the
egress policy there -- so it is a script you run, like the provider spike.

    python spike/check_openfigi.py
    python spike/check_openfigi.py --isin IE0002Y8CX98      # one instrument, verbose
    export OPENFIGI_API_KEY=...                             # optional, raises the limit

Unauthenticated the limit is about 25 requests/minute, so ten instruments in
one pass is comfortable. Calls are throttled anyway.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from portfolio.data.providers.openfigi import OpenFIGIProvider  # noqa: E402
from portfolio.data.resolve import candidates_from_listings      # noqa: E402
from portfolio.data.store import DataMode, DataStore             # noqa: E402
from portfolio.data.venues import (bloomberg_to_yahoo,           # noqa: E402
                                   classify_bloomberg_code)
from portfolio.tests.test_venues import (EXPECTED_FIGI_IDENTITY,  # noqa: E402
                                         OBSERVED_PRIMARY_LISTINGS)

SECONDS_BETWEEN_CALLS = 2.5      # ~24/min, inside the unauthenticated limit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isin", default="", help="check one ISIN and dump its listings")
    ap.add_argument("--json", default="", help="write raw responses to this path")
    args = ap.parse_args()

    store = DataStore(mode=DataMode.SEED, root=REPO / "portfolio" / "data_store")
    instruments = store.load_instruments()
    provider = OpenFIGIProvider()
    targets = [args.isin.upper()] if args.isin else sorted(instruments)

    raw: dict[str, list] = {}
    failures: list[str] = []
    print(f"{'ISIN':<14}{'rows':>5}{'cand':>6}  {'expected':<10}{'built':<10}"
          f"{'stored':<10}result")
    print("-" * 78)

    for i, isin in enumerate(targets):
        if i:
            time.sleep(SECONDS_BETWEEN_CALLS)
        inst = instruments.get(isin)
        try:
            listings = provider.listings_for_isin(isin)
        except Exception as exc:
            print(f"{isin:<14}{'-':>5}{'-':>6}  {type(exc).__name__}: {exc}")
            failures.append(f"{isin}: {exc}")
            continue

        raw[isin] = [l.__dict__ for l in listings]
        candidates, filtered = candidates_from_listings(isin, listings)
        symbols = {c.yahoo_symbol for c in candidates}

        expected = EXPECTED_FIGI_IDENTITY.get(isin)
        stored = inst.provider_symbols.get("yfinance", "-") if inst else "-"
        built = bloomberg_to_yahoo(*expected) if expected else None

        if stored == "-":
            result = "no stored symbol"
        elif built != stored:
            result = f"MAP MISMATCH (table says {built})"
            failures.append(f"{isin}: table builds {built}, stored {stored}")
        elif stored not in symbols:
            result = "OPENFIGI MISSING (stored symbol not among candidates)"
            failures.append(f"{isin}: OpenFIGI did not return the stored listing")
        else:
            recorded = set(OBSERVED_PRIMARY_LISTINGS.get(isin, []))
            gone = recorded - symbols
            new = symbols - recorded
            result = "ok"
            if gone:
                result += f"  (no longer listed: {', '.join(sorted(gone))})"
            if new:
                result += f"  (new: {', '.join(sorted(new))})"

        exp = f"{expected[0]}+{expected[1]}" if expected else "-"
        print(f"{isin:<14}{len(listings):>5}{len(candidates):>6}  {exp:<10}"
              f"{str(built or '-'):<10}{stored:<10}{result}")

        if args.isin:
            print(f"\n  filtered by class: {filtered}")
            print("  candidates:")
            for c in candidates:
                print(f"    {c.yahoo_symbol:<12}{c.ticker:<10}{c.bloomberg_code:<4}"
                      f"{c.venue_label}")
            print("\n  all listings:")
            for l in listings:
                print(f"    {l.ticker:<12}{l.exchange_code:<4}"
                      f"{classify_bloomberg_code(l.exchange_code).value:<16}{l.name}")

    if args.json and raw:
        pathlib.Path(args.json).write_text(json.dumps(raw, indent=2))
        print(f"\nraw responses -> {args.json}")

    print()
    if failures:
        print(f"{len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {len(targets)} instruments reproduce their stored symbol.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
