"""Test fixtures, and the two constraints the suite enforces rather than assumes.

1. No network. `socket.socket` is replaced for the whole session, so any test
   that reaches for the network fails loudly instead of quietly depending on a
   provider being up. Asserting this in prose would not survive contact with a
   future contributor.

2. No Streamlit. `core/` must be importable without it. That is checked in
   test_layering.py, because a green suite on a laptop that happens to have
   Streamlit installed proves nothing.
"""

from __future__ import annotations

import datetime as dt
import socket
from decimal import Decimal

import pytest

from portfolio.core.models import AssetClass, Instrument, Transaction, TransactionType
from portfolio.core.money import FxTable


@pytest.fixture(autouse=True, scope="session")
def _no_network():
    """Any socket call in the suite is a bug. Fail on it."""
    real = socket.socket

    def blocked(*args, **kwargs):
        raise RuntimeError(
            "network access attempted in the test suite; core/ must be testable "
            "offline")

    socket.socket = blocked
    yield
    socket.socket = real


D = dt.date


@pytest.fixture
def fx() -> FxTable:
    """Deliberately round rates so every expected value is hand-computable."""
    t = FxTable()
    t.add("USD", "EUR", D(2025, 1, 1), "0.90")
    t.add("USD", "EUR", D(2025, 6, 1), "0.80")
    t.add("GBP", "EUR", D(2025, 1, 1), "1.20")
    return t


@pytest.fixture
def two_asset_ledger() -> list[Transaction]:
    """The hand-computable fixture.

    A: 100 units @ 10.00 EUR, 5.00 fee  -> basis 1005.00
    B: 200 units @  4.00 EUR, 3.00 fee  -> basis  803.00
    """
    a, b = "IE0002Y8CX98", "IE000IAXNM41"
    return [
        Transaction(D(2025, 1, 10), a, TransactionType.BUY,
                    Decimal("100"), Decimal("10.00"), "EUR", Decimal("5.00")),
        Transaction(D(2025, 1, 10), b, TransactionType.BUY,
                    Decimal("200"), Decimal("4.00"), "EUR", Decimal("3.00")),
    ]


@pytest.fixture
def instruments() -> dict[str, Instrument]:
    return {
        "IE0002Y8CX98": Instrument("IE0002Y8CX98", "WisdomTree Europe Defence",
                                   AssetClass.ETF, "EUR", primary_symbol="EUDF",
                                   exchange="XETR", quote_currency="EUR"),
        "IE000IAXNM41": Instrument("IE000IAXNM41", "iShares Europe Defence",
                                   AssetClass.ETF, "EUR", primary_symbol="DFNC",
                                   exchange="XETR", quote_currency="EUR"),
        # Held zero times: the watchlist path.
        "LU1681048630": Instrument("LU1681048630", "Amundi Global Luxury",
                                   AssetClass.ETF, "EUR", primary_symbol="GLUX",
                                   exchange="XETR", quote_currency="EUR"),
    }
