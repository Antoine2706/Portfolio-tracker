"""Naming the holding that made the window, when one did.

A backtest over a period in which one holding nearly doubled is not a test of
a policy; it is a measurement of what that policy did about that holding. On
the real book IE00BMC38736 went from 61.12 to 109.16 -- up 78.6% -- while
everything else ran between -6% and +13%. Equal risk contribution trims the
most volatile holding, the most volatile holding was the one that ran, and
the verdict was read for weeks without that sentence next to it.

So the report says it. Computed rather than asserted, because the same note
on a window where nothing dominated would be a confident statement about
nothing -- which is the failure mode this project keeps finding, in prose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.research import names_the_window

NAMES = {"A": "Alpha ETF", "B": "Beta ETF", "C": "Gamma ETF", "D": "Delta ETF"}


def panel(totals, periods: int = 300) -> pd.DataFrame:
    """A price panel in which each column ends up at exactly `totals[column]`."""
    dates = pd.bdate_range("2026-01-01", periods=periods)
    out = {}
    for key, total in totals.items():
        out[key] = 100.0 * np.linspace(1.0, 1.0 + total, periods)
    return pd.DataFrame(out, index=dates)


class TestItNamesTheHoldingWhenOneRan:
    def test_the_real_shape_of_the_window(self):
        """One holding up 78.6%, the rest between -6% and +13%."""
        note = names_the_window(
            panel({"A": 0.786, "B": -0.06, "C": 0.13, "D": 0.04}), NAMES)
        assert note is not None
        assert "Alpha ETF" in note and "(A)" in note
        assert "+78.6%" in note
        assert "-6.0%" in note and "+13.0%" in note

    def test_it_says_what_the_reader_is_to_do_with_it(self):
        note = names_the_window(
            panel({"A": 0.786, "B": -0.06, "C": 0.13, "D": 0.04}), NAMES)
        assert "most volatile holding was the one that ran" in note
        assert "rather than as a property of the policy" in note

    def test_it_falls_back_to_the_isin_when_there_is_no_name(self):
        note = names_the_window(
            panel({"A": 0.786, "B": -0.06, "C": 0.13, "D": 0.04}), {})
        assert note is not None and "A" in note


class TestItStaysQuietWhenNothingDid:
    def test_an_even_window_gets_no_note(self):
        assert names_the_window(
            panel({"A": 0.12, "B": 0.09, "C": 0.11, "D": 0.10}), NAMES) is None

    def test_a_falling_window_gets_no_note(self):
        """The leader is the least bad, which explains nothing."""
        assert names_the_window(
            panel({"A": -0.02, "B": -0.30, "C": -0.25, "D": -0.19}),
            NAMES) is None

    def test_the_threshold_is_a_multiple_and_it_bites(self):
        """Three times the median of the others. Just under and just over."""
        under = panel({"A": 0.29, "B": 0.10, "C": 0.10, "D": 0.10})
        over = panel({"A": 0.31, "B": 0.10, "C": 0.10, "D": 0.10})
        assert names_the_window(under, NAMES) is None
        assert names_the_window(over, NAMES) is not None

    def test_two_holdings_are_not_a_window_one_holding_made(self):
        """If two ran, no single one explains the result and naming either
        would be picking a story out of a pair."""
        both = panel({"A": 0.80, "B": 0.75, "C": 0.05, "D": 0.04})
        assert names_the_window(both, NAMES) is None


class TestItRefusesRatherThanGuessing:
    def test_a_panel_too_short_to_measure(self):
        assert names_the_window(panel({"A": 0.8, "B": 0.1, "C": 0.1}, periods=1),
                                NAMES) is None

    def test_a_book_of_two_has_no_others_to_compare_against(self):
        assert names_the_window(panel({"A": 0.8, "B": 0.02}), NAMES) is None

    def test_a_column_with_no_prices_is_ignored_rather_than_fatal(self):
        prices = panel({"A": 0.786, "B": -0.06, "C": 0.13, "D": 0.04})
        prices["E"] = np.nan
        note = names_the_window(prices, NAMES)
        assert note is not None and "Alpha ETF" in note

    def test_a_late_listing_is_measured_from_its_own_first_price(self):
        """Not from a back-filled one, which would invent a return it never
        had. It joins the comparison on its own terms or not at all."""
        prices = panel({"A": 0.786, "B": -0.06, "C": 0.13, "D": 0.04})
        prices.iloc[:200, prices.columns.get_loc("D")] = np.nan
        note = names_the_window(prices, NAMES)
        assert note is not None
        assert "Alpha ETF" in note
