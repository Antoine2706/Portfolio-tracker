"""Alignment: the step most likely to be silently wrong.

The calendars used here are the real ones from the venue map -- Xetra,
Amsterdam, Milan and Paris -- because that is where the non-overlapping
holidays actually come from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.core.returns import (DEFAULT_LOOKBACK, InsufficientHistory,
                                    align_returns, simple_returns)

WDEF = "IE0002Y8CX98"      # EUDF.DE, 377 days in the real matrix
DFNC = "IE000IAXNM41"      # DFNC.DE, 320 days -- the binding instrument
GRAINS = "GB00B15KYL00"    # AIGG.MI, 503 days
LUXURY = "LU1681048630"    # GLUX.PA, 508 days


def series(n: int, *, end="2026-09-03", drop=(), seed=0, start_price=100.0):
    """A price series of n business days ending at `end`, minus `drop` dates."""
    idx = pd.bdate_range(end=end, periods=n)
    if drop:
        idx = idx.drop([pd.Timestamp(d) for d in drop if pd.Timestamp(d) in idx])
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1, len(idx)).cumsum()
    return pd.Series(start_price + steps, index=idx)


class TestSimpleReturns:
    def test_formula(self):
        px = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
        assert [round(r, 10) for r in simple_returns(px)["A"]] == [0.1, -0.1]

    def test_first_row_dropped(self):
        assert len(simple_returns(pd.DataFrame({"A": [1.0, 2.0, 3.0]}))) == 2


class TestCalendarAlignment:
    def test_intersection_of_mismatched_calendars(self):
        """15 August trades in Frankfurt but not Milan or Paris."""
        de = series(300, seed=1)
        mi = series(300, seed=2, drop=["2026-08-14", "2025-08-15"])
        returns, report = align_returns({WDEF: de, GRAINS: mi}, lookback=DEFAULT_LOOKBACK)
        assert set(returns.index) <= set(de.index) & set(mi.index)
        assert not returns.isna().any().any(), "no NaNs may survive alignment"

    def test_dropped_observations_are_counted(self):
        de = series(300, seed=1)
        mi = series(300, seed=2, drop=["2026-08-14"])
        _, report = align_returns({WDEF: de, GRAINS: mi})
        assert report.total_dropped > 0
        assert set(report.dropped_in_alignment) == {WDEF, GRAINS}

    def test_every_instrument_gets_the_same_window(self):
        """Window lengths are never mixed -- the spec is explicit about this."""
        returns, _ = align_returns({WDEF: series(300, seed=1),
                                    GRAINS: series(500, seed=2),
                                    LUXURY: series(508, seed=3)})
        assert returns.notna().all().all()
        assert len(set(returns.count())) == 1

    def test_non_overlapping_calendars_refuse(self):
        a = series(100, end="2020-06-01", seed=1)
        b = series(100, end="2026-09-03", seed=2)
        with pytest.raises(InsufficientHistory, match="common to all"):
            align_returns({WDEF: a, GRAINS: b})


class TestThinSeriesTrap:
    """Trap 2 from the matrix: AIGG.L and AIGE.L return 2 rows and look fine."""

    def test_thin_series_is_excluded_not_intersected(self):
        thin = pd.Series([10.0, 10.1], pd.to_datetime(["2026-07-17", "2026-09-03"]))
        returns, report = align_returns({WDEF: series(400, seed=1),
                                         LUXURY: series(500, seed=2),
                                         GRAINS: thin})
        assert GRAINS not in returns.columns
        assert len(returns) > 100, "a 2-row series must not truncate the others"

    def test_exclusion_names_the_instrument_and_the_count(self):
        thin = pd.Series([10.0, 10.1], pd.to_datetime(["2026-07-17", "2026-09-03"]))
        _, report = align_returns({WDEF: series(400, seed=1), GRAINS: thin})
        assert [e.isin for e in report.excluded] == [GRAINS]
        assert report.excluded[0].observations == 2
        assert "below the 60 minimum" in report.excluded[0].reason

    def test_all_thin_refuses_rather_than_returning_two_rows(self):
        thin = pd.Series([10.0, 10.1], pd.to_datetime(["2026-07-17", "2026-09-03"]))
        with pytest.raises(InsufficientHistory, match="too little history"):
            align_returns({WDEF: thin, GRAINS: thin})


class TestShortWindow:
    """The expected state whenever a recently launched thematic ETF is held."""

    def test_window_degrades_rather_than_failing(self):
        returns, report = align_returns({DFNC: series(320, seed=1),
                                         LUXURY: series(508, seed=2)},
                                        lookback=DEFAULT_LOOKBACK)
        assert report.effective_lookback == DEFAULT_LOOKBACK
        assert not report.window_was_shortened

    def test_shortening_names_the_binding_instrument(self):
        """DFNC has 320 days; ask for 400 and it is what constrains you."""
        returns, report = align_returns({DFNC: series(320, seed=1),
                                         LUXURY: series(508, seed=2)},
                                        lookback=400)
        assert report.window_was_shortened
        assert report.binding_instrument == DFNC
        assert report.binding_observations == 320
        assert report.effective_lookback == 319     # 320 prices -> 319 returns

    def test_summary_states_the_window_and_the_constraint(self):
        _, report = align_returns({DFNC: series(320, seed=1),
                                   LUXURY: series(508, seed=2)}, lookback=400)
        text = report.summary()
        assert "319 daily returns" in text
        assert DFNC in text and "400 requested" in text

    def test_warning_says_windows_are_not_mixed(self):
        _, report = align_returns({DFNC: series(320, seed=1),
                                   LUXURY: series(508, seed=2)}, lookback=400)
        assert any("never mixed" in w for w in report.warnings)

    def test_the_real_portfolio_shape_survives_252(self):
        """All ten instruments at their measured row counts, 252 lookback."""
        measured = {WDEF: 377, DFNC: 320, "IE000I7E6HL0": 356, "IE000OJ5TQP4": 505,
                    "IE00B6R52143": 508, GRAINS: 503, "JE00BN7KB664": 503,
                    "GB00B15KYB02": 503, "IE00BMW42637": 490, LUXURY: 508}
        prices = {isin: series(n, seed=i) for i, (isin, n) in enumerate(measured.items())}
        returns, report = align_returns(prices, lookback=252)
        assert report.effective_lookback == 252, report.summary()
        assert len(returns.columns) == 10
        assert not report.excluded


class TestRefusals:
    def test_empty_portfolio(self):
        with pytest.raises(InsufficientHistory, match="no price series"):
            align_returns({})

    def test_single_holding_aligns_fine(self):
        """Covariance of one asset is a 1x1 matrix; that is legal here."""
        returns, report = align_returns({WDEF: series(300, seed=1)})
        assert returns.shape[1] == 1

    def test_lookback_below_two_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            align_returns({WDEF: series(300)}, lookback=1)

    def test_duplicate_dates_are_collapsed(self):
        s = series(200, seed=1)
        dupes = pd.concat([s, s.iloc[[-1]]])
        returns, _ = align_returns({WDEF: dupes, LUXURY: series(200, seed=2)})
        assert returns.index.is_unique
