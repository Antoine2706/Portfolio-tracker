"""The reference check: T5, T6 and T7 on the exact bars the survey reads.

Built after the first real ladder run put SPY at about 19 bps against a true
half-spread near one, and every one of eleven lines between 8 and 34
whatever its truth. What can be tested here is that the tool asks the three
questions correctly and reports what it gets; what it will say about SPY
needs SPY's bars and is the user's run.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from portfolio.core.spread import edge, edge_components
from portfolio.data.providers.fixture import FixtureProvider
from portfolio.eval.spread_controls import simulate_bars
from portfolio.research import reference_check


class TestTheFourMomentConditions:
    def test_their_weighted_sum_is_the_estimator_exactly(self):
        for seed in range(4):
            o, h, l, c = simulate_bars(0.003, bars=700, seed=seed)
            parts = edge_components(o, h, l, c)
            assert parts.signed_square == pytest.approx(
                edge(o, h, l, c).signed_square, abs=1e-15)

    def test_a_contaminated_open_shows_in_the_open_products_only(self):
        """The diagnostic the components exist for: which prices carry the
        bounce. An open pushed off the market by 30 bps on a fifth of days
        inflates r1 r2 and r1 r5 and leaves r3 r4 where it was."""
        o, h, l, c = simulate_bars(2.0 * 5.0 / 10_000.0, bars=3000, seed=5)
        clean = edge_components(o, h, l, c)
        rng = np.random.default_rng(1)
        o2 = o.copy()
        picked = rng.choice(np.arange(1, 3000), 600, replace=False)
        o2[picked] *= np.where(rng.random(600) < 0.5, 1.003, 0.997)
        h2, l2 = np.maximum(h, o2), np.minimum(l, o2)
        dirty = edge_components(o2, h2, l2, c)
        assert dirty.open_previous_mid > 3 * clean.open_previous_mid
        assert dirty.open_previous_close > 3 * clean.open_previous_close
        assert abs(dirty.close_previous_mid - clean.close_previous_mid) \
            < 0.5 * abs(clean.close_previous_mid) + 1e-7

    def test_a_contaminated_close_shows_in_the_close_products_only(self):
        o, h, l, c = simulate_bars(2.0 * 5.0 / 10_000.0, bars=3000, seed=6)
        clean = edge_components(o, h, l, c)
        rng = np.random.default_rng(2)
        c2 = c.copy()
        picked = rng.choice(np.arange(0, 2999), 600, replace=False)
        c2[picked] *= np.where(rng.random(600) < 0.5, 1.003, 0.997)
        h2, l2 = np.maximum(h, c2), np.minimum(l, c2)
        dirty = edge_components(o, h2, l2, c2)
        assert dirty.close_previous_mid > 3 * clean.close_previous_mid
        assert abs(dirty.open_previous_mid - clean.open_previous_mid) \
            < 0.5 * abs(clean.open_previous_mid) + 1e-7

    def test_a_refusal_is_none(self):
        assert edge_components(*[np.full(50, 7.5)] * 4) is None


class Simulated(FixtureProvider):
    """A provider whose bars are the controls' simulator, at a known spread."""
    name = "fixture"

    def __init__(self, half_bps: float, bars: int = 2000) -> None:
        super().__init__()
        self.half_bps, self.bars_count = half_bps, bars

    def bars(self, symbol, start=None, *, period="2y"):
        import pandas as pd
        o, h, l, c = simulate_bars(2.0 * self.half_bps / 10_000.0,
                                   bars=self.bars_count, seed=42, tick_size=0.01)
        index = pd.bdate_range(end=pd.Timestamp.today().normalize(),
                               periods=self.bars_count)
        frame = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                              "volume": np.full(self.bars_count, 1e5)},
                             index=index)
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(start)]
        return frame


class TestTheReferenceCheck:
    @pytest.fixture
    def report(self, tmp_path):
        return reference_check("SIM", provider=Simulated(12.0),
                               data_root=tmp_path)

    def test_t5_runs_the_authors_package_on_the_same_arrays(self, report):
        pytest.importorskip("bidask", reason="the authors' package is not installed")
        assert report.theirs_half_bps is not None, report.theirs_note
        assert report.theirs_half_bps == pytest.approx(report.ours_half_bps, abs=1e-9)
        text = "\n".join(report.lines())
        assert "the two agree" in text

    def test_t5_says_so_when_the_package_is_missing(self, tmp_path, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "bidask", None)   # import raises
        report = reference_check("SIM", provider=Simulated(12.0), data_root=tmp_path)
        assert report.theirs_half_bps is None
        assert "not installed" in report.theirs_note

    def test_t6_runs_the_adjusted_series_and_a_constant_factor_changes_nothing(
            self, tmp_path):
        """The fixture's adjustment steps the factor every 63 bars; between
        the steps it is constant within a bar and cancels in every log
        ratio, so the two estimates sit within their error bars."""
        report = reference_check("SIM", provider=Simulated(12.0), data_root=tmp_path)
        assert report.adjusted is not None, report.adjusted_note
        gap = abs(report.adjusted.half_spread_bps - report.ours.half_spread_bps)
        assert gap < 2.0 * report.ours.half_spread_error_bps, (
            report.adjusted.describe(), report.ours.describe())

    def test_t7_cuts_the_history_at_the_dates_given(self, tmp_path):
        provider = Simulated(12.0, bars=2000)
        last = provider.bars("SIM").index[-1].date()
        cut_a = last - dt.timedelta(days=700)
        cut_b = last - dt.timedelta(days=300)
        report = reference_check("SIM", provider=provider, data_root=tmp_path,
                                 splits=(cut_a, cut_b))
        assert len(report.blocks) == 3
        assert report.blocks[0].last < cut_a <= report.blocks[1].first
        assert report.blocks[1].last < cut_b <= report.blocks[2].first
        assert sum(b.quality.bars for b in report.blocks) == report.rows
        for b in report.blocks:
            assert b.estimate.spread is not None and b.components is not None

    def test_the_report_prints_every_section_and_records_them(self, report):
        text = "\n".join(report.lines())
        for heading in ("T5  the authors' package", "T6  adjusted against",
                        "T7  blocks by date", "moment condition",
                        "per-bar autocorrelation"):
            assert heading in text, heading
        record = report.record()
        assert record["ours"]["half_spread_bps"] == pytest.approx(
            report.ours.half_spread_bps)
        assert record["components"]["weight"] == pytest.approx(report.components.weight)
        assert math.isfinite(record["ours_half_bps"])

    def test_the_command_line_runs_it(self, tmp_path, capsys):
        from portfolio.cli import main
        code = main(["controls", "--spread-reference", "IUSA.AS", "--provider",
                     "fixture", "--data-root", str(tmp_path),
                     "--split-at", "2025-06-01"])
        out = capsys.readouterr().out
        assert code == 0
        assert "Reference check: IUSA.AS on fixture" in out
        assert "T7  blocks by date" in out and "start to 2025-06-01" in out


class TestTheDriftTestAgainstAKnownBreak:
    """Point 3 of the brief: sabotage the drift test against a break like
    decimalisation and see whether it bites. It does, on a break the size
    of its own reference block's noise or larger; below that it cannot,
    because it compares every older block against the most recent 250
    bars, the noisiest window it has. Both halves are pinned."""

    def test_a_break_an_order_of_magnitude_wide_is_caught(self):
        from portfolio.core.spread import sweep_windows
        old = simulate_bars(2.0 * 30.0 / 10_000.0, bars=2000, sigma=0.012, seed=1)
        new = simulate_bars(2.0 * 1.0 / 10_000.0, bars=6000, sigma=0.012, seed=2)
        o, h, l, c = (np.concatenate([a, b]) for a, b in zip(old, new))
        sweep = sweep_windows(o, h, l, c)
        assert sweep.drifted_at is not None and sweep.chosen_bars <= 1000, (
            sweep.reason)

    def test_a_break_under_the_reference_blocks_floor_is_not(self):
        """Pre-decimal SPY at a few bps for 2000 bars, then 6000 at one:
        the old block's excess is about the size of the 250-bar floor, and
        the test is silent. Recorded as a measured limit of the design, not
        fixed here: the reference block is the choice to change, and that
        waits on the reference check."""
        from portfolio.core.spread import sweep_windows
        old = simulate_bars(2.0 * 5.0 / 10_000.0, bars=2000, sigma=0.012, seed=3)
        new = simulate_bars(2.0 * 1.0 / 10_000.0, bars=6000, sigma=0.012, seed=4)
        o, h, l, c = (np.concatenate([a, b]) for a, b in zip(old, new))
        sweep = sweep_windows(o, h, l, c)
        assert sweep.drifted_at is None, (
            "the drift test now catches a break under its reference block's "
            "floor; update this test and the note in ARCHITECTURE.md")
