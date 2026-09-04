"""Display projections and their plain sentences.

These are tested because they are logic. If they lived in the views they would
be arithmetic nothing checks, which is the erosion the layering guard cannot
catch on its own.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from portfolio.core.models import AssetClass, Instrument, Transaction, TransactionType as T
from portfolio.core.money import FxTable, Money
from portfolio.core.positions import PriceQuote, derive_positions
from portfolio.core.report import (divergence_rows, holdings_table,
                                   correlation_sentences, risk_metrics)
from portfolio.core.risk import (concentration, correlation_matrix, covariance_matrix,
                                 diversification_ratio, drawdown, high_correlation_pairs,
                                 risk_decomposition)

D = dt.date
A, B = "IE0002Y8CX98", "IE000IAXNM41"
TODAY = D(2026, 9, 3)


@pytest.fixture
def instruments():
    return {
        A: Instrument(A, "WisdomTree Europe Defence", AssetClass.ETF, "EUR"),
        B: Instrument(B, "iShares Europe Defence", AssetClass.ETF, "EUR"),
        "LU1681048630": Instrument("LU1681048630", "Amundi Global Luxury",
                                   AssetClass.ETF, "EUR"),
    }


@pytest.fixture
def ledger():
    return [
        Transaction(D(2025, 1, 10), A, T.BUY, Decimal("100"), Decimal("10.00"), "EUR"),
        Transaction(D(2025, 1, 10), B, T.BUY, Decimal("200"), Decimal("4.00"), "EUR"),
    ]


def quote(price, *, as_of=TODAY, ccy="EUR", stale=False, delay=15):
    return PriceQuote(Money(Decimal(price), ccy), as_of=as_of, source="test",
                      delay_minutes=delay, is_stale=stale)


class TestHoldingsTable:
    def test_rows_sorted_by_value_descending(self, instruments, ledger):
        pos = derive_positions(ledger, instruments,
                               quotes={A: quote("12.00"), B: quote("4.00")})
        table = holdings_table(pos, instruments, as_of=TODAY)
        assert [r.isin for r in table.rows] == [A, B]      # 1200 then 800

    def test_totals_and_weights(self, instruments, ledger):
        pos = derive_positions(ledger, instruments,
                               quotes={A: quote("12.00"), B: quote("4.00")})
        table = holdings_table(pos, instruments, as_of=TODAY)
        assert table.total_value == Money(Decimal("2000.00"), "EUR")
        assert sum(r.weight for r in table.rows) == Decimal("1")

    def test_unpriced_holding_is_excluded_and_named(self, instruments, ledger):
        """A total that silently omits a holding is wrong in a way nobody notices."""
        pos = derive_positions(ledger, instruments, quotes={A: quote("12.00")})
        table = holdings_table(pos, instruments, as_of=TODAY)
        assert table.total_value == Money(Decimal("1200.00"), "EUR")
        assert table.unpriced == (B,)
        row = next(r for r in table.rows if r.isin == B)
        assert row.has_warning and "excluded from the total" in row.warnings[0]

    def test_stale_price_warns_on_its_own_row(self, instruments, ledger):
        """The warning belongs beside the number, not in a sidebar panel."""
        pos = derive_positions(ledger, instruments,
                               quotes={A: quote("12.00", as_of=D(2025, 9, 10)),
                                       B: quote("4.00")})
        table = holdings_table(pos, instruments, as_of=TODAY)
        stale_row = next(r for r in table.rows if r.isin == A)
        fresh_row = next(r for r in table.rows if r.isin == B)
        assert any("NOT a current price" in w for w in stale_row.warnings)
        assert not fresh_row.warnings

    def test_last_close_fallback_is_labelled(self, instruments, ledger):
        pos = derive_positions(ledger, instruments,
                               quotes={A: quote("12.00", stale=True, delay=None),
                                       B: quote("4.00")})
        table = holdings_table(pos, instruments, as_of=TODAY)
        row = next(r for r in table.rows if r.isin == A)
        assert any("not a live quote" in w for w in row.warnings)

    def test_fx_rate_and_date_are_on_the_row(self, instruments, ledger):
        """A converted value must show the rate and when it was taken."""
        fx = FxTable().add("USD", "EUR", TODAY, "0.90")
        pos = derive_positions(ledger, instruments, rates=fx,
                               quotes={A: quote("12.00", ccy="USD"), B: quote("4.00")})
        table = holdings_table(pos, instruments, rates=fx, as_of=TODAY)
        row = next(r for r in table.rows if r.isin == A)
        assert row.fx_note and "USD->EUR at 0.9000" in row.fx_note
        assert TODAY.isoformat() in row.fx_note

    def test_delay_is_stated_per_row(self, instruments, ledger):
        pos = derive_positions(ledger, instruments,
                               quotes={A: quote("12.00"), B: quote("4.00")})
        table = holdings_table(pos, instruments, as_of=TODAY)
        assert "delayed ~15m" in table.rows[0].price_as_of

    def test_watchlist_entries_are_listed_not_shown_as_holdings(self, instruments, ledger):
        pos = derive_positions(ledger, instruments,
                               quotes={A: quote("12.00"), B: quote("4.00")})
        table = holdings_table(pos, instruments, as_of=TODAY)
        assert table.watchlist == ("LU1681048630",)
        assert "LU1681048630" not in {r.isin for r in table.rows}

    def test_underivable_holding_is_visible_and_excluded_from_the_total(
            self, instruments):
        """It must appear with its reason, not vanish and not be valued at zero."""
        from portfolio.core.positions import derive_positions as dp
        ledger = [
            Transaction(D(2025, 1, 10), A, T.BUY, Decimal("100"), Decimal("10"), "EUR"),
            Transaction(D(2025, 5, 20), B, T.BUY, Decimal("90"),
                        Decimal("742.50"), "GBp"),
        ]
        pos = dp(ledger, instruments, rates=FxTable(), strict=False,
                 quotes={A: quote("12.00")})
        table = holdings_table(pos, instruments, as_of=TODAY)
        row = next(r for r in table.rows if r.isin == B)
        assert "cannot be valued" in row.warnings[0]
        assert "GBP->EUR" in row.warnings[0]
        assert B in table.unpriced
        assert table.total_value == Money(Decimal("1200.00"), "EUR")

    def test_empty_portfolio(self, instruments):
        table = holdings_table(derive_positions([], instruments), instruments)
        assert table.is_empty and table.total_value.amount == 0


class TestDivergence:
    def _decomposition(self):
        cov = pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0001]],
                           index=[A, B], columns=[A, B])
        return risk_decomposition({A: 0.6, B: 0.4}, cov)

    def test_sorted_by_absolute_divergence(self, instruments):
        rows = divergence_rows(self._decomposition(), instruments)
        gaps = [abs(r.divergence) for r in rows]
        assert gaps == sorted(gaps, reverse=True)

    def test_divergence_is_risk_share_minus_weight(self, instruments):
        for row in divergence_rows(self._decomposition(), instruments):
            assert row.divergence == pytest.approx(row.risk_share - row.weight)

    def test_negative_divergence_sorts_by_magnitude_not_sign(self, instruments):
        """A holding that carries far less risk than capital is as interesting
        as one that carries more."""
        cov = pd.DataFrame([[0.0004, 0.0], [0.0, 0.000001]], index=[A, B], columns=[A, B])
        rows = divergence_rows(risk_decomposition({A: 0.5, B: 0.5}, cov), instruments)
        assert rows[0].divergence > 0 and rows[-1].divergence < 0
        assert abs(rows[0].divergence) >= abs(rows[-1].divergence)

    def test_sentence_names_the_fund_and_both_shares(self, instruments):
        sentence = divergence_rows(self._decomposition(), instruments)[0].sentence()
        assert "WisdomTree Europe Defence" in sentence
        assert "of your money" in sentence and "of your risk" in sentence

    def test_falls_back_to_isin_when_the_instrument_is_unknown(self):
        rows = divergence_rows(self._decomposition(), {})
        assert rows[0].name == rows[0].isin


class TestPlainSentences:
    def _metrics(self, **kw):
        cov = pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0001]],
                           index=[A, B], columns=[A, B])
        w = {A: 0.6, B: 0.4}
        return risk_metrics(risk_decomposition(w, cov), diversification_ratio(w, cov),
                            concentration(list(w.values())), **kw)

    def test_every_metric_has_a_sentence(self):
        for metric in self._metrics():
            assert metric.sentence and not metric.sentence.endswith(metric.value)

    def test_volatility_sentence_is_plain(self):
        metric = self._metrics()[0]
        assert "in a typical year" in metric.sentence.lower()
        assert "either way" in metric.sentence

    def test_diversification_sentence_explains_the_baseline(self):
        assert "1.0 would mean they all move together" in self._metrics()[1].sentence

    def test_effective_holdings_sentence(self):
        assert "independent ones" in self._metrics()[2].sentence

    def test_drawdown_sentence_covers_both_figures(self):
        values = pd.Series([100.0, 120.0, 90.0, 110.0],
                           index=pd.bdate_range("2026-01-01", periods=4))
        metric = self._metrics(drawdown_stats=drawdown(values))[3]
        assert "worst peak-to-trough" in metric.sentence
        assert "below the peak" in metric.sentence

    def test_beta_without_a_named_benchmark_warns(self):
        """A beta without a named benchmark is meaningless."""
        metric = self._metrics(beta_value=1.2)[-1]
        assert metric.warning and "meaningless" in metric.warning

    def test_beta_with_a_named_benchmark_does_not_warn(self):
        metric = self._metrics(beta_value=1.2, benchmark_name="MSCI Europe")[-1]
        assert metric.warning is None and "MSCI Europe" in metric.sentence


class TestCorrelationSentences:
    def test_names_both_funds(self, instruments):
        cov = pd.DataFrame([[0.0004, 0.00038], [0.00038, 0.0004]],
                           index=[A, B], columns=[A, B])
        pairs = high_correlation_pairs(correlation_matrix(cov))
        sentences = correlation_sentences(pairs, instruments)
        assert len(sentences) == 1
        assert "WisdomTree Europe Defence" in sentences[0]
        assert "iShares Europe Defence" in sentences[0]

    def test_no_pairs_gives_no_sentences(self, instruments):
        cov = pd.DataFrame([[0.0004, 0.0], [0.0, 0.0001]], index=[A, B], columns=[A, B])
        assert correlation_sentences(
            high_correlation_pairs(correlation_matrix(cov)), instruments) == []
