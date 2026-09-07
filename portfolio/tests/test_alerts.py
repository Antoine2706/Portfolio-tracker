"""Alert rules, their severities, and the order they come out in."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from portfolio.core.alerts import ROUTES, Alert, Severity, build_alerts
from portfolio.core.models import AssetClass, Instrument, Transaction, TransactionType as T
from portfolio.core.money import Money
from portfolio.core.positions import PriceQuote, derive_positions
from portfolio.core.report import holdings_table
from portfolio.core.returns import AlignmentReport, ExcludedInstrument
from portfolio.core.risk import ConcentrationStats, CorrelationCluster, CorrelationPair

D = dt.date
A, B, C, W = "IE0002Y8CX98", "IE000IAXNM41", "LU1681048630", "JE00BN7KB664"
TODAY = D(2026, 9, 3)


@pytest.fixture
def instruments():
    return {
        A: Instrument(A, "WisdomTree Europe Defence", AssetClass.ETF, "EUR"),
        B: Instrument(B, "iShares Europe Defence", AssetClass.ETF, "EUR"),
        C: Instrument(C, "Amundi Global Luxury", AssetClass.ETF, "EUR"),
        W: Instrument(W, "WisdomTree Copper", AssetClass.ETC, "USD"),
    }


@pytest.fixture
def ledger():
    return [
        Transaction(D(2025, 1, 10), A, T.BUY, Decimal("100"), Decimal("10.00"), "EUR"),
        Transaction(D(2025, 1, 10), B, T.BUY, Decimal("200"), Decimal("4.00"), "EUR"),
        Transaction(D(2025, 1, 10), C, T.BUY, Decimal("100"), Decimal("5.00"), "EUR"),
    ]


def quote(price, *, as_of=TODAY, stale=False, delay=15):
    return PriceQuote(Money(Decimal(price), "EUR"), as_of=as_of, source="test",
                      delay_minutes=delay, is_stale=stale)


def table(ledger, instruments, quotes):
    """A 1200, B 800, C 500 when all three are quoted at 12 / 4 / 5."""
    return holdings_table(derive_positions(ledger, instruments, quotes=quotes), instruments,
                          as_of=TODAY)


ALL_QUOTED = {A: quote("12.00"), B: quote("4.00"), C: quote("5.00")}


class TestRules:
    def test_a_healthy_book_raises_nothing(self, ledger, instruments):
        assert build_alerts(table(ledger, instruments, ALL_QUOTED), instruments, max_weight=0.5) == []

    def test_unpriced_holding_is_serious(self, ledger, instruments):
        t = table(ledger, instruments, {A: quote("12.00"), C: quote("5.00")})
        alerts = build_alerts(t, instruments, max_weight=0.8)
        assert [a.code for a in alerts] == ["unpriced_holding"]
        a = alerts[0]
        assert a.severity is Severity.SERIOUS and a.isins == (B,)
        assert "iShares Europe Defence" in a.title and a.route == "#/holdings"
        assert "excluded from the total" in a.detail

    def test_fetch_failures_are_serious_and_carry_the_message(self, ledger, instruments):
        alerts = build_alerts(table(ledger, instruments, ALL_QUOTED), instruments,
                              fetch_failures=["EUDF.DE: timed out"], max_weight=0.8)
        assert [(a.code, a.severity, a.detail) for a in alerts] == [
            ("fetch_failure", Severity.SERIOUS, "EUDF.DE: timed out")]
        assert alerts[0].route == "#/settings"

    def test_outdated_price_is_a_warning(self, ledger, instruments):
        quotes = dict(ALL_QUOTED, A=None)
        quotes = {A: quote("12.00", as_of=D(2025, 9, 10)), B: quote("4.00"), C: quote("5.00")}
        alerts = build_alerts(table(ledger, instruments, quotes), instruments, max_weight=0.8)
        assert [a.code for a in alerts] == ["stale_price"]
        assert alerts[0].severity is Severity.WARNING and alerts[0].isins == (A,)
        assert "NOT a current price" in alerts[0].detail

    def test_last_close_fallback_is_a_warning(self, ledger, instruments):
        quotes = {A: quote("12.00", stale=True, delay=None), B: quote("4.00"), C: quote("5.00")}
        alerts = build_alerts(table(ledger, instruments, quotes), instruments, max_weight=0.8)
        assert [a.code for a in alerts] == ["stale_price"]
        assert "not a live quote" in alerts[0].detail

    def test_unpriced_holding_does_not_also_count_as_stale(self, ledger, instruments):
        t = table(ledger, instruments, {B: quote("4.00"), C: quote("5.00")})
        codes = [a.code for a in build_alerts(t, instruments, max_weight=0.8)]
        assert codes == ["unpriced_holding"]

    def test_holding_above_max_weight(self, ledger, instruments):
        """A is 1200 / 2500 = 48%: above 35%, not above 50%."""
        t = table(ledger, instruments, ALL_QUOTED)
        alerts = build_alerts(t, instruments)
        assert [a.code for a in alerts] == ["concentrated_holding"]
        assert alerts[0].isins == (A,) and "48%" in alerts[0].title
        assert alerts[0].route == "#/risk"
        assert build_alerts(t, instruments, max_weight=0.5) == []

    def test_effective_holdings_below_half(self, ledger, instruments):
        t = table(ledger, instruments, ALL_QUOTED)
        low = ConcentrationStats(herfindahl=0.5, effective_holdings=2.0, actual_holdings=5)
        alerts = build_alerts(t, instruments, concentration=low, max_weight=0.8)
        assert [a.code for a in alerts] == ["low_effective_holdings"]
        assert alerts[0].severity is Severity.WARNING and "5 holdings behave like 2.0" in alerts[0].title

    def test_effective_holdings_at_half_or_above_is_fine(self, ledger, instruments):
        t = table(ledger, instruments, ALL_QUOTED)
        fine = ConcentrationStats(herfindahl=0.4, effective_holdings=2.5, actual_holdings=5)
        assert build_alerts(t, instruments, concentration=fine, max_weight=0.8) == []

    def test_single_holding_is_not_a_concentration_finding(self, ledger, instruments):
        t = table(ledger, instruments, ALL_QUOTED)
        one = ConcentrationStats(herfindahl=1.0, effective_holdings=1.0, actual_holdings=1)
        assert build_alerts(t, instruments, concentration=one, max_weight=0.8) == []

    def test_high_correlation_pair(self, ledger, instruments):
        t = table(ledger, instruments, ALL_QUOTED)
        alerts = build_alerts(t, instruments, pairs=[CorrelationPair(A, B, 0.93)], max_weight=0.8)
        assert [a.code for a in alerts] == ["correlated_pair"]
        a = alerts[0]
        assert a.severity is Severity.WARNING and a.isins == (A, B)
        assert "WisdomTree Europe Defence and iShares Europe Defence" in a.title
        assert "93%" in a.detail

    def test_cluster_of_three_is_serious_and_absorbs_its_pairs(self, ledger, instruments):
        t = table(ledger, instruments, ALL_QUOTED)
        cluster = CorrelationCluster((A, B, C), 0.9, 0.86, 0.75)
        pairs = [CorrelationPair(A, B, 0.93), CorrelationPair(A, C, 0.9),
                 CorrelationPair(B, C, 0.86), CorrelationPair(A, W, 0.88)]
        alerts = build_alerts(t, instruments, pairs=pairs, clusters=[cluster], max_weight=0.8)
        assert [a.code for a in alerts] == ["correlation_cluster", "correlated_pair"]
        c = alerts[0]
        assert c.severity is Severity.SERIOUS and c.isins == (A, B, C)
        assert "3 holdings are one bet" in c.title and "75%" in c.detail
        assert alerts[1].isins == (A, W), "the pair outside the cluster survives"

    def test_cluster_of_two_is_left_to_the_pair_rule(self, ledger, instruments):
        t = table(ledger, instruments, ALL_QUOTED)
        alerts = build_alerts(t, instruments, pairs=[CorrelationPair(A, B, 0.93)],
                              clusters=[CorrelationCluster((A, B), 0.93, 0.93, None)],
                              max_weight=0.8)
        assert [a.code for a in alerts] == ["correlated_pair"]

    def test_excluded_instrument_and_shortened_window_are_info(self, ledger, instruments):
        t = table(ledger, instruments, ALL_QUOTED)
        report = AlignmentReport(
            instruments=(A, B), requested_lookback=252, effective_lookback=200,
            first_date=None, last_date=None, raw_observations={A: 300, B: 201},
            dropped_in_alignment={}, binding_instrument=B, binding_observations=201,
            excluded=(ExcludedInstrument(C, 2, "only 2 observations, below the 60 minimum"),))
        alerts = build_alerts(t, instruments, alignment=report, max_weight=0.8)
        assert [(a.code, a.severity) for a in alerts] == [
            ("excluded_from_risk_model", Severity.INFO), ("window_shortened", Severity.INFO)]
        excluded, shortened = alerts
        assert excluded.isins == (C,) and excluded.route == "#/instruments"
        assert "Amundi Global Luxury" in excluded.title and "below the 60 minimum" in excluded.detail
        assert shortened.isins == (B,) and "200 days" in shortened.title

    def test_full_window_raises_nothing(self, ledger, instruments):
        t = table(ledger, instruments, ALL_QUOTED)
        report = AlignmentReport(instruments=(A, B), requested_lookback=252, effective_lookback=252,
                                 first_date=None, last_date=None, raw_observations={},
                                 dropped_in_alignment={}, binding_instrument=None,
                                 binding_observations=None)
        assert build_alerts(t, instruments, alignment=report, max_weight=0.8) == []

    def test_watchlist_only_universe_is_info(self, instruments):
        t = holdings_table(derive_positions([], instruments), instruments, as_of=TODAY)
        alerts = build_alerts(t, instruments)
        assert [(a.code, a.severity) for a in alerts] == [("watchlist_only", Severity.INFO)]
        assert set(alerts[0].isins) == set(instruments) and "4 instrument(s)" in alerts[0].detail

    def test_nothing_at_all_raises_nothing(self):
        t = holdings_table(derive_positions([], {}), {}, as_of=TODAY)
        assert build_alerts(t, {}) == []

    def test_unknown_instrument_falls_back_to_the_isin(self, ledger):
        t = table(ledger, {}, {A: quote("12.00"), C: quote("5.00")})
        alerts = build_alerts(t, {}, max_weight=0.8)
        assert alerts[0].title == f"{B} has no price"


class TestOrdering:
    def _everything(self, ledger, instruments):
        t = table(ledger, instruments, {A: quote("12.00", as_of=D(2025, 9, 10)), C: quote("5.00")})
        report = AlignmentReport(instruments=(A,), requested_lookback=252, effective_lookback=100,
                                 first_date=None, last_date=None, raw_observations={},
                                 dropped_in_alignment={}, binding_instrument=A,
                                 binding_observations=101)
        return build_alerts(
            t, instruments, fetch_failures=["DFNC.DE: no data"],
            concentration=ConcentrationStats(0.5, 2.0, 5),
            pairs=[CorrelationPair(A, C, 0.9)],
            clusters=[CorrelationCluster((A, B, C), 0.9, 0.86, None)],
            alignment=report)

    def test_severity_descending_then_title(self, ledger, instruments):
        alerts = self._everything(ledger, instruments)
        ranks = [a.severity.rank for a in alerts]
        assert ranks == sorted(ranks, reverse=True)
        for level in Severity:
            titles = [a.title for a in alerts if a.severity is level]
            assert titles == sorted(titles)

    def test_deterministic(self, ledger, instruments):
        assert self._everything(ledger, instruments) == self._everything(ledger, instruments)

    def test_every_route_is_a_known_spa_route(self, ledger, instruments):
        for a in self._everything(ledger, instruments):
            assert a.route in ROUTES.values()

    def test_serious_before_warning_before_info(self, ledger, instruments):
        alerts = self._everything(ledger, instruments)
        assert alerts[0].severity is Severity.SERIOUS
        assert alerts[-1].severity is Severity.INFO
        assert {a.code for a in alerts} == {
            "unpriced_holding", "fetch_failure", "stale_price", "concentrated_holding",
            "low_effective_holdings", "correlation_cluster", "window_shortened"}
        assert not any(a.code == "correlated_pair" for a in alerts), "A-C sits inside the cluster"


class TestSeverity:
    def test_rank_order(self):
        assert Severity.CRITICAL.rank > Severity.SERIOUS.rank > Severity.WARNING.rank > Severity.INFO.rank

    def test_alphabetical_order_is_not_urgency_order(self):
        """The trap the rank exists for: as strings, CRITICAL < INFO."""
        assert Severity.CRITICAL.value < Severity.INFO.value
        assert Severity.CRITICAL.rank > Severity.INFO.rank

    def test_is_a_string_enum_for_json(self):
        assert Severity.WARNING == "WARNING" and Severity("SERIOUS") is Severity.SERIOUS

    def test_alert_defaults(self):
        a = Alert("x", Severity.INFO, "t", "d")
        assert a.isins == () and a.route == ""
