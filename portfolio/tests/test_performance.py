"""Value history and returns, with every expected number worked by hand.

The fixtures use flat prices and round quantities so that a value, a flow or
a return can be checked against the arithmetic in the docstring without
running anything.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from portfolio.core.models import Transaction, TransactionType as T
from portfolio.core.money import FxTable
from portfolio.core.performance import (TRAILING_KEYS, annualised_return, drawdown_series,
                                        holding_contributions, money_weighted_return,
                                        monthly_table, performance_summary,
                                        period_returns, return_index, rolling_beta,
                                        rolling_volatility, time_weighted_return,
                                        trailing_returns, value_history, xirr)
from portfolio.core.positions import derive_positions
from portfolio.core.returns import TRADING_DAYS_PER_YEAR

D = dt.date
A, B = "IE0002Y8CX98", "IE000IAXNM41"
USD_ETC = "JE00BN7KB664"
WATCH = "LU1681048630"

# Wed 8 Jan 2025 to Wed 15 Jan 2025: 8, 9, 10, 13, 14, 15.
CAL = pd.bdate_range("2025-01-08", periods=6)
EUR = {A: "EUR", B: "EUR", USD_ETC: "USD"}


def flat(price: float, index=CAL) -> pd.Series:
    return pd.Series(float(price), index=index)


def buy(isin, on, qty, price, ccy="EUR", fee="0"):
    return Transaction(on, isin, T.BUY, Decimal(qty), Decimal(price), ccy, Decimal(fee))


# --------------------------------------------------------------------------
# value_history
# --------------------------------------------------------------------------

class TestValueHistory:
    def test_value_is_quantity_times_price(self, two_asset_ledger):
        """A: 100 x 10 = 1000, B: 200 x 4 = 800, from the first transaction date."""
        h = value_history(two_asset_ledger, {A: flat(10), B: flat(4)}, EUR)
        assert list(h.value.index) == list(CAL[2:])          # 10, 13, 14, 15 Jan
        assert h.value.tolist() == [1800.0] * 4
        assert h.per_holding.loc[CAL[2], A] == 1000.0
        assert h.per_holding.loc[CAL[2], B] == 800.0
        assert h.missing == ()

    def test_buy_flow_is_gross_plus_fees(self, two_asset_ledger):
        """1000 + 5 and 800 + 3 on the purchase date, nothing after."""
        h = value_history(two_asset_ledger, {A: flat(10), B: flat(4)}, EUR)
        assert h.flows_per_holding.loc[CAL[2], A] == 1005.0
        assert h.flows_per_holding.loc[CAL[2], B] == 803.0
        assert h.flows.tolist() == [1808.0, 0.0, 0.0, 0.0]
        assert h.invested.tolist() == [1808.0] * 4

    def test_cost_basis_agrees_with_derive_positions_on_the_final_date(self, two_asset_ledger):
        """The contract's own check: two replays of one ledger, one answer."""
        ledger = two_asset_ledger + [
            Transaction(D(2025, 1, 14), A, T.SELL, Decimal("50"), Decimal("11"), "EUR",
                        Decimal("1")),
        ]
        h = value_history(ledger, {A: flat(10), B: flat(4)}, EUR)
        positions = derive_positions(ledger)
        expected = sum(p.cost_basis.amount for p in positions.values() if p.is_open)
        assert h.cost_basis.iloc[-1] == float(expected) == 1305.5   # 1005 - 50 x 10.05 + 803

    def test_cost_basis_steps_on_the_transaction_date(self, two_asset_ledger):
        ledger = two_asset_ledger + [
            Transaction(D(2025, 1, 14), A, T.SELL, Decimal("50"), Decimal("11"), "EUR",
                        Decimal("1")),
        ]
        h = value_history(ledger, {A: flat(10), B: flat(4)}, EUR)
        assert h.cost_basis.tolist() == [1808.0, 1808.0, 1305.5, 1305.5]

    def test_sell_flow_is_proceeds_net_of_fee(self, two_asset_ledger):
        """50 x 11 = 550 gross, 1 fee: 549 reaches you, so the flow is -549."""
        ledger = two_asset_ledger + [
            Transaction(D(2025, 1, 14), A, T.SELL, Decimal("50"), Decimal("11"), "EUR",
                        Decimal("1")),
        ]
        h = value_history(ledger, {A: flat(10), B: flat(4)}, EUR)
        assert h.flows_per_holding.loc[CAL[4], A] == -549.0
        assert h.quantities[A].tolist() == [100.0, 100.0, 50.0, 50.0]
        assert h.invested.iloc[-1] == 1808.0 - 549.0

    def test_dividend_in_both_ledger_forms_is_money_out(self, two_asset_ledger):
        """Total-amount form: price 7, quantity 0 -> -7.
        Per-unit form: 100 x 0.05 = 5, fee 1 -> net 4 -> -4."""
        ledger = two_asset_ledger + [
            Transaction(D(2025, 1, 13), A, T.DIVIDEND, Decimal("0"), Decimal("7")),
            Transaction(D(2025, 1, 14), A, T.DIVIDEND, Decimal("100"), Decimal("0.05"), "EUR",
                        Decimal("1")),
        ]
        h = value_history(ledger, {A: flat(10), B: flat(4)}, EUR)
        assert h.flows_per_holding.loc[CAL[3], A] == -7.0
        assert h.flows_per_holding.loc[CAL[4], A] == -4.0
        assert h.value.tolist() == [1800.0] * 4, "a dividend does not change the holdings"

    def test_fee_is_money_in_that_buys_nothing(self, two_asset_ledger):
        """12 paid from outside: flow +12, value unchanged, so the day's
        return is 1800 / (1800 + 12) - 1."""
        ledger = two_asset_ledger + [
            Transaction(D(2025, 1, 14), A, T.FEE, Decimal("0"), Decimal("0"), "EUR",
                        Decimal("12"), "custody"),
        ]
        h = value_history(ledger, {A: flat(10), B: flat(4)}, EUR)
        assert h.flows.loc[CAL[4]] == 12.0
        assert h.value.loc[CAL[4]] == 1800.0
        r = time_weighted_return(h.value, h.flows)
        assert r.loc[CAL[4]] == pytest.approx(1800 / 1812 - 1)

    def test_usd_holding_converts_at_each_dates_rate(self, fx):
        """10 units x 10 USD: 90 EUR at 0.90 in January, 80 EUR at 0.80 in June.
        The flow converts once, at the purchase date: 90."""
        idx = pd.DatetimeIndex(["2025-01-10", "2025-06-02"])
        ledger = [buy(USD_ETC, D(2025, 1, 10), "10", "10", "USD")]
        h = value_history(ledger, {USD_ETC: flat(10, idx)}, EUR, rates=fx)
        assert h.value.tolist() == [90.0, 80.0]
        assert h.flows.tolist() == [90.0, 0.0]
        assert h.cost_basis.tolist() == [90.0, 90.0]

    def test_missing_fx_rate_excludes_and_names_the_holding(self, two_asset_ledger):
        ledger = two_asset_ledger + [buy(USD_ETC, D(2025, 1, 10), "10", "10", "USD")]
        histories = {A: flat(10), B: flat(4), USD_ETC: flat(10)}
        h = value_history(ledger, histories, EUR, rates=FxTable())
        assert h.missing == (USD_ETC,)
        assert any(USD_ETC in w and "USD->EUR" in w for w in h.warnings)
        assert list(h.per_holding.columns) == [A, B]
        assert h.value.tolist() == [1800.0] * 4

    def test_no_rates_with_a_foreign_currency_never_assumes_parity(self):
        ledger = [buy(USD_ETC, D(2025, 1, 10), "10", "10", "USD")]
        h = value_history(ledger, {USD_ETC: flat(10)}, EUR, rates=None)
        assert h.missing == (USD_ETC,)
        assert h.is_empty

    def test_missing_price_history_is_named_not_valued_at_zero(self, two_asset_ledger):
        """B has no series: it is out of the value AND out of the flows, and named."""
        h = value_history(two_asset_ledger, {A: flat(10)}, EUR)
        assert h.missing == (B,)
        assert any(B in w and "no price history" in w for w in h.warnings)
        assert h.value.tolist() == [1000.0] * 4
        assert h.invested.tolist() == [1005.0] * 4

    def test_price_gap_is_forward_filled_and_counted(self, two_asset_ledger):
        gappy = flat(10, CAL.drop(CAL[4]))                  # no 14 Jan price for A
        h = value_history(two_asset_ledger, {A: gappy, B: flat(4)}, EUR)
        assert h.per_holding.loc[CAL[4], A] == 1000.0
        assert any(A in w and "1 price gap" in w for w in h.warnings)

    def test_calendar_is_the_union_of_price_calendars(self, two_asset_ledger):
        """B trades on 16 Jan when A does not: the day exists and A carries."""
        extra = CAL.append(pd.DatetimeIndex(["2025-01-16"]))
        h = value_history(two_asset_ledger, {A: flat(10), B: flat(4, extra)}, EUR)
        assert h.value.index[-1] == pd.Timestamp("2025-01-16")
        assert h.per_holding.loc["2025-01-16", A] == 1000.0
        assert any(A in w and "last price 2025-01-15" in w for w in h.warnings)

    def test_weekend_transaction_lands_on_the_next_trading_day(self):
        """Bought on Saturday 11 Jan: nothing on Friday, everything on Monday."""
        ledger = [buy(A, D(2025, 1, 10), "1", "10"), buy(B, D(2025, 1, 11), "200", "4")]
        h = value_history(ledger, {A: flat(10), B: flat(4)}, EUR)
        assert h.quantities.loc[CAL[2], B] == 0.0
        assert h.quantities.loc[CAL[3], B] == 200.0
        assert h.flows_per_holding.loc[CAL[2], B] == 0.0
        assert h.flows_per_holding.loc[CAL[3], B] == 800.0

    def test_start_trims_the_calendar_and_carries_earlier_flows(self, two_asset_ledger):
        h = value_history(two_asset_ledger, {A: flat(10), B: flat(4)}, EUR,
                          start=D(2025, 1, 14))
        assert list(h.value.index) == list(CAL[4:])
        assert h.flows.tolist() == [1808.0, 0.0], "invested must stay complete"
        assert h.invested.tolist() == [1808.0, 1808.0]
        assert h.cost_basis.tolist() == [1808.0, 1808.0]

    def test_start_before_the_first_transaction_shows_zeros_not_a_trim(self, two_asset_ledger):
        h = value_history(two_asset_ledger, {A: flat(10), B: flat(4)}, EUR,
                          start=D(2025, 1, 8))
        assert h.value.tolist() == [0.0, 0.0, 1800.0, 1800.0, 1800.0, 1800.0]
        assert h.flows.loc[CAL[2]] == 1808.0

    def test_history_starts_when_every_open_holding_has_a_price(self, two_asset_ledger):
        """A's series begins on 13 Jan, after its purchase on the 10th. A zero on
        the 10th would read as a loss, so the history opens on the 13th."""
        late = flat(10, CAL[3:])
        h = value_history(two_asset_ledger, {A: late, B: flat(4)}, EUR)
        assert h.value.index[0] == CAL[3]
        assert h.value.iloc[0] == 1800.0
        assert h.flows.iloc[0] == 1808.0
        assert any("history starts 2025-01-13" in w and A in w for w in h.warnings)

    def test_later_purchase_before_its_first_price_is_carried_to_that_date(self):
        """A is held throughout; B is bought on the 13th but only priced from
        the 15th. B's flow lands on the 15th and it is valued at zero before."""
        ledger = [buy(A, D(2025, 1, 10), "100", "10"), buy(B, D(2025, 1, 13), "200", "4")]
        h = value_history(ledger, {A: flat(10), B: flat(4, CAL[5:])}, EUR)
        assert h.flows_per_holding[B].tolist() == [0.0, 0.0, 0.0, 800.0]
        assert h.per_holding[B].tolist() == [0.0, 0.0, 0.0, 800.0]
        assert h.quantities[B].tolist() == [0.0, 200.0, 200.0, 200.0]
        assert any(B in w and "before its first price on 2025-01-15" in w for w in h.warnings)

    def test_transaction_after_the_last_price_is_warned(self, two_asset_ledger):
        ledger = two_asset_ledger + [buy(A, D(2025, 2, 1), "1", "10")]
        h = value_history(ledger, {A: flat(10), B: flat(4)}, EUR)
        assert any("after the last price" in w and A in w for w in h.warnings)
        assert h.quantities[A].iloc[-1] == 100.0

    def test_quote_currency_is_inferred_from_transactions_with_a_warning(self, two_asset_ledger):
        h = value_history(two_asset_ledger, {A: flat(10), B: flat(4)}, {})
        assert h.value.tolist() == [1800.0] * 4
        assert any("no quote currency" in w and "using EUR" in w for w in h.warnings)

    def test_pence_quoted_series_is_rescaled_to_pounds(self, fx):
        """1000 GBp = 10 GBP = 12 EUR per unit; 10 units bought at 1000 GBp."""
        ledger = [buy(B, D(2025, 1, 10), "10", "1000", "GBp")]
        h = value_history(ledger, {B: flat(1000)}, {B: "GBp"}, rates=fx)
        assert h.value.tolist() == [120.0] * 4
        assert h.flows.iloc[0] == 120.0
        assert h.cost_basis.iloc[0] == 120.0

    def test_extra_price_series_are_ignored(self, two_asset_ledger):
        h = value_history(two_asset_ledger, {A: flat(10), B: flat(4), WATCH: flat(1)}, EUR)
        assert list(h.per_holding.columns) == [A, B]

    def test_frames_share_index_and_columns(self, two_asset_ledger):
        h = value_history(two_asset_ledger, {A: flat(10), B: flat(4)}, EUR)
        for frame in (h.per_holding, h.quantities, h.flows_per_holding):
            assert list(frame.columns) == [A, B]
            assert frame.index.equals(h.value.index)
        for series in (h.cost_basis, h.invested, h.flows):
            assert series.index.equals(h.value.index)

    def test_empty_ledger_is_the_empty_state_not_a_warning(self):
        h = value_history([], {A: flat(10)}, EUR)
        assert h.is_empty and h.warnings == () and h.missing == ()

    def test_no_price_on_or_after_the_start_is_named(self, two_asset_ledger):
        h = value_history(two_asset_ledger, {A: flat(10), B: flat(4)}, EUR,
                          start=D(2025, 3, 1))
        assert h.is_empty
        assert any("no price dates on or after 2025-03-01" in w for w in h.warnings)


# --------------------------------------------------------------------------
# Time-weighted return and the index
# --------------------------------------------------------------------------

class TestTimeWeightedReturn:
    def _series(self, values, flows):
        idx = pd.bdate_range("2026-01-01", periods=len(values))
        return pd.Series(values, index=idx, dtype=float), pd.Series(flows, index=idx, dtype=float)

    def test_by_hand(self):
        """Day 2: 100 held, 100 added, worth 220 -> 220/200 - 1 = 10%.
        Day 3: no flow, 220 -> 209 = -5%."""
        v, f = self._series([100, 220, 209], [100, 100, 0])
        assert time_weighted_return(v, f).round(10).tolist() == [0.0, 0.1, -0.05]

    @pytest.mark.parametrize("flow", [0.0, 10.0, 1000.0, 1e6])
    def test_invariant_to_flow_size_on_a_flat_price(self, flow):
        """Adding money to holdings whose price did not move is not a return."""
        v, f = self._series([100, 100 + flow, 100 + flow], [100, flow, 0])
        assert time_weighted_return(v, f).tolist() == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("flow", [10.0, 50.0, 99.0])
    def test_invariant_to_withdrawal_size_on_a_flat_price(self, flow):
        v, f = self._series([100, 100 - flow], [100, -flow])
        assert time_weighted_return(v, f).tolist() == [0.0, 0.0]

    def test_full_exit_is_valued_at_the_proceeds(self):
        """Holdings worth 100, sold for 99: a 1% loss, not the -100% that a
        start-of-day outflow would produce (0 / (100 - 99) - 1)."""
        v, f = self._series([100, 0], [100, -99])
        assert time_weighted_return(v, f).iloc[-1] == pytest.approx(-0.01)

    def test_full_exit_above_the_previous_close_is_a_gain(self):
        v, f = self._series([100, 0], [100, -101])
        assert time_weighted_return(v, f).iloc[-1] == pytest.approx(0.01)

    def test_partial_sell_at_the_close_is_exact(self):
        """10 units at 10; 5 sold at the day's close of 9.9. The 49.5 that left
        earned the day's -1% first, so the return is (49.5 + 49.5) / 100 - 1."""
        v, f = self._series([100, 49.5], [100, -49.5])
        assert time_weighted_return(v, f).iloc[-1] == pytest.approx(-0.01)

    def test_zero_denominator_gives_zero(self):
        """Nothing held and nothing added: no return, not a division error."""
        v, f = self._series([0, 0, 100, 110], [0, 0, 100, 0])
        assert time_weighted_return(v, f).round(10).tolist() == [0.0, 0.0, 0.0, 0.1]

    def test_first_day_is_zero_whatever_happened(self):
        v, f = self._series([150, 150], [100, 0])
        assert time_weighted_return(v, f).iloc[0] == 0.0

    def test_equals_price_return_when_there_are_no_flows(self):
        prices = [100.0, 110.0, 99.0, 105.0]
        v, f = self._series(prices, [100, 0, 0, 0])
        got = time_weighted_return(v, f).iloc[1:]
        assert np.allclose(got, pd.Series(prices).pct_change().iloc[1:])

    def test_missing_flow_dates_count_as_zero(self):
        v = pd.Series([100.0, 110.0], index=pd.bdate_range("2026-01-01", periods=2))
        f = pd.Series([100.0], index=v.index[:1])
        assert time_weighted_return(v, f).round(10).tolist() == [0.0, 0.1]


class TestReturnIndex:
    def test_constant_return_compounds_exactly(self):
        r = pd.Series([0.01] * 10)
        expected = [100 * 1.01 ** k for k in range(1, 11)]
        assert np.allclose(return_index(r), expected)

    def test_starts_at_start_when_the_first_return_is_zero(self):
        r = pd.Series([0.0, 0.5])
        assert return_index(r, start=1.0).tolist() == [1.0, 1.5]

    def test_drawdown_series_by_hand(self):
        """100 -> 120 -> 90 -> 110: 0, 0, -25%, -8.33%."""
        idx = pd.Series([100.0, 120.0, 90.0, 110.0])
        assert drawdown_series(idx).round(6).tolist() == [0.0, 0.0, -0.25, -0.083333]

    def test_drawdown_is_never_positive(self):
        rng = np.random.default_rng(0)
        idx = return_index(pd.Series(rng.normal(0, 0.02, 500)))
        assert (drawdown_series(idx) <= 0).all()


class TestAnnualisedReturn:
    def test_252_days_annualise_to_the_total_return(self):
        r = pd.Series([0.001] * TRADING_DAYS_PER_YEAR)
        assert annualised_return(r) == pytest.approx(1.001 ** 252 - 1, rel=1e-12)

    def test_half_a_year_squares_the_growth(self):
        r = pd.Series([0.001] * 126)
        total = 1.001 ** 126
        assert annualised_return(r) == pytest.approx(total ** 2 - 1, rel=1e-12)

    def test_empty_is_none(self):
        assert annualised_return(pd.Series(dtype=float)) is None

    def test_total_loss_is_none(self):
        """No rate compounds 1 to 0."""
        assert annualised_return(pd.Series([0.1, -1.0])) is None


# --------------------------------------------------------------------------
# Money-weighted return
# --------------------------------------------------------------------------

class TestXirr:
    def test_one_year_ten_percent(self):
        got = xirr([(D(2025, 1, 1), -1000.0), (D(2026, 1, 1), 1100.0)])
        assert got == pytest.approx(0.10, abs=1e-6)

    def test_two_years_compounds(self):
        """1000 x 1.1^2 = 1210 after 730 days = exactly 2 years at actual/365."""
        got = xirr([(D(2025, 1, 1), -1000.0), (D(2027, 1, 1), 1210.0)])
        assert got == pytest.approx(0.10, abs=1e-6)

    def test_three_flows(self):
        """NPV at 10%: -1000 - 1000/1.1 + 2310/1.21 = -1000 - 909.09 + 1909.09 = 0."""
        got = xirr([(D(2025, 1, 1), -1000.0), (D(2026, 1, 1), -1000.0),
                    (D(2027, 1, 1), 2310.0)])
        assert got == pytest.approx(0.10, abs=1e-6)

    def test_negative_rate(self):
        got = xirr([(D(2025, 1, 1), -1000.0), (D(2026, 1, 1), 900.0)])
        assert got == pytest.approx(-0.10, abs=1e-6)

    def test_bisection_rescues_newton(self):
        """-1000 then 10 a year later: Newton's first step from 0.1 lands far
        below -100%, so bisection has to find -99%."""
        got = xirr([(D(2025, 1, 1), -1000.0), (D(2026, 1, 1), 10.0)])
        assert got == pytest.approx(-0.99, abs=1e-6)

    def test_very_high_rate(self):
        got = xirr([(D(2025, 1, 1), -1.0), (D(2026, 1, 1), 1000.0)])
        assert got == pytest.approx(999.0, rel=1e-6)

    def test_fewer_than_two_flows_is_none(self):
        assert xirr([(D(2025, 1, 1), -1000.0)]) is None
        assert xirr([]) is None

    def test_zero_flows_do_not_count(self):
        assert xirr([(D(2025, 1, 1), -1000.0), (D(2026, 1, 1), 0.0)]) is None

    def test_all_same_sign_is_none(self):
        assert xirr([(D(2025, 1, 1), -1000.0), (D(2026, 1, 1), -100.0)]) is None
        assert xirr([(D(2025, 1, 1), 1000.0), (D(2026, 1, 1), 100.0)]) is None

    def test_order_of_flows_does_not_matter(self):
        flows = [(D(2026, 1, 1), 1100.0), (D(2025, 1, 1), -1000.0)]
        assert xirr(flows) == pytest.approx(0.10, abs=1e-6)


class TestMoneyWeightedReturn:
    def test_flips_the_sign_convention(self):
        """1000 in (portfolio sign +) worth 1100 a year later: +10%."""
        flows = pd.Series([1000.0], index=pd.DatetimeIndex(["2025-01-01"]))
        assert money_weighted_return(flows, 1100.0, D(2026, 1, 1)) == pytest.approx(0.10, abs=1e-6)

    def test_two_deposits(self):
        flows = pd.Series([1000.0, 1000.0], index=pd.DatetimeIndex(["2025-01-01", "2026-01-01"]))
        assert money_weighted_return(flows, 2310.0, D(2027, 1, 1)) == pytest.approx(0.10, abs=1e-6)

    def test_withdrawal_is_a_positive_investor_flow(self):
        """1000 in, 1100 out a year later, nothing left: still 10%."""
        flows = pd.Series([1000.0, -1100.0], index=pd.DatetimeIndex(["2025-01-01", "2026-01-01"]))
        assert money_weighted_return(flows, 0.0, D(2026, 1, 1)) == pytest.approx(0.10, abs=1e-6)

    def test_no_flows_is_none(self):
        flows = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        assert money_weighted_return(flows, 100.0, D(2026, 1, 1)) is None


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------

class TestPeriodReturns:
    R = pd.Series([0.1, -0.1, 0.05, 0.02],
                  index=pd.to_datetime(["2025-01-05", "2025-01-06", "2025-03-03", "2026-02-02"]))

    def test_monthly_compounds_within_the_month(self):
        """January: 1.1 x 0.9 - 1 = -1%; February absent; March 5%."""
        m = period_returns(self.R, "M")
        assert m.round(10).tolist() == [-0.01, 0.05, 0.02]
        assert [d.strftime("%Y-%m-%d") for d in m.index] == ["2025-01-31", "2025-03-31",
                                                             "2026-02-28"]

    def test_yearly(self):
        """2025: 1.1 x 0.9 x 1.05 - 1 = 3.95%; 2026: 2%."""
        y = period_returns(self.R, "Y")
        assert y.round(10).tolist() == [0.0395, 0.02]
        assert [d.year for d in y.index] == [2025, 2026]

    def test_month_end_aliases_are_accepted(self):
        assert period_returns(self.R, "ME").equals(period_returns(self.R, "M"))

    def test_unknown_frequency_is_refused(self):
        with pytest.raises(ValueError, match="'M' or 'Y'"):
            period_returns(self.R, "W")

    def test_empty(self):
        assert period_returns(pd.Series(dtype=float), "M").empty

    def test_monthly_table_shape_and_gaps(self):
        table = monthly_table(self.R)
        assert list(table.columns) == list(range(1, 13))
        assert list(table.index) == [2025, 2026]
        assert table.loc[2025, 1] == pytest.approx(-0.01)
        assert math.isnan(table.loc[2025, 2])
        assert table.loc[2025, 3] == pytest.approx(0.05)
        assert table.loc[2026, 2] == pytest.approx(0.02)
        assert table.dtypes.eq(float).all()

    def test_monthly_table_empty(self):
        table = monthly_table(pd.Series(dtype=float))
        assert table.empty and list(table.columns) == list(range(1, 13))


# --------------------------------------------------------------------------
# Rolling statistics
# --------------------------------------------------------------------------

class TestRolling:
    def test_rolling_volatility_by_hand(self):
        """Window of [0.01, 0.02, 0.03] has std (ddof=1) exactly 0.01."""
        r = pd.Series([0.01, 0.02, 0.03], index=pd.bdate_range("2026-01-01", periods=3))
        got = rolling_volatility(r, window=3)
        assert len(got) == 1
        assert got.iloc[0] == pytest.approx(0.01 * math.sqrt(252))

    def test_rolling_volatility_of_a_constant_is_zero(self):
        r = pd.Series([0.01] * 20, index=pd.bdate_range("2026-01-01", periods=20))
        got = rolling_volatility(r, window=5)
        assert len(got) == 16 and (got == 0).all()

    def test_rolling_beta_of_a_doubled_benchmark_is_two(self):
        rng = np.random.default_rng(1)
        b = pd.Series(rng.normal(0, 0.01, 100), index=pd.bdate_range("2026-01-01", periods=100))
        got = rolling_beta(b * 2, b, window=20)
        assert len(got) == 81
        assert np.allclose(got, 2.0)

    def test_rolling_beta_uses_shared_dates_only(self):
        rng = np.random.default_rng(2)
        idx = pd.bdate_range("2026-01-01", periods=100)
        b = pd.Series(rng.normal(0, 0.01, 100), index=idx)
        got = rolling_beta((b * 1.5).iloc[40:], b, window=10)
        assert len(got) == 60 - 9
        assert np.allclose(got, 1.5)


# --------------------------------------------------------------------------
# Trailing returns
# --------------------------------------------------------------------------

def linear_index() -> pd.Series:
    """Value = 100 + calendar days since 1 Jan 2025, on business days.

    A value that is a function of the date makes every anchor checkable:
    the base for an anchor is 100 + the day count of the last business day
    on or before it.
    """
    idx = pd.bdate_range("2025-01-01", "2026-03-31")
    return pd.Series(100.0 + (idx - pd.Timestamp("2025-01-01")).days, index=idx)


class TestTrailingReturns:
    def test_anchors_by_hand(self):
        """as_of Tue 31 Mar 2026 = 554.
        1w   Tue 24 Mar 2026 = 547       1m  Sat 28 Feb -> Fri 27 Feb = 522
        3m   Wed 31 Dec 2025 = 464       6m  Tue 30 Sep 2025 = 372
        ytd  Thu  1 Jan 2026 = 465       1y  Mon 31 Mar 2025 = 189
        all  Wed  1 Jan 2025 = 100"""
        got = trailing_returns(linear_index())
        assert got["1w"] == pytest.approx(554 / 547 - 1)
        assert got["1m"] == pytest.approx(554 / 522 - 1)
        assert got["3m"] == pytest.approx(554 / 464 - 1)
        assert got["6m"] == pytest.approx(554 / 372 - 1)
        assert got["ytd"] == pytest.approx(554 / 465 - 1)
        assert got["1y"] == pytest.approx(554 / 189 - 1)
        assert got["all"] == pytest.approx(554 / 100 - 1)

    def test_keys_are_exactly_the_contract(self):
        assert tuple(trailing_returns(linear_index())) == TRAILING_KEYS

    def test_as_of_uses_the_last_value_on_or_before(self):
        """Sunday 1 Feb 2026 -> Friday 30 Jan 2026 = 100 + 394 = 494; 1w anchor
        Sunday 25 Jan -> Friday 23 Jan = 487."""
        got = trailing_returns(linear_index(), as_of=D(2026, 2, 1))
        assert got["1w"] == pytest.approx(494 / 487 - 1)

    def test_none_when_the_index_does_not_reach_back(self):
        short = linear_index().iloc[-10:]           # ten business days
        got = trailing_returns(short)
        assert got["1w"] is not None
        assert got["1m"] is None and got["1y"] is None and got["ytd"] is None
        assert got["all"] is not None

    def test_ytd_needs_a_value_from_the_previous_year(self):
        this_year = linear_index().loc["2026-01-02":]
        assert trailing_returns(this_year)["ytd"] is None

    def test_as_of_before_the_index_is_all_none(self):
        got = trailing_returns(linear_index(), as_of=D(2024, 1, 1))
        assert all(v is None for v in got.values())

    def test_empty_index_is_all_none(self):
        got = trailing_returns(pd.Series(dtype=float, index=pd.DatetimeIndex([])))
        assert all(v is None for v in got.values())


# --------------------------------------------------------------------------
# Contributions
# --------------------------------------------------------------------------

class TestHoldingContributions:
    def _history(self):
        """A: 100 units bought at 10 on the 10th, price 10 -> 12 by the 15th.
        B: 200 at 4 on the 10th, flat; 100 more at 4 + 3 fee on the 14th."""
        ledger = [
            buy(A, D(2025, 1, 10), "100", "10"),
            buy(B, D(2025, 1, 10), "200", "4"),
            buy(B, D(2025, 1, 14), "100", "4", fee="3"),
        ]
        a = pd.Series([10.0, 10.0, 10.0, 11.0, 12.0, 12.0], index=CAL)
        return value_history(ledger, {A: a, B: flat(4)}, EUR)

    def test_by_hand(self):
        """A: 1200 - 1000 = +200. B: 1200 - 800 - 403 = -3 (the fee).
        Start value 1800: contributions 200/1800 and -3/1800."""
        rows = holding_contributions(self._history())
        assert [(r.isin, r.pnl) for r in rows] == [(A, 200.0), (B, -3.0)]
        assert rows[0].contribution == pytest.approx(200 / 1800)
        assert rows[1].contribution == pytest.approx(-3 / 1800)

    def test_sum_is_the_portfolio_gain_net_of_flows(self):
        h = self._history()
        rows = holding_contributions(h)
        total_gain = h.value.iloc[-1] - h.value.iloc[0] - h.flows.iloc[1:].sum()
        assert sum(r.pnl for r in rows) == pytest.approx(total_gain)

    def test_window(self):
        """From the 13th (A priced 11) to the 14th (A priced 12):
        A 1100 -> 1200 = +100; B 800 -> 1200 - 403 = -3.
        The start value is 1100 + 800 = 1900, so A's contribution is 100/1900."""
        rows = holding_contributions(self._history(), start=D(2025, 1, 13), end=D(2025, 1, 14))
        assert [(r.isin, r.pnl) for r in rows] == [(A, 100.0), (B, -3.0)]
        assert rows[0].contribution == pytest.approx(100 / 1900)

    def test_flows_on_the_start_date_are_inside_the_start_value(self):
        """Starting on the 14th, B's purchase that day is not a flow to subtract."""
        rows = holding_contributions(self._history(), start=D(2025, 1, 14))
        b = next(r for r in rows if r.isin == B)
        assert b.pnl == 0.0

    def test_window_before_anything_was_held_divides_by_the_inflows(self):
        rows = holding_contributions(self._history(), start=D(2025, 1, 8))
        a = next(r for r in rows if r.isin == A)
        # A: 1200 - 0 - 1000 = 200; inflows over the window 1000 + 800 + 403 = 2203
        assert a.pnl == 200.0
        assert a.contribution == pytest.approx(200 / 2203)

    def test_empty_history(self):
        assert holding_contributions(value_history([], {}, {})) == []


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def daily(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2026-01-01", periods=len(values)), dtype=float)


class TestPerformanceSummary:
    def test_sharpe_by_hand(self):
        """[0.02, 0.00, 0.01]: mean 0.01, std (ddof=1) 0.01.
        Sharpe = 0.01 x 252 / (0.01 x sqrt 252) = sqrt 252."""
        s = performance_summary(daily([0.02, 0.0, 0.01]))
        assert s.volatility == pytest.approx(0.01 * math.sqrt(252))
        assert s.sharpe == pytest.approx(math.sqrt(252))
        assert s.total_return == pytest.approx(1.02 * 1.01 - 1)
        assert s.observations == 3

    def test_sortino_by_hand(self):
        """[0.02, -0.01, 0.02, -0.01]: mean 0.005 -> 1.26 a year.
        Downside deviation = sqrt((0 + 0.0001 + 0 + 0.0001) / 4) x sqrt 252."""
        s = performance_summary(daily([0.02, -0.01, 0.02, -0.01]))
        downside = math.sqrt(0.00005) * math.sqrt(252)
        assert s.sortino == pytest.approx(1.26 / downside)

    def test_risk_free_rate_reduces_both_ratios(self):
        r = daily([0.02, 0.0, 0.01])
        assert performance_summary(r, risk_free=0.02).sharpe == pytest.approx(
            (0.01 * 252 - 0.02) / (0.01 * math.sqrt(252)))

    def test_constant_returns_have_no_sharpe(self):
        s = performance_summary(daily([0.01] * 10))
        assert s.volatility == 0.0 and s.sharpe is None
        assert s.sortino is None, "no downside, no ratio"

    def test_drawdown_and_calmar(self):
        """100 -> 120 -> 90 -> 110: max drawdown -25%, current -8.33%."""
        s = performance_summary(daily([0.2, -0.25, 110 / 90 - 1]))
        assert s.max_drawdown == pytest.approx(-0.25)
        assert s.current_drawdown == pytest.approx(-1 / 12)
        assert s.calmar == pytest.approx(s.annualised_return / 0.25)

    def test_no_drawdown_means_no_calmar(self):
        assert performance_summary(daily([0.01, 0.01, 0.02])).calmar is None

    def test_best_and_worst_day(self):
        s = performance_summary(daily([0.02, -0.03, 0.01]))
        assert s.best_day == 0.02 and s.worst_day == -0.03

    def test_months(self):
        """Jan +1%, Feb -2%, Mar 0%: one positive, one negative, one neither."""
        r = pd.Series([0.01, -0.02, 0.0],
                      index=pd.to_datetime(["2026-01-15", "2026-02-16", "2026-03-16"]))
        s = performance_summary(r)
        assert s.best_month == pytest.approx(0.01) and s.worst_month == pytest.approx(-0.02)
        assert s.positive_months == 1 and s.negative_months == 1

    def test_dates(self):
        s = performance_summary(daily([0.01, 0.02]))
        assert s.first_date == D(2026, 1, 1) and s.last_date == D(2026, 1, 2)

    def test_single_observation(self):
        s = performance_summary(daily([0.05]))
        assert s.total_return == pytest.approx(0.05) and s.observations == 1
        assert s.annualised_return is None and s.volatility is None and s.sharpe is None

    def test_empty(self):
        s = performance_summary(pd.Series(dtype=float))
        assert s.observations == 0 and s.total_return is None
        assert s.first_date is None and s.positive_months == 0

    def test_without_a_benchmark_every_benchmark_field_is_none(self):
        s = performance_summary(daily([0.01, 0.02, -0.01]))
        assert s.beta is s.alpha is s.correlation is s.tracking_error is None
        assert s.information_ratio is None and s.benchmark_total_return is None

    def test_against_a_doubled_benchmark(self):
        """p = 2b: beta 2, correlation 1, alpha 0 (the intercept mean(p) - 2 mean(b)),
        tracking error std(b) x sqrt 252, information ratio mean(b) x 252 / TE."""
        rng = np.random.default_rng(4)
        b = daily(rng.normal(0.001, 0.01, 120))
        s = performance_summary(b * 2, benchmark=b)
        assert s.beta == pytest.approx(2.0)
        assert s.correlation == pytest.approx(1.0)
        assert s.alpha == pytest.approx(0.0, abs=1e-12)
        te = float(b.std(ddof=1)) * math.sqrt(252)
        assert s.tracking_error == pytest.approx(te)
        assert s.information_ratio == pytest.approx(float(b.mean()) * 252 / te)
        assert s.benchmark_total_return == pytest.approx(float(np.prod(1 + b)) - 1)

    def test_alpha_by_hand(self):
        """p = b + 0.001 every day: beta 1, daily intercept 0.001, alpha 0.252."""
        rng = np.random.default_rng(5)
        b = daily(rng.normal(0, 0.01, 60))
        s = performance_summary(b + 0.001, benchmark=b)
        assert s.beta == pytest.approx(1.0)
        assert s.alpha == pytest.approx(0.001 * 252)
        assert s.tracking_error == pytest.approx(0.0, abs=1e-12)
        assert s.information_ratio is None, "zero tracking error, no ratio"

    def test_benchmark_on_shared_dates_only(self):
        rng = np.random.default_rng(6)
        b = daily(rng.normal(0, 0.01, 100))
        s = performance_summary((b * 1.5).iloc[30:], benchmark=b)
        assert s.beta == pytest.approx(1.5)
        assert s.benchmark_total_return == pytest.approx(float(np.prod(1 + b.iloc[30:])) - 1)

    def test_zero_variance_benchmark_gives_no_beta(self):
        s = performance_summary(daily([0.01, 0.02, 0.03]), benchmark=daily([0.0, 0.0, 0.0]))
        assert s.beta is None and s.alpha is None and s.correlation is None

    def test_too_little_overlap_gives_no_benchmark_figures(self):
        s = performance_summary(daily([0.01, 0.02]), benchmark=daily([0.01]))
        assert s.beta is None and s.benchmark_total_return is None

    def test_money_weighted_from_flows_and_terminal_value(self):
        """1000 in on the first day, worth 1100 a year later."""
        r = pd.Series([0.0, 0.1], index=pd.to_datetime(["2025-01-01", "2026-01-01"]))
        flows = pd.Series([1000.0, 0.0], index=r.index)
        s = performance_summary(r, flows=flows, terminal_value=1100.0)
        assert s.money_weighted == pytest.approx(0.10, abs=1e-6)

    def test_money_weighted_is_none_without_both_inputs(self):
        r = daily([0.01, 0.02])
        assert performance_summary(r, flows=pd.Series([1.0], index=r.index[:1])).money_weighted is None
        assert performance_summary(r, terminal_value=100.0).money_weighted is None

    def test_nans_are_dropped(self):
        s = performance_summary(daily([0.02, float("nan"), 0.0, 0.01]))
        assert s.observations == 3 and s.sharpe == pytest.approx(math.sqrt(252))
