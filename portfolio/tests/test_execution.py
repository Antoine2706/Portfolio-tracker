"""Trading costs, anchored to a contract note.

The first version of this model assumed a 1.32% transaction tax and a 2 EUR
minimum commission, and concluded that turnover-based strategies were not
viable in this account. A MeDirect contract note said 0.12% and zero. The
conclusion inverted, and nothing in the code had been wrong -- the inputs had
been assumed.

So the anchor case here is that document, reproduced to the cent, and the
rest of the file is about the model's refusal to invent the numbers it does
not have.
"""

from __future__ import annotations

import pytest

from portfolio.agents.execution import (BELGIAN_TOB_BANDS, FRENCH_FTT_RATE,
                                        KEYTRADE, MEDIRECT, CostModel,
                                        InstrumentCost, UnknownCost, cost_table)
from portfolio.core.models import AssetClass, Instrument

BANKS = "DE000A2QP372"          # the instrument on the contract note
GOLD = "IE00B579F325"
SCHNEIDER = "FR0000121972"


class TestTheContractNote:
    """MeDirect, 2 February 2026, 114 units at 17.76 on Euronext Amsterdam."""

    NOTIONAL = 2024.87
    TOB = 2.43
    TOTAL = 2027.30

    def test_the_note_is_internally_consistent(self):
        """Check the fixture before trusting it as one."""
        assert self.NOTIONAL + self.TOB == pytest.approx(self.TOTAL, abs=0.005)
        assert self.TOB / self.NOTIONAL == pytest.approx(0.0012, abs=5e-7)

    def test_the_model_reproduces_the_explicit_charges(self):
        """Spread and slippage are off: a contract note cannot contain them,
        because they are paid inside the execution price."""
        model = CostModel(account_value=15_000.0, slippage_bps=0.0,
                          per_instrument={BANKS: InstrumentCost(
                              broker="MeDirect", tob_rate=0.0012,
                              tob_observed=True, half_spread_bps=0.0,
                              spread_observed=True)})
        assert model.instrument_cost(BANKS, self.NOTIONAL, "buy") == \
            pytest.approx(self.TOB, abs=0.005)

    def test_commission_is_zero_at_this_broker(self):
        assert MEDIRECT.commission(self.NOTIONAL) == 0.0
        assert MEDIRECT.commission(50.0) == 0.0

    def test_the_band_that_was_assumed_before_was_eleven_times_larger(self):
        """The size of the error a single document removed."""
        assumed = BELGIAN_TOB_BANDS["accumulating_registered"]
        observed = BELGIAN_TOB_BANDS["observed_medirect_etf"]
        assert assumed / observed == pytest.approx(11.0)


class TestTwoBrokers:
    def test_a_flat_fee_makes_small_trades_expensive(self):
        assert KEYTRADE.commission(200.0) / 200.0 == pytest.approx(0.01225)
        assert KEYTRADE.commission(250.0) / 250.0 == pytest.approx(0.0098)

    def test_the_minimum_economic_trade_differs_by_broker(self):
        """The point: one number would be wrong at both."""
        assert MEDIRECT.minimum_economic_trade() == 0.0
        assert KEYTRADE.minimum_economic_trade() == 490.0

    def test_a_stricter_ceiling_raises_the_floor(self):
        assert KEYTRADE.minimum_economic_trade(25.0) == 980.0

    def test_an_unknown_tier_is_refused_not_extrapolated(self):
        """Only the first band was published; the rest is not knowledge."""
        with pytest.raises(UnknownCost, match="tiers stop at 250"):
            KEYTRADE.commission(1000.0)

    def test_an_unrecorded_broker_is_refused(self):
        model = CostModel(per_instrument={"X": InstrumentCost(broker="Nowhere")})
        with pytest.raises(UnknownCost, match="fee structure is not in this model"):
            model.instrument_cost("X", 1000.0, "buy")


class TestRefusingToGuess:
    def test_an_unrecorded_tax_band_refuses(self):
        """An ETC is a debt security and does not take a fund's band."""
        model = CostModel(per_instrument={GOLD: InstrumentCost(tob_rate=None)})
        with pytest.raises(UnknownCost, match="no transaction tax rate"):
            model.instrument_cost(GOLD, 500.0, "buy")

    def test_the_refusal_says_what_to_do_about_it(self):
        model = CostModel(per_instrument={GOLD: InstrumentCost(tob_rate=None)})
        with pytest.raises(UnknownCost, match="Read the rate off its contract note"):
            model.instrument_cost(GOLD, 500.0, "sell")

    def test_a_blank_cell_is_not_read_as_zero(self, tmp_path):
        """The silently-wrong failure this column exists to prevent."""
        from portfolio.data.store import DataMode, DataStore
        store = DataStore(mode=DataMode.USER, root=tmp_path)
        store.directory.mkdir(parents=True, exist_ok=True)
        store.instruments_path.write_text(
            "isin,name,tob_rate\n"
            f"{GOLD},Invesco Physical Gold ETC,\n", encoding="utf-8")
        assert store.load_instruments()[GOLD].tob_rate is None

    def test_assumptions_names_every_unobserved_input(self):
        model = CostModel(per_instrument={
            BANKS: InstrumentCost(tob_rate=0.0012, tob_observed=True,
                                  half_spread_bps=8.0, spread_observed=True),
            GOLD: InstrumentCost(tob_rate=None)})
        text = " ".join(model.assumptions())
        assert BANKS not in text, "an observed input was listed as an assumption"
        assert "NOT RECORDED" in text and GOLD in text

    def test_a_fully_observed_model_says_so(self):
        model = CostModel(per_instrument={
            BANKS: InstrumentCost(tob_rate=0.0012, tob_observed=True,
                                  half_spread_bps=8.0, spread_observed=True)})
        assert model.assumptions() == ["every cost input has been observed"]


class TestSidedTaxes:
    def test_the_french_tax_is_charged_on_purchases_only(self):
        model = CostModel(per_instrument={SCHNEIDER: InstrumentCost(
            tob_rate=BELGIAN_TOB_BANDS["equity"], half_spread_bps=4.0,
            buy_tax_rate=FRENCH_FTT_RATE)})
        buy = model.instrument_cost(SCHNEIDER, 1000.0, "buy")
        sell = model.instrument_cost(SCHNEIDER, 1000.0, "sell")
        assert buy - sell == pytest.approx(1000.0 * FRENCH_FTT_RATE)

    def test_a_share_costs_far_more_to_trade_than_the_funds(self):
        """The equity band plus the FTT, against the fund band. Worth knowing
        before a policy decides to rebalance it monthly."""
        model = CostModel(per_instrument={
            SCHNEIDER: InstrumentCost(tob_rate=BELGIAN_TOB_BANDS["equity"],
                                      half_spread_bps=4.0,
                                      buy_tax_rate=FRENCH_FTT_RATE),
            BANKS: InstrumentCost(tob_rate=0.0012, half_spread_bps=8.0)})
        assert model.round_trip_bps(SCHNEIDER, 1000.0) == pytest.approx(122.0)
        assert model.round_trip_bps(BANKS, 1000.0) == pytest.approx(44.0)

    def test_an_invalid_side_is_refused(self):
        with pytest.raises(ValueError, match="side must be"):
            CostModel().instrument_cost("X", 100.0, "hold")


class TestCostTable:
    def _instrument(self, isin, **kw):
        return Instrument(isin, kw.pop("name", "Some Fund"),
                          kw.pop("asset_class", AssetClass.ETF), "EUR", **kw)

    def test_it_projects_the_instrument_records(self):
        book = {
            BANKS: self._instrument(BANKS, broker="MeDirect", tob_rate=0.0012,
                                    tob_observed=True, half_spread_bps=8.0),
            GOLD: self._instrument(GOLD, broker="Keytrade", tob_rate=None,
                                   asset_class=AssetClass.ETC),
        }
        table = cost_table(book)
        assert table[BANKS].tob_rate == 0.0012 and table[BANKS].tob_observed
        assert table[GOLD].tob_rate is None and table[GOLD].broker == "Keytrade"

    def test_a_missing_spread_falls_back_conservatively_and_says_so(self):
        book = {BANKS: self._instrument(BANKS, broker="MeDirect",
                                        tob_rate=0.0012)}
        table = cost_table(book)
        assert table[BANKS].half_spread_bps == 8.0
        assert not table[BANKS].spread_observed
        assert "half-spread" in " ".join(
            CostModel(per_instrument=table).assumptions())

    def test_a_spread_cannot_be_observed_without_a_value(self):
        """Marking a number observed while leaving it blank would be the worst
        of both: an estimate wearing the label of evidence."""
        book = {BANKS: self._instrument(BANKS, tob_rate=0.0012,
                                        half_spread_bps=None,
                                        spread_observed=True)}
        assert not cost_table(book)[BANKS].spread_observed


class TestCostsScale:
    def test_cost_is_proportional_where_the_fee_is(self):
        model = CostModel(per_instrument={BANKS: InstrumentCost()})
        small = model.instrument_cost(BANKS, 500.0, "buy") / 500.0
        large = model.instrument_cost(BANKS, 5000.0, "buy") / 5000.0
        assert small == pytest.approx(large)

    def test_both_legs_of_a_rebalance_are_charged(self):
        model = CostModel(account_value=10_000.0,
                          per_instrument={"A": InstrumentCost(),
                                          "B": InstrumentCost()})
        cost = model.cost(0.1, {"A": 0.5, "B": 0.5}, {"A": 0.6, "B": 0.4})
        one_leg = model.instrument_cost("A", 1000.0, "buy")
        assert cost * 10_000.0 == pytest.approx(2 * one_leg)

    def test_moving_into_cash_costs_the_sell_side_only(self):
        model = CostModel(account_value=10_000.0,
                          per_instrument={"A": InstrumentCost()})
        cost = model.cost(0.1, {"A": 1.0}, {"A": 0.8})
        assert cost * 10_000.0 == pytest.approx(
            model.instrument_cost("A", 2000.0, "sell"))
