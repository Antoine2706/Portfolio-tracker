"""The account's hard constraints, and the one that would break a backtest quietly.

The gold ETC is held at a second broker. Moving weight between it and the rest
of the book is not a rebalance -- it is a sale at one institution, settlement,
a cash transfer, and a purchase at the other, taking about a week with the
portfolio out of position throughout. A rebalancing policy assumes weight can
move between holdings; across brokers it cannot.

An optimiser that is not told this proposes gold trades, and they are
unexecutable. The failure is quiet: the weights look reasonable, the backtest
runs, and the result describes a portfolio nobody could have held. So the
constraint is enforced structurally and tested here, including the case that
actually arose while building it -- a frozen holding drifting between the
decision and the execution, so that even proposing the weight it had a moment
ago implies an order by the time the order is placed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.base import Proposal
from portfolio.agents.execution import CostModel, InstrumentCost, UnknownCost
from portfolio.agents.referee import Referee, Refereed
from portfolio.eval.harness import Execution, Panel, buy_and_hold, walk_forward

GOLD = "IE00B579F325"
ETF_A, ETF_B = "IE00BKM4GZ66", "DE000A2QP372"


@pytest.fixture
def costs():
    return CostModel(account_value=15_000.0, per_instrument={
        GOLD: InstrumentCost(broker="Keytrade", tob_rate=None),
        ETF_A: InstrumentCost(broker="MeDirect"),
        ETF_B: InstrumentCost(broker="MeDirect")})


@pytest.fixture
def referee(costs):
    return Referee(costs=costs, tradeable={ETF_A, ETF_B})


HELD = {GOLD: 0.20, ETF_A: 0.40, ETF_B: 0.40}


class TestTheFrozenHolding:
    def test_a_proposal_that_moves_gold_is_refused(self, referee):
        """The test the constraint exists for."""
        verdict = referee.review(
            Proposal({GOLD: 0.25, ETF_A: 0.45, ETF_B: 0.30}, 0.5,
                     "trim the ETFs into gold", "p"), HELD)
        assert not verdict.allowed
        assert GOLD in verdict.explain()
        assert "not tradeable" in verdict.explain()

    def test_it_is_refused_however_small_the_move(self, referee):
        nudge = {GOLD: 0.2001, ETF_A: 0.3999, ETF_B: 0.40}
        assert not referee.review(Proposal(nudge, 0.5, "nudge", "p"), HELD).allowed

    def test_leaving_it_alone_is_allowed(self, referee):
        verdict = referee.review(
            Proposal({GOLD: 0.20, ETF_A: 0.50, ETF_B: 0.30}, 0.5,
                     "shift between the two ETFs", "p"), HELD)
        assert verdict.allowed and verdict.explain() == "accepted as proposed"

    def test_floating_point_noise_is_not_a_trade(self, referee):
        """An optimiser's last bit should not read as an attempt to trade."""
        noisy = {GOLD: 0.20 + 1e-15, ETF_A: 0.50, ETF_B: 0.30 - 1e-15}
        assert referee.review(Proposal(noisy, 0.5, "noise", "p"), HELD).allowed

    def test_the_wrapper_raises_rather_than_silently_correcting(self, referee):
        """Clipping the proposal would hide a policy solving the wrong problem."""
        class Greedy:
            name = "greedy"

            def observe(self, view):
                return Proposal({GOLD: 0.5, ETF_A: 0.25, ETF_B: 0.25}, 0.5,
                                "all the gold", self.name)

        class View:
            as_of = pd.Timestamp("2026-01-05")
            held = HELD

        with pytest.raises(ValueError, match="not tradeable"):
            Refereed(Greedy(), referee).observe(View())


class TestMinimumTradeSize:
    def test_it_is_per_broker_not_global(self, costs):
        """A flat fee at one broker and none at the other invert the economics."""
        assert costs.minimum_trade_value(ETF_A) == 0.0
        assert costs.minimum_trade_value(GOLD) == 490.0

    def test_a_trade_too_small_for_its_fee_is_skipped_not_rejected(self, costs):
        """Declining to trade is a sensible answer; refusing to run is not."""
        referee = Referee(costs=costs, tradeable={ETF_A, ETF_B, GOLD})
        # 0.5% of 15,000 is 75 EUR, well under Keytrade's 490 EUR floor.
        verdict = referee.review(
            Proposal({GOLD: 0.205, ETF_A: 0.395, ETF_B: 0.40}, 0.5, "tiny", "p"),
            HELD)
        assert verdict.allowed
        assert verdict.weights[GOLD] == pytest.approx(0.20)
        assert "below the 490 EUR" in verdict.explain()

    def test_a_trade_large_enough_goes_through(self, costs):
        referee = Referee(costs=costs, tradeable={ETF_A, ETF_B, GOLD})
        verdict = referee.review(
            Proposal({GOLD: 0.25, ETF_A: 0.35, ETF_B: 0.40}, 0.5, "big", "p"),
            HELD)
        assert verdict.allowed and verdict.weights[GOLD] == pytest.approx(0.25)


class TestLeverageAndSign:
    def test_a_levered_proposal_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unlevered"):
            Proposal({ETF_A: 0.8, ETF_B: 0.9}, 0.5, "levered", "p")

    def test_the_referee_re_checks_rather_than_trusting(self, referee):
        """It is the last thing between a proposal and an order."""
        sneaky = Proposal({ETF_A: 0.5, ETF_B: 0.5}, 0.5, "fine", "p")
        object.__setattr__(sneaky, "weights", {ETF_A: 0.9, ETF_B: 0.9})
        assert not referee.review(sneaky, HELD).allowed


# --------------------------------------------------------------------------
# The harness side of the same constraint
# --------------------------------------------------------------------------


def panel(seed: int = 1, periods: int = 400) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=periods)
    cols = [GOLD, ETF_A, ETF_B]
    rets = pd.DataFrame(rng.normal(0.0002, 0.011, (periods, 3)),
                        index=dates, columns=cols)
    rets.iloc[0] = 0.0
    return Panel(closes=100.0 * (1.0 + rets).cumprod())


class Target:
    """Proposes a fixed target, including for the frozen holding."""
    name = "target"

    def __init__(self, weights):
        self.weights = weights

    def observe(self, view):
        return Proposal(dict(self.weights), 0.5, "hold this target", self.name)


class TestTheHarnessNeverTradesAFrozenHolding:
    def test_a_frozen_weight_is_taken_from_the_drifted_book(self):
        """The bug this found: a frozen holding drifts between the decision
        and the execution one bar later, so holding its target still implies
        an order -- small enough to look like rounding, real enough to be an
        unexecutable trade and, here, an unpriceable one."""
        p = panel()
        costs = CostModel(account_value=15_000.0, per_instrument={
            GOLD: InstrumentCost(broker="Keytrade", tob_rate=None),
            ETF_A: InstrumentCost(), ETF_B: InstrumentCost()})
        start = {GOLD: 0.2, ETF_A: 0.4, ETF_B: 0.4}

        result = walk_forward(
            p, Target({GOLD: 0.2, ETF_A: 0.5, ETF_B: 0.3}), warmup=100,
            rebalance_every=21, cost_model=costs,
            execution=Execution.NEXT_CLOSE, initial_weights=start,
            frozen=frozenset({GOLD}))

        assert result.decisions
        for d in result.decisions:
            assert d.weights_after[GOLD] == pytest.approx(d.weights_before[GOLD]), (
                "the frozen holding was traded")

    def test_without_the_freeze_it_would_be_traded(self):
        """Prove the protection bites: remove it and the cost model refuses,
        because gold's tax band is deliberately not recorded."""
        p = panel()
        costs = CostModel(account_value=15_000.0, per_instrument={
            GOLD: InstrumentCost(broker="Keytrade", tob_rate=None),
            ETF_A: InstrumentCost(), ETF_B: InstrumentCost()})
        with pytest.raises(UnknownCost, match="no transaction tax rate"):
            walk_forward(p, Target({GOLD: 0.2, ETF_A: 0.5, ETF_B: 0.3}),
                         warmup=100, rebalance_every=21, cost_model=costs,
                         execution=Execution.NEXT_CLOSE,
                         initial_weights={GOLD: 0.2, ETF_A: 0.4, ETF_B: 0.4})

    def test_buy_and_hold_pays_nothing_at_all(self):
        """It starts from the book it already owns, so there is nothing to buy."""
        p = panel()
        costs = CostModel(account_value=15_000.0, per_instrument={
            GOLD: InstrumentCost(broker="Keytrade", tob_rate=None),
            ETF_A: InstrumentCost(), ETF_B: InstrumentCost()})
        result = buy_and_hold(p, {GOLD: 0.2, ETF_A: 0.4, ETF_B: 0.4},
                              warmup=100, cost_model=costs)
        assert result.total_cost == 0.0
        assert float(result.turnover.sum()) == pytest.approx(0.0, abs=1e-12)

    def test_the_starting_weights_are_honoured_and_drift(self):
        p = panel()
        start = {GOLD: 0.2, ETF_A: 0.4, ETF_B: 0.4}
        result = buy_and_hold(p, start, warmup=100)
        first = result.weights.iloc[0]
        assert float(first[ETF_A]) == pytest.approx(0.4, abs=0.03)
        assert not np.allclose(result.weights.iloc[0].to_numpy(),
                               result.weights.iloc[-1].to_numpy())

    def test_initial_weights_naming_an_absent_instrument_are_refused(self):
        with pytest.raises(ValueError, match="not in the panel"):
            buy_and_hold(panel(), {"XX0000000000": 1.0}, warmup=100)
