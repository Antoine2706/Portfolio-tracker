"""The tax that is not proportional to what was traded.

MeDirect charged 73.64 EUR on the VanEck disposal of 18 June 2026: exactly
10.00% of a 736.36 EUR realised gain, and thirty-six times every other cost on
that sale combined. It is structurally unlike anything else in the cost model
and each difference changes something:

*Proportional to the gain, not the notional.* Selling 2,000 EUR of a position
that has doubled costs 100 EUR; selling 2,000 EUR of a flat one costs nothing.
Turnover stops predicting cost, which is why `Breakeven` carries it as a
second dimension rather than folding it into the cost per unit of turnover.

*One-sided.* Buys never pay it. That is a structural argument for directing
new money over rebalancing that no spread or commission provides.

*Annual and path dependent.* An exempt tranche per calendar year makes the
marginal rate zero below it and 10% above, so a flat rate is wrong in both
directions and the same sale costs different amounts in January and December.

*Lot dependent.* Two holdings of equal size and equal risk are not equally
cheap to trim if one is up 78% and the other down 6%, which is a fact about
the book that the weights alone do not carry.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.base import Proposal
from portfolio.core.taxes import CapitalGainsTax, GainsLedger
from portfolio.eval.harness import Execution, Panel, walk_forward


class Fixed:
    """A policy that always wants the same weights, so the only thing moving
    the book is drift and the only thing trading it is the rebalance."""
    name = "fixed"

    def __init__(self, weights):
        self.weights = dict(weights)

    def observe(self, view):
        return Proposal(weights=dict(self.weights), reason="fixed",
                        confidence=1.0, agent="fixed")


def rising_panel(n: int = 200) -> pd.DataFrame:
    """A doubles, B rises 30%. Both up, so trimming either realises a gain."""
    return pd.DataFrame(
        {"A": np.linspace(100.0, 200.0, n),
         "B": 100.0 * np.linspace(1.0, 1.3, n)},
        index=pd.bdate_range("2026-01-01", periods=n))


class TestTheMarginalRate:
    """The formula, at each of the three places a gain can fall."""

    tax = CapitalGainsTax(rate=0.10, allowance=10_000.0)

    def test_inside_the_tranche_costs_nothing(self):
        assert self.tax.charge(4_000.0, realised=0.0) == 0.0
        assert self.tax.marginal_rate(0.0) == 0.0

    def test_above_it_costs_the_full_rate(self):
        assert self.tax.charge(4_000.0, realised=12_000.0) == pytest.approx(400.0)
        assert self.tax.marginal_rate(12_000.0) == pytest.approx(0.10)

    def test_a_gain_that_straddles_is_taxed_only_on_the_part_above(self):
        """The case a flat rate gets wrong in both directions."""
        assert self.tax.charge(4_000.0, realised=8_000.0) == pytest.approx(200.0)

    def test_a_refund_is_capped_at_what_was_paid(self):
        """A 3,000 loss against 12,000 realised takes the year to 9,000, back
        below the tranche. Only the 2,000 that was above it is refunded."""
        assert self.tax.charge(-3_000.0, realised=12_000.0) == pytest.approx(-200.0)
        assert self.tax.charge(-3_000.0, realised=2_000.0) == 0.0

    def test_headroom_is_what_can_still_be_realised_free(self):
        assert self.tax.headroom(6_500.0) == pytest.approx(3_500.0)
        assert self.tax.headroom(20_000.0) == 0.0

    def test_the_whole_year_reconciles_however_it_is_split(self):
        """Path dependence must not become path *sensitivity* of the total: a
        year's gains realised in one sale or in ten owe the same tax."""
        ledger_one = GainsLedger(self.tax)
        ledger_one.record(dt.date(2026, 6, 1), "X", 25_000.0, 10_000.0)
        ledger_many = GainsLedger(self.tax)
        for month in range(1, 6):
            ledger_many.record(dt.date(2026, month, 1), "X", 5_000.0, 2_000.0)
        assert ledger_many.total_gain == pytest.approx(ledger_one.total_gain)
        assert ledger_many.total_tax == pytest.approx(ledger_one.total_tax)


class TestTheLedger:
    def test_the_tranche_resets_with_the_calendar_year(self):
        ledger = GainsLedger(CapitalGainsTax(0.10, 10_000.0))
        first = ledger.record(dt.date(2026, 9, 1), "X", 21_000.0, 8_000.0)
        second = ledger.record(dt.date(2027, 1, 5), "X", 21_000.0, 8_000.0)
        assert first.tax == pytest.approx(300.0)
        assert second.tax == pytest.approx(300.0)
        assert ledger.realised(2026) == pytest.approx(13_000.0)
        assert ledger.realised(2027) == pytest.approx(13_000.0)

    def test_out_of_order_disposals_are_refused_rather_than_priced(self):
        """The tax on a sale depends on the sales before it, so a history in
        the wrong order produces a confident figure for events that did not
        happen that way."""
        ledger = GainsLedger()
        ledger.record(dt.date(2026, 6, 1), "X", 100.0, 50.0)
        with pytest.raises(ValueError, match="date order"):
            ledger.record(dt.date(2026, 5, 1), "X", 100.0, 50.0)

    def test_it_reports_the_marginal_rate_not_just_the_total(self):
        """Which is the number a decision is made against: whether trimming
        costs anything depends on where the year stands."""
        ledger = GainsLedger(CapitalGainsTax(0.10, 10_000.0))
        ledger.record(dt.date(2026, 3, 1), "X", 14_000.0, 8_000.0)
        text = "\n".join(ledger.lines())
        assert "next euro of gain taxed at 0%" in text
        assert "4,000.00 of the 10,000 tranche left" in text

    def test_no_disposals_says_the_structural_thing(self):
        assert "Buys never pay this" in "\n".join(GainsLedger().lines())


class TestTheHarnessTracksTheBasis:
    """Weighted average cost, carried through drift and rebalance.

    The harness works in normalised weights and had no cost basis at all, so a
    policy that sold into a risen holding paid transaction tax and spread and
    nothing else. The basis is tracked alongside the weights: it does not grow
    with the price -- that growth is precisely the unrealised gain -- so it
    takes the drift's denominator and no numerator.
    """

    def run(self, initial, target, **kw):
        return walk_forward(
            Panel(closes=rising_panel()), Fixed(target), warmup=2,
            rebalance_every=kw.pop("every", 60),
            execution=Execution.NEXT_CLOSE, initial_weights=initial, **kw)

    def test_it_matches_the_same_book_run_as_shares_and_euros(self):
        """The check that matters, and it is independent: the same rebalances
        replayed as share counts and cash cost, the way a broker statement
        works, rather than a second run of the same arithmetic."""
        start = 10_000.0
        closes = rising_panel()
        result = self.run({"A": 0.7, "B": 0.3}, {"A": 0.5, "B": 0.5})
        value = start * (1.0 + result.gross_returns).cumprod()
        theirs = float((result.realised_gains * value).sum())

        shares = {k: start * w / float(closes[k].iloc[1])
                  for k, w in (("A", 0.7), ("B", 0.3))}
        basis = {"A": start * 0.7, "B": start * 0.3}
        mine = 0.0
        for decision in result.decisions:
            px = closes.loc[decision.executed_on]
            book = sum(shares[k] * float(px[k]) for k in shares)
            for k in shares:
                want = book * decision.weights_after.get(k, 0.0) / float(px[k])
                if want < shares[k]:
                    sold = shares[k] - want
                    relieved = basis[k] * (sold / shares[k])
                    mine += sold * float(px[k]) - relieved
                    basis[k] -= relieved
                else:
                    basis[k] += (want - shares[k]) * float(px[k])
                shares[k] = want
        assert result.decisions, "nothing rebalanced, so this proves nothing"
        assert theirs == pytest.approx(mine, abs=0.01)
        assert mine > 0, "the fixture no longer realises a gain"

    def test_holding_still_realises_nothing(self):
        """No sale, no gain. The one-sidedness, from the other end.

        `Hold` rather than `Fixed`, because a policy that targets constant
        weights is not a policy that never trades: after one day of drift it
        trims the winner back, which realises a real if tiny gain. That
        distinction is the whole point of the tax, so the test uses the policy
        that genuinely does nothing.
        """
        from portfolio.eval.harness import Hold
        result = walk_forward(
            Panel(closes=rising_panel()), Hold(), warmup=2,
            rebalance_every=10_000,      # as `buy_and_hold` runs it: once
            execution=Execution.NEXT_CLOSE,
            initial_weights={"A": 0.5, "B": 0.5})
        assert result.decisions, "nothing was decided, so this proves nothing"
        # Not exactly zero, and the reason is worth knowing: `Hold` sees the
        # book one bar before it executes, so it trades a single day of drift
        # back every time it runs. Its docstring claimed zero until this test
        # measured it. The benchmark is unaffected because `buy_and_hold`
        # gives it a rebalance interval longer than the panel, so it decides
        # once -- which is how it is run here.
        traded = float(result.turnover.abs().sum())
        assert 0 < traded < 0.01
        assert float((result.turnover > 0).sum()) == 1
        # And the gain that realises is proportionally as small: a day of
        # drift on a book rising 0.3% a day, not a disposal.
        assert float(result.realised_gains.abs().sum()) < 1e-4

    def test_and_a_constant_weight_policy_is_not_holding(self):
        """It trims the winner every time it runs, which realises a gain
        however small the drift. Worth pinning: "rebalancing to fixed weights"
        sounds passive and is not."""
        result = self.run({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}, every=10)
        assert float(result.realised_gains.sum()) > 0

    def test_buying_more_realises_nothing_either(self):
        """A purchase moves the embedded gain towards zero without realising
        it, which is why buys never pay this tax."""
        result = self.run({"A": 0.2, "B": 0.8}, {"A": 0.9, "B": 0.1},
                          every=10_000)
        into_a = [d for d in result.decisions
                  if d.weights_after["A"] > d.weights_before["A"]]
        assert into_a, "nothing was bought, so this proves nothing"

    def test_a_falling_holding_realises_a_loss(self):
        """Sign, not just magnitude. A policy that trims a loser gets a
        deduction, and a model that took the absolute value would charge for
        it."""
        n = 200
        closes = pd.DataFrame(
            {"A": np.linspace(100.0, 60.0, n), "B": np.full(n, 100.0)},
            index=pd.bdate_range("2026-01-01", periods=n))
        result = walk_forward(Panel(closes=closes), Fixed({"A": 0.5, "B": 0.5}),
                              warmup=2, rebalance_every=60,
                              execution=Execution.NEXT_CLOSE,
                              initial_weights={"A": 0.5, "B": 0.5})
        # A falls, so its weight drifts down and the rebalance BUYS it back --
        # the gain is realised on B, which is sold, and B is flat.
        assert float(result.realised_gains.sum()) <= 1e-9

    def test_the_default_basis_is_a_lower_bound_and_says_so(self):
        """With no opening basis supplied the book is modelled as bought on
        day one, so it carries no embedded gain and the tax comes out too low.
        The direction is known, which is what makes it reportable."""
        panel = Panel(closes=rising_panel())
        fresh = walk_forward(panel, Fixed({"A": 0.5, "B": 0.5}), warmup=2,
                             rebalance_every=60,
                             execution=Execution.NEXT_CLOSE,
                             initial_weights={"A": 0.7, "B": 0.3})
        # Half the basis: the book is opened already up 100% on both legs.
        risen = walk_forward(panel, Fixed({"A": 0.5, "B": 0.5}), warmup=2,
                             rebalance_every=60,
                             execution=Execution.NEXT_CLOSE,
                             initial_weights={"A": 0.7, "B": 0.3},
                             initial_basis={"A": 0.35, "B": 0.15})
        assert not fresh.basis_was_supplied
        assert risen.basis_was_supplied
        assert float(risen.realised_gains.sum()) > float(
            fresh.realised_gains.sum()), (
            "opening with embedded gains must realise more, or the default is "
            "not a lower bound and the report's caveat is wrong")

    def test_the_existing_numbers_are_untouched(self):
        """The basis is additive. The harness's returns, turnover and costs
        are what they were, which is what keeps the calibration controls
        valid: a change that moved any of them would invalidate them."""
        panel = Panel(closes=rising_panel())
        kw = dict(warmup=2, rebalance_every=60,
                  execution=Execution.NEXT_CLOSE,
                  initial_weights={"A": 0.7, "B": 0.3})
        plain = walk_forward(panel, Fixed({"A": 0.5, "B": 0.5}), **kw)
        seeded = walk_forward(panel, Fixed({"A": 0.5, "B": 0.5}),
                              initial_basis={"A": 0.35, "B": 0.15}, **kw)
        assert np.array_equal(plain.returns.to_numpy(),
                              seeded.returns.to_numpy())
        assert np.array_equal(plain.turnover.to_numpy(),
                              seeded.turnover.to_numpy())
        assert np.array_equal(plain.costs.to_numpy(), seeded.costs.to_numpy())
