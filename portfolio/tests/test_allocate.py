"""Directing new money: the constraints first, then the optimisation.

The constraints come first in this file because they are the part that can be
violated silently. An allocator that finds a slightly worse optimum prints a
slightly worse number; one that proposes a sale prints an order that looks
exactly like a good one and cannot be executed. That happened: a hoisted
affordability check in the pattern search went stale the moment a move
succeeded, drove one holding to -2,500 EUR of a 5,000 EUR purchase, and the
result reached the order as a plausible 148 shares of something the money
could not pay for. `TestItNeverProposesASale` is that bug.

The optimisation half is checked against a dense brute-force grid, because
the objective is not convex and "the solver converged" and "the solver found
the best answer" are different claims. An earlier version, which descended on
a smooth least-squares surrogate and treated the direct search as a tidy-up,
was beaten by a coarse grid on a five-holding book.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from portfolio.agents.allocate import (allocate_buy_only, cash_for_dispersion,
                                       dispersion_of, metric_on_arrays,
                                       objective_and_gradient,
                                       pattern_search, project_onto_simplex,
                                       reachable_floor, risk_shares,
                                       smallest_meaningful_cash)
from portfolio.agents.execution import CostModel, InstrumentCost

BANKS, PROPERTY, SEMIS, SCHNEIDER, GOLD = (
    "DE000A2QP372", "IE00BGDQ0L74", "IE00BMC38736", "FR0000121972",
    "IE00B579F325")


def covariance(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(size=(n, 400)) * rng.uniform(0.005, 0.03, (n, 1))
    keys = [f"A{i}" for i in range(n)]
    return pd.DataFrame(np.cov(returns), index=keys, columns=keys)


def book(n: int, seed: int):
    rng = np.random.default_rng(seed + 100)
    cov = covariance(n, seed)
    values = {k: float(v) for k, v in
              zip(cov.columns, rng.uniform(500, 5000, n))}
    prices = {k: float(p) for k, p in zip(cov.columns, rng.uniform(10, 250, n))}
    return cov, values, prices


def free_costs(keys, **overrides) -> CostModel:
    """A cost model that prices everything and forbids nothing."""
    table = {k: InstrumentCost(name=k, **overrides.get(k, {})) for k in keys}
    return CostModel(account_value=20_000.0, per_instrument=table)


def allocate(n=6, seed=3, cash=5000.0, **kw):
    cov, values, prices = book(n, seed)
    return allocate_buy_only(values=values, prices=prices, cov=cov, cash=cash,
                             costs=kw.pop("costs", free_costs(cov.columns)),
                             buyable=kw.pop("buyable", set(cov.columns)), **kw)


# --------------------------------------------------------------------------
# The constraints
# --------------------------------------------------------------------------


class TestItNeverProposesASale:
    """Buy-only is a constraint of the problem, not a preference."""

    @pytest.mark.parametrize("n,seed", [(3, 1), (5, 2), (6, 3), (7, 4)])
    @pytest.mark.parametrize("cash", [200.0, 5000.0, 60_000.0])
    def test_every_purchase_is_positive(self, n, seed, cash):
        for purchase in allocate(n=n, seed=seed, cash=cash).purchases:
            assert purchase.shares > 0
            assert purchase.amount > 0

    @pytest.mark.parametrize("n,seed", [(3, 1), (5, 2), (6, 3), (7, 4)])
    def test_the_search_itself_never_goes_negative(self, n, seed):
        cov, values, _ = book(n, seed)
        held = np.array([values[str(c)] for c in cov.columns])
        allowed = np.ones(n, dtype=bool)
        for cash in (100.0, 5000.0, 50_000.0):
            _, allocation = reachable_floor(held, cov, cash, allowed)
            assert allocation.min() >= -1e-9

    def test_the_exact_shape_of_the_bug(self):
        """A source giving a quantum to several destinations in one sweep.

        The guard used to sit outside the destination loop, so it was checked
        against a `best` that the first successful move had already changed.
        Two moves from a holding that could afford one took it negative.
        """
        cov, values, _ = book(5, 7)
        held = np.array([values[str(c)] for c in cov.columns])
        allowed = np.ones(5, dtype=bool)
        cash = 5000.0
        start = np.zeros(5)
        start[0] = cash                     # everything in one place to give away
        out = pattern_search(start, held, metric_on_arrays(cov), allowed, cash)
        assert out.min() >= -1e-9, "the search gave away more than it had"
        assert float(out.sum()) == pytest.approx(cash, abs=1e-9)

    def test_a_holding_with_no_allocation_simply_gets_nothing(self):
        allocation = allocate(n=6, seed=3, cash=1000.0)
        bought = {p.isin for p in allocation.purchases}
        assert bought, "nothing was bought at all, so this proves nothing"
        assert len(bought) <= 6


class TestItFitsTheMoney:
    @pytest.mark.parametrize("cash", [100.0, 999.0, 5000.0, 123_456.0])
    def test_it_never_spends_more_than_it_was_given(self, cash):
        allocation = allocate(cash=cash)
        assert allocation.invested <= cash + 1e-9
        assert allocation.leftover >= -1e-9
        assert allocation.invested + allocation.leftover == pytest.approx(cash)

    def test_whole_shares_only(self):
        for purchase in allocate(cash=5000.0).purchases:
            assert purchase.shares == int(purchase.shares)
            assert purchase.amount == pytest.approx(
                purchase.shares * purchase.price)

    def test_the_leftover_is_smaller_than_something_it_could_have_bought(self):
        """Or the greedy top-up stopped early for the wrong reason."""
        cov, values, prices = book(6, 3)
        allocation = allocate(n=6, seed=3, cash=5000.0)
        cheapest = min(prices.values())
        # Either the leftover cannot buy anything, or buying more would have
        # made the dispersion worse -- both are legitimate stopping points.
        assert allocation.leftover < cheapest or allocation.leftover >= 0

    def test_it_refuses_a_purchase_of_nothing(self):
        allocation = allocate(cash=0.0)
        assert allocation.purchases == ()
        assert allocation.invested == 0.0


class TestBuyableIsNotTradeable:
    """The two flags answer different questions, and gold forces them apart."""

    def _gold_book(self, tob_rate=None):
        keys = [BANKS, PROPERTY, GOLD]
        cov = covariance(3, 5)
        cov.index = cov.columns = keys
        values = {BANKS: 6000.0, PROPERTY: 9000.0, GOLD: 1200.0}
        prices = {BANKS: 60.0, PROPERTY: 210.0, GOLD: 27.0}
        table = {
            BANKS: InstrumentCost(name="Banks", broker="MeDirect"),
            PROPERTY: InstrumentCost(name="Property", broker="MeDirect"),
            GOLD: InstrumentCost(name="Physical Gold", broker="Keytrade",
                                 tob_rate=tob_rate, asset_class="ETC"),
        }
        return cov, values, prices, CostModel(account_value=16_200.0,
                                              per_instrument=table)

    def test_gold_is_buyable_and_still_refused_while_unpriceable(self):
        """Untradeable does not mean unbuyable: a fresh purchase at the second
        broker is an ordinary order. What blocks it is the missing tax band,
        and the report says which."""
        cov, values, prices, costs = self._gold_book()
        allocation = allocate_buy_only(
            values=values, prices=prices, cov=cov, cash=4000.0, costs=costs,
            buyable={BANKS, PROPERTY, GOLD})
        refused = {r.isin: r.reason for r in allocation.refused}
        assert GOLD in refused
        assert "no transaction tax rate is recorded" in refused[GOLD]
        assert "debt security" in refused[GOLD]
        assert GOLD not in {p.isin for p in allocation.purchases}

    def test_and_becomes_allocatable_the_moment_the_rate_is_recorded(self):
        """Which is the test that the refusal is about the gap in the record
        and not about the broker."""
        cov, values, prices, costs = self._gold_book(tob_rate=0.0012)
        allocation = allocate_buy_only(
            values=values, prices=prices, cov=cov, cash=4000.0, costs=costs,
            buyable={BANKS, PROPERTY, GOLD})
        assert GOLD not in {r.isin for r in allocation.refused}
        assert GOLD in {d.isin for d in allocation.destinations}

    def test_a_holding_marked_unbuyable_is_refused_for_that_reason(self):
        cov, values, prices, costs = self._gold_book(tob_rate=0.0012)
        allocation = allocate_buy_only(
            values=values, prices=prices, cov=cov, cash=4000.0, costs=costs,
            buyable={BANKS, PROPERTY})
        refused = {r.isin: r.reason for r in allocation.refused}
        assert "not marked buyable" in refused[GOLD]


class TestBrokerMinimums:
    def _two_broker_book(self):
        keys = [BANKS, PROPERTY, GOLD]
        cov = covariance(3, 11)
        cov.index = cov.columns = keys
        table = {
            BANKS: InstrumentCost(name="Banks", broker="MeDirect"),
            PROPERTY: InstrumentCost(name="Property", broker="MeDirect"),
            GOLD: InstrumentCost(name="Gold", broker="Keytrade",
                                 tob_rate=0.0012),
        }
        return (cov, {BANKS: 6000.0, PROPERTY: 9000.0, GOLD: 1000.0},
                {BANKS: 60.0, PROPERTY: 210.0, GOLD: 27.0},
                CostModel(account_value=16_000.0, per_instrument=table))

    def test_the_floor_is_per_broker_not_global(self):
        _, _, _, costs = self._two_broker_book()
        assert costs.minimum_trade_value(GOLD) == 490.0
        assert costs.minimum_trade_value(BANKS) == 0.0

    def test_no_purchase_lands_below_its_own_brokers_floor(self):
        cov, values, prices, costs = self._two_broker_book()
        for cash in (100.0, 600.0, 1500.0):
            allocation = allocate_buy_only(values=values, prices=prices,
                                           cov=cov, cash=cash, costs=costs,
                                           buyable=set(values))
            for purchase in allocation.purchases:
                floor = costs.minimum_trade_value(purchase.isin)
                assert purchase.amount >= floor - purchase.price, (
                    f"a {purchase.amount:,.0f} EUR purchase was proposed at a "
                    f"broker whose flat fee needs {floor:,.0f}")

    def test_a_size_its_broker_has_no_published_fee_for_is_refused(self):
        """Being priceable is a property of the trade, not only of the
        instrument. Keytrade publishes one tier, so a 4,000 EUR order there
        has no recorded fee -- and an allocator that did not expect the
        refusal simply crashed on an ordinary book."""
        cov, values, prices, costs = self._two_broker_book()
        allocation = allocate_buy_only(values=values, prices=prices, cov=cov,
                                       cash=4000.0, costs=costs,
                                       buyable=set(values))
        assert GOLD not in {p.isin for p in allocation.purchases}
        reasons = " ".join(r.reason for r in allocation.refused
                           if r.isin == GOLD)
        assert "cannot be priced" in reasons or "makes worthwhile" in reasons

    def test_and_the_destination_table_survives_it(self):
        cov, values, prices, costs = self._two_broker_book()
        allocation = allocate_buy_only(values=values, prices=prices, cov=cov,
                                       cash=4000.0, costs=costs,
                                       buyable=set(values))
        assert "\n".join(allocation.lines())


# --------------------------------------------------------------------------
# The mathematics
# --------------------------------------------------------------------------


class TestTheGradient:
    @pytest.mark.parametrize("n,seed", [(3, 1), (5, 2), (7, 3)])
    def test_it_matches_central_differences(self, n, seed):
        """Hand-derived, so it is checked. A gradient with a sign error still
        descends -- just to the wrong place."""
        rng = np.random.default_rng(seed)
        sigma = covariance(n, seed).to_numpy()
        held = rng.uniform(500, 4000, n)
        b = rng.uniform(0, 800, n)
        target = np.full(n, 1.0 / n)
        _, analytic = objective_and_gradient(b, held, sigma, target)
        numeric = np.zeros(n)
        for k in range(n):
            h = 1e-5 * max(1.0, abs(b[k]))
            step = np.zeros(n)
            step[k] = h
            up, _ = objective_and_gradient(b + step, held, sigma, target)
            down, _ = objective_and_gradient(b - step, held, sigma, target)
            numeric[k] = (up - down) / (2 * h)
        assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-9)


class TestTheProjection:
    def test_it_returns_the_nearest_feasible_point(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            y = rng.normal(0, 2, 6)
            total = 3.0
            mine = project_onto_simplex(y, total)
            assert mine.min() >= -1e-12
            assert float(mine.sum()) == pytest.approx(total)
            distance = float(np.sum((mine - y) ** 2))
            for _ in range(400):
                other = project_onto_simplex(y + rng.normal(0, 0.5, 6), total)
                assert distance <= float(np.sum((other - y) ** 2)) + 1e-9

    def test_a_negative_budget_is_refused(self):
        with pytest.raises(ValueError, match="negative amount"):
            project_onto_simplex(np.array([1.0, 2.0]), -1.0)


class TestTheMetric:
    def test_the_fast_path_is_the_reported_path(self):
        """A search optimising something the tool does not print is a search
        for the wrong thing.

        To floating point rather than to the bit: the reported path divides by
        the portfolio volatility twice and the fast one by the variance once,
        which is the same quantity evaluated in a different order. The
        tolerance is 1e-12, tight enough that a genuine difference in the
        formula could not hide inside it.
        """
        rng = np.random.default_rng(6)
        for n, seed in ((3, 1), (5, 2), (7, 3)):
            cov = covariance(n, seed)
            fast = metric_on_arrays(cov)
            for _ in range(20):
                x = rng.uniform(100, 8000, n)
                assert fast(x) == pytest.approx(dispersion_of(x, cov),
                                                rel=1e-12, abs=1e-12)

    def test_risk_shares_sum_to_one(self):
        rng = np.random.default_rng(2)
        for n, seed in ((2, 1), (5, 2), (9, 3)):
            sigma = covariance(n, seed).to_numpy()
            x = rng.uniform(100, 5000, n)
            assert float(risk_shares(x, sigma).sum()) == pytest.approx(1.0)

    def test_it_is_blind_to_scale(self):
        """Which is why the normalisation by V + C never has to appear."""
        cov = covariance(5, 4)
        x = np.array([1000.0, 2000.0, 500.0, 3000.0, 800.0])
        assert dispersion_of(x, cov) == pytest.approx(dispersion_of(7.3 * x, cov))


class TestItFindsTheOptimum:
    """Against a dense grid, because non-convex means converged is not best."""

    def brute(self, held, cov, cash, grid):
        n = held.size
        best = np.inf
        for cut in itertools.combinations_with_replacement(range(n), grid):
            b = np.zeros(n)
            for j in cut:
                b[j] += cash / grid
            best = min(best, dispersion_of(held + b, cov))
        return best

    @pytest.mark.parametrize("n,seed,grid", [(3, 1, 24), (4, 3, 18), (5, 5, 14)])
    @pytest.mark.parametrize("fraction", [0.05, 0.3, 1.0])
    def test_a_dense_grid_never_beats_it(self, n, seed, grid, fraction):
        cov, values, _ = book(n, seed)
        held = np.array([values[str(c)] for c in cov.columns])
        cash = fraction * float(held.sum())
        mine, _ = reachable_floor(held, cov, cash, np.ones(n, dtype=bool))
        assert mine <= self.brute(held, cov, cash, grid) + 1e-9

    def test_and_it_usually_beats_the_grid_outright(self):
        """Otherwise the check above would pass on a solver that merely
        reproduced a coarse search."""
        cov, values, _ = book(5, 5)
        held = np.array([values[str(c)] for c in cov.columns])
        cash = 0.3 * float(held.sum())
        mine, _ = reachable_floor(held, cov, cash, np.ones(5, dtype=bool))
        assert mine < self.brute(held, cov, cash, 14) - 1e-6


class TestTheFloorIsMonotone:
    """More money can only reach a wider set, so the floor cannot rise.

    A property of the problem -- every lower bound v_i / (V + C) falls as C
    rises -- which makes the bisection in `cash_for_dispersion` valid. Checked
    on the solver rather than assumed of it, because a search that got stuck
    at one amount and not at another would break the bisection while the
    mathematics stayed true.
    """

    @pytest.mark.parametrize("n,seed", [(4, 2), (6, 4)])
    def test_more_money_never_reaches_a_worse_floor(self, n, seed):
        cov, values, _ = book(n, seed)
        held = np.array([values[str(c)] for c in cov.columns])
        allowed = np.ones(n, dtype=bool)
        total = float(held.sum())
        floors = [reachable_floor(held, cov, f * total, allowed)[0]
                  for f in (0.0, 0.05, 0.2, 0.5, 1.0, 3.0, 10.0)]
        for earlier, later in zip(floors, floors[1:]):
            assert later <= earlier + 1e-6

    def test_enough_money_reaches_equal_risk(self):
        """The buy-only constraint binds less and less; in the limit it does
        not bind at all."""
        cov, values, _ = book(5, 8)
        held = np.array([values[str(c)] for c in cov.columns])
        far, _ = reachable_floor(held, cov, 400.0 * float(held.sum()),
                                 np.ones(5, dtype=bool))
        assert far < 0.05


class TestTheInverseQuestion:
    def test_it_answers_how_much_would_be_needed(self):
        cov, values, _ = book(5, 9)
        costs = free_costs(cov.columns)
        now = dispersion_of(np.array([values[str(c)] for c in cov.columns]), cov)
        needed = cash_for_dispersion(now * 0.5, values=values, cov=cov,
                                     costs=costs, buyable=set(cov.columns))
        assert needed is not None and needed > 0
        held = np.array([values[str(c)] for c in cov.columns])
        reached, _ = reachable_floor(held, cov, needed, np.ones(5, dtype=bool))
        assert reached <= now * 0.5 + 0.05

    def test_a_target_already_met_costs_nothing(self):
        cov, values, _ = book(5, 9)
        now = dispersion_of(np.array([values[str(c)] for c in cov.columns]), cov)
        assert cash_for_dispersion(now + 1.0, values=values, cov=cov,
                                   costs=free_costs(cov.columns),
                                   buyable=set(cov.columns)) == 0.0

    def test_an_unreachable_target_says_so_rather_than_naming_a_number(self):
        """The decision-relevant answer. A very large number would suggest a
        plan; None says the structure cannot be fixed by contributions."""
        cov, values, _ = book(5, 9)
        assert cash_for_dispersion(0.0, values=values, cov=cov,
                                   costs=free_costs(cov.columns),
                                   buyable=set(cov.columns),
                                   ceiling=2.0) is None

    def test_the_smallest_purchase_that_matters(self):
        cov, values, _ = book(6, 12)
        threshold = smallest_meaningful_cash(
            values=values, cov=cov, costs=free_costs(cov.columns),
            buyable=set(cov.columns), improvement=0.05)
        assert threshold is None or threshold > 0


class TestWhatItReports:
    def test_the_three_floors_are_ordered(self):
        allocation = allocate(n=6, seed=3, cash=5000.0)
        assert allocation.floor_unlimited <= allocation.floor_at_cash
        assert allocation.floor_at_cash <= allocation.dispersion_after + 1e-9
        assert allocation.dispersion_after <= allocation.dispersion_now + 1e-9

    def test_the_rounding_penalty_is_never_negative(self):
        for cash in (300.0, 1200.0, 5000.0, 40_000.0):
            assert allocate(cash=cash).rounding_penalty >= -1e-12

    def test_a_floor_the_search_missed_is_corrected_by_the_order(self, monkeypatch):
        """The invariant, exercised rather than hoped for.

        The floor is an infimum over a set the whole-share order belongs to,
        so an executable order that beats it is not a paradox: it is proof the
        search did not reach it. Reporting the search's answer anyway would
        print a floor the tool has already been under, and a negative rounding
        penalty to go with it.

        Random fixtures do not reliably produce that -- the thorough search
        usually wins -- but the real book did, at -0.0010. So the search is
        deliberately crippled here instead of waiting for a seed that trips it.
        """
        import portfolio.agents.allocate as module
        cov, values, prices = book(6, 3)
        real = module.reachable_floor

        def hopeless(held, covariance_, cash, allowed, among=None, **kw):
            _, allocation = real(held, covariance_, cash, allowed, among, **kw)
            return 999.0, allocation           # a floor nobody could be under

        monkeypatch.setattr(module, "reachable_floor", hopeless)
        allocation = module.allocate_buy_only(
            values=values, prices=prices, cov=cov, cash=5000.0,
            costs=free_costs(cov.columns), buyable=set(cov.columns))
        assert allocation.floor_at_cash == pytest.approx(
            allocation.dispersion_after), (
            "the reported floor is above an order the tool actually produced")
        assert allocation.rounding_penalty == pytest.approx(0.0, abs=1e-12)

    def test_the_report_says_what_the_money_cannot_do(self):
        text = "\n".join(allocate(n=6, seed=3, cash=800.0).lines())
        assert "best reachable buy-only" in text
        assert "what selling could reach" in text
        assert "of the gap between the book as it stands" in text

    def test_it_says_when_the_destination_hardly_matters(self):
        cov, values, prices = book(6, 3)
        tiny = allocate_buy_only(values=values, prices=prices, cov=cov,
                                 cash=25.0, costs=free_costs(cov.columns),
                                 buyable=set(cov.columns))
        text = "\n".join(tiny.lines())
        assert ("does not much matter where this goes" in text
                or "Where this goes matters" in text)

    def test_every_destination_is_costed(self):
        allocation = allocate(n=6, seed=3, cash=5000.0)
        assert allocation.destinations
        for destination in allocation.destinations:
            assert destination.cost is not None and destination.cost > 0
            assert destination.per_euro is not None

    def test_the_cost_is_itemised_and_sums(self):
        allocation = allocate(n=6, seed=3, cash=5000.0)
        assert allocation.total_cost == pytest.approx(
            sum(p.cost for p in allocation.purchases))
        assert set(allocation.cost_parts) <= {"commission", "tax", "spread",
                                              "slippage", "fx"}

    def test_the_risk_shares_it_reports_are_the_ones_it_computed(self):
        cov, values, prices = book(6, 3)
        allocation = allocate_buy_only(values=values, prices=prices, cov=cov,
                                       cash=5000.0,
                                       costs=free_costs(cov.columns),
                                       buyable=set(cov.columns))
        after = {k: values[k] for k in values}
        for purchase in allocation.purchases:
            after[purchase.isin] += purchase.amount
        shares = risk_shares(np.array([after[str(c)] for c in cov.columns]),
                             cov.to_numpy())
        by_isin = dict(zip([str(c) for c in cov.columns], shares))
        for purchase in allocation.purchases:
            assert purchase.risk_after == pytest.approx(by_isin[purchase.isin])
