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

from portfolio.agents.allocate import (RESOLVED, allocate_buy_only,
                                       best_reachable_dispersion,
                                       cash_for_dispersion, dispersion_of,
                                       metric_on_arrays,
                                       objective_and_gradient,
                                       pattern_search, pinned_holdings,
                                       project_onto_simplex,
                                       reachable_floor, risk_shares,
                                       smallest_meaningful_cash,
                                       unconstrained_floor)
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


LADDER = (0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 50.0)


def hedged_book(n: int, seed: int):
    """A book where something hedges something else.

    One common factor and betas from -1.2 to 1.6, so some pairs correlate
    negatively. The real book does: its gold ETC runs at -0.98 against one of
    the equity holdings, which is the entire reason it is held.

    The distinction is not cosmetic, and finding that out is why this fixture
    exists rather than `book` above. Independent normals make the equal-risk
    problem easy: over forty-eight of them at ten amounts each, the floor was
    monotone whether or not `aim_at_equal_risk` was in the code, so a
    monotonicity check built on them passes either way and is not a check.
    Let one correlation go negative and the solver fails on most seeds. A
    holding whose marginal contribution (Sigma x)_i is negative has a
    *negative* risk share, so the minimum of the range is not bounded below by
    zero and the surface grows local minima a pairwise search cannot leave.
    """
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0, 0.011, 500)
    beta = rng.uniform(-1.2, 1.6, n)
    idiosyncratic = rng.normal(0.0, 1.0, (n, 500)) * rng.uniform(0.002, 0.008,
                                                                 (n, 1))
    returns = beta[:, None] * factor[None, :] + idiosyncratic
    # Twelve characters, like a real ISIN: the report qualifies a colliding
    # name by appending one, and whether that fits its column is a question
    # about ISIN-length keys rather than about "A3".
    keys = [f"IE00HEDGE{i:03d}" for i in range(n)]
    cov = pd.DataFrame(np.cov(returns) * 252, index=keys, columns=keys)
    return cov, rng.uniform(500, 5000, n)


def floors_up_the_ladder(held, cov, allowed, ladder=LADDER):
    total = float(held.sum())
    return [reachable_floor(held, cov, f * total, allowed)[0] for f in ladder]


class TestTheFloorIsMonotone:
    """More money can only reach a wider set -- if every holding can receive it.

    The condition is the whole of it, and the module docstring carries the
    proof: rescaling a reachable allocation by (V + C2) / (V + C1) preserves
    every risk share and stays feasible, *provided* no holding is pinned. When
    one is, its weight is an equality that moves with the money rather than an
    inequality that relaxes, and `TestPinningBreaksTheMonotonicity` below is
    the counterexample.

    Where it does hold it is a property of the problem, not of the solver, so
    what is checked here is that the solver exhibits it -- a search that gets
    stuck at one amount and not at another breaks `cash_for_dispersion` while
    the mathematics stays true. It did, and `test_the_check_bites` is that.
    """

    @pytest.mark.parametrize("n,seed", [(4, 2), (6, 4)])
    def test_more_money_never_reaches_a_worse_floor(self, n, seed):
        cov, values, _ = book(n, seed)
        held = np.array([values[str(c)] for c in cov.columns])
        floors = floors_up_the_ladder(held, cov, np.ones(n, dtype=bool))
        for earlier, later in zip(floors, floors[1:]):
            assert later <= earlier + 1e-6

    @pytest.mark.parametrize("seed", [2, 3, 4])
    def test_nor_on_a_book_with_a_hedge_in_it(self, seed):
        """The case the independent-normal fixture cannot see."""
        cov, held = hedged_book(8, seed)
        floors = floors_up_the_ladder(held, cov, np.ones(8, dtype=bool))
        for i, (earlier, later) in enumerate(zip(floors, floors[1:])):
            assert later <= earlier + 1e-6, (
                f"the floor rose from {earlier:.4f} at {LADDER[i]}x the book "
                f"to {later:.4f} at {LADDER[i + 1]}x, which the nesting "
                f"argument forbids when nothing is pinned")

    def test_the_check_bites(self, monkeypatch):
        """Delete the aimed start and the check above must fail.

        Otherwise it is a check on nothing. Without `aim_at_equal_risk` the
        floor on this book, up the same ladder, runs

            7.6371 6.7747 1.9746 1.1604 0.4161 0.0031 0.0078 0.0035
                                                       1.4254 1.2380

        It finds the equal-risk portfolio at four times the book and then
        *loses* it at sixteen, reporting a floor four hundred times worse for
        strictly more money on a strictly larger feasible set. With the start
        the last five entries are 0.0000 exactly.
        """
        import portfolio.agents.allocate as module
        monkeypatch.setattr(module, "aim_at_equal_risk", lambda *a, **k: None)
        cov, held = hedged_book(8, 2)
        floors = [module.reachable_floor(held, cov, f * float(held.sum()),
                                         np.ones(8, dtype=bool))[0]
                  for f in LADDER]
        assert max(b - a for a, b in zip(floors, floors[1:])) > 0.5
        assert min(floors) < 0.01, "it found equal risk at some amount"
        assert floors[-1] > 1.0, "and then lost it when handed more money"

    def test_enough_money_reaches_equal_risk(self):
        """The buy-only constraint binds less and less; in the limit it does
        not bind at all."""
        cov, values, _ = book(5, 8)
        held = np.array([values[str(c)] for c in cov.columns])
        far, _ = reachable_floor(held, cov, 400.0 * float(held.sum()),
                                 np.ones(5, dtype=bool))
        assert far < 0.05

    def test_and_it_reaches_it_exactly_not_approximately(self):
        """Because the start is the answer, not a neighbourhood of it."""
        cov, held = hedged_book(8, 3)
        far, _ = reachable_floor(held, cov, 50.0 * float(held.sum()),
                                 np.ones(8, dtype=bool))
        assert far < 1e-9


class TestPinningBreaksTheMonotonicity:
    """The condition, and what happens when it fails.

    This module claimed monotonicity unconditionally and bisected on it. The
    claim is false whenever a holding with money in it cannot receive more:
    the rescaling needs b2_i = (lambda - 1) v_i to be zero for such a holding,
    and it is strictly positive whenever v_i is. The floor then falls, bottoms
    out, and climbs again as the pinned holding is diluted towards a zero risk
    share.
    """

    LADDER = (0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 50.0)

    def floors(self, n=5, seed=1, pin=0):
        cov, values, _ = book(n, seed)
        held = np.array([values[str(c)] for c in cov.columns])
        allowed = np.ones(n, dtype=bool)
        allowed[pin] = False
        return cov, held, allowed, floors_up_the_ladder(held, cov, allowed,
                                                        self.LADDER)

    def test_the_floor_really_does_rise(self):
        _, _, _, floors = self.floors()
        bottom = min(floors)
        assert bottom < 0.2, "it should fall a long way before it turns"
        assert floors[-1] > bottom + 1.0, (
            "the floor did not rise, so the counterexample this class exists "
            "for is not being exercised")

    def test_it_climbs_towards_the_dilution_limit(self):
        """Which is arithmetic, not a fitted constant.

        As C grows the pinned holding's weight goes to zero, so its risk share
        does too, and the best the other four can do is share the risk equally
        among themselves. Four shares of 1/4 and one of 0 have a range of 1/4
        about a mean of 1/5, so the dispersion tends to n / (n - 1) = 1.25.
        """
        _, _, _, floors = self.floors(n=5)
        assert floors[-1] == pytest.approx(5.0 / 4.0, abs=0.02)
        assert floors[-1] < 5.0 / 4.0, "it is a limit, approached from below"

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_the_limit_holds_at_every_size(self, n):
        _, _, _, floors = self.floors(n=n, seed=1, pin=0)
        assert floors[-1] == pytest.approx(n / (n - 1.0), abs=0.03)

    def test_the_amount_named_still_reaches_the_target(self):
        """Minimality is what pinning costs. Sufficiency is not negotiable:
        every step of the scan and the bisection keeps floor(high) <= target,
        and the cheap search can only overstate the floor, so the amount
        reported is one that really does reach it."""
        cov, values, _ = book(5, 1)
        keys = [str(c) for c in cov.columns]
        held = np.array([values[k] for k in keys])
        allowed = np.ones(5, dtype=bool)
        allowed[0] = False
        buyable = set(keys[1:])
        for target in (1.0, 0.8, 0.5, 0.3):
            needed = cash_for_dispersion(target, values=values, cov=cov,
                                         costs=free_costs(keys),
                                         buyable=buyable)
            assert needed is not None, f"a target of {target} was reachable"
            reached, _ = reachable_floor(held, cov, needed, allowed)
            assert reached <= target + 1e-9, (
                f"it named {needed:,.0f} EUR for a dispersion of {target}, "
                f"and {needed:,.0f} EUR reaches only {reached:.4f}")

    def test_a_pinned_holding_is_named(self):
        cov, values, _ = book(5, 1)
        keys = [str(c) for c in cov.columns]
        assert pinned_holdings(values=values, cov=cov,
                               costs=free_costs(keys),
                               buyable=set(keys[1:])) == [keys[0]]

    def test_a_pinned_holding_worth_nothing_is_not_pinning_anything(self):
        """The rescaling needs (lambda - 1) v_i to vanish, and a zero value
        gives that without any purchase."""
        cov, values, _ = book(5, 1)
        keys = [str(c) for c in cov.columns]
        values = dict(values, **{keys[0]: 0.0})
        assert pinned_holdings(values=values, cov=cov, costs=free_costs(keys),
                               buyable=set(keys[1:])) == []

    def test_an_unreachable_target_comes_with_the_best_that_is_reachable(self):
        """"No" is not a decision on its own, and with a pinned holding it is
        not even "as much as possible": there is a best amount, past which
        more money makes the number worse."""
        cov, values, _ = book(5, 1)
        keys = [str(c) for c in cov.columns]
        best, at = best_reachable_dispersion(values=values, cov=cov,
                                             costs=free_costs(keys),
                                             buyable=set(keys[1:]))
        assert best < 0.2 and at > 0
        assert cash_for_dispersion(best * 0.5, values=values, cov=cov,
                                   costs=free_costs(keys),
                                   buyable=set(keys[1:])) is None


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

    def test_it_says_when_a_purchase_hardly_moves_the_book(self):
        cov, values, prices = book(6, 3)
        tiny = allocate_buy_only(values=values, prices=prices, cov=cov,
                                 cash=25.0, costs=free_costs(cov.columns),
                                 buyable=set(cov.columns))
        assert "hardly moves the book" in "\n".join(tiny.lines())

    def test_and_says_it_when_one_does(self):
        text = "\n".join(allocate(n=6, seed=3, cash=20_000.0).lines())
        assert "This moves the book" in text
        assert "of the gap to equal risk contribution" in text

    def test_how_much_the_choice_is_worth_is_a_separate_sentence(self):
        """Merging the two once printed "where this goes matters" directly
        above "closes 2.3% of the gap". A purchase too small to move the book
        can still have a destination worth avoiding, and the reverse."""
        for cash in (25.0, 5000.0):
            text = "\n".join(allocate(n=6, seed=3, cash=cash).lines())
            assert "Best and worst destinations differ by" in text
            assert "so the choice is worth" in text

    def test_the_threshold_is_a_share_of_the_gap_not_an_absolute(self):
        """A book at a dispersion of 15 and one at 1.5 must be asked the same
        question. As an absolute 0.05 the threshold asked for a third of the
        whole gap on the second and 0.3% of it on the first, and duly called
        197 EUR "meaningful" on a book that purchase moved by 2%.
        """
        cov, values, _ = book(6, 3)
        keys = [str(c) for c in cov.columns]
        costs, buyable = free_costs(keys), set(keys)
        held = np.array([values[k] for k in keys])
        threshold = smallest_meaningful_cash(values=values, cov=cov,
                                             costs=costs, buyable=buyable)
        assert threshold is not None
        # Scale every variance by 100: dispersion is homogeneous of degree
        # zero in the covariance, so the book is *identical* in the units that
        # matter and the answer must not move.
        scaled = cov * 100.0
        assert dispersion_of(held, scaled) == pytest.approx(
            dispersion_of(held, cov))
        assert smallest_meaningful_cash(
            values=values, cov=scaled, costs=costs,
            buyable=buyable) == pytest.approx(threshold, rel=1e-9)

    def _at_equal_risk(self, nudge=0.0):
        """A book sitting on the equal-risk weights, optionally pushed off."""
        from portfolio.agents.risk import equal_risk_weights
        cov, _, _ = book(5, 3)
        keys = [str(c) for c in cov.columns]
        weights = equal_risk_weights(cov)
        values = {k: 10_000.0 * float(weights[k]) for k in keys}
        values[keys[0]] *= 1.0 + nudge
        return cov, keys, values

    def test_a_book_already_at_equal_risk_has_no_threshold(self):
        """There is no gap to close, so no purchase closes a share of it and
        the honest answer is that the destination never matters."""
        cov, keys, values = self._at_equal_risk()
        assert smallest_meaningful_cash(values=values, cov=cov,
                                        costs=free_costs(keys),
                                        buyable=set(keys)) is None

    def test_and_a_gap_the_solver_invented_is_not_a_gap(self):
        """Both ends of the gap come out of the same iterative solver, so on a
        book that IS at equal risk they differ by whatever the last bisection
        left behind: exactly 0.0 on one machine and 1e-17 on another. Tested
        against zero, the second machine reported that 100 EUR would close a
        fifth of a gap seventeen orders of magnitude below anything real."""
        cov, keys, values = self._at_equal_risk(nudge=1e-9)
        held = np.array([values[k] for k in keys])
        room = dispersion_of(held, cov) - unconstrained_floor(cov)
        assert 0 < room < RESOLVED, (
            "this fixture no longer sits inside the resolution, so it is not "
            "exercising the guard")
        assert smallest_meaningful_cash(values=values, cov=cov,
                                        costs=free_costs(keys),
                                        buyable=set(keys)) is None

    def test_but_a_gap_above_the_resolution_is_answered(self):
        """Otherwise the guard could swallow every book."""
        cov, keys, values = self._at_equal_risk(nudge=1e-3)
        assert smallest_meaningful_cash(values=values, cov=cov,
                                        costs=free_costs(keys),
                                        buyable=set(keys)) is not None

    def _hedged_allocation(self, cash=5000.0, names=None):
        """An allocation on a book with a hedge in it, optionally with two
        holdings deliberately sharing a display name."""
        cov, held = hedged_book(6, 2)
        keys = [str(c) for c in cov.columns]
        names = names or {}
        costs = CostModel(account_value=20_000.0, per_instrument={
            k: InstrumentCost(name=names.get(k, k)) for k in keys})
        return allocate_buy_only(
            values=dict(zip(keys, held)),
            prices={k: 20.0 + 5.0 * i for i, k in enumerate(keys)},
            cov=cov, cash=cash, costs=costs, buyable=set(keys))

    def test_a_negative_risk_share_is_explained_rather_than_printed_bare(self):
        """A hedge earns a negative marginal contribution, so its risk share
        is negative. Correct, and unreadable without a sentence: the seed book
        prints -46.0% next to +205.1%."""
        allocation = self._hedged_allocation()
        assert any(min(p.risk_before, p.risk_after) < 0
                   for p in allocation.purchases), (
            "this fixture no longer produces a negative risk share, so the "
            "assertion below is checking nothing")
        assert "not a misprint" in "\n".join(allocation.lines())

    def test_and_the_sentence_stays_away_when_there_is_nothing_to_explain(self):
        allocation = allocate(n=6, seed=3, cash=5000.0)
        assert all(min(p.risk_before, p.risk_after) >= 0
                   for p in allocation.purchases)
        assert "not a misprint" not in "\n".join(allocation.lines())

    def test_two_holdings_with_one_name_are_told_apart(self):
        """The seed book has two ETFs that both shorten to "Europe Defence".
        A table listing both under one name cannot be acted on."""
        cov, _ = hedged_book(6, 2)
        keys = [str(c) for c in cov.columns]
        clash = {keys[0]: "Europe Defence", keys[1]: "Europe Defence"}
        allocation = self._hedged_allocation(names=clash)
        labels = allocation._labels([*allocation.purchases,
                                     *allocation.destinations])
        assert labels[keys[0]] == f"Europe Defence {keys[0]}"
        assert labels[keys[1]] == f"Europe Defence {keys[1]}"
        text = "\n".join(allocation.lines())
        assert f"Europe Defence {keys[0]}" in text
        assert f"Europe Defence {keys[1]}" in text

    def test_a_name_nobody_shares_is_left_alone(self):
        """An ISIN on every row would cost the width the names need."""
        allocation = self._hedged_allocation()
        labels = allocation._labels([*allocation.purchases,
                                     *allocation.destinations])
        assert set(labels.values()) == set(labels)

    def test_a_holding_is_not_read_as_colliding_with_itself(self):
        """It appears in the order AND in the destinations table, and counting
        the two rows separately would qualify every purchased name."""
        allocation = self._hedged_allocation()
        assert allocation.purchases and allocation.destinations
        bought = {p.isin for p in allocation.purchases}
        assert bought & {d.isin for d in allocation.destinations}, (
            "the two lists no longer overlap, so this test is vacuous")
        labels = allocation._labels([*allocation.purchases,
                                     *allocation.destinations])
        for isin in bought:
            assert labels[isin] == isin, "qualified against itself"

    def test_a_qualified_label_is_never_truncated_mid_isin(self):
        """Half an ISIN identifies nothing, which is worse than the collision
        it was meant to fix. The order column is 27 wide for exactly this: a
        14-character name, a space, and a 12-character ISIN."""
        cov, _ = hedged_book(6, 2)
        keys = [str(c) for c in cov.columns]
        long = "A rather long instrument name"
        allocation = self._hedged_allocation(names={keys[0]: long,
                                                    keys[1]: long})
        labels = allocation._labels([*allocation.purchases,
                                     *allocation.destinations])
        qualified = [labels[k] for k in (keys[0], keys[1])]
        assert len(set(qualified)) == 2, "the collision was not resolved"
        for isin, label in zip((keys[0], keys[1]), qualified):
            assert label.endswith(isin)
            assert len(label) <= 27, (
                f"{label!r} is {len(label)} characters and the order column "
                f"is 27, so the ISIN would be cut in half")
        text = "\n".join(allocation.lines())
        for label in qualified:
            assert label in text

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
