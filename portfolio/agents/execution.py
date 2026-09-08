"""What a trade costs, per broker and per instrument, from contract notes.

A backtest with costs switched off is not a backtest with an optimistic
assumption in it; it is a different experiment whose answer is known in
advance. Every policy that trades more looks better when trading is free.

The first version of this module assumed a 1.32% Belgian transaction tax and
a 2 EUR minimum commission, and concluded that turnover-based strategies were
not viable in this account. A contract note said otherwise on both counts. So
the design rule here is now explicit: **every number is either read off a
document or marked as an estimate, and the model can say which.**
`CostModel.assumptions()` lists everything not yet observed, and the backtest
prints it beside the result.

What the contract note said
---------------------------
MeDirect Bank SA, 2 February 2026, iShares EURO STOXX Banks 30-15 UCITS ETF
DE (DE000A2QP372), executed on Euronext Amsterdam, 114 units at 17.76:

    notional      2,024.87 EUR
    commission        0.00 EUR    0.000%
    TOB               2.43 EUR    0.120%
    total         2,027.30 EUR

Two facts follow, and they move the project further than any modelling
decision has.

**The tax band is 0.12%, not 1.32%.** Round trip 0.24%. At 200% annual
turnover that is 0.48% a year rather than 5.28%. Turnover-based strategies go
from dead to worth measuring, which is what the harness exists for.

**Commission is zero on this instrument at this broker.** Which means the
bid-ask spread -- invisible on any contract note, because it is paid inside
the execution price -- is now very likely the *largest* component of turnover
cost rather than the smallest. A cost model that omits it would overstate
every strategy. It is modelled per instrument, conservatively, and flagged as
an estimate until quotes are observed.

Two brokers, opposite shapes
----------------------------
Six holdings sit at MeDirect: zero commission, cost proportional to value.
The gold ETC sits at Keytrade: a flat fee per order, from 2.45 EUR. A flat
fee inverts the economics of small trades -- 2.45 EUR is 1.23% of a 200 EUR
order and 0.25% of a 1,000 EUR one -- so the minimum economically sensible
trade is a property of the *broker*, derived here rather than set as one
number in config.

The gold position is also, for now, unpriceable: an ETC is a debt security
rather than a fund and does not automatically sit in the band the ETF sits
in. Its rate is recorded as unknown and this module refuses to price a trade
in it, rather than defaulting to a plausible number. That refusal is the
project's convention, and it is also load-bearing: gold is held at a second
broker, so it cannot be traded against the rest of the book anyway.

Not modelled, deliberately
--------------------------
Market impact. At 15,000 EUR across seven instruments no order is large
enough relative to displayed depth for impact to be separable from the spread
already charged. Saying so beats inventing a coefficient.

Limit orders. The observed trade was a market order, which pays the full
half-spread. A systematic policy could work limit orders and pay materially
less. That is a design question for the execution agent, noted rather than
built.
"""

from __future__ import annotations

import dataclasses

__all__ = ["BrokerFees", "InstrumentCost", "CostModel", "UnknownCost", "cost_table",
           "MEDIRECT", "KEYTRADE", "BROKERS", "OBSERVED_TOB_RATE",
           "FRENCH_FTT_RATE", "BELGIAN_TOB_BANDS"]


class UnknownCost(RuntimeError):
    """A cost parameter has not been recorded, so the trade cannot be priced.

    Raised instead of falling back to a default. A backtest that silently
    prices a trade with a guessed tax rate produces a number that looks like
    a measurement, and the whole point of this module is that the numbers in
    it came from somewhere.
    """


# Observed on the MeDirect contract note above. The Belgian tax on
# stock-exchange transactions is charged on both purchase and sale.
OBSERVED_TOB_RATE = 0.0012

# The bands, for reference when confirming the other holdings. Which one
# applies turns on facts about each specific compartment -- fund or debt
# security, accumulating or distributing, registered for public distribution
# in Belgium or not -- that cannot be derived from an ISIN. Read each off its
# own contract note.
BELGIAN_TOB_BANDS = {
    "observed_medirect_etf": 0.0012,
    "accumulating_registered": 0.0132,
    "equity": 0.0035,
    "unknown": None,
}

# France taxes acquisitions of shares in French-headquartered companies above
# 1bn EUR of market capitalisation. Schneider Electric is on that list. Buy
# side only, regardless of the buyer's residence or the venue used, which is
# why `instrument_cost` takes a side.
FRENCH_FTT_RATE = 0.004


@dataclasses.dataclass(frozen=True)
class BrokerFees:
    """One broker's commission structure.

    Two shapes are represented because this account has both, and they are
    not variations of each other -- they invert which trades are affordable.

    *Proportional or free*: `fixed` + `rate` x value, floored at `minimum`.
    MeDirect on the observed instrument is all zeros, so cost scales with
    size and no trade is too small on fee grounds.

    *Tiered flat*: a fee per order that depends on the order's size band.
    Keytrade is this. A flat fee is a fixed cost, so the smaller the trade
    the worse the rate, and below some size the fee eats more than the trade
    can plausibly earn.

    >>> MEDIRECT.commission(200), MEDIRECT.commission(5000)
    (0.0, 0.0)
    >>> KEYTRADE.commission(200), KEYTRADE.commission(250)
    (2.45, 2.45)

    A tiered broker refuses rather than extrapolating past its known bands:

    >>> KEYTRADE.commission(1000)
    Traceback (most recent call last):
        ...
    portfolio.agents.execution.UnknownCost: Keytrade's fee for a 1000.00 EUR order is not recorded; the known tiers stop at 250 EUR. Read the fee off a contract note before pricing a trade this size.
    """
    name: str
    fixed: float = 0.0
    rate: float = 0.0
    minimum: float = 0.0
    maximum: float | None = None
    # Ascending (upper bound of the band, fee for that band). Empty means the
    # fixed/rate/minimum form above is used instead.
    tiers: tuple[tuple[float, float], ...] = ()

    def commission(self, value: float) -> float:
        if value <= 0:
            return 0.0
        if self.tiers:
            for upper, fee in self.tiers:
                if value <= upper:
                    return float(fee)
            raise UnknownCost(
                f"{self.name}'s fee for a {value:.2f} EUR order is not "
                f"recorded; the known tiers stop at {self.tiers[-1][0]:.0f} "
                f"EUR. Read the fee off a contract note before pricing a "
                f"trade this size.")
        fee = self.fixed + self.rate * value
        fee = max(fee, self.minimum)
        return float(fee if self.maximum is None else min(fee, self.maximum))

    def minimum_economic_trade(self, max_cost_bps: float = 50.0) -> float:
        """Smallest trade whose commission is within `max_cost_bps` of its value.

        Derived from the fee structure rather than configured, because it is
        a different number at each broker and setting one global figure would
        be wrong at both.

        A broker with no fixed component has no fee-driven minimum at all --
        cost is proportional, so a 50 EUR trade costs the same rate as a
        5,000 EUR one and only the spread argues against it:

        >>> MEDIRECT.minimum_economic_trade()
        0.0

        A flat fee gives a hard floor. At 2.45 EUR and a 50 bps ceiling, a
        trade below 490 EUR is spending more on the fee than the threshold
        allows:

        >>> KEYTRADE.minimum_economic_trade()
        490.0
        >>> KEYTRADE.minimum_economic_trade(max_cost_bps=25.0)
        980.0
        """
        if max_cost_bps <= 0:
            raise ValueError(f"max_cost_bps must be positive, got {max_cost_bps}")
        ceiling = max_cost_bps / 10_000.0
        if self.tiers:
            floor_fee = self.tiers[0][1]
        elif self.fixed or self.minimum:
            floor_fee = max(self.fixed, self.minimum)
        else:
            return 0.0
        return float(floor_fee / ceiling) if floor_fee else 0.0


# Zero commission observed on DE000A2QP372, 2 February 2026. Whether that
# generalises to the other five MeDirect holdings is NOT established -- it is
# one contract note for one instrument. `assumptions()` says so.
MEDIRECT = BrokerFees(name="MeDirect")

# "From 2.45 EUR for an order up to 250 EUR on Euronext Brussels, Paris,
# Amsterdam", tiering upward with order size. Only the first band is known,
# so the model refuses above it rather than extrapolating.
KEYTRADE = BrokerFees(name="Keytrade", tiers=((250.0, 2.45),))

BROKERS = {b.name: b for b in (MEDIRECT, KEYTRADE)}


@dataclasses.dataclass(frozen=True)
class InstrumentCost:
    """Everything cost-related that is a fact about one holding.

    `tob_rate` of None means not recorded, and pricing a trade in this
    instrument raises. That is the intended behaviour for the gold ETC: it is
    a debt security rather than a fund, so the band observed on an ETF does
    not carry over, and an ETC priced at the ETF's rate would be a plausible
    wrong number rather than a loud missing one.

    `half_spread_bps` may be an estimate -- spreads are paid inside the
    execution price and never appear on a contract note -- but it must be
    declared as one via `spread_observed`, so the model can report which of
    its inputs are evidence and which are judgement.
    """
    broker: str = "MeDirect"
    tob_rate: float | None = OBSERVED_TOB_RATE
    tob_observed: bool = False
    half_spread_bps: float = 8.0
    spread_observed: bool = False
    buy_tax_rate: float = 0.0            # the French FTT, where it applies
    needs_fx: bool = False
    note: str = ""


@dataclasses.dataclass(frozen=True)
class CostModel:
    """Cost of one rebalance, as a fraction of portfolio value.

    Called by the harness with the weights before and after a trade, so it
    sees each instrument's traded value and side rather than only aggregate
    turnover. That is what lets a flat fee bite on the small lines and a
    buy-side tax bite on one side, exactly as they do in reality.

    The observed contract note, reproduced exactly. Spread and slippage are
    switched off here because the note cannot contain them -- they are paid
    inside the execution price -- so this checks the model against the
    *explicit* charges only, which is all a contract note can settle:

    >>> note = CostModel(account_value=15_000.0, slippage_bps=0.0,
    ...                  per_instrument={
    ...     "DE000A2QP372": InstrumentCost(broker="MeDirect",
    ...                                    tob_rate=0.0012, tob_observed=True,
    ...                                    half_spread_bps=0.0,
    ...                                    spread_observed=True)})
    >>> round(note.instrument_cost("DE000A2QP372", 2024.87, "buy"), 2)
    2.43

    What the model adds on top is the implicit half, and at these fee levels
    the two are comparable: 24 bps of tax against 20 bps of estimated spread
    and slippage on a round trip, so 45% of the total cost is a number nobody
    has measured. On the thinner holdings -- a property or sector ETF rather
    than a broad tracker -- the estimated half is likely the larger one. That
    is the thing to hold in mind when reading any result from here.

    With commission at zero, a 1,000 EUR trade costs 1.20 of tax and 1.00 of
    estimated spread and slippage, and the same either way round:

    >>> etf = CostModel(per_instrument={"X": InstrumentCost(half_spread_bps=8.0)})
    >>> round(etf.instrument_cost("X", 1000.0, "buy"), 4)
    2.2
    >>> round(etf.instrument_cost("X", 1000.0, "sell"), 4)
    2.2

    Unless it is the French share, where the buy side carries the FTT as well
    and costs nearly twice the sell side:

    >>> fr = CostModel(per_instrument={"FR": InstrumentCost(
    ...     tob_rate=0.0035, buy_tax_rate=FRENCH_FTT_RATE)})
    >>> round(fr.instrument_cost("FR", 1000.0, "buy"), 4)
    8.5
    >>> round(fr.instrument_cost("FR", 1000.0, "sell"), 4)
    4.5

    An instrument whose tax band has not been recorded is refused, not guessed:

    >>> unknown = CostModel(per_instrument={"GOLD": InstrumentCost(tob_rate=None)})
    >>> unknown.instrument_cost("GOLD", 500.0, "buy")
    Traceback (most recent call last):
        ...
    portfolio.agents.execution.UnknownCost: no transaction tax rate is recorded for GOLD, so a trade in it cannot be priced. An ETC is a debt security rather than a fund and does not automatically take a fund's band. Read the rate off its contract note and set tob_rate.
    """

    # Costs are computed against a fixed notional rather than the running
    # portfolio value: a flat fee needs a euro amount, and letting that amount
    # drift with performance would make the cost of a trade depend on the
    # returns that preceded it -- a path dependence nobody wants to reason
    # about for a second-order effect.
    account_value: float = 15_000.0

    slippage_bps: float = 2.0            # the gap between the quote and the fill
    fx_spread_bps: float = 25.0          # on the currency leg, where there is one
    tob_cap: float = 1_300.0             # EUR per transaction, in the 0.12% band

    per_instrument: dict = dataclasses.field(default_factory=dict)
    brokers: dict = dataclasses.field(default_factory=lambda: dict(BROKERS))

    # The rate at which a trade is judged too small to be worth making. Not a
    # cost: a constraint the referee applies, derived per broker below.
    max_trade_cost_bps: float = 50.0

    def facts(self, isin: str) -> InstrumentCost:
        return self.per_instrument.get(isin, InstrumentCost())

    def broker_for(self, isin: str) -> BrokerFees:
        facts = self.facts(isin)
        broker = self.brokers.get(facts.broker)
        if broker is None:
            raise UnknownCost(
                f"{isin} is recorded at broker {facts.broker!r}, whose fee "
                f"structure is not in this model. Known: "
                f"{sorted(self.brokers)}.")
        return broker

    def minimum_trade_value(self, isin: str) -> float:
        """The smallest trade in this instrument worth making, in euros.

        Per broker, because it is set by the fee structure and the two
        brokers here have opposite ones. At MeDirect it is zero on fee
        grounds; at Keytrade a 2.45 EUR flat fee puts it near 490 EUR.

        >>> book = {"ETF": InstrumentCost(broker="MeDirect"),
        ...         "GOLD": InstrumentCost(broker="Keytrade", tob_rate=None)}
        >>> m = CostModel(per_instrument=book)
        >>> m.minimum_trade_value("ETF"), m.minimum_trade_value("GOLD")
        (0.0, 490.0)
        """
        return self.broker_for(isin).minimum_economic_trade(self.max_trade_cost_bps)

    def instrument_cost(self, isin: str, value: float, side: str = "buy") -> float:
        """Cost of trading `value` euros of one instrument, in euros."""
        if value <= 0:
            return 0.0
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        facts = self.facts(isin)

        if facts.tob_rate is None:
            raise UnknownCost(
                f"no transaction tax rate is recorded for {isin}, so a trade "
                f"in it cannot be priced. An ETC is a debt security rather "
                f"than a fund and does not automatically take a fund's band. "
                f"Read the rate off its contract note and set tob_rate.")

        commission = self.broker_for(isin).commission(value)
        spread = value * (facts.half_spread_bps + self.slippage_bps) / 10_000.0
        fx = value * self.fx_spread_bps / 10_000.0 if facts.needs_fx else 0.0
        tax = min(value * facts.tob_rate, self.tob_cap)
        if side == "buy":
            tax += value * facts.buy_tax_rate
        return float(commission + spread + fx + tax)

    def cost(self, turnover: float, before: dict[str, float],
             after: dict[str, float]) -> float:
        """Cost of moving from `before` to `after`, as a fraction of value.

        `turnover` is accepted for symmetry with the harness and unused: the
        per-instrument weight changes carry strictly more information, and
        recomputing from the aggregate would lose the flat-fee and one-sided-
        tax effects this model exists to capture.
        """
        total = 0.0
        for isin in set(before) | set(after):
            change = float(after.get(isin, 0.0)) - float(before.get(isin, 0.0))
            traded = abs(change) * self.account_value
            if traded <= 0:
                continue
            total += self.instrument_cost(isin, traded,
                                          "buy" if change > 0 else "sell")
        return float(total / self.account_value) if self.account_value else 0.0

    def round_trip_bps(self, isin: str, value: float) -> float:
        """Cost of buying and selling `value` euros, in basis points.

        The figure to quote when asking whether a strategy's edge can survive
        its trading. On the observed instrument at a conservative spread:

        >>> m = CostModel(per_instrument={"X": InstrumentCost()})
        >>> round(m.round_trip_bps("X", 2000.0), 1)
        44.0

        Twenty-four of those forty-four basis points are tax, read off a
        contract note, and twenty are spread and slippage, which are not.
        """
        both = (self.instrument_cost(isin, value, "buy")
                + self.instrument_cost(isin, value, "sell"))
        return float(10_000.0 * both / value) if value else 0.0

    def assumptions(self) -> list[str]:
        """Every input that is judgement rather than evidence.

        Printed beside any result this model contributed to. The point is
        that a reader can tell, without opening the source, which numbers
        came off a document and which are a considered guess -- and the
        spread, which is now the largest single component, is a guess.
        """
        out: list[str] = []
        for isin in sorted(self.per_instrument):
            facts = self.per_instrument[isin]
            if facts.tob_rate is None:
                out.append(f"{isin}: transaction tax NOT RECORDED; trades in "
                           f"it are refused rather than priced")
            elif not facts.tob_observed:
                out.append(f"{isin}: transaction tax {facts.tob_rate:.4%} "
                           f"assumed, not read off a contract note")
            if not facts.spread_observed:
                out.append(f"{isin}: half-spread {facts.half_spread_bps:.1f} bps "
                           f"estimated; spreads are paid inside the execution "
                           f"price and never appear on a contract note")
        if not out:
            out.append("every cost input has been observed")
        return out


def cost_table(instruments) -> dict:
    """Build the per-instrument cost facts from the instrument records.

    One source of truth. The broker, tax band, spread estimate and one-sided
    taxes are recorded on the instrument because they are facts about that
    holding at that broker; this projects them into the shape the cost model
    consumes, rather than maintaining a second table keyed by the same ISIN
    that would drift from the first.

    A blank half-spread falls back to a deliberately conservative 8 bps and is
    marked unobserved, so `assumptions()` names it. Being wrong in the cheap
    direction on the largest unmeasured component is how a backtest comes to
    flatter a strategy.
    """
    items = (instruments.items() if hasattr(instruments, "items")
             else [(i.isin, i) for i in instruments])
    out = {}
    for isin, inst in items:
        out[isin] = InstrumentCost(
            broker=inst.broker or "MeDirect",
            tob_rate=inst.tob_rate,
            tob_observed=bool(inst.tob_observed),
            half_spread_bps=(8.0 if inst.half_spread_bps is None
                             else float(inst.half_spread_bps)),
            spread_observed=bool(inst.spread_observed
                                 and inst.half_spread_bps is not None),
            buy_tax_rate=float(inst.buy_tax_rate),
        )
    return out
