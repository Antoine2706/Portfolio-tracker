"""What a trade costs, modelled explicitly enough to argue with.

A backtest with costs switched off is not a backtest with an optimistic
assumption in it; it is a different experiment, and one whose answer is known
in advance. Every policy that trades more looks better without costs, because
turnover is pure upside when it is free. The most common false positive in
this whole field is a strategy that is profitable gross and unprofitable net,
and a harness that cannot produce that finding is not measuring anything.

So costs are mandatory in `walk_forward`, and every parameter is a named
field with a default you can see and change rather than a number buried in
a formula.

The components, at this account size
------------------------------------
For a roughly 15,000 EUR book of UCITS ETFs on XETRA and Euronext plus one
large-cap French equity, four things bind, in this order of importance:

1.  **The Belgian transaction tax.** At 1.32% each way on an accumulating
    UCITS ETF this dwarfs everything else -- it is roughly forty times the
    commission on a 500 EUR trade -- and it alone determines whether any
    high-turnover policy can work. See the warning on `tax_rate` below: this
    number must be confirmed, not trusted.
2.  **The minimum commission.** A fixed 2-3 EUR minimum is 40-60 bps on a
    500 EUR trade and 10-15 bps on a 2,000 EUR one. At this size the
    commission is effectively a fixed cost per line touched, which is an
    argument for rebalancing fewer positions rather than all of them, and
    for a minimum trade size.
3.  **The half-spread**, paid on entry and again on exit. A liquid broad
    tracker sits around 3-5 bps; a thin sector or property ETF is several
    times that. Modelled per instrument, defaulting to the wider figure,
    because being wrong in the cheap direction is what produces a strategy
    that does not survive contact with a broker.
4.  **The French financial transaction tax**, 0.4% on *purchases* of shares
    in French-headquartered companies above 1bn EUR market capitalisation.
    Schneider Electric is one, so this book pays it, and it is charged on the
    buy leg only regardless of where the buyer lives or which venue is used.
    A cost function without a side argument cannot express that at all, which
    is why `instrument_cost` takes one.
5.  **Currency conversion** on the USD-denominated funds, once on the way in.

Market impact is deliberately not modelled. At 15,000 EUR spread over seven
instruments, no order is large enough relative to displayed size for impact
to be distinguishable from the spread already charged. Saying so is better
than adding a term with a made-up coefficient.

Everything here is in basis points of traded value unless named otherwise.
"""

from __future__ import annotations

import dataclasses

__all__ = ["CostModel", "InstrumentCost", "DEFAULT_COST_MODEL", "BELGIAN_TOB_RATES"]


# The Belgian tax on stock-exchange transactions (taxe sur les operations de
# bourse / beurstaks), charged on BOTH the purchase and the sale.
#
# ------------------------------------------------------------------------
# CONFIRM THESE BEFORE TRUSTING ANY RESULT THAT DEPENDS ON THEM.
#
# The rates change, and which one applies turns on facts about each specific
# fund -- whether it accumulates or distributes, and whether it is registered
# for public distribution in Belgium -- that this code cannot determine and
# must not guess. The figures below are the bands as understood when this was
# written and are here to make the model's sensitivity visible, not to state
# the law. Read them off a broker contract note, which shows the tax actually
# charged, and correct them here.
#
# This matters more than any other number in this module: at 1.32% each way,
# a policy rebalancing monthly at 20% turnover pays around 6% a year in tax
# alone, which is larger than the entire effect any of the strategies in the
# shortlist is expected to produce. If that rate is right, the honest
# conclusion is that turnover-based strategies are not viable in this account
# and the tracker's job is to say so.
# ------------------------------------------------------------------------
# The 0.12% and 1.32% bands differ by a factor of eleven and the choice
# between them is decided by whether each specific *compartment* is registered
# for public distribution in Belgium -- registering any share class of a
# sub-fund pulls the whole sub-fund into the higher band for accumulating
# classes. That is a per-instrument fact on the FSMA register, not something
# derivable from the ISIN, and this module defaults to the HIGHER rate on
# purpose: overstating cost understates a strategy, which is the direction an
# honest backtest should err in. Set `tax_rate` down once you have checked,
# and expect the drag figures to fall by roughly an order of magnitude if the
# funds turn out to be unregistered.
BELGIAN_TOB_RATES = {
    "etf_accumulating_be": 0.0132,   # accumulating fund registered in Belgium
    "etf_distributing_be": 0.0012,   # distributing fund registered in Belgium
    "etf_foreign": 0.0012,           # fund not registered for distribution in Belgium
    "equity": 0.0035,                # ordinary shares and bonds
    "none": 0.0,                     # for jurisdictions with no such tax
}


@dataclasses.dataclass(frozen=True)
class InstrumentCost:
    """Per-instrument overrides. Anything left None falls back to the model."""
    half_spread_bps: float | None = None
    tax_rate: float | None = None
    needs_fx: bool = False
    buy_tax_rate: float = 0.0     # buy-side only, e.g. the French FTT


@dataclasses.dataclass(frozen=True)
class CostModel:
    """Cost of one rebalance, as a fraction of portfolio value.

    Called by the harness with the weights before and after a trade, so it
    sees each instrument's traded value rather than only the aggregate
    turnover -- which is what lets the minimum commission bite on the small
    lines, exactly where it does in reality.

    >>> m = CostModel()
    >>> before = {"A": 0.5, "B": 0.5}
    >>> after = {"A": 0.6, "B": 0.4}
    >>> round(m.cost(0.1, before, after) * 10_000, 2)        # basis points
    31.07

    Ten percent one-way turnover on a 15,000 EUR book costs 31 bps of the
    whole portfolio. Six sevenths of that is tax: with the tax switched off
    the same trade costs less than a fifth as much.

    >>> free = dataclasses.replace(m, tax_rate=0.0)
    >>> round(free.cost(0.1, before, after) * 10_000, 2)
    4.67

    Small trades are dominated by the fixed minimum, which is why a
    rebalance that touches every line costs more than one that touches only
    the two lines that actually drifted -- for the same total turnover:

    >>> spread = {f"H{i}": 1 / 7 for i in range(7)}
    >>> nudged = {k: v for k, v in spread.items()}
    >>> nudged["H0"] += 0.007; nudged["H1"] -= 0.007
    >>> two_lines = m.cost(0.007, spread, nudged)
    >>> allseven = {k: v + (0.002 if i % 2 else -0.002)
    ...             for i, (k, v) in enumerate(spread.items())}
    >>> seven_lines = m.cost(0.007, spread, allseven)
    >>> round(two_lines * 10_000, 2), round(seven_lines * 10_000, 2)
    (4.65, 11.32)

    The identical 0.7% of turnover costs two and a half times as much spread
    over seven lines. Those seven trades are 30 EUR each, on which the 2 EUR
    minimum commission alone is 6.7% -- which is what `minimum_trade_value`
    exists to stop the referee from ever proposing.
    """

    # The book this is being modelled for. Costs are computed against a fixed
    # notional rather than the running portfolio value: the minimum commission
    # needs a euro amount, and letting that amount drift with performance
    # would make the cost of a trade depend on the returns that preceded it,
    # which is a path dependence nobody wants to reason about for a
    # second-order effect. Set it to the account's typical size.
    account_value: float = 15_000.0

    commission_fixed: float = 1.00        # EUR per order
    commission_rate: float = 0.0005       # 5 bps of value
    commission_minimum: float = 2.00      # EUR, the binding term at this size
    commission_maximum: float = 30.00     # EUR

    half_spread_bps: float = 8.0          # paid once per trade, each way
    slippage_bps: float = 2.0             # the gap between the quote and the fill
    fx_spread_bps: float = 25.0           # on the currency leg, when there is one

    tax_rate: float = BELGIAN_TOB_RATES["etf_accumulating_be"]
    tax_cap: float = 4_000.0              # EUR per transaction

    # France taxes acquisitions of shares in French-headquartered companies
    # above 1bn EUR of market capitalisation. Schneider Electric is on that
    # list. Applied per instrument via InstrumentCost.buy_tax_rate rather than
    # globally, because it is a fact about the issuer and not about the book.
    FRENCH_FTT: "float" = 0.004

    # Not a cost: a constraint the referee applies. It lives here because it
    # is decided by the cost structure -- below this, the minimum commission
    # eats more than the trade can plausibly earn.
    minimum_trade_value: float = 250.0

    per_instrument: dict = dataclasses.field(default_factory=dict)

    def instrument_cost(self, isin: str, value: float, side: str = "buy") -> float:
        """Cost of trading `value` euros of one instrument, in euros.

        `side` is "buy" or "sell". It exists because some taxes are one-sided:
        the French FTT is charged on acquisition only, and a model that
        charged it both ways -- or neither -- would misprice every rebalance
        touching a French share by tens of basis points.
        """
        if value <= 0:
            return 0.0
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        override: InstrumentCost = self.per_instrument.get(isin, InstrumentCost())

        commission = self.commission_fixed + self.commission_rate * value
        commission = min(max(commission, self.commission_minimum),
                         self.commission_maximum)

        half_spread = override.half_spread_bps
        if half_spread is None:
            half_spread = self.half_spread_bps
        spread_cost = value * (half_spread + self.slippage_bps) / 10_000.0

        fx_cost = value * self.fx_spread_bps / 10_000.0 if override.needs_fx else 0.0

        rate = self.tax_rate if override.tax_rate is None else override.tax_rate
        tax = min(value * rate, self.tax_cap)
        if side == "buy":
            tax += value * override.buy_tax_rate

        return commission + spread_cost + fx_cost + tax

    def cost(self, turnover: float, before: dict[str, float],
             after: dict[str, float]) -> float:
        """Total cost of moving from `before` to `after`, as a fraction of value.

        `turnover` is passed for symmetry with the harness and is not used:
        the per-instrument weight changes carry strictly more information, and
        recomputing the cost from the aggregate would lose exactly the minimum
        commission effect this model exists to capture.
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

    def annual_drag(self, annual_turnover: float, lines: int = 7) -> float:
        """Rough annual cost of running at a given turnover, as a fraction.

        A planning figure rather than a measurement: it assumes the turnover
        is spread evenly over `lines` instruments in equal trades. The harness
        reports the realised drag; this is for asking "is this policy even
        worth backtesting" before spending the effort.

        >>> round(CostModel().annual_drag(2.0) * 100, 2)
        5.97
        >>> round(CostModel().annual_drag(0.5) * 100, 2)
        1.61

        Two hundred percent annual one-way turnover costs about 6.0% a year;
        even a sedate 50% costs 1.6%. For context, volatility targeting is
        typically claimed to add something in the region of 0.1 to 0.3 to a
        Sharpe ratio, which on a 12%-volatility portfolio is roughly 1.2 to
        3.6% of return a year. The cost is the larger number. That is a
        finding, and it is available before a single line of strategy is
        written -- which is the whole argument for building the harness
        first.
        """
        if lines < 1:
            raise ValueError(f"lines must be at least 1, got {lines}")
        # One-way turnover of T means the weights moved by 2T in total --
        # something was sold to buy something else -- and both legs are
        # transactions that pay commission, spread and tax. Charging one leg
        # here while `cost` charges both would understate the drag by half,
        # in the one direction that flatters a trading strategy.
        traded = 2.0 * annual_turnover * self.account_value
        per_line = traded / lines
        # Half the traded value is bought and half sold, so the two sides are
        # averaged rather than one assumed.
        both = 0.5 * (self.instrument_cost("", per_line / 2, "buy")
                      + self.instrument_cost("", per_line / 2, "sell")) * 2
        return float(lines * both / self.account_value)


DEFAULT_COST_MODEL = CostModel()
