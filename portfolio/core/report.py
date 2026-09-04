"""Display-ready structures and their plain-language explanations.

Why this is in `core/` and not in `app/`
----------------------------------------
The rule is that a view renders and computes nothing. That rule is unenforceable
if assembling a table row -- deciding a weight, a percentage, a divergence --
happens inside the view, because then every view is quietly doing arithmetic
that nothing tests.

So the projection from Position and risk figures into rows lives here, pure and
tested, and the views do string formatting and layout only. The AST guard stops
`core` importing UI; nothing stops maths leaking into `app`, so the defence is
to leave nothing in `app` for it to leak into.

The plain sentences are here for the same reason. They are domain explanations
of what a number means, not presentation, and they should change under test if
the meaning changes.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal

import pandas as pd

from .models import Instrument
from .money import BASE_CURRENCY, FxRates, Money
from .positions import Position
from .returns import AlignmentReport
from .risk import (BROAD_EUROPEAN_EQUITY_VOLATILITY, ConcentrationStats,
                   CorrelationPair, DrawdownStats, RiskDecomposition,
                   annualise_volatility)

__all__ = [
    "HoldingRow", "HoldingsTable", "DivergenceRow", "Metric", "RiskReport",
    "holdings_table", "divergence_rows", "risk_metrics", "correlation_sentences",
    "volatility_context",
]


# --------------------------------------------------------------------------
# Holdings
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HoldingRow:
    """One row of the Holdings table, with its own warnings attached.

    `warnings` belongs on the row rather than in a page-level panel: a stale
    price is a fact about this holding, and a sidebar nobody reads is the same
    as no warning at all.
    """
    isin: str
    name: str
    quantity: Decimal
    average_cost: Money
    price: Money | None
    price_as_of: str
    market_value: Money | None
    unrealised: Money | None
    unrealised_pct: Decimal | None
    weight: Decimal | None
    fx_note: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def has_warning(self) -> bool:
        return bool(self.warnings)


@dataclasses.dataclass(frozen=True)
class HoldingsTable:
    rows: tuple[HoldingRow, ...]
    total_value: Money
    total_unrealised: Money
    unpriced: tuple[str, ...]
    watchlist: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.rows


def holdings_table(positions: dict[str, Position],
                   instruments: dict[str, Instrument],
                   rates: FxRates | None = None,
                   as_of: dt.date | None = None,
                   base: str = BASE_CURRENCY) -> HoldingsTable:
    """Project positions into display rows, warnings included.

    An unpriced holding is listed in `unpriced` and excluded from the total,
    never valued at zero: a total that silently omits a holding is wrong in a
    way nobody notices.
    """
    as_of = as_of or dt.date.today()
    rows: list[HoldingRow] = []
    total = Money.zero(base)
    total_pnl = Money.zero(base)
    unpriced: list[str] = []
    watchlist: list[str] = []

    valued: dict[str, Money] = {}
    for isin, pos in positions.items():
        if pos.is_watchlist:
            watchlist.append(isin)
            continue
        if not pos.is_open:
            continue
        if not pos.is_derivable:
            # Quantity is right, the money is not. Listing it as unpriced keeps
            # it visible and out of the total, which is the honest outcome.
            unpriced.append(isin)
            continue
        mv = pos.market_value(rates, as_of)
        if mv is None:
            unpriced.append(isin)
        else:
            valued[isin] = mv
            total = total + mv

    for isin, pos in positions.items():
        if not pos.is_open:
            continue
        inst = instruments.get(isin)
        mv = valued.get(isin)
        pnl = (mv - pos.cost_basis) if mv is not None else None
        pct = (pnl.amount / pos.cost_basis.amount
               if pnl is not None and pos.cost_basis.amount != 0 else None)
        weight = (mv.amount / total.amount
                  if mv is not None and total.amount != 0 else None)
        if pnl is not None:
            total_pnl = total_pnl + pnl

        quote = pos.quote
        warnings: list[str] = []
        as_of_text = "no price"
        fx_note = None

        if not pos.is_derivable:
            warnings.append(
                f"cannot be valued: {pos.error}. The quantity is correct; the "
                f"cost and value are not shown rather than shown wrong.")
        elif quote is None:
            warnings.append("no price returned; this holding is excluded from the total")
        else:
            as_of_text = (quote.as_of.isoformat()
                          if not isinstance(quote.as_of, dt.datetime)
                          else quote.as_of.isoformat(timespec="minutes"))
            stale = quote.staleness_note(as_of)
            if stale:
                warnings.append(stale)
            if quote.is_stale:
                warnings.append("last known close, not a live quote")
            if quote.delay_minutes:
                as_of_text += f" (delayed ~{quote.delay_minutes}m)"
            elif not quote.is_stale:
                as_of_text += " (delay unstated)"
            if mv is not None and quote.price.currency != base and rates is not None:
                rate = rates.rate(quote.price.currency, base, as_of)
                fx_note = (f"{quote.price.currency}->{base} at "
                           f"{rate:.4f} on {as_of.isoformat()}")

        rows.append(HoldingRow(
            isin=isin, name=inst.name if inst else isin,
            quantity=pos.quantity, average_cost=pos.average_cost,
            price=quote.price if quote else None, price_as_of=as_of_text,
            market_value=mv, unrealised=pnl, unrealised_pct=pct, weight=weight,
            fx_note=fx_note, warnings=tuple(warnings),
        ))

    rows.sort(key=lambda r: (r.market_value.amount if r.market_value else Decimal("-1")),
              reverse=True)
    return HoldingsTable(tuple(rows), total, total_pnl, tuple(unpriced), tuple(watchlist))


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DivergenceRow:
    """The headline of the whole application.

    The gap between an instrument's share of the money and its share of the
    risk is the number no broker app shows. Everything else in this report is
    supporting evidence for this column.
    """
    isin: str
    name: str
    weight: float
    risk_share: float
    divergence: float

    @property
    def divergence_pp(self) -> float:
        """Divergence in percentage points.

        The difference of two percentages is measured in percentage points, not
        percent, and the unit lets the column show an explicit sign. Here rather
        than in the view because it is a unit conversion on a reported figure.
        """
        return self.divergence * 100.0

    @property
    def direction(self) -> str:
        if self.divergence > 0:
            return "carries more risk than capital"
        if self.divergence < 0:
            return "carries less risk than capital"
        return "risk matches capital"

    def sentence(self) -> str:
        return (f"{self.name} is {self.weight:.1%} of your money but "
                f"{self.risk_share:.1%} of your risk "
                f"({self.divergence:+.1%}).")


def divergence_rows(decomposition: RiskDecomposition,
                    instruments: dict[str, Instrument]) -> list[DivergenceRow]:
    """Weight against risk share, sorted by absolute divergence descending.

    Sorted by magnitude rather than by weight so the holding most mispriced
    against intuition is the first thing on screen, whichever direction it errs.
    """
    frame = decomposition.as_frame()
    rows = [
        DivergenceRow(
            isin=str(isin),
            name=instruments[isin].name if isin in instruments else str(isin),
            weight=float(row["weight"]),
            risk_share=float(row["pct_of_risk"]),
            divergence=float(row["pct_of_risk"] - row["weight"]),
        )
        for isin, row in frame.iterrows()
    ]
    return sorted(rows, key=lambda r: abs(r.divergence), reverse=True)


@dataclasses.dataclass(frozen=True)
class Metric:
    """A number with the sentence that says what it means.

    The sentence is always shown, never a tooltip: a figure the reader cannot
    interpret is not information.
    """
    label: str
    value: str
    sentence: str
    warning: str | None = None


def risk_metrics(decomposition: RiskDecomposition,
                 diversification: float,
                 concentration_stats: ConcentrationStats,
                 drawdown_stats: DrawdownStats | None = None,
                 beta_value: float | None = None,
                 benchmark_name: str | None = None,
                 trading_days: int = 252) -> list[Metric]:
    """Headline metrics, each with a plain-language explanation."""
    annual = annualise_volatility(decomposition.portfolio_volatility, trading_days)
    # A volatility figure means little without something to measure it against.
    # A broad European equity index sits near 15%; most thematic ETFs here run
    # two to three times that, and that is what the concentration costs.
    multiple = annual / BROAD_EUROPEAN_EQUITY_VOLATILITY
    comparison = (f" That is about {multiple:.1f} times a broad European equity "
                  f"index, which typically sits near "
                  f"{BROAD_EUROPEAN_EQUITY_VOLATILITY:.0%}."
                  if multiple >= 1.15 else
                  f" A broad European equity index typically sits near "
                  f"{BROAD_EUROPEAN_EQUITY_VOLATILITY:.0%}.")
    out = [
        Metric("Annualised volatility", f"{annual:.2%}",
               f"In a typical year this portfolio moves about {annual:.1%} "
               f"either way." + comparison),
        Metric("Diversification ratio", f"{diversification:.2f}",
               f"Your holdings are about {diversification:.1f} times as "
               f"diversified as owning just one of them. 1.0 would mean they "
               f"all move together."),
        Metric("Effective holdings",
               f"{concentration_stats.effective_holdings:.1f} "
               f"of {concentration_stats.actual_holdings}",
               f"{concentration_stats.actual_holdings} positions behaving like "
               f"about {concentration_stats.effective_holdings:.0f} independent "
               f"ones. This is the measure that sees a cluster of holdings all "
               f"making the same bet; comparing pairs one at a time cannot."),
    ]
    if drawdown_stats is not None:
        out.append(Metric(
            "Maximum drawdown", f"{drawdown_stats.max_drawdown:.2%}",
            f"The worst peak-to-trough fall on record was "
            f"{abs(drawdown_stats.max_drawdown):.1%}. You are currently "
            f"{abs(drawdown_stats.current_drawdown):.1%} below the peak."))
    if beta_value is not None:
        name = benchmark_name or "the benchmark"
        out.append(Metric(
            f"Beta vs {name}" if benchmark_name else "Beta", f"{beta_value:.2f}",
            f"When {name} moves 1%, this portfolio has moved about "
            f"{beta_value:.2f}% in the same direction.",
            warning=None if benchmark_name else
            "no benchmark named; a beta without one is meaningless"))
    return out


def volatility_context(standalone: pd.Series,
                       instruments: dict[str, Instrument]) -> list[tuple[str, str, str]]:
    """Each holding's own annualised volatility, against a broad index.

    Returned as (name, value, comparison) so the view formats and does not
    compute. The contrast is the point: a 44% thematic ETF beside a 15%
    reference is what thematic concentration costs, stated plainly.
    """
    rows = []
    for isin, vol in standalone.items():
        name = instruments[isin].name if isin in instruments else str(isin)
        multiple = float(vol) / BROAD_EUROPEAN_EQUITY_VOLATILITY
        rows.append((name, f"{float(vol):.1%}", f"{multiple:.1f}x"))
    return rows


def correlation_sentences(pairs: list[CorrelationPair],
                          instruments: dict[str, Instrument]) -> list[str]:
    """High-correlation pairs as sentences naming both funds.

    A sentence beats a matrix cell the reader has to locate, which is why these
    appear before the heatmap rather than as an annotation on it.
    """
    names = {isin: inst.name for isin, inst in instruments.items()}
    return [pair.sentence(names) for pair in pairs]


@dataclasses.dataclass(frozen=True)
class RiskReport:
    """Everything the Risk view renders, assembled and ordered."""
    divergence: tuple[DivergenceRow, ...]
    alignment: AlignmentReport
    metrics: tuple[Metric, ...]
    correlation_notes: tuple[str, ...]
    correlation_matrix: pd.DataFrame
    excluded_note: str | None = None

    @property
    def headline(self) -> str:
        if not self.divergence:
            return "No risk decomposition available."
        top = self.divergence[0]
        return top.sentence()
