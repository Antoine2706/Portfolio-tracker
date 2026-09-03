"""Holdings view. Renders `core.report.holdings_table` and computes nothing."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...core.positions import derive_positions
from ...core.report import HoldingsTable, holdings_table
from .. import state


def _fmt_money(m) -> str:
    return "-" if m is None else f"{m.amount:,.2f}"


def _fmt_pct(v) -> str:
    return "-" if v is None else f"{float(v):.2%}"


def render() -> None:
    st.header("Holdings")
    store = state.store()
    instruments = store.load_instruments()
    transactions = store.load_transactions()

    if not transactions:
        st.info(
            "**No transactions yet.** Positions are derived from the transaction "
            "log, so there is nothing to show until something is bought.\n\n"
            "Add one on the **Transactions** page, or add an instrument to your "
            "watchlist on the **Instruments** page first.")
        if instruments:
            st.caption(f"{len(instruments)} instrument(s) on the watchlist, "
                       f"none held.")
        return

    quotes, failures = state.load_quotes(instruments)
    rates = state.fx_table()
    positions = derive_positions(transactions, instruments, rates=rates,
                             quotes=quotes, strict=False)
    table = holdings_table(positions, instruments, rates=rates)

    if table.is_empty:
        st.info("**Every position is closed.** The ledger has transactions but "
                "nothing is currently held.")
        return

    # One global provenance line; per-row stamps appear in the table where they
    # differ, and warnings sit on the row they concern.
    st.caption(state.last_refresh_text() +
               " - prices are delayed, never real time")

    frame = pd.DataFrame([{
        "Instrument": r.name,
        "ISIN": r.isin,
        "Qty": f"{float(r.quantity):,.4g}",
        "Avg cost": _fmt_money(r.average_cost),
        "Price": _fmt_money(r.price),
        "As of": r.price_as_of,
        "Value EUR": _fmt_money(r.market_value),
        "P&L": _fmt_money(r.unrealised),
        "P&L %": _fmt_pct(r.unrealised_pct),
        "Weight": _fmt_pct(r.weight),
        "": "!" if r.has_warning else "",
    } for r in table.rows])
    st.dataframe(frame, hide_index=True, width="stretch")

    st.markdown(
        f"**Total {table.total_value.amount:,.2f} EUR** &nbsp;&nbsp; "
        f"unrealised {table.total_unrealised.amount:+,.2f} EUR")

    # Warnings adjacent to the rows they affect, not in a sidebar panel.
    for row in table.rows:
        for warning in row.warnings:
            st.warning(f"**{row.name}** - {warning}", icon=":material/warning:")
        if row.fx_note:
            st.caption(f"{row.name}: converted {row.fx_note}")

    if table.unpriced:
        names = ", ".join(instruments[i].name if i in instruments else i
                          for i in table.unpriced)
        st.warning(f"Excluded from the total because no price was returned: "
                   f"{names}. The total above is therefore incomplete.",
                   icon=":material/warning:")
    for failure in failures:
        st.warning(f"Price fetch failed - {failure}", icon=":material/warning:")

    st.subheader("Weight by holding")
    st.caption("Bars rather than a pie: comparable across ten holdings and "
               "readable at a glance.")
    chart = pd.DataFrame(
        {"weight": [float(r.weight) for r in table.rows if r.weight is not None]},
        index=[r.name for r in table.rows if r.weight is not None])
    st.bar_chart(chart, horizontal=True)

    if table.watchlist:
        names = ", ".join(instruments[i].name if i in instruments else i
                          for i in table.watchlist)
        st.caption(f"Watchlist (no transactions, not held): {names}")
