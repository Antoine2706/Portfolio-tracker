"""Holdings view.

Layout follows the hierarchy real financial products use: the primary position
first and largest, then anything that qualifies it, then one chart, then the
dense table, then detail. Renders `core.report.holdings_table` and computes
nothing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...core.positions import derive_positions
from ...core.report import holdings_table
from .. import charts, state


def _num(value) -> float | None:
    """Decimal to float for st.column_config, which formats and right-aligns.

    Numbers are passed as numbers, never pre-formatted strings: a string column
    left-aligns and each row picks its own precision, which is what makes a
    table of figures unreadable.
    """
    return None if value is None else float(value)


def render() -> None:
    store = state.store()
    instruments = store.load_instruments()
    transactions = store.load_transactions()

    if not transactions:
        st.subheader("Holdings")
        st.info(
            "No transactions yet. Positions are derived from the transaction "
            "log, so there is nothing to show until something is bought. "
            "Record one on the Transactions page, or add an instrument to the "
            "watchlist on the Instruments page first.")
        if instruments:
            st.caption(f"{len(instruments)} instrument(s) on the watchlist, none held.")
        return

    quotes, failures = state.load_quotes(instruments)
    rates = state.fx_table()
    positions = derive_positions(transactions, instruments, rates=rates,
                                 quotes=quotes, strict=False)
    table = holdings_table(positions, instruments, rates=rates)

    if table.is_empty:
        st.subheader("Holdings")
        st.info("Every position is closed. The ledger has transactions but "
                "nothing is currently held.")
        return

    # ---- 1. Primary position, largest thing on the page ------------------
    # Colour means state: green is gain, red is loss. A zero P&L is neither, so
    # the delta is rendered neutral rather than picking up the default green.
    unrealised = table.total_unrealised.amount
    st.metric("Portfolio value",
              f"{table.total_value.amount:,.2f} EUR",
              delta=f"{unrealised:+,.2f} EUR unrealised",
              delta_color="normal" if unrealised != 0 else "off")
    st.caption(state.last_refresh_text() + " - prices are delayed, never real time")

    # ---- 2. Exceptions, immediately after ---------------------------------
    exceptions: list[str] = []
    for row in table.rows:
        for warning in row.warnings:
            exceptions.append(f"**{row.name}** - {warning}")
    if table.unpriced:
        names = ", ".join(instruments[i].name if i in instruments else i
                          for i in table.unpriced)
        exceptions.append(
            f"Excluded from the total because no price was returned: {names}. "
            f"The total above is therefore incomplete.")
    if exceptions or failures:
        with st.container(border=True):
            st.markdown("**Needs attention**")
            for note in exceptions:
                st.warning(note)
            # Fetch failures are collapsed into one line. Ten near-identical
            # warnings is not ten pieces of information, and an exception panel
            # long enough to fill the page stops being read at all.
            if failures:
                st.warning(f"{len(failures)} price fetch(es) failed. Values "
                           f"below are incomplete.")
                with st.expander(f"Show the {len(failures)} failures"):
                    for failure in failures:
                        st.caption(failure)

    # ---- 3. One chart ------------------------------------------------------
    st.subheader("Weight by holding")
    weights = [(r.name, float(r.weight)) for r in table.rows if r.weight is not None]
    if weights:
        st.altair_chart(charts.weight_bars(weights), use_container_width=True)

    # ---- 4. The dense table -----------------------------------------------
    st.subheader("Positions")
    frame = pd.DataFrame([{
        "Instrument": r.name,
        "ISIN": r.isin,
        "Quantity": _num(r.quantity),
        "Avg cost": _num(r.average_cost.amount),
        "Price": _num(r.price.amount) if r.price else None,
        "As of": r.price_as_of,
        "Value": _num(r.market_value.amount) if r.market_value else None,
        "P&L": _num(r.unrealised.amount) if r.unrealised else None,
        "P&L %": _num(r.unrealised_pct),
        "Weight": _num(r.weight),
    } for r in table.rows])

    st.dataframe(
        frame, hide_index=True, width="stretch",
        column_config={
            "Instrument": st.column_config.TextColumn(width="large"),
            "ISIN": st.column_config.TextColumn(width="small"),
            # Fixed precision per column, and numeric so the digits right-align.
            "Quantity": st.column_config.NumberColumn(format="%.2f"),
            "Avg cost": st.column_config.NumberColumn(format="%.2f"),
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "P&L": st.column_config.NumberColumn(format="%+.2f"),
            "P&L %": st.column_config.NumberColumn(format="percent"),
            "Weight": st.column_config.NumberColumn(format="percent"),
        })

    # ---- 5. Detail ---------------------------------------------------------
    fx_notes = [(r.name, r.fx_note) for r in table.rows if r.fx_note]
    if fx_notes:
        with st.container(border=True):
            st.markdown("**Currency conversions applied**")
            for name, note in fx_notes:
                st.caption(f"{name}: converted {note}")

    if table.watchlist:
        names = ", ".join(instruments[i].name if i in instruments else i
                          for i in table.watchlist)
        st.caption(f"Watchlist, not held: {names}")
