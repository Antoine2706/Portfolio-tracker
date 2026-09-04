"""Transactions view: the append-only ledger, and the form that feeds it.

Edit and delete are expressed as appends. Deleting a row writes a VOID
amendment naming it; the original stays on disk. That is what makes "why did
last month's value change" answerable.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal, InvalidOperation

import pandas as pd
import streamlit as st

from ...core.models import Transaction, TransactionType, ValidationError
from .. import state


def _add_form(instruments: dict) -> None:
    st.subheader("Record a transaction")
    if not instruments:
        st.info("Add an instrument first on the **Instruments** page. "
                "Transactions are keyed by ISIN.")
        return

    options = sorted(instruments, key=lambda i: instruments[i].name)
    with st.form("add_transaction", clear_on_submit=True):
        col1, col2 = st.columns(2)
        isin = col1.selectbox(
            "Instrument", options,
            format_func=lambda i: f"{instruments[i].name} ({i})")
        txn_type = col2.selectbox("Type", [t.value for t in TransactionType])
        col3, col4, col5 = st.columns(3)
        date = col3.date_input("Date", value=dt.date.today(), max_value=dt.date.today())
        quantity = col4.text_input("Quantity", value="0")
        price = col5.text_input("Price per unit", value="0")
        col6, col7 = st.columns(2)
        currency = col6.text_input(
            "Currency", value=instruments[isin].quote_currency or "EUR",
            help="The currency you actually paid in. Pence lines are entered as "
                 "GBp and converted to GBP automatically.")
        fees = col7.text_input("Fees", value="0")
        note = st.text_input("Note", value="")

        if st.form_submit_button("Append to ledger"):
            try:
                txn = Transaction(
                    date=date, isin=isin, type=TransactionType(txn_type),
                    quantity=Decimal(quantity or "0"),
                    price_per_unit=Decimal(price or "0"),
                    currency=currency.strip() or "EUR",
                    fees=Decimal(fees or "0"), note=note)
            except (ValidationError, InvalidOperation, ValueError) as exc:
                st.error(str(exc))
                return
            state.store().append_transaction(txn)
            st.success(f"Appended {txn.type.value} {txn.quantity} of "
                       f"{instruments[isin].name}")


def render() -> None:
    st.header("Transactions")
    store = state.store()
    instruments = store.load_instruments()
    _add_form(instruments)
    st.divider()

    live = store.load_transactions()
    everything = store.load_transactions(include_voided=True)
    voided_count = len(everything) - len(live)

    st.subheader("Ledger")
    if not live:
        st.info("The ledger is empty. Positions are derived from it, so nothing "
                "will appear in Holdings until a transaction is recorded.")
        return

    st.caption(f"{len(live)} live entries"
               + (f", {voided_count} voided and kept on disk" if voided_count else "")
               + ". The ledger is append-only: corrections are new rows, never "
                 "edits in place.")

    frame = pd.DataFrame([{
        "Date": t.date.isoformat(),
        "Instrument": instruments[t.isin].name if t.isin in instruments else t.isin,
        "Type": t.type.value,
        "Qty": f"{float(t.quantity):,.4g}",
        "Price": f"{float(t.price_per_unit):,.4f}",
        "Ccy": t.currency,
        "Fees": f"{float(t.fees):,.2f}",
        "Note": t.note,
        "id": t.id,
    } for t in sorted(live, key=lambda x: (x.date, x.id), reverse=True)])
    st.dataframe(frame, hide_index=True, width="stretch")

    st.subheader("Correct an entry")
    st.caption("A correction appends a void naming the original row, then a "
               "replacement. Nothing is overwritten.")
    labels = {t.id: (f"{t.date} {t.type.value} {float(t.quantity):g} "
                     f"{instruments[t.isin].name if t.isin in instruments else t.isin}")
              for t in live}
    target = st.selectbox("Entry", sorted(labels, key=lambda i: labels[i]),
                          format_func=lambda i: labels[i])
    reason = st.text_input("Reason", placeholder="quantity mistyped")
    if st.button("Void this entry"):
        store.void_transaction(target, reason=reason)
        st.success("Voided. The original row remains on disk for the audit trail.")
