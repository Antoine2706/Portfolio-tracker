"""Instruments view: resolve, confirm, save. Never a text box that writes.

The add flow is the critical path where a silent error would enter the system
and never leave, so nothing is written until the user has seen the ISIN, name,
exchange, currency, history length and verdict together and confirmed.
"""

from __future__ import annotations

import streamlit as st

from ...core.models import AssetClass, ValidationError, is_valid_isin
from ...core.universe import check_deletable, deactivate, reactivate
from ...data.provider import FigiListing, ListingProbe
from ...data.resolve import (Resolution, Verdict, instrument_from_candidate,
                             resolve_from_listings)
from .. import state

def _probe_from_cache(symbol: str, lookback: int) -> ListingProbe:
    import datetime as dt
    raw = state.probe_symbol(symbol, lookback)
    return ListingProbe(
        symbol=raw["symbol"], ok=raw["ok"], exchange=raw["exchange"],
        currency=raw["currency"], name=raw["name"],
        observations=raw["observations"],
        first_date=dt.date.fromisoformat(raw["first_date"]) if raw["first_date"] else None,
        last_date=dt.date.fromisoformat(raw["last_date"]) if raw["last_date"] else None,
        adjusted=raw["adjusted"], error=raw["error"])


def _resolve_cached(isin: str, lookback: int) -> Resolution:
    """Run the real pipeline over cached inputs.

    The filtering, gating and ranking stay in `data.resolve`; this supplies
    cached listings and a cached probe. Reimplementing any of that here would
    be the logic leak the layering rule exists to prevent -- and an uncached
    rerun would be one OpenFIGI call plus one probe per candidate per keystroke.
    """
    listings = [FigiListing(**row) for row in state.fetch_figi_listings(isin)]
    return resolve_from_listings(
        isin, listings, lambda symbol: _probe_from_cache(symbol, lookback),
        lookback=lookback)


def _add_form() -> None:
    st.subheader("Add an instrument")
    st.caption("ISIN is the reliable path: the same fund trades under different "
               "tickers on different exchanges, and a ticker alone can resolve "
               "to a different fund entirely.")
    isin = st.text_input("ISIN", placeholder="IE0002Y8CX98", key="add_isin").strip().upper()
    if not isin:
        return
    if not is_valid_isin(isin):
        st.error(f"`{isin}` is not a valid ISIN. Twelve characters with a "
                 f"correct check digit. A typo here would create a second, "
                 f"empty instrument rather than failing loudly.")
        return

    store = state.store()
    existing = store.load_instruments()
    if isin in existing:
        st.info(f"Already in your universe as **{existing[isin].name}**.")
        return

    if not st.button("Look up this ISIN", key="do_resolve"):
        return

    with st.spinner("Resolving via OpenFIGI, then probing each listing..."):
        try:
            resolution = _resolve_cached(isin, 252)
        except Exception as exc:
            st.error(f"Lookup failed: {exc}")
            return

    usable, refused = resolution.candidates, resolution.refused
    st.caption(resolution.summary())

    if not usable and not refused:
        st.error(resolution.block_reason() or "No listing on an allowlisted "
                                              "European venue.")
        return

    for candidate in refused:
        st.error(f"**{candidate.yahoo_symbol}** refused - "
                 + " ".join(candidate.reasons))

    selectable = [c for c in usable if c.verdict is not Verdict.FAILED]
    if not selectable:
        st.error(resolution.block_reason() or "Every listing failed to return data.")
        return
    if resolution.blocked:
        st.warning(resolution.block_reason())

    st.markdown("**Choose the listing to use.** Check the ISIN and exchange, not "
                "just the name: two different funds can share a ticker, an issuer "
                "and a theme.")
    default = next((i for i, c in enumerate(selectable) if c.selectable_as_primary), None)
    choice = st.radio(
        "Listing", options=list(range(len(selectable))), index=default,
        format_func=lambda i: selectable[i].describe(), key="candidate_choice")

    chosen = selectable[choice]
    for reason in chosen.reasons:
        st.warning(reason)
    if not chosen.selectable_as_primary:
        st.warning(f"**{chosen.verdict.value}** listings are never chosen for you. "
                   f"Selecting this one is a deliberate choice.")

    with st.form("confirm_instrument"):
        st.write("**Confirm before saving**")
        name = st.text_input("Name", value=chosen.name or "")
        issuer = st.text_input("Issuer", value="")
        col1, col2 = st.columns(2)
        asset_class = col1.selectbox("Asset class", [a.value for a in AssetClass],
                                     help="ETC is a collateralised note, not a "
                                          "UCITS fund, and carries issuer credit risk.")
        base_currency = col2.text_input(
            "Base currency", value="EUR",
            help="The fund's own currency, which is often not the listing's. "
                 "ISAE.AS quotes EUR for a USD-base fund, so this cannot be "
                 "inferred from the listing and must be entered.")
        if st.form_submit_button("Save instrument"):
            try:
                inst = instrument_from_candidate(
                    chosen, "yfinance", base_currency=base_currency.strip().upper(),
                    name=name, issuer=issuer, asset_class=asset_class)
            except ValidationError as exc:
                st.error(str(exc))
                return
            existing[inst.isin] = inst
            store.save_instruments(existing)
            st.success(f"Saved **{inst.name}** as {inst.provider_symbols['yfinance']}")


def _manage(instruments: dict) -> None:
    st.subheader("Your instrument universe")
    if not instruments:
        st.info("No instruments yet. Add one above.")
        return

    store = state.store()
    transactions = store.load_transactions()

    for isin, inst in sorted(instruments.items(), key=lambda kv: kv[1].name):
        status = "" if inst.active else " - deactivated"
        with st.expander(f"{inst.name} ({isin}){status}"):
            st.write(f"{inst.asset_class.value} - {inst.issuer or 'issuer unknown'} - "
                     f"base {inst.base_currency}, quoted {inst.quote_currency or '?'} "
                     f"on {inst.exchange or '?'}")
            if inst.note:
                st.caption(inst.note)

            st.write("**Provider symbols**")
            for provider, symbol in sorted(inst.provider_symbols.items()):
                field = f"provider_symbols.{provider}"
                marked = " *(set by hand)*" if inst.is_overridden(field) else ""
                st.write(f"- `{provider}`: `{symbol}`{marked}")

            with st.form(f"override_{isin}"):
                st.caption("Override a provider symbol by hand. Overridden fields "
                           "are marked and re-resolution will not overwrite them.")
                col1, col2 = st.columns(2)
                provider = col1.text_input("Provider", value="yfinance", key=f"p_{isin}")
                symbol = col2.text_input(
                    "Symbol", value=inst.provider_symbols.get("yfinance", ""),
                    key=f"s_{isin}")
                if st.form_submit_button("Save override"):
                    inst.override(f"provider_symbols.{provider.strip()}", symbol.strip())
                    store.save_instruments(instruments)
                    st.success(f"`{provider}` set to `{symbol}` and marked as manual.")

            check = check_deletable(isin, transactions)
            col1, col2 = st.columns(2)
            if inst.active:
                if col1.button("Deactivate", key=f"deact_{isin}"):
                    deactivate(inst, "hidden from holdings")
                    store.save_instruments(instruments)
                    st.success("Deactivated. It stays in the ledger and can be "
                               "reactivated at any time.")
            else:
                if col1.button("Reactivate", key=f"react_{isin}"):
                    reactivate(inst)
                    store.save_instruments(instruments)
                    st.success("Reactivated.")

            if check.allowed:
                col2.caption(f"{check.transaction_count} transactions reference this.")
                if col2.button("Delete permanently", key=f"del_{isin}"):
                    del instruments[isin]
                    store.save_instruments(instruments)
                    st.success("Deleted.")
            else:
                col2.caption(f"Cannot delete: {check.reason} {check.alternative}")


def render() -> None:
    st.header("Instruments")
    _add_form()
    st.divider()
    _manage(state.store().load_instruments())
