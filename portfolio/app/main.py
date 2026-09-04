"""Streamlit entry point.

    streamlit run portfolio/app/main.py

Deliberately plain. No custom theming, no login, no charts beyond the weight
bars and the correlation heatmap. Every number comes from `core`; this layer
reads and renders.
"""

from __future__ import annotations

import pathlib
import sys

import streamlit as st

if __package__ in (None, ""):                       # `streamlit run path/to/main.py`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from portfolio.app import state                      # noqa: E402
from portfolio.app.views import (holdings, instruments,  # noqa: E402
                                 risk, transactions)
from portfolio.data.store import DataMode            # noqa: E402

VIEWS = {
    "Holdings": holdings.render,
    "Risk": risk.render,
    "Instruments": instruments.render,
    "Transactions": transactions.render,
}


def main() -> None:
    st.set_page_config(page_title="Portfolio risk tracker", layout="wide")

    with st.sidebar:
        st.title("Portfolio")
        page = st.radio("View", list(VIEWS), label_visibility="collapsed")

        st.divider()
        st.caption("Data source")
        mode = st.radio(
            "Mode", [DataMode.SEED, DataMode.USER],
            format_func=lambda m: "Demo (seed data)" if m is DataMode.SEED
            else "My data",
            index=0 if state.current_mode() is DataMode.SEED else 1,
            label_visibility="collapsed")
        state.set_mode(mode)

        st.divider()
        # Prices are fetched on load and cached with a TTL. Nothing polls: free
        # tier rate limits make polling fatal, and this button is the only path
        # that refetches.
        if st.button("Refresh prices", width="stretch"):
            state.refresh_all()
        st.caption(state.last_refresh_text())

    state.mode_banner()
    VIEWS[page]()


main()
