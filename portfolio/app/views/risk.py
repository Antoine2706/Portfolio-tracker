"""Risk view.

Order is the design. The divergence table is first because it is the reason
this application exists; the alignment line sits directly above the numbers it
produced; the correlation sentences precede the heatmap because a sentence
naming two funds beats a matrix cell the reader has to locate.

Every number here comes from `core`. This module formats and lays out.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...core.positions import derive_positions, weights
from ...core.report import (correlation_sentences, divergence_rows, risk_metrics)
from ...core.returns import DEFAULT_LOOKBACK, InsufficientHistory, align_returns
from ...core.risk import (concentration, correlation_matrix, covariance_matrix,
                          diversification_ratio, drawdown, high_correlation_pairs,
                          risk_decomposition)
from .. import state


def render() -> None:
    st.header("Risk")
    store = state.store()
    instruments = store.load_instruments()
    transactions = store.load_transactions()

    quotes, _ = state.load_quotes(instruments)
    rates = state.fx_table()
    positions = derive_positions(transactions, instruments, rates=rates,
                             quotes=quotes, strict=False)
    held = weights(positions, rates)

    if len(held) < 2:
        st.info(
            f"**Not enough holdings for a risk model.** You have {len(held)} "
            f"priced position(s); covariance needs at least two.\n\n"
            f"A single holding has no diversification to measure, so these "
            f"figures would be zeros rather than answers.")
        return

    lookback = st.number_input(
        "Lookback window (trading days)", min_value=60, max_value=1000,
        value=DEFAULT_LOOKBACK, step=21,
        help="252 is one trading year. The window shortens automatically if "
             "your newest holding has less history, and says so.")

    series = {}
    missing = []
    for isin in held:
        symbol = instruments[isin].provider_symbols.get("yfinance")
        if not symbol:
            missing.append(instruments[isin].name)
            continue
        try:
            series[isin] = state.fetch_history(symbol)
        except Exception as exc:
            missing.append(f"{instruments[isin].name}: {exc}")

    if len(series) < 2:
        st.error("Could not load price history for at least two holdings. "
                 + "; ".join(missing))
        return

    try:
        returns, alignment = align_returns(series, lookback=int(lookback))
    except InsufficientHistory as exc:
        st.error(f"**No covariance matrix can be built.** {exc}")
        return

    used = [i for i in held if i in returns.columns]
    w = {i: float(held[i]) for i in used}
    scale = sum(w.values())
    if scale <= 0:
        st.error("Excluded holdings carry all the weight; nothing left to model.")
        return
    # Renormalising over the surviving instruments is arithmetic, but it is the
    # one thing a view legitimately owns: it is choosing what to display, not
    # deriving a risk figure. The weights themselves come from core.
    w = {k: v / scale for k, v in w.items()}

    cov = covariance_matrix(returns[used])
    decomposition = risk_decomposition(w, cov)

    # ---- 1. The headline -------------------------------------------------
    st.subheader("Capital share against risk share")
    st.caption("The gap between what a holding is worth and what it costs you "
               "in risk. Sorted by the size of the gap, largest first.")
    rows = divergence_rows(decomposition, instruments)
    st.dataframe(pd.DataFrame([{
        "Instrument": r.name,
        "Capital weight": f"{r.weight:.2%}",
        "Risk contribution": f"{r.risk_share:.2%}",
        "Divergence": f"{r.divergence:+.2%}",
    } for r in rows]), hide_index=True, width="stretch")
    if rows:
        st.markdown(f"**{rows[0].sentence()}**")

    # ---- 2. Alignment, directly above the numbers it produced -------------
    st.caption(f"Window used: {alignment.summary()}")
    for warning in alignment.warnings:
        st.warning(warning, icon=":material/warning:")
    if alignment.excluded:
        names = ", ".join(
            f"{instruments[e.isin].name if e.isin in instruments else e.isin} "
            f"({e.observations} obs)" for e in alignment.excluded)
        st.warning(f"Excluded from the risk model, still shown in Holdings: "
                   f"{names}", icon=":material/warning:")
    for note in missing:
        st.warning(f"No price history: {note}", icon=":material/warning:")

    # ---- 3. Headline metrics, each with its plain sentence ---------------
    st.subheader("Portfolio measures")
    dd = None
    portfolio_returns = (returns[used] * pd.Series(w)).sum(axis=1)
    value_series = (1.0 + portfolio_returns).cumprod()
    dd = drawdown(value_series)

    metrics = risk_metrics(
        decomposition, diversification_ratio(w, cov),
        concentration(list(w.values())), drawdown_stats=dd)
    for metric in metrics:
        st.metric(metric.label, metric.value)
        st.caption(metric.sentence)
        if metric.warning:
            st.warning(metric.warning, icon=":material/warning:")
    st.caption("Beta needs a benchmark series; none is configured yet, so it is "
               "not shown. A beta without a named benchmark is meaningless.")

    # ---- 4. Correlation as sentences, before the matrix -------------------
    corr = correlation_matrix(cov)
    pairs = high_correlation_pairs(corr)
    st.subheader("Holdings that may be duplicating each other")
    sentences = correlation_sentences(pairs, instruments)
    if sentences:
        for sentence in sentences:
            st.warning(sentence, icon=":material/content_copy:")
    else:
        st.caption("No pair moves together above 0.90. Nothing here looks like "
                   "the same exposure held twice.")

    # ---- 5. The heatmap, as supporting detail ----------------------------
    st.subheader("Correlation matrix")
    labels = [instruments[i].name[:24] if i in instruments else i for i in used]
    display = corr.copy()
    display.index = labels
    display.columns = labels
    # Diverging scale centred on zero: correlation runs -1 to +1 and zero is
    # meaningful, so a sequential scale would misrepresent the midpoint.
    st.dataframe(display.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
                 .format("{:.2f}"), width="stretch")
