"""Risk view.

The product is the gap between an instrument's share of the money and its share
of the risk. Everything else on this page is supporting evidence, and the
layout says so: the headline first, then anything that qualifies it, then the
divergence chart, then the table, then the measures, then the correlation grid
last as detail.

Every number comes from `core`. This module formats and lays out.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...core.positions import derive_positions, weights
from ...core.report import (correlation_sentences, divergence_rows, risk_metrics,
                            volatility_context)
from ...core.returns import (DEFAULT_LOOKBACK, InsufficientHistory,
                             align_returns, simple_returns)
from ...core.risk import (HIGH_CORRELATION_THRESHOLD, beta, concentration,
                          correlation_matrix, covariance_matrix,
                          diversification_ratio, drawdown, high_correlation_pairs,
                          normalise_weights, portfolio_return_series,
                          portfolio_value_series, risk_decomposition,
                          standalone_volatilities)
from ...data.benchmarks import BENCHMARKS, DEFAULT_BENCHMARK
from .. import charts, notices, state


def render() -> None:
    store = state.store()
    instruments = store.load_instruments()
    transactions = store.load_transactions()

    quotes, _ = state.load_quotes(instruments)
    rates = state.fx_table()
    positions = derive_positions(transactions, instruments, rates=rates,
                                 quotes=quotes, strict=False)
    held = weights(positions, rates)

    if len(held) < 2:
        st.subheader("Risk")
        st.info(
            f"Not enough holdings for a risk model. You have {len(held)} priced "
            f"position(s); covariance needs at least two. A single holding has "
            f"no diversification to measure, so these figures would be zeros "
            f"rather than answers.")
        return

    series, missing = {}, []
    for isin in held:
        symbol = instruments[isin].provider_symbols.get("yfinance")
        if not symbol:
            missing.append(f"{instruments[isin].name}: no symbol stored")
            continue
        try:
            series[isin] = state.fetch_history(symbol)
        except Exception as exc:
            missing.append(f"{instruments[isin].name}: {exc}")

    if len(series) < 2:
        st.subheader("Risk")
        st.error("Could not load price history for at least two holdings. "
                 + "; ".join(missing))
        return

    try:
        returns, alignment = align_returns(series, lookback=DEFAULT_LOOKBACK)
    except InsufficientHistory as exc:
        st.subheader("Risk")
        st.error(f"No covariance matrix can be built. {exc}")
        return

    used = [i for i in returns.columns if i in held]
    w = normalise_weights({i: float(held[i]) for i in held}, used)
    if not w:
        st.error("Excluded holdings carry all the weight; nothing left to model.")
        return

    cov = covariance_matrix(returns[used])
    decomposition = risk_decomposition(w, cov)
    rows = divergence_rows(decomposition, instruments)
    stats = concentration(list(w.values()))

    # ---- 1. The headline, largest on the page -----------------------------
    if rows:
        # No delta: st.metric always draws a direction arrow beside one, and an
        # arrow next to a fund name means nothing. The name is in the sentence
        # below, where it reads as language rather than as a metric.
        st.metric("Largest gap between capital and risk",
                  f"{rows[0].divergence:+.1%}")
        st.markdown(f"**{rows[0].sentence()}**")
    st.caption(f"Window used: {alignment.summary()}")

    # ---- 2. Exceptions, immediately after ----------------------------------
    corr = correlation_matrix(cov)
    pairs = high_correlation_pairs(corr)
    sentences = correlation_sentences(pairs, instruments)

    exceptions = list(alignment.warnings)
    if alignment.excluded:
        names = ", ".join(
            f"{instruments[e.isin].name if e.isin in instruments else e.isin} "
            f"({e.observations} obs)" for e in alignment.excluded)
        exceptions.append(f"Excluded from the risk model, still shown in "
                          f"Holdings: {names}")
    exceptions += [f"No price history: {note}" for note in missing]
    exceptions += sentences

    if exceptions:
        n = len(exceptions)
        notices.notices(
            f"{n} thing{'s' if n != 1 else ''} qualif{'y' if n != 1 else 'ies'} "
            f"the figures below.", exceptions)

    # ---- 3. One chart: the divergence -------------------------------------
    st.subheader("Capital share against risk share")
    st.caption("How far each holding's share of the risk sits from its share "
               "of the money. Bars at zero are behaving as expected.")
    st.altair_chart(
        charts.divergence_bars([(r.name, r.divergence) for r in rows]),
        use_container_width=True)

    # ---- 4. The dense table ------------------------------------------------
    st.dataframe(
        pd.DataFrame([{
            "Instrument": r.name,
            "Capital weight": r.weight,
            "Risk contribution": r.risk_share,
            "Divergence (pp)": r.divergence_pp,
        } for r in rows]),
        hide_index=True, width="stretch",
        column_config={
            "Instrument": st.column_config.TextColumn(width="large"),
            "Capital weight": st.column_config.NumberColumn(format="percent"),
            "Risk contribution": st.column_config.NumberColumn(format="percent"),
            # Percentage points, signed: this column is the product, so the
            # direction should be readable without comparing two other columns.
            "Divergence (pp)": st.column_config.NumberColumn(format="%+.2f"),
        })

    # ---- 5. Measures -------------------------------------------------------
    st.subheader("Portfolio measures")
    benchmark = st.selectbox(
        "Benchmark for beta", BENCHMARKS,
        index=BENCHMARKS.index(DEFAULT_BENCHMARK),
        format_func=lambda b: f"{b.index} - {b.name} ({b.symbol})",
        help="Named on screen because a beta without a stated benchmark is "
             "meaningless. All options are EUR-quoted, so the beta is not "
             "measuring a currency as well as an index.")
    st.caption(benchmark.note)

    beta_value = None
    portfolio_returns = portfolio_return_series(returns[used], w)
    try:
        beta_value = beta(portfolio_returns,
                          simple_returns(state.fetch_history(benchmark.symbol)))
    except Exception as exc:
        st.warning(f"Beta not shown: could not load {benchmark.symbol} ({exc}). "
                   f"A beta without its benchmark series is not a number worth "
                   f"guessing at.")

    metrics = risk_metrics(
        decomposition, diversification_ratio(w, cov), stats,
        drawdown_stats=drawdown(portfolio_value_series(returns[used], w)),
        beta_value=beta_value,
        benchmark_name=benchmark.label if beta_value is not None else None)

    # Effective holdings leads: it is the only reading that sees a cluster of
    # holdings all making the same bet. Pairwise correlation reports edges and
    # structurally cannot see the group.
    lead = next(m for m in metrics if m.label == "Effective holdings")
    st.metric(lead.label, lead.value)
    st.markdown(f"**{lead.sentence}**")

    for metric in metrics:
        if metric is lead:
            continue
        with st.container(border=True):
            st.metric(metric.label, metric.value)
            st.caption(metric.sentence)
            if metric.warning:
                st.warning(metric.warning)

    # ---- 6. Detail ---------------------------------------------------------
    st.subheader("How volatile each holding is on its own")
    st.caption("A broad European equity index typically sits near 15% a year. "
               "Most thematic ETFs run two to three times that, and that is "
               "what the concentration costs.")
    st.dataframe(
        pd.DataFrame([{"Instrument": name, "Annualised volatility": vol,
                       "Times a broad index": mult}
                      for name, vol, mult in
                      volatility_context(standalone_volatilities(cov), instruments)]),
        hide_index=True, width="stretch",
        column_config={"Instrument": st.column_config.TextColumn(width="medium")})

    st.subheader("Correlation")
    st.caption(
        f"Pairs above {HIGH_CORRELATION_THRESHOLD:.2f} are listed under Needs "
        f"attention. This grid is a pairwise measure: it shows one edge at a "
        f"time and cannot see a group of holdings that all make the same bet. "
        f"The scale stays centred on zero even when every value sits on one "
        f"arm, because nothing on the cool side means nothing here hedges "
        f"anything else, and that is itself the finding.")
    st.altair_chart(
        charts.correlation_heatmap(
            corr, [instruments[i].name if i in instruments else i for i in used]),
        use_container_width=True)
