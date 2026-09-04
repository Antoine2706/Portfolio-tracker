"""Chart construction. Presentation only -- no numbers are computed here.

Every colour is read from `.streamlit/config.toml` at runtime via
`st.get_option`, so the palette has exactly one home. Nothing is hardcoded and
nothing is injected as CSS.

The colour discipline, in one place
-----------------------------------
Three of this application's four visuals encode polarity or magnitude, not
identity. Getting that right is most of the visual improvement, and it is why
almost no categorical colour appears:

    weight bars        magnitude  -> one hue
    divergence bars    polarity   -> diverging, neutral at zero
    correlation grid   polarity   -> diverging, neutral at zero
    metrics            no colour

Assigning a different colour to each holding would be decoration: the colour
would carry no information the label does not already carry. Colour is reserved
for state, and in the data area that means polarity and nothing else.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

__all__ = ["weight_bars", "divergence_bars", "correlation_heatmap"]

LABEL_LIMIT = 34          # truncate long fund names rather than let them wrap


def _palette() -> dict[str, str]:
    """Palette read from config. One home for every colour, no CSS.

    The diverging ramp has ten stops, so the poles are its ends and the neutral
    is its middle -- never indices 0, 1, 2, which are three adjacent shades of
    the same pole and would collapse the encoding back to a single hue.

    Named cool/neutral/warm rather than negative/positive because the mapping is
    a design decision, not arithmetic: a holding carrying MORE risk than capital
    is the one wanting attention, so it gets the warm end.
    """
    diverging = st.get_option("theme.chartDivergingColors")
    return {
        "cool": diverging[0],
        "neutral": _midpoint(diverging),
        "warm": diverging[-1],
        "single": st.get_option("theme.primaryColor"),
        "text": st.get_option("theme.textColor"),
        "muted": st.get_option("theme.grayColor") or "#8a8880",
        "border": st.get_option("theme.borderColor"),
    }


def _midpoint(ramp: list[str]) -> str:
    """The true centre of a diverging ramp.

    An even-length ramp has no middle element, and either neighbour carries a
    cast toward its own pole. Averaging the two keeps a bar at zero genuinely
    neutral, which is the point: a holding behaving exactly as expected should
    be visually silent.
    """
    mid = len(ramp) // 2
    if len(ramp) % 2:
        return ramp[mid]
    a, b = ramp[mid - 1].lstrip("#"), ramp[mid].lstrip("#")
    return "#" + "".join(
        f"{(int(a[i:i+2], 16) + int(b[i:i+2], 16)) // 2:02x}" for i in (0, 2, 4))


def _base(frame: pd.DataFrame, height: int) -> alt.Chart:
    """A chart with the chrome removed. No gridlines, no legend, no title."""
    return alt.Chart(frame).properties(height=height)


def _axis_y(field: str) -> alt.Y:
    """Category axis: labels only, no title, no ticks, no domain line."""
    return alt.Y(f"{field}:N", sort=None, title=None,
                 axis=alt.Axis(labelLimit=200, labelFontSize=12, ticks=False,
                               domain=False, labelPadding=8))


def weight_bars(rows: list[tuple[str, float]], height: int | None = None) -> alt.LayerChart:
    """Capital weight per holding. Magnitude, so one hue.

    Sorted descending, value labelled directly on each bar. No legend: a legend
    for a single series is chrome, and a per-holding colour scale would be
    decoration dressed as information.
    """
    colour = _palette()
    frame = pd.DataFrame(
        [{"name": n[:LABEL_LIMIT], "weight": w} for n, w in rows])
    frame = frame.sort_values("weight", ascending=False)
    height = height or max(120, 30 * len(frame))

    bars = _base(frame, height).mark_bar(
        color=colour["single"], cornerRadiusEnd=3, size=18
    ).encode(
        x=alt.X("weight:Q", title=None,
                axis=alt.Axis(format=".0%", grid=False, domain=False, ticks=False,
                              labelColor=colour["muted"])),
        y=_axis_y("name"),
        tooltip=[alt.Tooltip("name:N", title="Instrument"),
                 alt.Tooltip("weight:Q", format=".2%", title="Weight")],
    )
    labels = _base(frame, height).mark_text(
        align="left", dx=6, fontSize=12, color=colour["text"]
    ).encode(x=alt.X("weight:Q"), y=_axis_y("name"),
             text=alt.Text("weight:Q", format=".1%"))
    return (bars + labels).configure_view(stroke=None)


def divergence_bars(rows: list[tuple[str, float]],
                    height: int | None = None) -> alt.LayerChart:
    """Risk share minus capital weight. The product, so it gets the most care.

    Polarity around a meaningful zero, so a diverging encoding: over-contributors
    on one arm, under-contributors on the other, neutral grey at the middle. A
    holding sitting at zero divergence is visually silent, which is exactly
    right -- it is behaving as expected and has earned no attention.

    Sorted by absolute divergence descending, so whatever is most mispriced
    against intuition is at the top.
    """
    colour = _palette()
    frame = pd.DataFrame(
        [{"name": n[:LABEL_LIMIT], "divergence": d, "magnitude": abs(d)}
         for n, d in rows]).sort_values("magnitude", ascending=False)
    height = height or max(120, 30 * len(frame))

    # An explicit three-point domain symmetric about zero. `domainMid` alone
    # leaves the range unbound and every bar renders in the first colour, which
    # silently turns a diverging encoding back into a single hue -- the exact
    # failure this chart exists to avoid.
    extent = float(frame["magnitude"].max()) or 1.0
    bars = _base(frame, height).mark_bar(cornerRadiusEnd=3, size=18).encode(
        x=alt.X("divergence:Q", title=None,
                axis=alt.Axis(format="+.0%", grid=False, domain=False, ticks=False,
                              labelColor=colour["muted"])),
        y=_axis_y("name"),
        color=alt.Color(
            "divergence:Q", legend=None,
            scale=alt.Scale(domain=[-extent, 0.0, extent],
                            range=[colour["cool"], colour["neutral"],
                                   colour["warm"]])),
        tooltip=[alt.Tooltip("name:N", title="Instrument"),
                 alt.Tooltip("divergence:Q", format="+.2%", title="Risk minus capital")],
    )
    # Zero line, so the midpoint is legible even when every bar is on one arm.
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color=colour["border"], strokeWidth=1).encode(x="x:Q")

    # Labels flip side with the bar. Drawn as two filtered layers because align
    # is a mark property, not an encoding, so it cannot be made conditional.
    def _labels(negative: bool):
        return _base(frame, height).transform_filter(
            (alt.datum.divergence < 0) if negative else (alt.datum.divergence >= 0)
        ).mark_text(align="right" if negative else "left",
                    dx=-6 if negative else 6,
                    fontSize=12, color=colour["text"]
        ).encode(x=alt.X("divergence:Q"), y=_axis_y("name"),
                 text=alt.Text("divergence:Q", format="+.1%"))

    return (zero + bars + _labels(True) + _labels(False)).configure_view(stroke=None)


def correlation_heatmap(matrix: pd.DataFrame, labels: list[str]) -> alt.Chart:
    """Correlation grid, diverging, centred on zero.

    The scale stays centred on zero even when every value sits on one arm.
    Negative correlation is genuinely meaningful, and its complete absence is
    itself a finding: a matrix with nothing on the cool side says nothing in
    this portfolio hedges anything else. Re-centring to make the picture
    prettier would hide exactly that.
    """
    colour = _palette()
    short = [l[:26] for l in labels]
    display = matrix.copy()
    display.index = short
    display.columns = short
    long = display.reset_index(names="row").melt(
        id_vars="row", var_name="column", value_name="correlation")

    # alt.Step sets the height of each band rather than the whole plot, which
    # is the right idiom for a categorical axis: rows keep a readable height
    # whatever the holding count. Pinning an absolute width and height instead
    # collapsed the plot area to zero and the cells vanished entirely, leaving
    # axes and a legend that looked like a styling problem rather than a
    # missing chart.

    return alt.Chart(long).mark_rect(stroke=None).encode(
        x=alt.X("column:N", sort=short, title=None,
                axis=alt.Axis(labelAngle=-45, labelLimit=180, labelFontSize=11,
                              ticks=False, domain=False, labelOverlap=False,
                              labelColor=colour["text"])),
        y=alt.Y("row:N", sort=short, title=None,
                axis=alt.Axis(labelLimit=180, labelFontSize=11, ticks=False,
                              domain=False, labelOverlap=False,
                              labelColor=colour["text"])),
        color=alt.Color(
            "correlation:Q", title=None,
            scale=alt.Scale(domain=[-1, 0, 1],
                            range=[colour["cool"], colour["neutral"],
                                   colour["warm"]]),
            legend=alt.Legend(orient="right", gradientLength=180, format=".1f",
                              labelColor=colour["muted"], title="correlation",
                              titleColor=colour["muted"])),
        tooltip=[alt.Tooltip("row:N", title=""), alt.Tooltip("column:N", title=""),
                 alt.Tooltip("correlation:Q", format=".3f", title="Correlation")],
    ).properties(height=alt.Step(34)).configure_view(stroke=None)
