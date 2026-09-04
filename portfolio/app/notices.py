"""How problems are shown. Presentation only.

The rule the rest of this application follows is that a warning belongs next to
the number it affects. That rule has a failure mode, and the degraded state
found it: with ten unreachable prices the page became eight stacked filled
amber blocks and no data at all.

At that density colour stops meaning state and becomes the background. Amber is
supposed to say "look here"; eighty percent of a viewport in amber says
nothing. So the treatment is graded rather than uniform:

    one filled block   the summary: what is wrong, and how many
    quiet amber lines  up to three specifics, as blockquotes with a left rule
    a disclosure       everything else

Same information, roughly a tenth of the visual weight, and the filled block
still works because it is the only one on the page.
"""

from __future__ import annotations

import streamlit as st

__all__ = ["notices", "MAX_VISIBLE"]

# Three is enough to see the shape of a problem. Past that the list stops being
# read and the count is the only part that still informs.
MAX_VISIBLE = 3


def notices(summary: str, items: list[str], *, tone: str = "orange",
            detail_label: str = "Show all") -> None:
    """One filled summary line, then quiet detail.

    `items` are rendered as markdown blockquotes, which Streamlit draws with a
    left rule and no fill -- readable as a list of problems without competing
    with the summary for attention.
    """
    if not items:
        return
    st.warning(summary)
    for item in items[:MAX_VISIBLE]:
        st.markdown(f"> :{tone}[{item}]")
    remaining = items[MAX_VISIBLE:]
    if remaining:
        with st.expander(f"{detail_label} ({len(remaining)} more)"):
            for item in remaining:
                st.markdown(f"> :{tone}[{item}]")
