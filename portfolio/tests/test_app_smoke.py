"""Smoke tests: every view renders without raising.

These need Streamlit, and the rest of the suite must run without it, so they
skip when it is absent. The core constraint is unaffected -- `core/` still
imports and tests clean on a machine with no Streamlit, which test_layering
verifies in a subprocess.

Network is unavailable here, so these also check the more important thing: that
the views degrade into visible warnings rather than exceptions when prices
cannot be fetched. A view that crashes on an unreachable provider is worse than
one that says so.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("streamlit", reason="UI smoke tests need streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

# Absolute: AppTest resolves relative paths against the calling file.
APP = str(pathlib.Path(__file__).resolve().parents[1] / "app" / "main.py")
TIMEOUT = 90


def run(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    if page and page != "Holdings":
        at.sidebar.radio[0].set_value(page).run()
    return at


class TestAppShell:
    def test_app_starts(self):
        at = run()
        assert not at.exception, [str(e) for e in at.exception]

    def test_four_views_are_offered(self):
        at = run()
        assert at.sidebar.radio[0].options == ["Holdings", "Risk", "Instruments",
                                               "Transactions"]

    def test_demo_mode_is_the_default_and_is_loud(self):
        """Real data shown while the user believes they are in demo, and demo
        data shown as real, are both bad enough to warrant a coloured block."""
        at = run()
        assert any("DEMO DATA" in e.value for e in at.error)

    def test_refresh_button_exists(self):
        at = run()
        assert "Refresh prices" in [b.label for b in at.sidebar.button]


@pytest.mark.parametrize("page", ["Holdings", "Risk", "Instruments", "Transactions"])
def test_every_view_renders_without_exception(page):
    at = run(page)
    assert not at.exception, f"{page}: {[str(e) for e in at.exception]}"


class TestGracefulDegradation:
    def test_holdings_warns_rather_than_crashing_without_prices(self):
        at = run("Holdings")
        assert not at.exception
        assert list(at.warning) or list(at.info) or list(at.error), \
            "an unreachable provider must produce a visible message"

    def test_risk_explains_itself_when_it_cannot_build_a_model(self):
        at = run("Risk")
        assert not at.exception
        messages = [m.value for m in list(at.info)] + \
                   [m.value for m in list(at.error)] + \
                   [m.value for m in list(at.warning)]
        assert messages, "the Risk view must explain why it has no numbers"
        assert any("holdings" in m.lower() or "history" in m.lower()
                   or "covariance" in m.lower() for m in messages), messages

    def test_transactions_lists_the_seed_ledger(self):
        at = run("Transactions")
        assert not at.exception
        assert at.dataframe, "the seed ledger should render as a table"
