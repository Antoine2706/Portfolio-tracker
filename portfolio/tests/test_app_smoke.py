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

    def test_demo_mode_is_the_default_and_is_announced(self):
        """Real data shown while the user believes they are in demo, and demo
        data shown as real, are both bad enough to state on every view."""
        at = run()
        assert any("DEMO DATA" in m.value for m in at.markdown)

    def test_the_mode_banner_does_not_spend_the_loss_colour(self):
        """Red means loss throughout this application. Using it for a mode
        indicator would weaken it exactly where it carries meaning."""
        at = run()
        assert not any("DEMO DATA" in e.value for e in at.error)

    def test_the_mode_banner_is_not_a_filled_block(self):
        """It cannot change without a deliberate click, so full weight on every
        view only pushes the first real content down the page."""
        at = run()
        assert not any("DEMO DATA" in w.value for w in at.warning)

    def test_refresh_button_exists(self):
        at = run()
        assert "Refresh prices" in [b.label for b in at.sidebar.button]


@pytest.mark.parametrize("page", ["Holdings", "Risk", "Instruments", "Transactions"])
def test_every_view_renders_without_exception(page):
    at = run(page)
    assert not at.exception, f"{page}: {[str(e) for e in at.exception]}"


class TestGracefulDegradation:
    """A view that crashes on an unreachable provider is worse than one saying so.

    The provider is stubbed to fail rather than relying on the machine having no
    network. These tests originally depended on the ambient environment: they
    passed in a sandbox with no egress and failed in CI, where the network works
    and the views correctly built a real risk model. A test that only passes
    when the network is down is not testing degradation, it is testing the
    runner.
    """

    @pytest.fixture(autouse=True)
    def _provider_is_down(self, monkeypatch):
        from portfolio.app import state

        def unreachable(*args, **kwargs):
            raise RuntimeError("provider unreachable (stubbed)")

        monkeypatch.setattr(state, "fetch_quote", unreachable)
        monkeypatch.setattr(state, "fetch_history", unreachable)
        monkeypatch.setattr(state, "fx_rates_raw", dict)

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


class TestRendersWithWorkingPrices:
    """The other half of the degradation tests, and their control.

    Without this, the degraded-state tests prove nothing on a machine with no
    network: they would pass whether or not the stub took effect. Here the stub
    supplies working prices, so if patching did not reach the app this class
    would fail -- which makes it the evidence that the other class is testing
    what it claims.

    It also covers the path that matters most and is hardest to reach in CI:
    the Risk view with a real covariance matrix.
    """

    @pytest.fixture(autouse=True)
    def _provider_works(self, monkeypatch):
        import datetime as dt

        import numpy as np
        import pandas as pd

        from portfolio.app import state

        rng = np.random.default_rng(7)
        rows = {"EUDF.DE": 377, "DFNC.DE": 320, "8RMY.DE": 356, "ASWC.DE": 505,
                "ISAE.AS": 508, "AIGG.MI": 503, "WEAT.MI": 503, "AIGE.MI": 503,
                "ESIE.DE": 490, "GLUX.PA": 508, "MEUD.PA": 511}
        common = rng.normal(0, 0.008, 520)
        series = {}
        for i, (symbol, n) in enumerate(rows.items()):
            steps = common[-n:] * 0.6 + rng.normal(0, 0.008, n)
            series[symbol] = pd.Series(
                100 * np.exp(np.cumsum(steps)),
                index=pd.bdate_range(end="2026-09-03", periods=n))

        def history(symbol):
            if symbol not in series:
                raise RuntimeError(f"no series for {symbol}")
            return series[symbol]

        def quote(symbol):
            return {"symbol": symbol, "price": str(round(float(history(symbol).iloc[-1]), 4)),
                    "currency": "EUR",
                    "as_of": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "source": "stub", "delay_minutes": 15, "is_stale": False}

        monkeypatch.setattr(state, "fetch_history", history)
        monkeypatch.setattr(state, "fetch_quote", quote)
        monkeypatch.setattr(state, "fx_rates_raw",
                            lambda: {"USD": "0.92", "GBP": "1.17"})

    def test_patching_reaches_the_app(self):
        """If this fails, the degraded-state tests are proving nothing."""
        at = run("Holdings")
        assert not at.exception
        assert any("EUR" in m.value for m in at.metric), \
            "a working provider must produce a portfolio value"

    def test_holdings_shows_a_value_and_no_fetch_failures(self):
        at = run("Holdings")
        warnings = [w.value for w in at.warning]
        assert not any("price fetch" in w for w in warnings), warnings

    def test_risk_builds_a_model_rather_than_explaining_why_it_cannot(self):
        at = run("Risk")
        assert not at.exception
        messages = [m.value for m in list(at.info)] + [m.value for m in list(at.error)]
        assert not any("Not enough holdings" in m for m in messages), messages

    def test_risk_shows_the_divergence_headline(self):
        """The reason the application exists."""
        at = run("Risk")
        assert any("of your money but" in m.value for m in at.markdown)

    def test_risk_names_its_benchmark(self):
        at = run("Risk")
        labels = [m.label for m in at.metric]
        assert any("Beta vs" in label for label in labels), labels
