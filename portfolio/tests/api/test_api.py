"""The API, end to end, against the seed data and the offline provider.

These are the smoke tests the Streamlit version ran in three minutes against
a network it could not reach, rewritten to run in seconds against a provider
that answers deterministically. They check the contract in
`api/schemas.py` field by field where a page depends on the field.
"""

from __future__ import annotations

import pytest

# The seed ledger: six open positions, one closed (AIGE), three never traded.
OPEN = {"IE0002Y8CX98", "IE000IAXNM41", "IE000I7E6HL0", "IE00B6R52143",
        "JE00BN7KB664", "LU1681048630"}
WATCHLIST = {"GB00B15KYL00", "IE000OJ5TQP4", "IE00BMW42637"}
CLOSED = "GB00B15KYB02"


class TestShell:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["mode"] == "seed" and body["provider"] == "fixture"

    def test_settings_reports_the_mode_and_the_cache(self, client):
        body = client.get("/api/settings").json()
        assert body["mode"] == "seed"
        assert body["cache"]["path"].endswith(".sqlite")
        assert body["quote_ttl_minutes"] == 15

    def test_mode_switch_is_explicit_and_reported(self, client):
        body = client.put("/api/settings", json={"mode": "user"}).json()
        assert body["mode"] == "user"
        assert client.get("/api/health").json()["mode"] == "user"
        # An empty user store is a legitimate state, not an error.
        snap = client.get("/api/snapshot").json()
        assert snap["holdings"] == [] and snap["meta"]["mode"] == "user"
        assert snap["risk"]["available"] is False
        assert snap["performance"]["available"] is False

    def test_errors_use_one_envelope(self, client):
        body = client.get("/api/holdings/XX0000000000").json()
        assert set(body) == {"error"} and set(body["error"]) == {"code", "message"}

    def test_unknown_api_path_is_404_not_index(self, client):
        assert client.get("/api/nothing").status_code == 404

    def test_openapi_is_served(self, client):
        assert client.get("/api/openapi.json").status_code == 200


class TestSnapshot:
    def test_holdings_are_the_open_positions(self, client):
        snap = client.get("/api/snapshot").json()
        assert {h["isin"] for h in snap["holdings"]} == OPEN
        assert {w["isin"] for w in snap["watchlist"]} == WATCHLIST
        assert CLOSED not in {h["isin"] for h in snap["holdings"]}

    def test_totals_add_up(self, client):
        snap = client.get("/api/snapshot").json()
        total = sum(h["value"] for h in snap["holdings"] if h["value"] is not None)
        assert snap["totals"]["value"] == pytest.approx(total, rel=1e-9)
        assert sum(h["weight"] for h in snap["holdings"]) == pytest.approx(1.0)
        assert snap["totals"]["priced_holdings"] == 6
        assert snap["totals"]["day_change"] is not None

    def test_every_holding_carries_its_provenance(self, client):
        snap = client.get("/api/snapshot").json()
        for h in snap["holdings"]:
            assert h["price_note"] and "as of" in h["price_note"]
            assert h["price_delay_minutes"] == 15
            assert len(h["sparkline"]) == 30
            assert h["symbol"]

    def test_risk_model_is_built(self, client):
        risk = client.get("/api/snapshot").json()["risk"]
        assert risk["available"] is True, risk["reason"]
        assert risk["headline"]["sentence"].endswith(").")
        assert len(risk["divergence"]) == 6
        assert sum(r["risk_share"] for r in risk["divergence"]) == pytest.approx(1.0)
        assert risk["correlation"]["matrix"][0][0] == pytest.approx(1.0)
        assert len(risk["correlation"]["isins"]) == 6
        assert risk["var"] and all(v["loss"] > 0 for v in risk["var"])
        assert risk["scenarios"] and risk["worst_periods"]
        assert risk["window"]["effective"] <= 252
        assert risk["beta"] is not None and risk["beta_benchmark"].startswith("STOXX")
        keys = {m["key"] for m in risk["metrics"]}
        assert {"volatility", "diversification", "effective_holdings",
                "max_drawdown", "beta"} <= keys

    def test_the_short_defence_history_binds_the_window(self, client):
        """DFNC.DE has 320 rows; the demo must say so, as the real book would."""
        window = client.get("/api/snapshot").json()["risk"]["window"]
        assert window["effective"] == 252 or window["binding_isin"] == "IE000IAXNM41"

    def test_performance_is_built(self, client):
        perf = client.get("/api/snapshot").json()["performance"]
        assert perf["available"] is True, perf["reason"]
        assert len(perf["dates"]) == len(perf["value"]) == len(perf["index"])
        assert perf["index"][0] == pytest.approx(100.0)
        assert perf["benchmark_index"][0] == pytest.approx(100.0)
        assert perf["summary"]["observations"] > 200
        assert perf["flows"] and perf["monthly"] and perf["yearly"]
        assert set(perf["trailing"]) == {"1w", "1m", "3m", "6m", "ytd", "1y", "all"}
        assert len(perf["contributions"]) >= 6

    def test_exposure_and_alerts(self, client):
        snap = client.get("/api/snapshot").json()
        assert set(snap["exposure"]) == {"asset_class", "issuer", "base_currency",
                                         "quote_currency", "exchange"}
        assert sum(s["weight"] for s in snap["exposure"]["asset_class"]) == pytest.approx(1.0)
        assert all(a["severity"] in {"INFO", "WARNING", "SERIOUS", "CRITICAL"}
                   for a in snap["alerts"])

    def test_benchmark_can_be_switched(self, client):
        snap = client.get("/api/snapshot", params={"benchmark": "IWDA.AS"}).json()
        assert snap["selected_benchmark"] == "IWDA.AS"
        assert snap["risk"]["beta_benchmark"].startswith("MSCI World")

    def test_lookback_is_honoured(self, client):
        snap = client.get("/api/snapshot", params={"lookback": 120}).json()
        assert snap["lookback"] == 120
        assert snap["risk"]["window"]["effective"] == 120

    def test_meta_names_the_demo_mode(self, client):
        meta = client.get("/api/snapshot").json()["meta"]
        assert meta["mode"] == "seed" and meta["provider"] == "fixture"
        assert meta["prices_as_of"] == "2026-09-04"
        assert meta["failures"] == []


class TestCaching:
    def test_second_snapshot_is_the_cached_analysis(self, client, service):
        client.get("/api/snapshot")
        first = service.analysis()
        client.get("/api/snapshot")
        assert service.analysis() is first

    def test_a_ledger_write_invalidates(self, client, service):
        client.get("/api/snapshot")
        first = service.analysis()
        r = client.post("/api/transactions", json={
            "date": "2026-09-01", "isin": "LU1681048630", "type": "BUY",
            "quantity": "1", "price_per_unit": "100", "currency": "EUR"})
        assert r.status_code == 201, r.text
        assert service.analysis() is not first

    def test_refresh_rebuilds_and_records_the_time(self, client, service):
        assert client.get("/api/settings").json()["last_refresh"] is None
        r = client.post("/api/refresh")
        assert r.status_code == 200
        assert client.get("/api/settings").json()["last_refresh"] is not None

    def test_prices_survive_a_restart_through_the_cache(self, client, service, data_root):
        client.get("/api/snapshot")
        stats = service.cache.stats()
        assert stats.symbols >= 9 and stats.rows > 1000


class TestDegradation:
    """A view that crashes on an unreachable provider is worse than one that says so."""

    @pytest.fixture(autouse=True)
    def _one_symbol_down(self, service):
        from portfolio.data.market import MarketData
        from portfolio.data.providers.fixture import FixtureProvider
        service.market_data = MarketData(
            FixtureProvider(fail=frozenset({"WEAT.MI"})), cache=None,
            base="EUR", max_workers=2)
        service.invalidate()

    def test_one_failure_does_not_blank_the_snapshot(self, client):
        snap = client.get("/api/snapshot").json()
        assert snap["meta"]["failures"], "the failure must be reported"
        assert any("WEAT" in f["message"] or f["key"] == "JE00BN7KB664"
                   for f in snap["meta"]["failures"])
        wheat = next(h for h in snap["holdings"] if h["isin"] == "JE00BN7KB664")
        assert wheat["value"] is None and wheat["warnings"]
        assert "JE00BN7KB664" in snap["unpriced"]
        assert snap["totals"]["priced_holdings"] == 5
        assert snap["risk"]["available"] is True
        assert any(a["severity"] in {"SERIOUS", "CRITICAL"} for a in snap["alerts"])


class TestHoldingDetail:
    def test_detail_for_a_held_instrument(self, client):
        body = client.get("/api/holdings/IE0002Y8CX98").json()
        assert body["holding"]["isin"] == "IE0002Y8CX98"
        assert len(body["prices"]["dates"]) == 377
        assert {m["type"] for m in body["markers"]} == {"BUY", "SELL"}
        assert len(body["transactions"]) == 3
        assert body["stats"]["volatility"] > 0
        assert body["stats"]["beta_vs_portfolio"] is not None
        assert len(body["correlations"]) == 5

    def test_detail_for_a_watchlist_instrument(self, client):
        body = client.get("/api/holdings/IE00BMW42637").json()
        assert body["holding"]["in_risk_model"] is True
        assert body["transactions"] == [] and body["markers"] == []

    def test_unknown_isin_is_404(self, client):
        assert client.get("/api/holdings/DE000A0F5UF5").status_code == 404


class TestInstruments:
    NEW = "DE000A0F5UF5"

    def test_list(self, client):
        rows = client.get("/api/instruments").json()
        assert len(rows) == 10
        assert sum(r["held"] for r in rows) == 7          # six open plus the closed one

    def test_resolve_rejects_a_bad_check_digit(self, client):
        r = client.post("/api/instruments/resolve", json={"isin": "IE0002Y8CX97"})
        assert r.status_code == 400 and "check digit" in r.json()["error"]["message"]

    def test_resolve_refuses_a_known_isin(self, client):
        r = client.post("/api/instruments/resolve", json={"isin": "IE0002Y8CX98"})
        assert r.status_code == 409

    def test_resolve_then_save_then_delete(self, client):
        r = client.post("/api/instruments/resolve", json={"isin": self.NEW})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["candidates"] and body["recommended"]
        assert body["filtered_by_class"]
        chosen = body["recommended"]

        r = client.post("/api/instruments", json={
            "isin": self.NEW, "yahoo_symbol": chosen, "name": "Test fund",
            "issuer": "Test", "asset_class": "ETF", "base_currency": "USD"})
        assert r.status_code == 201, r.text
        saved = r.json()
        assert saved["provider_symbols"] == {"fixture": chosen}
        assert saved["held"] is False and saved["base_currency"] == "USD"

        # It appears on the watchlist with a price, straight away.
        snap = client.get("/api/snapshot").json()
        assert any(w["isin"] == self.NEW and w["price"] for w in snap["watchlist"])

        assert client.delete(f"/api/instruments/{self.NEW}").status_code == 204
        assert client.delete(f"/api/instruments/{self.NEW}").status_code == 404

    def test_save_refuses_a_symbol_the_resolver_did_not_offer(self, client):
        r = client.post("/api/instruments", json={
            "isin": self.NEW, "yahoo_symbol": "WDEF", "name": "x"})
        assert r.status_code == 400

    def test_delete_refuses_a_held_instrument(self, client):
        r = client.delete("/api/instruments/IE0002Y8CX98")
        assert r.status_code == 409 and "Deactivate" in r.json()["error"]["message"]

    def test_patch_symbol_marks_it_manual(self, client):
        r = client.patch("/api/instruments/IE00BMW42637",
                         json={"provider_symbols": {"fixture": "ESIE.L"}})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["provider_symbols"]["fixture"] == "ESIE.L"
        assert "provider_symbols.fixture" in body["manual_overrides"]
        assert body["exchange"] == "XLON"

    def test_deactivate_hides_from_the_watchlist(self, client):
        r = client.patch("/api/instruments/IE00BMW42637", json={"active": False})
        assert r.status_code == 200 and r.json()["active"] is False
        snap = client.get("/api/snapshot").json()
        assert "IE00BMW42637" not in {w["isin"] for w in snap["watchlist"]}
        r = client.patch("/api/instruments/IE00BMW42637", json={"active": True})
        assert r.json()["active"] is True


class TestTransactions:
    def test_list_hides_voided_unless_asked(self, client):
        live = client.get("/api/transactions").json()
        everything = client.get("/api/transactions", params={"include_voided": True}).json()
        assert len(live) == 13 and len(everything) == 14
        voided = [t for t in everything if t["voided"]]
        assert len(voided) == 1 and "mistyped" in voided[0]["void_reason"]

    def test_append_and_void(self, client):
        r = client.post("/api/transactions", json={
            "date": "2026-09-01", "isin": "LU1681048630", "type": "BUY",
            "quantity": "2", "price_per_unit": "120.5", "currency": "EUR", "fees": "1"})
        assert r.status_code == 201, r.text
        txn = r.json()
        assert txn["net"] == pytest.approx(-242.0) and txn["name"].startswith("Amundi")
        r = client.post(f"/api/transactions/{txn['id']}/void", json={"reason": "test"})
        assert r.status_code == 200 and r.json()["voided"] is True
        assert client.post(f"/api/transactions/{txn['id']}/void", json={}).status_code == 409
        assert len(client.get("/api/transactions").json()) == 13

    def test_a_sell_of_more_than_held_is_refused_before_it_is_written(self, client):
        r = client.post("/api/transactions", json={
            "date": "2026-09-01", "isin": "LU1681048630", "type": "SELL",
            "quantity": "999", "price_per_unit": "120", "currency": "EUR"})
        assert r.status_code == 400 and "only 40" in r.json()["error"]["message"]
        assert len(client.get("/api/transactions").json()) == 13

    def test_an_unknown_isin_is_refused(self, client):
        r = client.post("/api/transactions", json={
            "date": "2026-09-01", "isin": "DE000A0F5UF5", "type": "BUY",
            "quantity": "1", "price_per_unit": "1"})
        assert r.status_code == 400 and "universe" in r.json()["error"]["message"]

    def test_voiding_a_buy_that_a_sell_depends_on_is_refused(self, client):
        # EUDF: bought 120 and 80, sold 60, so 140 are held. Sell them all,
        # then try to void the first buy: the later sells would exceed what
        # was ever bought, and the ledger must refuse rather than go negative.
        r = client.post("/api/transactions", json={
            "date": "2026-09-01", "isin": "IE0002Y8CX98", "type": "SELL",
            "quantity": "140", "price_per_unit": "60", "currency": "EUR"})
        assert r.status_code == 201, r.text
        buys = [t for t in client.get("/api/transactions").json()
                if t["isin"] == "IE0002Y8CX98" and t["type"] == "BUY"]
        first = min(buys, key=lambda t: t["date"])
        r = client.post(f"/api/transactions/{first['id']}/void", json={"reason": "x"})
        assert r.status_code == 409 and "sell" in r.json()["error"]["message"].lower()

    def test_import_preview_and_import(self, client):
        text = ("date;isin;type;quantity;price;currency;fees\n"
                "01.09.2026;LU1681048630;Kauf;3;120,50;EUR;1,00\n"
                "02.09.2026;IE0002Y8CX98;BUY;10;30.00;EUR;0\n"
                "03.09.2026;XX0000000000;BUY;1;1;EUR;0\n")
        r = client.post("/api/transactions/import/preview", json={"text": text})
        assert r.status_code == 200, r.text
        preview = r.json()
        assert preview["valid"] == 2 and preview["invalid"] == 1
        assert preview["mapping"]["decimal_comma"] is True
        r = client.post("/api/transactions/import", json={"text": text})
        assert r.status_code == 200 and r.json()["imported"] == 2
        assert len(client.get("/api/transactions").json()) == 15
        # Importing the same file again flags every row as a duplicate.
        again = client.post("/api/transactions/import/preview", json={"text": text}).json()
        assert again["valid"] == 0
        assert any("duplicate" in (row["error"] or "") for row in again["rows"])

    def test_exports(self, client):
        r = client.get("/api/export/transactions.csv")
        assert r.status_code == 200 and r.text.startswith("id,date,isin")
        r = client.get("/api/export/holdings.csv")
        assert r.status_code == 200 and "isin" in r.text.splitlines()[0]
        r = client.get("/api/export/workbook.xlsx")
        assert r.status_code == 200 and r.content[:2] == b"PK"


class TestSimulate:
    def test_equal_weight_preset(self, client):
        r = client.post("/api/simulate", json={"preset": "equal"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["holdings"]) == 6
        weights = [h["weight_after"] for h in body["holdings"]]
        assert all(w == pytest.approx(1 / 6) for w in weights)
        assert body["total_after"] == pytest.approx(body["total_before"])
        assert body["trades"]

    def test_risk_parity_equalises_risk_shares(self, client):
        body = client.post("/api/simulate", json={"preset": "risk_parity"}).json()
        shares = [h["risk_after"] for h in body["holdings"]]
        assert max(shares) - min(shares) < 1e-4

    def test_cash_change_moves_the_total(self, client):
        body = client.post("/api/simulate",
                           json={"changes": {"IE0002Y8CX98": 1000.0}}).json()
        assert body["total_after"] == pytest.approx(body["total_before"] + 1000.0)
        top = next(h for h in body["holdings"] if h["isin"] == "IE0002Y8CX98")
        assert top["value_after"] == pytest.approx(top["value_before"] + 1000.0)

    def test_a_watchlist_instrument_can_be_included(self, client):
        body = client.post("/api/simulate", json={
            "include": ["IE00BMW42637"], "changes": {"IE00BMW42637": 2000.0}}).json()
        assert any(h["isin"] == "IE00BMW42637" and h["value_after"] == 2000.0
                   for h in body["holdings"])
        assert any(t["isin"] == "IE00BMW42637" and t["action"] == "BUY"
                   for t in body["trades"])

    def test_unknown_isin_is_400(self, client):
        r = client.post("/api/simulate", json={"changes": {"DE000A0F5UF5": 1.0}})
        assert r.status_code == 400

    def test_nothing_requested_is_400(self, client):
        assert client.post("/api/simulate", json={}).status_code == 400


class TestClient_:
    def test_index_is_served_at_root_and_for_unknown_paths(self, client):
        from portfolio.api.app import WEB_DIR
        if not (WEB_DIR / "index.html").exists():
            pytest.skip("web client not present")
        assert client.get("/").status_code == 200
        assert "text/html" in client.get("/anything/else").headers["content-type"]
        assert client.get("/static/vendor/echarts.min.js").status_code == 200
