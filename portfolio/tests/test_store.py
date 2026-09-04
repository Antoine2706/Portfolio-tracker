"""Store round-trips, and the seed/user separation."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from portfolio.core.models import (AssetClass, Instrument, Transaction,
                                   TransactionType as T)
from portfolio.core.positions import derive_positions
from portfolio.data.store import DEFAULT_ROOT, DataMode, DataStore, resolve_mode

D = dt.date
A = "IE0002Y8CX98"


@pytest.fixture
def store(tmp_path) -> DataStore:
    return DataStore.open(DataMode.USER, root=tmp_path)


class TestModeSelection:
    def test_defaults_to_seed(self, monkeypatch):
        """Demo-when-you-wanted-real is annoying; real-on-a-demo-screen is a
        privacy failure. Default to the harmless one."""
        monkeypatch.delenv("PORTFOLIO_DATA_MODE", raising=False)
        assert resolve_mode() is DataMode.SEED

    def test_env_var_selects_user_mode(self, monkeypatch):
        monkeypatch.setenv("PORTFOLIO_DATA_MODE", "user")
        assert resolve_mode() is DataMode.USER

    def test_explicit_argument_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("PORTFOLIO_DATA_MODE", "user")
        assert resolve_mode(DataMode.SEED) is DataMode.SEED

    def test_mode_is_always_describable(self, tmp_path):
        assert "DEMO DATA" in DataStore.open(DataMode.SEED, tmp_path).describe()
        assert "LIVE DATA" in DataStore.open(DataMode.USER, tmp_path).describe()

    def test_seed_and_user_never_share_a_directory(self, tmp_path):
        assert DataStore.open(DataMode.SEED, tmp_path).directory != \
               DataStore.open(DataMode.USER, tmp_path).directory


class TestInstrumentRoundTrip:
    def test_round_trip_preserves_every_field(self, store):
        i = Instrument(A, "WisdomTree Europe Defence", AssetClass.ETF, "EUR",
                       issuer="WisdomTree", primary_symbol="EUDF", exchange="XETR",
                       quote_currency="EUR",
                       provider_symbols={"eodhd": "EUDF.XETRA", "yfinance": "EUDF.DE"})
        i.override("provider_symbols.eodhd", "WDEF.LSE")
        store.save_instruments({A: i})
        back = store.load_instruments()[A]
        assert back.name == i.name and back.issuer == "WisdomTree"
        assert back.provider_symbols == {"eodhd": "WDEF.LSE", "yfinance": "EUDF.DE"}
        assert back.manual_overrides == {"provider_symbols.eodhd"}

    def test_override_marks_survive_a_reload(self, store):
        """Otherwise re-resolution silently reverts the correction after a restart."""
        i = Instrument(A, "x", AssetClass.ETF, "EUR")
        i.override("provider_symbols.eodhd", "OD7S.XETRA")
        store.save_instruments({A: i})
        back = store.load_instruments()[A]
        assert back.is_overridden("provider_symbols.eodhd")
        assert back.apply_resolution({"provider_symbols": {"eodhd": "WRONG"}}) == \
            ["provider_symbols.eodhd"]
        assert back.provider_symbols["eodhd"] == "OD7S.XETRA"

    def test_deactivated_flag_survives(self, store):
        i = Instrument(A, "x", AssetClass.ETF, "EUR")
        i.active = False
        store.save_instruments({A: i})
        assert store.load_instruments()[A].active is False


class TestLedgerIsAppendOnly:
    def test_append_then_read(self, store):
        t = Transaction(D(2025, 1, 1), A, T.BUY, Decimal("10"), Decimal("20"),
                        "EUR", Decimal("1"))
        store.append_transaction(t)
        back = store.load_transactions()
        assert len(back) == 1 and back[0].id == t.id
        assert back[0].quantity == Decimal("10")

    def test_void_hides_the_row_but_leaves_it_on_disk(self, store):
        good = store.append_transaction(
            Transaction(D(2025, 1, 1), A, T.BUY, Decimal("10"), Decimal("20")))
        typo = store.append_transaction(
            Transaction(D(2025, 1, 2), A, T.BUY, Decimal("1000"), Decimal("20")))
        store.void_transaction(typo.id, "quantity mistyped")

        assert [t.id for t in store.load_transactions()] == [good.id]
        assert len(store.load_transactions(include_voided=True)) == 2
        raw = store.transactions_path.read_text()
        assert typo.id in raw, "the ledger file itself must never be rewritten"

    def test_edit_is_void_plus_append(self, store):
        typo = store.append_transaction(
            Transaction(D(2025, 1, 2), A, T.BUY, Decimal("2000"), Decimal("9.85")))
        store.void_transaction(typo.id, "off by 10x")
        fixed = store.append_transaction(
            Transaction(D(2025, 1, 2), A, T.BUY, Decimal("200"), Decimal("9.85")))
        live = store.load_transactions()
        assert [t.id for t in live] == [fixed.id]
        assert derive_positions(live)[A].quantity == Decimal("200")

    def test_pence_row_round_trips_as_pounds(self, store):
        store.append_transaction(Transaction(D(2025, 1, 1), A, T.BUY, Decimal("90"),
                                             Decimal("742.50"), "GBp"))
        back = store.load_transactions()[0]
        assert back.currency == "GBP" and back.price_per_unit == Decimal("7.4250")

    def test_missing_files_read_as_empty_not_an_error(self, tmp_path):
        fresh = DataStore.open(DataMode.USER, tmp_path)
        assert fresh.load_transactions() == [] and fresh.load_instruments() == {}


class TestShippedSeedData:
    """The committed seed set must actually load and derive."""

    @pytest.fixture
    def seed(self) -> DataStore:
        return DataStore.open(DataMode.SEED, DEFAULT_ROOT)

    def test_ten_instruments(self, seed):
        assert len(seed.load_instruments()) == 10

    def test_asset_classes_split_seven_three(self, seed):
        classes = [i.asset_class for i in seed.load_instruments().values()]
        assert classes.count(AssetClass.ETF) == 7
        assert classes.count(AssetClass.ETC) == 3

    def test_six_are_usd_base(self, seed):
        """FX is the majority case in this set, not an edge case."""
        insts = seed.load_instruments().values()
        assert sum(1 for i in insts if i.base_currency == "USD") == 6
        assert sum(1 for i in insts if i.base_currency == "EUR") == 4

    def test_the_voided_row_is_excluded_from_the_live_ledger(self, seed):
        assert len(seed.load_transactions(include_voided=True)) == \
            len(seed.load_transactions()) + 1

    def test_seed_ledger_derives_without_error(self, seed, fx):
        positions = derive_positions(seed.load_transactions(),
                                     seed.load_instruments(), rates=fx)
        assert len(positions) == 10
        held = [p for p in positions.values() if p.is_open]
        watch = [p for p in positions.values() if p.is_watchlist]
        assert held, "seed data should contain open positions"
        assert watch, "seed data should contain watchlist entries"

    def test_seed_contains_a_closed_position(self, seed, fx):
        positions = derive_positions(seed.load_transactions(),
                                     seed.load_instruments(), rates=fx)
        closed = [p for p in positions.values()
                  if not p.is_open and not p.is_watchlist]
        assert closed, "seed data should exercise the fully-exited path"
