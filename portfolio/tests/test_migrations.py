"""The schema migration, and the failure that made it necessary.

The bug it fixes was not a wrong line of code. `INSTRUMENT_COLUMNS` gained
seven trading-cost fields, appended so an older CSV would still load. It did
load. Every new field took its default, `tob_rate` defaults to None meaning
NOT RECORDED, and the cost model refuses to price a trade in an instrument
whose tax band is not recorded. Graceful loading and refusing to guess, each
correct on its own, composed into a tool that could not run at all on any
book written by the previous build.

No test caught it because no test loaded an old file and then tried to price
a trade from it. `TestTheFailureThisFixes` is that test, written to fail
against the code as it was.

The rest of this file is about the migration not doing too much. Filling a
column that is absent is helpful; filling a cell that is deliberately blank
would reintroduce the exact plausible-wrong-number failure the column was
added to prevent, which is why the trigger is the header and not the value.
"""

from __future__ import annotations

import pytest

from portfolio.agents.execution import CostModel, UnknownCost, cost_table
from portfolio.core.models import AssetClass
from portfolio.data.migrations import (OBSERVED_CONTRACT_NOTES,
                                       TRADING_COLUMNS, derived_facts,
                                       missing_columns)
from portfolio.data.store import DataMode, DataStore

BANKS = "DE000A2QP372"          # the one instrument with a contract note
PROPERTY = "IE00BGDQ0L74"       # a fund, held at the same broker
SEMIS = "IE00BMC38736"
SCHNEIDER = "FR0000121972"      # a French share: another band, plus the FTT
GOLD = "IE00B579F325"           # an ETC, and at a second broker

# A book in the shape the previous build wrote: no trading columns at all.
OLD_SCHEMA = (
    "isin,name,issuer,asset_class,base_currency,primary_symbol,exchange,"
    "quote_currency,provider_symbols,active,manual_overrides,note\n"
    f"{BANKS},iShares EURO STOXX Banks 30-15 UCITS ETF DE EUR ACC,iShares,ETF,"
    "EUR,EXX1,XAMS,EUR,yfinance=EXX1.AS,true,,\n"
    f"{PROPERTY},iShares European Property Yield UCITS ETF EUR Dist,iShares,"
    "ETF,EUR,IPRP,XAMS,EUR,yfinance=IPRP.AS,true,,\n"
    f"{SEMIS},VanEck Semiconductor UCITS ETF,VanEck,ETF,EUR,SMH,XAMS,EUR,"
    "yfinance=SMH.AS,true,,\n"
    f"{SCHNEIDER},Schneider Electric SE,,EQUITY,EUR,SU,XAMS,EUR,"
    "yfinance=SU.AS,true,,\n"
    f"{GOLD},Invesco Physical Gold ETC,Invesco,ETC,EUR,SGLD,XAMS,EUR,"
    "yfinance=SGLD.AS,true,,\n"
)

LEDGER = (
    "id,date,isin,type,quantity,price_per_unit,currency,fees,note\n"
    f"t1,2025-02-03,{BANKS},BUY,114,17.76,EUR,0,\n"
    f"t2,2025-02-10,{PROPERTY},BUY,60,25.10,EUR,0,\n"
    f"t3,2025-03-04,{SEMIS},BUY,32,38.40,EUR,0,\n"
    f"t4,2025-04-15,{SCHNEIDER},BUY,12,215.00,EUR,0,\n"
    f"t5,2025-05-06,{GOLD},BUY,45,26.50,EUR,0,\n"
)


def _refuse_to_write(self, instruments):
    raise OSError(30, "Read-only file system")


def store_with(tmp_path, instruments: str, ledger: str = "") -> DataStore:
    store = DataStore(mode=DataMode.USER, root=tmp_path)
    store.directory.mkdir(parents=True, exist_ok=True)
    store.instruments_path.write_text(instruments, encoding="utf-8")
    if ledger:
        store.transactions_path.write_text(ledger, encoding="utf-8")
    return store


def pre_migration_book() -> dict:
    """Exactly what the previous loader produced from this file.

    The row's own columns, and the dataclass default for every column the file
    does not have -- which for `tob_rate` is None, meaning NOT RECORDED.
    """
    import csv

    from portfolio.core.models import Instrument
    return {row["isin"]: Instrument(
                isin=row["isin"], name=row["name"], issuer=row["issuer"],
                asset_class=AssetClass(row["asset_class"]),
                base_currency=row["base_currency"])
            for row in csv.DictReader(OLD_SCHEMA.splitlines())}


class TestTheFailureThisFixes:
    """Written against the behaviour as reported, not as imagined."""

    def test_every_instrument_loaded_unpriceable(self):
        """The bug, and its real size: not one holding, every one of them.

        The property ETF was simply the first the optimiser happened to reach,
        which is why the report named it and why it looked like a fact about
        that fund.
        """
        book = pre_migration_book()
        assert len(book) == 5
        assert all(inst.tob_rate is None for inst in book.values())
        model = CostModel(per_instrument=cost_table(book))
        for isin in book:
            with pytest.raises(UnknownCost):
                model.instrument_cost(isin, 1000.0, "buy")

    def test_after_migration_every_fund_prices_and_only_the_etc_does_not(
            self, tmp_path):
        store = store_with(tmp_path, OLD_SCHEMA)
        model = CostModel(per_instrument=cost_table(store.load_instruments()))
        for isin in (BANKS, PROPERTY, SEMIS, SCHNEIDER):
            assert model.instrument_cost(isin, 1000.0, "buy") > 0
        with pytest.raises(UnknownCost, match="debt security"):
            model.instrument_cost(GOLD, 1000.0, "buy")


class TestWhatItFillsIn:
    @pytest.fixture()
    def book(self, tmp_path):
        return store_with(tmp_path, OLD_SCHEMA).load_instruments()

    def test_funds_take_the_fund_band(self, book):
        for isin in (PROPERTY, SEMIS):
            assert book[isin].tob_rate == pytest.approx(0.0012)
            assert not book[isin].tob_observed

    def test_the_share_takes_the_equity_band_and_the_french_tax(self, book):
        assert book[SCHNEIDER].tob_rate == pytest.approx(0.0035)
        assert book[SCHNEIDER].buy_tax_rate == pytest.approx(0.004)

    def test_the_etc_takes_nothing(self, book):
        """A debt security does not inherit a fund's band, and no contract
        note for this one has been read. Unpriceable is the honest state."""
        assert book[GOLD].tob_rate is None
        assert not book[GOLD].tob_observed

    def test_only_the_instrument_with_a_contract_note_is_marked_observed(self, book):
        observed = {i for i, inst in book.items() if inst.tob_observed}
        assert observed == {BANKS}
        assert set(OBSERVED_CONTRACT_NOTES) == {BANKS}

    def test_it_does_not_invent_a_broker(self, book):
        """Which institution holds a position is not on the instrument row.

        Guessing it would be worse than leaving it blank: the two brokers here
        have opposite fee shapes, so the wrong schedule changes which trades
        are affordable rather than changing a cost slightly.
        """
        assert all(not inst.broker for inst in book.values())

    def test_it_does_not_invent_a_spread(self, book):
        assert all(inst.half_spread_bps is None for inst in book.values())

    def test_the_derived_values_are_reported_as_assumptions(self, tmp_path):
        store = store_with(tmp_path, OLD_SCHEMA)
        model = CostModel(per_instrument=cost_table(store.load_instruments()))
        notes = model.assumptions()
        assert any(n.startswith(f"{PROPERTY}: transaction tax 0.1200% assumed")
                   for n in notes)
        assert any("no broker recorded" in n for n in notes)
        assert not [n for n in notes
                    if n.startswith(BANKS) and "transaction tax" in n], (
            "the rate read off a contract note was listed as an assumption")


class TestWhatItRefusesToTouch:
    """The half that matters. A migration that overwrites a recorded fact is
    worse than no migration, because the fact it destroys is the one somebody
    took the trouble to establish."""

    def test_a_blank_cell_under_a_present_column_survives(self, tmp_path):
        """The gold case. `tob_rate` is in the header and empty, which says
        "not established" -- a statement, not an absence."""
        text = OLD_SCHEMA.replace("note\n", "note,tob_rate\n").replace(
            ",true,,\n", ",true,,,\n")
        store = store_with(tmp_path, text)
        assert "tob_rate" not in missing_columns(store.instruments_path)
        book = store.load_instruments()
        assert all(inst.tob_rate is None for inst in book.values()), (
            "the migration filled a column the file already had, so a "
            "deliberately unrecorded rate would be overwritten with a guess")

    def test_a_recorded_rate_is_never_replaced(self, tmp_path):
        text = OLD_SCHEMA.replace("note\n", "note,tob_rate\n")
        text = text.replace(",true,,\n", ",true,,,0.0132\n")
        store = store_with(tmp_path, text)
        assert all(inst.tob_rate == pytest.approx(0.0132)
                   for inst in store.load_instruments().values())

    def test_an_observed_flag_is_never_set_without_its_rate(self, tmp_path):
        """A flag saying "read off a contract note" with no rate behind it is
        an estimate wearing the label of evidence."""
        text = OLD_SCHEMA.replace("note\n", "note,tob_rate\n").replace(
            ",true,,\n", ",true,,,\n")
        book = store_with(tmp_path, text).load_instruments()
        assert not any(inst.tob_observed for inst in book.values())

    def test_a_current_file_is_left_completely_alone(self, tmp_path):
        store = store_with(tmp_path, OLD_SCHEMA)
        store.load_instruments()                       # migrates
        before = store.instruments_path.read_bytes()
        store.last_migration = None
        store.load_instruments()                       # must do nothing
        assert store.last_migration is None
        assert store.instruments_path.read_bytes() == before

    def test_the_trigger_is_exactly_a_missing_column(self, tmp_path):
        store = store_with(tmp_path, OLD_SCHEMA)
        assert missing_columns(store.instruments_path) == list(TRADING_COLUMNS)
        store.load_instruments()
        assert missing_columns(store.instruments_path) == []


class TestItPreservesWhatItRewrites:
    def test_the_original_file_is_kept(self, tmp_path):
        store = store_with(tmp_path, OLD_SCHEMA)
        store.load_instruments()
        backup = store.instruments_path.parent / (
            store.instruments_path.name + DataStore.BACKUP_SUFFIX)
        assert backup.read_text(encoding="utf-8") == OLD_SCHEMA

    def test_every_original_field_survives_the_rewrite(self, tmp_path):
        store = store_with(tmp_path, OLD_SCHEMA)
        before = store.load_instruments()
        after = DataStore(mode=DataMode.USER, root=tmp_path).load_instruments()
        assert set(before) == set(after)
        for isin in before:
            for field in ("name", "issuer", "asset_class", "base_currency",
                          "primary_symbol", "exchange", "provider_symbols",
                          "active", "tob_rate", "buy_tax_rate"):
                assert getattr(before[isin], field) == getattr(after[isin], field)

    def test_a_read_only_directory_degrades_rather_than_failing(
            self, tmp_path, monkeypatch):
        """A book on a read-only volume must still open. The derived values
        apply for the run; they are simply derived again next time.

        Driven by making the write fail rather than by changing the directory
        mode, because the suite runs as root in CI and root ignores the mode
        -- the check would pass without exercising anything.
        """
        store = store_with(tmp_path, OLD_SCHEMA)
        monkeypatch.setattr(DataStore, "save_instruments", _refuse_to_write)
        book = store.load_instruments()
        assert book[PROPERTY].tob_rate == pytest.approx(0.0012)
        assert store.last_migration is not None
        assert not store.last_migration.persisted
        assert "NOT WRITTEN BACK" in "\n".join(store.last_migration.lines())

    def test_the_notice_names_what_it_did(self, tmp_path, capsys):
        store = store_with(tmp_path, OLD_SCHEMA)
        store.load_instruments()
        printed = capsys.readouterr().err
        assert "Migrated" in printed
        assert f"{SCHNEIDER} buy_tax_rate = 0.004" in printed
        assert GOLD in printed and "Still not priceable" in printed
        assert "portfolio instruments set" in printed


class TestDerivation:
    """The table, checked directly, so a change to it has to be deliberate."""

    @pytest.mark.parametrize("asset_class,expected", [
        (AssetClass.ETF, 0.0012),
        (AssetClass.FUND, 0.0012),
        (AssetClass.EQUITY, 0.0035),
        (AssetClass.ETC, None),
        (AssetClass.OTHER, None),
    ])
    def test_the_band_follows_the_asset_class(self, asset_class, expected):
        from portfolio.core.models import Instrument
        inst = Instrument(PROPERTY, "Something", asset_class)
        rates = [b.value for b in derived_facts(inst) if b.field == "tob_rate"]
        assert rates == ([expected] if expected is not None else [])

    def test_the_french_tax_is_only_for_french_shares(self):
        from portfolio.core.models import Instrument
        french = Instrument(SCHNEIDER, "Schneider Electric SE", AssetClass.EQUITY)
        dutch = Instrument("NL0011794037", "Koninklijke Ahold Delhaize NV",
                           AssetClass.EQUITY)
        assert any(b.field == "buy_tax_rate" for b in derived_facts(french))
        assert not any(b.field == "buy_tax_rate" for b in derived_facts(dutch))

    def test_a_french_fund_is_not_a_french_share(self):
        """The FTT is on acquisitions of shares, not of fund units."""
        from portfolio.core.models import Instrument
        fund = Instrument("FR0010315770", "Lyxor Core STOXX Europe 600",
                          AssetClass.ETF)
        assert not any(b.field == "buy_tax_rate" for b in derived_facts(fund))


@pytest.fixture(scope="module")
def book(tmp_path_factory):
    from portfolio.research import load_book
    root = tmp_path_factory.mktemp("book")
    store_with(root, OLD_SCHEMA, LEDGER)
    return load_book(mode="user", data_root=root, provider="fixture",
                     lookback=400)


class TestTheBookLoadsAndRuns:
    """End to end, on the fixture provider: the thing the user could not do.

    `research.load_book` had no test at all, which is the reason this defect
    reached a real book. A composition root nothing exercises is where two
    correct components get to compose into a broken one.
    """

    def test_it_loads(self, book):
        assert set(book.panel.closes.columns) == {
            BANKS, PROPERTY, SEMIS, SCHNEIDER, GOLD}

    def test_the_instrument_with_no_band_is_frozen_rather_than_fatal(self, book):
        """Refusing to price it stays absolute; refusing to run does not.

        A holding whose cost cannot be established is exactly a holding no
        policy may trade, and that is a constraint the harness already models.
        Raising instead meant one unrecorded rate produced no result at all.
        """
        assert GOLD in book.frozen
        assert GOLD not in book.tradeable
        assert set(book.tradeable) == {BANKS, PROPERTY, SEMIS, SCHNEIDER}

    def test_it_says_why_that_holding_is_frozen(self, book):
        note = next(n for n in book.notes if GOLD in n)
        assert "no transaction tax rate is recorded" in note
        assert "gap in the record rather than a fact about the account" in note
        assert "portfolio instruments set" in note

    def test_the_backtest_produces_a_result(self, book):
        from portfolio.research import run_equal_risk_contribution
        report = run_equal_risk_contribution(book, lookback=126,
                                             rebalance_every=21)
        assert report.rebalances > 0
        assert report.net.observations > 0
        # Turnover, not just a report: with every rate missing the policy has
        # nothing it may trade, produces a result full of zeros, and would
        # otherwise satisfy the two assertions above.
        assert report.mean_turnover > 0
        assert report.total_cost > 0

    def test_the_result_says_how_much_of_its_cost_is_evidence(self, book):
        from portfolio.research import run_equal_risk_contribution
        text = "\n".join(run_equal_risk_contribution(
            book, lookback=126, rebalance_every=21).lines())
        assert "How much of the cost figure rests on observed inputs" in text
        assert "1 of 5 rates was read off a contract note" in text
        assert "of the book by value" in text


class TestTheNoticeClaimsOnlyWhatApplies:
    """The same rule as the refusal message: say what is known, not what was
    true of the case the sentence was written for."""

    def test_a_file_missing_only_a_broker_column_is_not_called_unpriceable(
            self, tmp_path):
        text = OLD_SCHEMA.replace(
            "note\n", "note,tradeable,tob_rate,tob_observed,half_spread_bps,"
                      "spread_observed,buy_tax_rate,buyable\n")
        text = text.replace(",true,,\n",
                            ",true,,,true,0.0012,true,,false,0,true\n")
        store = store_with(tmp_path, text)
        store.load_instruments()
        printed = "\n".join(store.last_migration.lines())
        assert store.last_migration.columns_added == ("broker",)
        assert "no backtest could run" not in printed
        assert "missing the broker column" in printed

    def test_a_file_missing_the_rate_column_is(self, tmp_path):
        store = store_with(tmp_path, OLD_SCHEMA)
        store.load_instruments()
        printed = "\n".join(store.last_migration.lines())
        assert "tob_rate" in store.last_migration.columns_added
        assert "no backtest could run" in printed
