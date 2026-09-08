"""Recording a broker, a tax band or a freeze, without editing a CSV by hand.

These fields decide whether a trade can be priced at all, and until this
command existed the only way to populate them was to add seven columns across
every row of a spreadsheet -- on a machine whose locale uses a comma as the
decimal separator. That is not a chore, it is a mechanism for getting a
plausible wrong number into a cost model, which is the failure this project
keeps finding in other guises.

So the value goes through a parser that accepts every way a person actually
writes a rate and refuses the one that is ambiguous, and the flags that claim
a number was read off a document cannot be set without the number.
"""

from __future__ import annotations

import argparse

import pytest

from portfolio.cli import _rate, main
from portfolio.data.store import DataMode, DataStore

BANKS = "DE000A2QP372"
PROPERTY = "IE00BGDQ0L74"
SCHNEIDER = "FR0000121972"
GOLD = "IE00B579F325"

BOOK = (
    "isin,name,issuer,asset_class,base_currency,primary_symbol,exchange,"
    "quote_currency,provider_symbols,active,manual_overrides,note\n"
    f"{BANKS},iShares EURO STOXX Banks 30-15 UCITS ETF DE EUR ACC,iShares,ETF,"
    "EUR,EXX1,XAMS,EUR,yfinance=EXX1.AS,true,,\n"
    f"{PROPERTY},iShares European Property Yield UCITS ETF EUR Dist,iShares,"
    "ETF,EUR,IPRP,XAMS,EUR,yfinance=IPRP.AS,true,,\n"
    f"{SCHNEIDER},Schneider Electric SE,,EQUITY,EUR,SU,XAMS,EUR,"
    "yfinance=SU.AS,true,,\n"
    f"{GOLD},Invesco Physical Gold ETC,Invesco,ETC,EUR,SGLD,XAMS,EUR,"
    "yfinance=SGLD.AS,true,,\n"
)


@pytest.fixture()
def root(tmp_path):
    store = DataStore(mode=DataMode.USER, root=tmp_path)
    store.directory.mkdir(parents=True, exist_ok=True)
    store.instruments_path.write_text(BOOK, encoding="utf-8")
    return tmp_path


def run(root, *args) -> int:
    return main(["instruments", *args, "--mode", "user",
                 "--data-root", str(root)])


def book(root) -> dict:
    return DataStore(mode=DataMode.USER, root=root).load_instruments()


class TestTheRateParser:
    @pytest.mark.parametrize("text", ["0.0012", "0,0012", "0.12%", "0,12 %",
                                      " 0.12% ", "0,120%"])
    def test_every_way_a_person_writes_twelve_basis_points(self, text):
        assert _rate(text) == pytest.approx(0.0012)

    def test_the_ambiguous_one_is_refused(self):
        """`--tob-rate 0.12` meaning 0.12% would silently charge 12%.

        A hundredfold error in the direction that kills every strategy, from
        a keystroke, with nothing to notice it by. Refusing costs one retype.
        """
        with pytest.raises(argparse.ArgumentTypeError, match="12.0000%"):
            _rate("0.12")

    def test_a_percentage_sign_makes_it_unambiguous(self):
        assert _rate("1.32%") == pytest.approx(0.0132)
        assert _rate("0.0132") == pytest.approx(0.0132)

    def test_nonsense_is_refused(self):
        with pytest.raises(argparse.ArgumentTypeError, match="not a number"):
            _rate("twelve")
        with pytest.raises(argparse.ArgumentTypeError, match="negative"):
            _rate("-0.001")


class TestRecordingTheAccountsFacts:
    def test_one_command_sets_the_broker_on_every_holding(self, root):
        assert run(root, "set", "--all", "--broker", "MeDirect") == 0
        assert all(i.broker == "MeDirect" for i in book(root).values())

    def test_a_holding_can_be_frozen(self, root):
        assert run(root, "set", GOLD, "--broker", "Keytrade",
                   "--not-tradeable") == 0
        after = book(root)
        assert after[GOLD].broker == "Keytrade" and not after[GOLD].tradeable
        assert all(after[i].tradeable for i in (BANKS, PROPERTY, SCHNEIDER))

    def test_a_rate_read_off_a_document_can_be_marked_as_one(self, root):
        assert run(root, "set", BANKS, "--tob-rate", "0.12%", "--observed") == 0
        assert book(root)[BANKS].tob_rate == pytest.approx(0.0012)
        assert book(root)[BANKS].tob_observed

    def test_a_new_rate_with_no_stated_source_is_an_assumption(self, root):
        """Not a silent carry-over of the previous claim."""
        run(root, "set", BANKS, "--tob-rate", "0.12%", "--observed")
        run(root, "set", BANKS, "--tob-rate", "0.35%")
        assert book(root)[BANKS].tob_rate == pytest.approx(0.0035)
        assert not book(root)[BANKS].tob_observed

    def test_the_two_commands_reproduce_the_account_as_described(self, root):
        """The whole book, from a file that predates these columns, in two
        commands and no hand editing."""
        run(root, "list")                       # migrates on load
        assert run(root, "set", "--all", "--broker", "MeDirect") == 0
        assert run(root, "set", GOLD, "--broker", "Keytrade",
                   "--not-tradeable") == 0
        after = book(root)
        for isin in (BANKS, PROPERTY):
            assert after[isin].tob_rate == pytest.approx(0.0012)
            assert after[isin].broker == "MeDirect" and after[isin].tradeable
        assert after[SCHNEIDER].tob_rate == pytest.approx(0.0035)
        assert after[SCHNEIDER].buy_tax_rate == pytest.approx(0.004)
        assert after[GOLD].tob_rate is None
        assert after[GOLD].broker == "Keytrade" and not after[GOLD].tradeable
        assert {i for i, v in after.items() if v.tob_observed} == {BANKS}

    def test_an_edit_is_marked_as_a_manual_override(self, root):
        """So a later automatic re-resolution cannot quietly revert it."""
        run(root, "set", GOLD, "--broker", "Keytrade")
        assert book(root)[GOLD].is_overridden("broker")


class TestWhatItRefuses:
    def test_observed_without_a_rate(self, root, capsys):
        assert run(root, "set", GOLD, "--observed") == 2
        assert "needs that rate" in capsys.readouterr().err
        assert not book(root)[GOLD].tob_observed

    def test_spread_observed_without_a_spread(self, root, capsys):
        assert run(root, "set", BANKS, "--spread-observed") == 2
        assert "cannot be observed without a value" in capsys.readouterr().err

    def test_an_unknown_isin(self, root, capsys):
        assert run(root, "set", "IE00B579F324", "--broker", "MeDirect") == 2
        assert "not in" in capsys.readouterr().err

    def test_naming_nothing_at_all(self, root, capsys):
        assert run(root, "set", "--broker", "MeDirect") == 2
        assert "at least one ISIN" in capsys.readouterr().err

    def test_nothing_is_written_when_nothing_changes(self, root, capsys):
        run(root, "set", "--all", "--broker", "MeDirect")
        before = (root / "user" / "instruments.csv").read_bytes()
        assert run(root, "set", "--all", "--broker", "MeDirect") == 0
        assert "nothing to change" in capsys.readouterr().out
        assert (root / "user" / "instruments.csv").read_bytes() == before

    def test_a_broker_with_no_fee_schedule_is_flagged(self, root, capsys):
        """Saved, because the user may know something the model does not, but
        never quietly: the cost model will refuse to price a trade there."""
        assert run(root, "set", GOLD, "--broker", "Bolero") == 0
        assert "no fee schedule is recorded for Bolero" in capsys.readouterr().out


class TestListing:
    def test_it_shows_what_is_evidence_and_what_is_not(self, root, capsys):
        assert run(root, "list") == 0
        out = capsys.readouterr().out
        assert "0.1200% observed" in out          # the contract-note rate
        assert "0.1200% assumed" in out           # derived from the asset class
        assert "NOT RECORDED" in out              # the ETC
        assert "8.0 bps fallback" in out          # no quotes have been read

    def test_a_frozen_holding_is_visible_as_such(self, root, capsys):
        run(root, "set", GOLD, "--not-tradeable")
        capsys.readouterr()
        run(root, "list")
        line = next(ln for ln in capsys.readouterr().out.splitlines()
                    if ln.startswith(GOLD))
        assert " NO " in line

    def test_an_empty_store_says_so(self, tmp_path, capsys):
        assert main(["instruments", "list", "--mode", "user",
                     "--data-root", str(tmp_path)]) == 2
        assert "no instruments in" in capsys.readouterr().err
