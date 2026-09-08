"""Display names: derived, distinct, and overridable.

The doctests in `core/naming` cover the derivation rules case by case. What is
tested here is the part that cannot be shown one name at a time -- that a set
of names stays mutually distinguishable -- and the round trip through the
instrument and the store, since a label the user pinned must survive being
written to CSV and read back.
"""

from __future__ import annotations

import pytest

from portfolio.core.models import AssetClass, Instrument
from portfolio.core.naming import derive_issuer, short_names, shorten_name

A, B, C = "IE0002Y8CX98", "IE000IAXNM41", "IE00BMC38736"


def inst(isin: str, name: str, **kw) -> Instrument:
    return Instrument(isin, name, kw.pop("asset_class", AssetClass.ETF), "EUR", **kw)


class TestShortening:
    def test_wrapper_and_share_class_go_but_the_subject_stays(self):
        assert shorten_name("iShares Core MSCI World UCITS ETF USD Acc") == "Core MSCI World"

    def test_never_cuts_a_word_in_half(self):
        out = shorten_name("Amundi Index Solutions Global Aggregate Green Bond")
        assert out.endswith("…")
        # The ellipsis replaces whole words, so what remains is still words.
        assert all(w.strip("…") for w in out.split())
        assert "Aggre…" not in out

    def test_an_equity_keeps_its_identity(self):
        """The guard that stops the issuer strip emptying the name."""
        assert shorten_name("Schneider Electric SE", issuer="Schneider Electric") == \
            "Schneider Electric"

    def test_derivation_is_not_a_truncation(self):
        """The point of the whole module: the informative half is what survives."""
        full = "iShares MSCI Europe Industrials Sector UCITS ETF EUR Acc"
        assert shorten_name(full) == "MSCI Europe Industrials"
        assert full[:24] == "iShares MSCI Europe Indu"      # what truncation gives


class TestUniverseDistinctness:
    """The rule that one-at-a-time derivation cannot enforce."""

    def test_two_issuers_one_theme_stay_distinguishable(self):
        book = {A: inst(A, "WisdomTree Europe Defence UCITS ETF"),
                B: inst(B, "iShares Europe Defence UCITS ETF")}
        names = short_names(book)
        assert names[A] != names[B]
        assert names[A] == "WisdomTree Europe Defence"
        assert names[B] == "iShares Europe Defence"

    def test_the_collision_rule_bites_only_on_collision(self):
        """Deliberate sabotage of the premise: remove the clash and the
        issuer must come straight back off, or the rule is firing always."""
        book = {A: inst(A, "WisdomTree Europe Defence UCITS ETF"),
                C: inst(C, "VanEck Semiconductor UCITS ETF")}
        names = short_names(book)
        assert names[A] == "Europe Defence", "no clash, so the issuer should go"
        assert names[C] == "Semiconductor"

    def test_identical_names_fall_back_to_the_key(self):
        """When the name genuinely cannot separate them, the ISIN does."""
        book = {A: inst(A, "Europe Defence UCITS ETF"),
                B: inst(B, "Europe Defence UCITS ETF")}
        names = short_names(book)
        assert names[A] != names[B]
        assert names[A].endswith(A[-4:]) and names[B].endswith(B[-4:])

    def test_a_pinned_name_is_never_rewritten(self):
        book = {A: inst(A, "WisdomTree Europe Defence UCITS ETF", short_name="Defence EU"),
                B: inst(B, "iShares Europe Defence UCITS ETF")}
        assert short_names(book)[A] == "Defence EU"

    def test_empty_universe_is_not_an_error(self):
        assert short_names({}) == {}


class TestIssuerBackfill:
    def test_a_blank_issuer_is_derived_from_the_name(self):
        assert inst(B, "iShares Europe Defence UCITS ETF", issuer="").issuer == "iShares"

    def test_an_equity_is_its_own_issuer(self):
        e = inst("FR0000121972", "Schneider Electric SE",
                 asset_class=AssetClass.EQUITY, issuer="")
        assert e.issuer == "Schneider Electric"

    def test_an_unrecognised_name_is_left_blank_rather_than_guessed(self):
        assert derive_issuer("Some Unbranded Tracker") == ""
        assert inst(A, "Some Unbranded Tracker", issuer="").issuer == ""

    def test_a_recorded_issuer_is_never_overwritten(self):
        assert inst(A, "iShares Europe Defence", issuer="BlackRock").issuer == "BlackRock"


class TestInstrumentField:
    def test_display_name_derives_when_unset(self):
        assert inst(C, "VanEck Semiconductor UCITS ETF").display_name == "Semiconductor"

    def test_a_pinned_short_name_wins(self):
        assert inst(C, "VanEck Semiconductor UCITS ETF",
                    short_name="Chips").display_name == "Chips"

    def test_overriding_marks_the_field_protected(self):
        i = inst(C, "VanEck Semiconductor UCITS ETF")
        i.override("short_name", "Chips")
        assert i.is_overridden("short_name") and i.display_name == "Chips"

    def test_a_derived_label_follows_the_name_it_was_derived_from(self):
        """Nothing is stored, so renaming the fund relabels it too."""
        i = inst(C, "VanEck Semiconductor UCITS ETF")
        i.name = "VanEck Defense UCITS ETF"
        assert i.display_name == "Defense"

    def test_a_pinned_label_does_not_follow_the_name(self):
        i = inst(C, "VanEck Semiconductor UCITS ETF")
        i.override("short_name", "Chips")
        i.name = "VanEck Defense UCITS ETF"
        assert i.display_name == "Chips", "a pinned label is the user's, not the data's"


class TestStoreRoundTrip:
    def test_short_name_survives_a_write_and_read(self, tmp_path):
        from portfolio.data.store import DataMode, DataStore
        store = DataStore(mode=DataMode.USER, root=tmp_path)
        store.directory.mkdir(parents=True, exist_ok=True)
        i = inst(C, "VanEck Semiconductor UCITS ETF")
        i.override("short_name", "Chips")
        store.save_instruments({C: i})
        back = store.load_instruments()[C]
        assert back.short_name == "Chips" and back.is_overridden("short_name")

    def test_a_csv_without_the_column_still_loads(self, tmp_path):
        """Backward compatibility: an older ledger must not fail to open."""
        from portfolio.data.store import DataMode, DataStore
        store = DataStore(mode=DataMode.USER, root=tmp_path)
        store.directory.mkdir(parents=True, exist_ok=True)
        store.instruments_path.write_text(
            "isin,name,issuer,asset_class,base_currency\n"
            f"{C},VanEck Semiconductor UCITS ETF,VanEck,ETF,EUR\n", encoding="utf-8")
        back = store.load_instruments()[C]
        assert back.short_name == ""
        assert back.display_name == "Semiconductor", "a missing column means derive"


@pytest.mark.parametrize("name,expected", [
    ("iShares EURO STOXX Banks 30-15 UCITS ETF DE EUR Acc", "EURO STOXX Banks 30-15"),
    ("Invesco Physical Gold ETC", "Physical Gold"),
    ("iShares Core MSCI EM IMI UCITS ETF USD Acc", "Core MSCI EM IMI"),
    ("iShares European Property Yield UCITS ETF EUR Acc", "European Property Yield"),
    ("VanEck Semiconductor UCITS ETF", "Semiconductor"),
])
def test_the_real_portfolio(name, expected):
    """The seven instruments this was built against, all inside the limit."""
    out = shorten_name(name)
    assert out == expected
    assert len(out) <= 24 and "…" not in out
