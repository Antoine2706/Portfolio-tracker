"""CSV import: an English export, a German one, and the rows that go wrong.

The German file is the important one. Semicolons, dd.mm.yyyy, "1.234,56",
"Kauf"/"Verkauf" and a negative quantity on the sell are what a real
continental broker exports, and each of those has silently corrupted an
import somewhere."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from portfolio.core.models import Transaction, TransactionType as T
from portfolio.data.importers import (ColumnMapping, ImportPreview, ImportRow,
                                      detect_mapping, mark_duplicates, normalise_type,
                                      parse_date, parse_number, preview_import)

D = dt.date
A, B = "IE0002Y8CX98", "IE000IAXNM41"

ENGLISH = """Date,ISIN,Type,Quantity,Price,Currency,Fees,Note
2025-01-15,IE0002Y8CX98,Buy,120,22.40,EUR,3.90,first buy
2025-04-02,IE0002Y8CX98,Sell,-20,27.15,EUR,3.90,
2025-05-02,IE000IAXNM41,Dividend,,12.50,EUR,,distribution
"""

GERMAN = """Datum;ISIN;Typ;Anzahl;Kurs;Währung;Gebühren
15.01.2025;IE0002Y8CX98;Kauf;10;1.234,56;EUR;3,90
16.01.2025;IE0002Y8CX98;Verkauf;-5;1234,56;EUR;3,90
17.01.2025;IE000IAXNM41;Ausschüttung;0;12,5;EUR;0
"""


class TestDetectMapping:
    def test_english_headers(self):
        m = detect_mapping(["Date", "ISIN", "Type", "Quantity", "Price", "Currency",
                            "Fees", "Note"])
        assert m == ColumnMapping("Date", "ISIN", "Type", "Quantity", "Price",
                                  "Currency", "Fees", "Note")

    def test_synonyms_and_case(self):
        m = detect_mapping(["TRADE DATE", "isin", "Side", "Shares", "Unit Price",
                            "CCY", "Commission", "Description"])
        assert (m.date, m.type, m.quantity, m.price) == ("TRADE DATE", "Side", "Shares",
                                                           "Unit Price")
        assert (m.currency, m.fees, m.note) == ("CCY", "Commission", "Description")

    def test_german_headers_with_umlauts(self):
        m = detect_mapping(["Datum", "ISIN", "Typ", "Anzahl", "Kurs", "Währung", "Gebühren"])
        assert (m.currency, m.fees, m.quantity, m.price) == ("Währung", "Gebühren",
                                                              "Anzahl", "Kurs")
        assert m.note is None

    def test_unknown_headers_stay_none(self):
        m = detect_mapping(["foo", "bar"])
        assert m.missing_required() == ["date", "isin", "type", "quantity", "price"]

    def test_exact_match_beats_substring(self):
        """'Trade Date' must win over 'Settlement Date' whatever their order."""
        m = detect_mapping(["Settlement Date", "Trade Date", "ISIN"])
        assert m.date == "Trade Date"

    def test_a_header_is_claimed_once(self):
        m = detect_mapping(["Price", "Price Currency", "Currency"])
        assert m.price == "Price" and m.currency == "Currency"


class TestCellParsing:
    @pytest.mark.parametrize("text,flag,expected", [
        ("1,234.56", False, "1234.56"), ("1.234,56", False, "1234.56"),
        ("1234,56", False, "1234.56"), ("3,90", False, "3.90"), ("-12,5", False, "-12.5"),
        ("1,234", False, "1234"), ("1,234", True, "1.234"),
        ("1.5", True, "1.5"), ("3.900", True, "3900"), ("1.234.567", True, "1234567"),
        ("1,234,567.89", False, "1234567.89"), ("12", False, "12"), (" 7 ", False, "7"),
        ("1 234,56", False, "1234.56"), ("22.40 EUR", False, "22.40"),
    ])
    def test_numbers(self, text, flag, expected):
        assert parse_number(text, flag) == Decimal(expected)

    def test_not_a_number_raises(self):
        from decimal import InvalidOperation
        with pytest.raises(InvalidOperation):
            parse_number("abc")

    @pytest.mark.parametrize("text,expected", [
        ("2025-01-15", D(2025, 1, 15)), ("2025-01-15T10:30:00", D(2025, 1, 15)),
        ("15/01/2025", D(2025, 1, 15)), ("15.01.2025", D(2025, 1, 15)),
        ("15-01-2025", D(2025, 1, 15)), ("15.01.25", D(2025, 1, 15)),
    ])
    def test_dates(self, text, expected):
        assert parse_date(text) == expected

    def test_explicit_format_settles_month_first(self):
        assert parse_date("01/15/2025", "%m/%d/%Y") == D(2025, 1, 15)

    def test_unrecognised_date_is_a_sentence(self):
        with pytest.raises(ValueError) as exc:
            parse_date("yesterday")
        assert "not a recognised date" in str(exc.value)

    @pytest.mark.parametrize("text,expected", [
        ("Buy", T.BUY), ("KAUF", T.BUY), ("achat", T.BUY), ("Purchase", T.BUY),
        ("Sell", T.SELL), ("Verkauf", T.SELL), ("vente", T.SELL),
        ("Dividend", T.DIVIDEND), ("div", T.DIVIDEND), ("Ausschüttung", T.DIVIDEND),
        ("Fee", T.FEE), ("Custody charge", T.FEE),
        ("Kauf (Sparplan)", T.BUY), ("Verkauf Limit", T.SELL),
    ])
    def test_types(self, text, expected):
        assert normalise_type(text) is expected

    def test_unknown_type_is_a_sentence(self):
        with pytest.raises(ValueError) as exc:
            normalise_type("Transfer")
        assert "not a recognised transaction type" in str(exc.value)


class TestEnglishExport:
    def test_all_rows_valid(self):
        p = preview_import(ENGLISH)
        assert isinstance(p, ImportPreview)
        assert p.valid == 3 and p.invalid == 0
        assert p.headers == ["Date", "ISIN", "Type", "Quantity", "Price", "Currency",
                             "Fees", "Note"]
        assert p.mapping.decimal_comma is False

    def test_values(self):
        rows = preview_import(ENGLISH).rows
        buy = rows[0].transaction
        assert buy.date == D(2025, 1, 15) and buy.isin == A and buy.type is T.BUY
        assert buy.quantity == Decimal("120") and buy.price_per_unit == Decimal("22.40")
        assert buy.fees == Decimal("3.90") and buy.note == "first buy"
        assert buy.currency == "EUR"

    def test_negative_quantity_on_a_sell_becomes_positive(self):
        sell = preview_import(ENGLISH).rows[1].transaction
        assert sell.type is T.SELL and sell.quantity == Decimal("20")

    def test_dividend_with_blank_quantity_and_fees(self):
        div = preview_import(ENGLISH).rows[2].transaction
        assert div.type is T.DIVIDEND and div.quantity == 0
        assert div.price_per_unit == Decimal("12.50") and div.fees == 0

    def test_line_numbers_count_the_header(self):
        assert [r.line for r in preview_import(ENGLISH).rows] == [2, 3, 4]

    def test_raw_cells_are_kept_for_display(self):
        row = preview_import(ENGLISH).rows[0]
        assert isinstance(row, ImportRow) and row.raw["Price"] == "22.40"

    def test_transactions_helper_returns_only_ok_rows(self):
        p = preview_import(ENGLISH + "2025-06-01,IE0002Y8CX98,Buy,abc,1,EUR,0,\n")
        assert len(p.transactions()) == 3


class TestGermanExport:
    def test_semicolon_and_decimal_comma_are_detected(self):
        p = preview_import(GERMAN)
        assert p.valid == 3 and p.invalid == 0
        assert p.mapping.decimal_comma is True
        assert p.mapping.date == "Datum" and p.mapping.price == "Kurs"

    def test_values(self):
        rows = preview_import(GERMAN).rows
        buy, sell, div = (r.transaction for r in rows)
        assert buy.date == D(2025, 1, 15) and buy.type is T.BUY
        assert buy.price_per_unit == Decimal("1234.56") and buy.fees == Decimal("3.90")
        assert sell.type is T.SELL and sell.quantity == Decimal("5")
        assert sell.price_per_unit == Decimal("1234.56")
        assert div.type is T.DIVIDEND and div.price_per_unit == Decimal("12.5")

    def test_explicit_mapping_is_honoured_and_gaps_filled(self):
        given = ColumnMapping(date="Datum", date_format="%d.%m.%Y", decimal_comma=True)
        p = preview_import(GERMAN, given)
        assert p.valid == 3
        assert p.mapping.isin == "ISIN" and p.mapping.date_format == "%d.%m.%Y"

    def test_windows_line_endings_and_bom(self):
        text = "﻿" + GERMAN.replace("\n", "\r\n")
        assert preview_import(text).valid == 3


class TestBrokenRows:
    def test_bad_number_is_reported_on_its_row_only(self):
        text = ENGLISH + "2025-06-01,IE0002Y8CX98,Buy,abc,1,EUR,0,\n"
        p = preview_import(text)
        assert p.valid == 3 and p.invalid == 1
        bad = p.rows[3]
        assert bad.ok is False and bad.transaction is None and bad.line == 5
        assert "quantity" in bad.error and "'abc'" in bad.error
        assert "Traceback" not in bad.error

    def test_invalid_isin_uses_the_ledgers_own_sentence(self):
        text = ENGLISH + "2025-06-01,IE0002Y8CX97,Buy,1,1,EUR,0,\n"
        bad = preview_import(text).rows[3]
        assert not bad.ok and "not a valid ISIN" in bad.error

    def test_negative_buy_is_refused_by_the_model(self):
        text = ENGLISH + "2025-06-01,IE0002Y8CX98,Buy,-5,1,EUR,0,\n"
        bad = preview_import(text).rows[3]
        assert not bad.ok and "positive quantity" in bad.error

    def test_bad_date_and_bad_type(self):
        text = ENGLISH + ("yesterday,IE0002Y8CX98,Buy,1,1,EUR,0,\n"
                          "2025-06-01,IE0002Y8CX98,Transfer,1,1,EUR,0,\n")
        rows = preview_import(text).rows
        assert "not a recognised date" in rows[3].error
        assert "not a recognised transaction type" in rows[4].error

    def test_unmapped_required_column_marks_every_row_with_the_column_name(self):
        text = "When,ISIN,Type,Quantity,Price\n2025-01-15,IE0002Y8CX98,Buy,1,1\n"
        p = preview_import(text)
        assert p.valid == 0 and p.invalid == 1
        assert "no column identified for date" in p.rows[0].error
        assert p.headers == ["When", "ISIN", "Type", "Quantity", "Price"]

    def test_blank_lines_are_skipped_not_counted(self):
        text = ENGLISH.replace("\n2025-04-02", "\n\n,,,,,,,\n2025-04-02")
        p = preview_import(text)
        assert p.valid == 3 and p.invalid == 0

    def test_empty_input_is_an_empty_preview(self):
        p = preview_import("")
        assert p.rows == [] and p.valid == 0 and p.invalid == 0 and p.headers == []

    def test_short_row_defaults_the_missing_cells(self):
        text = "Date,ISIN,Type,Quantity,Price,Currency,Fees,Note\n2025-01-15,IE0002Y8CX98,Buy,1,2\n"
        row = preview_import(text).rows[0]
        assert row.ok and row.transaction.currency == "EUR" and row.transaction.fees == 0


class TestUnknownIsins:
    def test_listed_when_known_is_given_rows_stay_valid(self):
        p = preview_import(ENGLISH, known={A})
        assert p.unknown_isins == [B]
        assert p.valid == 3

    def test_empty_without_known(self):
        assert preview_import(ENGLISH).unknown_isins == []


class TestDuplicates:
    def test_existing_trade_is_flagged_with_its_id(self):
        existing = [Transaction(D(2025, 1, 15), A, T.BUY, Decimal("120"), Decimal("22.40"),
                                "EUR", Decimal("0"), note="entered by hand")]
        p = mark_duplicates(preview_import(ENGLISH), existing)
        dup = p.rows[0]
        assert dup.ok is False and dup.error.startswith(f"duplicate of {existing[0].id}")
        assert dup.transaction is not None, "the parsed row stays visible"
        assert p.valid == 2 and p.invalid == 1
        assert [r.ok for r in p.rows[1:]] == [True, True]

    def test_fees_and_note_do_not_make_a_trade_different(self):
        existing = [Transaction(D(2025, 1, 15), A, T.BUY, Decimal("120.0"), Decimal("22.4"),
                                "EUR", Decimal("9.99"))]
        assert mark_duplicates(preview_import(ENGLISH), existing).rows[0].ok is False

    def test_a_different_price_is_a_different_trade(self):
        existing = [Transaction(D(2025, 1, 15), A, T.BUY, Decimal("120"), Decimal("22.41"))]
        assert mark_duplicates(preview_import(ENGLISH), existing).rows[0].ok is True

    def test_already_invalid_rows_are_left_alone(self):
        text = ENGLISH + "2025-06-01,IE0002Y8CX98,Buy,abc,1,EUR,0,\n"
        p = mark_duplicates(preview_import(text), [])
        assert "quantity" in p.rows[3].error
