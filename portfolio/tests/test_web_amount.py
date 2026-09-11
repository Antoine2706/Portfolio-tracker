"""Reading a typed money amount, where guessing wrong costs a factor of 1000.

Runs the real `web/lib/amount.js` under node rather than a Python port of it,
because a port would be a second implementation and testing a copy proves
nothing about the file the browser loads.

Skipped when node is unavailable. The CI job that installs the app has it;
the core job does not, and does not need it.

Why this exists at all
----------------------
`allocate.js` and `simulator.js` each carried a parser that read "10.000" as
ten. That is how ten thousand is written in Belgium, so entering the amount
of a real purchase produced an allocation of one share and a page that looked
broken -- with no error anywhere, because the number parsed cleanly and was
simply the wrong number. `data/importers.py` had the rule right for CSV
import all along; the web layer had diverged from it twice.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

AMOUNT_JS = (pathlib.Path(__file__).resolve().parents[1]
             / "web" / "lib" / "amount.js")

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is not installed")


def parse(*inputs: str) -> list[dict]:
    """Run the browser's own parser over `inputs` and return its results."""
    script = (
        f"import {{ parseAmount }} from {json.dumps(str(AMOUNT_JS))};\n"
        f"const out = {json.dumps(list(inputs))}.map(parseAmount);\n"
        f"console.log(JSON.stringify(out));\n"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def value(text: str):
    return parse(text)[0]["value"]


def state(text: str) -> str:
    return parse(text)[0]["state"]


class TestTheUnambiguousCases:
    @pytest.mark.parametrize("text,expected", [
        ("5000", 5000.0),
        ("5 000", 5000.0),
        ("€5000", 5000.0),
        ("5000.50", 5000.50),
        ("5000,50", 5000.50),
        ("1,250.5", 1250.5),
        ("1.250,50", 1250.50),
        ("1.234.567", 1234567.0),
        ("1,234,567", 1234567.0),
        ("0.5", 0.5),
        ("0,5", 0.5),
        ("-250,75", -250.75),
        ("12", 12.0),
    ])
    def test_it_reads_what_was_meant(self, text, expected):
        assert value(text) == pytest.approx(expected), text

    def test_the_last_separator_is_the_decimal_point(self):
        """No convention puts a thousands separator after a decimal one, so
        whichever comes last settles it and neither reading is a guess."""
        assert value("1.250,50") == pytest.approx(1250.50)
        assert value("1,250.50") == pytest.approx(1250.50)

    def test_a_repeated_separator_is_thousands(self):
        """No convention has two decimal points."""
        assert value("1.234.567") == pytest.approx(1234567.0)

    def test_a_tail_that_is_not_three_digits_is_a_decimal(self):
        """A thousands group is always exactly three digits, so anything
        else settles it."""
        assert value("10.5") == pytest.approx(10.5)
        assert value("10,5") == pytest.approx(10.5)
        assert value("1.25") == pytest.approx(1.25)


class TestTheAmbiguousCase:
    """The bug. "10.000" is ten thousand in Brussels and ten in London."""

    @pytest.mark.parametrize("text", ["10.000", "10,000", "5.000", "2,500",
                                      "1.000", "250.000"])
    def test_it_refuses_rather_than_guessing(self, text):
        assert state(text) == "ambiguous", text

    def test_it_offers_both_readings_so_the_refusal_is_answerable(self):
        """A refusal the reader cannot act on is just a different way of not
        working. Larger reading first, because it is the one a European
        typing a purchase amount almost always means."""
        got = parse("10.000")[0]
        assert got["readings"] == [10000.0, 10.0]

    def test_the_old_behaviour_would_have_returned_ten(self):
        """What made this worth finding: the previous parser returned 10.0
        here, silently, and an allocation of one share looks like a broken
        page rather than a misread number."""
        assert value("10.000") is None
        assert state("10.000") != "ok"

    def test_a_sign_survives_the_refusal(self):
        assert parse("-10.000")[0]["readings"] == [-10000.0, -10.0]


class TestItSaysNothingRatherThanZero:
    @pytest.mark.parametrize("text", ["", "   ", "€", " ", "€€"])
    def test_blank_is_empty_not_zero(self, text):
        """Decoration without digits is "nothing typed", not "invalid".

        The distinction matters because this runs on every keystroke: a field
        showing "not a number" while someone is still reaching for the digits
        is worse than one that waits.
        """
        got = parse(text)[0]
        assert got["state"] == "empty" and got["value"] is None

    @pytest.mark.parametrize("text", ["abc", "1.2.3,4,5", "--5", "1e5", "."])
    def test_nonsense_is_invalid_not_a_number(self, text):
        got = parse(text)[0]
        assert got["value"] is None, text
        assert got["state"] in ("invalid", "ambiguous"), text

    def test_zero_is_a_number_not_a_blank(self):
        """Zero is a value the caller must be free to reject on its own
        terms; conflating it with 'nothing typed' hides which happened."""
        got = parse("0")[0]
        assert got["state"] == "ok" and got["value"] == 0.0


class TestItAgreesWithThePythonImporter:
    """The web layer diverged from `data/importers.py` and that is why this
    bug existed. Where the two are both unambiguous they must agree, or the
    same CSV row and the same typed string mean different amounts.
    """

    @pytest.mark.parametrize("text", [
        "5000", "5000.50", "5000,50", "1,250.5", "1.250,50", "1.234.567",
        "0.5", "0,5", "12",
    ])
    def test_same_answer_as_the_csv_parser(self, text):
        from portfolio.data.importers import parse_number
        # `decimal_comma=True` is the Belgian setting, and the cases here are
        # the ones where that flag cannot change the answer.
        assert value(text) == pytest.approx(float(parse_number(text, True))), text
