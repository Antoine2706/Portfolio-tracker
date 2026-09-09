"""The htm whitespace collapse, which has now shipped three times.

Three occurrences is a missing test, not three mistakes:

    "14.7135against"          a price fused to the next word
    "WheatJE00BN7KB664"       a fund name fused to its ISIN
    "Understood as\u20ac5,000.00"  a label fused to an amount

They are not all the same mechanism, and the difference decides what a test
can do about them. Reading the fixes back out of git: the allocate ones were
`text` / newline / `${...}`, which is the collapse below and is detectable.
The Wheat one was fixed with `</span>${" "}<span>` -- two elements adjacent on
one line, with no whitespace to collapse. Nothing was deleted there; nothing
was ever written. That is a judgement about whether a space was wanted, and
no static rule can make it.

So this covers the newline collapse and says plainly that it does not cover
the other. A check with a stated blind spot is worth having; one that claims
the whole family and quietly covers a third of it is what this project keeps
finding.

The rule, read out of the vendored parser rather than guessed
-------------------------------------------------------------
`web/vendor/preact-htm.module.js` builds each static text chunk and, before
the field that follows it, runs

    o = o.replace(/^\\s*\\n\\s*|\\s*\\n\\s*$/g, "")

So leading and trailing whitespace **containing a newline** is deleted from a
text chunk. A space on the same line survives; a line break and its
indentation do not. That is why

    html`Understood as
      ${amount}`

renders "Understood as€5,000.00" while `html`Understood as ${amount}`` is
fine. The collapse is not a quirk to remember, it is a documented consequence
of one regex, and it is therefore checkable.

Two mitigations are already in use in this codebase and both are accepted
here: an explicit `${" "}` before the interpolation, or an interpolated
expression whose own string literals begin with a space. Anything else that
puts a line break between text and an interpolation is flagged.

Deliberately a static check. It needs no browser, no node and no running
server, so it runs in both CI jobs and on every commit -- which is the point,
since this defect is introduced while writing a template and found weeks
later by reading a screenshot.
"""

from __future__ import annotations

import pathlib
import re

import pytest

WEB = pathlib.Path(__file__).resolve().parents[1] / "web"

# Text with content, then a line break, then either an interpolation or an
# OPENING tag. Both fuse: the parser calls its flush on `<` exactly as it does
# on a field, so `text\n  <strong>` loses the gap the same way `text\n  ${x}`
# does. That half was missing from the first version of this rule, and the
# sabotage below is what found it -- the check was written, run against a
# deliberately broken file, and passed. Adding tags immediately turned up two
# live occurrences on the Risk page ("contained.Parametric", "tails.Cornish").
#
# A CLOSING tag is excluded: `</div>` ends the element, so nothing follows to
# fuse with. The characters excluded before the break matter too: `>` ends a
# tag, and `=` `"` `'` sit inside attributes where there is no text node.
COLLAPSE = re.compile(r"[^\s<>{}=\"'`/][ \t]*\n[ \t]*(?:\$\{|<(?!/))")

# Block and line comments. Prose in a comment is not a template and its line
# breaks mean nothing; `Drawer.js` describes "the shell mounts one <DrawerHost
# /> that renders..." across two lines and is not a defect.
# `<` is in the lookbehind because htm closes a component with `<//>`, which
# a naive line-comment rule swallows along with the rest of the template.
COMMENTS = re.compile(r"/\*.*?\*/|(?<![:\\<])//[^\n]*", re.DOTALL)

# `${" "}` -- the project's existing way of putting the space back.
EXPLICIT_SPACE = re.compile(r"\$\{\s*[\"'] [\"']\s*\}")

# An interpolation whose first string literal opens with a space, e.g.
# `${gap > 0.01 ? " making" : " almost none"}`.
LEADING_SPACE_LITERAL = re.compile(r"\$\{[^\"'}]*[\"'] ")


def sources() -> list[pathlib.Path]:
    return [p for p in sorted(WEB.rglob("*.js")) if "vendor" not in p.parts]


def collapses(text: str) -> list[tuple[int, str]]:
    """Every unmitigated collapse in one file, as (line, context).

    Comments are blanked to spaces rather than removed, so reported line
    numbers still point at the real line.
    """
    text = COMMENTS.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    out = []
    for match in COLLAPSE.finditer(text):
        tail = text[match.end() - 2:match.end() + 60]
        if EXPLICIT_SPACE.match(tail) or LEADING_SPACE_LITERAL.match(tail):
            continue
        line = text[:match.start()].count("\n") + 1
        out.append((line, text[max(0, match.start() - 40):
                               match.end() + 40].replace("\n", "\\n")))
    return out


class TestNoTemplateFusesTextToAnInterpolation:
    def test_the_scan_actually_scans_something(self):
        """A loop over an empty list passes and proves nothing. The client is
        a dozen pages and a dozen components; if this drops to nothing, the
        path or the glob is wrong rather than the codebase being clean."""
        found = sources()
        assert len(found) >= 20, [p.name for p in found]
        assert any("${" in p.read_text(encoding="utf-8") for p in found)

    @pytest.mark.parametrize("path", sources(), ids=lambda p: p.name)
    def test_every_template_keeps_its_spaces(self, path):
        bad = collapses(path.read_text(encoding="utf-8"))
        assert not bad, "\n".join(
            [f"{path.name}: a line break between text and an interpolation. "
             f"htm deletes whitespace containing a newline from a text chunk, "
             f"so these render fused. Add ${{\" \"}} or move the "
             f"interpolation onto the same line."]
            + [f"  line {line}: ...{ctx}" for line, ctx in bad])


class TestTheCheckBites:
    """A lint nobody has seen fail is a lint nobody should trust.

    Every newline collapse this project has actually fixed, in the shape it
    was written in -- plus the two shapes this check deliberately does not
    catch, asserted as not caught so the boundary is a fact rather than a
    hope.
    """

    @pytest.mark.parametrize("template", [
        # The shape of every newline collapse this project has fixed, taken
        # from the diffs that added `${" "}` in 8173684 and b2a03ac.
        'html`<div>Understood as\n    ${fmt.money(v, ccy)}</div>`',
        'html`<div>Best and worst destinations differ by\n    ${fmt.num(gap)}</div>`',
        'html`<div>The best any purchase this size reaches is\n      ${fmt.num(x)}</div>`',
        'html`<span>closes\n  ${fmt.pct(share)} of the gap</span>`',
        # The tag half, which the first version of this rule missed and which
        # was shipping on the Risk page when it was added.
        'html`<div>the window never contained.\n      <strong>Parametric</strong></div>`',
        'html`<div>Understood as\n    <strong>${fmt.money(v, ccy)}</strong></div>`',
    ])
    def test_it_catches_what_actually_shipped(self, template):
        assert collapses(template), template

    @pytest.mark.parametrize("template", [
        # Two elements adjacent with no whitespace between them at all -- the
        # "WheatJE00BN7KB664" shape. Nothing is collapsed; a space was simply
        # never written, and whether one was wanted is not in the source.
        'html`<span>${name}</span><span>${isin}</span>`',
        # Value then value on separate lines. Indistinguishable from the
        # ordinary component-slot idiom, which appears 53 times in this
        # codebase and is correct every time.
        'html`<div>${children}\n    ${footer}</div>`',
    ])
    def test_the_stated_blind_spot_is_real(self, template):
        """Documented rather than silently absent. If a future change makes
        these detectable without noise, this test is the place that says they
        currently are not."""
        assert not collapses(template), template

    @pytest.mark.parametrize("template", [
        'html`<div>Understood as${" "}\n    ${fmt.money(v, ccy)}</div>`',
        'html`<div>Understood as ${fmt.money(v, ccy)}</div>`',
        'html`<div>worth\n    ${gap > 0.01 ? " making" : " almost none"}</div>`',
        'html`<div class="x">\n  ${children}\n</div>`',
        'html`<${Card} title=${t}>\n  ${body}\n<//>`',
    ])
    def test_it_stays_quiet_on_the_mitigated_forms(self, template):
        """Both existing mitigations, plus the ordinary shape of a component
        whose children sit on their own line -- where the chunk before the
        interpolation is pure whitespace and is meant to vanish."""
        assert not collapses(template), template

    def test_a_tag_boundary_is_not_a_collapse(self):
        """`>` ends a tag, so what follows is a fresh text chunk. Flagging
        this would make the check unusable and it would be turned off."""
        assert not collapses('html`<div>\n  ${x}\n</div>`')

    def test_a_closing_tag_is_not_a_collapse(self):
        """`</div>` ends the element; there is nothing after it to fuse."""
        assert not collapses('html`<div>some text\n</div>`')
        assert not collapses('html`<${Card}>some text\n<//>`')

    def test_prose_in_a_comment_is_not_a_template(self):
        """`Drawer.js` describes "<DrawerHost />" across two lines of a module
        docstring. Flagging that would train the reader to ignore this."""
        assert not collapses(
            "/* The shell mounts one\n   <DrawerHost /> that renders. */")
        assert not collapses("// see the note\n// <Thing /> is mounted once")
