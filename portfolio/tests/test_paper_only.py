"""The paper-trading boundary, enforced by the repository rather than a flag.

Why this is a test and not a configuration option
--------------------------------------------------
Software that places orders on someone's account without them acting on each
transaction is portfolio management under MiFID II, and providing it as a
service requires authorisation -- in Belgium, from the FSMA. ESMA has said
twice and directly that automatic execution without client action per
transaction constitutes portfolio management: the 2012 Investor Protection
Q&A (question 9) and the 2023 Copy Trading supervisory briefing.

Three consequences shape the code, not just the documentation:

*   Whose account and whose credentials are used changes nothing. The test is
    whether discretion is exercised, not who holds the assets.
*   A prominent "this is not investment advice" notice changes nothing
    either. ESMA states explicitly that such a disclaimer "would not in any
    way impact the characterisation of the service."
*   The authorisation requires two people in effective management and a
    compliance function that cannot be outsourced, so it is not obtainable by
    one person building this in the evenings.

The account this is built against is **Execution Only** -- confirmed on the
MeDirect statement of 30 June 2026. That is the classification in which the
bank gives no advice and assesses no suitability, and the holder makes every
decision themselves. It is the classification this tool is consistent with
and the one it must stay consistent with: a tool that produces a list for its
holder to type into their own broker sits inside Execution Only; the same
tool with an endpoint attached does not, whoever owns the account.

Trading one's own money automatically needs no authorisation. Building
something another person could point at their own account does. The line
between those two is a single endpoint constant, which is exactly the kind of
line that gets crossed by a helpful refactor eighteen months from now, long
after the reasoning has been forgotten.

So the boundary is structural: the live endpoints do not exist in the
repository, and this test fails if they appear. It is not a substitute for
judgement, it is a tripwire for the moment judgement lapses.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# Interactive Brokers separates paper from live by port and by host. These are
# the LIVE ones. 7497 (TWS) and 4002 (Gateway) are the paper equivalents and
# are permitted.
LIVE_MARKERS = {
    ":7496": "the live TWS API port (paper is 7497)",
    ":4001": "the live IB Gateway port (paper is 4002)",
    "api.ibkr.com": "the live Client Portal Web API host",
    "gdcdyn.interactivebrokers.com": "the live Client Portal gateway",
}

# Directories with no bearing on what this repository does: vendored
# third-party code we did not write, build artefacts, and binary assets.
SKIP_DIRS = {".git", "__pycache__", "node_modules", "vendor", ".pytest_cache",
             "screenshots", ".ruff_cache", "dist", "build", ".venv",
             # A nested worktree the review harness creates under .claude/
             # is a second copy of the whole repository, this file included.
             ".claude"}
TEXT_SUFFIXES = {".py", ".js", ".json", ".toml", ".cfg", ".ini", ".md", ".txt",
                 ".yml", ".yaml", ".css", ".html", ".sh", ".jsonl"}

# This file necessarily contains every marker it forbids.
SELF = pathlib.Path(__file__).resolve()


def _scannable() -> list[pathlib.Path]:
    out = []
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.resolve() == SELF:
            continue
        out.append(path)
    return sorted(out)


SCANNABLE = _scannable()


def test_there_is_something_to_scan():
    """Without this, an empty file list would make the guard pass vacuously.

    The most likely way for this protection to quietly stop working is not
    someone adding a live endpoint -- it is the glob above silently matching
    nothing after a directory is renamed.
    """
    assert len(SCANNABLE) > 50, (
        f"only {len(SCANNABLE)} files matched; the scan is not covering the "
        f"repository and would pass no matter what the code contained")
    assert any(p.suffix == ".py" for p in SCANNABLE)
    assert any("portfolio" in p.parts for p in SCANNABLE)


@pytest.mark.parametrize("marker,what", sorted(LIVE_MARKERS.items()))
def test_no_live_trading_endpoint_appears_anywhere(marker, what):
    hits = []
    for path in SCANNABLE:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if marker in text:
            hits.append(str(path.relative_to(REPO)))
    assert not hits, (
        f"{marker} ({what}) appears in {hits}. This phase is paper-trading "
        f"only, permanently: automatic execution on a real account is "
        f"portfolio management under MiFID II and needs FSMA authorisation "
        f"that a single person cannot obtain. If a live endpoint is genuinely "
        f"wanted, that is a decision to take deliberately and out loud, not "
        f"by editing a constant.")


def test_the_endpoint_guard_bites(tmp_path):
    """Prove the scan would catch a violation, by giving it one.

    A grep-based guard that has only ever been run against a clean tree is
    indistinguishable from a grep with a typo in the pattern.
    """
    planted = tmp_path / "broker_live.py"
    planted.write_text('HOST = "api.ibkr.com"\nPORT = ":7496"\n', encoding="utf-8")
    text = planted.read_text(encoding="utf-8")
    caught = [m for m in LIVE_MARKERS if m in text]
    assert set(caught) == {"api.ibkr.com", ":7496"}, (
        f"the marker list failed to catch a deliberately planted live "
        f"endpoint; it caught {caught}")

    innocent = tmp_path / "broker_paper.py"
    innocent.write_text('HOST = "127.0.0.1"\nPORT = ":7497"   # paper TWS\n',
                        encoding="utf-8")
    text = innocent.read_text(encoding="utf-8")
    assert not [m for m in LIVE_MARKERS if m in text], (
        "the guard flags a paper endpoint, so it would have to be disabled "
        "to do the legitimate work and would then protect nothing")


def test_no_broker_package_exists_yet():
    """The paper broker is step 7. Nothing should have crept in before it.

    Kept as a positive statement of where the project is, so that when
    broker/ does arrive it arrives with someone deliberately deleting this
    test and reading the boundary note above on the way past.
    """
    broker = REPO / "portfolio" / "broker"
    assert not broker.exists(), (
        "portfolio/broker/ exists. The build order puts the paper connection "
        "after the controls and the first two policies have been evaluated; "
        "if it is genuinely time for it, delete this test in the same commit "
        "that adds the package.")
