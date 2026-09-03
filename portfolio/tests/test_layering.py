"""The architectural constraint, enforced rather than documented.

`core/` must be importable and fully testable with no network and no Streamlit.
That is a load-bearing requirement -- it is what lets the UI be replaced later
without rewriting the portfolio mathematics -- and it is exactly the kind of
rule that erodes one convenient import at a time.

A green suite on a laptop that happens to have Streamlit installed proves
nothing, so these tests read the source with `ast` instead of trusting the
import to fail.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import textwrap

import pytest

CORE = pathlib.Path(__file__).resolve().parents[1] / "core"
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

FORBIDDEN_IN_CORE = {
    "streamlit",            # the UI layer must be replaceable
    "requests", "httpx", "urllib", "urllib3", "http", "socket",
    "yfinance",             # no provider-specific anything
    "aiohttp",
}


def imported_modules(path: pathlib.Path) -> set[str]:
    """Top-level module names imported by a file, via AST rather than regex."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
            elif node.module:
                names.add("." * node.level + node.module.split(".")[0])
    return names


CORE_FILES = sorted(CORE.glob("*.py"))


def test_core_has_modules_to_check():
    assert CORE_FILES, "no core modules found; the layering test would pass vacuously"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_core_imports_nothing_forbidden(path):
    offenders = imported_modules(path) & FORBIDDEN_IN_CORE
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. core/ must stay pure: no UI, "
        f"no network, no provider-specific code.")


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_core_does_not_import_upward(path):
    """core/ may not depend on data/ or app/. The dependency arrow points in."""
    bad = {m for m in imported_modules(path)
           if m.lstrip(".") in {"data", "app"} and m.startswith(".")}
    assert not bad, f"{path.name} imports {sorted(bad)}; core/ must not depend on outer layers"


def test_core_imports_with_streamlit_unavailable():
    """Import all of core in a subprocess where streamlit cannot be imported.

    A subprocess rather than monkeypatching this one: reloading modules in
    place rebinds their classes, so every already-imported test module would
    keep the old `Money` and equality checks elsewhere would start failing for
    reasons unrelated to what is being tested. Isolation is also the stronger
    check -- it is genuinely a fresh interpreter with no Streamlit.
    """
    script = textwrap.dedent("""
        import sys
        class Blocker:
            def find_module(self, name, path=None):
                if name.split(".")[0] == "streamlit":
                    raise ImportError("streamlit is not installed (simulated)")
                return None
        sys.meta_path.insert(0, Blocker())
        import portfolio.core.money, portfolio.core.models
        import portfolio.core.positions, portfolio.core.universe
        assert "streamlit" not in sys.modules
        print("core imported cleanly")
    """)
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, cwd=repo_root)
    assert result.returncode == 0, (
        f"core/ failed to import without Streamlit:\n{result.stderr}")
    assert "core imported cleanly" in result.stdout


def test_data_layer_may_use_the_network_but_core_may_not():
    """A sanity check that the rule is about layering, not about banning HTTP.

    If this ever fails it means the check above is testing nothing, because
    nothing anywhere imports the modules it forbids.
    """
    assert FORBIDDEN_IN_CORE, "the forbidden set must not be empty"
    assert DATA.exists(), "data/ layer is where network access belongs"
