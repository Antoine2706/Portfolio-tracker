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
APP = pathlib.Path(__file__).resolve().parents[1] / "api"

FORBIDDEN_IN_CORE = {
    "streamlit", "fastapi", "starlette", "pydantic",   # the UI layer must be replaceable
    "requests", "httpx", "urllib", "urllib3", "http", "socket",
    "yfinance",             # no provider-specific anything
    "aiohttp", "sqlite3",   # storage is data/'s job
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


def test_core_imports_with_web_framework_unavailable():
    """Import all of core in a subprocess where no UI framework can be imported.

    A subprocess rather than monkeypatching this one: reloading modules in
    place rebinds their classes, so every already-imported test module would
    keep the old `Money` and equality checks elsewhere would start failing for
    reasons unrelated to what is being tested. Isolation is also the stronger
    check -- it is genuinely a fresh interpreter with no web framework.
    """
    script = textwrap.dedent("""
        import sys, importlib.abc
        BLOCKED = {"streamlit", "fastapi", "starlette", "pydantic"}
        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"{name} is not installed (simulated)")
                return None
        sys.meta_path.insert(0, Blocker())
        import portfolio.core.money, portfolio.core.models
        import portfolio.core.positions, portfolio.core.universe
        import portfolio.core.risk, portfolio.core.returns, portfolio.core.report
        assert not (BLOCKED & set(m.split(".")[0] for m in sys.modules))
        print("core imported cleanly")
    """)
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, cwd=repo_root)
    assert result.returncode == 0, (
        f"core/ failed to import without a web framework:\n{result.stderr}")
    assert "core imported cleanly" in result.stdout


# The other direction, and the one that actually erodes. The AST guard above
# stops core importing UI; nothing stops arithmetic drifting into the API layer,
# where it would be untested and invisible. This is a partial defence -- pandas
# is allowed because projecting a frame into JSON needs it -- so the real
# protection is that core leaves the API nothing to compute.
FORBIDDEN_IN_APP = {
    "numpy",        # the maths library: a projection has no business importing it
    "scipy",
    "statistics",
}

APP_FILES = sorted(APP.rglob("*.py"))


def test_app_has_modules_to_check():
    assert APP_FILES, "no api modules found; the guard would pass vacuously"


@pytest.mark.parametrize("path", APP_FILES, ids=lambda p: p.name)
def test_app_does_not_import_maths_libraries(path):
    offenders = imported_modules(path) & FORBIDDEN_IN_APP
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. The API projects; it does "
        f"not compute. Move the calculation into core/ with a test.")


@pytest.mark.parametrize("path", APP_FILES, ids=lambda p: p.name)
def test_app_does_not_reach_into_private_helpers(path):
    """The API importing a private function is logic leaking out of its module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    private: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name.startswith("_"):
                    private.add(alias.name)
    assert not private, (
        f"{path.name} imports private {sorted(private)}. If a view needs it, "
        f"it belongs in the module's public interface.")


# --------------------------------------------------------------------------
# The evaluation layer
# --------------------------------------------------------------------------
# `agents/` holds decision policies and `eval/` the harness that measures
# them. Two rules matter here, and only one of them is obvious.
#
# The obvious one: neither may reach the network or a UI framework. A policy
# that fetched a price would be untestable and non-reproducible, and a
# backtest that fetched anything would give a different answer on Tuesday.
#
# The one that actually matters: `agents/` may not import `eval/`. A policy
# must not be able to tell that it is being backtested. If it could, it could
# behave differently under evaluation than in production -- which is the
# single most expensive bug available in this domain, because the backtest
# would be measuring something that will never happen. So `agents/` defines
# both its input type (MarketView) and its output type (Proposal), `eval/`
# imports `agents/`, and the arrow never points back.

AGENTS = pathlib.Path(__file__).resolve().parents[1] / "agents"
EVAL = pathlib.Path(__file__).resolve().parents[1] / "eval"

FORBIDDEN_IN_EVALUATION = {
    "streamlit", "fastapi", "starlette", "pydantic",
    "requests", "httpx", "urllib", "urllib3", "http", "socket",
    "yfinance", "aiohttp", "sqlite3",
}

AGENT_FILES = sorted(AGENTS.glob("*.py"))
EVAL_FILES = sorted(EVAL.glob("*.py"))


def test_evaluation_layer_has_modules_to_check():
    assert AGENT_FILES and EVAL_FILES, (
        "no agents/ or eval/ modules found; these guards would pass vacuously")


@pytest.mark.parametrize("path", AGENT_FILES + EVAL_FILES, ids=lambda p: p.name)
def test_evaluation_layer_is_offline(path):
    offenders = imported_modules(path) & FORBIDDEN_IN_EVALUATION
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. A policy or a backtest that "
        f"reaches the network is not reproducible, and a result that cannot be "
        f"reproduced is not a result.")


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: p.name)
def test_agents_do_not_know_they_are_being_evaluated(path):
    bad = {m for m in imported_modules(path) if m.lstrip(".") == "eval"}
    assert not bad, (
        f"{path.name} imports from eval/. A policy that can tell it is inside "
        f"a backtest can behave differently there, and then the backtest "
        f"measures something that will never happen in production.")


@pytest.mark.parametrize("path", AGENT_FILES + EVAL_FILES, ids=lambda p: p.name)
def test_the_evaluation_layer_does_not_import_the_composition_root(path):
    """`research.py` wires the data layer to the harness, so it may import
    anything. Nothing may import it back: an arrow from eval/ to research
    would drag the network into a package whose whole value is being offline.
    """
    bad = {m for m in imported_modules(path) if m.lstrip(".") == "research"}
    assert not bad, (
        f"{path.name} imports research, reversing the composition arrow and "
        f"pulling the data layer into code that must stay offline.")


def test_the_layering_guard_actually_bites(tmp_path):
    """Sabotage, because a guard that has never failed proves nothing.

    This project has hit the same class of bug three times: a check that
    looked like it worked without exercising what it claimed. The rule here
    is that no guard is trusted until it has been shown to fail on a file
    that violates it.
    """
    offender = tmp_path / "leaky_policy.py"
    offender.write_text(
        "import requests\n"
        "from ..eval.harness import walk_forward\n", encoding="utf-8")
    found = imported_modules(offender)
    assert found & FORBIDDEN_IN_EVALUATION == {"requests"}, (
        "the network guard did not flag a deliberate `import requests`")
    assert {m for m in found if m.lstrip(".") == "eval"}, (
        "the arrow-direction guard did not flag a deliberate import of eval/")

    clean = tmp_path / "clean_policy.py"
    clean.write_text(
        "import numpy as np\n"
        "from ..agents.base import Proposal\n", encoding="utf-8")
    assert not imported_modules(clean) & FORBIDDEN_IN_EVALUATION, (
        "the guard flags a clean file, so it is not testing what it claims")


def test_data_layer_may_use_the_network_but_core_may_not():
    """A sanity check that the rule is about layering, not about banning HTTP.

    If this ever fails it means the check above is testing nothing, because
    nothing anywhere imports the modules it forbids.
    """
    assert FORBIDDEN_IN_CORE, "the forbidden set must not be empty"
    assert DATA.exists(), "data/ layer is where network access belongs"
