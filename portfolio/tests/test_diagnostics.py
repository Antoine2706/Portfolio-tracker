"""Which copy is running -- and the two ways of asking that both lie.

This is the test for `portfolio/diagnostics.py`, and it exists because two
successive attempts to answer "is the running package the checkout or an
install" were both structurally incapable of returning the wrong answer:

  1. `python -c "import portfolio, os; print(os.path.dirname(
     portfolio.__file__))"`, run from the checkout. `sys.path[0]` is "" for
     `python -c`, meaning the current directory, so this reports the checkout
     whatever is installed. It was used to test a two-clone hypothesis and
     reported the clone, which was read as disproof. It was not a test.

  2. Swapping that for `importlib.metadata`. Better in concept -- the metadata
     is the right place to ask -- and the naive call has the same blind spot,
     because `importlib.metadata` also walks `sys.path`. From a checkout it
     finds the `portfolio_tracker.egg-info` left there by a build, and answers
     with relative paths and no `direct_url.json`, so an editable install
     reads as non-editable.

Both are demonstrated below in a temporary directory rather than argued from
the standard library's documentation, because the first one was argued from
reasoning that was correct in every respect except the conclusion.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

from portfolio.diagnostics import (Installation, _installed_search_path,
                                   inspect_installation)


def run_in(directory: pathlib.Path, code: str) -> str:
    """Run a snippet with `directory` as the working directory."""
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          cwd=directory, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


class TestTheObviousDiagnosticCannotFail:
    """The instance that started this: a check used as evidence."""

    def test_a_directory_named_portfolio_wins_over_any_install(self, tmp_path):
        """`import portfolio` from a directory containing ./portfolio finds
        the directory. Not the installed package, not the checkout that is
        actually running -- whatever happens to be underfoot."""
        fake = tmp_path / "portfolio"
        fake.mkdir()
        (fake / "__init__.py").write_text("MARKER = 'not the real package'\n")

        got = run_in(tmp_path, """
            import portfolio, os
            print(os.path.dirname(portfolio.__file__))
        """)
        assert got == str(fake), (
            f"expected {fake} to shadow the install, got {got}. "
            f"If this ever stops being true the diagnostic in the module "
            f"docstring becomes sound and this test should be deleted.")

        marker = run_in(tmp_path, """
            import portfolio
            print(getattr(portfolio, "MARKER", "the real package"))
        """)
        assert marker == "not the real package"

    def test_sys_path_zero_is_the_current_directory_for_dash_c(self, tmp_path):
        """The mechanism, stated as a fact about the interpreter rather than
        as something remembered about it."""
        assert run_in(tmp_path, "import sys; print(repr(sys.path[0]))") == "''"


class TestTheMetadataLookupHasTheSameBlindSpot:
    """The second instance, found while fixing the first."""

    def test_a_local_egg_info_shadows_the_real_distribution(self, tmp_path):
        """A build artefact in the working directory answers for the install.

        `pip install -e .` and `python setup.py` both leave a
        `<name>.egg-info` in the checkout. `importlib.metadata` searches
        `sys.path`, finds it first, and answers from it.
        """
        egg = tmp_path / "made_up_dist.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO").write_text(
            "Metadata-Version: 2.1\nName: made-up-dist\nVersion: 9.9.9\n")

        found = run_in(tmp_path, """
            import importlib.metadata as md
            print(md.distribution("made-up-dist").version)
        """)
        assert found == "9.9.9", (
            "a bare .egg-info in the working directory was not picked up; "
            "if that is no longer how importlib.metadata behaves, the "
            "exclusion in _installed_search_path is unnecessary")

    def test_excluding_the_search_path_makes_it_disappear(self, tmp_path):
        """The fix, demonstrated: with the current directory removed from the
        search, the same artefact is not found."""
        egg = tmp_path / "made_up_dist.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO").write_text(
            "Metadata-Version: 2.1\nName: made-up-dist\nVersion: 9.9.9\n")

        found = run_in(tmp_path, """
            import importlib.metadata as md, sys, pathlib
            path = [e for e in sys.path
                    if e and pathlib.Path(e).resolve() != pathlib.Path.cwd()]
            ctx = md.DistributionFinder.Context(name="made-up-dist", path=path)
            print(len(list(md.Distribution.discover(context=ctx))))
        """)
        assert found == "0"

    def test_the_search_path_drops_the_cwd_and_the_checkout(self):
        path = _installed_search_path()
        here = pathlib.Path(__file__).resolve().parents[2]
        assert "" not in path
        assert all(pathlib.Path(p).resolve() != here for p in path), path
        assert all(pathlib.Path(p).resolve() != pathlib.Path.cwd()
                   for p in path), path


class TestItReportsWhatIsActuallyServed:
    """The bug the whole file is downstream of."""

    def test_a_served_directory_missing_files_is_reported(self, tmp_path):
        """The reported symptom: a page in the repo, absent from the copy
        being served, and nothing anywhere saying so."""
        served = tmp_path / "web"
        (served / "pages").mkdir(parents=True)
        (served / "pages" / "overview.js").write_text("// present\n")
        report = inspect_installation(web_dir=served)
        assert not report.healthy
        assert report.served_is_a_snapshot
        assert any("allocate.js" in name for name in report.missing_from_served)

    def test_a_matching_directory_is_healthy(self):
        repo_web = pathlib.Path(__file__).resolve().parents[1] / "web"
        report = inspect_installation(web_dir=repo_web)
        assert report.healthy and not report.served_is_a_snapshot
        assert report.missing_from_served == ()

    def test_the_report_names_the_directory_it_is_talking_about(self, tmp_path):
        """A diagnostic that says something is wrong without saying where is
        the failure mode this replaces, not a fix for it."""
        served = tmp_path / "web"
        served.mkdir()
        text = "\n".join(inspect_installation(web_dir=served).lines())
        assert str(served) in text
        assert "render blank" in text or "renders" in text

    def test_the_lines_render_without_an_install(self):
        """`Installation` must print something sane when every lookup failed,
        because the case where nothing resolves is exactly when someone is
        reading this."""
        blank = Installation(
            imported_from=None, shadowed_by_cwd=False, metadata_location=None,
            dist_info=None, version=None, editable=None, web_dir=None,
            repo_web_dir=None)
        text = "\n".join(blank.lines())
        assert "not importable" in text and "not installed" in text
        assert blank.healthy


class TestTheCheckBites:
    def test_a_healthy_report_is_not_vacuous(self):
        """`healthy` must be capable of being False, or the serve-time
        warning it gates can never fire."""
        served_missing = Installation(
            imported_from=None, shadowed_by_cwd=False, metadata_location=None,
            dist_info=None, version=None, editable=None, web_dir=None,
            repo_web_dir=None, missing_from_served=("pages/allocate.js",))
        assert not served_missing.healthy
        assert "pages/allocate.js" in "\n".join(served_missing.lines())

    def test_shadowing_is_reported_when_it_happens(self):
        """Under pytest from the checkout, the package is imported from the
        checkout. Whether that counts as shadowing depends on `sys.path[0]`,
        so assert the property is computed rather than hardcoded False."""
        report = inspect_installation()
        assert isinstance(report.shadowed_by_cwd, bool)
        if report.shadowed_by_cwd:
            assert "CURRENT DIRECTORY" in "\n".join(report.lines())
