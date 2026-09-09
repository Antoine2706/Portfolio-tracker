"""Which copy of this package is actually running, and what it serves.

Why this file exists
--------------------
The allocate page was reported missing. It was in git and in the clone, and
absent from the directory the running server serves, because a plain
`pip install .` copies the package into site-packages and
`package-data = ["web/**/*"]` is a build-time glob: a page added afterwards is
not in the copy. Nothing said so. The page was simply blank.

The diagnostic reached for was:

    python -c "import portfolio, os; print(os.path.dirname(portfolio.__file__))"

run from inside the clone. For `python -c`, `sys.path[0]` is the empty string,
which means the current directory, so `import portfolio` finds `./portfolio`
before anything installed. Demonstrated rather than argued: a bare directory
named `portfolio` containing only an `__init__.py`, in the working directory,
is what that command reports. It answers "the clone" whenever it is run from
the clone, which is where anybody debugging this would run it, and it
therefore cannot contradict the hypothesis it was meant to test.

The `portfolio` console script does not have that problem -- its `sys.path[0]`
is the scripts directory, not the caller's -- which is exactly why the server
and the diagnostic disagreed.

So this reports several facts and, more importantly, whether they agree:

  * what `import portfolio` resolves to *here*, which is what code will use
  * whether the current directory is shadowing an installed package
  * what the installed distribution's metadata says with the current
    directory EXCLUDED from the search -- because `importlib.metadata` walks
    `sys.path` too, and run from a checkout it finds the `*.egg-info` build
    artefact sitting there and reports relative paths and no `direct_url.json`.
    Swapping `portfolio.__file__` for a naive `importlib.metadata` call moves
    the blind spot without closing it; this was found by running both from two
    directories rather than by reasoning about them
  * whether the install is editable, since that decides whether the served
    files are the clone's or a snapshot
  * the directory the web client is actually served from, and whether any
    file in this repo is missing from it

Only the last of those was the bug. Every other line is here because the
question "which copy is this" was answered wrongly once and cost two hours.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

__all__ = ["Installation", "inspect_installation", "DISTRIBUTION"]

DISTRIBUTION = "portfolio-tracker"


@dataclasses.dataclass(frozen=True)
class Installation:
    """Where this package is, from several angles that can disagree."""
    imported_from: pathlib.Path | None    # what `import portfolio` resolved to
    shadowed_by_cwd: bool                 # is ./portfolio winning over the install?
    metadata_location: pathlib.Path | None   # what importlib.metadata reports
    dist_info: pathlib.Path | None
    version: str | None
    editable: bool | None                 # None = could not tell
    web_dir: pathlib.Path | None          # the directory the client is served from
    repo_web_dir: pathlib.Path | None     # this source tree's, if we are in one
    missing_from_served: tuple[str, ...] = ()
    # What a naive `importlib.metadata.distribution()` answers, which from a
    # checkout is the local `*.egg-info` rather than the real install. Kept
    # because the disagreement between the two IS the diagnosis.
    naive_dist_info: pathlib.Path | None = None

    @property
    def metadata_is_shadowed(self) -> bool:
        if self.naive_dist_info is None or self.dist_info is None:
            return False
        return (not self.naive_dist_info.is_absolute()
                or self.naive_dist_info.resolve() != self.dist_info)

    @property
    def served_is_a_snapshot(self) -> bool:
        """Is the served directory a copy rather than this source tree?"""
        if self.web_dir is None or self.repo_web_dir is None:
            return False
        return self.web_dir.resolve() != self.repo_web_dir.resolve()

    @property
    def healthy(self) -> bool:
        return not self.missing_from_served

    def lines(self) -> list[str]:
        out = ["Installation", "=" * 60, ""]
        out.append(f"  import portfolio  -> {self.imported_from or 'not importable'}")
        if self.shadowed_by_cwd:
            out.append("      ^ this is the CURRENT DIRECTORY shadowing any "
                       "installed copy.")
            out.append("        `python -c \"import portfolio\"` run from a "
                       "checkout always")
            out.append("        reports the checkout, whatever is installed. "
                       "The console")
            out.append("        script does not, which is how the two can "
                       "disagree.")
        out.append(f"  distribution      -> {self.metadata_location or 'not installed'}"
                   + (f"  (v{self.version})" if self.version else ""))
        if self.metadata_is_shadowed:
            out.append(f"      ^ asked with the current directory excluded. "
                       f"Asked without, it")
            out.append(f"        answers {self.naive_dist_info}, which is a "
                       f"build artefact")
            out.append("        in the checkout, not an install. "
                       "importlib.metadata walks")
            out.append("        sys.path too, so it has the same blind spot "
                       "as __file__.")
        if self.editable is not None:
            out.append(f"  editable install  -> {'yes' if self.editable else 'NO'}"
                       + ("" if self.editable else
                          "  <- the served files are a copy taken when pip ran"))
        out.append(f"  web client served -> {self.web_dir or 'the app extra is not installed'}")
        if self.repo_web_dir is not None:
            out.append(f"  this source tree  -> {self.repo_web_dir}")
            out.append("  the two are "
                       + ("DIFFERENT" if self.served_is_a_snapshot else "the same"))
        out.append("")
        if self.missing_from_served:
            n = len(self.missing_from_served)
            out.append(f"  {n} file(s) in this source tree are NOT in the "
                       f"directory being served:")
            out.extend(f"      {name}" for name in self.missing_from_served[:10])
            if n > 10:
                out.append(f"      ... and {n - 10} more")
            out.append("")
            out.append("  A page whose module is missing does not error: the "
                       "browser asks for it,")
            out.append("  gets the single-page shell back, fails on the MIME "
                       "type and renders")
            out.append("  nothing. Reinstall to fix these; the failure returns "
                       "with the next")
            out.append("  file added, so prefer `pip install -e .` for a "
                       "checkout you edit.")
        else:
            out.append("  Every file in this source tree is present in the "
                       "directory being served.")
        return out


def _installed_search_path() -> list[str]:
    """`sys.path` with the caller's own directory removed.

    `importlib.metadata` searches `sys.path`, so from a checkout it finds the
    `portfolio_tracker.egg-info` left there by a build and answers from that:
    relative paths, and no `direct_url.json`, so the install looks
    non-editable when it is editable. Measured, not assumed -- the same call
    from `/tmp` finds the real dist-info in site-packages.

    Dropping "" (which means the current directory) and the checkout root is
    what makes the answer independent of where this was run.
    """
    here = pathlib.Path(__file__).resolve().parent.parent
    out = []
    for entry in sys.path:
        if not entry:
            continue                       # "" is the current directory
        try:
            resolved = pathlib.Path(entry).resolve()
        except OSError:
            continue
        if resolved == here or resolved == pathlib.Path.cwd():
            continue
        out.append(entry)
    return out


def _distribution(path: list[str] | None = None):
    """The distribution, optionally searching an explicit path."""
    import importlib.metadata as md
    if path is None:
        return md.distribution(DISTRIBUTION)
    context = md.DistributionFinder.Context(name=DISTRIBUTION, path=path)
    for dist in md.Distribution.discover(context=context):
        return dist
    raise md.PackageNotFoundError(DISTRIBUTION)


def _editable(dist) -> bool | None:
    """Is the distribution installed in editable mode? None if unknowable."""
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return bool(json.loads(raw).get("dir_info", {}).get("editable", False))
    except (ValueError, AttributeError):
        return None


def inspect_installation(web_dir: pathlib.Path | None = None) -> Installation:
    """Collect every angle on "which copy is running", and compare them.

    `web_dir` defaults to the one the API actually mounts, imported lazily so
    this works without the app extra.
    """
    try:
        import portfolio
        imported = pathlib.Path(portfolio.__file__).resolve().parent
    except Exception:
        imported = None

    # `sys.path[0]` is "" for `python -c` and for an interactive session, and
    # the script's directory otherwise. Either way, a `portfolio` package
    # sitting in it wins over anything installed.
    first = sys.path[0] if sys.path else ""
    cwd_candidate = (pathlib.Path(first or ".").resolve() / "portfolio")
    shadowed = bool(imported and cwd_candidate.exists()
                    and cwd_candidate.resolve() == imported)

    location = dist_info = None
    version = None
    editable = None
    naive_dist_info = None
    try:
        dist = _distribution(_installed_search_path())
        version = dist.version
        dist_info = pathlib.Path(str(dist._path)).resolve()
        location = pathlib.Path(str(dist.locate_file("portfolio"))).resolve()
        editable = _editable(dist)
    except Exception:
        pass
    try:
        naive = _distribution()
        naive_dist_info = pathlib.Path(str(naive._path))
    except Exception:
        pass

    if web_dir is None:
        try:
            from .api.app import WEB_DIR
            web_dir = WEB_DIR
        except Exception:
            web_dir = None

    repo_web = pathlib.Path(__file__).resolve().parent / "web"
    repo_web = repo_web if repo_web.exists() else None

    missing: tuple[str, ...] = ()
    if web_dir is not None and repo_web is not None:
        if web_dir.resolve() != repo_web.resolve():
            missing = tuple(
                str(p.relative_to(repo_web))
                for p in sorted(repo_web.rglob("*"))
                if p.is_file() and not (web_dir / p.relative_to(repo_web)).exists())

    return Installation(
        imported_from=imported, shadowed_by_cwd=shadowed,
        metadata_location=location, dist_info=dist_info, version=version,
        editable=editable, web_dir=web_dir, repo_web_dir=repo_web,
        missing_from_served=missing,
        naive_dist_info=naive_dist_info)
