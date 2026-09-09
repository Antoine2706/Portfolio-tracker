"""Every module the client imports must actually be served, at its own path.

The defect this exists for
--------------------------
The allocate page was reported as not existing. It did exist, in the clone and
in git; it was not in the directory the running server serves from, and three
things conspired to make that unreadable:

    GET /pages/allocate.js         200 text/html    <- the SPA catch-all
    GET /static/pages/allocate.js  404              <- the real answer
    GET /static/pages/overview.js  200 text/javascript

`app.mount("/static", StaticFiles(directory=WEB_DIR))` serves real files, and
`WEB_DIR` is `Path(portfolio/api/app.py).parents[1] / "web"` -- resolved
through the *imported* package. Under `pip install -e .` that is the clone and
is always current. Under a plain `pip install .` it is a **copy** in
site-packages taken when the install ran, and `package-data = ["web/**/*"]` is
a build-time glob, so a page added afterwards is simply not in the copy.
Same path, different vintage: overview.js was there at install time and
allocate.js was not.

The second half is worse than the first. `@app.get("/{path:path}")` returns
`index.html` for anything that is not `/api/...`, so a request for a missing
asset comes back **200 with HTML**. A browser importing a module and receiving
`text/html` fails on the MIME type and the page never mounts: the loudest
possible failure rendered as a blank screen. That is why the 200 on
`/pages/allocate.js` was not evidence of anything -- it is a false 200 that
appears whether or not the file exists.

What this test does
-------------------
Walks the client's whole import graph, and for each `/static/...` module asks
the app for that exact path and asserts it comes back 200 **as JavaScript**.

The content-type assertion is the load-bearing half. Without it the catch-all
would answer 200 for a missing file and this test would pass on the broken
state -- a check that cannot fail, which is the thing this project keeps
finding. `test_a_missing_page_is_caught` deletes a page and requires this test
to notice, so the check is demonstrated rather than assumed.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from portfolio.api.app import WEB_DIR

# `from "/static/..."`, `import "/static/..."` and dynamic `import("/static/...")`.
IMPORT = re.compile(r"""(?:from|import)\s*\(?\s*["'](/static/[^"']+)["']""")

JS_TYPES = ("text/javascript", "application/javascript")


def client_imports() -> dict[str, list[str]]:
    """Every /static path the client imports, and which files import it."""
    found: dict[str, list[str]] = {}
    for source in sorted(WEB_DIR.rglob("*.js")):
        if "vendor" in source.parts:
            continue                      # vendored bundles import nothing of ours
        for path in IMPORT.findall(source.read_text(encoding="utf-8")):
            found.setdefault(path, []).append(
                str(source.relative_to(WEB_DIR)))
    return found


def registry_imports() -> list[str]:
    """Just the ones `pages/index.js` imports -- the page registry itself."""
    index = WEB_DIR / "pages" / "index.js"
    return IMPORT.findall(index.read_text(encoding="utf-8"))


class TestThePageRegistryResolves:
    """The loop that would have caught it the moment the page was added."""

    def test_the_registry_is_not_empty(self):
        """A test that iterates an empty list passes and proves nothing."""
        assert len(registry_imports()) >= 9, registry_imports()

    @pytest.mark.parametrize("path", registry_imports())
    def test_every_registered_page_is_served_as_javascript(self, client, path):
        response = client.get(path)
        assert response.status_code == 200, (
            f"{path} is imported by pages/index.js and the server answers "
            f"{response.status_code}. The file is missing from the directory "
            f"the app serves ({WEB_DIR}), which under a non-editable install "
            f"is a copy taken when pip ran.")
        assert response.headers["content-type"].split(";")[0] in JS_TYPES, (
            f"{path} came back as "
            f"{response.headers['content-type']!r}. That is the SPA catch-all "
            f"answering for a file that is not there; a browser importing it "
            f"as a module fails on the MIME type and the page never mounts.")

    def test_allocate_specifically(self, client):
        """Named because it is the one that was reported missing."""
        r = client.get("/static/pages/allocate.js")
        assert r.status_code == 200
        assert "AllocatePage" in r.text or "export default" in r.text


class TestTheWholeImportGraphResolves:
    """Not only pages: a component or lib that 404s breaks the app equally."""

    @pytest.mark.parametrize("path", sorted(client_imports()))
    def test_every_imported_module_is_served_as_javascript(self, client, path):
        response = client.get(path)
        importers = ", ".join(client_imports()[path])
        assert response.status_code == 200, f"{path}, imported by {importers}"
        assert response.headers["content-type"].split(";")[0] in JS_TYPES, (
            f"{path}, imported by {importers}, came back as "
            f"{response.headers['content-type']!r}")


class TestTheCheckBites:
    """A registry test that cannot fail is worse than no registry test."""

    def test_a_missing_page_is_caught(self, client, monkeypatch, tmp_path):
        """Move a page out of the served directory and confirm the assertion
        above would fail. This is the exact shape of the reported bug: the
        file is in git, and absent from what the server serves."""
        page = WEB_DIR / "pages" / "allocate.js"
        hidden = tmp_path / "allocate.js"
        page.rename(hidden)
        try:
            response = client.get("/static/pages/allocate.js")
            # Either a clean 404, or the catch-all's false 200 -- and the
            # content-type assertion is what catches the second.
            assert (response.status_code == 404
                    or response.headers["content-type"].split(";")[0]
                    not in JS_TYPES), (
                "a page absent from the served directory still came back as "
                "JavaScript, so this whole file proves nothing")
        finally:
            hidden.rename(page)

    def test_a_missing_asset_does_not_masquerade_as_html(self, client):
        """The catch-all must not answer a request that is plainly for a file.

        Before this, /static was the only path that told the truth: everything
        else answered 200 with the SPA shell, so `curl /pages/allocate.js`
        looked healthy while the app was broken.
        """
        r = client.get("/pages/allocate.js")
        assert r.status_code == 404, (
            "a path ending in .js is a request for a file, and answering it "
            "with the SPA shell turns 'missing' into 'present but the wrong "
            "type', which is two hours of checking the browser cache")


class TestTheServedDirectoryMatchesTheClone:
    """The stale-install failure mode, caught before a browser is opened."""

    def test_every_page_in_the_repo_is_in_the_served_directory(self):
        """`WEB_DIR` resolves through the imported package. If that is a copy
        in site-packages rather than this clone, a page added since the last
        `pip install .` is missing and nothing says so until a page is blank.
        """
        here = pathlib.Path(__file__).resolve().parents[2] / "web"
        if here.resolve() == WEB_DIR.resolve():
            pytest.skip("editable install: the served directory is the clone, "
                        "so this comparison has nothing to say")
        missing = sorted(
            p.relative_to(here) for p in here.rglob("*.js")
            if not (WEB_DIR / p.relative_to(here)).exists())
        assert not missing, (
            f"{len(missing)} file(s) in the repo are absent from the "
            f"directory the app serves ({WEB_DIR}): {missing[:5]}. "
            f"Reinstalling fixes today's; the failure mode returns with the "
            f"next page added.")
