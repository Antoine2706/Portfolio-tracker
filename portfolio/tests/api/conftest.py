"""API test fixtures: a throwaway copy of the seed data, the offline provider.

The whole suite runs with sockets blocked (see ../conftest.py). FastAPI's
TestClient talks to the app in-process over an ASGI transport, so nothing
here opens one, and the FixtureProvider never would.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

pytest.importorskip("fastapi", reason="API tests need the app extra")

from fastapi.testclient import TestClient  # noqa: E402

from portfolio.api.app import create_app  # noqa: E402
from portfolio.api.config import Config  # noqa: E402
from portfolio.data.store import DEFAULT_ROOT, DataMode  # noqa: E402

SEED = DEFAULT_ROOT / "seed"


@pytest.fixture
def data_root(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "data_store"
    shutil.copytree(SEED, root / "seed")
    (root / "user").mkdir()
    return root


@pytest.fixture
def app(data_root):
    return create_app(Config(mode=DataMode.SEED, provider="fixture", data_root=data_root))


@pytest.fixture
def client(app) -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def service(app):
    return app.state.service
