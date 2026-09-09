"""FastAPI application factory.

    uvicorn portfolio.api.app:create_app --factory
    portfolio serve

Serves the JSON API under /api, the static client under /static, and
`index.html` for every other path so the client's hash router owns the URL.
"""

from __future__ import annotations

import pathlib

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..core.models import ValidationError
from ..core.money import MissingRate
from ..core.positions import InsufficientUnits
from ..core.universe import InstrumentInUse
from ..data.provider import ProviderError, RateLimited
from .config import VERSION, Config
from .routes import ROUTERS
from .services import PortfolioService

__all__ = ["create_app", "WEB_DIR"]

WEB_DIR = pathlib.Path(__file__).resolve().parents[1] / "web"


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message}})


def create_app(config: Config | None = None) -> FastAPI:
    config = config or Config.from_env()
    app = FastAPI(title="Portfolio tracker", version=VERSION,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.config = config
    app.state.service = PortfolioService(config)

    for router in ROUTERS:
        app.include_router(router, prefix="/api")

    # -- errors: every failure leaves as a sentence in one envelope ---------

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        codes = {400: "bad_request", 404: "not_found", 409: "conflict", 501: "unsupported"}
        return _error(exc.status_code, codes.get(exc.status_code, "error"), str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(400, "invalid", f"{where}: {first.get('msg', 'invalid request')}")

    @app.exception_handler(ValidationError)
    async def _domain(_: Request, exc: ValidationError):
        return _error(400, "validation", str(exc))

    @app.exception_handler(InsufficientUnits)
    async def _units(_: Request, exc: InsufficientUnits):
        return _error(400, "insufficient_units", str(exc))

    @app.exception_handler(InstrumentInUse)
    async def _in_use(_: Request, exc: InstrumentInUse):
        return _error(409, "instrument_in_use", str(exc))

    @app.exception_handler(MissingRate)
    async def _rate(_: Request, exc: MissingRate):
        return _error(409, "missing_rate", str(exc))

    @app.exception_handler(RateLimited)
    async def _limited(_: Request, exc: RateLimited):
        return _error(503, "rate_limited", str(exc))

    @app.exception_handler(ProviderError)
    async def _provider(_: Request, exc: ProviderError):
        return _error(502, "provider", str(exc))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        return _error(500, "internal", f"{type(exc).__name__}: {exc}")

    # -- the client ---------------------------------------------------------

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(WEB_DIR / "index.html")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            """The client router owns every path that is not a file.

            "Not a file" is load-bearing. This used to return `index.html` for
            anything outside `/api/`, which meant a request for a missing
            asset came back **200 with HTML**: a browser importing a module
            and receiving `text/html` fails on the MIME type, so a file that
            was simply absent presented as a blank page with no 404 anywhere
            to find. `curl /pages/allocate.js` answered 200 whether or not the
            file existed, which is a diagnostic that cannot fail.

            A last segment with an extension is a request for a file. If the
            static mount did not serve it, it is not there, and saying so is
            the whole job.
            """
            if path.startswith("api/"):
                raise HTTPException(status_code=404, detail=f"no such endpoint: /{path}")
            if "." in path.rsplit("/", 1)[-1]:
                raise HTTPException(
                    status_code=404,
                    detail=(f"no such file: /{path}. If the client imports it, "
                            f"it is missing from the directory this server "
                            f"serves ({WEB_DIR})."))
            return FileResponse(WEB_DIR / "index.html")

    return app
