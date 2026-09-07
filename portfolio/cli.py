"""Command line entry point.

    portfolio serve                      # http://127.0.0.1:8765, opens the browser
    portfolio serve --mode user          # your own data
    portfolio serve --provider fixture   # fully offline demo, synthetic prices
    portfolio check                      # run the core test suite

Deliberately thin: argument parsing and process startup only. Every decision
about data lives in `api.services`.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser

__all__ = ["main"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The application extra is not installed. Run:\n"
              "    pip install -e \".[app,data]\"", file=sys.stderr)
        return 2

    # The app factory reads its configuration from the environment so that
    # `uvicorn portfolio.api.app:create_app --factory` behaves identically.
    if args.mode:
        os.environ["PORTFOLIO_DATA_MODE"] = args.mode
    if args.provider:
        os.environ["PORTFOLIO_PROVIDER"] = args.provider
    if args.data_root:
        os.environ["PORTFOLIO_DATA_ROOT"] = args.data_root

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        # Open once the server is listening rather than racing it.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Portfolio tracker serving at {url}")
    uvicorn.run("portfolio.api.app:create_app", factory=True, host=args.host,
                port=args.port, reload=args.reload, log_level=args.log_level)
    return 0


def _check(_: argparse.Namespace) -> int:
    try:
        import pytest
    except ImportError:
        print("pytest is not installed. Run: pip install -e \".[test]\"", file=sys.stderr)
        return 2
    return int(pytest.main(["-q"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="portfolio",
                                     description="Portfolio risk tracker")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the web application")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--mode", choices=["seed", "user"], default=None,
                       help="seed = demo data, user = your data (default: seed)")
    serve.add_argument("--provider", choices=["yfinance", "fixture"], default=None,
                       help="fixture = deterministic offline prices")
    serve.add_argument("--data-root", default=None,
                       help="directory holding seed/, user/ and the price cache")
    serve.add_argument("--no-browser", action="store_true")
    serve.add_argument("--reload", action="store_true", help="developer auto-reload")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=_serve)

    check = sub.add_parser("check", help="run the test suite")
    check.set_defaults(func=_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
