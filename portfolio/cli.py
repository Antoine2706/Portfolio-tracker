"""Command line entry point.

    portfolio serve                      # http://127.0.0.1:8765, opens the browser
    portfolio serve --mode user          # your own data
    portfolio serve --provider fixture   # fully offline demo, synthetic prices
    portfolio check                      # run the core test suite
    portfolio controls                   # calibrate the backtest harness
    portfolio backtest erc               # equal risk contribution vs buy-and-hold

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


# Written here rather than generated, because a pre-registration is a
# statement of what was expected *before* the run, and text derived from the
# result is not that.
CONTROL_HYPOTHESES = {
    "negative control": (
        "A policy that ignores every price, trading at the same turnover as a "
        "real one, has no edge; the harness should say so."),
    "positive control": (
        "A policy given a known probability of foreseeing the next period has "
        "a Sharpe ratio available in closed form; the harness should recover "
        "it within sampling error."),
    "look-ahead canary": (
        "A policy that decides using tomorrow's close should produce a Sharpe "
        "ratio far outside anything achievable, and be flagged as implausible."),
    "leak detector": (
        "Rewriting every price after a date should leave every decision taken "
        "before it unchanged, and should not leave them unchanged for a policy "
        "that reads ahead."),
}

CONTROL_EXPECTATIONS = {
    "negative control": (
        "No significant edge, a rejection rate near the nominal 5%, and "
        "t-statistics with standard deviation 1. A spread far from 1 would "
        "mean the standard error formula is wrong."),
    "positive control": (
        "Measured Sharpe within roughly two standard errors of the injected "
        "one at every skill level, including zero."),
    "look-ahead canary": (
        "An annualised Sharpe above 10, flagged suspicious. Anything quieter "
        "means the future is not reaching the strategy and the test proves "
        "nothing."),
    "leak detector": (
        "Silent on a clean policy, loud on a peeking one, and refusing to run "
        "when wired so that it could not fail."),
}


def _controls(args: argparse.Namespace) -> int:
    """Calibrate the harness and print the result.

    This is the gate on everything else in the evaluation work: until the
    three controls pass, a backtest result is a number with nothing behind
    it. Exits non-zero when any control fails, so it can be a CI step.
    """
    from .eval.controls import NEGATIVE_CONTROL_SEEDS, run_calibration

    if args.quick:
        report = run_calibration(seeds=args.seeds or 40, periods=700,
                                 warmup=126, positive_seeds=6)
    else:
        report = run_calibration(seeds=args.seeds or NEGATIVE_CONTROL_SEEDS)
    print("\n".join(report.lines()))

    if args.register:
        from .eval.registry import Preregistration, Registry, TrialResult
        registry = Registry()
        for outcome in report.outcomes:
            trial = registry.register(Preregistration(
                hypothesis=CONTROL_HYPOTHESES[outcome.name.split(" - ")[0]],
                policy=outcome.name.split(" - ")[0],
                parameters=outcome.numbers,
                success_criterion=outcome.name.split(" - ")[-1],
                expected_outcome=CONTROL_EXPECTATIONS[outcome.name.split(" - ")[0]],
                is_control=True))
            registry.record(TrialResult(
                trial_id=trial.id,
                sharpe_per_period=outcome.numbers.get("mean_sharpe"),
                observations=int(outcome.numbers.get("seeds", 0)),
                independent_observations=int(outcome.numbers.get("seeds", 0)),
                verdict="passed" if outcome.passed else "FAILED",
                met_criterion=outcome.passed))
        print()
        print(f"Registered in {registry.path}: {registry.summary()}")
    return 0 if report.passed else 1


def _backtest(args: argparse.Namespace) -> int:
    """Evaluate one policy against buy-and-hold on the real book.

    Prints gross and net side by side, the turnover at which the gross edge
    is fully consumed, and the cost inputs that are estimates rather than
    observations -- because the largest single component of trading cost
    here, the bid-ask spread, is paid inside the execution price and appears
    on no contract note.
    """
    from .eval.registry import Preregistration, Registry, TrialResult
    from .research import load_book, run_equal_risk_contribution

    try:
        book = load_book(mode=args.mode, data_root=args.data_root,
                         provider=args.provider, lookback=args.history,
                         account_value=args.account_value)
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        print(f"Could not load the book: {exc}", file=sys.stderr)
        return 2

    if args.register and args.provider == "fixture":
        print("Refusing to register a run on synthetic prices. The trial count "
              "feeds the deflated Sharpe ratio, and an attempt that was never "
              "about the real portfolio would deflate every real attempt that "
              "follows it. Run against real prices, or drop --register.",
              file=sys.stderr)
        return 2

    registry = Registry()
    trial = None
    if args.register:
        trial = registry.register(Preregistration(
            hypothesis=(
                "Equalising each holding's contribution to portfolio volatility "
                "beats leaving the book alone, after the trading it requires."),
            policy="equal-risk-contribution",
            parameters={"lookback": args.lookback,
                        "rebalance_every": args.rebalance,
                        "frozen": sorted(book.frozen),
                        "provider": args.provider,
                        "account_value": book.account_value},
            success_criterion=(
                "net annualised Sharpe above buy-and-hold, with the gross edge "
                "surviving the policy's own turnover"),
            expected_outcome=(
                "A small reduction in volatility and a small improvement in "
                "risk-adjusted return, most or all of it consumed by turnover. "
                "Written before the run.")))

    comparison = run_equal_risk_contribution(
        book, lookback=args.lookback, rebalance_every=args.rebalance,
        trials=args.trials, trial_sharpe_sd=args.trial_spread)
    print("\n".join(comparison.lines()))

    if trial is not None:
        registry.record(TrialResult(
            trial_id=trial.id,
            sharpe_per_period=comparison.net.sharpe_per_period,
            observations=comparison.net.observations,
            independent_observations=comparison.net.independent_observations,
            verdict=comparison.net.verdict(),
            met_criterion=bool(
                comparison.net.sharpe is not None
                and comparison.benchmark.sharpe is not None
                and comparison.net.sharpe > comparison.benchmark.sharpe)))
        print()
        print(f"Registered in {registry.path}: {registry.summary()}")
    return 0


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

    controls = sub.add_parser(
        "controls",
        help="run the three harness controls and report whether it discriminates")
    controls.add_argument("--seeds", type=int, default=None,
                          help="negative-control runs (default 200; see the "
                               "note on the upper bound in eval/controls.py)")
    controls.add_argument("--quick", action="store_true",
                          help="a smaller, faster version for a sanity check")
    controls.add_argument("--register", action="store_true",
                          help="append the controls to the pre-registration log")
    controls.set_defaults(func=_controls)

    backtest = sub.add_parser(
        "backtest", help="evaluate a policy against buy-and-hold on your book")
    backtest.add_argument("policy", choices=["erc"],
                          help="erc = equal risk contribution")
    backtest.add_argument("--mode", choices=["seed", "user"], default="user")
    backtest.add_argument("--provider", choices=["yfinance", "fixture"],
                          default="yfinance",
                          help="fixture = deterministic synthetic prices; the "
                               "machinery runs but the result is not about "
                               "your holdings")
    backtest.add_argument("--data-root", default=None)
    backtest.add_argument("--lookback", type=int, default=252,
                          help="covariance window, in trading days")
    backtest.add_argument("--rebalance", type=int, default=21,
                          help="trading days between decisions")
    backtest.add_argument("--history", type=int, default=750,
                          help="price history to evaluate over, in rows")
    backtest.add_argument("--account-value", type=float, default=None,
                          help="notional for the cost model; defaults to the "
                               "book's own market value")
    backtest.add_argument("--trials", type=int, default=1,
                          help="pre-registered attempts, for the deflated Sharpe")
    backtest.add_argument("--trial-spread", type=float, default=0.0,
                          help="spread of Sharpe estimates across those trials")
    backtest.add_argument("--register", action="store_true",
                          help="append this attempt to the pre-registration log")
    backtest.set_defaults(func=_backtest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
