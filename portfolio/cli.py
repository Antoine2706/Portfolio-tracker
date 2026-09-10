"""Command line entry point.

    portfolio serve                      # http://127.0.0.1:8765, opens the browser
    portfolio serve --mode user          # your own data
    portfolio serve --provider fixture   # fully offline demo, synthetic prices
    portfolio check                      # run the core test suite
    portfolio doctor                     # which copy is running, and what it serves
    portfolio instruments list           # where each holding is held, and its costs
    portfolio instruments set ISIN ...   # record a broker, a tax band, a freeze
    portfolio controls                   # calibrate the backtest harness
    portfolio backtest erc               # equal risk contribution vs buy-and-hold
    portfolio allocate 5000              # where new money should go
    portfolio spreads                    # each instrument's bid-ask spread

Deliberately thin: argument parsing and process startup only. Every decision
about data lives in `api.services`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
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

    # Say which client is about to be served, before anything renders. A page
    # whose module is missing from the served directory does not error: the
    # browser gets the single-page shell back, fails on the MIME type, and
    # shows nothing. One line here is the difference between that and a
    # two-hour hunt through the browser cache.
    from .diagnostics import inspect_installation
    installed = inspect_installation()
    if not installed.healthy:
        print("", file=sys.stderr)
        for line in installed.lines():
            print(line, file=sys.stderr)
        print("", file=sys.stderr)
        print("Serving anyway. Pages listed above will render blank.",
              file=sys.stderr)
    elif installed.served_is_a_snapshot:
        print(f"Client served from {installed.web_dir} (a copy, not this "
              f"checkout).")

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        # Open once the server is listening rather than racing it.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Portfolio tracker serving at {url}")
    uvicorn.run("portfolio.api.app:create_app", factory=True, host=args.host,
                port=args.port, reload=args.reload, log_level=args.log_level)
    return 0


def _doctor(_: argparse.Namespace) -> int:
    """Which copy of this package is running, and does it serve every page.

    Exists because the obvious way to ask that question cannot answer it:
    `python -c "import portfolio"` run from a checkout reports the checkout
    whatever is installed, because `sys.path[0]` is the current directory.
    The console script has no such problem, so run this rather than that.
    """
    from .diagnostics import inspect_installation
    report = inspect_installation()
    print("\n".join(report.lines()))
    return 0 if report.healthy else 1


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

    if getattr(args, "spread_reference", None):
        return _spread_reference(args)
    if getattr(args, "spread_ladder", False):
        return _spread_ladder(args)

    if args.spread:
        # A separate family: these calibrate the spread estimator against
        # simulated markets, not the backtest harness against simulated
        # returns. Run on their own so the estimator can be re-validated
        # without paying for the harness controls, and vice versa.
        from .eval.spread_controls import run_spread_controls
        reports = run_spread_controls(quick=args.quick)
        print("Spread estimator calibration")
        print("=" * 60)
        print()
        for report in reports:
            print("\n".join(report.lines()))
            print()
        print("-" * 60)
        if all(r.passed for r in reports):
            print("The estimator recovers spreads it was not told, declines")
            print("the ones it cannot resolve, and its error bar predicts its")
            print("own dispersion. Estimates from it may enter the cost model.")
            return 0
        print("AT LEAST ONE CONTROL FAILED. No spread estimate means anything")
        print("until the failure above is understood; the cost model should")
        print("keep its declared constant instead.")
        return 1

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


def _parse_rung(text: str):
    """`SYMBOL=LOW-HIGH[:REGION]`, e.g. `VWCE.DE=1-6:EU`."""
    import math
    from .research import Rung
    try:
        # rsplit, because Yahoo symbols carry "=" themselves (ES=F, EURUSD=X)
        # and the band is always the last "="-separated piece.
        symbol, rest = text.rsplit("=", 1)
        band, _, region = rest.partition(":")
        low, high = (float(x) for x in band.split("-", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected SYMBOL=LOW-HIGH[:REGION], the band in "
            f"half-spread bps, e.g. VWCE.DE=1-6:EU") from None
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
        raise argparse.ArgumentTypeError(f"{text!r}: the band must be 0 <= "
                                         f"LOW < HIGH, both finite")
    region = (region or ("EU" if "." in symbol else "US")).upper()
    if region not in ("US", "EU", "THIN"):
        raise argparse.ArgumentTypeError(f"{text!r}: region must be US, EU "
                                         f"or THIN")
    return Rung(symbol.strip(), "supplied on the command line", low, high,
                region if region != "THIN" else "thin")


def _parse_pair(text: str) -> tuple[str, str]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"{text!r}: expected two symbols, "
                                         f"A,B")
    return parts[0], parts[1]


def _spread_ladder(args: argparse.Namespace) -> int:
    """The estimator on instruments whose spread is not in doubt.

    A control on the data path rather than on the estimator, and one that
    needs the real provider: see the note above `research.LADDER`. Exits
    non-zero on any rung outside its band, so that it can gate `--write`
    by hand the way the ranking check gates it automatically.
    """
    from .research import LADDER, PAIRS, run_ladder

    rungs = tuple(args.rung) if args.rung else LADDER
    if args.rung and not args.replace_ladder:
        rungs = LADDER + tuple(args.rung)
    pairs = tuple(args.pair) if args.pair else PAIRS
    if args.provider == "fixture":
        print("The fixture provider invents bars whose spreads bear no "
              "relation to the bands,")
        print("so this run is expected to FAIL. That is the control being "
              "seen to bite, not a")
        print("result about any market.")
        print()
    report = run_ladder(provider=args.provider, data_root=args.data_root,
                        rungs=rungs, pairs=pairs)
    print("\n".join(report.lines()))
    return 0 if report.passed else 1


def _parse_date(text: str):
    import datetime as dt
    try:
        return dt.date.fromisoformat(text.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: expected YYYY-MM-DD") from None


def _spread_reference(args: argparse.Namespace) -> int:
    """T5, T6 and T7 on one symbol: see the note above `research.reference_check`."""
    from .research import reference_check
    report = reference_check(args.spread_reference, provider=args.provider,
                             data_root=args.data_root,
                             splits=tuple(args.split_at or ()))
    print("\n".join(report.lines()))
    return 0


def _backtest(args: argparse.Namespace) -> int:
    """Evaluate one policy against buy-and-hold on the real book.

    Prints gross and net side by side, the turnover at which the gross edge
    is fully consumed, and the cost inputs that are estimates rather than
    observations -- because the largest single component of trading cost
    here, the bid-ask spread, is paid inside the execution price and appears
    on no contract note.
    """
    from .eval.registry import Preregistration, Registry, TrialResult
    from .research import (load_book, replay_the_ledger,
                           run_equal_risk_contribution)

    try:
        book = load_book(mode=args.mode, data_root=args.data_root,
                         provider=args.provider, lookback=args.history,
                         account_value=args.account_value)
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        print(f"Could not load the book: {exc}", file=sys.stderr)
        return 2

    if args.policy == "allocator":
        # Not registered as a trial: the allocator makes no claim about
        # returns, so it consumes none of the deflation budget that exists to
        # keep return claims honest. Its claim is about risk structure, and
        # that is computed rather than estimated.
        try:
            replay = replay_the_ledger(
                book, mode=args.mode, data_root=args.data_root,
                lookback=args.lookback,
                on_sale="stop" if args.stop_at_first_sale else "prorata")
        except ValueError as exc:
            print(f"Could not replay the ledger: {exc}", file=sys.stderr)
            return 2
        print("\n".join(replay.lines()))
        return 0

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


def _rate(text: str) -> float:
    """Parse a rate written any of the ways a person actually writes one.

    A contract note says "0.120%" and a spreadsheet on a machine with a comma
    decimal separator says "0,0012". Both mean the same thing, and typing the
    wrong one of them into a CSV by hand is exactly the silent, plausible,
    wrong number this project keeps finding.

    >>> _rate("0.0012"), _rate("0,0012"), _rate("0.12%"), _rate("0,120 %")
    (0.0012, 0.0012, 0.0012, 0.0012)

    A bare number large enough to be a percentage typed without its sign is
    refused rather than accepted as a 12% tax:

    >>> _rate("0.12")
    Traceback (most recent call last):
        ...
    argparse.ArgumentTypeError: 0.12 as a bare rate means 12.0000%, which is far outside any transaction tax. Write it as '0.12%' if that is what you meant, or as a fraction such as 0.0012.
    """
    s = text.strip().replace(" ", "").replace("\xa0", "").replace(",", ".")
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        value = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value > 0.1:
        raise argparse.ArgumentTypeError(
            f"{s} as a bare rate means {value:.4%}, which is far outside any "
            f"transaction tax. Write it as '{s}%' if that is what you meant, "
            f"or as a fraction such as 0.0012.")
    if value < 0:
        raise argparse.ArgumentTypeError(f"a rate cannot be negative: {s}")
    return value


def _open_store(args: argparse.Namespace):
    import pathlib

    from .data.store import DataMode, DataStore
    root = pathlib.Path(args.data_root) if args.data_root else None
    return DataStore.open(DataMode(args.mode), root=root)


def _instrument_rows(instruments) -> list[str]:
    """The trading facts, as a table, so they can be read rather than inferred."""
    head = (f"{'ISIN':14}{'name':26}{'class':8}{'broker':10}{'trade':6}"
            f"{'transaction tax':17}{'half-spread':18}{'buy tax':8}")
    out = [head, "-" * len(head)]
    for isin in sorted(instruments):
        inst = instruments[isin]
        if inst.tob_rate is None:
            tax = "NOT RECORDED"
        else:
            tax = f"{inst.tob_rate:.4%} " + ("observed" if inst.tob_observed
                                             else "assumed")
        if inst.half_spread_bps is None:
            spread = "8.0 bps fallback"
        else:
            spread = f"{inst.half_spread_bps:.1f} bps " + (
                "obs" if inst.spread_observed else "est")
        out.append(
            f"{isin:14}{inst.display_name[:25]:26}{inst.asset_class.value:8}"
            f"{(inst.broker or '-'):10}{('yes' if inst.tradeable else 'NO'):6}"
            f"{tax:17}{spread:18}"
            f"{(f'{inst.buy_tax_rate:.2%}' if inst.buy_tax_rate else '-'):8}")
    return out


def _instruments(args: argparse.Namespace) -> int:
    """Read and edit the trading facts on the instrument records.

    These fields decide whether a trade can be priced at all, and before this
    existed the only way to populate them was to hand-edit a seven-column
    addition across every row of a CSV -- on a machine whose spreadsheet uses
    a comma decimal separator. That is precisely how a plausible wrong number
    gets into a cost model, so it is code with validation instead.
    """
    store = _open_store(args)
    instruments = store.load_instruments()
    if not instruments:
        print(f"no instruments in {store.directory}", file=sys.stderr)
        return 2

    if args.action == "list":
        print(f"{store.describe()}\n")
        print("\n".join(_instrument_rows(instruments)))
        return 0

    if not args.all and not args.isin:
        print("name at least one ISIN, or pass --all.", file=sys.stderr)
        return 2
    targets = sorted(instruments) if args.all else [i.strip().upper()
                                                    for i in args.isin]
    unknown = [i for i in targets if i not in instruments]
    if unknown:
        print(f"not in {store.instruments_path}: {', '.join(unknown)}",
              file=sys.stderr)
        return 2

    # An observed flag with no number behind it is an estimate wearing the
    # label of evidence, which is worse than either honestly.
    if args.observed and args.tob_rate is None:
        if any(instruments[i].tob_rate is None for i in targets):
            print("--observed marks a rate as read off a contract note, so it "
                  "needs that rate: pass --tob-rate too, or set it first.",
                  file=sys.stderr)
            return 2
    if args.spread_observed and args.half_spread_bps is None:
        if any(instruments[i].half_spread_bps is None for i in targets):
            print("--spread-observed needs --half-spread-bps: a spread cannot "
                  "be observed without a value.", file=sys.stderr)
            return 2
    if args.commission_observed and args.commission is None:
        if any(instruments[i].commission is None for i in targets):
            print("--commission-observed needs --commission: a charge cannot "
                  "be read off a confirmation without a figure.",
                  file=sys.stderr)
            return 2

    changes: list[str] = []
    for isin in targets:
        inst = instruments[isin]
        for field, value in (("broker", args.broker),
                             ("tradeable", args.tradeable),
                             ("tob_rate", args.tob_rate),
                             ("half_spread_bps", args.half_spread_bps),
                             ("buy_tax_rate", args.buy_tax_rate),
                             ("venue", args.venue),
                             ("commission", args.commission),
                             ("short_name", args.short_name),
                             ("note", args.note)):
            if value is None:
                continue
            before = getattr(inst, field)
            if before == value:
                continue
            inst.override(field, value)
            changes.append(f"{isin} {field}: {before!r} -> {value!r}")
        # The observed flags travel with the value they describe, and setting
        # a new rate without saying where it came from resets the claim.
        for flag, field, value in (("observed", "tob_observed", args.observed),
                                   ("spread_observed", "spread_observed",
                                    args.spread_observed),
                                   ("commission_observed", "commission_observed",
                                    args.commission_observed)):
            source = {"tob_observed": args.tob_rate,
                      "spread_observed": args.half_spread_bps,
                      "commission_observed": args.commission}[field]
            if value is None and source is None:
                continue
            new = bool(value) if value is not None else False
            if getattr(inst, field) != new:
                inst.override(field, new)
                changes.append(f"{isin} {field}: {new}")

    if not changes:
        print("nothing to change; every field already has the value asked for.")
        return 0
    store.save_instruments(instruments)
    print("\n".join(changes))

    from .agents.execution import BROKERS
    strangers = sorted({instruments[i].broker for i in targets
                        if instruments[i].broker
                        and instruments[i].broker not in BROKERS})
    if strangers:
        print(f"\nWarning: no fee schedule is recorded for "
              f"{', '.join(strangers)}, so the cost model will refuse to price "
              f"a trade there. Known: {', '.join(sorted(BROKERS))}.")
    print(f"\nWritten to {store.instruments_path}.\n")
    print("\n".join(_instrument_rows(instruments)))
    return 0


def _spreads(args: argparse.Namespace) -> int:
    """Estimate the bid-ask spread for every instrument, from its own bars.

    Read-only unless `--write` is given. Looking at what the estimator says
    and committing it to the instrument records are different actions, and a
    number that is about to become an input to every cost figure in the tool
    should have to be looked at first.
    """
    from .research import load_book, survey_spreads

    try:
        book = load_book(mode=args.mode, data_root=args.data_root,
                         provider=args.provider, lookback=args.history,
                         account_value=args.account_value)
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        print(f"Could not load the book: {exc}", file=sys.stderr)
        return 2

    survey = survey_spreads(book, mode=args.mode, data_root=args.data_root,
                            provider=args.provider,
                            fallback_bps=args.fallback)
    print("\n".join(survey.lines()))

    # Every run is recorded, and the failed ones are the point of recording:
    # a control that failed and kept its numbers is a result; one that failed
    # and scrolled off is an anecdote. Under the mode's own directory, next
    # to the instruments it describes, so the demo book and the real one do
    # not share a log.
    from .data.store import DataMode, DataStore
    from .research import record_survey
    root = pathlib.Path(args.data_root) if args.data_root else None
    store = DataStore.open(DataMode(args.mode), root=root)
    log = store.directory / "spread-runs.jsonl"
    try:
        run_number = record_survey(survey, log)
        print()
        print(f"Recorded as run {run_number} in {log}")
    except OSError as exc:
        print(f"Could not record the run in {log}: {exc}", file=sys.stderr)

    if not survey.ranking.passed:
        print()
        print("The ranking check did not pass. Nothing is written on that "
              "basis: an", file=sys.stderr)
        print("ordering that contradicts liquidity is more likely to be a "
              "wiring fault", file=sys.stderr)
        print("than a market fact, and writing it would bake the fault into "
              "every", file=sys.stderr)
        print("cost figure the tool prints.", file=sys.stderr)
        print(file=sys.stderr)
        print("The check cannot say which of its two inputs is wrong. Two "
              "things can:", file=sys.stderr)
        print("the bars set aside per instrument above, if the estimate "
              "with them excluded", file=sys.stderr)
        print("ranks differently; and `portfolio controls --spread-ladder`, "
              "which runs the", file=sys.stderr)
        print("same path on instruments whose spread is not in doubt.",
              file=sys.stderr)
        return 1

    if args.write:
        from .agents.spreads import BOUNDED, ESTIMATED
        instruments = store.load_instruments()
        counts = {ESTIMATED: 0, BOUNDED: 0}
        for isin, decision in survey.decisions.items():
            # A bound is written as well as a measurement, and stored under
            # its own source so that nothing downstream can mistake the two.
            # An instrument that was not measurable at all is left alone.
            if decision.source not in counts or isin not in instruments:
                continue
            instruments[isin] = dataclasses.replace(
                instruments[isin],
                half_spread_bps=round(decision.half_spread_bps, 2),
                spread_observed=False,
                spread_source=decision.source)
            counts[decision.source] += 1
        store.save_instruments(instruments)
        print()
        print(f"Wrote {counts[ESTIMATED]} measured half-spread(s) and "
              f"{counts[BOUNDED]} upper bound(s) to")
        print(f"{store.directory}, each under its own source. Instruments "
              f"that were not")
        print("measurable at all were left alone, keeping the declared "
              "constant.")
    return 0


def _allocate(args: argparse.Namespace) -> int:
    """Where a purchase of new money should go.

    No sale is ever proposed, so this is the only rebalancing channel in an
    account with no regular contribution and no leverage: the purchase was
    going to happen anyway, and choosing its destination costs nothing extra.
    """
    from .research import allocate_new_money, load_book

    try:
        book = load_book(mode=args.mode, data_root=args.data_root,
                         provider=args.provider, lookback=args.history,
                         account_value=args.account_value)
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        print(f"Could not load the book: {exc}", file=sys.stderr)
        return 2
    if args.amount <= 0:
        print("an amount to invest is needed, and it has to be positive",
              file=sys.stderr)
        return 2

    allocation = allocate_new_money(book, args.amount, lookback=args.lookback)
    print("\n".join(allocation.lines()))
    if args.target is not None:
        from .agents.allocate import (best_reachable_dispersion,
                                      cash_for_dispersion, pinned_holdings)
        from .core.returns import simple_returns
        from .core.risk import covariance_matrix
        window = simple_returns(book.panel.closes).dropna().iloc[-args.lookback:]
        cov = covariance_matrix(window)
        shared = dict(values=book.values, cov=cov, costs=book.costs,
                      buyable=book.buyable)
        needed = cash_for_dispersion(args.target, **shared)
        pinned = pinned_holdings(**shared)
        total = sum(book.values.values())
        print()
        if needed is None:
            # "No" is not a decision on its own, and with a pinned holding it
            # is not even "as much as possible": the floor bottoms out and
            # then rises as the money dilutes a risk share it may not buy.
            best, at = best_reachable_dispersion(**shared)
            print(f"No purchase reaches a dispersion of {args.target:.2f} "
                  f"buy-only, at any size this tool searched. The best it "
                  f"found was {best:.2f}, at about {at:,.0f} EUR against a "
                  f"book of {total:,.0f} EUR.")
        else:
            print(f"Reaching a dispersion of {args.target:.2f} buy-only would "
                  f"take about {needed:,.0f} EUR of new money, against a book "
                  f"of {total:,.0f} EUR.")
        if pinned:
            frozen = sum(book.values.get(k, 0.0) for k in pinned)
            many = len(pinned) > 1
            print(f"\nThat is a search rather than a proof. "
                  f"{len(pinned)} holding{'s' if many else ''} worth "
                  f"{frozen:,.0f} EUR cannot receive new money, so past some "
                  f"amount more cash dilutes {'them' if many else 'it'} "
                  f"towards a zero risk share faster than it evens the rest "
                  f"out, and the floor starts rising again. Every amount named "
                  f"above does reach what it claims; what a pinned holding "
                  f"costs is the guarantee that nothing smaller would.")
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

    def _store_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mode", choices=["seed", "user"], default="user")
        p.add_argument("--data-root", default=None,
                       help="directory holding seed/ and user/")

    instruments = sub.add_parser(
        "instruments",
        help="read and edit where a holding is held and what trading it costs")
    action = instruments.add_subparsers(dest="action", required=True)

    listing = action.add_parser("list", help="show the trading facts on record")
    _store_args(listing)
    listing.set_defaults(func=_instruments)

    setter = action.add_parser(
        "set", help="record a broker, a tax band, a spread or a freeze",
        description="Rates accept 0.0012, 0,0012 or 0.12% interchangeably.")
    _store_args(setter)
    setter.add_argument("isin", nargs="*", help="one or more ISINs")
    setter.add_argument("--all", action="store_true",
                        help="every instrument in the store")
    setter.add_argument("--broker", default=None,
                        help="the institution that holds it, e.g. MeDirect")
    tradeable = setter.add_mutually_exclusive_group()
    tradeable.add_argument("--tradeable", dest="tradeable", action="store_true",
                           default=None)
    tradeable.add_argument("--not-tradeable", dest="tradeable",
                           action="store_false", default=None,
                           help="no automated policy may change its weight")
    setter.add_argument("--tob-rate", type=_rate, default=None,
                        help="transaction tax, charged each way")
    seen = setter.add_mutually_exclusive_group()
    seen.add_argument("--observed", dest="observed", action="store_true",
                      default=None, help="the tax rate was read off a document")
    seen.add_argument("--assumed", dest="observed", action="store_false",
                      default=None)
    setter.add_argument("--half-spread-bps", type=float, default=None,
                        help="from observed quotes, not from a contract note")
    quoted = setter.add_mutually_exclusive_group()
    quoted.add_argument("--spread-observed", dest="spread_observed",
                        action="store_true", default=None)
    quoted.add_argument("--spread-estimated", dest="spread_observed",
                        action="store_false", default=None)
    setter.add_argument("--buy-tax-rate", type=_rate, default=None,
                        help="one-sided taxes, e.g. the French FTT")
    setter.add_argument("--venue", default=None,
                        help="MIC the trade executes on, off the confirmation. "
                             "Not the primary listing used to fetch prices: "
                             "the same holding traded XETA in February and "
                             "JPEU in June. It is the key for tick size")
    setter.add_argument("--commission", type=float, default=None,
                        help="what the broker actually charged, per order. "
                             "Per instrument because the same broker charges "
                             "differently by type: MeDirect takes nothing on "
                             "an ETF and 7.00 EUR on a share")
    charged = setter.add_mutually_exclusive_group()
    charged.add_argument("--commission-observed", dest="commission_observed",
                         action="store_true", default=None,
                         help="read off a confirmation")
    charged.add_argument("--commission-assumed", dest="commission_observed",
                         action="store_false", default=None)
    setter.add_argument("--short-name", default=None,
                        help="the label used on charts and in dense tables")
    setter.add_argument("--note", default=None)
    setter.set_defaults(func=_instruments)

    doctor = sub.add_parser(
        "doctor",
        help="which copy of this package is running, and does it serve every "
             "page")
    doctor.set_defaults(func=_doctor)

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
    controls.add_argument("--spread", action="store_true",
                          help="calibrate the bid-ask spread estimator instead "
                               "of the backtest harness")
    controls.add_argument("--spread-ladder", action="store_true",
                          dest="spread_ladder",
                          help="run the spread estimator's whole data path on "
                               "instruments whose spread is not in doubt "
                               "(SPY, AAPL, SU.PA, MEUD.PA, a thin fund) and "
                               "say which end of the path is wrong. Needs "
                               "the real provider")
    controls.add_argument("--provider", choices=["yfinance", "fixture"],
                          default="yfinance",
                          help="for --spread-ladder; the fixture is expected "
                               "to fail it, which is how the check is seen "
                               "to bite")
    controls.add_argument("--data-root", default=None,
                          help="for --spread-ladder: where the bar cache lives")
    controls.add_argument("--rung", action="append", type=_parse_rung,
                          default=None, metavar="SYMBOL=LOW-HIGH[:REGION]",
                          help="add a rung to the ladder, band in half-spread "
                               "bps, e.g. VWCE.DE=1-6:EU; repeatable")
    controls.add_argument("--replace-ladder", action="store_true",
                          help="with --rung: use only the rungs given")
    controls.add_argument("--pair", action="append", type=_parse_pair,
                          default=None, metavar="A,B",
                          help="two listings of the same fund to compare; "
                               "repeatable")
    controls.add_argument("--spread-reference", metavar="SYMBOL",
                          dest="spread_reference", default=None,
                          help="one symbol, three questions on the exact bars "
                               "the survey reads: the authors' bidask package "
                               "on the same arrays, adjusted against "
                               "unadjusted prices, and blocks by date "
                               "(--split-at); plus the four moment conditions "
                               "separately")
    controls.add_argument("--split-at", action="append", type=_parse_date,
                          default=None, metavar="YYYY-MM-DD",
                          help="with --spread-reference: cut the history here; "
                               "repeatable")
    controls.set_defaults(func=_controls)

    backtest = sub.add_parser(
        "backtest", help="evaluate a policy against buy-and-hold on your book")
    backtest.add_argument("policy", choices=["erc", "allocator"],
                          help="erc = equal risk contribution; allocator = "
                               "replay your real purchases with only the "
                               "destination changed")
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
    backtest.add_argument("--stop-at-first-sale", action="store_true",
                          help="allocator only: end the replay at the first "
                               "disposal instead of applying it to both arms "
                               "pro-rata. Every figure then describes a book "
                               "that really was held, over a shorter window")
    backtest.add_argument("--register", action="store_true",
                          help="append this attempt to the pre-registration log")
    backtest.set_defaults(func=_backtest)

    allocate = sub.add_parser(
        "allocate",
        help="where a purchase of new money should go, so risk shares even out")
    allocate.add_argument("amount", type=float,
                          help="how much new money there is, in EUR")
    allocate.add_argument("--mode", choices=["seed", "user"], default="user")
    allocate.add_argument("--provider", choices=["yfinance", "fixture"],
                          default="yfinance")
    allocate.add_argument("--data-root", default=None)
    allocate.add_argument("--lookback", type=int, default=252,
                          help="covariance window, in trading days")
    allocate.add_argument("--history", type=int, default=750,
                          help="price history to load, in rows")
    allocate.add_argument("--account-value", type=float, default=None)
    allocate.add_argument("--target", type=float, default=None,
                          help="also answer: how much would reaching this "
                               "dispersion take, buy-only?")
    allocate.set_defaults(func=_allocate)

    spreads = sub.add_parser(
        "spreads",
        help="estimate each instrument's bid-ask spread from its own bars")
    spreads.add_argument("--mode", choices=["seed", "user"], default="user")
    spreads.add_argument("--provider", choices=["yfinance", "fixture"],
                         default="yfinance")
    spreads.add_argument("--data-root", default=None)
    spreads.add_argument("--history", type=int, default=750,
                         help="price history to load, in rows")
    spreads.add_argument("--account-value", type=float, default=None)
    spreads.add_argument("--fallback", type=float, default=8.0,
                         help="the declared constant, in bps, charged where "
                              "an estimate cannot be supported (default 8)")
    spreads.add_argument("--write", action="store_true",
                         help="store the estimates that cleared every gate "
                              "onto the instrument records")
    spreads.set_defaults(func=_spreads)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
