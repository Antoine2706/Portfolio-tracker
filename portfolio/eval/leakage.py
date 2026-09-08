"""Proving the harness cannot see the future, by changing the future.

This project has now met the same class of bug three times: a check that
looked like it worked without exercising what it claimed. The gate that fired
before the staleness rule and left it dead. The layering guard that needed
deliberate sabotage to show it failed when violated. The CI test that passed
locally only because the machine had no network. Each one produced a green
tick and measured nothing.

A backtest is the same artefact at a larger scale. It returns a confident
Sharpe ratio whether or not the decisions behind it were takeable, and the
single most common reason they were not is that something in the pipeline
read a price that had not happened yet. Reviewing for that by eye does not
work: the leak is usually one index, in a line that looks right.

So it is tested by experiment rather than inspection.

The experiment
--------------
Run the backtest. Then rewrite every price *strictly after* some date T into a
completely different path, and run it again. Everything the first run decided
on or before T, and every return it realised on or before T, must come back
bit-identical -- because none of it was allowed to depend on anything that
changed.

If any of it moves, something read past T. It does not matter where: the
estimator, a normaliser fitted on the whole sample, an execution price taken
from the wrong bar, or a policy closing over the full frame instead of using
the view it was handed. All of them show up as the same failure, which is
what makes this stronger than a checklist of the traps in section 2.6 -- it
catches the trap nobody wrote down.

The detector is only worth having if it can fail. `portfolio/tests/test_leakage.py`
runs it against a policy that deliberately reads tomorrow's close and asserts
that it screams. A leak detector that has never caught a leak is a green tick
with nothing behind it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .harness import BacktestResult, Panel

__all__ = ["LeakReport", "VacuousLeakCheck", "perturb_after",
           "check_lookahead", "DEFAULT_SPLIT_FRACTIONS", "default_splits"]


class VacuousLeakCheck(RuntimeError):
    """The perturbation changed nothing, so the check proved nothing.

    Raised when rewriting every price after the split leaves even the returns
    *after* the split identical. That cannot happen if the backtest is reading
    the panel it was handed, so it means the `run` callable ignored its
    argument -- most often by closing over the original panel when building
    its policy or cost model.

    This is the failure this whole module exists to prevent, turned on itself.
    A leak detector that silently passes because it was wired up wrongly is
    worse than no leak detector, because it produces a green tick that
    everything downstream is then trusted on.
    """


@dataclasses.dataclass(frozen=True)
class LeakReport:
    """What survived the future being rewritten."""
    leaked: bool
    split: pd.Timestamp
    compared_returns: int
    compared_decisions: int
    first_divergence: pd.Timestamp | None
    detail: str

    def __bool__(self) -> bool:
        """True when clean, so `assert check_lookahead(...)` reads correctly."""
        return not self.leaked


def perturb_after(panel: Panel, split: int, seed: int = 20260908) -> Panel:
    """A panel identical up to row `split`, on a different path after it.

    The replacement is a fresh geometric random walk starting from each
    column's last unperturbed price, with a deliberately large volatility.
    Continuity at the join matters: a discontinuous jump would be detectable
    as an outlier return and might be filtered by a policy for the wrong
    reason, which would weaken the test. Everything after the join is
    unrelated to what really happened, which is the point.

    NaNs are preserved exactly. An instrument that was not trading on a date
    must still not be trading on it, or the perturbed run would differ because
    the universe changed rather than because something leaked.
    """
    if not 0 <= split < len(panel.closes) - 1:
        raise ValueError(
            f"split must leave at least one row to perturb; got {split} in a "
            f"panel of {len(panel.closes)}")
    rng = np.random.default_rng(seed)

    def rewrite(frame: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        n_post = len(frame) - split - 1
        for col in frame.columns:
            base = anchor[col].iloc[:split + 1].dropna()
            start = float(base.iloc[-1]) if len(base) else 100.0
            steps = rng.normal(0.0015, 0.05, n_post)
            path = start * np.exp(np.cumsum(steps))
            tail = out[col].iloc[split + 1:]
            out.iloc[split + 1:, out.columns.get_loc(col)] = np.where(
                tail.isna(), np.nan, path)
        return out

    closes = rewrite(panel.closes, panel.closes)
    opens = None if panel.opens is None else rewrite(panel.opens, panel.closes)
    return Panel(closes=closes, opens=opens)


def _decision_key(d) -> tuple:
    """Everything the POLICY produced, so any of it moving counts as a leak.

    `proposed`, not `weights_after`. The executed target is allowed to depend
    on the execution bar -- a frozen holding's weight is read from the drifted
    book on the day the order is placed, which is after the decision -- so
    comparing it would report a leak for a policy that never looked forward.
    That false positive is not hypothetical: it appeared the first time a
    frozen holding was run through this, and the fix was to record what the
    policy said separately from what the harness did with it.

    The *reason* and confidence are included because they are outputs of the
    policy too, and they are often the more sensitive signal: a policy that
    peeks one bar ahead over four assets picks the same asset a quarter of the
    time by chance even when the future has been rewritten, but a reason
    string quoting the number it saw changes every time. Detection should not
    depend on a coin flip.
    """
    return (tuple(sorted((k, round(float(v), 12)) for k, v in d.proposed.items())),
            d.reason, round(float(d.confidence), 12))


# Spread across the sample so a leak is not missed because one split landed
# in a quiet stretch. Fractions of the panel rather than absolute rows, so the
# same defaults work on any length of history.
DEFAULT_SPLIT_FRACTIONS = (0.45, 0.55, 0.65, 0.75, 0.85)


def default_splits(panel: Panel) -> list[int]:
    """Row indices at the default fractions, deduplicated and in range."""
    n = len(panel.closes)
    out = sorted({min(n - 2, max(1, int(n * f))) for f in DEFAULT_SPLIT_FRACTIONS})
    return out


def check_lookahead(run, panel: Panel, splits=None, seed: int = 20260908) -> LeakReport:
    """Run the backtest twice per split and compare what must not have moved.

    `splits` is a row index or an iterable of them. Several is the default for
    a reason: a policy that peeks exactly one bar ahead only touches perturbed
    data at the single decision sitting on the split, so one split gives one
    chance to notice. Over four assets that is a 75% chance, which is not a
    standard to hold a leak detector to. Five splits make it 99.9%, and the
    cost is five cheap backtests.

    `run` takes a Panel and returns a BacktestResult. It must build
    *everything that reads prices* from the panel it is given -- the policy
    included. Closing over the outer panel instead is the one way to make
    this check pass while testing nothing, and it is caught: see
    `VacuousLeakCheck`.

    The rebalance interval, cost model and warmup should be closed over, so
    that this function has no way to vary them between the two runs.
    """
    if splits is None:
        splits = default_splits(panel)
    wanted = [splits] if isinstance(splits, int) else list(splits)
    if not wanted:
        raise ValueError("at least one split is needed")
    original: BacktestResult = run(panel)

    # A split before the first decision has nothing for the perturbation to
    # affect. The default fractions cannot know the warmup, so the usable
    # ones are selected here rather than making every caller do the
    # arithmetic -- and it is only an error if none of them are usable.
    positions = {d: i for i, d in enumerate(panel.closes.index)}
    first_decision = min((positions[d.decided_on] for d in original.decisions),
                         default=None)
    if first_decision is None:
        raise ValueError(
            "the backtest took no decisions at all, so there is nothing to "
            "check for look-ahead")
    usable = [int(w) for w in wanted if int(w) >= first_decision]
    if not usable:
        raise ValueError(
            f"no decision was taken at or before row {max(wanted)}; the first "
            f"is at row {first_decision}. Choose a split after the warmup "
            f"period.")

    last: LeakReport | None = None
    for want in usable:
        report = _check_one(run, panel, original, int(want), seed)
        if report.leaked:
            return report
        last = report
    assert last is not None
    return last


def _check_one(run, panel: Panel, original: BacktestResult, split: int,
               seed: int) -> LeakReport:

    # The split is snapped BACK to the latest decision day at or before it.
    #
    # This is not a detail. A policy that peeks exactly one bar ahead reads
    # bar d+1 when deciding on bar d, so it only touches perturbed data if
    # some decision sits on the split itself. With monthly rebalancing an
    # arbitrary split falls between decisions nineteen times in twenty, the
    # peek reads an unperturbed bar, and the check passes on a policy that is
    # openly reading the future -- which is exactly how a leak detector comes
    # to certify a leak.
    positions = {d: i for i, d in enumerate(panel.closes.index)}
    on_or_before = [positions[d.decided_on] for d in original.decisions
                    if positions[d.decided_on] <= split]
    if not on_or_before:
        raise ValueError(
            f"no decision was taken at or before row {split}; there is "
            f"nothing for the perturbation to affect. Choose a split after "
            f"the warmup period.")
    split = max(on_or_before)

    altered: BacktestResult = run(perturb_after(panel, split, seed))
    cut = panel.closes.index[split]


    # Before comparing what must NOT have changed, confirm that something
    # did. If the whole series is identical then the second run never saw the
    # rewritten prices, and every comparison below would pass for a reason
    # that has nothing to do with look-ahead.
    after_a = original.returns[original.returns.index > cut]
    after_b = altered.returns[altered.returns.index > cut]
    unchanged_after = (len(after_a) == len(after_b)
                       and (len(after_a) == 0
                            or bool(np.array_equal(after_a.to_numpy(),
                                                   after_b.to_numpy()))))
    if unchanged_after:
        raise VacuousLeakCheck(
            f"rewriting every price after {cut:%Y-%m-%d} left the returns "
            f"after it identical as well, which is impossible for a backtest "
            f"that is reading the panel it was handed. The `run` callable is "
            f"ignoring its argument -- typically by building its policy from "
            f"an outer-scope panel instead of the one passed in. Fix the "
            f"wiring: as it stands this check would pass whatever the code "
            f"under test did.")

    a = original.returns[original.returns.index <= cut]
    b = altered.returns[altered.returns.index <= cut]

    if not a.index.equals(b.index):
        return LeakReport(
            True, cut, len(a), 0, None,
            f"the two runs produced different dates before the split "
            f"({len(a)} against {len(b)}); the panel's shape leaked")

    first = None
    if len(a):
        diff = (a.to_numpy() - b.to_numpy())
        bad = np.flatnonzero(np.abs(diff) > 0)
        if bad.size:
            first = a.index[int(bad[0])]

    da = [d for d in original.decisions if d.decided_on <= cut]
    db = [d for d in altered.decisions if d.decided_on <= cut]
    decision_detail = ""
    if len(da) != len(db):
        decision_detail = (f"{len(da)} decisions before the split against "
                           f"{len(db)} after the future was rewritten")
    else:
        for x, y in zip(da, db):
            if _decision_key(x) != _decision_key(y):
                decision_detail = (
                    f"the decision taken on {x.decided_on:%Y-%m-%d} changed "
                    f"when only data after {cut:%Y-%m-%d} was altered")
                first = first or x.decided_on
                break

    leaked = first is not None or bool(decision_detail)
    if leaked:
        detail = decision_detail or (
            f"the return realised on {first:%Y-%m-%d} changed when only data "
            f"after {cut:%Y-%m-%d} was altered")
        detail += (". Something in the pipeline read a price that had not "
                   "happened yet at the time it was used.")
    else:
        detail = (f"{len(a)} returns and {len(da)} decisions up to "
                  f"{cut:%Y-%m-%d} were unchanged by rewriting every price "
                  f"after it")
    return LeakReport(leaked, cut, len(a), len(da), first, detail)
