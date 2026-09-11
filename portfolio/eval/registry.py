"""The pre-registration log: what was tried, written down before it was tried.

This is not paperwork. It is an input to a formula.

The deflated Sharpe ratio corrects an observed Sharpe for the number of
attempts it was selected from, because the expected maximum of N estimates
drawn from a worthless null is strictly positive and grows with N. That
correction takes N as an argument. An N that quietly omits the variants that
did not work produces a threshold that is too low and a discovery that is not
one -- and the omission is invisible afterwards, because a discarded backtest
leaves no trace. So the count has to be recorded at the moment of the attempt,
by the person who would benefit from forgetting it.

The rules the format enforces
-----------------------------
*   A hypothesis, a success criterion and an expected outcome are all
    required and are all validated as non-empty. "I expect this to fail after
    costs" written before the run is a scientific statement; the same sentence
    written afterwards is a rationalisation, and the file cannot tell them
    apart unless the field was mandatory.
*   Append-only, exactly like the ledger. A result is a new line naming the
    trial it belongs to, never an edit of the trial's line. Rewriting history
    here would defeat the only thing the file is for.
*   Controls are registered too, and flagged, and excluded from the trial
    count. A control is a calibration of the instrument, not an attempt to
    find an edge: counting the look-ahead canary as a trial would inflate the
    deflation threshold for every real strategy that follows it.

The file is plain JSON Lines so it can be read without this module, and is
meant to be committed: a trial count that lives only on the machine that ran
it is a trial count nobody can check.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
import statistics
import uuid

__all__ = ["Preregistration", "TrialResult", "Registry", "DEFAULT_REGISTRY_PATH"]

DEFAULT_REGISTRY_PATH = pathlib.Path(__file__).resolve().parents[2] / "research" / "registry.jsonl"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@dataclasses.dataclass(frozen=True)
class Preregistration:
    """One declared attempt, written before the backtest runs."""
    hypothesis: str
    policy: str
    parameters: dict
    success_criterion: str
    expected_outcome: str
    is_control: bool = False
    id: str = dataclasses.field(default_factory=lambda: f"trial_{uuid.uuid4().hex[:10]}")
    registered_at: str = dataclasses.field(default_factory=_now)

    def __post_init__(self) -> None:
        for field in ("hypothesis", "policy", "success_criterion", "expected_outcome"):
            if not str(getattr(self, field)).strip():
                raise ValueError(
                    f"{field} is required. A pre-registration missing its "
                    f"{field.replace('_', ' ')} records that something was "
                    f"tried without recording what would have counted as it "
                    f"working, which is the part that makes the log worth "
                    f"keeping.")

    def as_row(self) -> dict:
        return {"kind": "trial", **dataclasses.asdict(self)}


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """What a registered trial actually produced."""
    trial_id: str
    sharpe_per_period: float | None
    observations: int
    independent_observations: int
    verdict: str
    met_criterion: bool | None = None
    detail: dict = dataclasses.field(default_factory=dict)
    recorded_at: str = dataclasses.field(default_factory=_now)

    def as_row(self) -> dict:
        return {"kind": "result", **dataclasses.asdict(self)}


class Registry:
    """Append-only JSON Lines log of pre-registered trials and their results."""

    def __init__(self, path: "pathlib.Path | str" = DEFAULT_REGISTRY_PATH) -> None:
        self.path = pathlib.Path(path)

    # -- writing -----------------------------------------------------------

    def register(self, entry: Preregistration) -> Preregistration:
        """Append a trial. Returns it, so the caller can hold its id."""
        self._append(entry.as_row())
        return entry

    def record(self, result: TrialResult) -> TrialResult:
        """Append a result. The trial it names must already be registered.

        Checked rather than assumed: a result whose trial is missing means
        either the registration was skipped or the id was mistyped, and both
        of those corrupt the trial count the deflation depends on.
        """
        known = {e["id"] for e in self.entries() if e.get("kind") == "trial"}
        if result.trial_id not in known:
            raise ValueError(
                f"{result.trial_id} was never registered. Register the trial "
                f"before running it -- a result without a pre-registration is "
                f"exactly what the deflation correction cannot see.")
        self._append(result.as_row())
        return result

    def _append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    # -- reading -----------------------------------------------------------

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def trials(self, include_controls: bool = False) -> list[dict]:
        return [e for e in self.entries()
                if e.get("kind") == "trial"
                and (include_controls or not e.get("is_control"))]

    def results(self) -> list[dict]:
        return [e for e in self.entries() if e.get("kind") == "result"]

    # -- what the deflation needs -----------------------------------------

    def deflation_inputs(self) -> tuple[int, float]:
        """(trial count, spread of their per-period Sharpe estimates).

        The count excludes controls, which calibrate the harness rather than
        search for an edge.

        The spread is the sample standard deviation of the Sharpe ratios the
        trials actually produced. With fewer than two recorded results it is
        undefined, and 0.0 is returned -- which sets the deflation threshold
        to zero and makes the deflated Sharpe equal the plain probabilistic
        one. That is the correct behaviour and not a silent degradation: with
        one trial there is no selection to correct for.

        >>> import tempfile, pathlib
        >>> d = pathlib.Path(tempfile.mkdtemp())
        >>> r = Registry(d / "reg.jsonl")
        >>> r.deflation_inputs()
        (0, 0.0)
        >>> t = r.register(Preregistration("h", "p", {}, "c", "e"))
        >>> r.deflation_inputs()
        (1, 0.0)
        >>> _ = r.record(TrialResult(t.id, 0.05, 252, 252, "ran"))
        >>> r.deflation_inputs()
        (1, 0.0)

        A control is registered but does not count as a trial:

        >>> c = r.register(Preregistration("h", "canary", {}, "c", "e", is_control=True))
        >>> r.deflation_inputs()
        (1, 0.0)
        """
        counted = {t["id"] for t in self.trials()}
        sharpes = [r["sharpe_per_period"] for r in self.results()
                   if r["trial_id"] in counted and r.get("sharpe_per_period") is not None]
        spread = float(statistics.stdev(sharpes)) if len(sharpes) >= 2 else 0.0
        return len(counted), spread

    def summary(self) -> str:
        n, spread = self.deflation_inputs()
        controls = len(self.trials(include_controls=True)) - n
        done = len(self.results())
        return (f"{n} pre-registered trial(s), {controls} control(s), "
                f"{done} result(s) recorded; Sharpe spread across trials "
                f"{spread:.4f} per period")
