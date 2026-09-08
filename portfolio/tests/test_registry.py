"""The pre-registration log: append-only, mandatory fields, honest counts.

The registry's only job is to make the trial count auditable, so what is
tested is the ways that count could be quietly wrong.
"""

from __future__ import annotations

import json

import pytest

from portfolio.eval.registry import Preregistration, Registry, TrialResult


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "registry.jsonl")


def trial(**kw):
    base = dict(hypothesis="equal risk contribution beats buy-and-hold net of costs",
                policy="erc", parameters={"lookback": 252},
                success_criterion="deflated Sharpe above 0.95",
                expected_outcome="a small improvement, erased by turnover")
    base.update(kw)
    return Preregistration(**base)


class TestMandatoryFields:
    @pytest.mark.parametrize("field", ["hypothesis", "success_criterion",
                                       "expected_outcome", "policy"])
    def test_a_blank_field_is_refused(self, field):
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            trial(**{field: "   "})

    def test_the_expected_outcome_must_be_written_before_the_run(self):
        """The field that makes it a pre-registration rather than a record.

        "I expected this to fail" is a scientific statement written first and
        a rationalisation written afterwards, and the file cannot tell them
        apart unless the field is mandatory.
        """
        with pytest.raises(ValueError, match="expected outcome"):
            trial(expected_outcome="")


class TestAppendOnly:
    def test_a_result_is_a_new_line_not_an_edit(self, registry):
        t = registry.register(trial())
        registry.record(TrialResult(t.id, 0.05, 252, 252, "ran"))
        rows = [json.loads(line) for line in
                registry.path.read_text(encoding="utf-8").splitlines()]
        assert [r["kind"] for r in rows] == ["trial", "result"]
        assert rows[0]["id"] == t.id and rows[1]["trial_id"] == t.id

    def test_a_result_for_an_unregistered_trial_is_refused(self, registry):
        registry.register(trial())
        with pytest.raises(ValueError, match="never registered"):
            registry.record(TrialResult("trial_doesnotexist", 0.05, 252, 252, "ran"))

    def test_the_file_is_readable_without_this_module(self, registry):
        registry.register(trial())
        line = registry.path.read_text(encoding="utf-8").splitlines()[0]
        assert json.loads(line)["hypothesis"].startswith("equal risk")

    def test_an_absent_file_reads_as_empty_rather_than_failing(self, tmp_path):
        assert Registry(tmp_path / "nothing.jsonl").entries() == []
        assert Registry(tmp_path / "nothing.jsonl").deflation_inputs() == (0, 0.0)


class TestTheTrialCount:
    def test_controls_are_recorded_but_not_counted(self, registry):
        registry.register(trial())
        registry.register(trial(policy="canary", is_control=True))
        registry.register(trial(policy="coin-flip", is_control=True))
        assert registry.deflation_inputs()[0] == 1
        assert len(registry.trials(include_controls=True)) == 3
        assert len(registry.entries()) == 3

    def test_the_spread_needs_two_results(self, registry):
        a = registry.register(trial(policy="a"))
        registry.record(TrialResult(a.id, 0.05, 252, 252, "ran"))
        assert registry.deflation_inputs() == (1, 0.0)
        b = registry.register(trial(policy="b"))
        registry.record(TrialResult(b.id, 0.09, 252, 252, "ran"))
        n, spread = registry.deflation_inputs()
        assert n == 2 and spread == pytest.approx(0.02828427, rel=1e-5)

    def test_a_control_result_does_not_widen_the_spread(self, registry):
        a = registry.register(trial(policy="a"))
        b = registry.register(trial(policy="b"))
        c = registry.register(trial(policy="canary", is_control=True))
        for t, sr in ((a, 0.05), (b, 0.09), (c, 2.9)):
            registry.record(TrialResult(t.id, sr, 252, 252, "ran"))
        n, spread = registry.deflation_inputs()
        assert n == 2
        assert spread == pytest.approx(0.02828427, rel=1e-5), (
            "the canary's Sharpe of 2.9 leaked into the cross-trial spread, "
            "which would inflate the deflation threshold for every real "
            "strategy that follows it")

    def test_a_missing_sharpe_is_skipped_not_counted_as_zero(self, registry):
        a = registry.register(trial(policy="a"))
        b = registry.register(trial(policy="b"))
        registry.record(TrialResult(a.id, 0.05, 252, 252, "ran"))
        registry.record(TrialResult(b.id, None, 10, 10, "too short to rate"))
        assert registry.deflation_inputs() == (2, 0.0)

    def test_the_summary_says_what_it_holds(self, registry):
        registry.register(trial())
        registry.register(trial(policy="canary", is_control=True))
        text = registry.summary()
        assert "1 pre-registered trial" in text and "1 control" in text
