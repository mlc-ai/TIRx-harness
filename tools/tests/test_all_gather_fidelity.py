"""Keep the CUTLASS-schedule comparison separate from derived implementations."""

import pytest

from benchmarks.peer.fidelity import summarize, validate_summary


def measurement(config="baseline-best", **updates):
    return {
        "shape": "example", "origin": "cutlass", "comparison_kind": "port_fidelity",
        "config": config, "errors": {"cutlass": None, "tirx": None},
        "bitwise_equal": True, "origin_us": 100.0, "tirx_us": 100.0, "ratio": 1.0,
        **updates,
    }


def test_fidelity_uses_the_same_baseline_selected_configuration():
    rows = [measurement(tirx_us=103.0, ratio=1.03),
            measurement("different", origin_us=120.0, tirx_us=90.0, ratio=0.75)]
    result = summarize(rows)

    assert result[0]["config"] == "baseline-best"
    assert result[0]["ratio"] == 1.03
    with pytest.raises(RuntimeError, match="99% parity"):
        validate_summary(result, ["example"])


@pytest.mark.parametrize("kind", (None, "cross_implementation"))
def test_derived_or_unlabelled_results_are_not_fidelity_evidence(kind):
    assert summarize([measurement(comparison_kind=kind)]) == []


def test_fidelity_rejects_missing_or_incorrect_shapes():
    with pytest.raises(RuntimeError, match="missing other"):
        validate_summary(summarize([measurement()]), ["example", "other"])
    with pytest.raises(RuntimeError, match="numerical comparison failed"):
        validate_summary(summarize([measurement(bitwise_equal=False)]), ["example"])


def test_equal_outputs_do_not_hide_a_reference_failure():
    row = measurement(errors={"cutlass": "reference mismatch", "tirx": "reference mismatch"})
    with pytest.raises(RuntimeError, match="numerical comparison failed"):
        validate_summary(summarize([row]), ["example"])


@pytest.mark.parametrize("ratio", (0, float("nan"), float("inf"), 1 / 0.989))
def test_fidelity_rejects_invalid_or_slow_timings(ratio):
    with pytest.raises(RuntimeError, match="99% parity"):
        validate_summary(summarize([measurement(ratio=ratio, tirx_us=100 * ratio)]), ["example"])


def test_fidelity_accepts_the_unchanged_99_percent_threshold():
    validate_summary(summarize([measurement(ratio=1 / 0.99, tirx_us=100 / 0.99)]), ["example"])
