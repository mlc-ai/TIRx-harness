"""Exercise the MoE RMS gate through the real suite runner on CPU."""

import sys
from types import ModuleType

import pytest
import torch

from evolution.benchmark import adapter

bench = adapter.import_benchmark("moe")
from flashinfer_bench_evolve.benchmark_common import compare_outputs  # noqa: E402


@pytest.fixture
def cpu_suite(monkeypatch):
    baseline = ModuleType(f"{bench.__package__}.baseline")
    baseline.run = lambda output: output.clone()
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    monkeypatch.setattr(bench, "choose_device", lambda setting: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    def run(reference, transform):
        monkeypatch.setattr(bench, "make_inputs", lambda entry, device: [reference.clone()])
        legacy = compare_outputs(
            transform(reference.clone()),
            reference,
            bench.ATOL,
            bench.RTOL,
            bench.REQUIRED_MATCHED_RATIO,
        )
        assert legacy[0] and legacy[4] == 1.0
        rows = bench.run_suite(
            config=bench.BenchConfig(
                device="cpu",
                trials=1,
                correctness_runs=3,
                check_after_timing=False,
                require_repeatable_outputs=False,
            ),
            candidate_fn=transform,
            workloads=[
                {
                    "suite": "official",
                    "id": "low-amplitude",
                    "axes": {"seq_len": reference.shape[0]},
                    "timed": False,
                }
            ],
        )
        assert len(rows) == 1
        return rows[0]

    return run


@pytest.mark.parametrize(
    ("scale", "passed"),
    [(0.0, False), (0.1, False), (0.5, False), (1.0, True), (1.01, True)],
    ids=["zero", "tenth", "half", "exact", "small-perturbation"],
)
def test_low_amplitude_output(cpu_suite, scale, passed):
    reference = torch.linspace(-0.25, 0.25, 128, dtype=torch.bfloat16).reshape(4, 32)
    row = cpu_suite(reference, lambda output: output * scale)
    assert row["passed"] is passed
    assert row["matched"] == 1.0
    assert row["max_rms_ratio"] == pytest.approx(abs(1.0 - scale), abs=0.003)
    if passed:
        assert row["correctness_checks"] == 3
    else:
        assert row["verdict"] == "FAIL"
        assert "output 0 normalized RMS ratio" in row["note"]


def test_genuine_zero_reference(cpu_suite):
    # A token with no routed local expert can legitimately have a zero result.
    row = cpu_suite(torch.zeros((1, 32), dtype=torch.bfloat16), torch.clone)
    assert row["passed"]
    assert row["max_rms_ratio"] == 0.0
    assert row["correctness_checks"] == 3


@pytest.mark.parametrize("affected_tokens", [9, 1])
def test_sparse_missing_contribution(cpu_suite, affected_tokens):
    # Even one missing token must fail when the workload is small enough that
    # allowing it would exceed the token mismatch budget.
    reference = torch.full((100, 32), 0.125, dtype=torch.bfloat16)

    def omit_contribution(output):
        result = output.clone()
        result[:affected_tokens] = 0
        return result

    row = cpu_suite(reference, omit_contribution)
    assert not row["passed"]
    assert row["max_rms_ratio"] == pytest.approx((affected_tokens / 100) ** 0.5)
    assert row["matched"] == 1.0
    if affected_tokens == 1:
        assert "per-token normalized RMS matched ratio" in row["note"]


@pytest.mark.parametrize(
    ("tokens", "outliers", "passed"), [(9999, 1, False), (10000, 1, True), (10000, 2, False)]
)
def test_rare_token_outlier_budget(cpu_suite, tokens, outliers, passed):
    reference = torch.full((tokens, 32), 0.125, dtype=torch.bfloat16)

    def token_outliers(output):
        result = output.clone()
        result[:outliers] = 0
        return result

    row = cpu_suite(reference, token_outliers)
    assert row["passed"] is passed
    if not passed:
        assert "per-token normalized RMS matched ratio" in row["note"]
