"""Check the FA4 forward RMS gate and its suite wiring on CPU."""

import sys
from types import ModuleType

import pytest
import torch

from evolution.benchmark import adapter

bench = adapter.import_benchmark("fa4_forward")
from flashinfer_bench_evolve.benchmark_common import compare_outputs  # noqa: E402


@pytest.mark.parametrize(
    ("scale", "passed"),
    [(0.0, False), (0.5, False), (1.0, True), (1.001, True)],
    ids=["zero", "half", "exact", "small-perturbation"],
)
def test_low_amplitude_output_quality(monkeypatch, scale, passed):
    # Every element is below ATOL, so even zero and half-sized outputs
    # satisfy the old elementwise rule.
    reference = torch.tensor(
        [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04], dtype=torch.float16
    ).reshape(1, 4, 1, 2)
    candidate = reference * scale
    tolerances = (bench.ATOL, bench.RTOL, bench.REQUIRED_MATCHED_RATIO)
    legacy = compare_outputs(candidate, reference, *tolerances)
    assert legacy[0] and legacy[4] == 1.0

    gated = compare_outputs(
        candidate, reference, *tolerances, required_rms_error_ratios=bench.REQUIRED_RMS_ERROR_RATIOS
    )
    assert gated[0] is passed
    assert gated[4] == 1.0
    if passed:
        assert gated[3] < 0.2
    else:
        assert gated[3] == pytest.approx(1.0 - scale, abs=1e-5)
        assert "normalized RMS ratio" in gated[5]

    # Exercise the real run_suite -> run_benchmark -> comparator path
    # without importing flash-attn or entering GPU timing.
    baseline = ModuleType(f"{bench.__package__}.baseline")
    baseline.run = lambda output: output
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    monkeypatch.setattr(bench, "choose_device", lambda setting: torch.device("cpu"))
    monkeypatch.setattr(bench, "make_inputs", lambda entry, device: [reference.clone()])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    rows = bench.run_suite(
        config=bench.BenchConfig(
            device="cpu",
            trials=1,
            correctness_runs=1,
            check_after_timing=False,
            require_repeatable_outputs=False,
        ),
        candidate_fn=lambda output: output * scale,
        workloads=[
            {"suite": "official", "id": "low-amplitude", "axes": {"seq_len": 4}, "timed": False}
        ],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["passed"] is passed
    assert row["matched"] == 1.0
    assert row["max_rms_ratio"] == pytest.approx(gated[3])
    if not passed:
        assert row["verdict"] == "FAIL"
        assert "normalized RMS ratio" in row["note"]
