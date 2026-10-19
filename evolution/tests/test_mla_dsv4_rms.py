"""Check both MLA dtype gates, including FP8's formerly accepted half output."""

import sys
from types import ModuleType

import pytest
import torch

from evolution.benchmark import adapter

bench = adapter.import_benchmark("mla_dsv4")


@pytest.mark.parametrize("kv_dtype,amplitude", [("bfloat16", 0.0005), ("float8_e4m3fn", 0.08)])
@pytest.mark.parametrize("scale,passed", [(0.0, False), (0.5, False), (1.0, True), (0.875, True)])
def test_dtype_gate_and_suite(monkeypatch, kv_dtype, amplitude, scale, passed):
    # MLA returns BF16 for both BF16 and FP8 input pools.
    reference = (torch.full((2, 8, 512), amplitude, dtype=torch.bfloat16),)

    def candidate(output):
        return (output * scale,)

    atol, rtol, limits = bench.gate_for(kv_dtype)
    actual = candidate(*reference)
    legacy = bench.compare(actual, reference, atol, rtol, bench.REQUIRED_MATCHED_RATIO)
    assert legacy[0] and legacy[4] == 1.0
    gated = bench.compare(actual, reference, atol, rtol, bench.REQUIRED_MATCHED_RATIO, limits)
    assert gated[0] is passed
    if not passed:
        assert gated[3] == pytest.approx(1.0 - scale, abs=3e-5)
        assert "output 0: normalized RMS ratio" in gated[5]

    baseline = ModuleType(f"{bench.__package__}.baseline")
    baseline.run = baseline.run_prepared = lambda output: (output,)
    baseline.prepare = lambda *outputs: outputs
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    monkeypatch.setattr(bench, "choose_device", lambda setting: torch.device("cpu"))
    monkeypatch.setattr(bench, "make_inputs", lambda entry, device: reference)
    monkeypatch.setattr(bench, "load_task_reference", lambda name: baseline.run)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    rows = bench.run_suite(
        config=bench.BenchConfig(
            device="cpu",
            correctness_runs=1,
            check_after_timing=False,
            require_repeatable_outputs=False,
        ),
        candidate_fn=candidate,
        workloads=[
            {
                "suite": "regression",
                "id": "small-signal",
                "axes": {"num_heads": 8, "kv_dtype": kv_dtype},
                "timed": False,
            }
        ],
    )
    assert len(rows) == 1
    assert rows[0]["passed"] is passed
    assert rows[0]["matched"] == 1.0
    assert rows[0]["max_rms_ratio"] == pytest.approx(gated[3])


@pytest.mark.parametrize("kv_dtype", ["bfloat16", "float8_e4m3fn"])
def test_zero_reference_and_elementwise_outlier(kv_dtype):
    atol, rtol, limits = bench.gate_for(kv_dtype)

    def compare(actual, reference):
        return bench.compare(actual, reference, atol, rtol, bench.REQUIRED_MATCHED_RATIO, limits)

    zeros = (torch.zeros(4096, dtype=torch.bfloat16),)
    assert compare(zeros, zeros)[0]
    reference = (torch.ones_like(zeros[0]),)
    actual = (reference[0].clone(),)
    actual[0][0] = 2.0
    result = compare(actual, reference)
    assert not result[0]
    assert result[3] < 0.2
