"""Check every KDA training output independently, including small gradients."""

import sys
from types import ModuleType

import pytest
import torch

from evolution.benchmark import adapter

bench = adapter.import_benchmark("kda_backward")


@pytest.mark.parametrize("output_index", range(10), ids=bench.OUTPUT_NAMES)
@pytest.mark.parametrize("scale,passed", [(0.0, False), (0.5, False), (1.0, True), (0.875, True)])
def test_each_output_gate_and_suite(monkeypatch, output_index, scale, passed):
    dtypes = [torch.bfloat16, torch.float32, *([torch.bfloat16] * 5), *([torch.float32] * 3)]
    # Unequal sizes ensure that a small bad gradient is not diluted by the
    # larger, accurate forward result in a single aggregate RMS calculation.
    reference = tuple(
        torch.full((4096 if i == 0 else 16,), 0.004, dtype=dtype) for i, dtype in enumerate(dtypes)
    )

    def candidate(*outputs):
        return tuple(
            value * scale if i == output_index else value.clone() for i, value in enumerate(outputs)
        )

    actual = candidate(*reference)
    tolerances = (bench.ATOL, bench.RTOL, bench.REQUIRED_MATCHED_RATIO)
    legacy = bench.compare(actual, reference, *tolerances)
    assert legacy[0] and legacy[4] == 1.0
    gated = bench.compare(actual, reference, *tolerances, bench.REQUIRED_RMS_ERROR_RATIOS)
    assert gated[0] is passed
    if not passed:
        assert gated[3] == pytest.approx(1.0 - scale, abs=1e-5)
        assert f"output {output_index}: normalized RMS ratio" in gated[5]

    baseline = ModuleType(f"{bench.__package__}.baseline")
    baseline.run = baseline.run_prepared = lambda *outputs: outputs
    baseline.prepare = lambda *outputs: outputs
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    monkeypatch.setattr(bench, "choose_device", lambda setting: torch.device("cpu"))
    monkeypatch.setattr(bench, "make_inputs", lambda entry, device: reference)
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
                "axes": {"total_tokens": 16},
                "timed": False,
            }
        ],
    )
    assert len(rows) == 1
    assert rows[0]["passed"] is passed
    assert rows[0]["matched"] == 1.0
    assert rows[0]["max_rms_ratio"] == pytest.approx(gated[3])


def test_zero_reference_and_elementwise_outlier():
    def compare(actual, reference):
        return bench.compare(
            actual,
            reference,
            bench.ATOL,
            bench.RTOL,
            bench.REQUIRED_MATCHED_RATIO,
            bench.REQUIRED_RMS_ERROR_RATIOS,
        )

    zeros = tuple(torch.zeros(4096, dtype=torch.float32) for _ in bench.OUTPUT_NAMES)
    assert compare(zeros, zeros)[0]
    reference = tuple(torch.ones_like(value) for value in zeros)
    actual = tuple(value.clone() for value in reference)
    actual[-1][0] = 1.125
    result = compare(actual, reference)
    assert not result[0]
    assert result[3] < 0.2
