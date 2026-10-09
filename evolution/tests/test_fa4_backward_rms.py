"""Check that the backward gate rejects missing or rescaled gradients on CPU."""

import sys
from types import ModuleType

import pytest
import torch

from evolution.benchmark import adapter

bench = adapter.import_benchmark("fa4_backward")
from flashinfer_bench_evolve.benchmark_common import BenchConfig, compare_outputs  # noqa: E402


@pytest.fixture
def cpu_suite(monkeypatch):
    values = torch.linspace(-0.004, 0.004, 128).reshape(2, 4, 2, 8)
    reference = tuple((values * scale).to(torch.float16) for scale in (1.0, 2.0, 0.5))
    baseline = ModuleType(f"{bench.__package__}.baseline")
    baseline.run = lambda *gradients: tuple(gradient.clone() for gradient in gradients)
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    monkeypatch.setattr(bench, "choose_device", lambda setting: torch.device("cpu"))
    monkeypatch.setattr(
        bench,
        "make_inputs",
        lambda entry, device: tuple(gradient.clone() for gradient in reference),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    config = BenchConfig(
        device="cpu",
        include_official=False,
        trials=1,
        correctness_runs=3,
        check_after_timing=False,
        require_repeatable_outputs=False,
    )
    workloads = [dict(suite="official", id="rms", axes={"seq_len": 4}, timed=False)]

    def run(scale, corrupted_indices):
        def candidate(*gradients):
            # Every call allocates fresh outputs: the runner poisons previous results.
            return tuple(
                gradient * scale if index in corrupted_indices else gradient.clone()
                for index, gradient in enumerate(gradients)
            )

        legacy = compare_outputs(
            candidate(*reference), reference, bench.ATOL, bench.RTOL, bench.REQUIRED_MATCHED_RATIO
        )
        assert legacy[0] and legacy[4] == 1.0
        rows = bench.run_suite(config=config, candidate_fn=candidate, workloads=workloads)
        assert len(rows) == 1
        return rows[0]

    return run


@pytest.mark.parametrize("scale", [0.0, 0.5], ids=["zero", "half"])
@pytest.mark.parametrize(
    "corrupted_indices", [(0,), (1,), (2,), (0, 1, 2)], ids=["dq", "dk", "dv", "all"]
)
def test_backward_run_suite_rejects_corrupted_gradients(cpu_suite, scale, corrupted_indices):
    row = cpu_suite(scale, corrupted_indices)
    assert not row["passed"]
    assert row["verdict"] == "FAIL"
    assert row["matched"] == 1.0
    assert row["max_rms_ratio"] == pytest.approx(1.0 - scale, rel=1e-3)
    for index in range(3):
        message = f"output {index} normalized RMS ratio"
        assert (message in row["note"]) == (index in corrupted_indices)


@pytest.mark.parametrize("scale", [1.0, 1.001], ids=["exact", "small-perturbation"])
def test_backward_run_suite_accepts_accurate_gradients(cpu_suite, scale):
    row = cpu_suite(scale, (0, 1, 2))
    assert row["passed"]
    assert row["correctness_checks"] == 3
    assert row["matched"] == 1.0
    assert row["max_rms_ratio"] < 0.2
    if scale == 1.0:
        assert row["max_rms_ratio"] == 0.0
    else:
        assert row["max_rms_ratio"] > 0.0
