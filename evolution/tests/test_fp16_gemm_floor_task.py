"""Check the fp16 GEMM correctness gate on CPU."""

import pytest
import torch

from evolution.benchmark import adapter

adapter._bench_repo_on_path()
from flashinfer_bench_evolve.benchmark_common import BenchConfig, compare_outputs  # noqa: E402

bench = adapter.import_benchmark("fp16_gemm_floor")


def compare(candidate, reference, required=bench.REQUIRED_MATCHED_RATIO):
    return compare_outputs(candidate, reference, bench.ATOL, bench.RTOL, required)


@pytest.mark.parametrize("bad_count", [18, 28, 100])
def test_sparse_large_errors_are_rejected(bad_count):
    # 0.09%, 0.14% and 0.5% corruptions all passed the old 99% gate.
    reference = torch.ones(20_000, dtype=torch.float16)
    candidate = reference.clone()
    candidate[:bad_count] += 30

    old = compare(candidate, reference, required=0.99)
    current = compare(candidate, reference)
    assert old[0]
    assert not current[0]
    assert current[4] == pytest.approx(1 - bad_count / reference.numel())
    assert "elementwise matched ratio" in current[5]


@pytest.mark.parametrize("bad_count, passed", [(1, True), (2, True), (3, False)])
def test_matched_ratio_boundary(bad_count, passed):
    reference = torch.ones(20_000, dtype=torch.float16)
    candidate = reference.clone()
    candidate[:bad_count] += 30
    result = compare(candidate, reference)
    assert result[0] == passed
    assert result[4] == pytest.approx(1 - bad_count / reference.numel())


@pytest.mark.parametrize("perturb", [False, True])
def test_exact_and_in_tolerance_outputs_pass(perturb):
    reference = torch.tensor([0.0, 100.0], dtype=torch.float16)
    candidate = reference.clone()
    if perturb:
        # Exercise absolute and relative tolerances independently.
        candidate += torch.tensor([0.05, 0.5], dtype=torch.float16)
    result = compare(candidate, reference)
    assert result[0]
    assert result[4] == 1.0


@pytest.mark.parametrize("bad_count", [0, 100])
def test_run_suite_uses_task_correctness_gate(monkeypatch, bad_count):
    from flashinfer_bench_evolve.tasks.fp16_gemm_floor import baseline

    monkeypatch.setattr(bench, "choose_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    def candidate(a, b):
        output = baseline.run(a, b)
        output.view(-1)[:bad_count] += 30
        return output

    workload = {
        "suite": "official",
        "id": "cpu-correctness",
        "axes": {"M": 100, "N": 200, "K": 1},
        "timed": False,
    }
    rows = bench.run_suite(
        config=BenchConfig(device="cpu", trials=1, correctness_runs=1),
        candidate_fn=candidate,
        workloads=[workload],
    )
    assert len(rows) == 1
    assert rows[0]["passed"] == (bad_count == 0)
    assert rows[0]["matched"] == pytest.approx(1 - bad_count / 20_000)
    assert rows[0]["baseline_ms"] is None
    assert rows[0]["kernel_ms"] is None
