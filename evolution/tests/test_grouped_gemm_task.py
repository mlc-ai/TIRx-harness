"""Check task packaging and the valid-row correctness boundary on CPU."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from evolution.benchmark import adapter
from evolution.preparation.declare import load_task

bench = adapter.import_benchmark("grouped_gemm_fp8")
KEY = "grouped_gemm/fp8"


def test_task_plan_and_bundle():
    task = load_task(adapter.REPO_ROOT / "evolution/tasks/grouped_gemm_fp8.yaml")
    assert adapter.workload_key(task.workload_dir) == KEY
    name, warmup, repeat, mode = adapter.PACKAGED[KEY]
    overrides, rows, candidate = adapter.plan(
        name,
        adapter.CANDIDATES_ROOT / KEY,
        "baseline",
        warmup=warmup,
        repeat=repeat,
        shape_mode=mode,
    )
    assert rows == bench.make_workloads(replace(bench.default_config(), **overrides))
    assert len(rows) == 4 and candidate is None
    expected = [
        (4, 8192, 6144, 7168, 1),
        (4, 8192, 4096, 4096, 3),
        (8, 4096, 7168, 3072, 6),
        (8, 4096, 4096, 2048, 8),
    ]
    for row, (g, m, n, k, seed) in zip(rows, expected, strict=True):
        config, device = bench.make_inputs(row, "cuda:0")
        assert config == dict(num_groups=g, expected_m_per_group=m, N=n, K=k, seed=seed)
        assert device == "cuda:0"
    sources = adapter.harness_sources(name)
    assert f"flashinfer_bench_evolve/tasks/{name}/baseline.py" in sources


def test_candidate_adapter_and_valid_rows(monkeypatch):
    from flashinfer_bench_evolve.benchmark_common import poison_outputs

    output = torch.zeros((8, 2), dtype=torch.bfloat16)
    data = dict(
        a=object(),
        b=object(),
        sfa=object(),
        sfb=object(),
        d=output,
        grouped_layout=object(),
        actual_ms=(2, 1),
        aligned_ms=(4, 4),
        alignment=4,
        num_groups=2,
        M=8,
        N=2,
        K=4,
    )
    monkeypatch.setattr(bench.baseline, "prepare_data", lambda *args: data)

    def setup(inputs, *shape):
        assert shape == (2, 8, 2, 4)
        assert inputs["A"] is data["a"] and inputs["SFB"] is data["sfb"]
        assert inputs["grouped_layout"] is data["grouped_layout"]
        return lambda: inputs["D"].fill_(1)

    launch, views = bench.tirx_prepare(SimpleNamespace(setup=setup), {}, "cpu")
    poison_outputs(views)
    assert torch.isnan(output[:2]).all() and torch.isnan(output[4:5]).all()
    assert not output[2:4].any() and not output[5:].any()
    bench.tirx_run(launch, views)
    reference = tuple(v.clone() for v in views)
    assert bench.compare_outputs(views, reference)[0]
    views[-1].zero_()
    assert not bench.compare_outputs(views, reference)[0]
    views[-1].fill_(float("nan"))
    assert not bench.compare_outputs(views, reference)[0]
