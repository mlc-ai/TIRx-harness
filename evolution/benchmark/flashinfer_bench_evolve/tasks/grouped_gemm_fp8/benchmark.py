"""Packaged Grouped GEMM task for the shared local and KCoral adapters."""

from __future__ import annotations

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    compare_tensors,
    config_from_env,
    load_task_workloads,
    run_benchmark,
)
from . import baseline

TASK_NAME = "grouped_gemm_fp8"
DIFF_THRESHOLD = 1e-6


def default_config() -> BenchConfig:
    return config_from_env(warmup=3, iters=30, trials=3)


def make_workloads(config: BenchConfig | None = None):
    cfg = config or default_config()
    if not cfg.include_official:
        raise ValueError("the Grouped GEMM task requires its official workload")
    rows = load_task_workloads(
        TASK_NAME,
        cfg.max_official,
        shape_mode=cfg.shape_mode,
        shape_axes=("G", "expected_m_per_group", "N", "K"),
    )
    if len(rows) != 4:
        raise ValueError("expected all four Grouped GEMM workloads")
    return [
        {
            "suite": "official",
            "id": row["workload"]["uuid"],
            "axes": row["workload"]["axes"],
            "raw": row["workload"],
        }
        for row in rows
    ]


def make_inputs(entry, device):
    config = dict(entry["axes"], seed=entry["raw"]["seed"])
    config["num_groups"] = config.pop("G")
    return config, device


def tirx_prepare(solution_module, config, device):
    data = baseline.prepare_data(config, device)
    inputs = {key.upper(): data[key] for key in ("a", "b", "sfa", "sfb", "d")}
    inputs.update(
        {key: data[key] for key in ("grouped_layout", "actual_ms", "aligned_ms", "alignment")}
    )
    launch = solution_module.setup(inputs, data["num_groups"], data["M"], data["N"], data["K"])
    return launch, baseline.output_views(data["d"], data["actual_ms"], data["aligned_ms"])


tirx_run = baseline.run_prepared


def compare_outputs(candidate, reference, *_unused):
    from tirx_kernels.ported.deepgemm._sm100_fp8_fp4_gemm_1d1d.data import calc_diff

    # Reuse shared shape/dtype/finite checks and diagnostic statistics.
    stats = compare_tensors(candidate, reference, float("inf"), 0.0)
    if not stats[0]:
        return stats
    diff = max(calc_diff(got, ref) for got, ref in zip(candidate, reference, strict=True))
    passed = diff < DIFF_THRESHOLD
    return (passed, *stats[1:4], float(passed), f"max group diff={diff:.3e}")


def run_suite(config=None, candidate_fn=None, workloads=None, candidate_prepare_fn=None):
    cfg = config or default_config()
    entries = make_workloads(cfg) if workloads is None else workloads
    if not entries:
        raise ValueError("the Grouped GEMM workload list cannot be empty")
    if candidate_fn is None:
        candidate_fn, candidate_prepare_fn = baseline.run_prepared, baseline.prepare
    return run_benchmark(
        name=TASK_NAME,
        workloads=entries,
        make_inputs=make_inputs,
        candidate_fn=candidate_fn,
        candidate_prepare_fn=candidate_prepare_fn,
        baseline_fn=baseline.run_prepared,
        baseline_prepare_fn=baseline.prepare,
        compare_fn=compare_outputs,
        device=choose_device(cfg.device),
        warmup=cfg.warmup,
        iters=cfg.iters,
        trials=cfg.trials,
        atol=0.0,
        rtol=0.0,
        required_matched_ratio=1.0,
        group_axis="G",
        correctness_runs=cfg.correctness_runs,
        check_after_timing=cfg.check_after_timing,
        require_repeatable_outputs=cfg.require_repeatable_outputs,
    )
