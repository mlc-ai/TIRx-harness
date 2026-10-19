"""Standalone benchmark for Wan hybrid (BF16-QK, NVFP4-P/V) attention candidates."""

from __future__ import annotations

from functools import partial

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    compare_tensors,
    config_from_env,
    load_candidate_override,
    load_task_reference,
    load_task_workloads,
    run_benchmark,
)

TASK_NAME = "nvfp4_attention"

# PR #4543: BF16 SDPA reference, allclose AND cosine >= .995 AND MAE <= .025.
ATOL = 1.0
RTOL = 0.1
REQUIRED_MATCHED_RATIO = None
REQUIRED_RMS_ERROR_RATIOS = None
MIN_COSINE = 0.995
MAX_MAE = 0.025
compare = partial(compare_tensors, cast_to_float32=False, min_cosine=MIN_COSINE, max_mae=MAX_MAE)


# The PR's tests and benchmarks seed their generator with 4254.
RANDOM_SEED = 4254


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=3, iters=50, trials=3)


def make_workloads(config: BenchConfig | None = None):
    cfg = config or default_config()
    official = []
    if cfg.include_official:
        for row in load_task_workloads(
            TASK_NAME,
            cfg.max_official,
            shape_mode=cfg.shape_mode,
            shape_axes=("seq_len", "num_heads"),
        ):
            workload = row["workload"]
            official.append(
                {
                    "suite": "official",
                    "id": workload["uuid"],
                    "axes": workload["axes"],
                    "raw": workload,
                }
            )
    return official


def build_inputs(axes, seed, device, *, sm_scale):
    """Unit-normal bf16 q/k/v drawn in the PR's order (q, k, v)."""

    shape = (
        int(axes["batch_size"]),
        int(axes["seq_len"]),
        int(axes["num_heads"]),
        int(axes["head_dim"]),
    )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    tensors = [
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        for _ in range(3)
    ]
    return [*tensors, float(sm_scale)]


def make_inputs(entry, device):
    return build_inputs(
        entry["axes"],
        RANDOM_SEED,
        device,
        sm_scale=float(entry["raw"]["inputs"]["sm_scale"]["value"]),
    )


def tirx_prepare(solution_module, q, k, v, sm_scale):
    """Bind fresh gate inputs to a TIRx attention candidate outside timing."""

    data = {
        "q": q,
        "k": k,
        "v": v,
        "sm_scale": float(sm_scale),
        "output": torch.empty_like(q),
    }
    kernel_fn = solution_module.setup(data, q.shape[0], q.shape[1])
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["output"]


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
):
    """Run the correctness gate and timing sweep for this task."""

    from .baseline import run as baseline_attention

    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_NVFP4_ATTENTION_KERNEL",
            baseline_attention,
            "nvfp4_attention_candidate",
        )
    return run_benchmark(
        name="wan_hybrid_attention_b1_s4800_h40_d128_bf16",
        workloads=make_workloads(cfg) if workloads is None else workloads,
        make_inputs=make_inputs,
        baseline_fn=baseline_attention,
        candidate_fn=candidate,
        candidate_prepare_fn=candidate_prepare_fn,
        device=device,
        warmup=cfg.warmup,
        iters=cfg.iters,
        trials=cfg.trials,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        required_rms_error_ratios=REQUIRED_RMS_ERROR_RATIOS,
        reference_fn=load_task_reference(TASK_NAME),
        compare_fn=compare,
        group_axis="seq_len",
        correctness_runs=cfg.correctness_runs,
        check_after_timing=cfg.check_after_timing,
        require_repeatable_outputs=cfg.require_repeatable_outputs,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
