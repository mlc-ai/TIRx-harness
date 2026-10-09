"""Standalone benchmark for fp16 FlashAttention-4 forward candidates."""

from __future__ import annotations

import hashlib
import math

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    config_from_env,
    load_candidate_override,
    load_task_workloads,
    run_benchmark,
)
TASK_NAME = "fa4_forward"

ATOL = 5e-2
RTOL = 5e-2
REQUIRED_MATCHED_RATIO = 0.999
REQUIRED_RMS_ERROR_RATIOS = (0.2,)


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
            shape_axes=("seq_len", "batch_size", "causal"),
        ):
            wl = row["workload"]
            official.append({"suite": "official", "id": wl["uuid"][:8], "axes": wl["axes"], "raw": wl})
    return official


def _entry_seed(entry):
    raw = f"fa4-forward:{entry['id']}:{entry['axes']}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _randn(shape, dtype, device, scale=1.0):
    return (torch.randn(shape, dtype=torch.float32, device=device) * scale).to(dtype)


def make_inputs(entry, device):
    axes = entry["axes"]
    batch = axes["batch_size"]
    seq_len = axes["seq_len"]
    num_heads = axes["num_heads"]
    head_dim = axes["head_dim"]
    causal = bool(axes.get("causal", True))
    sm_scale = float(axes.get("sm_scale", 1.0 / math.sqrt(head_dim)))
    torch.manual_seed(_entry_seed(entry))
    return [
        _randn((batch, seq_len, num_heads, head_dim), torch.float16, device, 0.5),
        _randn((batch, seq_len, num_heads, head_dim), torch.float16, device, 0.5),
        _randn((batch, seq_len, num_heads, head_dim), torch.float16, device, 0.5),
        causal,
        sm_scale,
    ]


def tirx_prepare(solution_module, q, k, v, causal, sm_scale):
    """Compile and bind a TIRx FA4-forward candidate outside timing."""

    import tvm

    if causal:
        print("warning: current TIRx forward kernel ignores causal masking")
    batch, seq_len, num_heads, head_dim = q.shape
    total_heads = batch * num_heads
    q_flat = q.permute(0, 2, 1, 3).reshape(
        total_heads, seq_len, head_dim
    ).contiguous()
    k_flat = k.permute(0, 2, 1, 3).reshape(
        total_heads, seq_len, head_dim
    ).contiguous()
    vt_flat = v.permute(0, 2, 3, 1).reshape(
        total_heads, head_dim, seq_len
    ).contiguous()
    output = torch.empty(
        (total_heads, seq_len, head_dim), dtype=q.dtype, device=q.device
    )
    lse = torch.empty(
        (total_heads, seq_len), dtype=torch.float32, device=q.device
    )

    kernel = solution_module.tir_kernel(total_heads, seq_len, head_dim)
    target = tvm.target.Target("cuda")
    with target:
        executable = tvm.compile(
            tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx"
        )
    kernel_args = (q_flat, k_flat, vt_flat, output, lse)

    def kernel_fn():
        executable(*kernel_args)

    kernel_fn()
    return kernel_fn, output, batch, num_heads, seq_len, head_dim


def tirx_run(kernel_fn, output, batch, num_heads, seq_len, head_dim):
    kernel_fn()
    return output.reshape(batch, num_heads, seq_len, head_dim).permute(
        0, 2, 1, 3
    )


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
):
    """Run the correctness gate and timing sweep for this task."""

    from .baseline import run as baseline_forward

    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_FA4_FORWARD_KERNEL",
            baseline_forward,
            "fa4_forward_candidate",
        )
    return run_benchmark(
        name="fa4_forward_fp16",
        workloads=make_workloads(cfg) if workloads is None else workloads,
        make_inputs=make_inputs,
        baseline_fn=baseline_forward,
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
