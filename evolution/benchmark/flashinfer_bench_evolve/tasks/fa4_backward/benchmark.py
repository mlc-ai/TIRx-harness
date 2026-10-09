"""Standalone backward-only benchmark for fp16 FlashAttention-4 candidates."""

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
TASK_NAME = "fa4_backward"

ATOL = 1e-1
RTOL = 1e-1
REQUIRED_MATCHED_RATIO = 0.995
REQUIRED_RMS_ERROR_RATIOS = (0.2, 0.2, 0.2)


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=3, iters=30, trials=2)


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
            official.append(
                {
                    "suite": "official",
                    "id": wl["uuid"][:8],
                    "axes": wl["axes"],
                    "raw": wl,
                }
            )
    return official


def _entry_seed(entry):
    raw = f"fa4-backward:{entry['id']}:{entry['axes']}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _randn(shape, dtype, device, scale=1.0):
    return (torch.randn(shape, dtype=torch.float32, device=device) * scale).to(dtype)


def make_inputs(entry, device):
    from flash_attn.cute.interface import _flash_attn_fwd

    axes = entry["axes"]
    batch = axes["batch_size"]
    seq_len = axes["seq_len"]
    num_heads = axes["num_heads"]
    head_dim = axes["head_dim"]
    causal = bool(axes.get("causal", True))
    sm_scale = float(axes.get("sm_scale", 1.0 / math.sqrt(head_dim)))
    torch.manual_seed(_entry_seed(entry))
    shape = (batch, seq_len, num_heads, head_dim)
    q = _randn(shape, torch.float16, device, 0.5)
    k = _randn(shape, torch.float16, device, 0.5)
    v = _randn(shape, torch.float16, device, 0.5)
    with torch.no_grad():
        # Starred unpack tolerates both cute API generations: pre-940cd96
        # returns (out, lse); newer builds return (out, lse, p, row_max).
        out, lse, *_ = _flash_attn_fwd(
            q=q,
            k=k,
            v=v,
            softmax_scale=sm_scale,
            causal=causal,
            return_lse=True,
        )
    return [
        q,
        k,
        v,
        out,
        _randn(shape, torch.float16, device, 0.25),
        lse,
        causal,
        sm_scale,
    ]


def tirx_prepare(
    solution_module,
    q,
    k,
    v,
    output,
    grad_out,
    lse,
    causal,
    sm_scale,
):
    """Bind fresh gate inputs to a TIRx FA4-backward candidate outside timing."""

    if causal:
        print("warning: current TIRx backward kernel ignores causal masking")
    batch, seq_len, num_heads, head_dim = q.shape
    data = {
        "Q": q,
        "K": k,
        "V": v,
        "O": output,
        "dO": grad_out,
        "LSE": lse,
        "dQ": torch.empty_like(q),
        "dK": torch.empty_like(k),
        "dV": torch.empty_like(v),
        "causal": causal,
        "softmax_scale": float(sm_scale),
    }
    kernel_fn = solution_module.setup(data, batch, num_heads, seq_len, head_dim)
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["dQ"], data["dK"], data["dV"]


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
):
    """Run the correctness gate and timing sweep for this task."""

    from .baseline import run as baseline_backward

    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_FA4_BACKWARD_KERNEL",
            baseline_backward,
            "fa4_backward_candidate",
        )
    return run_benchmark(
        name="fa4_backward_fp16",
        workloads=make_workloads(cfg) if workloads is None else workloads,
        make_inputs=make_inputs,
        baseline_fn=baseline_backward,
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
