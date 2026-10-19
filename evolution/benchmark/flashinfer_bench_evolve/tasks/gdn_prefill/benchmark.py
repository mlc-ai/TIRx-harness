"""Standalone benchmark for GDN prefill candidates."""

from __future__ import annotations

import hashlib
import math

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    config_from_env,
    include_synthetic_workloads,
    load_candidate_override,
    load_safetensor,
    load_task_workloads,
    run_benchmark,
)
TASK_NAME = "gdn_prefill"

ATOL = 1e-2
RTOL = 1e-2
REQUIRED_MATCHED_RATIO = None
# Bound each output independently so small signals cannot be replaced by zeros.
REQUIRED_RMS_ERROR_RATIOS = (0.2, 0.2)

LARGE_WORKLOADS = [
    {"total_seq_len": 16384, "num_seqs": 64, "len_cu_seqlens": 65},
    {"total_seq_len": 32768, "num_seqs": 128, "len_cu_seqlens": 129},
]


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
            shape_axes=("total_seq_len", "num_seqs"),
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
    large = []
    if include_synthetic_workloads(cfg.shape_mode):
        large = [
            {
                "suite": "large",
                "id": f"T{w['total_seq_len']}_N{w['num_seqs']}",
                "axes": dict(w),
                "raw": None,
            }
            for w in LARGE_WORKLOADS
        ]
    return official + large


def _balanced_cu_seqlens(total_seq_len: int, num_seqs: int, device):
    base = total_seq_len // num_seqs
    rem = total_seq_len % num_seqs
    lens = torch.full((num_seqs,), base, dtype=torch.int64, device=device)
    if rem:
        lens[:rem] += 1
    return torch.cat([torch.zeros(1, dtype=torch.int64, device=device), lens.cumsum(0)])


def _entry_seed(entry):
    raw = f"{entry['suite']}:{entry['id']}:{entry['axes']}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def make_inputs(entry, device):
    axes = entry["axes"]
    total_seq_len = axes["total_seq_len"]
    num_seqs = axes["num_seqs"]
    scale = 1.0 / math.sqrt(128)
    cu_seqlens = _balanced_cu_seqlens(total_seq_len, num_seqs, device)
    if entry["raw"] is not None:
        inputs = entry["raw"]["inputs"]
        cu_seqlens = load_safetensor(inputs["cu_seqlens"], device)
        scale = float(inputs["scale"]["value"])

    torch.manual_seed(_entry_seed(entry))
    return [
        (torch.randn((total_seq_len, 4, 128), dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16),
        (torch.randn((total_seq_len, 4, 128), dtype=torch.float32, device=device) * 0.1).to(torch.bfloat16),
        (torch.randn((total_seq_len, 8, 128), dtype=torch.float32, device=device) * 0.15).to(torch.bfloat16),
        torch.randn((num_seqs, 8, 128, 128), dtype=torch.float32, device=device) * 0.01,
        torch.rand((8,), dtype=torch.float32, device=device) * 2.0 - 1.0,
        (torch.randn((total_seq_len, 8), dtype=torch.float32, device=device) * 1.5).to(torch.bfloat16),
        torch.rand((8,), dtype=torch.float32, device=device) * 6.0 - 4.0,
        torch.randn((total_seq_len, 8), dtype=torch.float32, device=device).to(torch.bfloat16),
        cu_seqlens,
        scale,
    ]


def tirx_prepare(
    solution_module,
    q,
    k,
    v,
    state,
    a_log,
    a,
    dt_bias,
    b_gate,
    cu_seqlens,
    scale,
):
    """Bind fresh gate inputs to a TIRx GDN-prefill candidate outside timing."""

    data = {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "A_log": a_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b_gate,
        "cu_seqlens": cu_seqlens,
        "scale": scale,
        "output": torch.empty_like(v),
        "new_state": torch.empty_like(state),
    }
    kernel_fn = solution_module.setup(data, state.shape[0], q.shape[0])
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["output"], data["new_state"]


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
):
    """Run the correctness gate and timing sweep for this task."""

    from . import baseline as baseline_module

    baseline_prefill = baseline_module.run
    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_GDN_PREFILL_KERNEL",
            baseline_prefill,
            "gdn_prefill_candidate",
        )
    # Real candidates hoist their setup outside timing, so the baseline is
    # timed kernel-only through its prepare step. The baseline-vs-baseline
    # sanity path keeps the raw run() on both arms so it still reads ~1.0x.
    baseline_kwargs = {"baseline_fn": baseline_prefill}
    if candidate is not baseline_prefill:
        baseline_kwargs = {
            "baseline_fn": baseline_module.run_prepared,
            "baseline_prepare_fn": baseline_module.prepare,
        }
    return run_benchmark(
        name="gdn_prefill_qk4_v8_d128_k_last",
        workloads=make_workloads(cfg) if workloads is None else workloads,
        make_inputs=make_inputs,
        candidate_fn=candidate,
        **baseline_kwargs,
        candidate_prepare_fn=candidate_prepare_fn,
        device=device,
        warmup=cfg.warmup,
        iters=cfg.iters,
        trials=cfg.trials,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        required_rms_error_ratios=REQUIRED_RMS_ERROR_RATIOS,
        group_axis="total_seq_len",
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
