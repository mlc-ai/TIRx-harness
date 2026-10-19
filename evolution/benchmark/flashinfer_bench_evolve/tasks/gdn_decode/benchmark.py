"""Standalone benchmark for GDN decode candidates."""

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
    load_task_workloads,
    rand_tensor,
    run_benchmark,
)

TASK_NAME = "gdn_decode"

ATOL = 1e-2
RTOL = 1e-2
REQUIRED_MATCHED_RATIO = None

LARGE_WORKLOADS = [
    {"batch_size": 128},
    {"batch_size": 256},
    {"batch_size": 512},
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
            shape_axes=("batch_size",),
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
                "id": f"B{w['batch_size']}",
                "axes": dict(w),
                "raw": None,
            }
            for w in LARGE_WORKLOADS
        ]
    return official + large


def _entry_seed(entry):
    raw = f"{entry['suite']}:{entry['id']}:{entry['axes']}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def make_inputs(entry, device):
    batch_size = entry["axes"]["batch_size"]
    torch.manual_seed(_entry_seed(entry))
    scale = 1.0 / math.sqrt(128)
    if entry["raw"] is not None:
        scale = float(entry["raw"]["inputs"]["scale"]["value"])
    return [
        rand_tensor((batch_size, 1, 4, 128), torch.bfloat16, device),
        rand_tensor((batch_size, 1, 4, 128), torch.bfloat16, device),
        rand_tensor((batch_size, 1, 8, 128), torch.bfloat16, device),
        rand_tensor((batch_size, 8, 128, 128), torch.float32, device),
        rand_tensor((8,), torch.float32, device),
        rand_tensor((batch_size, 1, 8), torch.bfloat16, device),
        rand_tensor((8,), torch.float32, device),
        rand_tensor((batch_size, 1, 8), torch.bfloat16, device),
        scale,
    ]


def tirx_prepare(solution_module, q, k, v, state, a_log, a, dt_bias, b_gate, scale):
    """Bind fresh gate inputs to a TIRx GDN-decode candidate outside timing."""

    data = {
        "q": q,
        "k": k,
        "v": v,
        "state": state,
        "A_log": a_log,
        "a": a,
        "dt_bias": dt_bias,
        "b": b_gate,
        "scale": scale,
        "output": torch.empty_like(v),
        "new_state": torch.empty_like(state),
    }
    kernel_fn = solution_module.setup(data, q.shape[0])
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

    baseline_decode = baseline_module.run
    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_GDN_DECODE_KERNEL",
            baseline_decode,
            "gdn_decode_candidate",
        )
    # Real candidates hoist their setup outside timing, so the baseline is
    # timed kernel-only through its prepare step. The baseline-vs-baseline
    # sanity path keeps the raw run() on both arms so it still reads ~1.0x.
    baseline_kwargs = {"baseline_fn": baseline_decode}
    if candidate is not baseline_decode:
        baseline_kwargs = {
            "baseline_fn": baseline_module.run_prepared,
            "baseline_prepare_fn": baseline_module.prepare,
        }
    return run_benchmark(
        name="gdn_decode_qk4_v8_d128_k_last",
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
        group_axis="batch_size",
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
