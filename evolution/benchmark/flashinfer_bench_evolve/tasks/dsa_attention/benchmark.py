"""Standalone benchmark for DSA sparse attention candidates."""

from __future__ import annotations

import hashlib

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    config_from_env,
    include_synthetic_workloads,
    load_candidate_override,
    load_safetensor,
    load_task_workloads,
    rand_tensor,
    run_benchmark,
)

TASK_NAME = "dsa_attention"

ATOL = 1e-2
RTOL = 1e-2
REQUIRED_MATCHED_RATIO = 0.999

LARGE_WORKLOADS = [
    {"num_tokens": 8, "num_pages": 8192},
    {"num_tokens": 8, "num_pages": 32768},
    {"num_tokens": 16, "num_pages": 32768},
    {"num_tokens": 32, "num_pages": 32768},
    {"num_tokens": 64, "num_pages": 32768},
    {"num_tokens": 128, "num_pages": 32768},
    {"num_tokens": 256, "num_pages": 32768},
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
            shape_axes=("num_tokens", "num_pages"),
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
                "id": f"T{w['num_tokens']}_P{w['num_pages']}",
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
    axes = entry["axes"]
    num_tokens = axes["num_tokens"]
    num_pages = axes["num_pages"]
    torch.manual_seed(_entry_seed(entry))
    if entry["raw"] is not None:
        spec = entry["raw"]["inputs"]["sparse_indices"]
        sparse_indices = load_safetensor(spec, device).to(torch.int32)
        sm_scale = float(entry["raw"]["inputs"]["sm_scale"]["value"])
    else:
        total_tokens = num_pages * 64
        sparse_indices = torch.randint(
            0, total_tokens, (num_tokens, 2048), dtype=torch.int32, device=device
        )
        sm_scale = 0.1352337788608801

    return [
        rand_tensor((num_tokens, 16, 512), torch.bfloat16, device),
        rand_tensor((num_tokens, 16, 64), torch.bfloat16, device),
        rand_tensor((num_pages, 64, 512), torch.bfloat16, device),
        rand_tensor((num_pages, 64, 64), torch.bfloat16, device),
        sparse_indices,
        sm_scale,
    ]


def tirx_prepare(
    solution_module,
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    sparse_indices,
    sm_scale,
):
    """Bind fresh gate inputs to a TIRx DSA candidate outside timing."""

    num_tokens = q_nope.shape[0]
    num_pages = ckv_cache.shape[0]
    data = {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "sparse_indices": sparse_indices,
        "sm_scale": sm_scale,
        "O": torch.empty_like(q_nope),
        "LSE": torch.empty(
            (num_tokens, q_nope.shape[1]),
            dtype=torch.float32,
            device=q_nope.device,
        ),
    }
    kernel_fn = solution_module.setup(data, num_tokens, num_pages)
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return (data["O"],)


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
):
    """Run the correctness gate and timing sweep for this task."""

    from . import baseline as baseline_module

    baseline_attention = baseline_module.run
    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_DSA_ATTENTION_KERNEL",
            baseline_attention,
            "dsa_attention_candidate",
        )
    # Real candidates hoist their setup outside timing, so the baseline is
    # timed kernel-only through its prepare step. The baseline-vs-baseline
    # sanity path keeps the raw run() on both arms so it still reads ~1.0x.
    baseline_kwargs = {"baseline_fn": baseline_attention}
    if candidate is not baseline_attention:
        baseline_kwargs = {
            "baseline_fn": baseline_module.run_prepared,
            "baseline_prepare_fn": baseline_module.prepare,
        }
    return run_benchmark(
        name="dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
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
        group_axis="num_tokens",
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
