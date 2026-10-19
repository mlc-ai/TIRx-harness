"""Standalone benchmark for KDA training-step (forward + backward) candidates."""

from __future__ import annotations

from functools import partial

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    compare_tensors,
    config_from_env,
    load_candidate_override,
    load_task_workloads,
    run_benchmark,
)

TASK_NAME = "kda_backward"

# PR #4636: all ten training outputs vs FLA, torch.testing.assert_close.
ATOL = 1e-2
RTOL = 1e-2
REQUIRED_MATCHED_RATIO = None
REQUIRED_RMS_ERROR_RATIOS = None
compare = partial(compare_tensors, cast_to_float32=False)


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=2, iters=20, trials=2)


def make_workloads(config: BenchConfig | None = None):
    cfg = config or default_config()
    official = []
    if cfg.include_official:
        for row in load_task_workloads(
            TASK_NAME,
            cfg.max_official,
            shape_mode=cfg.shape_mode,
            # Prioritize head count so max selects the packed 1024x8 H=96 case.
            shape_axes=("num_qk_heads", "num_v_heads", "total_tokens", "batch_size"),
        ):
            workload = row["workload"]
            official.append(
                {
                    "suite": "official",
                    "id": workload["uuid"],
                    "axes": workload["axes"],
                    "seq_lens": tuple(workload["inputs"]["cu_seqlens"]["value"]),
                    "raw": workload,
                }
            )
    return official


def build_inputs(seq_lens, num_qk_heads, num_v_heads, seed, device, *, scale=128**-0.5):
    """PR #4636's ``_make_inputs``: same draws, same order, same scales."""

    generator = torch.Generator(device=device).manual_seed(int(seed))
    total_tokens = sum(seq_lens)
    qk_shape = (1, total_tokens, num_qk_heads, 128)
    value_shape = (1, total_tokens, num_v_heads, 128)
    state_shape = (len(seq_lens), num_v_heads, 128, 128)

    def randn(shape, multiplier=1.0, dtype=torch.float32):
        return (torch.randn(shape, generator=generator, device=device) * multiplier).to(dtype)

    q = randn(qk_shape, dtype=torch.bfloat16)
    k = randn(qk_shape, dtype=torch.bfloat16)
    v = randn(value_shape, dtype=torch.bfloat16)
    g = randn(value_shape, 0.1, torch.bfloat16)
    beta = randn(value_shape[:-1], dtype=torch.bfloat16)
    A_log = torch.log(torch.rand((num_v_heads,), generator=generator, device=device) + 1.0)
    dt_bias = randn((num_v_heads, 128), 0.1).reshape(-1)
    initial_state = randn(state_shape, 0.02)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int64, device=device
    )
    do = randn(value_shape, 0.1, torch.bfloat16)
    dfinal_state = randn(state_shape, 0.1)
    return [
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        float(scale),
        initial_state,
        cu_seqlens,
        do,
        dfinal_state,
    ]


def make_inputs(entry, device):
    axes = entry["axes"]
    return build_inputs(
        tuple(entry["seq_lens"]),
        axes["num_qk_heads"],
        axes["num_v_heads"],
        axes["seed"],
        device,
        scale=float(entry["raw"]["inputs"]["scale"]["value"]),
    )


def tirx_prepare(
    solution_module,
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    scale,
    initial_state,
    cu_seqlens,
    do,
    dfinal_state,
):
    """Bind fresh gate inputs to a TIRx KDA training-step candidate outside timing."""

    batch_size, total_tokens, num_heads, _ = q.shape
    data = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": float(scale),
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "do": do,
        "dfinal_state": dfinal_state,
        "output": torch.empty_like(v),
        "final_state": torch.empty_like(initial_state),
        "dq": torch.empty_like(q),
        "dk": torch.empty_like(k),
        "dv": torch.empty_like(v),
        "dg": torch.empty_like(g),
        "dbeta": torch.empty_like(beta),
        "dA_log": torch.empty_like(A_log),
        "ddt_bias": torch.empty_like(dt_bias),
        "dinitial_state": torch.empty_like(initial_state),
    }
    kernel_fn = solution_module.setup(data, batch_size, total_tokens, num_heads)
    return kernel_fn, data


OUTPUT_NAMES = (
    "output",
    "final_state",
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
    "dA_log",
    "ddt_bias",
    "dinitial_state",
)


def tirx_run(kernel_fn, data):
    kernel_fn()
    return tuple(data[name] for name in OUTPUT_NAMES)


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
):
    """Run the correctness gate and timing sweep for this task."""

    from . import baseline as baseline_module

    baseline_backward = baseline_module.run
    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_KDA_BACKWARD_KERNEL",
            baseline_backward,
            "kda_backward_candidate",
        )
    # Real candidates hoist their setup outside timing, so the baseline is
    # timed as a CUDA-graph replay through its prepare step (kernel-only span,
    # no host launch gaps). The baseline-vs-baseline sanity path keeps the raw
    # run() on both arms so it still reads ~1.0x.
    baseline_kwargs = {"baseline_fn": baseline_backward}
    if candidate is not baseline_backward:
        baseline_kwargs = {
            "baseline_fn": baseline_module.run_prepared,
            "baseline_prepare_fn": baseline_module.prepare,
        }
    return run_benchmark(
        name="kda_backward_k128_v128_bf16",
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
        compare_fn=compare,
        group_axis="total_tokens",
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
