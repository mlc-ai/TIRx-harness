"""Standalone benchmark for recurrent KDA decode candidates."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    compare_tensors,
    config_from_env,
    load_candidate_override,
    load_task_workloads,
    run_benchmark,
)

TASK_NAME = "kda_decode"

# PR #4279: output and complete state pool vs CuTe-DSL, cast to FP32.
ATOL = 1e-2
RTOL = 1e-2
REQUIRED_MATCHED_RATIO = None
REQUIRED_RMS_ERROR_RATIOS = None
compare = partial(compare_tensors, cast_to_float32=True)


# flashinfer PR #4279's benchmark seeds every case's generator with 42.
RANDOM_SEED = 42
LOWER_BOUND = -5.0
STATE_STD = 0.01


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
            shape_axes=("num_tokens", "num_seqs"),
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


def build_inputs(axes, seed, device):
    """Generate one row's contract inputs with the PR benchmark's draw order."""

    num_tokens = int(axes["num_tokens"])
    num_seqs = int(axes["num_seqs"])
    num_qk_heads = int(axes["num_qk_heads"])
    num_v_heads = int(axes["num_v_heads"])
    head_dim = int(axes["head_dim"])
    lower_bound_gate = bool(axes["lower_bound_gate"])
    standard_decode = num_tokens == 1
    total_tokens = num_seqs * num_tokens
    token_shape = (num_seqs, 1) if standard_decode else (1, total_tokens)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    def rand(shape, dtype=torch.bfloat16):
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    q = rand((*token_shape, num_qk_heads, head_dim))
    k = rand((*token_shape, num_qk_heads, head_dim))
    v = rand((*token_shape, num_v_heads, head_dim))
    beta = torch.sigmoid(randn((*token_shape, num_v_heads)))
    if lower_bound_gate:
        g = randn((*token_shape, num_v_heads, head_dim))
        A_log = torch.log(rand((num_qk_heads,), torch.float32) + 1.0)
        dt_bias = randn((num_qk_heads * head_dim,), torch.float32)
        lower_bound = LOWER_BOUND
    else:
        g = F.logsigmoid(randn((*token_shape, num_v_heads, head_dim), torch.float32))
        g = g.to(torch.bfloat16)
        A_log = dt_bias = lower_bound = None
    if standard_decode:
        cu_seqlens = ssm_state_indices = num_accepted_tokens = num_spec_tokens = None
        num_state_slots = num_seqs
    else:
        cu_seqlens = torch.arange(0, total_tokens + 1, num_tokens, dtype=torch.int32, device=device)
        ssm_state_indices = torch.arange(
            1, total_tokens + 1, dtype=torch.int32, device=device
        ).reshape(num_seqs, num_tokens)
        num_accepted_tokens = torch.ones(num_seqs, dtype=torch.int32, device=device)
        num_spec_tokens = num_tokens - 1
        num_state_slots = total_tokens + 6
    initial_state = (
        randn((num_state_slots, num_v_heads, head_dim, head_dim), torch.float32) * STATE_STD
    ).to(torch.bfloat16)
    return [
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        head_dim**-0.5,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        num_spec_tokens,
        num_accepted_tokens,
        lower_bound,
    ]


def make_inputs(entry, device):
    return build_inputs(entry["axes"], RANDOM_SEED, device)


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
    ssm_state_indices,
    num_spec_tokens,
    num_accepted_tokens,
    lower_bound,
):
    """Bind fresh gate inputs to a TIRx KDA-decode candidate outside timing."""

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
        "ssm_state_indices": ssm_state_indices,
        "num_spec_tokens": num_spec_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "lower_bound": lower_bound,
        "output": torch.empty_like(v),
        # Only the checkpoint slots are written; the rest must carry over.
        "final_state": initial_state.clone(),
    }
    num_seqs = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
    kernel_fn = solution_module.setup(data, num_seqs, q.shape[0] * q.shape[1] // num_seqs)
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["output"], data["final_state"]


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
            "BENCH_KDA_DECODE_KERNEL",
            baseline_decode,
            "kda_decode_candidate",
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
        name="kda_decode_recurrent_d128_h16_bf16",
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
        compare_fn=compare,
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
