"""Standalone benchmark for FP8 block-scale MoE candidates."""

from __future__ import annotations

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    compare_outputs,
    config_from_env,
    include_synthetic_workloads,
    load_candidate_override,
    load_safetensor,
    load_task_workloads,
    normalize_outputs,
    rand_tensor,
    run_benchmark,
)
TASK_NAME = "moe"

ATOL = 1.0
RTOL = 0.3
REQUIRED_MATCHED_RATIO = 0.9
# Reject zeroed and rescaled outputs even when they fit within ATOL.
REQUIRED_RMS_ERROR_RATIOS = (0.2,)
# Allow rare reference/baseline outliers in large workloads while
# rejecting missing contributions that affect only a few tokens.
REQUIRED_TOKEN_MATCHED_RATIO = 0.9999

HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS_GLOBAL = 256
NUM_LOCAL_EXPERTS = 32
BLOCK_SIZE = 128

LARGE_WORKLOADS = [
    {"seq_len": 32768},
]


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=3, iters=30, trials=2)


def _entry_seed(entry):
    raw_id = entry["raw"]["uuid"] if entry["raw"] is not None else entry["id"]
    return int.from_bytes(f"{entry['suite']}:{raw_id}:{entry['axes']}".encode(), "little") % (2**32)


def _positive_scale(shape, device):
    return torch.empty(shape, dtype=torch.float32, device=device).uniform_(0.01, 0.1)


def make_workloads(config: BenchConfig | None = None):
    # The official rows are always included for this task; unlike the other
    # benchmarks there is no BENCH_INCLUDE_OFFICIAL opt-out here.
    cfg = config or default_config()
    official = []
    for row in load_task_workloads(
        TASK_NAME,
        cfg.max_official,
        shape_mode=cfg.shape_mode,
        shape_axes=("seq_len",),
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
                "id": f"T{w['seq_len']}",
                "axes": dict(w),
                "raw": None,
            }
            for w in LARGE_WORKLOADS
        ]
    return official + large


def make_inputs(entry, device):
    seq_len = entry["axes"]["seq_len"]
    torch.manual_seed(_entry_seed(entry))
    if entry["raw"] is not None:
        inputs = entry["raw"]["inputs"]
        routing_logits = load_safetensor(inputs["routing_logits"], device)
        routing_bias = load_safetensor(inputs["routing_bias"], device)
        local_expert_offset = int(inputs["local_expert_offset"]["value"])
        routed_scaling_factor = float(inputs["routed_scaling_factor"]["value"])
    else:
        routing_logits = rand_tensor((seq_len, NUM_EXPERTS_GLOBAL), torch.float32, device)
        routing_bias = torch.zeros((NUM_EXPERTS_GLOBAL,), dtype=torch.bfloat16, device=device)
        local_expert_offset = 0
        routed_scaling_factor = 2.5

    return [
        routing_logits,
        routing_bias,
        rand_tensor((seq_len, HIDDEN_SIZE), torch.float8_e4m3fn, device),
        _positive_scale((HIDDEN_SIZE // BLOCK_SIZE, seq_len), device),
        rand_tensor(
            (NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        _positive_scale(
            (
                NUM_LOCAL_EXPERTS,
                (2 * INTERMEDIATE_SIZE) // BLOCK_SIZE,
                HIDDEN_SIZE // BLOCK_SIZE,
            ),
            device,
        ),
        rand_tensor(
            (NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        _positive_scale(
            (
                NUM_LOCAL_EXPERTS,
                HIDDEN_SIZE // BLOCK_SIZE,
                INTERMEDIATE_SIZE // BLOCK_SIZE,
            ),
            device,
        ),
        local_expert_offset,
        routed_scaling_factor,
    ]


def tirx_prepare(
    solution_module,
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """Bind fresh gate inputs to a TIRx MoE candidate outside timing."""

    seq_len = routing_logits.shape[0]
    data = {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "local_expert_offset": local_expert_offset,
        "routed_scaling_factor": routed_scaling_factor,
        "output": torch.empty(
            (seq_len, hidden_states.shape[1]),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        ),
    }
    kernel_fn = solution_module.setup(data, seq_len)
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["output"]


def compare_moe_outputs(
    candidate, baseline, atol, rtol, required_matched_ratio, required_rms_error_ratios
):
    """Apply global and per-token normalized RMS checks to the MoE output."""
    result = compare_outputs(
        candidate, baseline, atol, rtol, required_matched_ratio, required_rms_error_ratios
    )
    if not result[0]:
        return result

    output = normalize_outputs(candidate)[0].float()
    reference = normalize_outputs(baseline)[0].float()
    if reference.numel() == 0:
        return result
    error = output - reference
    token_rms = error.square().mean(dim=-1).sqrt() / (reference.square().mean(dim=-1).sqrt() + 1e-8)
    bad_tokens = (token_rms >= required_rms_error_ratios[0]) & (error.abs().amax(dim=-1) > 1e-6)
    bad_count = int(bad_tokens.sum().item())
    token_matched = 1.0 - bad_count / bad_tokens.numel()
    if token_matched < REQUIRED_TOKEN_MATCHED_RATIO:
        return (
            False,
            *result[1:5],
            f"per-token normalized RMS matched ratio {token_matched:.6g} "
            f"< {REQUIRED_TOKEN_MATCHED_RATIO:.6g} ({bad_count} failing tokens)",
        )
    return result


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
):
    """Run the correctness gate and timing sweep for this task."""

    from .baseline import run as baseline_moe

    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_MOE_KERNEL",
            baseline_moe,
            "moe_candidate",
        )
    return run_benchmark(
        name="moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
        workloads=make_workloads(cfg) if workloads is None else workloads,
        make_inputs=make_inputs,
        baseline_fn=baseline_moe,
        candidate_fn=candidate,
        candidate_prepare_fn=candidate_prepare_fn,
        compare_fn=compare_moe_outputs,
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
