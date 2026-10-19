"""Standalone benchmark for FP8 block-scale MoE layer (Alpha-MoE) candidates."""

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

TASK_NAME = "alphamoe"

# PR #4287: independent BF16 oracle and per-element accumulation-order bound.
ATOL = 0.1
RTOL = 0.1
REQUIRED_MATCHED_RATIO = None
REQUIRED_RMS_ERROR_RATIOS = None


def compare(
    candidate,
    reference,
    atol=ATOL,
    rtol=RTOL,
    required_matched_ratio=None,
    required_rms_error_ratios=None,
):
    """The oracle returns (expected, per-element tolerance), outside timing."""
    expected, bound = reference
    return compare_tensors(
        candidate,
        expected,
        atol,
        rtol,
        required_matched_ratio,
        required_rms_error_ratios,
        absolute_bounds=(bound,),
    )


BLOCK_SIZE = 128
FP8_MAX = 448.0
HIDDEN_STATES_STD = 0.25
WEIGHT_STD = 0.125
# Local reproducible performance sweep. PR #4287 does not publish the seed
# or harness for its performance table. Its correctness cases have own seeds.
RANDOM_SEED = 42


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
            shape_axes=(
                "num_tokens",
                "hidden_size",
                "intermediate_size",
                "num_experts",
            ),
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


def quantize_block_2d(values):
    """Per-expert 128x128 block FP8 E4M3 quantization of ``[E, rows, cols]``."""

    experts, rows, columns = values.shape
    groups = values.float().reshape(
        experts, rows // BLOCK_SIZE, BLOCK_SIZE, columns // BLOCK_SIZE, BLOCK_SIZE
    )
    scales = groups.abs().amax(dim=(2, 4)).clamp_min(1.0e-8) / FP8_MAX
    quantized = (groups / scales[:, :, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return (
        quantized.to(torch.float8_e4m3fn).reshape(experts, rows, columns).contiguous(),
        scales.contiguous(),
    )


def make_routing(
    num_tokens,
    num_experts,
    top_k,
    num_shared_experts,
    generator,
    device,
    *,
    balancedness=1.0,
):
    """Seeded top-k routing as in the PR's contract cases: shared experts take
    the last expert slots and trailing routes; ``balancedness`` < 1 makes
    routed expert 0 hot."""

    routed_top_k = top_k - num_shared_experts
    routed_experts = num_experts - num_shared_experts
    scores = torch.randn(
        (num_tokens, routed_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    scores[:, 0] += (1.0 - balancedness) * 6.0
    topk_ids = torch.topk(scores, routed_top_k, dim=-1).indices.to(torch.int32)
    if num_shared_experts:
        shared = torch.arange(routed_experts, num_experts, dtype=torch.int32, device=device).expand(
            num_tokens, num_shared_experts
        )
        topk_ids = torch.cat((topk_ids, shared), dim=-1)
    topk_weights = torch.softmax(
        torch.randn((num_tokens, top_k), dtype=torch.float32, device=device, generator=generator),
        dim=-1,
    )
    return topk_ids, topk_weights


def build_inputs(axes, seed, device, *, routed_scaling_factor, balancedness=1.0):
    """Generate one workload's contract inputs from its axes and a seed."""

    num_tokens = int(axes["num_tokens"])
    hidden_size = int(axes["hidden_size"])
    intermediate_size = int(axes["intermediate_size"])
    num_experts = int(axes["num_experts"])
    num_shared_experts = int(axes.get("num_shared_experts", 0))
    top_k = int(axes["top_k"])
    generator = torch.Generator(device=device).manual_seed(int(seed))

    hidden_states = (
        torch.randn(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * HIDDEN_STATES_STD
    )
    gemm1 = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * WEIGHT_STD
    )
    gemm2 = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * WEIGHT_STD
    )
    gemm1_weights, gemm1_weights_scale = quantize_block_2d(gemm1)
    gemm2_weights, gemm2_weights_scale = quantize_block_2d(gemm2)
    del gemm1, gemm2

    topk_ids, topk_weights = make_routing(
        num_tokens,
        num_experts,
        top_k,
        num_shared_experts,
        generator,
        device,
        balancedness=balancedness,
    )
    return [
        hidden_states,
        topk_ids,
        topk_weights,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        top_k,
        float(routed_scaling_factor),
    ]


def make_inputs(entry, device):
    inputs = entry["raw"]["inputs"]
    return build_inputs(
        entry["axes"],
        int(entry["raw"].get("seed", RANDOM_SEED)),
        device,
        routed_scaling_factor=float(inputs["routed_scaling_factor"]["value"]),
        balancedness=float(entry["raw"].get("balancedness", 1.0)),
    )


def tirx_prepare(
    solution_module,
    hidden_states,
    topk_ids,
    topk_weights,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    top_k,
    routed_scaling_factor,
):
    """Bind the preselected-route contract; quantization/alignment stay timed."""

    num_tokens, hidden_size = hidden_states.shape
    data = {
        "hidden_states": hidden_states,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "top_k": int(top_k),
        "routed_scaling_factor": float(routed_scaling_factor),
        "output": torch.zeros(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        ),
    }
    kernel_fn = solution_module.setup(data, num_tokens)
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

    from . import baseline as baseline_module

    baseline_moe = baseline_module.run
    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_ALPHAMOE_KERNEL",
            baseline_moe,
            "alphamoe_candidate",
        )
    baseline_sanity = candidate_fn is None and candidate is baseline_moe
    if baseline_sanity:
        print("TRT-LLM baseline sanity/timing only; no AlphaMoE candidate is being validated.")
        candidate = baseline_module.run_prepared
        candidate_prepare_fn = baseline_module.prepare
    reference = partial(
        load_task_reference(TASK_NAME, "reference_for_check"),
        atol=ATOL,
        rtol=RTOL,
    )
    rows = run_benchmark(
        name=(
            "alphamoe_trtllm_baseline_sanity"
            if baseline_sanity
            else "alphamoe_fp8_block_scale_moe_bf16"
        ),
        workloads=make_workloads(cfg) if workloads is None else workloads,
        make_inputs=make_inputs,
        candidate_fn=candidate,
        baseline_fn=baseline_module.run_prepared,
        baseline_prepare_fn=baseline_module.prepare,
        candidate_prepare_fn=candidate_prepare_fn,
        device=device,
        warmup=cfg.warmup,
        iters=cfg.iters,
        trials=cfg.trials,
        atol=0.0 if baseline_sanity else ATOL,
        rtol=0.0 if baseline_sanity else RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        required_rms_error_ratios=REQUIRED_RMS_ERROR_RATIOS,
        # TRT-LLM is the timing baseline. Candidate correctness uses the oracle.
        # The no-candidate mode only checks eager vs captured TRT-LLM.
        reference_fn=baseline_moe if baseline_sanity else reference,
        compare_fn=None if baseline_sanity else compare,
        group_axis="num_tokens",
        correctness_runs=cfg.correctness_runs,
        check_after_timing=cfg.check_after_timing,
        require_repeatable_outputs=cfg.require_repeatable_outputs,
    )
    for row in rows:
        row["correctness_policy"] = (
            "trtllm_baseline_sanity" if baseline_sanity else "pr4287_alphamoe_oracle"
        )
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
