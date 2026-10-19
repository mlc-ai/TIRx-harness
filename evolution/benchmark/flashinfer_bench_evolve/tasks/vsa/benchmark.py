"""Standalone benchmark for VSA block-sparse attention candidates."""

from __future__ import annotations

import math
from functools import partial
from itertools import groupby

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

TASK_NAME = "vsa"

# PR #4593: every element vs the independent masked FP32 attention oracle.
# PR #4612's blk64 checks use assert_close(reference.float(), output.float()).
ATOL = 1e-2
RTOL = 1e-2
REQUIRED_MATCHED_RATIO = None
REQUIRED_RMS_ERROR_RATIOS = None
compare = partial(compare_tensors, cast_to_float32=False)
compare_blk64 = partial(compare_tensors, cast_to_float32=True, reference_first=True)


# flashinfer PR #4612's sweeps seed with 0; every row here reseeds so it is
# reproducible on its own.
RANDOM_SEED = 0


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


def pooled_topk_indices(q, k, block_size, topk, sm_scale):
    """PR #4612's compress step: the top-k KV blocks per (head, query block)
    by softmax of the mean-pooled block scores, as ascending block ids."""

    S, H, D = q.shape
    MB, NB = S // block_size, k.shape[0] // block_size
    q_pooled = q.view(MB, block_size, H, D).float().mean(dim=1).permute(1, 0, 2)
    k_pooled = k.view(NB, block_size, H, D).float().mean(dim=1).permute(1, 0, 2)
    scores = torch.softmax(q_pooled @ k_pooled.transpose(-1, -2) * sm_scale, dim=-1)
    selected = torch.topk(scores, topk, dim=-1).indices.sort(dim=-1).values
    return selected.to(torch.int32).contiguous()


def shared_random_indices(num_heads, num_blocks, topk, seed):
    """PR #4612's head-shared random BSR: topk sorted random columns per row."""

    generator = torch.Generator().manual_seed(int(seed))
    rows = [
        torch.randperm(num_blocks, generator=generator)[:topk].sort().values
        for _ in range(num_blocks)
    ]
    selected = torch.stack(rows).to(torch.int32)
    return selected.unsqueeze(0).expand(num_heads, -1, -1).contiguous()


def build_inputs(axes, seed, device):
    """Generate one row's contract inputs from its axes."""

    seq_len = int(axes["seq_len"])
    num_heads = int(axes["num_heads"])
    head_dim = int(axes["head_dim"])
    block_size = int(axes["block_size"])
    topk = int(axes["topk"])
    num_blocks = seq_len // block_size
    sm_scale = 1.0 / math.sqrt(head_dim)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    def randn():
        return torch.randn(
            (seq_len, num_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )

    q, k, v = randn(), randn(), randn()
    if axes["head_shared_mask"]:
        q2k_indices = shared_random_indices(num_heads, num_blocks, topk, seed).to(device)
    else:
        q2k_indices = pooled_topk_indices(q, k, block_size, topk, sm_scale)
    kv_block_lens = None
    num_partial = int(axes["num_partial_blocks"])
    if num_partial:
        # FastVideo flattens (t, h, w) tiles, so the partial last w-tile of
        # every (t, h) row recurs once per num_blocks // num_partial blocks.
        period = num_blocks // num_partial
        kv_block_lens = torch.full((num_blocks,), block_size, dtype=torch.int32, device=device)
        kv_block_lens[period - 1 :: period] = int(axes["partial_block_len"])
    return [q, k, v, q2k_indices, kv_block_lens, block_size, sm_scale]


def make_inputs(entry, device):
    return build_inputs(entry["axes"], RANDOM_SEED, device)


def tirx_prepare(solution_module, q, k, v, q2k_indices, kv_block_lens, block_size, sm_scale):
    """Bind fresh gate inputs to a TIRx VSA candidate outside timing."""

    data = {
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "kv_block_lens": kv_block_lens,
        "block_size": int(block_size),
        "sm_scale": float(sm_scale),
        "output": torch.empty_like(q),
    }
    kernel_fn = solution_module.setup(
        data, q.shape[0], q.shape[1], int(block_size), q2k_indices.shape[-1]
    )
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

    baseline_attention = baseline_module.run
    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_VSA_KERNEL",
            baseline_attention,
            "vsa_candidate",
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
    selected = make_workloads(cfg) if workloads is None else workloads
    # Group adjacent rows so the comparison precision follows the block size
    # without changing the caller's workload order.
    groups = [
        (block_size, list(group))
        for block_size, group in groupby(selected, key=lambda w: int(w["axes"]["block_size"]))
    ] or [(128, [])]
    rows = []
    for block_size, group in groups:
        rows.extend(
            run_benchmark(
                name="vsa_block_sparse_attention_d128_bf16",
                workloads=group,
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
                reference_fn=load_task_reference(TASK_NAME),
                compare_fn=compare_blk64 if block_size == 64 else compare,
                group_axis="block_size",
                correctness_runs=cfg.correctness_runs,
                check_after_timing=cfg.check_after_timing,
                require_repeatable_outputs=cfg.require_repeatable_outputs,
            )
        )
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
