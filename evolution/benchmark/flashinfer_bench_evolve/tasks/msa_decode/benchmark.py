"""Standalone benchmark for MiniMax sparse attention (MSA) decode candidates."""

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

TASK_NAME = "msa_decode"

# PR #4355: independent sparse-attention oracle; tolerance follows KV dtype.
ATOL = 1e-2
RTOL = 1e-2
REQUIRED_MATCHED_RATIO = None
REQUIRED_RMS_ERROR_RATIOS = None
FP8_ATOL = 0.1
FP8_RTOL = 0.1


def gate_for(kv_dtype):
    return (FP8_ATOL, FP8_RTOL) if str(kv_dtype) == "float8_e4m3fn" else (ATOL, RTOL)


compare = partial(compare_tensors, cast_to_float32=False)


HEAD_DIM = 128
BLOCK_SIZE = 128
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


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
            shape_axes=("total_q", "seqlen_kv"),
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


def make_q2k_indices(batch_size, seqlen_q, seqlen_kv, num_kv_heads, topk, seed, device):
    """The PR benchmark's random-valid, bottom-right-causal block selection.

    Per (request, query token, kv head) in that draw order: a random subset of
    the blocks visible to the token, ascending, -1 padded."""

    total_q = batch_size * seqlen_q
    out = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    offset = seqlen_kv - seqlen_q
    for row in range(total_q):
        visible_blocks = (offset + row % seqlen_q + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
        for kv_head in range(num_kv_heads):
            selected = torch.randperm(visible_blocks, generator=generator)
            selected = selected[: min(topk, visible_blocks)].sort().values
            out[kv_head, row, : selected.numel()] = selected.to(torch.int32)
    return out.to(device)


def to_pages(logical, batch_size, seqlen_kv):
    """The PR benchmark's paged layout: 128-token pages stored in reverse order."""

    total_k, num_kv_heads, head_dim = logical.shape
    pages_per_seq = (seqlen_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_pages = batch_size * pages_per_seq
    padded = logical.view(batch_size, seqlen_kv, num_kv_heads, head_dim)
    if pages_per_seq * BLOCK_SIZE != seqlen_kv:
        padded = logical.new_zeros((batch_size, pages_per_seq * BLOCK_SIZE, num_kv_heads, head_dim))
        padded[:, :seqlen_kv] = logical.view(batch_size, seqlen_kv, num_kv_heads, head_dim)
    pages = (
        padded.view(batch_size, pages_per_seq, BLOCK_SIZE, num_kv_heads, head_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(total_pages, num_kv_heads, BLOCK_SIZE, head_dim)
    )
    page_table = torch.arange(total_pages - 1, -1, -1, dtype=torch.int32, device=logical.device)
    return pages.flip(0).contiguous(), page_table.view(batch_size, pages_per_seq).contiguous()


def build_inputs(axes, device):
    """Generate one row's contract inputs with the PR benchmark's draw order."""

    batch_size = int(axes["batch_size"])
    seqlen_q = int(axes["seqlen_q"])
    seqlen_kv = int(axes["seqlen_kv"])
    num_qo_heads = int(axes["num_qo_heads"])
    num_kv_heads = int(axes["num_kv_heads"])
    topk = int(axes["topk"])
    seed = int(axes["seed"])
    q_dtype = DTYPES[axes["q_dtype"]]
    kv_dtype = DTYPES[axes["kv_dtype"]]
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dtype):
        values = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
        return (values / 3.0).to(dtype)

    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_kv
    q = randn((total_q, num_qo_heads, HEAD_DIM), q_dtype)
    k = randn((total_k, num_kv_heads, HEAD_DIM), kv_dtype)
    v = randn((total_k, num_kv_heads, HEAD_DIM), kv_dtype)
    cu_seqlens_k = torch.arange(0, total_k + 1, seqlen_kv, dtype=torch.int32, device=device)
    q2k_indices = make_q2k_indices(
        batch_size, seqlen_q, seqlen_kv, num_kv_heads, topk, seed, device
    )
    page_table = seqused_k = None
    if axes["kv_layout"] == "paged":
        k, page_table = to_pages(k, batch_size, seqlen_kv)
        v, _ = to_pages(v, batch_size, seqlen_kv)
        seqused_k = torch.full((batch_size,), seqlen_kv, dtype=torch.int32, device=device)
        cu_seqlens_k = None
    return [q, k, v, q2k_indices, cu_seqlens_k, page_table, seqused_k, seqlen_q, HEAD_DIM**-0.5]


def make_inputs(entry, device):
    return build_inputs(entry["axes"], device)


def tirx_prepare(
    solution_module,
    q,
    k,
    v,
    q2k_indices,
    cu_seqlens_k,
    page_table,
    seqused_k,
    seqlen_q,
    softmax_scale,
):
    """Bind fresh gate inputs to a TIRx MSA-decode candidate outside timing."""

    data = {
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "cu_seqlens_k": cu_seqlens_k,
        "page_table": page_table,
        "seqused_k": seqused_k,
        "seqlen_q": int(seqlen_q),
        "softmax_scale": float(softmax_scale),
        "output": torch.empty_like(q),
    }
    kernel_fn = solution_module.setup(data, q.shape[0] // int(seqlen_q), int(seqlen_q))
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return (data["output"],)


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
            "BENCH_MSA_DECODE_KERNEL",
            baseline_attention,
            "msa_decode_candidate",
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
    groups = {}
    for entry in selected:
        groups.setdefault(str(entry["axes"]["kv_dtype"]), []).append(entry)
    rows = []
    for kv_dtype, group in (groups or {"bfloat16": []}).items():
        atol, rtol = gate_for(kv_dtype)
        rows.extend(
            run_benchmark(
                name="msa_sparse_decode_hd128_blk128",
                workloads=group,
                make_inputs=make_inputs,
                candidate_fn=candidate,
                **baseline_kwargs,
                candidate_prepare_fn=candidate_prepare_fn,
                device=device,
                warmup=cfg.warmup,
                iters=cfg.iters,
                trials=cfg.trials,
                atol=atol,
                rtol=rtol,
                required_matched_ratio=REQUIRED_MATCHED_RATIO,
                required_rms_error_ratios=REQUIRED_RMS_ERROR_RATIOS,
                reference_fn=load_task_reference(TASK_NAME),
                compare_fn=compare,
                group_axis="seqlen_q",
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
