"""Standalone benchmark for DeepSeek-V4 sparse MLA decode candidates."""

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

TASK_NAME = "mla_dsv4"

# PR #4573: test_trtllm_gen_sparse_mla_dsv4.py::_assert_close.
ATOL = 8e-4
RTOL = 2e-2
REQUIRED_MATCHED_RATIO = None
REQUIRED_RMS_ERROR_RATIOS = None
FP8_ATOL = 1e-1
FP8_RTOL = 1e-1
FP8_REQUIRED_RMS_ERROR_RATIOS = None
compare = partial(compare_tensors, cast_to_float32=True)


KV_DTYPES = {"bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn}


def gate_for(kv_dtype):
    """(atol, rtol, required_rms_error_ratios) for a row's kv_dtype."""

    if KV_DTYPES[str(kv_dtype)] == torch.float8_e4m3fn:
        return FP8_ATOL, FP8_RTOL, FP8_REQUIRED_RMS_ERROR_RATIOS
    return ATOL, RTOL, REQUIRED_RMS_ERROR_RATIOS


# Upstream random input amplitudes (same scale for BF16 and FP8 rows).
SEED_BASE = 2026
SWA_TOPK = 128
QUERY_RANDOM_SCALE = 0.05
KV_RANDOM_SCALE = 0.05
SWA_KV_OFFSET = -0.20
COMPRESSED_KV_OFFSET = 0.25
SINK_MEAN = 0.0
SINK_STD = 0.05
BMM1_SCALE = 512**-0.55


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=3, iters=50, trials=3)


def make_workloads(config: BenchConfig | None = None):
    cfg = config or default_config()
    official = []
    if cfg.include_official:
        # Seeds follow the row's position in the full published list.
        seeds = {
            row["workload"]["uuid"]: SEED_BASE + idx
            for idx, row in enumerate(load_task_workloads(TASK_NAME, 0))
        }
        for row in load_task_workloads(
            TASK_NAME,
            cfg.max_official,
            shape_mode=cfg.shape_mode,
            shape_axes=("num_heads", "sparse_topk", "swa_seq_len"),
        ):
            workload = row["workload"]
            official.append(
                {
                    "suite": "official",
                    "id": workload["uuid"],
                    "axes": workload["axes"],
                    "raw": workload,
                    "seed": seeds[workload["uuid"]],
                }
            )
    return official


def _flat_indices(logical, block_table, page_size):
    """Map per-request logical token positions to flat pool rows (-1 kept)."""

    valid = logical >= 0
    safe = logical.clamp_min(0)
    physical = block_table[safe // page_size] * page_size + safe % page_size
    return torch.where(valid, physical, torch.full_like(physical, -1)).to(torch.int32)


def build_inputs(axes, seed, device):
    """Generate one row's inputs with upstream scales and distinct KV offsets.

    Query and pools are drawn in bf16 and cast to the row's ``kv_dtype``; pools
    are laid out per ``kv_layout`` (HND ``[pages, 1, page, 512]`` or NHD
    ``[pages, page, 1, 512]``)."""

    num_heads = int(axes["num_heads"])
    num_seqs = int(axes["num_seqs"])
    max_q_len = int(axes["max_q_len"])
    head_dim = int(axes["head_dim"])
    swa_page = int(axes["swa_page_size"])
    comp_page = int(axes["compressed_page_size"])
    comp_topk = int(axes["compressed_topk"])
    generator = torch.Generator(device=device).manual_seed(int(seed))

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    def randperm(n):
        return torch.randperm(n, device=device, generator=generator)

    def pool(seq_len_base, page_size, value_offset):
        seq_lens = seq_len_base + page_size * torch.arange(num_seqs, device=device)
        pages_per_seq = (int(seq_lens.max()) + page_size - 1) // page_size
        block_table = randperm(num_seqs * pages_per_seq).view(num_seqs, pages_per_seq)
        cache = (
            (randn((num_seqs * pages_per_seq, 1, page_size, head_dim)) * KV_RANDOM_SCALE)
            .add_(value_offset)
            .clamp_(-1.0, 1.0)
        )
        return seq_lens.to(torch.int32), block_table, cache

    dense_query = str(axes.get("query_layout", "varlen")) == "dense"
    if dense_query:
        # upstream's dense-q guard: every request has max_q_len queries, the
        # query is [B, max_q_len, H, 512] and cum_seq_lens_q / max_q_len are None.
        q_lens = [max_q_len] * num_seqs
        cum_seq_lens_q = None
    else:
        # q_lens = round(linspace(ceil(s_q / 2), s_q, B)) -> [3, 4, 5] for s_q = 5.
        min_len = (max_q_len + 1) // 2
        q_lens = [
            round(min_len + (max_q_len - min_len) * i / max(num_seqs - 1, 1))
            for i in range(num_seqs)
        ]
        cum_seq_lens_q = torch.tensor(
            [0, *torch.tensor(q_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
    sum_q = sum(q_lens)
    query = randn((sum_q, num_heads, head_dim)) * QUERY_RANDOM_SCALE
    if dense_query:
        query = query.view(num_seqs, max_q_len, num_heads, head_dim)
    sinks = randn((num_heads,), torch.float32) * SINK_STD + SINK_MEAN

    seq_lens, swa_table, swa_kv_cache = pool(int(axes["swa_seq_len"]), swa_page, SWA_KV_OFFSET)
    swa_columns = torch.arange(SWA_TOPK, device=device)
    if comp_page:
        comp_base = max(int(axes["compressed_seq_len"]), comp_topk, max_q_len)
        comp_seq_lens, comp_table, compressed_kv_cache = pool(
            comp_base, comp_page, COMPRESSED_KV_OFFSET
        )
    else:
        compressed_kv_cache = None

    index_rows = []
    lens = []
    for b in range(num_seqs):
        for q_idx in range(q_lens[b]):
            # Causal window of the last 128 positions, valid entries first and
            # -1 padding at the tail when shorter (upstream's layout).
            token = int(seq_lens[b]) - q_lens[b] + q_idx
            num_valid = min(SWA_TOPK, token + 1)
            logical = torch.full_like(swa_columns, -1)
            logical[:num_valid] = torch.arange(token - num_valid + 1, token + 1, device=device)
            row = [_flat_indices(logical, swa_table[b], swa_page)]
            active = SWA_TOPK
            if comp_page:
                # Random sample without replacement (random order) of the
                # request's compressed positions.
                c_len = int(comp_seq_lens[b])
                sample = randperm(c_len)[:comp_topk]
                logical = torch.full((comp_topk,), -1, device=device, dtype=torch.int64)
                logical[: sample.numel()] = sample
                row.append(_flat_indices(logical, comp_table[b], comp_page))
                if comp_page == 2:
                    active += min(c_len, comp_topk)
                else:
                    active += min(max(comp_topk - (comp_topk // 16) * b - q_idx, 1), comp_topk)
            index_rows.append(torch.cat(row))
            lens.append(active)
    sparse_indices = torch.stack(index_rows).contiguous()
    sparse_topk_lens = torch.tensor(lens, dtype=torch.int32, device=device)

    kv_dtype = KV_DTYPES[str(axes.get("kv_dtype", "bfloat16"))]
    kv_layout = str(axes.get("kv_layout", "HND"))

    def as_row(pool):
        if pool is None:
            return None
        pool = pool.to(kv_dtype)
        return pool.transpose(1, 2).contiguous() if kv_layout == "NHD" else pool

    return [
        query.to(kv_dtype),
        as_row(swa_kv_cache),
        as_row(compressed_kv_cache),
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        cum_seq_lens_q,
        None if dense_query else max_q_len,
        sinks,
        BMM1_SCALE,
        1.0,
        kv_layout,
    ]


def make_inputs(entry, device):
    return build_inputs(entry["axes"], entry.get("seed", SEED_BASE), device)


def tirx_prepare(
    solution_module,
    query,
    swa_kv_cache,
    compressed_kv_cache,
    sparse_indices,
    sparse_topk_lens,
    seq_lens,
    cum_seq_lens_q,
    max_q_len,
    sinks,
    bmm1_scale,
    bmm2_scale,
    kv_layout,
):
    """Bind fresh gate inputs to a TIRx DSv4 sparse-MLA candidate outside timing."""

    data = {
        "query": query,
        "swa_kv_cache": swa_kv_cache,
        "compressed_kv_cache": compressed_kv_cache,
        "sparse_indices": sparse_indices,
        "sparse_topk_lens": sparse_topk_lens,
        "seq_lens": seq_lens,
        "cum_seq_lens_q": cum_seq_lens_q,
        "max_q_len": None if max_q_len is None else int(max_q_len),
        "sinks": sinks,
        "bmm1_scale": float(bmm1_scale),
        "bmm2_scale": float(bmm2_scale),
        "kv_layout": str(kv_layout),
        "output": torch.empty(query.shape, dtype=torch.bfloat16, device=query.device),
    }
    kernel_fn = solution_module.setup(data, sparse_indices.shape[0], sparse_indices.shape[1])
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

    baseline_decode = baseline_module.run
    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_MLA_DSV4_KERNEL",
            baseline_decode,
            "mla_dsv4_candidate",
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
    # bf16 and fp8 rows carry different gates, so each dtype runs as its own
    # sweep; the rows are returned in one list.
    selected = make_workloads(cfg) if workloads is None else workloads
    groups = {
        kv_dtype: [w for w in selected if str(w["axes"].get("kv_dtype", "bfloat16")) == kv_dtype]
        for kv_dtype in KV_DTYPES
    }
    groups = {k: v for k, v in groups.items() if v} or {"bfloat16": []}
    rows = []
    for kv_dtype, group in groups.items():
        atol, rtol, rms_ratios = gate_for(kv_dtype)
        rows.extend(
            run_benchmark(
                name="mla_dsv4_sparse_decode_h512",
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
                required_rms_error_ratios=rms_ratios,
                reference_fn=load_task_reference(TASK_NAME),
                compare_fn=compare,
                group_axis="num_heads",
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
