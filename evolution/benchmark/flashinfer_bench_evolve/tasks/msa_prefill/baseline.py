#!/usr/bin/env python3
"""MiniMax MSA (``fmha_sm100.sparse_atten_func``) baseline for MSA prefill.

flashinfer PR #4355 benchmarks its kernel against MiniMax's public sparse
attention, MiniMax-AI/MSA at 80434d7f (``benchmarks/bench_blackwell_msa_sm100.py``,
``baseline_mode="minimax_public"``): ``fmha_sm100.build_k2q_csr`` turns the
contract's ``q2k_indices`` into the kernel's CSR reverse index and forward
schedule, then ``sparse_atten_func`` runs on the contract's tensors as they
are, flat varlen or paged, bf16 or fp8 E4M3 K/V under a bf16 q. This module is
that arm: the CSR build, the schedule and the JIT are ``prepare()`` work, the
timed span is the kernel.

MiniMax's forward accepts bf16 and fp8 storage only (the PR's fp16 rows are
candidate-only there), and its CSR kernel fails NVVM codegen for GQA ratios
below 8 on every cutlass-dsl 4.6-4.8 build tried; :func:`baseline_arm` routes
those rows to flashinfer's trtllm-gen block-sparse bridge (one single-token
request per query token) so every row keeps a baseline.
"""

from __future__ import annotations

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache

BLOCK_SIZE = 128
MINIMAX_STORAGE_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
# GQA ratios whose MiniMax CSR forward kernel compiles here: ratios 1, 2 and 4
# fail NVVM codegen (flat and paged alike, prefill and single-token shapes) on
# cutlass-dsl 4.6.0, 4.6.3, 4.7.0, 4.7.1 and 4.8.0.dev0, so those rows take
# the bridge.
MINIMAX_GQA_RATIOS = (8, 16)
MINIMAX_TOPK = (4, 8, 16, 32)  # sparse_atten_func's accepted top-k values
_WORKSPACE = {}


def baseline_arm(q, k, q2k_indices):
    """``"minimax"`` where MiniMax's forward runs, ``"trtllm_bridge"`` otherwise."""

    if (
        q.dtype in MINIMAX_STORAGE_DTYPES
        and k.dtype in MINIMAX_STORAGE_DTYPES
        and q.shape[1] // k.shape[1] in MINIMAX_GQA_RATIOS
        and int(q2k_indices.shape[-1]) in MINIMAX_TOPK
    ):
        return "minimax"
    return "trtllm_bridge"


# --- MiniMax arm ------------------------------------------------------------


def _cu_seqlens_k(cu_seqlens_k, seqused_k):
    """MiniMax wants cumulative KV lengths on paged rows too (its bench builds them)."""

    if cu_seqlens_k is not None:
        return cu_seqlens_k
    cu = torch.zeros(seqused_k.numel() + 1, dtype=torch.int32, device=seqused_k.device)
    cu[1:] = torch.cumsum(seqused_k.to(torch.int32), 0)
    return cu


def _minimax_args(
    q, k, v, q2k_indices, cu_seqlens_q, cu_seqlens_k, page_table, seqused_k, softmax_scale
):
    """Build the CSR reverse index and the forward schedule, as the PR's bench does outside timing."""

    import fmha_sm100

    cu_seqlens_k = _cu_seqlens_k(cu_seqlens_k, seqused_k)
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    kv_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    max_seqlen_q = int(q_lens.max())
    max_seqlen_k = int(kv_lens.max())
    total_rows = int(((kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE).sum())
    k2q_row_ptr, k2q_q_indices, schedule = fmha_sm100.build_k2q_csr(
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        BLOCK_SIZE,
        total_k=int(cu_seqlens_k[-1]),
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        total_rows=total_rows,
        qhead_per_kv=q.shape[1] // k.shape[1],
        return_schedule=True,
    )
    return (
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        int(q2k_indices.shape[-1]),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        seqused_k,
        schedule,
        float(softmax_scale),
    )


def _minimax_launch(
    q,
    k,
    v,
    k2q_row_ptr,
    k2q_q_indices,
    topk,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    page_table,
    seqused_k,
    schedule,
    softmax_scale,
):
    """The kernel call alone, as ``bench_blackwell_msa_sm100.py`` makes it."""

    import fmha_sm100

    return fmha_sm100.sparse_atten_func(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        blk_kv=BLOCK_SIZE,
        causal=True,
        softmax_scale=softmax_scale,
        return_softmax_lse=False,
        page_table=page_table,
        seqused_k=seqused_k,
        schedule=schedule,
    )


# --- trtllm-gen bridge (fp16 rows and GQA ratios below 8) -------------------


def _workspace(device):
    key = str(device)
    if key not in _WORKSPACE:
        _WORKSPACE[key] = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return _WORKSPACE[key]


def _paged_kv(k, v, cu_seqlens_k, page_table, seqused_k):
    """Return (k_pages, v_pages, page_table, kv_lens) in trtllm's HND page layout.

    MSA's paged layout already is trtllm's ``[num_pages, Hkv, 128, D]`` HND
    layout, so paged rows pass through; flat rows are scattered into 128-token
    pages here, once, outside the timed region."""

    if page_table is not None:
        return k, v, page_table.long(), seqused_k.long()
    device = k.device
    cu_seqlens_k = cu_seqlens_k.long()
    kv_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    pages_per_seq = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    page_base = torch.cumsum(pages_per_seq, 0) - pages_per_seq
    total_pages = int(pages_per_seq.sum())
    seqs = torch.arange(kv_lens.numel(), device=device)
    seq_of_token = torch.repeat_interleave(seqs, kv_lens, output_size=k.shape[0])
    local = torch.arange(k.shape[0], device=device) - cu_seqlens_k[:-1][seq_of_token]
    page = page_base[seq_of_token] + local // BLOCK_SIZE
    slot = local % BLOCK_SIZE
    k_pages = k.new_zeros((total_pages, k.shape[1], BLOCK_SIZE, k.shape[2]))
    v_pages = torch.zeros_like(k_pages)
    k_pages[page, :, slot] = k
    v_pages[page, :, slot] = v
    seq_of_page = torch.repeat_interleave(seqs, pages_per_seq, output_size=total_pages)
    pages = torch.arange(total_pages, device=device)
    page_table = torch.zeros(
        (kv_lens.numel(), int(pages_per_seq.max())), dtype=torch.int64, device=device
    )
    page_table[seq_of_page, pages - page_base[seq_of_page]] = pages
    return k_pages, v_pages, page_table, kv_lens


def _sparse_tables(q2k_indices, cu_seqlens_q, page_table, kv_lens):
    """One trtllm request per query token: its selected pages and surviving tokens.

    ``block_tables[h, r, :n]`` are the physical pages of the ``n`` blocks
    selected for kv head ``h`` and query token ``r``; ``seq_lens[h, r]`` counts
    the tokens of those pages up to the token's own causal position, so only
    the last (diagonal) page can be partial."""

    total_q = q2k_indices.shape[1]
    device = q2k_indices.device
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
    seq_of_q = torch.repeat_interleave(
        torch.arange(q_lens.numel(), device=device), q_lens, output_size=total_q
    )
    local_q = torch.arange(total_q, device=device) - cu_seqlens_q[:-1].long()[seq_of_q]
    q_pos = kv_lens[seq_of_q] - q_lens[seq_of_q] + local_q
    valid = q2k_indices >= 0
    count = valid.sum(-1)
    physical = page_table[seq_of_q.view(1, -1, 1), q2k_indices.clamp(min=0).long()]
    block_tables = torch.where(valid, physical, 0).to(torch.int32).contiguous()
    last = q2k_indices.gather(-1, (count - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1).long()
    tail = torch.clamp(q_pos.view(1, -1) + 1 - last * BLOCK_SIZE, max=BLOCK_SIZE)
    seq_lens = ((count - 1) * BLOCK_SIZE + tail).clamp(min=0).to(torch.int32).contiguous()
    return block_tables, seq_lens


def _bridge_args(
    q, k, v, q2k_indices, cu_seqlens_q, cu_seqlens_k, page_table, seqused_k, softmax_scale
):
    """Turn MSA's inputs into trtllm-gen block-sparse decode arguments.

    trtllm-gen's block-sparse decode takes per-(kv head, request) page lists in
    place of MSA's per-(kv head, query token) block ids, so every query token
    becomes one single-token request. The conversion, the flat->paged KV copy
    and the ``max_seq_len`` sync happen here, outside the timed region."""

    k_pages, v_pages, page_table, kv_lens = _paged_kv(k, v, cu_seqlens_k, page_table, seqused_k)
    block_tables, seq_lens = _sparse_tables(q2k_indices, cu_seqlens_q, page_table, kv_lens)
    max_seq_len = int(seq_lens.max().item())
    return q, k_pages, v_pages, block_tables, seq_lens, max_seq_len, float(softmax_scale)


def _bridge_launch(q, k_pages, v_pages, block_tables, seq_lens, max_seq_len, softmax_scale):
    """The kernel call alone, on arguments from :func:`_bridge_args`."""

    return trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=(k_pages, v_pages),
        workspace_buffer=_workspace(q.device),
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=softmax_scale,
        bmm2_scale=1.0,
        kv_layout="HND",
        backend="trtllm-gen",
        enable_block_sparse_attention=True,
    )


# --- contract entry points ---------------------------------------------------


def _arm(q, k, q2k_indices):
    if baseline_arm(q, k, q2k_indices) == "minimax":
        return _minimax_args, _minimax_launch
    return _bridge_args, _bridge_launch


@torch.no_grad()
def prepare(*contract_args):
    """Capture the kernel call in a CUDA graph outside timing.

    The harness times the CUPTI span of one call. Both arms are a few launches
    (MiniMax's CSR forward plus its combine, or trtllm-gen's prep and attention
    kernels), and eager launching leaves a host-dependent gap between them
    inside that span (a few us on the prefill rows, 10-15 us on the decode
    task's rows); replaying a graph removes the gap, so the timed span is the
    kernels alone. PR #4355 itself times eager CUPTI spans
    (``use_cuda_graph=False``) on its own host, gaps included; the replay is a
    deliberate deviation for reproducibility on a shared host. The output is
    bitwise equal to the eager call (alphamoe precedent). The captured input
    buffers ride along in the prepared state: a graph holds raw pointers, and
    the harness drops its own clones once prepare returns."""

    make_args, launch = _arm(contract_args[0], contract_args[1], contract_args[3])
    args = make_args(*contract_args)
    launch(*args)  # JIT / extension load and workspace allocation
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = launch(*args)
    return graph, output, args


@torch.no_grad()
def run_prepared(graph, output, args):
    """Kernel-only dispatch: replay the graph captured by :func:`prepare`."""

    graph.replay()
    return output


@torch.no_grad()
def run(q, k, v, q2k_indices, cu_seqlens_q, cu_seqlens_k, page_table, seqused_k, softmax_scale):
    """Call the row's baseline kernel on MSA inputs (eager)."""

    make_args, launch = _arm(q, k, q2k_indices)
    return launch(
        *make_args(
            q, k, v, q2k_indices, cu_seqlens_q, cu_seqlens_k, page_table, seqused_k, softmax_scale
        )
    )
