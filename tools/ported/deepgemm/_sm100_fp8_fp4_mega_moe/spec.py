# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Configuration, layout, heuristics, and compile/launch plumbing for MegaMoE.

The kernel body lives in :mod:`.kernel`; test and benchmark data preparation in
:mod:`.data`.

Upstream sources: deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh, csrc/apis/mega.h,
csrc/jit_kernels/heuristics/mega_moe.h.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
DEEPGEMM_SYM_BUFFER_MAX_RANKS = 72
_CUDA_COMPILE_MODE_LOCK = threading.RLock()
_PREPARED_LIBRARY_ENV = {
    False: "TIRX_PREPARED_MEGAMOE_LIBRARY_NO_STATS",
    True: "TIRX_PREPARED_MEGAMOE_LIBRARY_STATS",
}


@dataclass(frozen=True)
class MegaMoeConfig:
    num_processes: int = 1
    num_max_tokens_per_rank: int = 128
    num_tokens: int = 96
    hidden: int = 1024
    intermediate_hidden: int = 512
    num_experts: int = 8
    num_topk: int = 2
    num_shared_experts: int = 0
    activation_clamp: float = 10.0
    fast_math: int = 1

    def validate(self) -> None:
        if self.num_processes <= 0:
            raise ValueError("num_processes must be positive")
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if self.num_tokens > self.num_max_tokens_per_rank:
            raise ValueError("num_tokens must not exceed num_max_tokens_per_rank")
        if self.hidden % 128 != 0 or self.intermediate_hidden % 128 != 0:
            raise ValueError("hidden and intermediate_hidden must be multiples of 128")
        if self.intermediate_hidden > 4096:
            raise ValueError(
                "intermediate_hidden must satisfy DeepGEMM L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N"
            )
        if self.num_experts % self.num_processes != 0:
            raise ValueError("num_experts must be divisible by num_processes")
        if self.num_topk <= 0 or self.num_topk > self.num_experts:
            raise ValueError("num_topk must be in [1, num_experts]")
        if self.num_shared_experts < 0:
            raise ValueError("num_shared_experts must be non-negative")
        if self.num_shared_experts > 0:
            # Mirrors the host assert in DeepGEMM `csrc/apis/mega.hpp` and the
            # combine-loop static assert in the kernel (one extra slot per token).
            if self.shared_intermediate_hidden % 128 != 0:
                raise ValueError(
                    "num_shared_experts * intermediate_hidden must be a multiple of 128"
                )
            if self.num_topk + 1 > 32:
                raise ValueError("num_topk + 1 must not exceed the 32-lane combine ballot width")

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_experts // self.num_processes

    @property
    def shared_intermediate_hidden(self) -> int:
        return self.intermediate_hidden * self.num_shared_experts

    @property
    def has_shared_experts(self) -> bool:
        return self.num_shared_experts > 0


@dataclass
class MegaMoeCase:
    config: MegaMoeConfig
    rank_idx: int
    num_ranks: int
    group: Any
    deep_gemm: Any
    symm_buffer: Any
    x_fp8: tuple[torch.Tensor, torch.Tensor]
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    transformed_l1_weights: tuple[torch.Tensor, torch.Tensor]
    transformed_l2_weights: tuple[torch.Tensor, torch.Tensor]
    workspace_layout: DeepGemmWorkspaceLayout
    symm_buffer_layout: DeepGemmSymmBufferLayout
    cumulative_local_expert_recv_stats: torch.Tensor | None = None
    # Shared-expert weights, already through `transform_weights_for_mega_moe`.
    # 2-D (no expert-group dim) and FP8 e4m3; `None` when num_shared_experts == 0.
    transformed_shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
    transformed_shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
    # Host-built shared L1 input SF plane in the UTCCP-transposed, BLOCK_M-dependent
    # layout; the kernel never writes it.
    shared_l1_acts_sf: torch.Tensor | None = None


@dataclass
class TirxMegaMoeLaunchContext:
    config: MegaMoeConfig
    rank_idx: int
    num_ranks: int
    symm_buffer: Any
    transformed_l1_weights: tuple[torch.Tensor, torch.Tensor]
    transformed_l2_weights: tuple[torch.Tensor, torch.Tensor]
    cumulative_local_expert_recv_stats: torch.Tensor | None
    workspace_layout: DeepGemmWorkspaceLayout
    symm_buffer_layout: DeepGemmSymmBufferLayout
    transformed_shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
    transformed_shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass(frozen=True)
class DeepGemmLaunchConfig:
    num_sms: int
    num_ctas_per_cluster: int
    block_m: int
    block_n: int
    block_k: int
    load_block_m: int
    load_block_n: int
    store_block_m: int
    num_dispatch_threads: int
    num_non_epilogue_threads: int
    num_epilogue_threads: int
    num_bytes_per_pull: int
    num_topk: int
    hidden: int
    intermediate_hidden: int

    @property
    def num_dispatch_warps(self) -> int:
        return self.num_dispatch_threads // 32

    @property
    def num_non_epilogue_warps(self) -> int:
        return self.num_non_epilogue_threads // 32

    @property
    def num_epilogue_warps(self) -> int:
        return self.num_epilogue_threads // 32

    @property
    def num_epilogue_wgs(self) -> int:
        return self.num_epilogue_warps // 4

    @property
    def wg_block_m(self) -> int:
        """Rows one epilogue warpgroup stores (`WG_BLOCK_M`)."""
        return self.block_m // self.num_epilogue_wgs

    @property
    def atom_m(self) -> int:
        """Rows per `stmatrix` atom in the epilogue store (`ATOM_M`)."""
        return 8

    @property
    def l1_out_block_n(self) -> int:
        """Linear-1 emits gate and up halves per block, so half of `BLOCK_N`."""
        return self.block_n // 2

    @property
    def sf_block_m(self) -> int:
        """Scale factors are UTCCP-copied in 128-row chunks."""
        return _align_up(self.block_m, 128)

    @property
    def num_total_warps(self) -> int:
        return self.num_dispatch_warps + self.num_non_epilogue_warps + self.num_epilogue_warps

    @property
    def num_threads(self) -> int:
        return self.num_total_warps * 32

    @property
    def num_warpgroups(self) -> int:
        return self.num_total_warps // 4

    @property
    def num_threads_per_cta(self) -> int:
        return self.num_threads

    @property
    def num_warpgroups_per_cta(self) -> int:
        return self.num_warpgroups

    @property
    def num_tokens_per_warp(self) -> int:
        return 32 // self.num_topk

    @property
    def num_activate_lanes(self) -> int:
        return self.num_tokens_per_warp * self.num_topk

    @property
    def load_a_warp_idx(self) -> int:
        return self.num_dispatch_warps

    @property
    def load_b_warp_idx(self) -> int:
        return self.num_dispatch_warps + 1

    @property
    def mma_issue_warp_idx(self) -> int:
        return self.num_dispatch_warps + 2

    @property
    def reserved_non_epilogue_warp_idx(self) -> int:
        return self.num_dispatch_warps + 3

    @property
    def epilogue_warp_start_idx(self) -> int:
        return self.num_dispatch_warps + self.num_non_epilogue_warps


@dataclass(frozen=True)
class DeepGemmWorkspaceLayout:
    num_ranks: int
    num_experts: int
    num_experts_per_rank: int
    num_max_tokens_per_rank: int
    num_topk: int
    block_m: int
    num_max_recv_tokens_per_expert: int
    num_ring_tokens: int
    num_ring_blocks: int
    num_sf_ring_tokens: int
    num_max_pool_tokens: int
    num_shared_l2_pool_blocks: int
    token_src_metadata_bytes: int
    barrier_offset: int
    l1_task_count_offset: int
    l2_task_count_offset: int
    shared_l1_task_count_offset: int
    shared_l2_task_count_offset: int
    expert_send_count_offset: int
    expert_recv_count_offset: int
    expert_recv_count_sum_offset: int
    l1_full_count_offset: int
    l1_empty_count_offset: int
    l2_full_count_offset: int
    l2_empty_count_offset: int
    shared_l2_full_count_offset: int
    src_token_topk_idx_offset: int
    token_src_metadata_offset: int
    total_bytes: int


@dataclass(frozen=True)
class DeepGemmSymmBufferLayout:
    workspace_bytes: int
    input_token_offset: int
    input_sf_offset: int
    input_topk_idx_offset: int
    input_topk_weights_offset: int
    shared_l1_token_offset: int
    shared_l1_sf_offset: int
    shared_l2_token_offset: int
    shared_l2_sf_offset: int
    num_max_shared_sf_tokens: int
    l1_token_offset: int
    l1_sf_offset: int
    l1_topk_weights_offset: int
    l2_token_offset: int
    l2_sf_offset: int
    combine_token_offset: int
    total_bytes: int


@dataclass
class TirxMegaMoeInvocation:
    executable: Any
    y: torch.Tensor
    cumulative_local_expert_recv_stats: torch.Tensor
    symm_buffer_offsets: tuple[int, ...]
    tensor_maps: dict[str, _AlignedTensorMap]


@dataclass
class TirxMegaMoePrepared:
    context: TirxMegaMoeLaunchContext
    invocation: TirxMegaMoeInvocation


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


# Mirror of deep_gemm::layout constants from
# deep_gemm/include/deep_gemm/layout/mega_moe.cuh. The shared symm buffer is
# allocated by deep_gemm host code; the upstream layout is block_m-agnostic
# (uses LCM of the candidate set for token alignment and Min for the barrier-
# array sizing) so the same buffer is reusable across all candidate block_m's.
_K_CANDIDATE_BLOCK_M: tuple[int, ...] = (8, 16, 32, 64, 96, 128, 192)
_K_MIN_CANDIDATE_BLOCK_M = 8
_K_MAX_CANDIDATE_BLOCK_M = 192
_K_LCM_CANDIDATE_BLOCK_M = 384


def _get_num_max_pool_tokens(
    *, num_ranks: int, num_max_tokens_per_rank: int, num_topk: int, num_experts_per_rank: int
) -> int:
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return _align_up(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (_K_MAX_CANDIDATE_BLOCK_M - 1),
        _K_LCM_CANDIDATE_BLOCK_M,
    )


def _get_aligned_num_max_tokens_per_rank(config: MegaMoeConfig) -> int:
    return _align_up(config.num_max_tokens_per_rank, _K_LCM_CANDIDATE_BLOCK_M)


def _get_num_sf_ring_tokens(num_ring_tokens: int) -> int:
    return max(
        (num_ring_tokens // block_m) * _align_up(block_m, 128) for block_m in _K_CANDIDATE_BLOCK_M
    )


def _get_num_l1_warmup_waves(
    *, num_total_m_blocks: int, num_clusters: int, num_l1_n_clusters: int, num_l2_n_clusters: int
) -> int:
    """Mirror of `sched::get_num_l1_warmup_waves` in
    `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`: minimal L1 warmup waves
    so the interleaved L1/L2 schedule cannot deadlock on the ring buffer."""
    num_first_l2_wave_m_blocks = _ceil_div(num_clusters, num_l2_n_clusters)
    num_l1_warmup_clusters_for_first_l2_wave = _ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters
    )
    num_interleave_cluster_diff_per_m_block = (
        num_l1_n_clusters - num_l2_n_clusters if num_l1_n_clusters > num_l2_n_clusters else 0
    )
    num_warmup_waves_for_interleave_schedule = (
        _ceil_div(
            num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
            num_clusters,
        )
        + 1
    )
    return max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule)


def _get_num_max_live_pool_blocks(
    *, num_total_m_blocks: int, num_sms: int, hidden: int, intermediate_hidden: int
) -> int:
    """Mirror of `sched::get_num_max_live_pool_blocks` in
    `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`."""
    mega_moe_block_n = 128
    num_ctas_per_cluster = 2
    if (intermediate_hidden * 2) % (num_ctas_per_cluster * mega_moe_block_n) != 0:
        raise ValueError("MegaMoE requires (intermediate_hidden * 2) % 256 == 0")
    if hidden % (num_ctas_per_cluster * mega_moe_block_n) != 0:
        raise ValueError("MegaMoE requires hidden % 256 == 0")
    num_clusters = num_sms // num_ctas_per_cluster
    num_l1_n_clusters = intermediate_hidden * 2 // (num_ctas_per_cluster * mega_moe_block_n)
    num_l2_n_clusters = hidden // (num_ctas_per_cluster * mega_moe_block_n)
    num_l1_clusters = num_total_m_blocks * num_l1_n_clusters
    num_l1_waves = _ceil_div(num_l1_clusters, num_clusters)
    num_min_l1_warmup_waves = _get_num_l1_warmup_waves(
        num_total_m_blocks=num_total_m_blocks,
        num_clusters=num_clusters,
        num_l1_n_clusters=num_l1_n_clusters,
        num_l2_n_clusters=num_l2_n_clusters,
    )
    num_l1_warmup_waves = min(num_min_l1_warmup_waves, num_l1_waves)
    num_l1_warmup_clusters = min(num_l1_warmup_waves * num_clusters, num_l1_clusters)
    num_live_blocks_after_warmup = _ceil_div(num_l1_warmup_clusters, num_l1_n_clusters)
    frontier_growth = (
        _ceil_div(num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters), num_l2_n_clusters)
        if num_l2_n_clusters > num_l1_n_clusters
        else 0
    )
    wave_margin = _ceil_div(num_clusters, min(num_l1_n_clusters, num_l2_n_clusters))
    return min(num_total_m_blocks, num_live_blocks_after_warmup + frontier_growth + wave_margin)


def _get_num_ring_tokens_for_mega_moe(config: MegaMoeConfig) -> int:
    """Mirror of the ring sizing loop in `csrc/apis/mega.hpp`
    `get_symm_buffer_size_for_mega_moe`: the worst-case live pool blocks over
    all candidate block sizes bounds the ring."""
    num_sms = _get_num_sms_for_mega_moe()
    num_max_tokens_per_rank = _get_aligned_num_max_tokens_per_rank(config)
    num_active_topk = min(config.num_topk, config.num_experts_per_rank)
    num_max_routed_tokens = num_max_tokens_per_rank * config.num_processes * num_active_topk
    num_ring_tokens = 0
    for block_m in _K_CANDIDATE_BLOCK_M:
        num_pool_blocks = _ceil_div(num_max_routed_tokens, block_m) + config.num_experts_per_rank
        num_live_pool_blocks = _get_num_max_live_pool_blocks(
            num_total_m_blocks=num_pool_blocks,
            num_sms=num_sms,
            hidden=config.hidden,
            intermediate_hidden=config.intermediate_hidden,
        )
        num_ring_tokens = max(num_ring_tokens, num_live_pool_blocks * block_m)
    return _align_up(num_ring_tokens, _K_LCM_CANDIDATE_BLOCK_M)


def get_deepgemm_workspace_layout(config: MegaMoeConfig) -> DeepGemmWorkspaceLayout:
    launch = get_deepgemm_launch_config(config)
    num_ranks = config.num_processes
    num_experts = config.num_experts
    num_experts_per_rank = config.num_experts_per_rank
    aligned_num_max_tokens_per_rank = _get_aligned_num_max_tokens_per_rank(config)
    num_max_recv_tokens_per_expert = num_ranks * aligned_num_max_tokens_per_rank
    num_max_pool_tokens = _get_num_max_pool_tokens(
        num_ranks=num_ranks,
        num_max_tokens_per_rank=aligned_num_max_tokens_per_rank,
        num_topk=config.num_topk,
        num_experts_per_rank=num_experts_per_rank,
    )
    num_ring_tokens = _get_num_ring_tokens_for_mega_moe(config)
    num_ring_blocks = num_ring_tokens // _K_MIN_CANDIDATE_BLOCK_M
    num_sf_ring_tokens = _get_num_sf_ring_tokens(num_ring_tokens)
    num_shared_l2_pool_blocks = _ceil_div(aligned_num_max_tokens_per_rank, _K_MIN_CANDIDATE_BLOCK_M)

    # `Workspace::kNumBarrierSignalBytes` = 128: [0..15] grid sync counters,
    # [16..19] NVLink barrier counter, [20..27] NVLink barrier signals,
    # [28..31] L1 / [32..35] L2 / [36..39] shared-L1 / [40..43] shared-L2
    # schedule task counters, [44..127] padding isolating the hot atomics.
    barrier_offset = 0
    l1_task_count_offset = 28
    l2_task_count_offset = 32
    shared_l1_task_count_offset = 36
    shared_l2_task_count_offset = 40
    cursor = 128

    expert_send_count_offset = cursor
    cursor += num_experts * 8

    expert_recv_count_offset = cursor
    cursor += num_experts * 8

    expert_recv_count_sum_offset = cursor
    cursor += num_experts_per_rank * 8

    l1_full_count_offset = cursor
    cursor += num_ring_blocks * 4

    l1_empty_count_offset = cursor
    cursor += num_ring_blocks * 4

    l2_full_count_offset = cursor
    cursor += num_ring_blocks * 4

    l2_empty_count_offset = cursor
    cursor += num_ring_blocks * 4

    shared_l2_full_count_offset = cursor
    cursor += num_shared_l2_pool_blocks * 4

    src_token_topk_idx_offset = cursor
    cursor += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * 4

    token_src_metadata_offset = cursor
    token_src_metadata_bytes = num_max_pool_tokens * 12
    cursor += token_src_metadata_bytes

    total_bytes = _align_up(cursor, 16)
    return DeepGemmWorkspaceLayout(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_experts_per_rank=num_experts_per_rank,
        num_max_tokens_per_rank=aligned_num_max_tokens_per_rank,
        num_topk=config.num_topk,
        block_m=launch.block_m,
        num_max_recv_tokens_per_expert=num_max_recv_tokens_per_expert,
        num_ring_tokens=num_ring_tokens,
        num_ring_blocks=num_ring_blocks,
        num_sf_ring_tokens=num_sf_ring_tokens,
        num_max_pool_tokens=num_max_pool_tokens,
        num_shared_l2_pool_blocks=num_shared_l2_pool_blocks,
        token_src_metadata_bytes=token_src_metadata_bytes,
        barrier_offset=barrier_offset,
        l1_task_count_offset=l1_task_count_offset,
        l2_task_count_offset=l2_task_count_offset,
        shared_l1_task_count_offset=shared_l1_task_count_offset,
        shared_l2_task_count_offset=shared_l2_task_count_offset,
        expert_send_count_offset=expert_send_count_offset,
        expert_recv_count_offset=expert_recv_count_offset,
        expert_recv_count_sum_offset=expert_recv_count_sum_offset,
        l1_full_count_offset=l1_full_count_offset,
        l1_empty_count_offset=l1_empty_count_offset,
        l2_full_count_offset=l2_full_count_offset,
        l2_empty_count_offset=l2_empty_count_offset,
        shared_l2_full_count_offset=shared_l2_full_count_offset,
        src_token_topk_idx_offset=src_token_topk_idx_offset,
        token_src_metadata_offset=token_src_metadata_offset,
        total_bytes=total_bytes,
    )


def get_deepgemm_symm_buffer_layout(config: MegaMoeConfig) -> DeepGemmSymmBufferLayout:
    workspace = get_deepgemm_workspace_layout(config)
    aligned_num_max_tokens_per_rank = workspace.num_max_tokens_per_rank
    cursor = workspace.total_bytes

    input_token_offset = cursor
    cursor += aligned_num_max_tokens_per_rank * config.hidden

    input_sf_offset = cursor
    cursor += aligned_num_max_tokens_per_rank * (config.hidden // 32)

    input_topk_idx_offset = cursor
    cursor += aligned_num_max_tokens_per_rank * config.num_topk * 8

    input_topk_weights_offset = cursor
    cursor += aligned_num_max_tokens_per_rank * config.num_topk * 4

    # Shared-expert regions (empty at `num_shared_experts == 0`, so the routed
    # ring buffers keep their zero-shared offsets). The shared L1 activations
    # alias the input tokens in place; only the SF plane is a separate region,
    # and it is written by the host, never by the kernel.
    shared_intermediate_hidden = config.shared_intermediate_hidden
    num_max_shared_sf_tokens = (
        _ceil_div(aligned_num_max_tokens_per_rank, _K_MIN_CANDIDATE_BLOCK_M) * 128
        if config.has_shared_experts
        else 0
    )

    shared_l1_token_offset = input_token_offset

    shared_l1_sf_offset = cursor
    cursor += num_max_shared_sf_tokens * (config.hidden // 32)

    shared_l2_token_offset = cursor
    cursor += (
        aligned_num_max_tokens_per_rank * shared_intermediate_hidden
        if config.has_shared_experts
        else 0
    )

    shared_l2_sf_offset = cursor
    cursor += num_max_shared_sf_tokens * (shared_intermediate_hidden // 32)

    l1_token_offset = cursor
    cursor += workspace.num_ring_tokens * config.hidden

    l1_sf_offset = cursor
    cursor += workspace.num_sf_ring_tokens * (config.hidden // 32)

    l1_topk_weights_offset = cursor
    cursor += workspace.num_ring_tokens * 4

    l2_token_offset = cursor
    cursor += workspace.num_ring_tokens * config.intermediate_hidden

    l2_sf_offset = cursor
    cursor += workspace.num_sf_ring_tokens * (config.intermediate_hidden // 32)

    combine_token_offset = cursor
    # Shared experts add one combine slot per token, written locally (no NVLink).
    num_combine_slots = config.num_topk + (1 if config.has_shared_experts else 0)
    cursor += num_combine_slots * aligned_num_max_tokens_per_rank * config.hidden * 2

    return DeepGemmSymmBufferLayout(
        workspace_bytes=workspace.total_bytes,
        input_token_offset=input_token_offset,
        input_sf_offset=input_sf_offset,
        input_topk_idx_offset=input_topk_idx_offset,
        input_topk_weights_offset=input_topk_weights_offset,
        shared_l1_token_offset=shared_l1_token_offset,
        shared_l1_sf_offset=shared_l1_sf_offset,
        shared_l2_token_offset=shared_l2_token_offset,
        shared_l2_sf_offset=shared_l2_sf_offset,
        num_max_shared_sf_tokens=num_max_shared_sf_tokens,
        l1_token_offset=l1_token_offset,
        l1_sf_offset=l1_sf_offset,
        l1_topk_weights_offset=l1_topk_weights_offset,
        l2_token_offset=l2_token_offset,
        l2_sf_offset=l2_sf_offset,
        combine_token_offset=combine_token_offset,
        total_bytes=cursor,
    )


def _tensor_offset_bytes(base: torch.Tensor, view: torch.Tensor) -> int:
    return int(view.data_ptr()) - int(base.data_ptr())


def validate_runtime_symm_buffer_layout(
    *, symm_buffer: Any, layout: DeepGemmSymmBufferLayout, config: MegaMoeConfig
) -> None:
    workspace = get_deepgemm_workspace_layout(config)
    actual_offsets = {
        "input_token_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.x),
        "input_sf_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.x_sf),
        "input_topk_idx_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.topk_idx),
        "input_topk_weights_offset": _tensor_offset_bytes(
            symm_buffer.buffer, symm_buffer.topk_weights
        ),
        "l1_token_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.l1_acts),
        "l1_sf_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.l1_acts_sf),
        "l2_token_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.l2_acts),
        "l2_sf_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.l2_acts_sf),
    }
    expected_offsets = {
        "input_token_offset": layout.input_token_offset,
        "input_sf_offset": layout.input_sf_offset,
        "input_topk_idx_offset": layout.input_topk_idx_offset,
        "input_topk_weights_offset": layout.input_topk_weights_offset,
        "l1_token_offset": layout.l1_token_offset,
        "l1_sf_offset": layout.l1_sf_offset,
        "l2_token_offset": layout.l2_token_offset,
        "l2_sf_offset": layout.l2_sf_offset,
    }
    if config.has_shared_experts:
        actual_offsets.update(
            {
                "shared_l1_token_offset": _tensor_offset_bytes(
                    symm_buffer.buffer, symm_buffer.shared_l1_acts
                ),
                "shared_l1_sf_offset": _tensor_offset_bytes(
                    symm_buffer.buffer, symm_buffer.shared_l1_acts_sf
                ),
                "shared_l2_token_offset": _tensor_offset_bytes(
                    symm_buffer.buffer, symm_buffer.shared_l2_acts
                ),
                "shared_l2_sf_offset": _tensor_offset_bytes(
                    symm_buffer.buffer, symm_buffer.shared_l2_acts_sf
                ),
            }
        )
        expected_offsets.update(
            {
                "shared_l1_token_offset": layout.shared_l1_token_offset,
                "shared_l1_sf_offset": layout.shared_l1_sf_offset,
                "shared_l2_token_offset": layout.shared_l2_token_offset,
                "shared_l2_sf_offset": layout.shared_l2_sf_offset,
            }
        )
    for key, expected in expected_offsets.items():
        actual = actual_offsets[key]
        if actual != expected:
            raise ValueError(
                f"DeepGEMM symm buffer offset mismatch for {key}: expected {expected}, got {actual}"
            )

    expected_shapes = {
        "x": (workspace.num_max_tokens_per_rank, config.hidden),
        "x_sf": (workspace.num_max_tokens_per_rank, config.hidden // 128),
        "topk_idx": (workspace.num_max_tokens_per_rank, config.num_topk),
        "topk_weights": (workspace.num_max_tokens_per_rank, config.num_topk),
        "l1_acts": (workspace.num_ring_tokens, config.hidden),
        "l1_acts_sf": (workspace.num_sf_ring_tokens, config.hidden // 128),
        "l2_acts": (workspace.num_ring_tokens, config.intermediate_hidden),
        "l2_acts_sf": (workspace.num_sf_ring_tokens, config.intermediate_hidden // 128),
    }
    if config.has_shared_experts:
        num_shared_sf = layout.num_max_shared_sf_tokens
        expected_shapes.update(
            {
                "shared_l1_acts": (workspace.num_max_tokens_per_rank, config.hidden),
                "shared_l1_acts_sf": (num_shared_sf, config.hidden // 128),
                "shared_l2_acts": (
                    workspace.num_max_tokens_per_rank,
                    config.shared_intermediate_hidden,
                ),
                "shared_l2_acts_sf": (num_shared_sf, config.shared_intermediate_hidden // 128),
            }
        )
        # Both shared SF planes are MN-major with the shared-SF-token stride.
        for name in ("shared_l1_acts_sf", "shared_l2_acts_sf"):
            actual_stride = tuple(getattr(symm_buffer, name).stride())
            if actual_stride != (1, num_shared_sf):
                raise ValueError(
                    f"DeepGEMM symm buffer stride mismatch for {name}: "
                    f"expected {(1, num_shared_sf)}, got {actual_stride}"
                )
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(getattr(symm_buffer, name).shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"DeepGEMM symm buffer shape mismatch for {name}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    # Upstream 559d79f removed `SymmBuffer.num_ring_tokens`; the ring capacity is
    # implied by the `l1_acts` view shape checked above.
    actual_num_ring_tokens = int(symm_buffer.l1_acts.shape[0])
    if actual_num_ring_tokens != workspace.num_ring_tokens:
        raise ValueError(
            "DeepGEMM ring capacity mismatch: "
            f"expected {workspace.num_ring_tokens}, got {actual_num_ring_tokens}"
        )

    if int(symm_buffer.buffer.nbytes) != layout.total_bytes:
        raise ValueError(
            f"DeepGEMM symm buffer total size mismatch: "
            f"expected {layout.total_bytes}, got {int(symm_buffer.buffer.nbytes)}"
        )


def _get_block_config_for_mega_moe(
    *, num_ranks: int, num_experts: int, num_tokens: int, num_topk: int
) -> tuple[int, int, int, int, int]:
    """Pick ``(cluster, block_m, store_m, block_k, epilogue_wgs)`` by
    expected tokens-per-expert. Mirrors `get_block_config_for_mega_moe` in
    `csrc/jit_kernels/heuristics/mega_moe.hpp` so the schedule tracks upstream's
    per-batch-size tuning instead of always picking the prefill-sized 192."""
    expected = float(num_tokens) * num_ranks * num_topk / num_experts
    if expected <= 8.5:
        cfg = (2, 16, 8, 256, 2)
    elif expected <= 16.5:
        cfg = (2, 32, 16, 128, 2)
    elif expected <= 32.5:
        cfg = (2, 64, 32, 128, 1)
    elif expected <= 64.5:
        cfg = (2, 96, 16, 128, 2)
    elif expected <= 96.5:
        cfg = (2, 128, 32, 128, 2)
    else:
        cfg = (2, 192, 32, 128, 2)
    assert cfg[1] in _K_CANDIDATE_BLOCK_M
    return cfg


def _get_num_bytes_per_pull(hidden: int) -> int:
    num_bytes_per_pull = hidden
    while num_bytes_per_pull > 4096:
        if num_bytes_per_pull % 2 != 0:
            raise ValueError("MegaMoE dispatch pull bytes must remain divisible by 2")
        num_bytes_per_pull //= 2
    return num_bytes_per_pull


def _get_num_sms_for_mega_moe() -> int:
    override = os.environ.get("TIRX_DEEPGEMM_NUM_SMS_OVERRIDE")
    if override is not None:
        return int(override)
    from tirx_kernels.runner import hardware_num_sms

    return hardware_num_sms()


def get_deepgemm_launch_config(config: MegaMoeConfig) -> DeepGemmLaunchConfig:
    block_n = 128
    num_sms = _get_num_sms_for_mega_moe()
    if num_sms <= 1:
        raise ValueError("MegaMoE launch must satisfy DeepGEMM kNumSMs > 1")
    if num_sms % 2 != 0:
        raise ValueError("MegaMoE launch must satisfy DeepGEMM kNumSMs % 2 == 0")
    num_ctas_per_cluster_env = os.environ.get("TIRX_DEEPGEMM_NUM_CTAS_PER_CLUSTER_OVERRIDE")
    cluster_size, block_m, store_block_m, block_k, num_epilogue_wgs = (
        _get_block_config_for_mega_moe(
            num_ranks=config.num_processes,
            num_experts=config.num_experts,
            num_tokens=config.num_tokens,
            num_topk=config.num_topk,
        )
    )
    if num_ctas_per_cluster_env is not None and int(num_ctas_per_cluster_env) != cluster_size:
        raise ValueError(
            f"MegaMoE must use DeepGEMM-equivalent num_ctas_per_cluster={cluster_size}"
        )
    load_block_m = block_m // 2
    launch = DeepGemmLaunchConfig(
        num_sms=num_sms,
        num_ctas_per_cluster=cluster_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        load_block_m=load_block_m,
        load_block_n=block_n,
        store_block_m=store_block_m,
        num_dispatch_threads=128,
        num_non_epilogue_threads=128,
        num_epilogue_threads=num_epilogue_wgs * 128,
        num_bytes_per_pull=_get_num_bytes_per_pull(config.hidden),
        num_topk=config.num_topk,
        hidden=config.hidden,
        intermediate_hidden=config.intermediate_hidden,
    )
    if launch.num_epilogue_warps != num_epilogue_wgs * 4:
        raise ValueError("MegaMoE launch num_epilogue_warps must equal kNumEpilogueWarpgroups * 4")
    if launch.epilogue_warp_start_idx % 4 != 0 or launch.num_epilogue_warps % 4 != 0:
        raise ValueError("MegaMoE launch must satisfy DeepGEMM epilogue warpgroup alignment")
    if launch.block_m % num_epilogue_wgs != 0:
        raise ValueError("MegaMoE launch must satisfy BLOCK_M % kNumEpilogueWarpgroups == 0")
    if launch.wg_block_m % launch.store_block_m != 0:
        raise ValueError("MegaMoE launch must satisfy WG_BLOCK_M % STORE_BLOCK_M == 0")
    if launch.store_block_m % launch.atom_m != 0:
        raise ValueError("MegaMoE launch must satisfy STORE_BLOCK_M % ATOM_M == 0")
    # Upstream relaxed `WG_BLOCK_M % 32 == 0` to a runtime lane-bound guard in the
    # SF weight-cache load (see kernel body). Keep only the `atom_m | 32` part here.
    if 32 % launch.atom_m != 0:
        raise ValueError("MegaMoE launch must satisfy 32 % ATOM_M == 0")
    if launch.block_n != 128:
        raise ValueError("MegaMoE launch must satisfy BLOCK_N == 128")
    if launch.block_k % launch.block_n != 0:
        raise ValueError("MegaMoE launch must satisfy BLOCK_K % BLOCK_N == 0")
    if (config.intermediate_hidden * 2 // launch.block_n) % 2 != 0:
        raise ValueError("MegaMoE launch must satisfy kNumL1BlockNs % 2 == 0")
    if (config.hidden // launch.block_n) % 2 != 0:
        raise ValueError("MegaMoE launch must satisfy kNumL2BlockNs % 2 == 0")
    # Shared-expert shape asserts from `MegaMoEScheduler` (scheduler/mega_moe.cuh:164-167).
    # The other two are redundant: SHARED_L1_SHAPE_K == L1_SHAPE_K and
    # SHARED_L2_SHAPE_N == L2_SHAPE_N, both already covered above.
    shared_intermediate_hidden = config.shared_intermediate_hidden
    if (shared_intermediate_hidden * 2) % (launch.block_n * 2) != 0:
        raise ValueError("MegaMoE launch must satisfy SHARED_L1_SHAPE_N % (BLOCK_N * 2) == 0")
    if shared_intermediate_hidden % launch.block_k != 0:
        raise ValueError("MegaMoE launch must satisfy SHARED_L2_SHAPE_K % BLOCK_K == 0")
    return launch


def get_tirx_launch_param_tags() -> list[str]:
    return ["blockIdx.x", "clusterCtaIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_tma_aligned_size(x: int, element_size: int) -> int:
    return _align_up(x * element_size, 16) // element_size


def _tensor_map_swizzle_from_mode(mode: int, base: int = 0) -> int:
    if base != 0:
        if base == 32 and mode == 128:
            return 4
        raise ValueError(f"Unsupported tensor map swizzle base={base}, mode={mode}")
    if mode in (0, 16):
        return 0
    if mode == 32:
        return 1
    if mode == 64:
        return 2
    if mode == 128:
        return 3
    raise ValueError(f"Unsupported tensor map swizzle mode={mode}")


def _torch_dtype_to_tvm_dtype(t: torch.Tensor) -> str:
    if t.dtype == torch.int8:
        return "int8"
    if t.dtype == torch.uint8:
        return "uint8"
    if t.dtype == torch.int32:
        return "int32"
    if t.dtype == torch.uint32:
        return "uint32"
    if t.dtype == torch.float32:
        return "float32"
    if t.dtype == torch.bfloat16:
        return "bfloat16"
    if t.dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    raise TypeError(f"Unsupported tensor dtype for TMA descriptor: {t.dtype}")


class _AlignedTensorMap:
    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


_CUDA_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B = 14
_CUDA_TENSOR_MAP_INTERLEAVE_NONE = 0
_CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B = 3
_CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
_CUDA_DRIVER = None


def _get_cuda_driver() -> ctypes.CDLL:
    global _CUDA_DRIVER
    if _CUDA_DRIVER is None:
        libcuda_path = ctypes.util.find_library("cuda") or "libcuda.so.1"
        driver = ctypes.CDLL(libcuda_path)
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        result = driver.cuInit(0)
        if result != 0:
            raise RuntimeError(f"cuInit failed with CUresult={result}")
        driver.cuTensorMapEncodeTiled.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        driver.cuTensorMapEncodeTiled.restype = ctypes.c_int
        _CUDA_DRIVER = driver
    return _CUDA_DRIVER


def _encode_fp4_align16_tma_2d_desc(
    *,
    desc: _AlignedTensorMap,
    tensor: torch.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    gmem_outer_stride_bytes: int,
    swizzle: int,
) -> None:
    global_shape = (ctypes.c_uint64 * 2)(int(gmem_inner_dim), int(gmem_outer_dim))
    global_strides = (ctypes.c_uint64 * 1)(int(gmem_outer_stride_bytes))
    box_dim = (ctypes.c_uint32 * 2)(int(smem_inner_dim), int(smem_outer_dim))
    element_strides = (ctypes.c_uint32 * 2)(1, 1)
    result = _get_cuda_driver().cuTensorMapEncodeTiled(
        desc.ptr,
        _CUDA_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        ctypes.c_uint32(2),
        ctypes.c_void_p(int(tensor.data_ptr())),
        global_shape,
        global_strides,
        box_dim,
        element_strides,
        _CUDA_TENSOR_MAP_INTERLEAVE_NONE,
        int(swizzle),
        _CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B,
        _CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed for FP4 align16 with CUresult={result}")


def _encode_tma_2d_desc(
    *,
    encode_tensormap: Any,
    tensor: torch.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    gmem_outer_stride: int,
    swizzle_mode: int,
    swizzle_base: int = 0,
    tensor_dtype: Any | None = None,
) -> _AlignedTensorMap:
    elem_size = int(tensor.element_size())
    if swizzle_mode != 0:
        smem_inner_dim = swizzle_mode // elem_size
    desc = _AlignedTensorMap()
    swizzle = _tensor_map_swizzle_from_mode(swizzle_mode, swizzle_base)
    if tensor_dtype == "float4_e2m1fn":
        _encode_fp4_align16_tma_2d_desc(
            desc=desc,
            tensor=tensor,
            gmem_inner_dim=gmem_inner_dim,
            gmem_outer_dim=gmem_outer_dim,
            smem_inner_dim=smem_inner_dim,
            smem_outer_dim=smem_outer_dim,
            gmem_outer_stride_bytes=int(gmem_outer_stride * elem_size),
            swizzle=swizzle,
        )
    else:
        encode_tensormap(
            desc.ptr,
            _torch_dtype_to_tvm_dtype(tensor) if tensor_dtype is None else tensor_dtype,
            2,
            ctypes.c_void_p(int(tensor.data_ptr())),
            int(gmem_inner_dim),
            int(gmem_outer_dim),
            int(gmem_outer_stride * elem_size),
            int(smem_inner_dim),
            int(smem_outer_dim),
            1,
            1,
            0,
            swizzle,
            3,
            0,
        )
    return desc


def _encode_tma_sf_desc(
    *,
    encode_tensormap: Any,
    tensor: torch.Tensor,
    shape_mn: int,
    shape_k: int,
    block_mn: int,
    gran_k: int,
    num_groups: int,
    smem_outer_dim: int = 1,
) -> _AlignedTensorMap:
    aligned_shape_mn = _get_tma_aligned_size(shape_mn, int(tensor.element_size()))
    packed_shape_k = _ceil_div(shape_k, gran_k * (1 if tensor.dtype == torch.float32 else 4))
    return _encode_tma_2d_desc(
        encode_tensormap=encode_tensormap,
        tensor=tensor,
        gmem_inner_dim=aligned_shape_mn,
        gmem_outer_dim=packed_shape_k * num_groups,
        smem_inner_dim=block_mn,
        smem_outer_dim=smem_outer_dim,
        gmem_outer_stride=aligned_shape_mn,
        swizzle_mode=0,
    )


def _build_tirx_tensor_maps(
    *,
    case: MegaMoeCase,
    l1_acts: torch.Tensor,
    l2_acts: torch.Tensor,
    l1_weights: torch.Tensor,
    l1_weights_sf: torch.Tensor,
    l2_weights: torch.Tensor,
    l2_weights_sf: torch.Tensor,
    shared_l1_acts: torch.Tensor | None = None,
    shared_l2_acts: torch.Tensor | None = None,
    shared_l1_weights: torch.Tensor | None = None,
    shared_l1_weights_sf: torch.Tensor | None = None,
    shared_l2_weights: torch.Tensor | None = None,
    shared_l2_weights_sf: torch.Tensor | None = None,
) -> dict[str, _AlignedTensorMap]:
    import tvm

    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    launch = get_deepgemm_launch_config(case.config)
    workspace = case.workspace_layout
    gran_k = 32
    swizzle_acts_mode = 128
    swizzle_weights_mode = 128
    num_experts_per_rank = case.config.num_experts_per_rank
    sf_block_m = _align_up(launch.block_m, 128)

    tensor_maps = {
        "tensor_map_l1_acts": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l1_acts,
            gmem_inner_dim=case.config.hidden,
            gmem_outer_dim=workspace.num_ring_tokens,
            smem_inner_dim=launch.block_k,
            smem_outer_dim=launch.load_block_m,
            gmem_outer_stride=int(l1_acts.stride(-2)),
            swizzle_mode=swizzle_acts_mode,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_l1_acts_sf": _encode_tma_sf_desc(
            encode_tensormap=encode_tensormap,
            tensor=case.symm_buffer.l1_acts_sf,
            shape_mn=workspace.num_sf_ring_tokens,
            shape_k=case.config.hidden,
            block_mn=sf_block_m,
            gran_k=gran_k,
            num_groups=1,
            smem_outer_dim=launch.block_k // 128,
        ),
        "tensor_map_l1_weights": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l1_weights,
            gmem_inner_dim=case.config.hidden,
            gmem_outer_dim=num_experts_per_rank * case.config.intermediate_hidden * 2,
            smem_inner_dim=launch.block_k,
            smem_outer_dim=launch.load_block_n,
            gmem_outer_stride=int(l1_weights.stride(-2)),
            swizzle_mode=swizzle_weights_mode,
            tensor_dtype="float4_e2m1fn",
        ),
        "tensor_map_l1_weights_sf": _encode_tma_sf_desc(
            encode_tensormap=encode_tensormap,
            tensor=l1_weights_sf,
            shape_mn=case.config.intermediate_hidden * 2,
            shape_k=case.config.hidden,
            block_mn=launch.block_n,
            gran_k=gran_k,
            num_groups=num_experts_per_rank,
            smem_outer_dim=launch.block_k // 128,
        ),
        "tensor_map_l1_output": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l2_acts,
            gmem_inner_dim=case.config.intermediate_hidden,
            gmem_outer_dim=workspace.num_ring_tokens,
            smem_inner_dim=launch.block_n // 2,
            smem_outer_dim=launch.store_block_m,
            gmem_outer_stride=int(l2_acts.stride(-2)),
            swizzle_mode=swizzle_acts_mode // 2,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_l2_acts": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l2_acts,
            gmem_inner_dim=case.config.intermediate_hidden,
            gmem_outer_dim=workspace.num_ring_tokens,
            smem_inner_dim=launch.block_k,
            smem_outer_dim=launch.load_block_m,
            gmem_outer_stride=int(l2_acts.stride(-2)),
            swizzle_mode=swizzle_acts_mode,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_l2_acts_sf": _encode_tma_sf_desc(
            encode_tensormap=encode_tensormap,
            tensor=case.symm_buffer.l2_acts_sf,
            shape_mn=workspace.num_sf_ring_tokens,
            shape_k=case.config.intermediate_hidden,
            block_mn=sf_block_m,
            gran_k=gran_k,
            num_groups=1,
            smem_outer_dim=launch.block_k // 128,
        ),
        "tensor_map_l2_weights": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l2_weights,
            gmem_inner_dim=case.config.intermediate_hidden,
            gmem_outer_dim=num_experts_per_rank * case.config.hidden,
            smem_inner_dim=launch.block_k,
            smem_outer_dim=launch.load_block_n,
            gmem_outer_stride=int(l2_weights.stride(-2)),
            swizzle_mode=swizzle_weights_mode,
            tensor_dtype="float4_e2m1fn",
        ),
        "tensor_map_l2_weights_sf": _encode_tma_sf_desc(
            encode_tensormap=encode_tensormap,
            tensor=l2_weights_sf,
            shape_mn=case.config.hidden,
            shape_k=case.config.intermediate_hidden,
            block_mn=launch.block_n,
            gran_k=gran_k,
            num_groups=num_experts_per_rank,
            smem_outer_dim=launch.block_k // 128,
        ),
    }

    # Shared-expert descriptors. Mirrors the upstream launcher
    # (csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:216-269): the nine slots
    # always exist in the ABI, and at `num_shared_experts == 0` the host binds the
    # corresponding routed descriptor into each so the signature never changes.
    shared_names = (
        "tensor_map_shared_l1_acts",
        "tensor_map_shared_l1_acts_sf",
        "tensor_map_shared_l1_weights",
        "tensor_map_shared_l1_weights_sf",
        "tensor_map_shared_l1_output",
        "tensor_map_shared_l2_acts",
        "tensor_map_shared_l2_acts_sf",
        "tensor_map_shared_l2_weights",
        "tensor_map_shared_l2_weights_sf",
    )
    if not case.config.has_shared_experts:
        for name in shared_names:
            tensor_maps[name] = tensor_maps[name.replace("_shared", "", 1)]
        return tensor_maps

    shared_intermediate_hidden = case.config.shared_intermediate_hidden
    num_max_shared_sf_tokens = case.symm_buffer_layout.num_max_shared_sf_tokens
    num_max_tokens_per_rank = workspace.num_max_tokens_per_rank
    tensor_maps.update(
        {
            # The shared L1 activations alias the input tokens in place.
            "tensor_map_shared_l1_acts": _encode_tma_2d_desc(
                encode_tensormap=encode_tensormap,
                tensor=shared_l1_acts,
                gmem_inner_dim=case.config.hidden,
                gmem_outer_dim=num_max_tokens_per_rank,
                smem_inner_dim=launch.block_k,
                smem_outer_dim=launch.load_block_m,
                gmem_outer_stride=int(shared_l1_acts.stride(-2)),
                swizzle_mode=swizzle_acts_mode,
                tensor_dtype="float8_e4m3fn",
            ),
            "tensor_map_shared_l1_acts_sf": _encode_tma_sf_desc(
                encode_tensormap=encode_tensormap,
                tensor=case.symm_buffer.shared_l1_acts_sf,
                shape_mn=num_max_shared_sf_tokens,
                shape_k=case.config.hidden,
                block_mn=sf_block_m,
                gran_k=gran_k,
                num_groups=1,
                smem_outer_dim=launch.block_k // 128,
            ),
            # Shared weights are plain FP8 e4m3, not packed FP4, and carry no
            # expert-group dimension.
            "tensor_map_shared_l1_weights": _encode_tma_2d_desc(
                encode_tensormap=encode_tensormap,
                tensor=shared_l1_weights,
                gmem_inner_dim=case.config.hidden,
                gmem_outer_dim=shared_intermediate_hidden * 2,
                smem_inner_dim=launch.block_k,
                smem_outer_dim=launch.load_block_n,
                gmem_outer_stride=int(shared_l1_weights.stride(-2)),
                swizzle_mode=swizzle_weights_mode,
                tensor_dtype="float8_e4m3fn",
            ),
            "tensor_map_shared_l1_weights_sf": _encode_tma_sf_desc(
                encode_tensormap=encode_tensormap,
                tensor=shared_l1_weights_sf,
                shape_mn=shared_intermediate_hidden * 2,
                shape_k=case.config.hidden,
                block_mn=launch.block_n,
                gran_k=gran_k,
                num_groups=1,
                smem_outer_dim=launch.block_k // 128,
            ),
            "tensor_map_shared_l1_output": _encode_tma_2d_desc(
                encode_tensormap=encode_tensormap,
                tensor=shared_l2_acts,
                gmem_inner_dim=shared_intermediate_hidden,
                gmem_outer_dim=num_max_tokens_per_rank,
                smem_inner_dim=launch.block_n // 2,
                smem_outer_dim=launch.store_block_m,
                gmem_outer_stride=int(shared_l2_acts.stride(-2)),
                swizzle_mode=swizzle_acts_mode // 2,
                tensor_dtype="float8_e4m3fn",
            ),
            "tensor_map_shared_l2_acts": _encode_tma_2d_desc(
                encode_tensormap=encode_tensormap,
                tensor=shared_l2_acts,
                gmem_inner_dim=shared_intermediate_hidden,
                gmem_outer_dim=num_max_tokens_per_rank,
                smem_inner_dim=launch.block_k,
                smem_outer_dim=launch.load_block_m,
                gmem_outer_stride=int(shared_l2_acts.stride(-2)),
                swizzle_mode=swizzle_acts_mode,
                tensor_dtype="float8_e4m3fn",
            ),
            "tensor_map_shared_l2_acts_sf": _encode_tma_sf_desc(
                encode_tensormap=encode_tensormap,
                tensor=case.symm_buffer.shared_l2_acts_sf,
                shape_mn=num_max_shared_sf_tokens,
                shape_k=shared_intermediate_hidden,
                block_mn=sf_block_m,
                gran_k=gran_k,
                num_groups=1,
                smem_outer_dim=launch.block_k // 128,
            ),
            "tensor_map_shared_l2_weights": _encode_tma_2d_desc(
                encode_tensormap=encode_tensormap,
                tensor=shared_l2_weights,
                gmem_inner_dim=shared_intermediate_hidden,
                gmem_outer_dim=case.config.hidden,
                smem_inner_dim=launch.block_k,
                smem_outer_dim=launch.load_block_n,
                gmem_outer_stride=int(shared_l2_weights.stride(-2)),
                swizzle_mode=swizzle_weights_mode,
                tensor_dtype="float8_e4m3fn",
            ),
            "tensor_map_shared_l2_weights_sf": _encode_tma_sf_desc(
                encode_tensormap=encode_tensormap,
                tensor=shared_l2_weights_sf,
                shape_mn=case.config.hidden,
                shape_k=shared_intermediate_hidden,
                block_mn=launch.block_n,
                gran_k=gran_k,
                num_groups=1,
                smem_outer_dim=launch.block_k // 128,
            ),
        }
    )
    return tensor_maps


def _view_symm_matrix(
    case: MegaMoeCase | TirxMegaMoeLaunchContext, offset: int, rows: int, cols: int
) -> torch.Tensor:
    return case.symm_buffer.buffer.narrow(0, offset, rows * cols).view(rows, cols)


def _get_mega_moe_cuda_compile_mode() -> str:
    mode = os.environ.get(
        "TIRX_MEGAMOE_CUDA_COMPILE_MODE", os.environ.get("TVM_CUDA_COMPILE_MODE", "nvcc")
    ).lower()
    if mode not in ("nvcc", "nvrtc"):
        raise ValueError(f"TIRX_MEGAMOE_CUDA_COMPILE_MODE must be 'nvcc' or 'nvrtc', got {mode!r}")
    return mode


@contextmanager
def _cuda_compile_mode(mode: str):
    """Select the synchronous TVM CUDA callback backend without leaking it."""
    with _CUDA_COMPILE_MODE_LOCK:
        previous = os.environ.get("TVM_CUDA_COMPILE_MODE")
        os.environ["TVM_CUDA_COMPILE_MODE"] = mode
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("TVM_CUDA_COMPILE_MODE", None)
            else:
                os.environ["TVM_CUDA_COMPILE_MODE"] = previous


@cache
def _compile_tirx_mega_moe_for_config(
    *,
    num_processes: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    num_topk: int,
    num_shared_experts: int,
    activation_clamp: float,
    fast_math: int,
    collect_stats: bool,
    cuda_compile_mode: str,
) -> Any:
    import tvm
    from tirx_kernels.runner import cuda_target

    # Deferred: `kernel` imports this module for its layout and launch config.
    from .kernel import get_kernel

    kernel = get_kernel(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        num_shared_experts=num_shared_experts,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        collect_stats=collect_stats,
    )
    # The block-scale tcgen05 MMA below uses ``scale_vec::1X``, which ptxas
    # rejects for a family-only target. The prepared compile profile therefore
    # supplies the exact architecture-specific target validated for the runtime
    # GPU (currently sm_100a, sm_103a, or sm_107a).
    target = cuda_target()
    mod = tvm.IRModule({"main": kernel})
    with _cuda_compile_mode(cuda_compile_mode):
        return tvm.compile(mod, target=target, tir_pipeline="tirx")


def _compile_tirx_mega_moe(case: MegaMoeCase | TirxMegaMoeLaunchContext) -> Any:
    config = case.config
    collect_stats = getattr(case, "cumulative_local_expert_recv_stats", None) is not None
    prepared_path = os.environ.get(_PREPARED_LIBRARY_ENV[collect_stats])
    if prepared_path is not None:
        return _load_prepared_mega_moe_executable(prepared_path)
    return _compile_tirx_mega_moe_for_config(
        num_processes=config.num_processes,
        num_max_tokens_per_rank=config.num_max_tokens_per_rank,
        num_tokens=config.num_tokens,
        hidden=config.hidden,
        intermediate_hidden=config.intermediate_hidden,
        num_experts=config.num_experts,
        num_topk=config.num_topk,
        num_shared_experts=config.num_shared_experts,
        activation_clamp=config.activation_clamp,
        fast_math=config.fast_math,
        collect_stats=collect_stats,
        cuda_compile_mode=_get_mega_moe_cuda_compile_mode(),
    )


@cache
def _load_prepared_mega_moe_executable(library_path: str) -> Any:
    import tvm

    return tvm.runtime.Executable(tvm.runtime.load_module(library_path))


def _require_mega_moe_tuple(name: str, value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a 2-tuple of tensors")
    first, second = value
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError(f"{name} must contain tensors")
    return first, second


def _make_symm_buffer_offsets(case: MegaMoeCase | TirxMegaMoeLaunchContext) -> tuple[int, ...]:
    buffer_ptrs = tuple(int(ptr) for ptr in case.symm_buffer.handle.buffer_ptrs)
    if len(buffer_ptrs) > DEEPGEMM_SYM_BUFFER_MAX_RANKS:
        raise ValueError(
            "TIRx MegaMoE supports at most "
            f"{DEEPGEMM_SYM_BUFFER_MAX_RANKS} symmetric-buffer ranks, got {len(buffer_ptrs)}"
        )
    local_base = int(case.symm_buffer.buffer.data_ptr())
    expected_local_base = buffer_ptrs[case.rank_idx]
    if local_base != expected_local_base:
        raise ValueError(
            "sym_buffer.buffer data_ptr does not match handle.buffer_ptrs[rank_idx]: "
            f"{local_base} != {expected_local_base}"
        )
    offsets = tuple(ptr - local_base for ptr in buffer_ptrs)
    return offsets + (0,) * (DEEPGEMM_SYM_BUFFER_MAX_RANKS - len(offsets))


def _make_tirx_mega_moe_launch_context(
    *,
    y: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer: Any,
    cumulative_local_expert_recv_stats: torch.Tensor | None,
    recipe: tuple[int, int, int],
    activation: str,
    activation_clamp: float | None,
    fast_math: bool,
    shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> TirxMegaMoeLaunchContext:
    if tuple(recipe) != (1, 1, 32):
        raise NotImplementedError("TIRx MegaMoE currently supports recipe=(1, 1, 32) only")
    if activation != "swiglu":
        raise NotImplementedError("TIRx MegaMoE currently supports activation='swiglu' only")
    if y.dtype != torch.bfloat16:
        raise TypeError(f"y must have dtype torch.bfloat16, got {y.dtype}")
    if not y.is_cuda:
        raise ValueError("y must be a CUDA tensor")
    if not y.is_contiguous():
        raise ValueError("y must be contiguous")
    if y.dim() != 2:
        raise ValueError(f"y must be 2D, got shape {tuple(y.shape)}")

    l1_weights = _require_mega_moe_tuple("l1_weights", l1_weights)
    l2_weights = _require_mega_moe_tuple("l2_weights", l2_weights)
    for tensor_name, tensor in (
        ("l1_weights[0]", l1_weights[0]),
        ("l1_weights[1]", l1_weights[1]),
        ("l2_weights[0]", l2_weights[0]),
        ("l2_weights[1]", l2_weights[1]),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{tensor_name} must be a CUDA tensor")
    # Mirrors `csrc/apis/mega.hpp:210-232`: the shared-expert count is derived
    # from the L2 weight width, and both shared pairs are 2-D FP8 e4m3.
    num_shared_experts = 0
    if (shared_l1_weights is None) != (shared_l2_weights is None):
        raise ValueError("shared_l1_weights and shared_l2_weights must be supplied together")
    if shared_l1_weights is not None:
        shared_l1_weights = _require_mega_moe_tuple("shared_l1_weights", shared_l1_weights)
        shared_l2_weights = _require_mega_moe_tuple("shared_l2_weights", shared_l2_weights)
        intermediate_hidden = int(sym_buffer.intermediate_hidden)
        shared_intermediate_hidden = int(shared_l2_weights[0].shape[1])
        if shared_intermediate_hidden % intermediate_hidden != 0:
            raise ValueError(
                "shared_l2_weights width must be a multiple of intermediate_hidden, got "
                f"{shared_intermediate_hidden} vs {intermediate_hidden}"
            )
        num_shared_experts = shared_intermediate_hidden // intermediate_hidden
        for tensor_name, tensor in (
            ("shared_l1_weights[0]", shared_l1_weights[0]),
            ("shared_l1_weights[1]", shared_l1_weights[1]),
            ("shared_l2_weights[0]", shared_l2_weights[0]),
            ("shared_l2_weights[1]", shared_l2_weights[1]),
        ):
            if not tensor.is_cuda:
                raise ValueError(f"{tensor_name} must be a CUDA tensor")
        if shared_l1_weights[0].dim() != 2 or shared_l2_weights[0].dim() != 2:
            raise ValueError("shared weights must be 2D (no expert-group dimension)")
        if shared_l1_weights[0].shape[0] != shared_intermediate_hidden * 2:
            raise ValueError(
                "shared_l1_weights must have 2 * shared_intermediate_hidden rows, got "
                f"{shared_l1_weights[0].shape[0]}"
            )
        if shared_l1_weights[0].shape[1] != int(y.shape[1]):
            raise ValueError("shared_l1_weights must have `hidden` columns")
        if shared_l2_weights[0].shape[0] != int(y.shape[1]):
            raise ValueError("shared_l2_weights must have `hidden` rows")
        for tensor_name, tensor in (
            ("shared_l1_weights[0]", shared_l1_weights[0]),
            ("shared_l2_weights[0]", shared_l2_weights[0]),
        ):
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{tensor_name} must have dtype torch.float8_e4m3fn")
            if not tensor.is_contiguous():
                raise ValueError(f"{tensor_name} must be contiguous")

    num_ranks = int(sym_buffer.group.size())
    rank_idx = int(sym_buffer.group.rank())
    config = MegaMoeConfig(
        num_processes=num_ranks,
        num_max_tokens_per_rank=int(sym_buffer.num_max_tokens_per_rank),
        num_tokens=int(y.shape[0]),
        hidden=int(y.shape[1]),
        intermediate_hidden=int(sym_buffer.intermediate_hidden),
        num_experts=int(sym_buffer.num_experts),
        num_topk=int(sym_buffer.num_topk),
        num_shared_experts=num_shared_experts,
        activation_clamp=math.inf if activation_clamp is None else float(activation_clamp),
        fast_math=int(bool(fast_math)),
    )
    config.validate()
    if int(sym_buffer.hidden) != config.hidden:
        raise ValueError(
            f"y hidden dimension {config.hidden} does not match sym_buffer.hidden {sym_buffer.hidden}"
        )
    if cumulative_local_expert_recv_stats is not None:
        if cumulative_local_expert_recv_stats.dtype != torch.int32:
            raise TypeError(
                "cumulative_local_expert_recv_stats must have dtype torch.int32, "
                f"got {cumulative_local_expert_recv_stats.dtype}"
            )
        if cumulative_local_expert_recv_stats.numel() != config.num_experts_per_rank:
            raise ValueError(
                "cumulative_local_expert_recv_stats must have one element per local expert, "
                f"expected {config.num_experts_per_rank}, got "
                f"{cumulative_local_expert_recv_stats.numel()}"
            )
        if not cumulative_local_expert_recv_stats.is_contiguous():
            raise ValueError("cumulative_local_expert_recv_stats must be contiguous")
        if not cumulative_local_expert_recv_stats.is_cuda:
            raise ValueError("cumulative_local_expert_recv_stats must be a CUDA tensor")
        if cumulative_local_expert_recv_stats.device != y.device:
            raise ValueError("cumulative_local_expert_recv_stats must be on the same device as y")

    workspace_layout = get_deepgemm_workspace_layout(config)
    symm_buffer_layout = get_deepgemm_symm_buffer_layout(config)
    validate_runtime_symm_buffer_layout(
        symm_buffer=sym_buffer, layout=symm_buffer_layout, config=config
    )
    return TirxMegaMoeLaunchContext(
        config=config,
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        symm_buffer=sym_buffer,
        transformed_l1_weights=l1_weights,
        transformed_l2_weights=l2_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        workspace_layout=workspace_layout,
        symm_buffer_layout=symm_buffer_layout,
        transformed_shared_l1_weights=shared_l1_weights,
        transformed_shared_l2_weights=shared_l2_weights,
    )


def _prepare_tirx_invocation(
    case: MegaMoeCase | TirxMegaMoeLaunchContext, y: torch.Tensor | None = None
) -> TirxMegaMoeInvocation:
    l1_weights = case.transformed_l1_weights[0]
    l1_weights_sf = case.transformed_l1_weights[1].permute(0, 2, 1)
    l2_weights = case.transformed_l2_weights[0]
    l2_weights_sf = case.transformed_l2_weights[1].permute(0, 2, 1)
    symm_layout = case.symm_buffer_layout
    l1_acts = _view_symm_matrix(
        case, symm_layout.l1_token_offset, case.workspace_layout.num_ring_tokens, case.config.hidden
    )
    l1_acts_sf = case.symm_buffer.l1_acts_sf.transpose(0, 1)
    l2_acts = _view_symm_matrix(
        case,
        symm_layout.l2_token_offset,
        case.workspace_layout.num_ring_tokens,
        case.config.intermediate_hidden,
    )
    l2_acts_sf = case.symm_buffer.l2_acts_sf.transpose(0, 1)
    shared_kwargs: dict[str, torch.Tensor] = {}
    if case.config.has_shared_experts:
        shared_l1_weights, shared_l1_weights_sf = case.transformed_shared_l1_weights
        shared_l2_weights, shared_l2_weights_sf = case.transformed_shared_l2_weights
        shared_kwargs = {
            # Shared L1 activations alias the input token buffer in place.
            "shared_l1_acts": _view_symm_matrix(
                case,
                symm_layout.input_token_offset,
                case.workspace_layout.num_max_tokens_per_rank,
                case.config.hidden,
            ),
            "shared_l2_acts": _view_symm_matrix(
                case,
                symm_layout.shared_l2_token_offset,
                case.workspace_layout.num_max_tokens_per_rank,
                case.config.shared_intermediate_hidden,
            ),
            "shared_l1_weights": shared_l1_weights,
            "shared_l1_weights_sf": shared_l1_weights_sf,
            "shared_l2_weights": shared_l2_weights,
            "shared_l2_weights_sf": shared_l2_weights_sf,
        }
    tensor_maps = _build_tirx_tensor_maps(
        case=case,
        l1_acts=l1_acts,
        l2_acts=l2_acts,
        l1_weights=l1_weights,
        l1_weights_sf=l1_weights_sf,
        l2_weights=l2_weights,
        l2_weights_sf=l2_weights_sf,
        **shared_kwargs,
    )
    if y is None:
        y = torch.empty(
            (case.config.num_tokens, case.config.hidden), dtype=torch.bfloat16, device="cuda"
        )
    cumulative_local_expert_recv_stats = getattr(case, "cumulative_local_expert_recv_stats", None)
    if cumulative_local_expert_recv_stats is None:
        # TVM buffer parameters require a tensor even when the no-stats kernel
        # specialization compiles all accesses away.
        cumulative_local_expert_recv_stats = torch.empty(
            case.config.num_experts_per_rank, dtype=torch.int32, device=y.device
        )
    return TirxMegaMoeInvocation(
        executable=_compile_tirx_mega_moe(case),
        y=y,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        symm_buffer_offsets=_make_symm_buffer_offsets(case),
        tensor_maps=tensor_maps,
    )


def _prepare_global_barrier(executable: Any) -> None:
    try:
        prepare_global_barrier = executable.mod.get_function("__tvm_prepare_global_barrier")
    except AttributeError:
        prepare_global_barrier = None
    if prepare_global_barrier is not None:
        prepare_global_barrier()


def _launch_tirx_mega_moe(
    case: MegaMoeCase | TirxMegaMoeLaunchContext, invocation: TirxMegaMoeInvocation
) -> None:
    tensor_maps = invocation.tensor_maps
    _prepare_global_barrier(invocation.executable)
    invocation.executable.mod(
        invocation.y.view(-1),
        invocation.cumulative_local_expert_recv_stats,
        case.symm_buffer.buffer,
        *invocation.symm_buffer_offsets,
        tensor_maps["tensor_map_l1_acts"].ptr,
        tensor_maps["tensor_map_l1_acts_sf"].ptr,
        tensor_maps["tensor_map_l1_weights"].ptr,
        tensor_maps["tensor_map_l1_weights_sf"].ptr,
        tensor_maps["tensor_map_l1_output"].ptr,
        tensor_maps["tensor_map_l2_acts"].ptr,
        tensor_maps["tensor_map_l2_acts_sf"].ptr,
        tensor_maps["tensor_map_l2_weights"].ptr,
        tensor_maps["tensor_map_l2_weights_sf"].ptr,
        tensor_maps["tensor_map_shared_l1_acts"].ptr,
        tensor_maps["tensor_map_shared_l1_acts_sf"].ptr,
        tensor_maps["tensor_map_shared_l1_weights"].ptr,
        tensor_maps["tensor_map_shared_l1_weights_sf"].ptr,
        tensor_maps["tensor_map_shared_l1_output"].ptr,
        tensor_maps["tensor_map_shared_l2_acts"].ptr,
        tensor_maps["tensor_map_shared_l2_acts_sf"].ptr,
        tensor_maps["tensor_map_shared_l2_weights"].ptr,
        tensor_maps["tensor_map_shared_l2_weights_sf"].ptr,
        case.config.num_tokens,
        case.rank_idx,
    )


def prepare_tirx_fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer: Any,
    shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    cumulative_local_expert_recv_stats: torch.Tensor | None = None,
    recipe: tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: float | None = None,
    fast_math: bool = True,
) -> TirxMegaMoePrepared:
    context = _make_tirx_mega_moe_launch_context(
        y=y,
        l1_weights=l1_weights,
        l2_weights=l2_weights,
        sym_buffer=sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    return TirxMegaMoePrepared(context=context, invocation=_prepare_tirx_invocation(context, y=y))


def launch_prepared_tirx_fp8_fp4_mega_moe(prepared: TirxMegaMoePrepared) -> None:
    _launch_tirx_mega_moe(prepared.context, prepared.invocation)


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer: Any,
    shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    cumulative_local_expert_recv_stats: torch.Tensor | None = None,
    recipe: tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: float | None = None,
    fast_math: bool = True,
) -> None:
    prepared = prepare_tirx_fp8_fp4_mega_moe(
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=shared_l1_weights,
        shared_l2_weights=shared_l2_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    launch_prepared_tirx_fp8_fp4_mega_moe(prepared)
