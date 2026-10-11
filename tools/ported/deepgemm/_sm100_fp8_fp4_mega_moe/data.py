# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Case construction, references, and the distributed bench harness for MegaMoE.

The kernel body lives in :mod:`.kernel`; configuration and launch plumbing in
:mod:`.spec`.

Upstream sources: deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh, csrc/apis/mega.h,
csrc/jit_kernels/heuristics/mega_moe.h.
"""

from __future__ import annotations

import inspect
import math
import os
import random
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any
from unittest import SkipTest

import torch
import torch.multiprocessing as mp

from .spec import (
    MegaMoeCase,
    MegaMoeConfig,
    _align_up,
    _launch_tirx_mega_moe,
    _prepare_tirx_invocation,
    fp8_fp4_mega_moe,
    get_deepgemm_symm_buffer_layout,
    get_deepgemm_workspace_layout,
    validate_runtime_symm_buffer_layout,
)

_DEEP_GEMM_MODULE_NAME = "deep_gemm"


def load_deep_gemm_mega() -> tuple[Any, str]:
    try:
        import deep_gemm as module
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM mega_moe runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc
    if not hasattr(module, "fp8_fp4_mega_moe"):
        raise SkipTest("DeepGEMM mega_moe runtime unavailable: missing fp8_fp4_mega_moe")
    return module, "installed"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _distributed_env(port: int):
    old_master_addr = os.environ.get("MASTER_ADDR")
    old_master_port = os.environ.get("MASTER_PORT")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        yield
    finally:
        if old_master_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = old_master_addr
        if old_master_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = old_master_port


def _cast_grouped_weights_to_fp4(
    deep_gemm: Any, bf16_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = bf16_weights.shape
    weights = []
    scales = []
    for group_idx in range(num_groups):
        weight, scale = deep_gemm.utils.per_token_cast_to_fp4(
            bf16_weights[group_idx], use_ue8m0=True, gran_k=32
        )
        weights.append(weight)
        scales.append(scale)
    packed_weights = torch.stack(weights, dim=0).contiguous()
    raw_scales = torch.stack(scales, dim=0).contiguous()
    transformed_scales = deep_gemm.transform_sf_into_required_layout(
        raw_scales, n, k, (1, 32), num_groups
    )
    return packed_weights, transformed_scales


def _cast_fp8_for_mega_moe(
    deep_gemm: Any, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of `_cast_fp8_for_mega_moe` (DeepGEMM tests/test_mega_moe.py:53-60).

    Returns ``(x_fp8, x_sf, x_sf_tma)``. The third entry is the TMA-strided SF
    view required by ``check_sf_layout(..., tma_stride_check=true)``; shared
    weights use ``[0::2]`` of this tuple, i.e. ``(x_fp8, x_sf_tma)``.
    """
    x_fp8, x_sf = deep_gemm.utils.per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    mn, packed_sf_k = x_sf.shape
    x_sf_tma = torch.empty_strided(
        (mn, packed_sf_k), (1, _align_up(mn, 4)), dtype=x_sf.dtype, device=x_sf.device
    )
    x_sf_tma.copy_(x_sf)
    return x_fp8, x_sf, x_sf_tma


def _to_shared_mega_moe_sf_layout(
    sf: torch.Tensor, block_m: int, num_max_sf_tokens: int
) -> torch.Tensor:
    """Port of `_to_shared_mega_moe_sf_layout` (tests/test_mega_moe.py:36-50).

    The shared L1 input SF plane is host-written in the UTCCP-transposed,
    BLOCK_M-dependent layout; the kernel never writes it. Padding rows stay
    ZERO: the plane is allocated at full size, zeroed, and only ``[0,
    num_tokens)`` is filled.
    """
    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = _align_up(block_m, 128)
    num_m_blocks = (num_tokens + block_m - 1) // block_m
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k), (1, num_max_sf_tokens), dtype=sf.dtype, device=sf.device
    )
    result.zero_()
    for block_idx in range(num_m_blocks):
        num_block_tokens = min(block_m, num_tokens - block_idx * block_m)
        for m_idx in range(num_block_tokens):
            transposed_m_idx = (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
            result[block_idx * aligned_block_m + transposed_m_idx].copy_(
                sf[block_idx * block_m + m_idx]
            )
    return result


def _copy_fp8_sf(dst: torch.Tensor, src: torch.Tensor, num_tokens: int) -> None:
    """Port of `_copy_fp8_sf` (tests/test_mega_moe.py:62-70).

    At this call site ``dst.shape == src.shape`` (both are the full shared SF
    plane), so the plain copy path is taken and the replicate-last-row branch is
    dead; it is kept only to mirror the reference helper exactly.
    """
    if num_tokens == 0:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    dst[:num_tokens].copy_(src)
    if num_tokens < dst.shape[0]:
        dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))


def create_case(
    deep_gemm: Any, config: MegaMoeConfig, group: Any, rank_idx: int, num_ranks: int
) -> MegaMoeCase:
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        config.num_experts,
        config.num_max_tokens_per_rank,
        config.num_topk,
        config.hidden,
        config.intermediate_hidden,
        config.num_shared_experts,
    )
    num_tokens = config.num_tokens
    num_experts_per_rank = config.num_experts // num_ranks

    x = torch.randn((num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda")
    l1_weights = torch.randn(
        (num_experts_per_rank, config.intermediate_hidden * 2, config.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2_weights = torch.randn(
        (num_experts_per_rank, config.hidden, config.intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    shared_intermediate_hidden = config.shared_intermediate_hidden
    if config.has_shared_experts:
        # One fused shared FFN of intermediate width S * I; no expert-group dim.
        shared_l1_weights_bf16 = torch.randn(
            (shared_intermediate_hidden * 2, config.hidden), dtype=torch.bfloat16, device="cuda"
        )
        shared_l2_weights_bf16 = torch.randn(
            (config.hidden, shared_intermediate_hidden), dtype=torch.bfloat16, device="cuda"
        )
    scores = torch.randn((num_tokens, config.num_experts), dtype=torch.float32, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, config.num_topk, dim=-1, largest=True, sorted=False)

    x_fp8_data, x_sf, _x_sf_tma = _cast_fp8_for_mega_moe(deep_gemm, x)
    x_fp8 = (x_fp8_data, x_sf)
    transformed_l1_input = _cast_grouped_weights_to_fp4(deep_gemm, l1_weights)
    transformed_l2_input = _cast_grouped_weights_to_fp4(deep_gemm, l2_weights)
    transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(
        transformed_l1_input, transformed_l2_input
    )
    transformed_shared_l1_weights = None
    transformed_shared_l2_weights = None
    shared_l1_acts_sf = None
    if config.has_shared_experts:
        # `[0::2]` selects `(fp8, tma-strided SF)`, matching the reference test.
        shared_l1_input = _cast_fp8_for_mega_moe(deep_gemm, shared_l1_weights_bf16)[0::2]
        shared_l2_input = _cast_fp8_for_mega_moe(deep_gemm, shared_l2_weights_bf16)[0::2]
        transformed_shared_l1_weights, transformed_shared_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(shared_l1_input, shared_l2_input)
        )
        # The shared L1 SF plane is host-written in a BLOCK_M-dependent layout.
        block_m = deep_gemm.get_block_m_for_mega_moe(
            num_ranks,
            config.num_experts,
            symm_buffer.num_max_tokens_per_rank,
            num_tokens,
            config.num_topk,
            "fp8xfp4",
        )
        shared_l1_acts_sf = _to_shared_mega_moe_sf_layout(
            x_sf, block_m, symm_buffer.shared_l1_acts_sf.shape[0]
        )
    workspace_layout = get_deepgemm_workspace_layout(config)
    symm_buffer_layout = get_deepgemm_symm_buffer_layout(config)
    validate_runtime_symm_buffer_layout(
        symm_buffer=symm_buffer, layout=symm_buffer_layout, config=config
    )
    return MegaMoeCase(
        config=config,
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        group=group,
        deep_gemm=deep_gemm,
        symm_buffer=symm_buffer,
        x_fp8=x_fp8,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        transformed_l1_weights=transformed_l1_weights,
        transformed_l2_weights=transformed_l2_weights,
        workspace_layout=workspace_layout,
        symm_buffer_layout=symm_buffer_layout,
        transformed_shared_l1_weights=transformed_shared_l1_weights,
        transformed_shared_l2_weights=transformed_shared_l2_weights,
        shared_l1_acts_sf=shared_l1_acts_sf,
    )


def _copy_inputs_into_symm_buffer(case: MegaMoeCase) -> None:
    num_tokens = case.config.num_tokens
    case.symm_buffer.x[:num_tokens].copy_(case.x_fp8[0])
    case.symm_buffer.x_sf[:num_tokens].copy_(case.x_fp8[1])
    if case.config.has_shared_experts:
        # Kernel-read, host-written: refreshed on every launch because debug mode
        # zeroes the whole symmetric buffer between calls.
        _copy_fp8_sf(case.symm_buffer.shared_l1_acts_sf, case.shared_l1_acts_sf, num_tokens)
    case.symm_buffer.topk_idx[:num_tokens].copy_(case.topk_idx)
    case.symm_buffer.topk_weights[:num_tokens].copy_(case.topk_weights)


def run_deepgemm_reference(
    case: MegaMoeCase, cumulative_local_expert_recv_stats: torch.Tensor | None = None
) -> torch.Tensor:
    _copy_inputs_into_symm_buffer(case)
    y = torch.empty(
        (case.config.num_tokens, case.config.hidden), dtype=torch.bfloat16, device="cuda"
    )
    case.deep_gemm.fp8_fp4_mega_moe(
        y,
        case.transformed_l1_weights,
        case.transformed_l2_weights,
        case.symm_buffer,
        shared_l1_weights=case.transformed_shared_l1_weights,
        shared_l2_weights=case.transformed_shared_l2_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        activation_clamp=case.config.activation_clamp,
        fast_math=bool(case.config.fast_math),
    )
    return y


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    abs_diff = (lhs.float() - rhs.float()).abs()
    return 0.0 if abs_diff.numel() == 0 else float(abs_diff.max().item())


def run_tirx_mega_moe(
    case: MegaMoeCase, cumulative_local_expert_recv_stats: torch.Tensor | None = None
) -> torch.Tensor:
    _copy_inputs_into_symm_buffer(case)
    y = torch.empty(
        (case.config.num_tokens, case.config.hidden), dtype=torch.bfloat16, device="cuda"
    )
    fp8_fp4_mega_moe(
        y,
        case.transformed_l1_weights,
        case.transformed_l2_weights,
        case.symm_buffer,
        shared_l1_weights=case.transformed_shared_l1_weights,
        shared_l2_weights=case.transformed_shared_l2_weights,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        activation_clamp=case.config.activation_clamp,
        fast_math=bool(case.config.fast_math),
    )
    return y


def _cleanup_distinct_cases(*cases: MegaMoeCase | None) -> None:
    destroyed: set[int] = set()
    for case in cases:
        if case is None:
            continue
        key = id(case.symm_buffer)
        if key in destroyed:
            continue
        case.symm_buffer.destroy()
        destroyed.add(key)


def _destroy_process_group() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _bench_megamoe_mode(
    funcs: dict[str, Any],
    kernel_names: dict[str, str],
    bench_kineto: Any,
    barrier: Any,
    between_impls: Any,
    *,
    rounds: int,
    cooldown_s: float,
) -> dict[str, Any]:
    """Run the exact benchmark protocol used by latest DeepGEMM MegaMoE."""
    if funcs.keys() != kernel_names.keys():
        raise ValueError("MegaMoE benchmark funcs and kernel_names must have identical keys")
    num_tests = int(inspect.signature(bench_kineto).parameters["num_tests"].default)
    round_samples: dict[str, list[float]] = {name: [] for name in funcs}
    round_orders: list[list[str]] = []
    items = list(funcs.items())
    for round_idx in range(rounds):
        if round_idx > 0:
            time.sleep(cooldown_s)
        round_items = items if round_idx % 2 == 0 else list(reversed(items))
        round_orders.append([name for name, _ in round_items])

        def run_pair() -> None:
            for impl_idx, (_, fn) in enumerate(round_items):
                if impl_idx > 0:
                    between_impls()
                fn()

        round_kernel_names = tuple(kernel_names[name] for name, _ in round_items)
        round_times = bench_kineto(run_pair, round_kernel_names, barrier=barrier)
        for (name, _), seconds in zip(round_items, round_times):
            seconds = float(seconds)
            if not math.isfinite(seconds) or seconds <= 0:
                raise RuntimeError(
                    f"DeepGEMM bench_kineto returned invalid time for {name}: {seconds}"
                )
            round_samples[name].append(seconds * 1e6)

    return {
        "impls": {name: sum(samples) / len(samples) for name, samples in round_samples.items()},
        "round_samples": round_samples,
        "errors": {},
        "timer": "megamoe",
        "benchmark_protocol": {
            "source": "deep_gemm.testing.bench_kineto",
            "kernel_names": kernel_names,
            "num_tests": num_tests,
            "flush_l2": True,
            "flush_l2_bytes": int(8e9),
            "gpu_sleep_cycles": int(2e7),
            "rank_barrier": "stream all_reduce",
            "rank_barrier_outside_kernel_timing": True,
            "paired_profile_session": True,
            "cold_setup_per_implementation": True,
            "rounds": rounds,
            "round_aggregate": "mean",
            "round_cooldown_s": cooldown_s,
            "round_orders": round_orders,
        },
    }


def _init_dist_on_assigned_device(
    local_rank: int, num_local_ranks: int, physical_device_index: int
) -> tuple[int, int, Any]:
    """Initialize the process-group shape without conflating rank and device."""
    import torch.distributed as dist

    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))
    world_size = num_nodes * num_local_ranks
    rank = node_rank * num_local_ranks + local_rank

    torch.cuda.set_device(physical_device_index)
    params: dict[str, Any] = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": world_size,
        "rank": rank,
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        params["device_id"] = torch.device("cuda", physical_device_index)
    dist.init_process_group(**params)
    torch.set_default_device("cuda")
    torch.cuda.set_device(physical_device_index)
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(world_size)))


def _run_worker(
    local_rank: int,
    physical_device_index: int,
    physical_device_uuid: str,
    cfg_dict: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    worker_kwargs = dict(cfg_dict)
    warmup = worker_kwargs.pop("warmup", None)
    repeat = worker_kwargs.pop("repeat", None)
    warmup = None if warmup is None else int(warmup)
    repeat = None if repeat is None else int(repeat)
    timer = worker_kwargs.pop("timer", None)
    timer = None if timer is None else str(timer)
    rounds = int(worker_kwargs.pop("rounds", 1))
    cooldown_s = float(worker_kwargs.pop("cooldown_s", 1.0))
    config = MegaMoeConfig(**worker_kwargs)
    config.validate()

    if config.num_processes > torch.cuda.device_count():
        raise SkipTest(
            f"Requested {config.num_processes} processes, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )

    from tirx_kernels.runner import (
        bind_cuda_assignment,
        external_references_enabled,
        validate_current_cuda_assignment,
    )

    bind_cuda_assignment((physical_device_index,), (physical_device_uuid,))
    deep_gemm, source = load_deep_gemm_mega()
    case = None
    dg_case = None
    tirx_case = None
    default_device_before = torch.get_default_device()
    cuda_device_before = (
        torch.cuda.current_device()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    try:
        if (
            hasattr(torch.distributed, "destroy_process_group")
            and torch.distributed.is_initialized()
        ):
            _destroy_process_group()
        rank_idx, num_ranks, group = _init_dist_on_assigned_device(
            local_rank, config.num_processes, physical_device_index
        )
        validate_current_cuda_assignment("after DeepGEMM distributed init")

        if mode == "test":
            case = create_case(deep_gemm, config, group, rank_idx, num_ranks)
            initial_stats = torch.arange(
                config.num_experts_per_rank, dtype=torch.int32, device="cuda"
            )
            deepgemm_stats = initial_stats.clone()
            tirx_stats = initial_stats.clone()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            y_ref = run_deepgemm_reference(case, deepgemm_stats)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            checksum = float(y_ref.float().sum().item())
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                y_tir = run_tirx_mega_moe(case, tirx_stats)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except NotImplementedError as exc:
                return {
                    "status": "SKIP",
                    "reason": str(exc),
                    "reference_source": source,
                    "reference_checksum": checksum,
                    "num_tokens": config.num_tokens,
                }
            deepgemm_max_abs_diff = _max_abs_diff(y_tir, y_ref)
            stats_max_abs_diff = int(
                (tirx_stats.to(torch.int64) - deepgemm_stats.to(torch.int64)).abs().max().item()
            )
            return {
                "status": "OK",
                "reference_source": source,
                "reference_checksum": checksum,
                "deepgemm_max_abs_diff": deepgemm_max_abs_diff,
                "stats_max_abs_diff": stats_max_abs_diff,
            }

        if mode == "bench":
            from tirx_kernels.runner import bench

            with_references = external_references_enabled()

            # bench()'s proton/event/cudagraph_proton timers are single-process:
            # each rank derives n_warmup/n_repeat from its own per-call estimate
            # with no cross-rank sync, so a cross-rank mega_moe kernel (in-kernel
            # collectives per launch) deadlocks when ranks run different iteration
            # counts. Multi-process bench must use the purpose-built megamoe
            # harness (DeepGEMM bench_kineto + barrier reset); default to it and
            # reject an explicit single-process timer.
            if config.num_processes > 1:
                if timer is None:
                    timer = "megamoe"
                elif timer != "megamoe":
                    raise ValueError(
                        "multi-process mega_moe bench requires timer='megamoe' (or omit "
                        f"--timer); {timer!r} is a single-process timer and would "
                        "deadlock the cross-rank kernel collectives"
                    )

            dg_case = None
            if with_references:
                dg_case = create_case(deep_gemm, config, group, rank_idx, num_ranks)
            tirx_case = create_case(deep_gemm, config, group, rank_idx, num_ranks)
            deepgemm_stats = None
            tirx_stats = None
            if timer == "megamoe":
                initial_stats = torch.zeros(
                    config.num_experts_per_rank, dtype=torch.int32, device="cuda"
                )
                if with_references:
                    deepgemm_stats = initial_stats.clone()
                tirx_stats = initial_stats.clone()
                tirx_case.cumulative_local_expert_recv_stats = tirx_stats
            if dg_case is not None:
                _copy_inputs_into_symm_buffer(dg_case)
            _copy_inputs_into_symm_buffer(tirx_case)
            y_deepgemm = None
            if with_references:
                y_deepgemm = torch.empty(
                    (config.num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda"
                )
            tirx_invocation = _prepare_tirx_invocation(tirx_case)

            def deepgemm_step() -> None:
                assert dg_case is not None and y_deepgemm is not None
                dg_case.deep_gemm.fp8_fp4_mega_moe(
                    y_deepgemm,
                    dg_case.transformed_l1_weights,
                    dg_case.transformed_l2_weights,
                    dg_case.symm_buffer,
                    shared_l1_weights=dg_case.transformed_shared_l1_weights,
                    shared_l2_weights=dg_case.transformed_shared_l2_weights,
                    cumulative_local_expert_recv_stats=deepgemm_stats,
                    activation_clamp=dg_case.config.activation_clamp,
                    fast_math=bool(dg_case.config.fast_math),
                )

            def tirx_step() -> None:
                _launch_tirx_mega_moe(tirx_case, tirx_invocation)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            if with_references:
                deepgemm_step()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            tirx_step()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            deepgemm_max_abs_diff = None
            if with_references:
                assert y_deepgemm is not None
                deepgemm_max_abs_diff = _max_abs_diff(tirx_invocation.y, y_deepgemm)
                if deepgemm_max_abs_diff != 0.0:
                    raise AssertionError(f"TIRx diff={deepgemm_max_abs_diff}")
                if timer == "megamoe" and not torch.equal(tirx_stats, deepgemm_stats):
                    raise AssertionError("TIRx cumulative expert stats differ from DeepGEMM")

            if timer == "megamoe":
                if warmup is not None or repeat is not None:
                    raise ValueError(
                        "timer='megamoe' uses DeepGEMM's fixed bench_kineto protocol; "
                        "do not pass warmup/repeat overrides"
                    )

                def deepgemm_megamoe_step() -> None:
                    nonlocal y_deepgemm
                    assert dg_case is not None
                    _copy_inputs_into_symm_buffer(dg_case)
                    y_deepgemm = torch.empty(
                        (config.num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda"
                    )
                    deepgemm_step()

                def tirx_megamoe_step() -> None:
                    _copy_inputs_into_symm_buffer(tirx_case)
                    tirx_invocation.y = torch.empty(
                        (config.num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda"
                    )
                    tirx_step()

                # DeepGEMM's tests/test_mega_moe.py barrier: an all-reduce queued
                # behind the sleep, so the ranks leave it together on the GPU.
                # `torch.distributed.barrier()` blocks the host instead, and each
                # rank's launch then starts its kernel on its own host's time.
                def stream_barrier() -> None:
                    torch.distributed.all_reduce(torch.empty(1, device="cuda"))

                def reset_between_implementations() -> None:
                    torch.empty(int(8e9 // 4), dtype=torch.int, device="cuda").zero_()
                    torch.cuda._sleep(int(2e7))
                    stream_barrier()

                from deep_gemm.testing import bench_kineto

                validate_current_cuda_assignment("before DeepGEMM MegaMoE timing", restore=True)
                funcs = {"tirx": tirx_megamoe_step}
                kernel_names = {"tirx": "mega_moe_kernel"}
                if with_references:
                    funcs["deepgemm"] = deepgemm_megamoe_step
                    kernel_names["deepgemm"] = "sm100_fp8_fp4_mega_moe_impl"
                bench_result = _bench_megamoe_mode(
                    funcs,
                    kernel_names,
                    bench_kineto,
                    stream_barrier,
                    reset_between_implementations,
                    rounds=rounds,
                    cooldown_s=cooldown_s,
                )
            else:
                validate_current_cuda_assignment("before TIRx MegaMoE timing", restore=True)
                bench_result = bench(
                    {"tirx": tirx_step},
                    warmup=warmup,
                    repeat=repeat,
                    timer=timer,
                    references={"deepgemm": lambda: deepgemm_step},
                    rounds=rounds,
                    cooldown_s=cooldown_s,
                )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            impls = bench_result["impls"]
            expected_impls = {"tirx", "deepgemm"} if with_references else {"tirx"}
            missing = expected_impls - set(impls)
            if missing:
                raise RuntimeError(f"Benchmark did not report timings for: {sorted(missing)}")
            result = {
                "status": "OK",
                "reference_source": source,
                "impls": {name: float(impls[name]) for name in sorted(expected_impls)},
                "round_samples": bench_result.get("round_samples", {}),
                "errors": bench_result["errors"],
                "timer": bench_result.get("timer"),
                "benchmark_protocol": bench_result.get("benchmark_protocol", {}),
            }
            if deepgemm_max_abs_diff is not None:
                result["deepgemm_max_abs_diff"] = deepgemm_max_abs_diff
            return result

        raise ValueError(f"Unsupported mode: {mode}")
    finally:
        try:
            _cleanup_distinct_cases(case, dg_case, tirx_case)
            _destroy_process_group()
        finally:
            torch.set_default_device(default_device_before)
            if cuda_device_before is not None:
                torch.cuda.set_device(cuda_device_before)


def _worker_entry(
    local_rank: int,
    device_indices: tuple[int, ...],
    device_uuids: tuple[str, ...],
    cfg_dict: dict[str, Any],
    mode: str,
    result_queue: mp.SimpleQueue | None,
) -> None:
    result = _run_worker(
        local_rank, int(device_indices[local_rank]), str(device_uuids[local_rank]), cfg_dict, mode
    )
    if result_queue is not None:
        result_queue.put((local_rank, result))


def _aggregate_rank_results(rank_results: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    rank_results = sorted(rank_results, key=lambda item: item[0])
    results = [result for _, result in rank_results]
    for result in results:
        if result["status"] == "SKIP":
            return result
    first = results[0]
    if "impls" in first:
        impl_names = sorted({name for result in results for name in result.get("impls", {})})
        round_samples: dict[str, list[float]] = {}
        for name in impl_names:
            per_rank_samples = []
            for result in results:
                samples = result.get("round_samples", {}).get(name)
                if samples is None:
                    raise RuntimeError(f"Rank result is missing round samples for {name!r}")
                per_rank_samples.append([float(value) for value in samples])
            sample_counts = {len(samples) for samples in per_rank_samples}
            if len(sample_counts) != 1:
                raise RuntimeError(
                    f"Ranks reported different round counts for {name!r}: {sorted(sample_counts)}"
                )
            num_rounds = sample_counts.pop()
            round_samples[name] = [
                max(samples[round_idx] for samples in per_rank_samples)
                for round_idx in range(num_rounds)
            ]
        impls = {
            name: sum(samples) / len(samples) for name, samples in round_samples.items() if samples
        }
        errors = {}
        for result in results:
            errors.update(result.get("errors", {}))
        return {
            **first,
            "impls": impls,
            "round_samples": round_samples,
            "errors": errors,
            "deepgemm_max_abs_diff": max(
                float(result.get("deepgemm_max_abs_diff", 0.0)) for result in results
            ),
            "rank_results": [
                {
                    "rank": rank,
                    "impls": result.get("impls", {}),
                    "round_samples": result.get("round_samples", {}),
                    "deepgemm_max_abs_diff": float(result.get("deepgemm_max_abs_diff", 0.0)),
                }
                for rank, result in rank_results
            ],
        }
    if "deepgemm_max_abs_diff" not in first:
        return first
    return {
        **first,
        "deepgemm_max_abs_diff": max(float(result["deepgemm_max_abs_diff"]) for result in results),
        "stats_max_abs_diff": max(int(result["stats_max_abs_diff"]) for result in results),
        "rank_results": [
            {
                "rank": rank,
                "deepgemm_max_abs_diff": float(result["deepgemm_max_abs_diff"]),
                "stats_max_abs_diff": int(result["stats_max_abs_diff"]),
                "reference_checksum": float(result["reference_checksum"]),
            }
            for rank, result in rank_results
        ],
    }


def _run_distributed(
    config: MegaMoeConfig,
    mode: str,
    *,
    device_indices: tuple[int, ...] | None = None,
    device_uuids: tuple[str, ...] | None = None,
    **kwargs,
) -> dict[str, Any]:
    cfg_dict = {**asdict(config), **kwargs}
    if device_indices is None:
        device_indices = tuple(range(config.num_processes))
    if device_uuids is None:
        from tirx_kernels.runner import physical_cuda_uuids

        device_uuids = physical_cuda_uuids(device_indices)
    if len(device_indices) != config.num_processes or len(device_uuids) != config.num_processes:
        raise ValueError(
            f"MegaMoE assignment must contain {config.num_processes} devices, got "
            f"indices={device_indices!r}, uuids={device_uuids!r}"
        )
    if config.num_processes > torch.cuda.device_count():
        raise SkipTest(
            f"Requested {config.num_processes} processes, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )
    if config.num_processes == 1:
        last_error = None
        for _ in range(32):
            port = _find_free_port()
            try:
                with _distributed_env(port):
                    return _run_worker(
                        0, int(device_indices[0]), str(device_uuids[0]), cfg_dict, mode
                    )
            except Exception as exc:
                message = str(exc)
                if "EADDRINUSE" not in message and "address already in use" not in message:
                    raise
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "Unable to allocate a free TCP port for single-process distributed init."
        )

    port = _find_free_port()
    with _distributed_env(port):
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        mp.spawn(
            _worker_entry,
            args=(tuple(device_indices), tuple(device_uuids), cfg_dict, mode, result_queue),
            nprocs=config.num_processes,
            join=True,
        )
        rank_results = [result_queue.get() for _ in range(config.num_processes)]
        return _aggregate_rank_results(rank_results)
