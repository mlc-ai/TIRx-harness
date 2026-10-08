# Copyright 2026. Derived portions copyright the SGLang contributors, Qwen Team,
# Tri Dao, Songlin Yang, Yu Zhang, and the vLLM contributors.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Qwen/Qwen3.8-27B text inference, including all model definitions.

Public API:
    model = Qwen38('/path/to/Qwen3.8-27B')
    cache = model.new_cache(batch_size=2)
    logits = model.forward_step([[11, 12], [13]], cache)  # [2, vocabulary]
    logits = model.forward_step([[14], [15]], cache)
    output_ids = model.generate([[11, 12], [13]])  # new tokens, including EOS

A flat input list represents one request; nested lists represent a batch.
forward_step consumes every supplied token, updates the supplied cache in place,
and returns next-token logits. It accepts both prefill and cached continuation.
generate returns a flat list for flat input, otherwise a list of lists. It uses
greedy decoding and stops each request independently at EOS.

Kernel source: SGLang afe90a8bc908002219993794404c7c95dd3ced1d (2026-09-08).
Matching SGLang settings: single GPU, bfloat16 weights and KV cache,
--bf16-gemm-backend torch
--attention-backend trtllm_mha --page-size 64 --linear-attn-backend triton
--linear-attn-prefill-backend flashinfer --mamba-ssm-dtype float32
--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
Use DecodeRunner for startup capture and repeated single-step execution.
--disable-radix-cache --disable-overlap-schedule, temperature=0.
The fused decode projection/conv is enabled, as in upstream. No quantization,
image input is used. The prepared benchmark also supports a four-token linear
chain target-verification forward with per-draft intermediate state output.

Dependencies: torch, triton, flashinfer-python==0.6.18,
safetensors, apache-tvm-ffi, and a CUDA 13 C++ toolchain. The tested target is
NVIDIA B200. tokenizers and jinja2 are only used by the optional text CLI.
No SGLang or transformers Python modules are imported. Extracted Triton kernels
and the exact upstream C++ headers are included below. The latter are compiled
in a local cache without a source checkout. FlashInfer may download its prebuilt
kernel binaries on first use unless they are already installed or cached.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
from pathlib import Path
from functools import lru_cache
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open

SGLANG_COMMIT = "afe90a8bc908002219993794404c7c95dd3ced1d"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PAD_SLOT_ID = -1
MAX_ROWS_PER_BLOCK = 4
_is_npu = False
_is_hip = False


@lru_cache
def is_arch_support_pdl():
    return torch.cuda.get_device_capability()[0] >= 9


# Upstream fused rotary code uses this flag at launch, not at import time.
_ENABLE_PDL = True
cdiv = triton.cdiv
next_power_of_2 = triton.next_power_of_2
device_context = torch.cuda.device
exp = tl.exp  # Upstream default FLA_USE_FAST_OPS=0.


def calc_rows_per_block(M, device):
    # Same upstream heuristic with CUDA graphs and batch-invariant mode disabled.
    return min(next_power_of_2(cdiv(M, 2 * _get_sm_count(device))), MAX_ROWS_PER_BLOCK)

# Extracted from python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py
@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    mrope_axis_map_ptr,
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_cos_t,
    stride_pos_axis,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    MROPE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= NUM_Q_HEADS
    local_head = tl.where(is_k, head - NUM_Q_HEADS, head)
    out_dtype = tl.float16 if FP16 else tl.bfloat16

    if is_k:
        in_base = k_ptr + token * stride_k_t + local_head * HEAD_DIM
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * stride_ko_t + local_head * HEAD_DIM
    else:
        if HAS_GATE:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * 2 * HEAD_DIM
        else:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * HEAD_DIM
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM

    # Full load -> RMSNorm variance
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < HEAD_DIM
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(var + EPS)
    x_norm = (x * inv_rms * (w + 1.0)).to(out_dtype).to(tl.float32)

    # Pass-through tail [rotary_dim, head_dim)
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= ROTARY_DIM)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    # Reload rotary portion from L1 -> re-norm -> RoPE
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < HALF_ROTARY
    xr1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    xr2 = tl.load(in_base + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    wr1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    wr2 = tl.load(w_ptr + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    xr1 = (xr1 * inv_rms * (wr1 + 1.0)).to(out_dtype).to(tl.float32)
    xr2 = (xr2 * inv_rms * (wr2 + 1.0)).to(out_dtype).to(tl.float32)

    if MROPE:
        axis = tl.load(mrope_axis_map_ptr + rot_offs, mask=rot_mask, other=0)
        pos = tl.load(
            positions_ptr + axis * stride_pos_axis + token, mask=rot_mask, other=0
        )
    else:
        pos = tl.load(positions_ptr + token)
    cache_off = pos.to(tl.int64) * stride_cos_t
    cos = tl.load(
        cos_sin_cache_ptr + cache_off + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    tl.store(out_base + rot_offs, (xr1 * cos - xr2 * sin), mask=rot_mask)
    tl.store(out_base + HALF_ROTARY + rot_offs, (xr2 * cos + xr1 * sin), mask=rot_mask)

    # Gate copy (Q heads only)
    if HAS_GATE and not is_k:
        gate_in = in_base + HEAD_DIM
        gate_out = gate_out_ptr + token * stride_gate_t + local_head * HEAD_DIM
        g = tl.load(gate_in + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out + head_offs, g, mask=head_mask)

    # PDL: signal dependent kernels (attention/allreduce) can start early.
    # Only available on NVIDIA Hopper+ (sm_90+); guarded for AMD/other backends.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def fused_qk_gemma_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    has_gate: bool = True,
    mrope_axis_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fused QK GemmaRMSNorm + NeoX RoPE + gate deinterleave.

    Args:
        q_gate: [T, num_q_heads * (1 + has_gate) * head_dim] — interleaved Q+Gate if has_gate
        k: [T, num_kv_heads * head_dim]
        q_weight, k_weight: [head_dim] — raw GemmaRMSNorm weights (kernel adds +1.0)
        cos_sin_cache: [max_seq_len, rotary_dim] — [cos..., sin...]
        positions: [T] token positions, or [3, T] mrope rows (temporal, height, width)
        mrope_axis_map: [rotary_dim // 2] — the axis owning each rotary lane, from
            MRotaryEmbedding
    """
    assert positions.dim() in (1, 2), f"want [T] or [3, T], got {positions.shape}"
    mrope = positions.dim() == 2
    assert mrope == (mrope_axis_map is not None), "mrope_axis_map needs [3, T]"
    if mrope:
        assert positions.shape[0] == 3 and positions.stride(1) == 1, (
            f"want [3, T] contiguous over T, got {positions.shape} "
            f"stride {positions.stride()}"
        )
        lanes = rotary_dim // 2
        assert mrope_axis_map.shape == (lanes,), f"want one axis per lane ({lanes})"
    T = q_gate.shape[0]
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim

    q_out = torch.empty(T, q_size, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(T, kv_size, dtype=k.dtype, device=k.device)
    gate_out = (
        torch.empty(T, num_q_heads, head_dim, dtype=q_gate.dtype, device=q_gate.device)
        if has_gate
        else q_out
    )

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)

    grid = (T, num_q_heads + num_kv_heads)
    _fused_qk_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        mrope_axis_map,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        positions.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY=half_rotary,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rot_half_block,
        EPS=eps,
        FP16=q_gate.dtype == torch.float16,
        HAS_PASS=rotary_dim < head_dim,
        HAS_GATE=has_gate,
        MROPE=mrope,
        ENABLE_PDL=_ENABLE_PDL,
    )

    return q_out, k_out, gate_out if has_gate else None



# Extracted from python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py
def qwen3_5_gdn_prefill_projection_views(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    """Return strided views accepted by the prefill GDN consumers."""
    tokens = mixed_qkvz.shape[0]
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = mixed_qkvz[:, :qkv_dim]
    z = mixed_qkvz[:, qkv_dim:].view(tokens, num_heads_v, head_v)
    b = mixed_ba[:, :num_heads_v]
    a = mixed_ba[:, num_heads_v : 2 * num_heads_v]
    return mixed_qkv, z, b, a


@triton.jit
def _fused_qkvzba_causal_conv1d_update_contiguous_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    conv_state,
    conv_weight,
    conv_bias,
    conv_state_indices,
    stride_qkvz_batch: tl.constexpr,
    stride_qkvz_dim: tl.constexpr,
    stride_ba_batch: tl.constexpr,
    stride_ba_dim: tl.constexpr,
    stride_state_batch: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_pos: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_state_indices: tl.constexpr,
    QKV_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    NUM_STATE_SLOTS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    qkv_mask = dim_idx < QKV_DIM

    x = tl.load(
        mixed_qkvz + batch_idx * stride_qkvz_batch + dim_idx * stride_qkvz_dim,
        mask=qkv_mask,
        other=0.0,
    )

    state_slot = tl.load(conv_state_indices + batch_idx * stride_state_indices).to(
        tl.int64
    )
    # Treat every out-of-range index as padding so stale replay metadata cannot
    # turn an indexed state update into an OOB access.
    valid_slot = (
        (state_slot != PAD_SLOT_ID) & (state_slot >= 0) & (state_slot < NUM_STATE_SLOTS)
    )
    state_base = (
        conv_state + state_slot * stride_state_batch + dim_idx * stride_state_dim
    )

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(conv_bias + dim_idx, mask=qkv_mask, other=0.0).to(tl.float32)

    # Match the deployed direct-Triton update exactly. Its effective decode
    # state length is width-1 even when the physical cache tensor is wider.
    for pos in tl.static_range(KERNEL_WIDTH - 1):
        state_value = tl.load(
            state_base + pos * stride_state_pos,
            mask=qkv_mask & valid_slot,
            other=0.0,
        )
        weight_value = tl.load(
            conv_weight + dim_idx * stride_weight_dim + pos * stride_weight_width,
            mask=qkv_mask,
            other=0.0,
        )
        # Do not force an FP32 multiply here. This expression deliberately
        # retains the operand types/order of causal_conv1d_triton.py.
        acc += state_value * weight_value

    last_weight = tl.load(
        conv_weight
        + dim_idx * stride_weight_dim
        + (KERNEL_WIDTH - 1) * stride_weight_width,
        mask=qkv_mask,
        other=0.0,
    )
    acc += x * last_weight
    if SILU_ACTIVATION:
        conv_out = acc / (1.0 + tl.exp(-acc))
    else:
        conv_out = acc

    # The legacy kernel leaves padded rows' input unchanged.
    conv_out = tl.where(valid_slot, conv_out, x)
    tl.store(
        mixed_qkv + batch_idx * QKV_DIM + dim_idx,
        conv_out,
        mask=qkv_mask,
    )

    # The direct-Triton wrapper sets effective state_len=width-1 for decode.
    for pos in tl.static_range(KERNEL_WIDTH - 2):
        next_value = tl.load(
            state_base + (pos + 1) * stride_state_pos,
            mask=qkv_mask & valid_slot,
            other=0.0,
        )
        tl.store(
            state_base + pos * stride_state_pos,
            next_value,
            mask=qkv_mask & valid_slot,
        )
    tl.store(
        state_base + (KERNEL_WIDTH - 2) * stride_state_pos,
        x,
        mask=qkv_mask & valid_slot,
    )

    # The first feature lanes also materialize the smaller downstream tensors.
    z_mask = dim_idx < V_DIM
    z_value = tl.load(
        mixed_qkvz
        + batch_idx * stride_qkvz_batch
        + (QKV_DIM + dim_idx) * stride_qkvz_dim,
        mask=z_mask,
        other=0.0,
    )
    tl.store(z + batch_idx * V_DIM + dim_idx, z_value, mask=z_mask)

    gate_mask = dim_idx < NUM_V_HEADS
    b_value = tl.load(
        mixed_ba + batch_idx * stride_ba_batch + dim_idx * stride_ba_dim,
        mask=gate_mask,
        other=0.0,
    )
    a_value = tl.load(
        mixed_ba
        + batch_idx * stride_ba_batch
        + (NUM_V_HEADS + dim_idx) * stride_ba_dim,
        mask=gate_mask,
        other=0.0,
    )
    tl.store(b + batch_idx * NUM_V_HEADS + dim_idx, b_value, mask=gate_mask)
    tl.store(a + batch_idx * NUM_V_HEADS + dim_idx, a_value, mask=gate_mask)


def can_use_fused_qkvzba_causal_conv1d_update_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state_indices: torch.Tensor,
    *,
    qkv_dim: int,
    v_dim: int,
    num_v_heads: int,
    activation: str | None,
) -> tuple[bool, str]:
    """Return an explicit eligibility decision for the decode fusion."""
    tensors = (mixed_qkvz, mixed_ba, conv_state, conv_weight, conv_state_indices)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False, "all inputs must be torch.Tensor instances"
    if not all(tensor.is_cuda for tensor in tensors):
        return False, "CUDA tensors are required"
    if mixed_qkvz.ndim != 2 or mixed_ba.ndim != 2:
        return False, "projection outputs must be rank-2"
    if conv_state.ndim != 3 or conv_weight.ndim != 2:
        return False, "Conv1D state/weight ranks must be 3/2"
    if conv_state_indices.ndim != 1:
        return False, "conv_state_indices must be rank-1"
    batch = mixed_qkvz.shape[0]
    if mixed_ba.shape[0] != batch or conv_state_indices.shape[0] != batch:
        return False, "batch dimensions must match"
    if qkv_dim <= 0 or v_dim <= 0 or num_v_heads <= 0:
        return False, "TP-local dimensions must be positive"
    if mixed_qkvz.shape[1] != qkv_dim + v_dim:
        return False, "qkvz layout is not contiguous [Q|K|V|Z]"
    if mixed_ba.shape[1] != 2 * num_v_heads:
        return False, "ba layout is not contiguous [B|A]"
    if conv_state.shape[1] != qkv_dim or conv_weight.shape[0] != qkv_dim:
        return False, "Conv1D feature dimension does not match packed QKV"
    width = conv_weight.shape[1]
    if width < 2 or width > 4:
        return False, "only Conv1D widths 2 through 4 are supported"
    if conv_state.shape[2] < width - 1:
        return False, "Conv1D state is shorter than width - 1"
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if mixed_qkvz.dtype not in supported_dtypes:
        return False, "QKVZ activation dtype must be FP16, BF16, or FP32"
    if conv_state.dtype != mixed_qkvz.dtype or conv_weight.dtype != mixed_qkvz.dtype:
        return False, "QKVZ, Conv1D state, and weight dtypes must match"
    if mixed_ba.dtype not in supported_dtypes:
        return False, "BA activation dtype must be FP16, BF16, or FP32"
    if conv_bias is not None:
        if (
            not isinstance(conv_bias, torch.Tensor)
            or not conv_bias.is_cuda
            or conv_bias.ndim != 1
            or conv_bias.shape[0] != qkv_dim
            or conv_bias.dtype != mixed_qkvz.dtype
        ):
            return False, "Conv1D bias contract is incompatible"
    if activation not in (None, "silu", "swish"):
        return False, "activation must be None, silu, or swish"
    if mixed_qkvz.stride(1) != 1 or mixed_ba.stride(1) != 1:
        return False, "projection feature dimensions must be contiguous"
    if conv_weight.stride(1) != 1:
        return False, "Conv1D weight width dimension must be contiguous"
    if conv_state_indices.dtype not in (torch.int32, torch.int64):
        return False, "conv_state_indices must be int32 or int64"
    return True, "eligible"


def fused_qkvzba_causal_conv1d_update_contiguous(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state_indices: torch.Tensor,
    *,
    qkv_dim: int,
    v_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    activation: str | None,
    pad_slot_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode-only fused Qwen3.5 projection unpack and Conv1D state update."""
    eligible, reason = can_use_fused_qkvzba_causal_conv1d_update_contiguous(
        mixed_qkvz,
        mixed_ba,
        conv_state,
        conv_weight,
        conv_bias,
        conv_state_indices,
        qkv_dim=qkv_dim,
        v_dim=v_dim,
        num_v_heads=num_v_heads,
        activation=activation,
    )
    if not eligible:
        raise ValueError(f"Ineligible fused GDN decode projection/Conv1D: {reason}")
    if v_dim != num_v_heads * head_v_dim:
        raise ValueError(
            "Ineligible fused GDN decode projection/Conv1D: "
            "v_dim must equal num_v_heads * head_v_dim"
        )

    batch = mixed_qkvz.shape[0]
    mixed_qkv = torch.empty(
        (batch, qkv_dim), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    z = torch.empty(
        (batch, num_v_heads, head_v_dim),
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        (batch, num_v_heads),
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)

    block_size = 256
    grid = (batch, triton.cdiv(qkv_dim, block_size))
    _fused_qkvzba_causal_conv1d_update_contiguous_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        conv_state,
        conv_weight,
        conv_bias,
        conv_state_indices,
        mixed_qkvz.stride(0),
        mixed_qkvz.stride(1),
        mixed_ba.stride(0),
        mixed_ba.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        conv_state_indices.stride(0),
        QKV_DIM=qkv_dim,
        V_DIM=v_dim,
        NUM_V_HEADS=num_v_heads,
        NUM_STATE_SLOTS=conv_state.shape[0],
        STATE_LEN=conv_state.shape[2],
        KERNEL_WIDTH=conv_weight.shape[1],
        HAS_BIAS=conv_bias is not None,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        PAD_SLOT_ID=pad_slot_id,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=2,
    )
    return mixed_qkv, z, b, a


@triton.jit
def fused_qkv_split_gdn_prefill_kernel(
    q,
    k,
    v,
    mixed_qkv,
    MIXED_QKV_STRIDE_T: tl.constexpr,
    MIXED_QKV_STRIDE_D: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    HEAD_V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_t = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    q_dim: tl.constexpr = NUM_Q_HEADS * HEAD_Q
    k_dim: tl.constexpr = NUM_K_HEADS * HEAD_K
    v_dim: tl.constexpr = NUM_V_HEADS * HEAD_V
    qk_dim: tl.constexpr = q_dim + k_dim
    qkv_dim: tl.constexpr = qk_dim + v_dim

    mask = offsets < qkv_dim
    values = tl.load(
        mixed_qkv + i_t * MIXED_QKV_STRIDE_T + offsets * MIXED_QKV_STRIDE_D,
        mask=mask,
    )

    q_mask = offsets < q_dim
    tl.store(q + i_t * q_dim + offsets, values, mask=q_mask)

    k_offsets = offsets - q_dim
    k_mask = (offsets >= q_dim) & (offsets < qk_dim)
    tl.store(k + i_t * k_dim + k_offsets, values, mask=k_mask)

    v_offsets = offsets - qk_dim
    v_mask = (offsets >= qk_dim) & (offsets < qkv_dim)
    tl.store(v + i_t * v_dim + v_offsets, values, mask=v_mask)


def fused_qkv_split_gdn_prefill(
    mixed_qkv: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
):
    """Split packed post-conv GDN QKV into contiguous FLA prefill tensors.

    `mixed_qkv` is laid out per token as `[all_q | all_k | all_v]`. The FLA
    chunk kernels consume separate contiguous `[1, T, H, D]` tensors, so this
    fused split replaces three independent `aten::copy_` kernels from the
    generic FLA input guard. `mixed_qkv` may be a strided `[T, qkv_dim]` view.
    """
    seq_len = mixed_qkv.shape[0]
    q = torch.empty(
        (1, seq_len, num_q_heads, head_q),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    k = torch.empty(
        (1, seq_len, num_k_heads, head_k),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    v = torch.empty(
        (1, seq_len, num_v_heads, head_v),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )

    qkv_dim = num_q_heads * head_q + num_k_heads * head_k + num_v_heads * head_v
    if _is_hip and seq_len == 0:
        return q, k, v
    fused_qkv_split_gdn_prefill_kernel[(seq_len,)](
        q,
        k,
        v,
        mixed_qkv,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_q,
        head_k,
        head_v,
        BLOCK_SIZE=triton.next_power_of_2(qkv_dim),
        num_warps=8,
        num_stages=3,
    )
    return q, k, v



# Extracted from python/sglang/kernels/ops/mamba/causal_conv1d_triton.py
@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    # base of the sequence
    x_base = (
        x_ptr
        + sequence_start_index.to(tl.int64) * stride_x_token
        + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states. Cast to x's dtype so col* keep a single
            # dtype across the whole kernel: when x is fp16 but the conv-state
            # cache is bf16 (e.g. MiniCPM-V GDN prefill), the chunk_offset==0
            # branch would otherwise produce bf16 cols while the chunk_offset>0
            # else branch (and the sliding-window reassignment) produce fp16,
            # which trips Triton's if/else phi type check on col0.
            x_elem_ty = x_ptr.dtype.element_ty
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0).to(x_elem_ty)
        else:
            # prior-tokens are zeros (same x dtype as every other col* source)
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + (
                    (sequence_start_index + idx_tokens_last).to(tl.int64)
                    * stride_x_token
                )[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset.to(tl.int64) * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token).to(tl.int64)
            * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    **kwargs,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    seq_lens_cpu: (batch) int32
        The sequence lengths of the sequences in the batch
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3

    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    out = torch.empty_like(x)

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (
            num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2]
        )
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        # assert stride_istate_dim == 1
    if out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        padded_batch = query_start_loc.size(0) - 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert conv_states is not None, (
                "ERROR: `has_initial_state` is used, which needs also `conv_states`"
            )
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    def grid(META):
        max_seq_len = max(seq_lens_cpu)
        return (
            len(seq_lens_cpu),  # batch_size
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # launch_cooperative_grid=True
        BLOCK_M=8,
        BLOCK_N=256,
        num_stages=2,
    )
    return out



# Extracted from python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py
@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
    blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    seq_len = 1
    stride_a = a.stride(0)
    stride_b = b.stride(0)
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        stride_a,
        stride_b,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output



# Extracted from python/sglang/kernels/ops/attention/fla/fused_recurrent.py
@triton.jit
def fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    dev = mixed_qkv.device
    if any(
        t.device != dev
        for t in (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)
    ):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements (got A_log.numel()={A_log.numel()}, dt_bias.numel()={dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(
            f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK})."
        )
    BV = min(triton.next_power_of_2(V), 32)
    num_stages = 3
    num_warps = 1

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    NV = triton.cdiv(V, BV)
    grid = (NV, B * HV)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state



# Extracted from python/sglang/kernels/ops/attention/fla/l2norm.py
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            T=T,
            D=D,
            BD=BD,
            BT=16,
            num_warps=8,
            num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    return y.view(x_shape_og)


@triton.jit(do_not_specialize=["T"])
def gdn_prefill_qkv_prepare_kernel(
    q,
    k,
    v,
    q_out,
    k_out,
    v_out,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    T,
    H_QK: tl.constexpr,
    H_V: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Materialize strided Q/K/V into token-major tensors in one launch."""
    token_block = tl.program_id(0)
    head_idx = tl.program_id(1)

    if head_idx < H_QK:
        # Match l2norm_fwd_kernel's block layout so the BF16 reduction tree is
        # unchanged for strided inputs.
        q_block = tl.make_block_ptr(
            q + head_idx * q_stride_h,
            (T, D),
            (q_stride_t, q_stride_d),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        k_block = tl.make_block_ptr(
            k + head_idx * k_stride_h,
            (T, D),
            (k_stride_t, k_stride_d),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        q_values = tl.load(q_block, boundary_check=(0, 1)).to(tl.float32)
        k_values = tl.load(k_block, boundary_check=(0, 1)).to(tl.float32)
        q_output_block = tl.make_block_ptr(
            q_out + head_idx * D,
            (T, D),
            (H_QK * D, 1),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        k_output_block = tl.make_block_ptr(
            k_out + head_idx * D,
            (T, D),
            (H_QK * D, 1),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        tl.store(
            q_output_block,
            q_values.to(q_output_block.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            k_output_block,
            k_values.to(k_output_block.dtype.element_ty),
            boundary_check=(0, 1),
        )
    else:
        value_head = head_idx - H_QK
        v_block = tl.make_block_ptr(
            v + value_head * v_stride_h,
            (T, D),
            (v_stride_t, v_stride_d),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        v_values = tl.load(v_block, boundary_check=(0, 1))
        v_output_block = tl.make_block_ptr(
            v_out + value_head * D,
            (T, D),
            (H_V * D, 1),
            (token_block * BT, 0),
            (BT, BD),
            (1, 0),
        )
        tl.store(v_output_block, v_values, boundary_check=(0, 1))


def gdn_prefill_qkv_prepare_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q/K/V for FlashInfer, materializing only strided inputs."""
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError(
            "GDN fused prepare requires equal Q/K [T, Hqk, D] shapes and "
            "V [T, Hv, D], got "
            f"{q.shape=}, {k.shape=}, {v.shape=}"
        )
    if v.shape[0] != q.shape[0] or v.shape[2] != q.shape[2]:
        raise ValueError(
            "GDN fused prepare requires common token and head-dim axes, got "
            f"{q.shape=}, {v.shape=}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("GDN fused prepare requires Q/K/V on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("GDN fused prepare requires equal Q/K/V dtypes")

    T, H_QK, D = q.shape
    H_V = v.shape[1]
    if D > 512:
        raise ValueError(f"GDN fused prepare supports head dim <= 512, got {D}")
    if q.is_contiguous() and k.is_contiguous() and v.is_contiguous():
        return l2norm_fwd(q, eps), l2norm_fwd(k, eps), v

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    v_out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    BT = 16
    BD = triton.next_power_of_2(D)
    grid = (triton.cdiv(T, BT), H_QK + H_V)
    gdn_prefill_qkv_prepare_kernel[grid](
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        T=T,
        H_QK=H_QK,
        H_V=H_V,
        D=D,
        BT=BT,
        BD=BD,
        num_warps=4,
        num_stages=2,
    )
    return l2norm_fwd(q_out, eps), l2norm_fwd(k_out, eps), v_out



# Extracted from python/sglang/kernels/ops/attention/fla/layernorm_gated.py
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        if Z_IS_3D:
            z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        else:
            z_row_offsets = rows[:, None] * stride_z_row
        Z_base = Z + z_row_offsets + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    # Compute mean and variance per row (reduce along axis 1)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        # Store mean for each row
        mean_offsets = group * M + rows
        mean_mask = rows < M
        tl.store(Mean + mean_offsets, mean, mask=mean_mask)
        # Broadcast mean back to 2D for subtraction
        xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
        mean = 0.0  # Placeholder for RMS norm

    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Load weights and biases (broadcast across rows)
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        if Z_IS_3D:
            z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        else:
            z_row_offsets = rows[:, None] * stride_z_row
        Z_base = Z + z_row_offsets + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    # Write output
    tl.store(Y_base, y, mask=mask)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


@lru_cache
def _get_sm_count(device: torch.device) -> int:
    """Get and cache the SM count for a given device."""
    if device.type == "xpu":
        assert torch.xpu.is_available(), "XPU device is not available"
        return torch.xpu.get_device_properties(device).gpu_subslice_count
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    z_is_3d = z is not None and z.ndim == 3
    if z is not None:
        assert z.stride(-1) == 1
        if z_is_3d:
            assert z.shape[0] * z.shape[1] == M
            assert z.shape[2] == N
        else:
            assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    # Calculate rows per block based on SM count
    rows_per_block = calc_rows_per_block(M, x.device)
    # Update grid to use rows_per_block
    grid = (cdiv(M, rows_per_block), ngroups)
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    # Workaround for PyTorch <= 2.12: torch.xpu.device is not Dynamo-compatible
    # in that release — it creates a DynamoConfigPatchProxy that
    # SourcelessBuilder cannot wrap, causing a hard error under
    # torch.compile(fullgraph=True).  The device context is a functional no-op
    # for Triton kernel launches (device is determined by the tensor, not the
    # surrounding context), so we simply skip it when Dynamo is tracing.
    # PyTorch main already has the proper fix (XPUDeviceVariable registered in
    # torch/_dynamo/variables/ctx_manager.py analogous to CUDADeviceVariable).
    # TODO: remove this branch once we upgrade from PyTorch 2.12.
    device_ctx = (
        nullcontext()
        if x.device.type == "xpu" and torch.compiler.is_compiling()
        else device_context(x.device)
    )
    with device_ctx:
        _layer_norm_fwd_1pass_kernel[grid](
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None and not z_is_3d else 0,
            z.stride(0) if z_is_3d else 0,
            z.stride(1) if z_is_3d else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            ROWS_PER_BLOCK=rows_per_block,
            HAS_BIAS=bias is not None,
            HAS_Z=z is not None,
            Z_IS_3D=z_is_3d,
            Z_HEADS=z.shape[1] if z_is_3d else 1,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
            ACTIVATION=activation,
            **pdl_kwargs,
        )
    return out, mean, rstd


def rms_norm_gated(
    *,
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

    x_shape_og = x.shape
    # reshape input data into 2D tensor
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        if z.shape == x_shape_og:
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        else:
            assert len(x_shape_og) == 2
            assert z.ndim == 3
            assert z.shape[0] * z.shape[1] == x.shape[0]
            assert z.shape[2] == x.shape[1]
            assert z.stride(-1) == 1
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if _is_npu:
        assert activation == "swish", "NPU only supports swish activation"
    y, mean, rstd = _layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )
    return y.reshape(x_shape_og)



# Extracted from python/sglang/kernels/ops/elementwise/elementwise.py
@triton.jit
def _fused_sigmoid_mul_kernel(
    output_ptr,
    attn_output_ptr,
    gate_ptr,
    gate_stride_row,
    gate_stride_head,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fuse sigmoid(gate) * attn_output into a single kernel."""
    pid_row = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1)

    offsets = pid_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    d = offsets - head * HEAD_DIM

    attn_off = pid_row * hidden_dim + offsets
    attn = tl.load(attn_output_ptr + attn_off, mask=mask, other=0.0).to(tl.float32)

    gate_off = pid_row * gate_stride_row + head * gate_stride_head + d
    g = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

    result = attn * tl.sigmoid(g)
    tl.store(output_ptr + attn_off, result, mask=mask)


def fused_sigmoid_mul(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Fused sigmoid-mul for attention output gating.

    Equivalent to: attn_output * sigmoid(gate)

    The production Qwen3.5 path passes a 3D strided gate. A single hidden-block
    Triton kernel handles both that path and flat contiguous inputs.

    When inplace=True, writes result back to attn_output and returns it.

    Supports strided gate: if gate is 3D (num_tokens, num_heads, head_dim)
    and attn_output is 2D (num_tokens, hidden_dim), the kernel reads gate
    via explicit strides without requiring a contiguous copy.
    """
    if gate.ndim == 3 and attn_output.ndim == 2:
        # Strided gate path: gate is 3D (num_tokens, num_heads, head_dim)
        num_tokens, num_heads, head_dim = gate.shape
        hidden_dim = num_heads * head_dim
        assert attn_output.shape == (num_tokens, hidden_dim)
        gate_stride_row = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        # Flat path: both tensors have the same shape
        assert attn_output.shape == gate.shape, (
            "attn_output and gate must have the same shape"
        )
        hidden_dim = attn_output.shape[-1]
        num_tokens = attn_output.numel() // hidden_dim
        head_dim = hidden_dim
        gate_stride_row = hidden_dim
        gate_stride_head = hidden_dim

    out = attn_output if inplace else torch.empty_like(attn_output)
    block_h = 1024 if num_tokens < 1024 else 2048
    grid = (num_tokens, triton.cdiv(hidden_dim, block_h))
    _fused_sigmoid_mul_kernel[grid](
        out,
        attn_output,
        gate,
        gate_stride_row,
        gate_stride_head,
        hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out



# Exact upstream C++/CUDA headers, materialized only for compilation.
_EMBEDDED_HEADERS = {
    'elementwise/activation.cuh': r'''#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace sglang {

enum class ActivationKind : uint32_t {
  kSiLU,
  kGELU,
  kGELUTanh,
  kReLU2,
};

template <ActivationKind kAct>
SGL_DEVICE float apply_activation_f32(float x_f32) {
  if constexpr (kAct == ActivationKind::kSiLU) {
    return x_f32 / (1.0f + expf(-x_f32));
  } else if constexpr (kAct == ActivationKind::kGELU) {
    constexpr auto kSqrt1Over2 = 0.7071067811865475f;
    return x_f32 * (0.5f * (1.0f + erff(x_f32 * kSqrt1Over2)));
  } else if constexpr (kAct == ActivationKind::kGELUTanh) {
    constexpr auto kGeluTanhAlpha = 0.044715f;
    constexpr auto kGeluTanhBeta = 0.7978845608028654f;
    const float cdf = 0.5f * (1.0f + tanhf(kGeluTanhBeta * (x_f32 + kGeluTanhAlpha * x_f32 * x_f32 * x_f32)));
    return x_f32 * cdf;
  } else if constexpr (kAct == ActivationKind::kReLU2) {
    const float relu = x_f32 > 0.0f ? x_f32 : 0.0f;
    return relu * relu;
  } else {
    static_assert(host::dependent_false_v<decltype(kAct)>, "unsupported activation kind");
    return 0.0f;
  }
}

struct ActivationParams {
  const void* __restrict__ input;
  void* __restrict__ out;
  uint32_t hidden_dim;
  uint32_t num_tokens;
  // Optional MoE expert filtering: when expert_ids != nullptr, a token is
  // skipped if expert_ids[token_id / expert_step] == -1. expert_step is 1
  // for per-token routing and BLOCK_SIZE_M for sorted/TMA routing.
  const int32_t* __restrict__ expert_ids;
  uint32_t expert_step;
};

template <
    typename T,
    ActivationKind kAct,
    bool kUsePDL,
    bool kFilterExpert,
    bool kRoundActivation = false,
    bool kReuseInput = false>
__global__ void act_and_mul_kernel(const __grid_constant__ ActivationParams params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kMaxVecBytes / sizeof(T)>;
  const auto num_vecs = params.hidden_dim / kVecSize;  // per token
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto token_id = tid / num_vecs;

  if (token_id >= params.num_tokens) return;
  if constexpr (kFilterExpert) {
    if (params.expert_ids[token_id / params.expert_step] == -1) return;
  }
  const auto offset = tid % num_vecs;
  const auto input_offset = token_id * (num_vecs * 2) + offset;
  const auto output_offset = kReuseInput ? input_offset : tid;
  PDLWaitPrimary<kUsePDL>();
  const auto gate = device::load_as<vec_t>(params.input, input_offset);
  const auto up = device::load_as<vec_t>(params.input, input_offset + num_vecs);
  vec_t out;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const float gate_f32 = device::cast<fp32_t>(gate[i]);
    const float up_f32 = device::cast<fp32_t>(up[i]);
    if constexpr (kRoundActivation) {
      const T activated = device::cast<T>(apply_activation_f32<kAct>(gate_f32));
      out[i] = device::cast<T>(device::cast<fp32_t>(activated) * up_f32);
    } else {
      out[i] = device::cast<T>(apply_activation_f32<kAct>(gate_f32) * up_f32);
    }
  }
  if constexpr (kReuseInput) {
    device::store_as<vec_t>(const_cast<void*>(params.input), out, output_offset);
  } else {
    device::store_as<vec_t>(params.out, out, output_offset);
  }
  PDLTriggerSecondary<kUsePDL>();
}

struct UnaryActivationParams {
  const void* __restrict__ input;
  void* __restrict__ out;
  uint32_t num_vecs;
};

template <typename T, ActivationKind kAct, bool kUsePDL>
__global__ void act_kernel(const __grid_constant__ UnaryActivationParams params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kMaxVecBytes / sizeof(T)>;
  const auto vec_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_id >= params.num_vecs) return;
  PDLWaitPrimary<kUsePDL>();
  const auto in = device::load_as<vec_t>(params.input, vec_id);
  vec_t out;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    out[i] = device::cast<T>(apply_activation_f32<kAct>(device::cast<fp32_t>(in[i])));
  }
  device::store_as<vec_t>(params.out, out, vec_id);
  PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct ActivationKernel {
  static constexpr auto kVecSize = device::kMaxVecBytes / sizeof(T);
  static constexpr auto kBlockSize = 256u;

  using kernel_fn_t = decltype(&act_and_mul_kernel<T, ActivationKind::kSiLU, kUsePDL, false>);
  using unary_kernel_fn_t = decltype(&act_kernel<T, ActivationKind::kReLU2, kUsePDL>);

  template <ActivationKind kAct, bool kFilterExpert, bool kRoundActivation = false, bool kReuseInput = false>
  static constexpr kernel_fn_t activation_kernel =
      act_and_mul_kernel<T, kAct, kUsePDL, kFilterExpert, kRoundActivation, kReuseInput>;

  static_assert(device::kMaxVecBytes % sizeof(T) == 0, "unsupported data type");

  template <bool kFilterExpert, bool kRoundActivation = false, bool kReuseInput = false>
  static kernel_fn_t select_kernel(const std::string& type) {
    using namespace host;
    if (type == "silu") {
      return activation_kernel<ActivationKind::kSiLU, kFilterExpert, kRoundActivation, kReuseInput>;
    } else if (type == "gelu") {
      return activation_kernel<ActivationKind::kGELU, kFilterExpert, kRoundActivation, kReuseInput>;
    } else if (type == "gelu_tanh") {
      return activation_kernel<ActivationKind::kGELUTanh, kFilterExpert, kRoundActivation, kReuseInput>;
    } else {
      Panic("unsupported activation type: ", type);
    }
    return nullptr;
  }

  template <bool kRoundActivation = false, bool kReuseInput = false>
  static void launch(
      const tvm::ffi::TensorView& input,
      const tvm::ffi::TensorView& out,
      const std::string& type,
      const int32_t* expert_ids,
      uint32_t expert_step) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto D_in = SymbolicSize{"input_width"};
    auto D_out = SymbolicSize{"output_width"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    if constexpr (kReuseInput) {
      TensorMatcher({N, D_out}).with_strides({D_in, 1}).with_dtype<T>().with_device(device_).verify(out);
    } else {
      TensorMatcher({N, D_out}).with_dtype<T>().with_device(device_).verify(out);
    }
    TensorMatcher({N, D_in})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(input);

    const auto hidden_size = static_cast<uint32_t>(D_out.unwrap());
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto device = device_.unwrap();
    if (num_tokens == 0) return;
    RuntimeCheck(hidden_size * 2 == D_in.unwrap(), "invalid activation dimension");
    RuntimeCheck(hidden_size % kVecSize == 0, "hidden size must be divisible by vector size");
    if constexpr (kReuseInput) {
      RuntimeCheck(input.data_ptr() == out.data_ptr(), "in-place activation output must alias input");
    }
    // only get once to avoid overhead
    const auto num_total_items = num_tokens * (hidden_size / kVecSize);
    RuntimeCheck(num_total_items <= std::numeric_limits<uint32_t>::max(), "too many items for 32-bit indexing");
    const auto num_blocks = div_ceil(static_cast<uint32_t>(num_total_items), kBlockSize);
    const auto params = ActivationParams{
        .input = input.data_ptr(),
        .out = kReuseInput ? nullptr : out.data_ptr(),
        .hidden_dim = hidden_size,
        .num_tokens = num_tokens,
        .expert_ids = expert_ids,
        .expert_step = expert_step,
    };
    if (expert_ids != nullptr) {
      RuntimeCheck(expert_step > 0, "expert_step must be positive");
      const auto kernel = select_kernel<true, kRoundActivation, kReuseInput>(type);
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
    } else {
      const auto kernel = select_kernel<false, kRoundActivation, kReuseInput>(type);
      LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
    }
  }

  static void run_activation(const tvm::ffi::TensorView input, const tvm::ffi::TensorView out, std::string type) {
    launch(input, out, type, /*expert_ids=*/nullptr, /*expert_step=*/1);
  }

  static void
  run_activation_with_rounding(const tvm::ffi::TensorView input, const tvm::ffi::TensorView out, std::string type) {
    launch<true>(input, out, type, /*expert_ids=*/nullptr, /*expert_step=*/1);
  }

  static void run_activation_with_rounding_input_inplace(
      const tvm::ffi::TensorView input, const tvm::ffi::TensorView out, std::string type) {
    launch<true, true>(input, out, type, /*expert_ids=*/nullptr, /*expert_step=*/1);
  }

  static void run_activation_filtered(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView expert_ids,
      int64_t expert_step,
      std::string type) {
    using namespace host;
    RuntimeCheck(is_type<int32_t>(expert_ids.dtype()), "expert_ids must have dtype int32");
    RuntimeCheck(expert_step >= 1, "expert_step must be positive");
    launch(input, out, type, static_cast<const int32_t*>(expert_ids.data_ptr()), static_cast<uint32_t>(expert_step));
  }

  template <ActivationKind kAct>
  static constexpr auto unary_kernel = act_kernel<T, kAct, kUsePDL>;

  // Use the explicit non-const function-pointer type (mirrors select_kernel's
  // kernel_fn_t) rather than a trailing `decltype(unary_kernel<...>)` return,
  // which deduces a const-qualified pointer that clang-HIP (gfx942) refuses to
  // initialize from an lvalue / nullptr. nvcc accepts both; this form works for
  // CUDA and ROCm alike.
  static unary_kernel_fn_t select_unary_kernel(const std::string& type) {
    using namespace host;
    if (type == "relu2") {
      return ActivationKernel::template unary_kernel<ActivationKind::kReLU2>;
    } else {
      Panic("unsupported unary activation type: ", type);
    }
    return nullptr;
  }

  static void run_unary_activation(const tvm::ffi::TensorView input, const tvm::ffi::TensorView out, std::string type) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D})  //
        .with_dtype<T>()
        .with_device(device_)
        .verify(out)
        .verify(input);

    const auto num_elems = static_cast<int64_t>(N.unwrap()) * D.unwrap();
    const auto device = device_.unwrap();
    if (num_elems == 0) return;
    RuntimeCheck(num_elems % kVecSize == 0, "num elements must be divisible by vector size");
    const auto num_vecs = num_elems / kVecSize;
    RuntimeCheck(num_vecs <= std::numeric_limits<uint32_t>::max(), "too many items for 32-bit indexing");
    const auto num_blocks = div_ceil(static_cast<uint32_t>(num_vecs), kBlockSize);
    const auto params = UnaryActivationParams{
        .input = input.data_ptr(),
        .out = out.data_ptr(),
        .num_vecs = static_cast<uint32_t>(num_vecs),
    };
    const auto kernel = select_unary_kernel(type);
    LaunchKernel(num_blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
''',
    'elementwise/kvcache.cuh': r'''#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cassert>
#include <cstdint>

namespace sglang {

struct StoreKVCacheParams {
  const void* __restrict__ k;
  const void* __restrict__ v;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ indices;
  int64_t stride_k_bytes;
  int64_t stride_v_bytes;
  // Independent slot strides: head_dim != v_head_dim gives K and V different row widths.
  int64_t stride_k_cache_bytes;
  int64_t stride_v_cache_bytes;
  int64_t stride_indices;
  uint32_t batch_size;
  int64_t size_limit;
  int64_t reserved_skip_index;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

/**
 * \brief How a warp vectorizes one row of kElementBytes: the widest aligned
 * vector type it can use, and how many full loop iterations that takes.
 * Shared by the interleaved and single-row copies so the two cannot drift.
 * kElementBytes == 0 is a valid (empty) plan, so a zero-width tail can be
 * queried before being branched away.
 */
template <int64_t kElementBytes>
struct RowVecPlan {
  static constexpr int64_t kAlignment = (kElementBytes % (16 * device::kWarpThreads) == 0) ? 16
                                        : kElementBytes % (8 * device::kWarpThreads) == 0  ? 8
                                        : kElementBytes % (4 * device::kWarpThreads) == 0  ? 4
                                        : kElementBytes % 4 == 0                           ? 4
                                                                                           : 0;

  static_assert(kAlignment > 0, "Element size must be multiple of 4 bytes");

  using vec_t = device::AlignedStorage<uint32_t, kAlignment / 4>;
  static constexpr int64_t kLoopBytes = sizeof(vec_t) * device::kWarpThreads;
  static constexpr int64_t kLoopCount = kElementBytes / kLoopBytes;
  static constexpr int64_t kElementCount = kElementBytes / sizeof(vec_t);
  static constexpr bool kHasEpilogue = kLoopCount * kLoopBytes < kElementBytes;
};

/**
 * \brief Use a single warp to copy key and value data from source to destination.
 * Each thread in the warp copies a portion of the data in a coalesced manner.
 * Both loads are issued before either store: the two rows live in different
 * tensors, and the params' __restrict__ does not survive into the kernel body,
 * so the compiler cannot prove k_dst and v_src disjoint and will not sink the
 * V load past the K store on its own.
 * \tparam kElementBytes The size of each key/value element in bytes.
 * \param k_src Pointer to the source key data.
 * \param v_src Pointer to the source value data.
 * \param k_dst Pointer to the destination key data.
 * \param v_dst Pointer to the destination value data.
 */
template <int64_t kElementBytes>
SGL_DEVICE void copy_kv_warp(
    const void* __restrict__ k_src,
    const void* __restrict__ v_src,
    void* __restrict__ k_dst,
    void* __restrict__ v_dst) {
  using namespace device;
  using plan_t = RowVecPlan<kElementBytes>;
  using vec_t = typename plan_t::vec_t;
  constexpr auto kLoopCount = plan_t::kLoopCount;

  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll kLoopCount
  for (int64_t i = 0; i < kLoopCount; ++i) {
    const auto k = gmem.load(k_src, i);
    const auto v = gmem.load(v_src, i);
    gmem.store(k_dst, k, i);
    gmem.store(v_dst, v, i);
  }

  // handle the epilogue if any
  if constexpr (plan_t::kHasEpilogue) {
    if (gmem.in_bound(plan_t::kElementCount, kLoopCount)) {
      const auto k = gmem.load(k_src, kLoopCount);
      const auto v = gmem.load(v_src, kLoopCount);
      gmem.store(k_dst, k, kLoopCount);
      gmem.store(v_dst, v, kLoopCount);
    }
  }
}

/**
 * \brief Use a single warp to copy one row from source to destination.
 * Serves the width by which asymmetric K/V rows differ, which has no counterpart
 * row to interleave with.
 * \tparam kElementBytes The size of the row in bytes.
 * \param src Pointer to the source data.
 * \param dst Pointer to the destination data.
 */
template <int64_t kElementBytes>
SGL_DEVICE void copy_row_warp(const void* __restrict__ src, void* __restrict__ dst) {
  using namespace device;
  using plan_t = RowVecPlan<kElementBytes>;
  using vec_t = typename plan_t::vec_t;
  constexpr auto kLoopCount = plan_t::kLoopCount;

  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll kLoopCount
  for (int64_t i = 0; i < kLoopCount; ++i) {
    gmem.store(dst, gmem.load(src, i), i);
  }

  // handle the epilogue if any
  if constexpr (plan_t::kHasEpilogue) {
    if (gmem.in_bound(plan_t::kElementCount, kLoopCount)) {
      gmem.store(dst, gmem.load(src, kLoopCount), kLoopCount);
    }
  }
}

/**
 * \brief Copy a K row of kKBytes and a V row of kVBytes with one warp.
 * The overlapping prefix goes through the interleaved copy; only the width by
 * which the rows differ is left as a serial tail. Equal widths degenerate to a
 * single interleaved copy with no tail.
 */
template <int64_t kKBytes, int64_t kVBytes>
SGL_DEVICE void copy_kv_rows_warp(
    const void* __restrict__ k_src,
    const void* __restrict__ v_src,
    void* __restrict__ k_dst,
    void* __restrict__ v_dst) {
  using namespace device;
  constexpr auto kCommon = kKBytes < kVBytes ? kKBytes : kVBytes;
  constexpr auto kTail = (kKBytes < kVBytes ? kVBytes : kKBytes) - kCommon;

  // The interleaved copy indexes BOTH rows with kCommon's vector width, so that
  // width must divide each row's split offset -- the narrower row's alignment
  // does not imply the wider one's (e.g. 512 picks 16B, but 516 is not 16B
  // aligned). The tail's own width must likewise divide its kCommon start.
  // Whatever these gates admit is alignment-safe for the strides too, since a
  // stride is a whole multiple of its split size.
  constexpr auto kTailOrCommon = kTail == 0 ? kCommon : kTail;
  constexpr auto kCommonAlign = RowVecPlan<kCommon>::kAlignment;
  constexpr auto kTailAlign = RowVecPlan<kTailOrCommon>::kAlignment;
  constexpr bool kCanInterleave =
      kKBytes % kCommonAlign == 0 && kVBytes % kCommonAlign == 0 && kCommon % kTailAlign == 0;

  if constexpr (kCanInterleave) {
    copy_kv_warp<kCommon>(k_src, v_src, k_dst, v_dst);
    if constexpr (kTail > 0) {
      if constexpr (kKBytes > kVBytes) {
        copy_row_warp<kTail>(pointer::offset(k_src, kCommon), pointer::offset(k_dst, kCommon));
      } else {
        copy_row_warp<kTail>(pointer::offset(v_src, kCommon), pointer::offset(v_dst, kCommon));
      }
    }
  } else {
    copy_row_warp<kKBytes>(k_src, k_dst);
    copy_row_warp<kVBytes>(v_src, v_dst);
  }
}

/**
 * \brief Kernel to store key-value pairs into the KV cache.
 * Each element is split into multiple parts to allow parallel memory copy.
 * \tparam kKElementBytes The size of each key element in bytes.
 * \tparam kVElementBytes The size of each value element in bytes. Differs from
 *         kKElementBytes for asymmetric KV (head_dim != v_head_dim).
 * \tparam kSplit The number of warps that handle each element.
 * \tparam kUsePDL Whether to use PDL feature.
 * \tparam T The data type of the indices (`int32_t` or `int64_t`).
 */
template <int64_t kKElementBytes, int64_t kVElementBytes, int kSplit, bool kUsePDL, typename T>
__global__ void store_kvcache(const __grid_constant__ StoreKVCacheParams params) {
  using namespace device;
  constexpr auto kKSplitSize = kKElementBytes / kSplit;
  constexpr auto kVSplitSize = kVElementBytes / kSplit;
  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint32_t item_id = warp_id / kSplit;
  const uint32_t split_id = warp_id % kSplit;
  const auto& [
    k_input, v_input, k_cache, v_cache, indices, // ptr
    stride_k, stride_v, stride_k_cache, stride_v_cache, stride_indices, batch_size, // size
    size_limit, reserved_skip_index // bounds and reserved sink
  ] = params;
  if (item_id >= batch_size) return;

  const auto index_ptr = static_cast<const T*>(indices) + item_id * stride_indices;
  PDLWaitPrimary<kUsePDL>();

  const auto index = *index_ptr;
  // A stale/OOB slot id would cause an illegal memory access in the store below;
  // fail fast at the culprit instead. always-on (kvcache JIT compiles without NDEBUG).
  assert(index >= 0 && index < size_limit);
  const auto k_src = pointer::offset(k_input, item_id * stride_k, split_id * kKSplitSize);
  const auto v_src = pointer::offset(v_input, item_id * stride_v, split_id * kVSplitSize);
  const auto k_dst = pointer::offset(k_cache, index * stride_k_cache, split_id * kKSplitSize);
  const auto v_dst = pointer::offset(v_cache, index * stride_v_cache, split_id * kVSplitSize);

  if (index != reserved_skip_index) {
    copy_kv_rows_warp<kKSplitSize, kVSplitSize>(k_src, v_src, k_dst, v_dst);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kKElementBytes, int64_t kVElementBytes, bool kUsePDL>
struct StoreKVCacheKernel {
  static_assert(kKElementBytes > 0 && kKElementBytes % 4 == 0);
  static_assert(kVElementBytes > 0 && kVElementBytes % 4 == 0);

  template <int kSplit, typename T>
  static constexpr auto store_kernel = store_kvcache<kKElementBytes, kVElementBytes, kSplit, kUsePDL, T>;

  template <typename T>
  static auto get_kernel(const int num_split) {
    using namespace host;
    // only apply split optimization when both element sizes are aligned
    if constexpr (kKElementBytes % (4 * 128) == 0 && kVElementBytes % (4 * 128) == 0) {
      if (num_split == 4) return store_kernel<4, T>;
    }
    if constexpr (kKElementBytes % (2 * 128) == 0 && kVElementBytes % (2 * 128) == 0) {
      if (num_split == 2) return store_kernel<2, T>;
    }
    if (num_split == 1) return store_kernel<1, T>;
    Panic("Unsupported num_split {} for element sizes k={} v={}", num_split, kKElementBytes, kVElementBytes);
  }

  static void
  run(const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView indices,
      const int num_split,
      const int64_t size_limit,
      const int64_t reserved_skip_index) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto DK = SymbolicSize{"k_element_size"};
    auto DV = SymbolicSize{"v_element_size"};
    auto KS = SymbolicSize{"k_stride"};
    auto VS = SymbolicSize{"v_stride"};
    auto SK = SymbolicSize{"k_cache_stride"};
    auto SV = SymbolicSize{"v_cache_stride"};
    auto I = SymbolicSize{"indices_stride"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    auto indice_dtype = SymbolicDType{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({B, DK})  //
        .with_strides({KS, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(k);
    TensorMatcher({B, DV})  //
        .with_strides({VS, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(v);
    TensorMatcher({-1, DK})  //
        .with_strides({SK, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(k_cache);
    TensorMatcher({-1, DV})  //
        .with_strides({SV, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(v_cache);
    TensorMatcher({B})  //
        .with_strides({I})
        .with_dtype<int32_t, int64_t>(indice_dtype)
        .with_device(device)
        .verify(indices);

    const int64_t dtype_size = dtype_bytes(dtype.unwrap());
    const uint32_t num_elements = static_cast<uint32_t>(B.unwrap());
    RuntimeCheck(kKElementBytes == dtype_size * DK.unwrap());
    RuntimeCheck(kVElementBytes == dtype_size * DV.unwrap());

    const auto params = StoreKVCacheParams{
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .stride_k_bytes = KS.unwrap() * dtype_size,
        .stride_v_bytes = VS.unwrap() * dtype_size,
        .stride_k_cache_bytes = SK.unwrap() * dtype_size,
        .stride_v_cache_bytes = SV.unwrap() * dtype_size,
        .stride_indices = I.unwrap(),
        .batch_size = static_cast<uint32_t>(B.unwrap()),
        .size_limit = size_limit,
        .reserved_skip_index = reserved_skip_index,
    };
    // select kernel and update num_split if needed
    const auto use_int32 = indice_dtype.is_type<int32_t>();
    const auto kernel = use_int32 ? get_kernel<int32_t>(num_split) : get_kernel<int64_t>(num_split);
    const auto num_blocks = div_ceil(num_elements * num_split, kNumWarps);
    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
''',
    'sgl_kernel/runtime.cuh': r'''/// \file runtime.cuh
/// \brief Host-side CUDA runtime query helpers.
///
/// Thin wrappers around CUDA occupancy and device-property APIs with
/// automatic error checking via `RuntimeDeviceCheck`.

#pragma once

#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#ifndef USE_ROCM
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif
#ifndef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#endif
#ifndef cudaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#endif
#ifndef cudaDevAttrComputeCapabilityMinor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif
#ifndef cudaRuntimeGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#endif
#ifndef cudaOccupancyAvailableDynamicSMemPerBlock
inline hipError_t
cudaOccupancyAvailableDynamicSMemPerBlock(std::size_t* smem, const void* func, int num_blocks, int block_size) {
  // HIP does not expose this directly; return max shared mem as conservative estimate
  hipDeviceProp_t prop;
  int device;
  hipGetDevice(&device);
  hipGetDeviceProperties(&prop, device);
  *smem = prop.sharedMemPerBlock;
  return hipSuccess;
}
#endif
#endif

namespace sglang {

namespace host::runtime {

// Return the maximum number of active blocks per SM for the given kernel
template <typename T>
inline auto get_blocks_per_sm(T&& kernel, int32_t block_dim, std::size_t dynamic_smem = 0) -> uint32_t {
  int num_blocks_per_sm = 0;
  RuntimeDeviceCheck(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, block_dim, dynamic_smem));
  return static_cast<uint32_t>(num_blocks_per_sm);
}

// Return the number of SMs for the given device
inline auto get_sm_count(int device_id) -> uint32_t {
  int sm_count;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  return static_cast<uint32_t>(sm_count);
}

// Return the Major compute capability for the given device
inline auto get_cc_major(int device_id) -> int {
  int cc_major;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id));
  return cc_major;
}

// Return the Minor compute capability for the given device
inline auto get_cc_minor(int device_id) -> int {
  int cc_minor;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device_id));
  return cc_minor;
}

// Return the SM version (major * 10 + minor) for the given device
inline auto get_sm_version(int device_id) -> int {
  return get_cc_major(device_id) * 10 + get_cc_minor(device_id);
}

// Return the runtime version
inline auto get_runtime_version() -> int {
  int runtime_version;
  RuntimeDeviceCheck(cudaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}

// Return the maximum dynamic shared memory per block for the given kernel
template <typename T>
inline auto get_available_dynamic_smem_per_block(T&& kernel, int num_blocks, int block_size) -> std::size_t {
  std::size_t smem_size;
  RuntimeDeviceCheck(cudaOccupancyAvailableDynamicSMemPerBlock(&smem_size, kernel, num_blocks, block_size));
  return smem_size;
}

}  // namespace host::runtime

}  // namespace sglang
''',
    'sgl_kernel/source_location.h': r'''/// \file source_location.h
/// \brief Portable `source_location` wrapper.
///
/// Uses `std::source_location` when available (C++20), otherwise falls
/// back to a minimal stub that returns empty/zero values.

#pragma once
#include <version>

#if defined(__cpp_lib_source_location)
#include <source_location>
#endif

namespace sglang {

/// NOTE: fallback to a minimal source_location implementation
#if defined(__cpp_lib_source_location)

using source_location_t = std::source_location;

#else

struct source_location_fallback {
 public:
  static constexpr source_location_fallback current() noexcept {
    return source_location_fallback{};
  }
  constexpr source_location_fallback() noexcept = default;
  constexpr unsigned line() const noexcept {
    return 0;
  }
  constexpr unsigned column() const noexcept {
    return 0;
  }
  constexpr const char* file_name() const noexcept {
    return "";
  }
  constexpr const char* function_name() const noexcept {
    return "";
  }
};

using source_location_t = source_location_fallback;

#endif

}  // namespace sglang
''',
    'sgl_kernel/tensor.h': r'''/// \file tensor.h
/// \brief Tensor validation and symbolic matching utilities.
///
/// Provides the `TensorMatcher` fluent API for validating tensor shapes,
/// strides, dtypes, and devices at kernel entry points, along with
/// `SymbolicSize`, `SymbolicDType`, and `SymbolicDevice` for capturing
/// and cross-checking tensor metadata across multiple tensors.
///
/// See the "Tensor Checking" section in the JIT kernel dev guide for
/// usage examples.

#pragma once
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
#include <sgl_kernel/utils.cuh>
#elif defined(__HIPCC__)
#include <sgl_kernel/utils.cuh>
#endif

namespace sglang {

namespace host {

namespace details {

inline constexpr auto kAnyDeviceID = -1;
inline constexpr auto kAnySize = static_cast<int64_t>(-1);
inline constexpr auto kNullSize = static_cast<int64_t>(-1);
inline constexpr auto kNullDType = static_cast<DLDataTypeCode>(18u);
inline constexpr auto kNullDevice = static_cast<DLDeviceType>(-1);

struct SizeRef;
struct DTypeRef;
struct DeviceRef;

template <typename T>
struct DLDataTypeTrait {};

template <std::integral T>
struct DLDataTypeTrait<T> {
  inline static constexpr DLDataType value = {
      .code = std::is_signed_v<T> ? DLDataTypeCode::kDLInt : DLDataTypeCode::kDLUInt,
      .bits = static_cast<std::uint8_t>(sizeof(T) * 8),
      .lanes = 1};
};

template <std::floating_point T>
struct DLDataTypeTrait<T> {
  inline static constexpr DLDataType value = {
      .code = DLDataTypeCode::kDLFloat, .bits = static_cast<std::uint8_t>(sizeof(T) * 8), .lanes = 1};
};

#ifdef __CUDACC__
template <>
struct DLDataTypeTrait<fp16_t> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLFloat, .bits = 16, .lanes = 1};
};
template <>
struct DLDataTypeTrait<bf16_t> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLBfloat, .bits = 16, .lanes = 1};
};
template <>
struct DLDataTypeTrait<fp8_e4m3_t> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLFloat8_e4m3fn, .bits = 8, .lanes = 1};
};
#elif defined(__HIPCC__)
template <>
struct DLDataTypeTrait<fp16_t> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLFloat, .bits = 16, .lanes = 1};
};
template <>
struct DLDataTypeTrait<bf16_t> {
  inline static constexpr DLDataType value = {.code = DLDataTypeCode::kDLBfloat, .bits = 16, .lanes = 1};
};
#endif

template <DLDeviceType Code>
struct DLDeviceTrait {
  inline static constexpr DLDevice value = {.device_type = Code, .device_id = kAnyDeviceID};
};

template <typename... Ts>
inline constexpr auto kDTypeList = std::array<DLDataType, sizeof...(Ts)>{DLDataTypeTrait<Ts>::value...};

template <DLDeviceType... Codes>
inline constexpr auto kDeviceList = std::array<DLDevice, sizeof...(Codes)>{DLDeviceTrait<Codes>::value...};

template <typename T>
struct PrintAbleSpan {
  explicit PrintAbleSpan(std::span<const T> data) : data(data) {}
  std::span<const T> data;
};

// define DLDataType comparison and printing in root namespace
inline constexpr auto kDeviceStringMap = [] {
  constexpr auto map = std::array<std::pair<DLDeviceType, const char*>, 16>{
      std::pair{DLDeviceType::kDLCPU, "cpu"},
      std::pair{DLDeviceType::kDLCUDA, "cuda"},
      std::pair{DLDeviceType::kDLCUDAHost, "cuda_host"},
      std::pair{DLDeviceType::kDLOpenCL, "opencl"},
      std::pair{DLDeviceType::kDLVulkan, "vulkan"},
      std::pair{DLDeviceType::kDLMetal, "metal"},
      std::pair{DLDeviceType::kDLVPI, "vpi"},
      std::pair{DLDeviceType::kDLROCM, "rocm"},
      std::pair{DLDeviceType::kDLROCMHost, "rocm_host"},
      std::pair{DLDeviceType::kDLExtDev, "ext_dev"},
      std::pair{DLDeviceType::kDLCUDAManaged, "cuda_managed"},
      std::pair{DLDeviceType::kDLOneAPI, "oneapi"},
      std::pair{DLDeviceType::kDLWebGPU, "webgpu"},
      std::pair{DLDeviceType::kDLHexagon, "hexagon"},
      std::pair{DLDeviceType::kDLMAIA, "maia"},
      std::pair{DLDeviceType::kDLTrn, "trn"},
  };
  constexpr auto max_type = stdr::max(map | stdv::keys);
  auto result = std::array<std::string_view, max_type + 1>{};
  for (const auto& [code, name] : map) {
    result[static_cast<std::size_t>(code)] = name;
  }
  return result;
}();

struct PrintableDevice {
  DLDevice device;
};

inline auto& operator<<(std::ostream& os, DLDevice device) {
  const auto& mapping = kDeviceStringMap;
  const auto entry = static_cast<std::size_t>(device.device_type);
  RuntimeCheck(entry < mapping.size());
  const auto name = mapping[entry];
  RuntimeCheck(!name.empty(), "Unknown device: ", int(device.device_type));
  os << name;
  if (device.device_id != kAnyDeviceID && device.device_type != DLDeviceType::kDLCPU) {
    os << ":" << device.device_id;
  }
  return os;
}

inline auto& operator<<(std::ostream& os, PrintableDevice pd) {
  return os << pd.device;
}

template <typename T>
inline auto& operator<<(std::ostream& os, PrintAbleSpan<T> span) {
  os << "[";
  for (const auto i : irange(span.data.size())) {
    if (i > 0) {
      os << ", ";
    }
    os << span.data[i];
  }
  os << "]";
  return os;
}

}  // namespace details

/// \brief Check whether `dtype` matches the DLDataType for C++ type `T`.
template <typename T>
inline bool is_type(DLDataType dtype) {
  return dtype == details::DLDataTypeTrait<T>::value;
}

/**
 * \brief A symbolic dimension size that can be bound once and
 *        verified across multiple tensors.
 *
 * Create with an optional annotation string for error messages:
 * \code
 *   auto N = SymbolicSize{"num_tokens"};
 * \endcode
 *
 * Call `verify()` during tensor matching to either bind the first
 * observed value or check subsequent values match. Call `unwrap()`
 * to retrieve the bound value (panics if unset).
 */
struct SymbolicSize {
 public:
  SymbolicSize(std::string_view annotation = {}) : m_value(details::kNullSize), m_annotation(annotation) {}
  SymbolicSize(const SymbolicSize&) = delete;
  SymbolicSize& operator=(const SymbolicSize&) = delete;

  auto get_name() const -> std::string_view {
    return m_annotation;
  }

  auto set_value(int64_t value) -> void {
    RuntimeCheck(!this->has_value(), "Size value already set");
    m_value = value;
  }

  auto has_value() const -> bool {
    return m_value != details::kNullSize;
  }

  auto get_value() const -> std::optional<int64_t> {
    return this->has_value() ? std::optional{m_value} : std::nullopt;
  }

  auto unwrap(DebugInfo info = {}) const -> int64_t {
    RuntimeCheck(info, this->has_value(), "Size value is not set");
    return m_value;
  }

  auto verify(int64_t value, const char* prefix, int64_t dim) -> void {
    if (this->has_value()) {
      if (m_value != value) {
        [[unlikely]];
        Panic("Size mismatch for ", m_name_str(prefix, dim), ": expected ", m_value, " but got ", value);
      }
    } else {
      this->set_value(value);
    }
  }

  auto value_or_name(const char* prefix, int64_t dim) const -> std::string {
    if (const auto value = this->get_value()) {
      return std::to_string(*value);
    } else {
      return m_name_str(prefix, dim);
    }
  }

 private:
  auto m_name_str(const char* prefix, int64_t dim) const -> std::string {
    std::ostringstream os;
    os << prefix << '#' << dim;
    if (!m_annotation.empty()) os << "('" << m_annotation << "')";
    return std::move(os).str();
  }

  std::int64_t m_value;
  std::string_view m_annotation;
};

inline auto operator==(DLDevice lhs, DLDevice rhs) -> bool {
  return lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id;
}

/**
 * \brief A symbolic data type that can be constrained and verified.
 *
 * Optionally restrict allowed types via `set_options<fp16_t, bf16_t>()`.
 * Use `verify()` to bind/check the dtype, and `unwrap()` to retrieve it.
 */
struct SymbolicDType {
 public:
  SymbolicDType() : m_value({details::kNullDType, 0, 0}) {}
  SymbolicDType(const SymbolicDType&) = delete;
  SymbolicDType& operator=(const SymbolicDType&) = delete;

  auto set_value(DLDataType value) -> void {
    RuntimeCheck(!this->has_value(), "Dtype value already set");
    RuntimeCheck(
        m_check(value), "Dtype value [", value, "] not in the allowed options: ", details::PrintAbleSpan{m_options});
    m_value = value;
  }

  auto has_value() const -> bool {
    return m_value.code != details::kNullDType;
  }

  auto get_value() const -> std::optional<DLDataType> {
    return this->has_value() ? std::optional{m_value} : std::nullopt;
  }

  auto unwrap(DebugInfo info = {}) const -> DLDataType {
    RuntimeCheck(info, this->has_value(), "Dtype value is not set");
    return m_value;
  }

  auto set_options(std::span<const DLDataType> options) -> void {
    m_options = options;
  }

  template <typename... Ts>
  auto set_options() -> void {
    m_options = details::kDTypeList<Ts...>;
  }

  auto verify(DLDataType dtype) -> void {
    if (this->has_value()) {
      RuntimeCheck(m_value == dtype, "DType mismatch: expected ", m_value, " but got ", dtype);
    } else {
      this->set_value(dtype);
    }
  }

  template <typename T>
  auto is_type() const -> bool {
    return host::is_type<T>(m_value);
  }

 private:
  auto m_check(DLDataType value) const -> bool {
    return stdr::empty(m_options) || (stdr::find(m_options, value) != stdr::end(m_options));
  }

  std::span<const DLDataType> m_options;
  DLDataType m_value;
};

/**
 * \brief A symbolic device that can be constrained and verified.
 *
 * Optionally restrict allowed device types via
 * `set_options<kDLCUDA, kDLCPU>()`. The device id can be wildcarded.
 */
struct SymbolicDevice {
 public:
  SymbolicDevice() : m_value({details::kNullDevice, details::kAnyDeviceID}) {}
  SymbolicDevice(const SymbolicDevice&) = delete;
  SymbolicDevice& operator=(const SymbolicDevice&) = delete;

  auto set_value(DLDevice value) -> void {
    RuntimeCheck(!this->has_value(), "Device value already set");
    RuntimeCheck(
        m_check(value),
        "Device value [",
        details::PrintableDevice{value},
        "] not in the allowed options: ",
        details::PrintAbleSpan{m_options});
    m_value = value;
  }

  auto has_value() const -> bool {
    return m_value.device_type != details::kNullDevice;
  }

  auto get_value() const -> std::optional<DLDevice> {
    return this->has_value() ? std::optional{m_value} : std::nullopt;
  }

  auto unwrap(DebugInfo info = {}) const -> DLDevice {
    RuntimeCheck(info, this->has_value(), "Device value is not set");
    return m_value;
  }

  auto set_options(std::span<const DLDevice> options) -> void {
    m_options = options;
  }

  template <DLDeviceType... Codes>
  auto set_options() -> void {
    m_options = details::kDeviceList<Codes...>;
  }

  auto verify(DLDevice device) -> void {
    if (this->has_value()) {
      RuntimeCheck(
          m_value == device,
          "Device mismatch: expected ",
          details::PrintableDevice{m_value},
          " but got ",
          details::PrintableDevice{device});
    } else {
      this->set_value(device);
    }
  }

 private:
  auto m_check(DLDevice value) const -> bool {
    return stdr::empty(m_options) || (stdr::any_of(m_options, [value](const DLDevice& opt) {
             // device type must exactly match
             if (opt.device_type != value.device_type) return false;
             // device id can be wildcarded
             return opt.device_id == details::kAnyDeviceID || opt.device_id == value.device_id;
           }));
  }

  std::span<const DLDevice> m_options;
  DLDevice m_value;
};

namespace details {

template <typename T>
struct BaseRef {
 public:
  BaseRef(const BaseRef&) = delete;
  BaseRef& operator=(const BaseRef&) = delete;

  auto operator->() const -> T* {
    return m_ref;
  }
  auto operator*() const -> T& {
    return *m_ref;
  }
  auto rebind(T& other) -> void {
    m_ref = &other;
  }

  explicit BaseRef() : m_ref(&m_cache), m_cache() {}
  BaseRef(T& size) : m_ref(&size), m_cache() {}

 private:
  T* m_ref;
  T m_cache;
};

struct SizeRef : BaseRef<SymbolicSize> {
  using BaseRef::BaseRef;
  SizeRef(int64_t value) {
    if (value != kAnySize) {
      (**this).set_value(value);
    } else {
      // otherwise, we can match any size
    }
  }
};

struct DTypeRef : BaseRef<SymbolicDType> {
  using BaseRef::BaseRef;
  DTypeRef(DLDataType options) {
    (**this).set_value(options);
  }
  DTypeRef(std::initializer_list<DLDataType> options) {
    (**this).set_options(options);
  }
  DTypeRef(std::span<const DLDataType> options) {
    (**this).set_options(options);
  }
};

struct DeviceRef : BaseRef<SymbolicDevice> {
  using BaseRef::BaseRef;
  DeviceRef(DLDevice options) {
    (**this).set_value(options);
  }
  DeviceRef(std::initializer_list<DLDevice> options) {
    (**this).set_options(options);
  }
  DeviceRef(std::span<const DLDevice> options) {
    (**this).set_options(options);
  }
};

}  // namespace details

/**
 * \brief Fluent API for validating tensor shape, strides, dtype, and device.
 *
 * Construct with the expected shape (using `SymbolicSize` or literal
 * integers), chain `.with_strides()`, `.with_dtype<...>()`, and
 * `.with_device<...>()`, then call `.verify(tensor)`.
 *
 * Example:
 * \code
 *   auto N = SymbolicSize{"N"};
 *   TensorMatcher({N, 128})
 *       .with_dtype<fp16_t, bf16_t>()
 *       .with_device<kDLCUDA>()
 *       .verify(input_tensor);
 * \endcode
 *
 * \note `TensorMatcher` is a move-only temporary. Do not store in a variable.
 */
struct TensorMatcher {
 private:
  using SizeRef = details::SizeRef;
  using DTypeRef = details::DTypeRef;
  using DeviceRef = details::DeviceRef;

 public:
  TensorMatcher(const TensorMatcher&) = delete;
  TensorMatcher& operator=(const TensorMatcher&) = delete;

  explicit TensorMatcher(std::initializer_list<SizeRef> shape) : m_shape(shape), m_strides(), m_dtype() {}

  auto with_strides(std::initializer_list<SizeRef> strides) && -> TensorMatcher&& {
    // no partial update allowed
    RuntimeCheck(m_strides.size() == 0, "Strides already specified");
    RuntimeCheck(m_shape.size() == strides.size(), "Strides size must match shape size");
    m_strides = strides;
    return std::move(*this);
  }

  template <typename... Ts>
  auto with_dtype(DTypeRef&& dtype) && -> TensorMatcher&& {
    m_init_dtype();
    m_dtype.rebind(*dtype);
    m_dtype->set_options<Ts...>();
    return std::move(*this);
  }

  template <typename... Ts>
  auto with_dtype() && -> TensorMatcher&& {
    static_assert(sizeof...(Ts) > 0, "At least one dtype option must be specified");
    m_init_dtype();
    m_dtype->set_options<Ts...>();
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
  auto with_device(DeviceRef&& device) && -> TensorMatcher&& {
    m_init_device();
    m_device.rebind(*device);
    m_device->set_options<Codes...>();
    return std::move(*this);
  }

  template <DLDeviceType... Codes>
  auto with_device() && -> TensorMatcher&& {
    static_assert(sizeof...(Codes) > 0, "At least one device option must be specified");
    m_init_device();
    m_device->set_options<Codes...>();
    return std::move(*this);
  }

  // once we start verification, we cannot modify anymore
  auto verify(tvm::ffi::TensorView view, DebugInfo info = {}) const&& -> const TensorMatcher&& {
    try {
      m_verify_impl(view);
    } catch (PanicError& e) {
      auto oss = std::ostringstream{};
      oss << "Tensor match failed for ";
      s_print_tensor(oss, view);
      oss << " at " << info.file_name() << ":" << info.line() << "\n- Root cause: " << e.root_cause();
      throw PanicError(std::move(oss).str());
    }
    return std::move(*this);
  }

 private:
  static auto s_print_tensor(std::ostringstream& oss, tvm::ffi::TensorView view) -> void {
    oss << "Tensor<";
    int64_t dim = 0;
    for (const auto& size : view.shape()) {
      if (dim++ > 0) oss << ", ";
      oss << size;
    }
    oss << ">[strides=<";
    dim = 0;
    for (const auto& stride : view.strides()) {
      if (dim++ > 0) {
        oss << ", ";
      }
      oss << stride;
    }
    oss << ">, dtype=" << view.dtype();
    oss << ", device=" << details::PrintableDevice{view.device()} << "]";
  }

  auto m_verify_impl(tvm::ffi::TensorView view) const -> void {
    const auto dim = static_cast<std::size_t>(view.dim());
    RuntimeCheck(dim == m_shape.size(), "Tensor dimension mismatch: expected ", m_shape.size(), " but got ", dim);
    for (const auto i : irange(dim)) {
      m_shape[i]->verify(view.size(i), "shape", i);
    }
    if (m_has_strides()) {
      for (const auto i : irange(dim)) {
        if (view.size(i) != 1 || !m_strides[i]->has_value()) {
          // skip stride check for size 1 dimension
          m_strides[i]->verify(view.stride(i), "stride", i);
        }
      }
    } else {
      RuntimeCheck(view.is_contiguous(), "Tensor is not contiguous as expected");
    }
    // since we may double verify, we will force to check
    m_dtype->verify(view.dtype());
    m_device->verify(view.device());
  }

  auto m_init_dtype() -> void {
    RuntimeCheck(!m_has_dtype, "DType already specified");
    m_has_dtype = true;
  }

  auto m_init_device() -> void {
    RuntimeCheck(!m_has_device, "Device already specified");
    m_has_device = true;
  }

  auto m_has_strides() const -> bool {
    return !m_strides.empty();
  }

  std::span<const SizeRef> m_shape;
  std::span<const SizeRef> m_strides;
  DTypeRef m_dtype;
  DeviceRef m_device;
  bool m_has_dtype = false;
  bool m_has_device = false;
};

}  // namespace host

}  // namespace sglang
''',
    'sgl_kernel/tile.cuh': r'''/// \file tile.cuh
/// \brief Tiled memory access helpers for coalesced global memory I/O.
///
/// `tile::Memory<T>` represents a contiguous memory region where multiple
/// threads cooperatively load/store elements. The three factory methods
/// determine the thread group:
/// - `thread()` - single thread (no tiling).
/// - `warp()`   - all threads in a warp cooperate.
/// - `cta()`    - all threads in the CTA cooperate.

#pragma once
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

namespace device::tile {

/**
 * \brief Represents a contiguous memory region for cooperative tiled access.
 *
 * Each instance is parameterized by an element type `T` and bound to a
 * specific thread id (`tid`) within a group of `tsize` threads.
 *
 * \tparam T The storage element type (e.g. `AlignedVector<packed_t<float>, 4>`).
 */
template <typename T>
struct Memory {
 public:
  SGL_DEVICE constexpr Memory(uint32_t tid, uint32_t tsize) : tid(tid), tsize(tsize) {}
  /// \brief Create a Memory accessor for a single thread (no cooperation).
  SGL_DEVICE static constexpr Memory thread() {
    return Memory{0, 1};
  }
  /// \brief Create a Memory accessor distributed across warp threads.
  SGL_DEVICE static Memory warp(int warp_threads = kWarpThreads) {
    return Memory{static_cast<uint32_t>(threadIdx.x % warp_threads), static_cast<uint32_t>(warp_threads)};
  }
  /// \brief Create a Memory accessor distributed across all CTA threads.
  SGL_DEVICE static Memory cta(int cta_threads = blockDim.x) {
    return Memory{static_cast<uint32_t>(threadIdx.x), static_cast<uint32_t>(cta_threads)};
  }
  /// \brief Load one element from `ptr` at the position assigned to this thread.
  /// \param ptr  Base pointer (cast to `const T*`).
  /// \param offset  Optional tile offset (multiplied by `tsize`).
  SGL_DEVICE T load(const void* ptr, int64_t offset = 0) const {
    return static_cast<const T*>(ptr)[tid + offset * tsize];
  }
  /// \brief Store one element to `ptr` at the position assigned to this thread.
  SGL_DEVICE void store(void* ptr, T val, int64_t offset = 0) const {
    static_cast<T*>(ptr)[tid + offset * tsize] = val;
  }
  /// \brief Check whether this thread's element index is within bounds.
  SGL_DEVICE bool in_bound(int64_t element_count, int64_t offset = 0) const {
    return tid + offset * tsize < element_count;
  }

 private:
  uint32_t tid;
  uint32_t tsize;
};

}  // namespace device::tile

}  // namespace sglang
''',
    'sgl_kernel/type.cuh': r'''/// \file type.cuh
/// \brief Dtype trait system for CUDA scalar/packed types.
///
/// `DTypeTrait<T>` provides per-type metadata: packed type alias,
/// conversion functions (`from`), and unary/binary math operations.
/// Use `device::cast<To>(from_value)` for type conversion on device.

#pragma once
#include <sgl_kernel/utils.cuh>

#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace sglang {

template <typename T>
struct DTypeTrait {};

#define SGL_REGISTER_PACKED(SELF, PACKED) \
  using self_t = SELF;                    \
  using packed_t = PACKED

#define SGL_REGISTER_UNPACK(UNPACK, N) \
  using unpacked_t = UNPACK;           \
  static constexpr size_t kVecSize = N

#define SGL_REGISTER_FROM_DEFAULT()               \
  template <typename S>                           \
  SGL_DEVICE static self_t from(const S& value) { \
    return static_cast<self_t>(value);            \
  }                                               \
  static_assert(true)

#define SGL_REGISTER_FROM_FUNCTION(FROM, FN)     \
  SGL_DEVICE static self_t from(const FROM& x) { \
    return FN(x);                                \
  }                                              \
  static_assert(true)

#define SGL_REGISTER_UNARY_FUNCTION(NAME, FN)      \
  SGL_DEVICE static self_t NAME(const self_t& x) { \
    return FN(x);                                  \
  }                                                \
  static_assert(true)

// Also emits a `kHas_<NAME>` flag so reduction dispatch can detect the op via
// plain member SFINAE (see details::HasMax below) - hipcc mis-evaluates
// requires-expressions that probe device functions, so detection must only
// ever look at data members.
#define SGL_REGISTER_BINARY_FUNCTION(NAME, FN)                      \
  static constexpr bool kHas_##NAME = true;                         \
  SGL_DEVICE static self_t NAME(const self_t& x, const self_t& y) { \
    return FN(x, y);                                                \
  }                                                                 \
  static_assert(true)

template <std::integral T>
struct DTypeTrait<T> {
  SGL_REGISTER_PACKED(T, void);
  SGL_REGISTER_UNPACK(T, 1);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_UNARY_FUNCTION(abs, ::abs);
  SGL_REGISTER_BINARY_FUNCTION(max, ::max);
  SGL_REGISTER_BINARY_FUNCTION(min, ::min);
  static constexpr T kZeroBits = 0;
};

template <>
struct DTypeTrait<fp32_t> {
  SGL_REGISTER_PACKED(fp32_t, fp32x2_t);
  SGL_REGISTER_UNPACK(fp32_t, 1);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp16_t, __half2float);
  SGL_REGISTER_FROM_FUNCTION(bf16_t, __bfloat162float);
  SGL_REGISTER_UNARY_FUNCTION(abs, fabsf);
  SGL_REGISTER_UNARY_FUNCTION(sqrt, sqrtf);
  SGL_REGISTER_UNARY_FUNCTION(rsqrt, rsqrtf);
  SGL_REGISTER_UNARY_FUNCTION(exp, expf);
  SGL_REGISTER_UNARY_FUNCTION(sin, sinf);
  SGL_REGISTER_UNARY_FUNCTION(cos, cosf);
  SGL_REGISTER_BINARY_FUNCTION(max, fmaxf);
  SGL_REGISTER_BINARY_FUNCTION(min, fminf);
  static constexpr float kFloatMax = std::numeric_limits<float>::max();
  static constexpr uint32_t kZeroBits = 0x00000000;
};

template <>
struct DTypeTrait<fp32x2_t> {
  SGL_REGISTER_PACKED(fp32x2_t, fp32x4_t);
  SGL_REGISTER_UNPACK(fp32_t, 2);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp16x2_t, __half22float2);
  SGL_REGISTER_FROM_FUNCTION(bf16x2_t, __bfloat1622float2);
};

template <>
struct DTypeTrait<fp32x4_t> {
  SGL_REGISTER_PACKED(fp32x4_t, void);
  SGL_REGISTER_UNPACK(fp32_t, 4);
  SGL_REGISTER_FROM_DEFAULT();
};

template <>
struct DTypeTrait<fp16_t> {
  SGL_REGISTER_PACKED(fp16_t, fp16x2_t);
  SGL_REGISTER_UNPACK(fp16_t, 1);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp32_t, __float2half_rn);
  SGL_REGISTER_UNARY_FUNCTION(abs, __habs);
  SGL_REGISTER_BINARY_FUNCTION(max, __hmax);
  SGL_REGISTER_BINARY_FUNCTION(min, __hmin);
  // CUDA fp16 max clamp value
  static constexpr float kFloatMax = 65504.0f;
  static constexpr uint16_t kZeroBits = 0x0000;
};

template <>
struct DTypeTrait<fp16x2_t> {
  SGL_REGISTER_PACKED(fp16x2_t, void);
  SGL_REGISTER_UNPACK(fp16_t, 2);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp32x2_t, __float22half2_rn);
  SGL_REGISTER_UNARY_FUNCTION(abs, __habs2);
#ifndef USE_ROCM
  SGL_REGISTER_BINARY_FUNCTION(add, __hadd2);
  SGL_REGISTER_BINARY_FUNCTION(max, __hmax2);
  SGL_REGISTER_BINARY_FUNCTION(min, __hmin2);
#else
  // HIP only provides __hmax2/__hmin2 for __hip_bfloat162, not __half2.
  // No `add` registered on HIP (packed SUM falls back to lane-wise scalar).
  static constexpr bool kHas_max = true;
  static constexpr bool kHas_min = true;
  SGL_DEVICE static self_t max(const self_t& x, const self_t& y) {
    return self_t{__hmax(x.x, y.x), __hmax(x.y, y.y)};
  }
  SGL_DEVICE static self_t min(const self_t& x, const self_t& y) {
    return self_t{__hmin(x.x, y.x), __hmin(x.y, y.y)};
  }
#endif
};

template <>
struct DTypeTrait<bf16_t> {
  SGL_REGISTER_PACKED(bf16_t, bf16x2_t);
  SGL_REGISTER_UNPACK(bf16_t, 1);
  SGL_REGISTER_FROM_DEFAULT();
#ifndef USE_ROCM
  SGL_REGISTER_FROM_FUNCTION(fp32_t, __float2bfloat16_rn);
#else
  // HIP has no _rn-suffixed variant; __float2bfloat16 rounds to nearest.
  SGL_REGISTER_FROM_FUNCTION(fp32_t, __float2bfloat16);
#endif
  SGL_REGISTER_UNARY_FUNCTION(abs, __habs);
  SGL_REGISTER_BINARY_FUNCTION(max, __hmax);
  SGL_REGISTER_BINARY_FUNCTION(min, __hmin);
  // CUDA bf16 max clamp value
  static constexpr float kFloatMax = 3.38953139e38f;
  static constexpr uint16_t kZeroBits = 0x0000;
};

template <>
struct DTypeTrait<bf16x2_t> {
  SGL_REGISTER_PACKED(bf16x2_t, void);
  SGL_REGISTER_UNPACK(bf16_t, 2);
  SGL_REGISTER_FROM_DEFAULT();
  SGL_REGISTER_FROM_FUNCTION(fp32x2_t, __float22bfloat162_rn);
  SGL_REGISTER_UNARY_FUNCTION(abs, __habs2);
#ifndef USE_ROCM
  // No `add` on HIP: bf162 __hadd2 is unverified there (packed SUM falls
  // back to lane-wise scalar).
  SGL_REGISTER_BINARY_FUNCTION(add, __hadd2);
#endif
  SGL_REGISTER_BINARY_FUNCTION(max, __hmax2);
  SGL_REGISTER_BINARY_FUNCTION(min, __hmin2);
};

#ifndef USE_ROCM
template <>
struct DTypeTrait<fp8_e4m3_t> {
  SGL_REGISTER_PACKED(fp8_e4m3_t, fp8x2_e4m3_t);
  SGL_REGISTER_UNPACK(fp8_e4m3_t, 1);
  SGL_REGISTER_FROM_DEFAULT();
  // NOTE: CUDA fp8 support explicit cast (i.e. use default from is ok)

  static constexpr float kFloatMax = 448.0f;  // CUDA fp8 max clamp value
  static constexpr uint8_t kZeroBits = 0x00;
};

template <>
struct DTypeTrait<fp8x2_e4m3_t> {
  SGL_REGISTER_PACKED(fp8x2_e4m3_t, fp8x4_e4m3_t);
  SGL_REGISTER_UNPACK(fp8_e4m3_t, 2);
  SGL_REGISTER_FROM_DEFAULT();
  // NOTE: CUDA fp8 support explicit cast (i.e. use default from is ok)
};

template <>
struct DTypeTrait<fp8x4_e4m3_t> {
  SGL_REGISTER_PACKED(fp8x4_e4m3_t, void);
  SGL_REGISTER_UNPACK(fp8_e4m3_t, 4);
  SGL_REGISTER_FROM_DEFAULT();
  // NOTE: CUDA fp8 support explicit cast (i.e. use default from is ok)
};
#endif

#undef SGL_REGISTER_PACKED
#undef SGL_REGISTER_UNPACK
#undef SGL_REGISTER_FROM_DEFAULT
#undef SGL_REGISTER_FROM_FUNCTION
#undef SGL_REGISTER_UNARY_FUNCTION
#undef SGL_REGISTER_BINARY_FUNCTION

/// \brief Alias: the packed (x2) type for `T`.
template <typename T>
using packed_t = typename DTypeTrait<T>::packed_t;

namespace device {

/**
 * \brief Cast a value from type `From` to type `To` on device.
 *
 * Dispatches through `DTypeTrait<To>::from()`, which uses the appropriate
 * CUDA intrinsic (e.g. `__half2float`, `__float22half2_rn`).
 */
template <typename To, typename From>
SGL_DEVICE To cast(const From& value) {
  return DTypeTrait<To>::from(value);
}

/**
 * \brief View a packed value as an array of its `unpacked_t` elements.
 *
 * Returns a reference to `value` reinterpreted as `unpacked_t[kVecSize]`,
 * so element writes propagate back to the original packed value.
 * Constness of `value` is preserved.
 */
template <typename T>
SGL_DEVICE auto& unpack(T& value) {
  using Trait = DTypeTrait<std::remove_const_t<T>>;
  using U = typename Trait::unpacked_t;
  constexpr size_t kVecSize = Trait::kVecSize;
  static_assert(sizeof(T) == sizeof(U) * kVecSize, "packed type must be layout-compatible");
  using A = std::conditional_t<std::is_const_v<T>, const U, U>;
  return reinterpret_cast<A(&)[kVecSize]>(value);
}

enum class ReductionOp : uint8_t { SUM, MAX, MIN };

template <ReductionOp Op, typename T>
struct ReductionTrait {};

namespace details {

// Op detection via the `kHas_*` data members emitted by
// SGL_REGISTER_BINARY_FUNCTION. Deliberately classic void_t member SFINAE:
// hipcc mis-evaluates requires-expressions in device instantiation contexts
// (observed: even `requires { a + b; }` on float came out false), so detection
// must never probe function-call expressions.
template <typename T, typename = void>
struct HasAdd : std::false_type {};
template <typename T>
struct HasAdd<T, std::void_t<decltype(DTypeTrait<T>::kHas_add)>> : std::true_type {};

template <typename T, typename = void>
struct HasMax : std::false_type {};
template <typename T>
struct HasMax<T, std::void_t<decltype(DTypeTrait<T>::kHas_max)>> : std::true_type {};

template <typename T, typename = void>
struct HasMin : std::false_type {};
template <typename T>
struct HasMin<T, std::void_t<decltype(DTypeTrait<T>::kHas_min)>> : std::true_type {};

template <ReductionOp Op, typename T>
SGL_DEVICE T reduce_recursive(const T& x, const T& y) {
  using U = typename DTypeTrait<T>::unpacked_t;
  constexpr size_t kVecSize = DTypeTrait<T>::kVecSize;
  static_assert(kVecSize > 1, "unsupported scalar type for reduction");
  using Trait = ReductionTrait<Op, U>;
  auto& x_unpacked = device::unpack(x);
  auto& y_unpacked = device::unpack(y);
  T result{};
  auto& z_unpacked = device::unpack(result);
#pragma unroll
  for (size_t i = 0; i < kVecSize; ++i) {
    z_unpacked[i] = Trait::reduce(x_unpacked[i], y_unpacked[i]);
  }
  return result;
}

}  // namespace details

// Dispatch rules, chosen so correctness never depends on detection:
// scalars (kVecSize == 1) call the trait member / operator directly - a
// missing op is a clear compile error at the call line; packed types use the
// native op when the trait registered one and fall back to lane-wise
// recursion otherwise (worst case for a mis-detecting compiler is a slightly
// slower but still correct lane-wise path).
template <typename T>
struct ReductionTrait<ReductionOp::SUM, T> {
  SGL_DEVICE static T reduce(const T& x, const T& y) {
    if constexpr (details::HasAdd<T>::value) {
      return DTypeTrait<T>::add(x, y);
    } else if constexpr (DTypeTrait<T>::kVecSize == 1) {
      return static_cast<T>(x + y);
    } else {
      return details::reduce_recursive<ReductionOp::SUM>(x, y);
    }
  }
};

template <typename T>
struct ReductionTrait<ReductionOp::MAX, T> {
  SGL_DEVICE static T reduce(const T& x, const T& y) {
    if constexpr (DTypeTrait<T>::kVecSize == 1) {
      return DTypeTrait<T>::max(x, y);
    } else if constexpr (details::HasMax<T>::value) {
      return DTypeTrait<T>::max(x, y);
    } else {
      return details::reduce_recursive<ReductionOp::MAX>(x, y);
    }
  }
};

template <typename T>
struct ReductionTrait<ReductionOp::MIN, T> {
  SGL_DEVICE static T reduce(const T& x, const T& y) {
    if constexpr (DTypeTrait<T>::kVecSize == 1) {
      return DTypeTrait<T>::min(x, y);
    } else if constexpr (details::HasMin<T>::value) {
      return DTypeTrait<T>::min(x, y);
    } else {
      return details::reduce_recursive<ReductionOp::MIN>(x, y);
    }
  }
};

}  // namespace device

// ---------------------------------------------------------------------------
// FP8 max clamp value - platform-dependent
//   CUDA (e4m3fn):      448.0f
//   AMD FNUZ (e4m3fnuz): 224.0f
//   AMD E4M3 (e4m3fn):  448.0f
// ---------------------------------------------------------------------------
#ifndef USE_ROCM
inline constexpr float kFP8E4M3Max = 448.0f;
#else  // USE_ROCM
#if HIP_FP8_TYPE_FNUZ
inline constexpr float kFP8E4M3Max = 224.0f;
#else   // HIP_FP8_TYPE_E4M3
inline constexpr float kFP8E4M3Max = 448.0f;
#endif  // HIP_FP8_TYPE_FNUZ
#endif  // USE_ROCM

}  // namespace sglang
''',
    'sgl_kernel/utils.cuh': r'''/// \file utils.cuh
/// \brief Core CUDA/device utilities: type aliases, PDL helpers,
///        typed pointer access, kernel launch wrapper, and error checking.
///
/// This header is included (directly or transitively) by nearly every
/// JIT kernel. It provides:
/// - Scalar/packed type aliases (`fp16_t`, `bf16_t`, `fp8_e4m3_t`, ...).
/// - `SGL_DEVICE` macro (forced-inline device function qualifier).
/// - `kWarpThreads` constant (32).
/// - PDL (Programmatic Dependent Launch) helpers for Hopper (sm_90+).
/// - Typed `load_as` / `store_as` for void-pointer access.
/// - `pointer::offset` for safe void-pointer arithmetic.
/// - `host::LaunchKernel` - kernel launcher with optional PDL.
/// - `host::RuntimeDeviceCheck` - CUDA error checking.

#pragma once

#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <type_traits>
#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#ifndef __grid_constant__
#define __grid_constant__
#endif
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaLaunchConfig_t = hipLaunchConfig_t;
using cudaLaunchAttribute = hipLaunchAttribute;
inline constexpr auto cudaSuccess = hipSuccess;
#define cudaStreamPerThread hipStreamPerThread
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#endif

namespace sglang {

#ifndef USE_ROCM
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;

using fp32x2_t = float2;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
using fp8x2_e5m2_t = __nv_fp8x2_e5m2;
using fp8x4_e4m3_t = __nv_fp8x4_e4m3;
using fp8x4_e5m2_t = __nv_fp8x4_e5m2;

using fp32x4_t = float4;
#else
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __hip_bfloat16;
using fp8_e4m3_t = uint8_t;
using fp8_e5m2_t = uint8_t;
using fp32x2_t = float2;
using fp16x2_t = half2;
using bf16x2_t = __hip_bfloat162;
using fp8x2_e4m3_t = uint16_t;
using fp8x2_e5m2_t = uint16_t;
using fp8x4_e4m3_t = uint32_t;
using fp8x4_e5m2_t = uint32_t;
using fp32x4_t = float4;
#endif

/*
 * LDG Support
 */
#ifndef USE_ROCM
#define SGLANG_LDG(arg) __ldg(arg)
#else
#define SGLANG_LDG(arg) *(arg)
#endif

// DLPack device type for the current platform
#ifndef USE_ROCM
inline constexpr auto kDLGPU = kDLCUDA;
inline constexpr auto kDLGPUHost = kDLCUDAHost;
#else
inline constexpr auto kDLGPU = kDLROCM;
inline constexpr auto kDLGPUHost = kDLROCMHost;
#endif

namespace device {

/// \brief Macro: forced-inline device function qualifier.
#define SGL_DEVICE __forceinline__ __device__

// Architecture detection: SGL_CUDA_ARCH is injected by load_jit() and is
// available in both host and device compilation passes, whereas __CUDA_ARCH__
// is only defined by nvcc during the device pass.
#if !defined(USE_ROCM)
#if !defined(SGL_CUDA_ARCH)
#error "SGL_CUDA_ARCH is not defined. JIT compilation must inject -DSGL_CUDA_ARCH via load_jit()."
#endif
#if defined(__CUDA_ARCH__)
static_assert(
    __CUDA_ARCH__ == SGL_CUDA_ARCH, "SGL_CUDA_ARCH mismatch: injected arch flag does not match device target");
#endif
#define SGL_ARCH_HOPPER_OR_GREATER (SGL_CUDA_ARCH >= 900)
#define SGL_ARCH_BLACKWELL_OR_GREATER ((SGL_CUDA_ARCH >= 1000) && (CUDA_VERSION >= 12090))
#else  // USE_ROCM
#define SGL_ARCH_HOPPER_OR_GREATER 0
#define SGL_ARCH_BLACKWELL_OR_GREATER 0
#endif

// Maximum vector size in bytes supported by current architecture.
// Pre-Blackwell / AMD: 128-bit (16 bytes)
// Blackwell or greater: 256-bit (32 bytes)
inline constexpr std::size_t kMaxVecBytes = SGL_ARCH_BLACKWELL_OR_GREATER ? 32 : 16;

/// \brief Number of threads per warp (always 32 on NVIDIA/AMD GPUs).
inline constexpr auto kWarpThreads = 32u;
/// \brief Full warp active mask (all 32 lanes).
#ifndef USE_ROCM
inline constexpr auto kFullMask = 0xffffffffu;
#else
inline constexpr auto kFullMask = 0xffffffffffffffffULL;
#endif

/**
 * \brief PDL (Programmatic Dependent Launch): wait for the primary kernel.
 *
 * On Hopper (sm_90+), inserts a `griddepcontrol.wait` instruction to
 * synchronize with a preceding kernel in the same stream. On older
 * architectures or ROCm this is a no-op.
 */
template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {
#if SGL_ARCH_HOPPER_OR_GREATER
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

/**
 * \brief PDL: trigger dependent (secondary) kernel launch.
 *
 * On Hopper (sm_90+), inserts a `griddepcontrol.launch_dependents`
 * instruction. On older architectures or ROCm this is a no-op.
 */
template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {
#if SGL_ARCH_HOPPER_OR_GREATER
  if constexpr (kUsePDL) {
    // The "memory" clobber is load-bearing: without it the compiler may sink
    // this kernel's stores past the trigger, and the dependent grid's
    // griddepcontrol.wait only covers writes issued BEFORE launch_dependents.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
#endif
}

template <std::integral T, std::integral U>
SGL_DEVICE constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

/**
 * \brief Load data with the specified type and offset from a void pointer.
 * \tparam T The type to load.
 * \param ptr The base pointer.
 * \param offset The offset in number of elements of type T.
 */
template <typename T>
SGL_DEVICE T load_as(const void* ptr, int64_t offset = 0) {
  return static_cast<const T*>(ptr)[offset];
}

/**
 * \brief Store data with the specified type and offset to a void pointer.
 * \tparam T The type to store.
 * \param ptr The base pointer.
 * \param val The value to store.
 * \param offset The offset in number of elements of type T.
 * \note we use type_identity_t to force the caller to explicitly specify
 * the template parameter `T`, which can avoid accidentally using the wrong type.
 */
template <typename T>
SGL_DEVICE void store_as(void* ptr, std::type_identity_t<T> val, int64_t offset = 0) {
  static_cast<T*>(ptr)[offset] = val;
}

/// \brief Safe void-pointer arithmetic (byte-level by default).
namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T = char, std::integral... U>
SGL_DEVICE auto offset(void* ptr, U... offset) -> void* {
  return static_cast<T*>(ptr) + (... + offset);
}

template <typename T = char, std::integral... U>
SGL_DEVICE auto offset(const void* ptr, U... offset) -> const void* {
  return static_cast<const T*>(ptr) + (... + offset);
}

}  // namespace pointer

/// PTX pragma that lets the compiler spill registers into shared memory
SGL_DEVICE void enable_smem_spilling() {
#if defined(__CUDA_ARCH__) && CUDART_VERSION >= 13000
  asm(".pragma \"enable_smem_spilling\";");
#endif
}

}  // namespace device

namespace host {

/**
 * \brief Check the CUDA error code and panic with location info on failure.
 */
inline void RuntimeDeviceCheck(::cudaError_t error, DebugInfo location = {}) {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

/// \brief Check the last CUDA error (calls `cudaGetLastError`).
inline void RuntimeDeviceCheck(DebugInfo location = {}) {
  return RuntimeDeviceCheck(::cudaGetLastError(), location);
}

/**
 * \brief Kernel launcher with automatic stream resolution and PDL support.
 *
 * Usage:
 * \code
 *   host::LaunchKernel(grid, block, device)
 *       .enable_pdl(true)(my_kernel, arg0, arg1);
 *   host::LaunchKernel(grid, block, stream)
 *       .config({.use_pdl = true, .cluster_dim = cluster_dim})(my_kernel, arg0);
 * \endcode
 *
 * The constructor resolves the CUDA stream from a `DLDevice` (via `TVMFFIEnvGetStream`)
 * or accepts a raw `cudaStream_t`. The call operator launches the kernel and checks for errors.
 */
struct LaunchKernel {
 private:
  struct KernelConfig {
    bool use_pdl = false;
    std::optional<dim3> cluster_dim = std::nullopt;
  };

 public:
  explicit LaunchKernel(
      dim3 grid_dim,
      dim3 block_dim,
      DLDevice device,
      std::size_t dynamic_shared_mem_bytes = 0,
      DebugInfo location = {}) noexcept
      : m_config(s_make_config(grid_dim, block_dim, resolve_device(device), dynamic_shared_mem_bytes)),
        m_location(location) {}

  explicit LaunchKernel(
      dim3 grid_dim,
      dim3 block_dim,
      cudaStream_t stream,
      std::size_t dynamic_shared_mem_bytes = 0,
      DebugInfo location = {}) noexcept
      : m_config(s_make_config(grid_dim, block_dim, stream, dynamic_shared_mem_bytes)), m_location(location) {}

  LaunchKernel(const LaunchKernel&) = delete;
  LaunchKernel& operator=(const LaunchKernel&) = delete;

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  auto enable_pdl(bool enabled = true) -> LaunchKernel& {
#ifdef USE_ROCM
    (void)enabled;
    m_config.numAttrs = 0;
#else
    if (enabled) {
      auto& attr = m_attrs[m_config.numAttrs++];
      attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed = true;
      m_config.attrs = m_attrs;
    }
#endif
    return *this;
  }

  auto enable_cluster(dim3 cluster_dim) -> LaunchKernel& {
#ifdef USE_ROCM
    (void)cluster_dim;
#else
    auto& attr = m_attrs[m_config.numAttrs++];
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {cluster_dim.x, cluster_dim.y, cluster_dim.z};
    m_config.attrs = m_attrs;
#endif
    return *this;
  }

  /**
   * \brief Configure the kernel launch with the given options.
   * \param config The kernel configuration options.
   * \return A reference to this `LaunchKernel` for chaining.
   * \note This is a convenience method that applies multiple configurations at once.
   * We are in favor of this instead of `enable_pdl` and `enable_cluster`.
   * We enforce use of designated initializers for better readability.
   */
  auto config(const KernelConfig& config) -> LaunchKernel& {
    if (config.use_pdl) this->enable_pdl(true);
    if (config.cluster_dim) this->enable_cluster(*config.cluster_dim);
    return *this;
  }

  template <typename T, typename... Args>
  auto operator()(T&& kernel, Args&&... args) const -> void {
#ifdef USE_ROCM
    hipLaunchKernelGGL(
        std::forward<T>(kernel),
        m_config.gridDim,
        m_config.blockDim,
        m_config.dynamicSmemBytes,
        m_config.stream,
        std::forward<Args>(args)...);
    RuntimeDeviceCheck(m_location);
#else
    RuntimeDeviceCheck(::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...), m_location);
#endif
  }

  template <typename T, typename... Args>
  auto launch(T&& kernel, Args&&... args) const -> void {
    return (*this)(std::forward<T>(kernel), std::forward<Args>(args)...);
  }

 private:
  static auto s_make_config(  // Make a config for kernel launch
      dim3 grid_dim,
      dim3 block_dim,
      cudaStream_t stream,
      std::size_t smem) -> cudaLaunchConfig_t {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }

  cudaLaunchConfig_t m_config;
  const DebugInfo m_location;
  cudaLaunchAttribute m_attrs[2];
};

// The empty-true-branch if/else form keeps a trailing `else` in user code
// bound to the user's `if`, not to the macro's.
#define CHECK_CUDA(COND)                                              \
  if (const auto error = (COND); error == ::cudaSuccess) [[likely]] { \
  } else                                                              \
    host::Error() << "CUDA error: " << ::cudaGetErrorString(error) << ". "

}  // namespace host

}  // namespace sglang
''',
    'sgl_kernel/utils.h': r'''/// \file utils.h
/// \brief Host-side C++ utilities used by JIT kernel wrappers.

#pragma once

// ref: https://forums.developer.nvidia.com/t/c-20s-source-location-compilation-error-when-using-nvcc-12-1/258026/3
#ifdef __CUDACC__
#include <cuda.h>
#if CUDA_VERSION <= 12010

#pragma push_macro("__cpp_consteval")
#pragma push_macro("_NODISCARD")
#pragma push_macro("__builtin_LINE")

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#define __cpp_consteval 201811L
#pragma clang diagnostic pop

#ifdef _NODISCARD
#undef _NODISCARD
#define _NODISCARD
#endif

#define consteval constexpr

#include "source_location.h"

#undef consteval
#pragma pop_macro("__cpp_consteval")
#pragma pop_macro("_NODISCARD")
#else  // __CUDACC__ && CUDA_VERSION > 12010
#include "source_location.h"
#endif
#else  // no __CUDACC__
#include "source_location.h"
#endif

#include <dlpack/dlpack.h>

#include <concepts>
#include <cstddef>
#include <ostream>
#include <ranges>
#include <sstream>
#include <utility>

namespace sglang {

namespace host {

template <typename>
inline constexpr bool dependent_false_v = false;

/// \brief Source-location wrapper for debug/error messages.
struct DebugInfo : public source_location_t {
  DebugInfo(source_location_t loc = source_location_t::current()) : source_location_t(loc) {}
};

/// \brief Exception type thrown by `RuntimeCheck` and `Panic`.
struct PanicError : public std::runtime_error {
 public:
  explicit PanicError(std::string msg) : runtime_error(msg), m_message(std::move(msg)) {}
  auto root_cause() const -> std::string_view {
    const auto str = std::string_view{m_message};
    const auto pos = str.find(": ");
    return pos == std::string_view::npos ? str : str.substr(pos + 2);
  }

 private:
  std::string m_message;
};

/// \brief Unconditionally abort with a formatted error message.
template <typename... Args>
[[noreturn]]
inline auto panic(DebugInfo location, Args&&... args) -> void {
  std::ostringstream os;
  os << "Failed at " << location.file_name() << ":" << location.line();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  } else {
    os << " in " << location.function_name();
  }
  throw PanicError(std::move(os).str());
}

/**
 * \brief Runtime assertion: panics with a formatted message when `condition`
 *        is false. Extra `args` are streamed to the error message.
 *
 * Example:
 * \code
 *   RuntimeCheck(n > 0, "n must be positive, got ", n);
 * \endcode
 */
template <typename... Args>
struct RuntimeCheck {
  template <typename Cond>
  explicit RuntimeCheck(Cond&& condition, Args&&... args, DebugInfo location = {}) {
    if (condition) return;
    [[unlikely]] host::panic(location, std::forward<Args>(args)...);
  }
  template <typename Cond>
  explicit RuntimeCheck(DebugInfo location, Cond&& condition, Args&&... args) {
    if (condition) return;
    [[unlikely]] host::panic(location, std::forward<Args>(args)...);
  }
};

template <typename... Args>
struct Panic {
  explicit Panic(Args&&... args, DebugInfo location = {}) {
    host::panic(location, std::forward<Args>(args)...);
  }
  explicit Panic(DebugInfo location, Args&&... args) {
    host::panic(location, std::forward<Args>(args)...);
  }
  [[noreturn]] ~Panic() {
    std::terminate();
  }
};

template <typename Cond, typename... Args>
explicit RuntimeCheck(Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <typename Cond, typename... Args>
explicit RuntimeCheck(DebugInfo, Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <typename... Args>
explicit Panic(Args&&...) -> Panic<Args...>;

template <typename... Args>
explicit Panic(DebugInfo, Args&&...) -> Panic<Args...>;

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T = char, std::integral... U>
inline auto offset(void* ptr, U... offset) -> void* {
  return static_cast<T*>(ptr) + (... + offset);
}

template <typename T = char, std::integral... U>
inline auto offset(const void* ptr, U... offset) -> const void* {
  return static_cast<const T*>(ptr) + (... + offset);
}

}  // namespace pointer

/// \brief Integer ceiling division: ceil(a / b).
template <std::integral T, std::integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

/// \brief Returns the byte width of a DLPack data type.
inline auto dtype_bytes(DLDataType dtype) -> std::size_t {
  return static_cast<std::size_t>(dtype.bits / 8);
}

namespace stdr = std::ranges;
namespace stdv = stdr::views;

/// \brief Python-style integer range: `irange(n)` -> `[0, n)`.
template <std::integral T>
inline auto irange(T end) {
  return stdv::iota(static_cast<T>(0), end);
}

/// \brief Python-style integer range: `irange(start, end)` -> `[start, end)`.
template <std::integral T>
inline auto irange(T start, T end) {
  return stdv::iota(start, end);
}

/** \brief Error class for stream-style error logging. */
struct Error {
  Error(DebugInfo location = {}) {
    m_oss << "Failed at " << location.file_name() << ":" << location.line() << ": ";
  }

  template <typename T>
  Error& operator<<(T&& arg) {
    m_oss << std::forward<T>(arg);
    return *this;
  }

  [[noreturn]]
  ~Error() noexcept(false) {
    throw PanicError(std::move(m_oss).str());
  }

 private:
  std::ostringstream m_oss;
};

/**
 * \brief 0-overhead CHECK macro for host code. This can avoid unnecessary
 * instantiation of error messages when the condition is true.
 *
 * Usage: CHECK_HOST(ptr != nullptr) << "Pointer must not be null";
 */
// The empty-true-branch if/else form keeps a trailing `else` in user code
// bound to the user's `if`, not to the macro's.
#define CHECK_HOST(COND) \
  if (COND) [[likely]] { \
  } else                 \
    host::Error()

}  // namespace host

}  // namespace sglang
''',
    'sgl_kernel/vec.cuh': r'''/// \file vec.cuh
/// \brief Aligned vector types for coalesced global memory access.
///
/// `AlignedVector<T, N>` wraps `N` elements of type `T` in a naturally
/// aligned struct so that the compiler emits wide (vectorized) load/store
/// instructions (e.g. `LDG.128`). The maximum supported vector width is
/// 256 bits (32 bytes), matching CUDA's widest vector load.

#pragma once
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>

namespace sglang {

namespace device {

namespace details {

/// \brief Maps byte-width to the corresponding unsigned integer type.
template <std::size_t N>
struct uint_trait {};

template <>
struct uint_trait<1> {
  using type = uint8_t;
};

template <>
struct uint_trait<2> {
  using type = uint16_t;
};

template <>
struct uint_trait<4> {
  using type = uint32_t;
};

template <>
struct uint_trait<8> {
  using type = uint64_t;
};

/// \brief Alias: maps `sizeof(T)` to matching unsigned int type.
template <typename T>
using sized_int = typename uint_trait<sizeof(T)>::type;

}  // namespace details

/// \brief Raw aligned storage for `N` elements of type `T`.
template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};

/**
 * \brief Aligned vector for vectorized memory access on GPU.
 *
 * Stores `N` elements of type `T` with natural alignment so that a single
 * `load`/`store` call compiles to a wide memory transaction.
 *
 * \tparam T Element type (e.g. `fp16_t`, `bf16_t`, `float`).
 * \tparam N Number of elements. Must be a power of two and
 *           `sizeof(T) * N <= 32` (256 bits).
 *
 * Example:
 * \code
 *   AlignedVector<fp16_t, 8> vec;  // 16 bytes, 128-bit aligned
 *   vec.load(input_ptr, tid);      // vectorized load
 *   vec[0] = vec[0] + 1;
 *   vec.store(output_ptr, tid);    // vectorized store
 * \endcode
 */
template <typename T, std::size_t N>
struct AlignedVector {
 private:
  static_assert(
      (N > 0 && (N & (N - 1)) == 0) && sizeof(T) * N <= kMaxVecBytes,
      "CUDA vector size exceeds arch limit: max 16 bytes on pre-Blackwell/AMD, "
      "32 bytes on Blackwell or greater");
  using element_t = typename details::sized_int<T>;
  using storage_t = AlignedStorage<element_t, N>;

 public:
  /// \brief Vectorized load from `ptr` at the given element `offset`.
  SGL_DEVICE void load(const void* ptr, int64_t offset = 0) {
    m_storage = reinterpret_cast<const storage_t*>(ptr)[offset];
  }
  /// \brief Vectorized store to `ptr` at the given element `offset`.
  SGL_DEVICE void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<storage_t*>(ptr)[offset] = m_storage;
  }
  /// \brief Fill all N elements with the same `value`.
  SGL_DEVICE void fill(T value) {
    const auto store_value = *reinterpret_cast<element_t*>(&value);
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      m_storage.data[i] = store_value;
    }
  }

  SGL_DEVICE auto operator[](std::size_t idx) -> T& {
    return reinterpret_cast<T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto operator[](std::size_t idx) const -> T {
    return reinterpret_cast<const T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto data() -> T* {
    return reinterpret_cast<T*>(&m_storage);
  }
  SGL_DEVICE auto data() const -> const T* {
    return reinterpret_cast<const T*>(&m_storage);
  }

 private:
  storage_t m_storage;
};

/// Sum `M` vectors element-wise into one, accumulating in fp32 regardless of
/// the packed element type. Used by every collective that reduces peer
/// contributions in registers.
template <bool kFP32Acc = true, typename T2, size_t N, size_t M>
SGL_DEVICE auto reduce_vec(device::AlignedVector<T2, N> (&vec)[M]) -> device::AlignedVector<T2, N> {
  static_assert(DTypeTrait<T2>::kVecSize == 2, "reduce_vec only supports 2-element vectors for now");
  static_assert(M > 0, "reduce_vec requires at least one vector to reduce");
  if constexpr (kFP32Acc) {
    fp32x2_t acc[N];
#pragma unroll
    for (size_t i = 0; i < M; ++i) {
#pragma unroll
      for (size_t j = 0; j < N; ++j) {
        const auto [x, y] = cast<fp32x2_t>(vec[i][j]);
        acc[j].x = i == 0 ? x : acc[j].x + x;
        acc[j].y = i == 0 ? y : acc[j].y + y;
      }
    }
    device::AlignedVector<T2, N> out_vec;
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      out_vec[j] = cast<T2>(acc[j]);
    }
    return out_vec;
  } else {
    using SumOp = ReductionTrait<ReductionOp::SUM, T2>;
    device::AlignedVector<T2, N> out_vec = vec[0];
#pragma unroll
    for (size_t i = 1; i < M; ++i) {
#pragma unroll
      for (size_t j = 0; j < N; ++j) {
        out_vec[j] = SumOp::apply(out_vec[j], vec[i][j]);
      }
    }
    return out_vec;
  }
}

}  // namespace device

}  // namespace sglang
''',
}

@lru_cache
def _compiled_ops():
    """Compile the embedded upstream activation and cache-store kernels."""
    from tvm_ffi.cpp import load_inline

    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError("This extracted kernel configuration requires an SM100/SM103 GPU.")
    digest = hashlib.sha256("".join(_EMBEDDED_HEADERS.values()).encode()).hexdigest()[:16]
    root = Path(os.environ.get("QWEN38_KERNEL_CACHE", Path.home() / ".cache/qwen38_standalone")) / digest
    root.mkdir(parents=True, exist_ok=True)
    for name, source in _EMBEDDED_HEADERS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != source:
            # Atomic rename also permits separate processes compiling the same sources.
            tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            tmp.write_text(source)
            tmp.replace(path)
    source = '''
#include <elementwise/activation.cuh>
#include <elementwise/kvcache.cuh>
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu, (ActivationKernel<bf16_t, true>::run_activation));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(store, (StoreKVCacheKernel<2048, 2048, true>::run));
}
'''
    return load_inline(
        name=f"qwen38_ops_{digest}_sm{major}{minor}",
        cuda_sources=source,
        extra_include_paths=[str(root)],
        extra_cflags=["-std=c++20", "-O3"],
        extra_cuda_cflags=[f"-arch=sm_{major}{minor}a", f"-DSGL_CUDA_ARCH={major * 100 + minor * 10}", "-std=c++20", "-O3", "--expt-relaxed-constexpr"],
    )


# Upstream target-verification kernels; source hashes in verify_sources.json.
@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accept_tokens_ptr,
    intermediate_conv_window_ptr,
    intermediate_state_indices_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    retrieve_parent_token_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_intermediate_state_indices: tl.constexpr,
    stride_retrieve_next_token_seq: tl.constexpr,
    stride_retrieve_next_token_token: tl.constexpr,
    stride_retrieve_next_sibling_seq: tl.constexpr,
    stride_retrieve_next_sibling_token: tl.constexpr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    # ruff: noqa: E501
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
        if SAVE_INTERMEDIATE:
            intermediate_state_batch_coord = tl.load(
                intermediate_state_indices_ptr
                + idx_seq * stride_intermediate_state_indices
            ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = tl.load(num_accept_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # The conv_state updates works in a sliding window manner,
    # at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            :, None
        ]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        idx_tokens = tl.arange(0, NP2_SEQLEN)  # [BLOCK_M]
        # Update parent mapping for all tokens at once using vectorized operations
        mask_retrieve = idx_tokens < seqlen
        retrieve_next_token_base = (
            retrieve_next_token_ptr
            + (idx_seq * stride_retrieve_next_token_seq)
            + idx_tokens * stride_retrieve_next_token_token
        )
        retrieve_next_tokens = tl.load(retrieve_next_token_base, mask_retrieve)
        retrieve_next_sibling_base = (
            retrieve_next_sibling_ptr
            + (idx_seq * stride_retrieve_next_sibling_seq)
            + idx_tokens * stride_retrieve_next_sibling_token
        )
        retrieve_next_siblings = tl.load(retrieve_next_sibling_base, mask_retrieve)
        parent_idx_tokens = tl.zeros((NP2_SEQLEN,), dtype=tl.int32)

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # set the parent index of the next token in the eagle tree
            # next token's parent is the current token
            retrieve_next_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_tokens, 0)
            )
            if retrieve_next_token_idx != -1:  # pad slot id
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_next_token_idx,
                    idx_token,
                    parent_idx_tokens,
                )
            # next token's parent is the parent of the current token
            retrieve_sibling_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_siblings, 0)
            )
            if retrieve_sibling_token_idx != -1:  # pad slot id
                parent_idx_token = tl.sum(
                    tl.where(idx_tokens == idx_token, parent_idx_tokens, 0)
                )
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_sibling_token_idx,
                    parent_idx_token,
                    parent_idx_tokens,
                )
            # tl.device_print("am", parent_idx_tokens)

            _idx_token = idx_token
            x_ptrs_1d = x_base_1d + _idx_token * stride_x_token  # [BLOCK_N]
            matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            # convolution operation: itself * wcol[-1] + parent * wcol[-2] + grand-parent * wcol[-3] + ...
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 0:
                        matrix_w = w_col1
                    else:
                        matrix_w = w_col0
                elif KERNEL_WIDTH == 3:
                    if j == 0:
                        matrix_w = w_col2
                    elif j == 1:
                        matrix_w = w_col1
                    else:
                        matrix_w = w_col0
                elif KERNEL_WIDTH == 4:
                    if j == 0:
                        matrix_w = w_col3
                    elif j == 1:
                        matrix_w = w_col2
                    elif j == 2:
                        matrix_w = w_col1
                    else:
                        matrix_w = w_col0

                if SAVE_INTERMEDIATE:
                    # Save the window state after consuming this token
                    # Layout: [seq(cache line), step, dim, win(K-1)]
                    base_ptr = (
                        intermediate_conv_window_ptr
                        + intermediate_state_batch_coord * stride_inter_seq
                        + idx_token * stride_inter_step
                        + idx_feats * stride_inter_dim
                    )

                    # store itself in KERNEL_WIDTH-2 slot, parent in KERNEL_WIDTH-3 slot, grand-parent in KERNEL_WIDTH-4 slot, ...
                    if KERNEL_WIDTH - j - 2 >= 0:
                        tl.store(
                            base_ptr + (KERNEL_WIDTH - j - 2) * stride_inter_win,
                            matrix_x,
                            mask=mask_w,
                        )

                acc += matrix_x * matrix_w

                # move to parent for next iteration
                if _idx_token > 0:
                    _idx_token = tl.sum(
                        tl.where(idx_tokens == _idx_token, parent_idx_tokens, 0)
                    )
                    x_ptrs_1d = x_base_1d + _idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                else:
                    # no parent within the current chunk, load from prev conv state: col[-1] (idx 0's parent), col[-2] (idx 0's grand parent), ...
                    if KERNEL_WIDTH == 2:
                        if _idx_token == 0:
                            matrix_x = col0
                    elif KERNEL_WIDTH == 3:
                        if _idx_token == 0:
                            matrix_x = col1
                        else:
                            matrix_x = col0
                    elif KERNEL_WIDTH == 4:
                        if _idx_token == 0:
                            matrix_x = col2
                        elif _idx_token == -1:
                            matrix_x = col1
                        else:
                            matrix_x = col0
                    _idx_token = _idx_token - 1
        else:
            matrix_w = w_col0
            matrix_x = col0

            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 1:  # KERNEL_WIDTH-1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

                acc += matrix_x * matrix_w  # [BLOCK_N]

            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x

            if SAVE_INTERMEDIATE:
                # Save the window state after consuming this token
                # Layout: [seq(cache line), step, dim, win(K-1)]
                base_ptr = (
                    intermediate_conv_window_ptr
                    + intermediate_state_batch_coord * stride_inter_seq
                    + idx_token * stride_inter_step
                    + idx_feats * stride_inter_dim
                )
                if KERNEL_WIDTH >= 2:
                    tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
                if KERNEL_WIDTH >= 3:
                    tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
                if KERNEL_WIDTH >= 4:
                    tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)

        # fuse: store calculated retrieve_parent_token to tensor
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            tl.store(
                retrieve_parent_token_ptr
                + idx_seq * stride_retrieve_parent_token_seq
                + idx_tokens * stride_retrieve_parent_token_token,
                parent_idx_tokens,
                mask=mask_retrieve,
            )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accept_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    retrieve_next_token: Optional[torch.Tensor] = None,
    retrieve_next_sibling: Optional[torch.Tensor] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        )
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape
            assert intermediate_state_indices is not None
            assert (batch,) == intermediate_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = torch.empty_like(x)
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    stride_intermediate_state_indices = (
        intermediate_state_indices.stride(0)
        if intermediate_state_indices is not None
        else 0
    )
    if num_accept_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    np2_seqlen = triton.next_power_of_2(seqlen)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    # prepare retrieve next token buffer strides if provided
    if retrieve_next_token is not None:
        stride_retrieve_next_token_seq, stride_retrieve_next_token_token = (
            retrieve_next_token.stride(0),
            retrieve_next_token.stride(1),
        )
    else:
        stride_retrieve_next_token_seq = stride_retrieve_next_token_token = 0

    # prepare retrieve next sibling buffer strides if provided
    if retrieve_next_sibling is not None:
        stride_retrieve_next_sibling_seq, stride_retrieve_next_sibling_token = (
            retrieve_next_sibling.stride(0),
            retrieve_next_sibling.stride(1),
        )
    else:
        stride_retrieve_next_sibling_seq = stride_retrieve_next_sibling_token = 0

    # prepare retrieve parent token buffer strides if provided
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token = (
            retrieve_parent_token.stride(0),
            retrieve_parent_token.stride(1),
        )
    else:
        stride_retrieve_parent_token_seq = stride_retrieve_parent_token_token = 0

    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    _causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accept_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_intermediate_state_indices,
        stride_retrieve_next_token_seq,
        stride_retrieve_next_token_token,
        stride_retrieve_next_sibling_seq,
        stride_retrieve_next_sibling_token,
        stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accept_tokens is not None,
        NP2_STATELEN=np2_statelen,
        NP2_SEQLEN=np2_seqlen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_next_token is not None,
        **pdl_kwargs,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ================================================
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    # ReplaySSM fused ring-write. Pointers stay None and CACHE_RING False for
    # decode / flag-off -> byte-identical. The gate ring layout follows IS_KDA
    # (see the store below).
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
    stride_rawv_slot: tl.constexpr = 0,
    stride_rawk_slot: tl.constexpr = 0,
    stride_g_slot: tl.constexpr = 0,
    stride_beta_slot: tl.constexpr = 0,
    MAX_CACHE_LEN: tl.constexpr = 0,
    CACHE_RING: tl.constexpr = False,
    SPLIT_N_HV_GRID: tl.constexpr = False,
    USE_GDC: tl.constexpr = False,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    if SPLIT_N_HV_GRID:
        i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        # The GPU wrapper asserts NK == 1. Keep N and HV on independent grid
        # axes so large GLM5 decode batches do not exceed CUDA's grid limit.
        i_k = 0
    else:
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
    # PDL: overlap this kernel's prologue with the producer (the KDA/GDN
    # conv1d_update). All global loads below happen after the wait, so
    # numerics are unchanged. The immediate trigger releases the LAUNCH of
    # the next PDL kernel so its prologue overlaps this whole body;
    # consumers' own gdc_wait still fences on full completion.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # Slot stride comes from the caller (h0_source.stride(0)): the state pool
        # may be an envelope-strided view (page-major / unified memory), where the
        # per-slot pitch spans ALL layers' state, not HV*K*V. int64: envelope
        # pitches overflow an int32 index product.
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * stride_h0_source
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled. int64: the buffer is
    # contiguous but `cache_idx * cache_steps * HV * K * V` can exceed int32 for
    # large slot counts.
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias
        if USE_LOWER_BOUND:
            # KDA safe gate: lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
            b_g = lower_bound * tl.sigmoid(tl.exp(b_A_log) * x)
        else:
            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            beta_x = softplus_beta * x
            # Apply softplus with numerical stability
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
                x,
            )
            b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # fused ring-write: stash this step's raw inputs + in-kernel gate/beta
        # into the per-slot ring for the commit fold to replay. Must sit here --
        # b_k is still pre-l2norm, b_v still pre-delta, b_g/b_beta are formed,
        # so the fold's replay is bit-identical to the update below. rawk uses
        # the k-head i_h (shared across a GQA group); rawv/g/beta use the v-head
        # i_hv. step_idx < MAX_CACHE_LEN: absorb-inflated rows can exceed the
        # ring; the overflow steps are past the committable prefix, so drop them
        # (writing them would smash the next slot's ring).
        if CACHE_RING:
            ring_slot = tl.load(h0_indices + i_n).to(tl.int64)
            if ring_slot >= 0 and step_idx < MAX_CACHE_LEN:
                tl.store(
                    replayssm_rawv
                    + ring_slot * stride_rawv_slot
                    + i_hv * MAX_CACHE_LEN * V
                    + step_idx * V
                    + o_v,
                    b_v.to(replayssm_rawv.dtype.element_ty),
                    mask=mask_v,
                )
                if i_v == 0:
                    tl.store(
                        replayssm_rawk
                        + ring_slot * stride_rawk_slot
                        + i_h * MAX_CACHE_LEN * K
                        + step_idx * K
                        + o_k,
                        b_k.to(replayssm_rawk.dtype.element_ty),
                        mask=mask_k,
                    )
                    # b_g follows IS_KDA: KDA loads a/dt_bias with mask_k, so the
                    # gate is a per-K vector and the ring row is K wide; GDN's is
                    # a scalar per (head, step). The two layouts are not
                    # interchangeable -- storing one into the other's stride is a
                    # shape error, not a slow path -- and memory_pool.py sizes
                    # replayssm_g off the same is_kda test.
                    if IS_KDA:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN * K
                            + step_idx * K
                            + o_k,
                            b_g,
                            mask=mask_k,
                        )
                    else:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_g,
                        )
                    if i_k == 0:
                        tl.store(
                            replayssm_beta
                            + ring_slot * stride_beta_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_beta,
                        )

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * stride_h0_source
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    lower_bound: Optional[float] = None,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
    # fused ReplaySSM ring-write (spec verify). When cache_ring, each draft step
    # stores pre-norm k / raw v / gate / beta into these per-slot rings,
    # replacing the eager ring-write. Off by default -> decode unchanged.
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis stride.
    # For 2D a ([T, ...]) this is stride(0); for 3D a ([B, T, ...]) this is stride(1).
    # Using stride()[-2] covers GDN [T, HV] and KDA layouts ([T, HV*K] / [B, T, HV*K]).
    # KDA decode also passes 4-D [B, T, H, K], where [-2] is the head stride, not the
    # token stride; take dim 1 explicitly for that layout.
    stride_a = a.stride()[1] if a.ndim == 4 else a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    split_n_hv_grid = q.device.type == "cuda"
    grid = (NV, N, HV) if split_n_hv_grid else (NK, NV, N * HV)

    # Adaptive spec changes the runtime draft count without changing the
    # allocated per-request pitch, which is preserved in stride(0).
    if intermediate_states_buffer is not None:
        cache_stride_steps = intermediate_states_buffer.stride(0) // (HV * K * V)
    elif cache_steps is not None and cache_steps > 0:
        cache_stride_steps = cache_steps
    else:
        cache_stride_steps = 0

    # ring strides (per-slot rings are contiguous [num_slots, heads, L, dim];
    # the kernel offsets within a slot with MAX_CACHE_LEN and the dim extents).
    if cache_ring:
        # stride(0) is used as the slot pitch, so a tensor still carrying the
        # layer dim would scribble outside its slot. The gate ring is the one
        # whose rank depends on the model: per-K vector for KDA, per-head scalar
        # for GDN, matching g_shape in memory_pool.py and the IS_KDA branch in
        # the store above.
        assert (
            replayssm_rawv.dim() == 4
            and replayssm_rawk.dim() == 4
            and replayssm_g.dim() == (4 if is_kda else 3)
            and replayssm_beta.dim() == 3
        ), "cache_ring expects per-layer ring views"
        max_cache_len = replayssm_rawv.shape[-2]
        stride_rawv_slot = replayssm_rawv.stride(0)
        stride_rawk_slot = replayssm_rawk.stride(0)
        stride_g_slot = replayssm_g.stride(0)
        stride_beta_slot = replayssm_beta.stride(0)
    else:
        max_cache_len = 0
        stride_rawv_slot = stride_rawk_slot = stride_g_slot = stride_beta_slot = 0

    # PDL (sm90+): chain this kernel behind its producer conv1d_update, which
    # already launches dependents. Bit-exact (scheduling only) — benefits both
    # KDA and GDN recurrent paths.
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        lower_bound=lower_bound if lower_bound is not None else 0.0,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        # Envelope-strided state pools (page-major / unified memory) have a
        # per-slot pitch != HV*K*V; contiguous pools pass exactly HV*K*V.
        stride_h0_source=(
            initial_state_source.stride(0) if initial_state_source is not None else 0
        ),
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_stride_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        scale=scale,
        T=T,
        stride_a=stride_a,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        USE_LOWER_BOUND=lower_bound is not None,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        replayssm_rawv=replayssm_rawv,
        replayssm_rawk=replayssm_rawk,
        replayssm_g=replayssm_g,
        replayssm_beta=replayssm_beta,
        stride_rawv_slot=stride_rawv_slot,
        stride_rawk_slot=stride_rawk_slot,
        stride_g_slot=stride_g_slot,
        stride_beta_slot=stride_beta_slot,
        MAX_CACHE_LEN=max_cache_len,
        CACHE_RING=cache_ring,
        SPLIT_N_HV_GRID=split_n_hv_grid,
        num_warps=num_warps,
        num_stages=num_stages,
        **pdl_kwargs,
    )
    o = o.squeeze(0)
    return o



@dataclasses.dataclass
class HybridCache:
    """Per-request keys, values, convolution history, and recurrent state.

    Slots are stable across calls. request_indices selects/reorders active slots;
    lengths are counts of tokens already consumed, not generated tokens pending
    consumption. Do not mutate a cache concurrently from multiple calls.
    """
    owner: int
    lengths: list[int]
    capacity: int
    max_context: int
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]]
    conv: dict[int, torch.Tensor]
    recurrent: dict[int, torch.Tensor]

    def clone(self):
        return HybridCache(self.owner, self.lengths.copy(), self.capacity, self.max_context,
                           {i: (k.clone(), v.clone()) for i, (k, v) in self.kv.items()},
                           {i: x.clone() for i, x in self.conv.items()},
                           {i: x.clone() for i, x in self.recurrent.items()})

    @torch.inference_mode()
    def reset(self, request_indices=None):
        slots = list(range(len(self.lengths))) if request_indices is None else list(request_indices)
        if len(set(slots)) != len(slots) or any(i < 0 or i >= len(self.lengths) for i in slots):
            raise ValueError("Invalid or duplicate cache slots")
        for slot in slots:
            self.lengths[slot] = 0
            for state in self.conv.values():
                state[slot].zero_()
            for state in self.recurrent.values():
                state[slot].zero_()


def _token_batch(input_ids):
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim not in (1, 2) or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a 1D or 2D integer tensor")
        input_ids = input_ids.tolist()
    if not isinstance(input_ids, (list, tuple)) or len(input_ids) == 0:
        raise ValueError("input_ids must contain at least one nonempty sequence")
    single = isinstance(input_ids[0], int)
    rows = [input_ids] if single else input_ids
    result = []
    for row in rows:
        if isinstance(row, torch.Tensor):
            if row.ndim != 1 or row.dtype not in (torch.int32, torch.int64):
                raise ValueError("Each request must be a 1D integer sequence")
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("Empty requests and padded tensors are not accepted; pass unpadded lists")
        if any(type(t) is not int for t in row):
            raise ValueError("Token IDs must be integers")
        result.append(list(row))
    return result, single


class Qwen38:
    """Single-GPU Qwen3.8-27B with genuine packed multi-request execution."""

    @torch.inference_mode()
    def __init__(self, model_path="/path/to/Qwen3.8-27B", device="cuda:0", max_context=None):
        import flashinfer
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
        # sglang-kernel 0.4.6.post1 delegates these BF16 calls to FlashInfer.
        # Import that same implementation directly, without its Torch ABI binding.
        from flashinfer.norm import gemma_rmsnorm, gemma_fused_add_rmsnorm

        self.path = Path(model_path)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA is required")
        torch.cuda.set_device(self.device)
        if torch.cuda.get_device_capability(self.device)[0] != 10:
            raise ValueError("This extracted configuration supports NVIDIA SM100/SM103")
        self.dtype = torch.bfloat16
        self.config = json.loads((self.path / "config.json").read_text())["text_config"]
        c = self.config
        expected = {"hidden_size": 5120, "intermediate_size": 17408,
                    "num_attention_heads": 24, "num_key_value_heads": 4,
                    "head_dim": 256, "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                    "num_hidden_layers": 64, "vocab_size": 248320}
        for key, value in expected.items():
            if c.get(key) != value:
                raise ValueError(f"Not the supported Qwen3.8-27B configuration: {key}={c.get(key)}")
        if c.get("attention_bias") or c.get("hidden_act") != "silu" or not c.get("attn_output_gate"):
            raise ValueError("Unsupported attention/activation configuration")
        if c.get("output_gate_type", "swish") != "swish" or c.get("quantization_config"):
            raise ValueError("Only unquantized weights with the swish output gate are supported")
        self.eps = c["rms_norm_eps"]
        self.max_context = c["max_position_embeddings"] if max_context is None else int(max_context)
        if not 1 <= self.max_context <= c["max_position_embeddings"]:
            raise ValueError("max_context must be within the model context limit")
        self.layer_types = c["layer_types"]
        if self.layer_types != ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]:
            raise ValueError("Unexpected hybrid attention layout")
        generation = json.loads((self.path / "generation_config.json").read_text())
        eos = generation.get("eos_token_id", c["eos_token_id"])
        self.eos_token_ids = {eos} if isinstance(eos, int) else set(eos)
        self._norm = gemma_rmsnorm
        self._add_norm = gemma_fused_add_rmsnorm
        self._gdn_prefill = chunk_gated_delta_rule
        self._ops = _compiled_ops()
        self.page_size = 64
        self.workspace = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self._context_attention = flashinfer.prefill.trtllm_batch_context_with_kv_cache
        self._decode_attention = flashinfer.decode.trtllm_batch_decode_with_kv_cache
        sm_count = flashinfer.utils.get_device_sm_count(self.device)
        counter_bytes = flashinfer.utils.get_trtllm_gen_multi_ctas_kv_counter_bytes(8192, 24, sm_count)
        self._attention_counter = torch.zeros(counter_bytes, dtype=torch.uint8, device=self.device)
        self._load_weights()
        # RoPE is the same partial, non-interleaved rotation as the text path in
        # SGLang's multimodal rotary embedding (all three axes equal for text).
        rp = c["rope_parameters"]
        self.rotary_dim = int(c["head_dim"] * rp["partial_rotary_factor"])
        if self.rotary_dim != 64 or rp["rope_type"] != "default":
            raise ValueError("Unsupported rotary embedding configuration")
        # Upstream builds this cache under the CUDA model-loading context.
        # Computing it on CPU changes transcendental rounding before rotation.
        inv_freq = 1.0 / (rp["rope_theta"] ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=self.device) / self.rotary_dim))
        positions = torch.arange(self.max_context, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.rope = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    def _load_weights(self):
        index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        loaded = set()

        def read(name, shape=None, dtype=None):
            if name not in index:
                raise ValueError(f"Missing weight: {name}")
            with safe_open(self.path / index[name], framework="pt", device="cpu") as f:
                tensor = f.get_tensor(name)
            if shape is not None and tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"Wrong shape for {name}: {tensor.shape}, expected {shape}")
            if tensor.dtype not in (torch.bfloat16, torch.float32, torch.float16):
                raise ValueError(f"Quantized/unsupported weight: {name} ({tensor.dtype})")
            loaded.add(name)
            return tensor.to(device=self.device, dtype=dtype or self.dtype).contiguous()

        prefix = "model.language_model."
        self.embedding = read(prefix + "embed_tokens.weight", (248320, 5120))
        self.final_norm = read(prefix + "norm.weight", (5120,))
        self.lm_head = read("lm_head.weight", (248320, 5120))
        self.layers = []
        for i, kind in enumerate(self.layer_types):
            p = prefix + f"layers.{i}."
            w = {"input_norm": read(p + "input_layernorm.weight", (5120,)),
                 "post_norm": read(p + "post_attention_layernorm.weight", (5120,)),
                 "down": read(p + "mlp.down_proj.weight", (5120, 17408))}
            w["gate_up"] = torch.cat((read(p + "mlp.gate_proj.weight", (17408, 5120)), read(p + "mlp.up_proj.weight", (17408, 5120))))
            if kind == "full_attention":
                p += "self_attn."
                w["qkv"] = torch.cat([read(p + name + ".weight", (dim, 5120)) for name, dim in [("q_proj", 12288), ("k_proj", 1024), ("v_proj", 1024)]])
                w["out"] = read(p + "o_proj.weight", (5120, 6144))
                w["q_norm"] = read(p + "q_norm.weight", (256,))
                w["k_norm"] = read(p + "k_norm.weight", (256,))
            else:
                p += "linear_attn."
                if p + "in_proj_qkvz.weight" in index:
                    w["qkvz"] = read(p + "in_proj_qkvz.weight", (16384, 5120))
                else:
                    w["qkvz"] = torch.cat((read(p + "in_proj_qkv.weight", (10240, 5120)), read(p + "in_proj_z.weight", (6144, 5120))))
                if p + "in_proj_ba.weight" in index:
                    w["ba"] = read(p + "in_proj_ba.weight", (96, 5120))
                else:
                    w["ba"] = torch.cat((read(p + "in_proj_b.weight", (48, 5120)), read(p + "in_proj_a.weight", (48, 5120))))
                w["conv"] = read(p + "conv1d.weight", (10240, 1, 4)).view(10240, 4)
                w["A_log"] = read(p + "A_log", (48,), dtype=torch.float32)
                w["dt_bias"] = read(p + "dt_bias", (48,))
                w["norm"] = read(p + "norm.weight", (128,))
                w["out"] = read(p + "out_proj.weight", (5120, 6144))
            self.layers.append(w)
        unexpected = [name for name in index if name.startswith(prefix) and name not in loaded and ".mtp." not in name]
        if unexpected:
            raise ValueError(f"Unconsumed language-model weights: {unexpected[:8]}")
        self.loaded_weight_names = sorted(loaded)

    @torch.inference_mode()
    def new_cache(self, batch_size=1, initial_capacity=256, max_context=None):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        limit = self.max_context if max_context is None else int(max_context)
        if not 1 <= limit <= self.max_context or initial_capacity < 1:
            raise ValueError("Invalid cache capacity/context limit")
        capacity = math.ceil(min(initial_capacity, limit) / self.page_size) * self.page_size
        kv, conv, recurrent = {}, {}, {}
        for i, kind in enumerate(self.layer_types):
            if kind == "full_attention":
                shape = (batch_size * capacity + self.page_size, 4, 256)
                # Upstream initializes the entire pool, including unused page
                # tails and the reserved page, to zero. Preserve that invariant.
                kv[i] = tuple(torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in range(2))
            else:
                conv[i] = torch.zeros((batch_size, 10240, 3), dtype=self.dtype, device=self.device)
                recurrent[i] = torch.zeros((batch_size, 48, 128, 128), dtype=torch.float32, device=self.device)
        return HybridCache(id(self), [0] * batch_size, capacity, limit, kv, conv, recurrent)

    def _reserve(self, cache, needed):
        if needed > cache.max_context:
            raise ValueError(f"Context limit exceeded: {needed} > {cache.max_context}")
        if needed <= cache.capacity:
            return
        capacity = math.ceil(min(cache.max_context, max(needed, cache.capacity * 2)) / self.page_size) * self.page_size
        new_kv = {}
        for i, pair in cache.kv.items():
            new_pair = []
            for old in pair:
                new = torch.zeros((len(cache.lengths) * capacity + self.page_size, 4, 256), dtype=self.dtype, device=self.device)
                for slot, length in enumerate(cache.lengths):
                    new[self.page_size + slot * capacity:self.page_size + slot * capacity + length].copy_(old[self.page_size + slot * cache.capacity:self.page_size + slot * cache.capacity + length])
                new_pair.append(new)
            new_kv[i] = tuple(new_pair)
        cache.kv = new_kv
        cache.capacity = capacity

    @torch.inference_mode()
    def forward_step(self, input_ids, kv_cache, *, request_indices=None, return_all_logits=False):
        """Consume packed sequences, update the supplied cache, and return full logits."""
        slots, lengths, inputs = self._prepare_step(
            input_ids, kv_cache, request_indices, return_all_logits
        )
        logits = self._forward(kv_cache, **inputs)
        for slot, length in zip(slots, lengths):
            kv_cache.lengths[slot] = length
        return logits

    def _prepare_step(self, input_ids, kv_cache, request_indices=None,
                      return_all_logits=False, fixed_pages=False):
        rows, _ = _token_batch(input_ids)
        cache = kv_cache
        if not isinstance(cache, HybridCache) or cache.owner != id(self):
            raise ValueError("kv_cache must have been created by this model")
        slots = list(range(len(cache.lengths))) if request_indices is None else list(request_indices)
        if len(slots) != len(rows) or len(set(slots)) != len(slots) or any(type(s) is not int or s < 0 or s >= len(cache.lengths) for s in slots):
            raise ValueError("request_indices must identify one distinct cache slot per input sequence")
        flat = [t for row in rows for t in row]
        if any(t < 0 or t >= self.config["vocab_size"] for t in flat):
            raise ValueError("Token ID outside model vocabulary")
        lens = [len(r) for r in rows]
        prefixes = [cache.lengths[s] for s in slots]
        lengths = [p + n for p, n in zip(prefixes, lens)]
        if hasattr(cache, "slot_capacities"):
            if any(n > cache.slot_capacities[s] for s, n in zip(slots, lengths)):
                raise ValueError("Prepared cache capacity exceeded")
        else:
            self._reserve(cache, max(lengths))
        offsets = getattr(cache, "slot_offsets", [s * cache.capacity for s in range(len(cache.lengths))])
        is_decode = all(n == 1 and p > 0 for n, p in zip(lens, prefixes))
        cu = [0]
        kv_cu = [0]
        for n, length in zip(lens, lengths):
            cu.append(cu[-1] + n)
            kv_cu.append(kv_cu[-1] + length)
        def tensor(data, dtype=torch.int32):
            return torch.tensor(data, dtype=dtype, device=self.device)
        query_start = tensor(cu)
        state_indices = tensor(slots)
        positions = tensor([p + j for p, n in zip(prefixes, lens) for j in range(n)], torch.int64)
        locations = tensor([self.page_size + offsets[s] + p + j for s, p, n in zip(slots, prefixes, lens) for j in range(n)], torch.int64)
        pages_per_slot = cache.capacity // self.page_size
        max_pages = pages_per_slot if fixed_pages else math.ceil(max(lengths) / self.page_size)
        page_table = tensor([[1 + offsets[s] // self.page_size + j if j < math.ceil(n / self.page_size) else 0 for j in range(max_pages)] for s, n in zip(slots, lengths)])
        sequence_lengths = tensor(lengths)
        kv_starts = tensor(kv_cu)
        has_initial = tensor([p > 0 for p in prefixes], torch.bool)
        inputs = dict(
            token_ids=tensor(flat, torch.int64), query_start=query_start,
            state_indices=state_indices, positions=positions, locations=locations,
            page_table=page_table, sequence_lengths=sequence_lengths,
            kv_starts=kv_starts, has_initial=has_initial, is_decode=is_decode,
            max_q_len=max(lens), query_lens=lens,
            last_indices=None if return_all_logits else tensor([end - 1 for end in cu[1:]], torch.int64),
        )
        return slots, lengths, inputs

    def _forward(self, cache, *, token_ids, query_start, state_indices, positions,
                 locations, page_table, sequence_lengths, kv_starts, has_initial,
                 is_decode, max_q_len, query_lens, last_indices, verify=False):
        hidden = F.embedding(token_ids, self.embedding)
        residual = None
        for i, (kind, w) in enumerate(zip(self.layer_types, self.layers)):
            if residual is None:
                residual = hidden
                hidden = self._norm(hidden, w["input_norm"], self.eps)
            else:
                self._add_norm(hidden, residual, w["input_norm"], self.eps)
            if kind == "full_attention":
                projected = F.linear(hidden, w["qkv"])
                qg, k, v = projected.split([12288, 1024, 1024], dim=-1)
                q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(qg, k, w["q_norm"], w["k_norm"], self.rope, positions, self.eps, 24, 4, 256, self.rotary_dim, has_gate=True)
                kc, vc = cache.kv[i]
                self._ops.store(k.view(-1, 1024), v, kc.view(-1, 1024), vc.view(-1, 1024), locations, 4, kc.shape[0], 0)
                paged_kv = tuple(t.view(-1, self.page_size, 4, 256).permute(0, 2, 1, 3) for t in (kc, vc))
                attention_args = dict(query=q.view(-1, 24, 256), kv_cache=paged_kv, workspace_buffer=self.workspace, block_tables=page_table, seq_lens=sequence_lengths, bmm1_scale=256 ** -0.5, bmm2_scale=1.0, window_left=-1, sinks=None, skip_softmax_threshold_scale_factor=None, out_dtype=self.dtype)
                if is_decode or verify:
                    extra = {"q_len_per_req": max_q_len} if verify else {}
                    out = self._decode_attention(**attention_args, max_seq_len=self.max_context, multi_ctas_kv_counter_buffer=self._attention_counter, **extra)
                else:
                    out = self._context_attention(**attention_args, max_q_len=max_q_len, max_kv_len=self.max_context, batch_size=state_indices.numel(), cum_seq_lens_q=query_start, cum_seq_lens_kv=kv_starts)
                out = out.view(-1, 6144)
                out = fused_sigmoid_mul(out, gate.view(-1, 6144), inplace=True)
                hidden = F.linear(out, w["out"])
            else:
                qkvz = F.linear(hidden, w["qkvz"])
                ba = F.linear(hidden, w["ba"])
                if is_decode:
                    mixed, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(qkvz, ba, cache.conv[i], w["conv"], None, state_indices, qkv_dim=10240, v_dim=6144, num_v_heads=48, head_v_dim=128, activation="silu")
                    out = mixed.new_empty(state_indices.numel(), 1, 48, 128)
                    fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv=mixed, a=a, b=b, A_log=w["A_log"], dt_bias=w["dt_bias"], scale=128 ** -0.5, initial_state=cache.recurrent[i], out=out, ssm_state_indices=state_indices, use_qk_l2norm_in_kernel=True)
                    z = z.reshape(-1, 128)
                elif verify:
                    mixed, z, b, a = qwen3_5_gdn_prefill_projection_views(qkvz, ba, 16, 48, 128, 128)
                    mixed = causal_conv1d_update(
                        mixed.view(state_indices.numel(), max_q_len, -1).transpose(1, 2),
                        cache.conv[i], w["conv"], activation="silu",
                        conv_state_indices=state_indices,
                        intermediate_conv_window=cache.verify_conv[i],
                        intermediate_state_indices=state_indices,
                    ).transpose(1, 2).reshape(-1, 10240)
                    q, k, v = mixed.split([2048, 2048, 6144], dim=-1)
                    out = fused_sigmoid_gating_delta_rule_update(
                        A_log=w["A_log"], dt_bias=w["dt_bias"], a=a, b=b,
                        q=q.view(1, -1, 16, 128), k=k.view(1, -1, 16, 128),
                        v=v.view(1, -1, 48, 128),
                        initial_state_source=cache.recurrent[i], initial_state_indices=state_indices,
                        cu_seqlens=query_start, use_qk_l2norm_in_kernel=True,
                        softplus_beta=1.0, softplus_threshold=20.0,
                        disable_state_update=True,
                        intermediate_states_buffer=cache.verify_recurrent[i],
                        intermediate_state_indices=state_indices, cache_steps=max_q_len,
                    )
                else:
                    mixed, z, b, a = qwen3_5_gdn_prefill_projection_views(qkvz, ba, 16, 48, 128, 128)
                    mixed = causal_conv1d_fn(mixed.T, w["conv"], None, cache.conv[i], query_start, query_lens, cache_indices=state_indices, has_initial_state=has_initial, activation="silu").T
                    q, k, v = fused_qkv_split_gdn_prefill(mixed, 16, 16, 48, 128, 128, 128)
                    q, k, v = gdn_prefill_qkv_prepare_fwd(q[0], k[0], v[0])
                    g, beta = fused_gdn_gating(w["A_log"], a, b, w["dt_bias"])
                    initial = cache.recurrent[i][state_indices.to(torch.int64)].contiguous()
                    final = torch.empty_like(initial)
                    out, final = self._gdn_prefill(q=q, k=k, v=v, g=torch.exp(g[0].float()), beta=beta[0].float(), scale=None, initial_state=initial, output_final_state=True, cu_seqlens=query_start.to(torch.int64), use_qk_l2norm_in_kernel=False, output_state=final)
                    cache.recurrent[i].index_copy_(0, state_indices.to(torch.int64), final)
                out = rms_norm_gated(x=out.reshape(-1, 128), weight=w["norm"], bias=None, z=z, eps=self.eps, norm_before_gate=True, is_rms_norm=True, activation="swish")
                hidden = F.linear(out.reshape(-1, 6144), w["out"])
            self._add_norm(hidden, residual, w["post_norm"], self.eps)
            gate_up = F.linear(hidden, w["gate_up"])
            activated = torch.empty((token_ids.numel(), 17408), dtype=self.dtype, device=self.device)
            self._ops.silu(gate_up, activated, "silu")
            hidden = F.linear(activated, w["down"])
        self._add_norm(hidden, residual, self.final_norm, self.eps)
        if last_indices is not None:
            hidden = hidden[last_indices]
        logits = F.linear(hidden, self.lm_head).float()
        return logits

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens=None, eos_token_ids=None, prefill_chunk_size=512):
        """Generate new tokens until each request emits EOS, including that EOS.

        max_new_tokens=None means no artificial generation cutoff. Exhausting
        the model context without EOS raises RuntimeError. An explicit positive
        max_new_tokens returns at the requested cutoff even without EOS.
        Ragged requests are packed together for every prefill/decode call.
        """
        rows, single = _token_batch(input_ids)
        if max_new_tokens is not None and (type(max_new_tokens) is not int or max_new_tokens < 0):
            raise ValueError("max_new_tokens must be a nonnegative integer or None")
        if type(prefill_chunk_size) is not int or prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be a positive integer")
        eos = self.eos_token_ids if eos_token_ids is None else ({eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids))
        if not eos or any(type(t) is not int or t < 0 or t >= self.config["vocab_size"] for t in eos):
            raise ValueError("eos_token_ids must contain valid vocabulary IDs")
        output = [[] for _ in rows]
        if max_new_tokens == 0:
            return output[0] if single else output
        cache = self.new_cache(len(rows))
        if any(len(row) > cache.max_context for row in rows):
            raise ValueError("Input exceeds model context")
        last_logits = {}
        for offset in range(0, max(map(len, rows)), prefill_chunk_size):
            slots = [i for i, row in enumerate(rows) if offset < len(row)]
            chunks = [rows[i][offset:offset + prefill_chunk_size] for i in slots]
            logits = self.forward_step(chunks, cache, request_indices=slots)
            for j, slot in enumerate(slots):
                last_logits[slot] = logits[j]
        active = list(range(len(rows)))
        logits = torch.stack([last_logits[i] for i in active])
        while active:
            tokens = logits.argmax(-1).tolist()
            next_active, next_inputs = [], []
            for slot, token in zip(active, tokens):
                output[slot].append(token)
                if token in eos or (max_new_tokens is not None and len(output[slot]) >= max_new_tokens):
                    continue
                if cache.lengths[slot] >= cache.max_context:
                    raise RuntimeError(f"Request {slot} exhausted context without EOS; produced {len(output[slot])} tokens")
                next_active.append(slot)
                next_inputs.append([token])
            active = next_active
            if active:
                logits = self.forward_step(next_inputs, cache, request_indices=active)
        return output[0] if single else output


class DecodeRunner:
    """Own a fixed cache and one reusable decode graph for an exact batch size.

    Prefill with model.forward_step(..., runner.cache), then call prepare() and
    forward() for each decode. Only forward() belongs in prepared-step timing.
    forward_step() includes host preparation. Output storage is reused: clone it
    before retaining a result across another replay. Reset/reuse cache slots for
    new requests; this does not recapture. The original arbitrary-cache API stays
    available on the model. A runner is sequential and not thread-safe.
    """

    @torch.inference_mode()
    def __init__(self, model, batch_size=1, max_context=None):
        self.model = model
        self.batch_size = batch_size
        limit = model.max_context if max_context is None else max_context
        self.cache = model.new_cache(batch_size, initial_capacity=limit, max_context=limit)
        self.cache.lengths[:] = [1] * batch_size
        _, _, self.inputs = model._prepare_step(
            [[0]] * batch_size, self.cache, fixed_pages=True
        )
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=model.device)
        stream.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                model._forward(self.cache, **self.inputs)
        torch.cuda.current_stream(model.device).wait_stream(stream)
        with torch.cuda.graph(self.graph):
            self.logits = model._forward(self.cache, **self.inputs)
        # Capture only touches this fresh runner's dummy state, never a request.
        for pair in self.cache.kv.values():
            for tensor in pair:
                tensor.zero_()
        self.cache.reset()
        self.pending = None
        self.captures = 1
        self.replays = 0

    @torch.inference_mode()
    def prepare(self, input_ids, *, request_indices=None):
        """Validate and upload current metadata outside prepared-forward timing."""
        if self.pending is not None:
            raise RuntimeError("The prepared step has not been executed")
        rows, _ = _token_batch(input_ids)
        slots = list(range(self.batch_size)) if request_indices is None else list(request_indices)
        if len(rows) != self.batch_size or any(len(row) != 1 for row in rows):
            raise ValueError("DecodeRunner requires exactly one token per captured row")
        if len(slots) != self.batch_size or set(slots) != set(range(self.batch_size)):
            raise ValueError("DecodeRunner requires a permutation of its cache slots")
        if any(self.cache.lengths[slot] == 0 for slot in slots):
            raise ValueError("Prefill each selected slot before decoding")
        slots, lengths, inputs = self.model._prepare_step(
            rows, self.cache, slots, fixed_pages=True
        )
        for name, destination in self.inputs.items():
            if isinstance(destination, torch.Tensor):
                destination.copy_(inputs[name])
        self.pending = slots, lengths

    @torch.inference_mode()
    def forward(self):
        """Execute one prepared model step, including logits and cache writes."""
        if self.pending is None:
            raise RuntimeError("Call prepare before each forward")
        self.graph.replay()
        slots, lengths = self.pending
        for slot, length in zip(slots, lengths):
            self.cache.lengths[slot] = length
        self.pending = None
        self.replays += 1
        return self.logits

    def forward_step(self, input_ids, *, request_indices=None):
        self.prepare(input_ids, request_indices=request_indices)
        return self.forward()


def forward_step(model, input_ids, kv_cache, **kwargs):
    """Functional form of Qwen38.forward_step."""
    return model.forward_step(input_ids, kv_cache, **kwargs)


def generate(model, input_ids, **kwargs):
    """Functional form of Qwen38.generate."""
    return model.generate(input_ids, **kwargs)
