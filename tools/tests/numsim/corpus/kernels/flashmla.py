"""Small, CPU-owned numerical cases for the canonical FlashMLA kernels."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from tirx_harness.numsim.cases import (
    ComparisonSpec,
    NumSimCase,
    TensorMap,
)
from tests.numsim.support._tirx_kernels import load_tirx_kernel


def _encode_bfloat16(values: np.ndarray) -> np.ndarray:
    """Round float32 values to BF16 bits without relying on NumSim's decoder."""

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _decode_bfloat16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _encode_exact_e4m3(values: np.ndarray) -> np.ndarray:
    """Encode the small exact E4M3 value set used by the decode oracle."""

    values = np.asarray(values, dtype=np.float32)
    magnitude_codes = {
        np.float32(0.0): np.uint8(0x00),
        np.float32(0.5): np.uint8(0x30),
        np.float32(1.0): np.uint8(0x38),
    }
    encoded = np.empty(values.shape, dtype=np.uint8)
    for magnitude, code in magnitude_codes.items():
        encoded[np.abs(values) == magnitude] = code
    if not np.all(np.isin(np.abs(values), tuple(magnitude_codes))):
        raise ValueError("decode oracle uses a value without an exact E4M3 encoding")
    encoded[values < 0] |= np.uint8(0x80)
    return encoded


def _sparse_decode_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build four physical MODEL1 blocks and two distinguishable logical splits."""

    num_tokens = 256
    rows = np.arange(num_tokens, dtype=np.int32)[:, None]
    cols = np.arange(512, dtype=np.int32)[None, :]
    kv_values = ((3 * rows + 2 * cols + (rows // 5) * (cols % 3)) % 5 - 2).astype(
        np.float32
    ) * np.float32(0.5)

    heads = np.arange(64, dtype=np.int32)[:, None]
    q_values = ((2 * heads + 3 * cols + (heads // 7) * (cols % 4)) % 5 - 2).astype(
        np.float32
    ) * np.float32(0.125)
    q = _encode_bfloat16(q_values.reshape((1, 1, 64, 512)))

    # MODEL1 stores 64 token rows of 576 bytes followed by 64 packed scale
    # records of 8 bytes.  The first 448 token bytes are E4M3; the final 128
    # bytes are 64 BF16 RoPE values.  ue8m0 code 127 is an exact scale of 1.
    block_bytes = 37_440
    token_bytes = 576
    storage = np.zeros((4 * block_bytes,), dtype=np.uint8)
    fp8 = _encode_exact_e4m3(kv_values[:, :448])
    rope = _encode_bfloat16(kv_values[:, 448:]).view(np.uint8).reshape((num_tokens, 128))
    for token in range(num_tokens):
        block = token // 64
        row = token % 64
        row_start = block * block_bytes + row * token_bytes
        storage[row_start : row_start + 448] = fp8[token]
        storage[row_start + 448 : row_start + token_bytes] = rope[token]
        scale_start = block * block_bytes + 64 * token_bytes + row * 8
        storage[scale_start : scale_start + 7] = np.uint8(127)

    blocks = [
        block * 64 + np.roll(np.arange(64, dtype=np.int32), shift)
        for block, shift in enumerate((11, 23, 37, 49))
    ]
    indices = np.concatenate(blocks)
    indices[5] = -1
    indices[90] = -1
    indices[170] = -1
    indices[230] = -1
    return q, storage.view(np.uint16).reshape((260, 288)), indices, kv_values


def _sparse_decode_reference(
    q_bits: np.ndarray, kv_values: np.ndarray, indices: np.ndarray
) -> dict[str, np.ndarray]:
    """Independent sparse-attention oracle spanning main and split combine."""

    q = _decode_bfloat16(q_bits)[0, 0]
    valid = indices >= 0
    selected = kv_values[np.clip(indices, 0, kv_values.shape[0] - 1)]
    logits = (q @ selected.T).astype(np.float32) * np.float32(512**-0.55)
    logits[:, ~valid] = -np.inf
    row_max = np.max(logits, axis=1)
    weights = np.exp(logits - row_max[:, None]).astype(np.float32)
    weights[:, ~valid] = np.float32(0)
    denominator = np.sum(weights, axis=1, dtype=np.float32)
    out = (weights @ selected).astype(np.float32) / denominator[:, None]
    lse = row_max + np.log(denominator).astype(np.float32)
    return {"k1:lse": lse, "k1:out": out.reshape(-1)}


def _sparse_prefill_inputs(
    *, s_q: int, s_kv: int, topk: int, d_qk: int, h_q: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct deterministic nonzero data that distinguishes every attention axis."""

    q = np.zeros((s_q, h_q, d_qk), dtype=np.float32)
    kv = np.zeros((s_kv, d_qk), dtype=np.float32)
    heads = np.arange(h_q, dtype=np.int32)
    rows = np.arange(s_kv, dtype=np.int32)
    probe_dims = (0, 1, 257, d_qk - 1)
    for query in range(s_q):
        q[query, :, probe_dims[0]] = ((heads + 2 * query) % 7 - 3) * np.float32(0.5)
        q[query, :, probe_dims[1]] = ((heads // 3 + query) % 5 - 2) * np.float32(0.5)
        q[query, :, probe_dims[2]] = ((heads // 5 + 2 * query) % 3 - 1) * np.float32(0.75)
        q[query, :, probe_dims[3]] = ((heads // 7 + query) % 5 - 2) * np.float32(0.25)

    kv[:, probe_dims[0]] = (rows % 9 - 4) * np.float32(0.5)
    kv[:, probe_dims[1]] = ((rows // 2) % 7 - 3) * np.float32(0.5)
    kv[:, probe_dims[2]] = ((rows // 3) % 5 - 2) * np.float32(0.75)
    kv[:, probe_dims[3]] = ((rows // 5) % 7 - 3) * np.float32(0.25)
    # These dimensions do not participate in the selected Q probes, so they
    # independently identify which V rows and output columns were combined.
    kv[:, 2] = (rows % 11 - 5) * np.float32(0.25)
    kv[:, 3] = ((rows * 3) % 13 - 6) * np.float32(0.25)
    kv[:, 17] = ((rows * 5) % 9 - 4) * np.float32(0.5)

    indices = np.empty((s_q, topk), dtype=np.int32)
    for query in range(s_q):
        indices[query] = (np.arange(topk, dtype=np.int32) * 5 + 3 + 7 * query) % s_kv
        if topk >= 18:
            indices[query, 5] = -1
            indices[query, 17] = s_kv
    return _encode_bfloat16(q), _encode_bfloat16(kv), indices


def _sparse_prefill_reference(
    q_bits: np.ndarray,
    kv_bits: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Independent NumPy oracle for sparse attention phase 1."""

    q = _decode_bfloat16(q_bits)
    kv = _decode_bfloat16(kv_bits)
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    out = np.zeros((s_q, h_q, 512), dtype=np.float32)
    max_logits = np.empty((s_q, h_q), dtype=np.float32)
    lse = np.empty((s_q, h_q), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(d_qk))

    for query in range(s_q):
        row_indices = indices[query]
        valid = (row_indices >= 0) & (row_indices < s_kv)
        safe = np.clip(row_indices, 0, s_kv - 1)
        selected = kv[safe]
        logits = (q[query] @ selected.T).astype(np.float32) * scale
        logits[:, ~valid] = -np.inf
        row_max = np.max(logits, axis=1)
        weights = np.exp(logits - row_max[:, None]).astype(np.float32)
        weights[:, ~valid] = np.float32(0)
        denominator = np.sum(weights, axis=1, dtype=np.float32)
        out[query] = (weights @ selected[:, :512]).astype(np.float32) / denominator[:, None]
        max_logits[query] = row_max
        lse[query] = row_max + np.log(denominator).astype(np.float32)
    return {"out": out, "max_logits": max_logits, "lse": lse}


def prepare_sparse_prefill_case(
    module_name: str,
    *,
    s_q: int,
    s_kv: int,
    topk: int,
    d_qk: int,
    h_q: int,
) -> NumSimCase:
    """Exercise sparse gather, QK, masking, softmax, and V aggregation."""

    module = load_tirx_kernel(module_name)
    q, kv_matrix, indices_matrix = _sparse_prefill_inputs(
        s_q=s_q, s_kv=s_kv, topk=topk, d_qk=d_qk, h_q=h_q
    )
    kv = kv_matrix.reshape(-1)
    indices = indices_matrix.reshape(-1)
    attn_sink = np.zeros((h_q,), dtype=np.float32)
    topk_length = np.full((s_q,), topk, dtype=np.int32)
    out = np.full((s_q, h_q, 512), np.uint16(0x7FC0), dtype=np.uint16)
    max_logits = np.full((s_q, h_q), np.nan, dtype=np.float32)
    lse = np.full((s_q, h_q), np.nan, dtype=np.float32)

    kernel = module.get_kernel(
        s_q=s_q,
        s_kv=s_kv,
        topk=topk,
        d_qk=d_qk,
        h_q=h_q,
        have_attn_sink=False,
        have_topk_length=False,
    )
    args = {
        "q": q,
        "kv": kv,
        "indices": indices,
        "attn_sink": attn_sink,
        "topk_length": topk_length,
        "out": out,
        "max_logits": max_logits,
        "lse": lse,
    }
    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs=("out", "max_logits", "lse"),
        reference=lambda: _sparse_prefill_reference(q, kv_matrix, indices_matrix),
        comparisons={
            "out": ComparisonSpec(rtol=4.01 / 128, atol=5e-3, actual_encoding="bfloat16"),
            "max_logits": ComparisonSpec(rtol=2.01 / 65536, atol=1e-6),
            "lse": ComparisonSpec(rtol=2.01 / 65536, atol=1e-6),
        },
    )


def _specialize_runtime_scalars(kernel: Any, values: dict[str, int | float]) -> Any:
    substitutions = {param: values[param.name] for param in kernel.params if param.name in values}
    return kernel.specialize(substitutions)


def prepare_sparse_decode_head64_case() -> NumSimCase:
    """Exercise nonzero MODEL1 attention, ring reuse, and a two-way split combine."""

    module = load_tirx_kernel("sparse_flashmla_decode_head64")
    # The public loader derives this launch dimension from the device.  Give it
    # a deterministic two-SM topology so the CPU-owned case exercises two real
    # split rows without requiring CUDA during test discovery or execution.
    with (
        patch.object(module.torch.cuda, "is_available", return_value=True),
        patch.object(
            module.torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=2),
        ),
    ):
        kernels = module.get_kernel(
            model_type="MODEL1",
            b=1,
            s_q=1,
            s_kv=256,
            topk=256,
            page_block_size=64,
            device="cuda:0",
        )
    main_scalars = {
        "sm_scale_div_log2": 512**-0.55 * math.log2(math.e),
        "stride_q_b": 32768,
        "stride_q_s_q": 32768,
        "stride_q_h_q": 512,
        "stride_kv_block": 37440,
        "stride_kv_row": 584,
        "stride_indices_b": 256,
        "stride_indices_s_q": 256,
        "stride_lse_b": 64,
        "stride_lse_s_q": 64,
        "stride_o_b": 32768,
        "stride_o_s_q": 32768,
        "stride_o_h_q": 512,
        "stride_extra_kv_block": 0,
        "stride_extra_kv_row": 0,
        "tma_coords_step_per_block": 65,
        "tma_coords_step_per_extra_block": 0,
        "stride_extra_indices_b": 0,
        "stride_extra_indices_s_q": 0,
        "stride_lse_accum_split": 64,
        "stride_lse_accum_s_q": 64,
        "stride_o_accum_split": 32768,
        "stride_o_accum_s_q": 32768,
        "stride_o_accum_h_q": 512,
        "b": 1,
        "s_q": 1,
        "topk": 256,
        "extra_topk": 0,
        "num_blocks": 4,
        "extra_num_blocks": 0,
        "page_block_size": 64,
        "extra_page_block_size": 0,
        "num_sm_parts": 2,
    }
    combine_scalars = {
        key: main_scalars[key]
        for key in (
            "stride_lse_b",
            "stride_lse_s_q",
            "stride_o_b",
            "stride_o_s_q",
            "stride_o_h_q",
            "stride_lse_accum_split",
            "stride_lse_accum_s_q",
            "stride_o_accum_split",
            "stride_o_accum_s_q",
            "stride_o_accum_h_q",
            "b",
            "s_q",
            "num_sm_parts",
        )
    }
    combine_scalars.update(h_q=64, d_v=512)
    specialized = [
        _specialize_runtime_scalars(kernels[0], main_scalars),
        _specialize_runtime_scalars(kernels[1], combine_scalars),
    ]

    q, kv, indices, kv_values = _sparse_decode_inputs()
    lse = np.full((64,), np.nan, dtype=np.float32)
    out = np.full((32768,), np.uint16(0x7FC0), dtype=np.uint16)
    lse_accum = np.full((192,), np.nan, dtype=np.float32)
    o_accum = np.full((98304,), np.nan, dtype=np.float32)
    # The first CTA consumes three consecutive blocks, so its two-stage SMEM
    # ring has a real stage-0 -> stage-1 -> stage-0 lifetime.  The second CTA
    # owns the final block, retaining an independently combined split row.
    scheduler = np.asarray([[0, 0, 0, 3, 0, 1, 1, 0], [0, 0, 3, 4, 1, 1, 1, 0]], dtype=np.int32)
    num_splits = np.asarray([0, 2], dtype=np.int32)
    topk_length = np.zeros((1,), dtype=np.int32)
    attn_sink = np.zeros((64,), dtype=np.float32)

    q_binding = q.reshape(-1)
    kv_binding = kv.reshape(-1)
    kv_nope_binding = kv.view(np.int64).reshape(-1)
    kv_rope_binding = kv.reshape(-1)[224:]
    indices_binding = indices
    topk_length_binding = topk_length
    attn_sink_binding = attn_sink
    lse_binding = lse
    out_binding = out
    lse_accum_binding = lse_accum
    o_accum_binding = o_accum
    scheduler_binding = scheduler.reshape(-1)
    num_splits_binding = num_splits

    kv_rope_tensormap = TensorMap(
        base=kv_rope_binding,
        dtype="bfloat16",
        global_shape=(64, 260),
        global_strides=(576,),
        box_shape=(64, 1),
        element_strides=(1, 1),
        swizzle="128B",
        interleave=None,
        fill_mode="none",
    ).numpy()
    kv_nope_tensormap = TensorMap(
        base=kv_nope_binding,
        dtype="int64",
        global_shape=(56, 260),
        global_strides=(576,),
        box_shape=(56, 1),
        element_strides=(1, 1),
        swizzle=None,
        interleave=None,
        fill_mode="none",
    ).numpy()
    q_strided_tensormap = TensorMap(
        base=q_binding,
        dtype="bfloat16",
        global_shape=(512, 64, 1, 1),
        global_strides=(1024, 65536, 65536),
        box_shape=(64, 64, 1, 1),
        element_strides=(1, 1, 1, 1),
        swizzle="128B",
        interleave=None,
        fill_mode="none",
    ).numpy()
    out_tensormap = TensorMap(
        base=out_binding,
        dtype="bfloat16",
        global_shape=(512, 64, 1, 1),
        global_strides=(1024, 65536, 65536),
        box_shape=(64, 64, 1, 1),
        element_strides=(1, 1, 1, 1),
        swizzle="128B",
        interleave=None,
        fill_mode="none",
    ).numpy()

    # Both phases bind views of the same host allocations; NumSim derives the
    # device-side aliasing from their overlapping backing-address ranges.
    args = {
        "k0:q": q_binding,
        "k0:kv": kv_binding,
        "k0:indices": indices_binding,
        "k0:topk_length": topk_length_binding,
        "k0:attn_sink": attn_sink_binding,
        "k0:lse": lse_binding,
        "k0:out": out_binding,
        "k0:lse_accum": lse_accum_binding,
        "k0:o_accum": o_accum_binding,
        "k0:tile_scheduler_metadata": scheduler_binding,
        "k0:num_splits": num_splits_binding,
        "k0:extra_kv": kv_binding,
        "k0:extra_indices": indices_binding,
        "k0:extra_topk_length": topk_length_binding,
        "k0:kv_rope_tensormap": kv_rope_tensormap,
        "k0:kv_nope_tensormap": kv_nope_tensormap,
        "k0:extra_kv_rope_tensormap": kv_rope_tensormap,
        "k0:extra_kv_nope_tensormap": kv_nope_tensormap,
        "k0:q_strided_tensormap": q_strided_tensormap,
        "k0:out_tensormap": out_tensormap,
        "k1:lse": lse_binding,
        "k1:out": out_binding,
        "k1:lse_accum": lse_accum_binding,
        "k1:o_accum": o_accum_binding,
        "k1:num_splits": num_splits_binding,
        "k1:attn_sink": attn_sink_binding,
    }
    return NumSimCase(
        kernel=specialized,
        args=args,
        outputs=("k1:lse", "k1:out"),
        reference=lambda: _sparse_decode_reference(q, kv_values, indices),
        comparisons={
            "k1:lse": ComparisonSpec(rtol=8.01 / 65536, atol=1e-6),
            "k1:out": ComparisonSpec(rtol=2.01 / 128, atol=1e-3, actual_encoding="bfloat16"),
        },
    )


__all__ = ["prepare_sparse_decode_head64_case", "prepare_sparse_prefill_case"]
