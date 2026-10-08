"""NumSim-owned numerical corpus for the DeepGEMM kernel families."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any
from unittest.mock import patch

import numpy as np
from tvm import tirx
from tvm.ir import Call
from tvm_ffi import structural_map

from tests.numsim.support._tirx_kernels import config_params, load_tirx_kernel

_E2M1_POSITIVE_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_E2M1_SAMPLE_CODES = np.array(
    [0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF], dtype=np.uint8
)
_E4M3_SAMPLE_CODES = np.array(
    [
        0x00,
        0x18,
        0x20,
        0x28,
        0x30,
        0x34,
        0x38,
        0x3C,
        0x40,
        0x98,
        0xA0,
        0xA8,
        0xB0,
        0xB4,
        0xB8,
        0xBC,
        0xC0,
    ],
    dtype=np.uint8,
)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _e2m1_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8) & np.uint8(0xF)
    magnitude = _E2M1_POSITIVE_VALUES[(bits & np.uint8(0x7)).astype(np.intp)]
    return np.where((bits & np.uint8(0x8)) != 0, -magnitude, magnitude).astype(np.float32)


def _pack_e2m1(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if values.shape[-1] % 2 != 0:
        raise ValueError("E2M1 packing requires an even final dimension")
    return (
        (values[..., 0::2] & np.uint8(0xF)) | ((values[..., 1::2] & np.uint8(0xF)) << np.uint8(4))
    ).astype(np.uint8)


def _unpack_e2m1(values: Any) -> np.ndarray:
    packed = np.asarray(values, dtype=np.uint8)
    unpacked = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    unpacked[..., 0::2] = packed & np.uint8(0xF)
    unpacked[..., 1::2] = packed >> np.uint8(4)
    return unpacked


def _e4m3fn_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (bits & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1.0) + mantissa / np.float32(8.0), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8.0), -6)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    result = np.where((bits & np.uint8(0x80)) != 0, -magnitude, magnitude)
    is_nan = (exponent == 0xF) & ((bits & np.uint8(0x7)) == np.uint8(0x7))
    return np.where(is_nan, np.float32(np.nan), result).astype(np.float32)


def _pack_e8m0_words(exponents: Any) -> np.ndarray:
    exponents = np.asarray(exponents, dtype=np.int16)
    if exponents.shape[-1] != 4 or np.any(exponents < -127) or np.any(exponents > 127):
        raise ValueError("packed E8M0 words require four exponents in [-127, 127]")
    codes = (exponents + np.int16(127)).astype(np.uint32)
    return (
        codes[..., 0]
        | (codes[..., 1] << np.uint32(8))
        | (codes[..., 2] << np.uint32(16))
        | (codes[..., 3] << np.uint32(24))
    ).astype(np.uint32)


def _unpack_e8m0_words(words: Any) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint32)[..., None]
    shifts = np.arange(4, dtype=np.uint32) * np.uint32(8)
    return ((words >> shifts) & np.uint32(0xFF)).astype(np.uint8)


def _e8m0_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8)
    exponent = np.where(bits == np.uint8(0xFF), 0, bits.astype(np.int16) - 127)
    result = np.ldexp(np.ones(bits.shape, dtype=np.float32), exponent)
    return np.where(bits == np.uint8(0xFF), np.float32(np.nan), result).astype(np.float32)


def _float32_to_bfloat16_bits(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _round_to_bfloat16_float32(values: Any) -> np.ndarray:
    bits = _float32_to_bfloat16_bits(values).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _decode_fp4_rows(packed: np.ndarray, scale_words: np.ndarray) -> np.ndarray:
    values = _e2m1_bits_to_float32(_unpack_e2m1(packed))
    if values.shape[-1] != 128:
        raise ValueError("FP4 MQA NumSim cases require head_dim=128")
    scales = _e8m0_bits_to_float32(_unpack_e8m0_words(scale_words))
    decoded = (values.reshape(*values.shape[:-1], 4, 32) * scales[..., :, None]).reshape(
        values.shape
    )
    return _round_to_bfloat16_float32(decoded)


def _dense_ranges(
    seq_len: int, seq_len_kv: int, *, disable_cp: bool
) -> tuple[np.ndarray, np.ndarray]:
    if disable_cp:
        starts = np.zeros(seq_len, dtype=np.int32)
        ends = np.arange(seq_len, dtype=np.int32) + np.int32(seq_len_kv - seq_len)
        return starts, ends
    if seq_len_kv % seq_len != 0 or seq_len % 2 != 0:
        raise ValueError("cooperative schedule requires divisible KV length and even query length")
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    starts = np.zeros(seq_len, dtype=np.int32)
    ends = np.empty(seq_len, dtype=np.int32)
    offsets = np.arange(chunk_size, dtype=np.int32)
    ends[:chunk_size] = np.int32(cp_id * chunk_size) + offsets
    ends[chunk_size:] = np.int32((cp_size * 2 - 1 - cp_id) * chunk_size) + offsets
    return starts, ends


def _dense_mqa_reference(q: Any, kv: Any, weights: Any, starts: Any, ends: Any) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    kv = np.asarray(kv, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int32)
    ends = np.asarray(ends, dtype=np.int32)
    if q.ndim != 3 or kv.ndim != 2 or weights.shape != q.shape[:2]:
        raise ValueError("invalid dense MQA reference shapes")
    output = np.full((q.shape[0], kv.shape[0]), -np.inf, dtype=np.float32)
    columns = np.arange(kv.shape[0], dtype=np.int32)
    for row_start in range(0, q.shape[0], 32):
        row_end = min(row_start + 32, q.shape[0])
        scores = np.einsum("mhd,nd->hmn", q[row_start:row_end], kv, optimize=True, dtype=np.float32)
        logits = np.sum(
            np.maximum(scores, np.float32(0.0)) * weights[row_start:row_end].T[:, :, None],
            axis=0,
            dtype=np.float32,
        )
        mask = (columns[None, :] >= starts[row_start:row_end, None]) & (
            columns[None, :] < ends[row_start:row_end, None]
        )
        output[row_start:row_end] = np.where(mask, logits, -np.inf)
    return output


def _encode_logits(values: np.ndarray, logits_dtype: str) -> np.ndarray:
    if logits_dtype == "float32":
        return np.asarray(values, dtype=np.float32)
    if logits_dtype == "bfloat16":
        return _float32_to_bfloat16_bits(values)
    raise ValueError(f"unsupported logits dtype: {logits_dtype}")


def _empty_logits(shape: tuple[int, ...], logits_dtype: str) -> np.ndarray:
    if logits_dtype == "float32":
        return np.full(shape, -np.inf, dtype=np.float32)
    if logits_dtype == "bfloat16":
        return np.full(shape, np.uint16(0xFF80), dtype=np.uint16)
    raise ValueError(f"unsupported logits dtype: {logits_dtype}")


def _tensor_map(
    array: np.ndarray,
    *,
    dtype: str,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
    swizzle: str | None,
    fp4_shared_layout: str | None = None,
):
    """Bind a public TensorMap parameter to its physical backing array."""
    from tirx_harness.numsim.cases import TensorMap

    return TensorMap(
        base=array,
        dtype="float4_e2m1fn" if fp4_shared_layout is not None else dtype,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
        fp4_shared_layout=fp4_shared_layout,
        swizzle=swizzle,
        interleave=None,
        fill_mode="none",
    ).numpy()


def _inline_wrelu_reduce(kernel: Any, num_heads: int) -> Any:
    """Inline the package's private expression helper for NumSim execution."""

    def replace(node: Any) -> Any:
        if type(node).__name__ != "Call" or type(getattr(node, "op", None)).__name__ != "GlobalVar":
            return node
        if "wrelu_reduce" not in str(node.op) or len(node.args) != 2:
            return node
        accum_load = node.args[0].args[0]
        weights_load = node.args[1].args[0]
        if type(accum_load).__name__ != "TensorLoad" or type(weights_load).__name__ != "TensorLoad":
            raise ValueError("wrelu_reduce arguments must be address-of TensorLoad expressions")
        zero = tirx.const(0.0, "float32")
        total = zero
        for head in range(num_heads):
            accum = tirx.BufferLoad(accum_load.source, [head])
            weight = tirx.BufferLoad(weights_load.source, [weights_load.indices[0], head])
            total = total + tirx.Select(accum > zero, accum, zero) * weight
        return total

    return kernel.with_body(structural_map(kernel.body, (Call, replace)))


def prepare_dense_mqa_case(config: Any, get_kernel: Any, *, input_format: str):
    """Build a CUDA-free dense/compressed MQA case from physical low-precision inputs."""
    from tirx_harness.numsim.cases import ComparisonRegion, ComparisonSpec, NumSimCase

    rng = np.random.default_rng(config.seed)
    q_shape = (config.seq_len, config.num_heads, config.head_dim)
    kv_shape = (config.seq_len_kv, config.head_dim)
    weights = rng.choice(
        np.array([-1.0, -0.5, 0.25, 0.75, 1.0], dtype=np.float32),
        size=(config.seq_len, config.num_heads),
    ).astype(np.float32)
    starts, ends = _dense_ranges(config.seq_len, config.seq_len_kv, disable_cp=config.disable_cp)

    if input_format == "fp8":
        q_data = rng.choice(_E4M3_SAMPLE_CODES, size=q_shape).astype(np.uint8)
        kv_data = rng.choice(_E4M3_SAMPLE_CODES, size=kv_shape).astype(np.uint8)
        kv_scales = np.ldexp(
            np.ones(config.seq_len_kv, dtype=np.float32),
            rng.integers(-2, 2, size=config.seq_len_kv, dtype=np.int16),
        )
        q_dequant = _round_to_bfloat16_float32(_e4m3fn_bits_to_float32(q_data))
        kv_dequant = _round_to_bfloat16_float32(
            _e4m3fn_bits_to_float32(kv_data) * kv_scales[:, None]
        )
        swizzle = {32: "32B", 64: "64B", 128: "128B"}[config.head_dim]
        kernel_inputs = {
            "kv_scales_map": _tensor_map(
                kv_scales,
                dtype="float32",
                global_shape=(config.seq_len_kv,),
                global_strides=(),
                box_shape=(config.block_kv,),
                swizzle=None,
            ),
            "kv_map": _tensor_map(
                kv_data,
                dtype="uint8",
                global_shape=(config.head_dim, config.seq_len_kv),
                global_strides=(config.head_dim,),
                box_shape=(config.head_dim, config.block_kv),
                swizzle=swizzle,
            ),
            "q_map": _tensor_map(
                q_data.reshape(config.seq_len * config.num_heads, config.head_dim),
                dtype="uint8",
                global_shape=(config.head_dim, config.seq_len * config.num_heads),
                global_strides=(config.head_dim,),
                box_shape=(config.head_dim, config.block_q * config.num_heads),
                swizzle=swizzle,
            ),
        }
    elif input_format == "fp4":
        q_codes = rng.choice(_E2M1_SAMPLE_CODES, size=q_shape).astype(np.uint8)
        kv_codes = rng.choice(_E2M1_SAMPLE_CODES, size=kv_shape).astype(np.uint8)
        q_scale_words = _pack_e8m0_words(
            rng.integers(-2, 2, size=(*q_shape[:-1], 4), dtype=np.int16)
        )
        kv_scale_words = _pack_e8m0_words(
            rng.integers(-2, 2, size=(config.seq_len_kv, 4), dtype=np.int16)
        )
        q_data = _pack_e2m1(q_codes)
        kv_data = _pack_e2m1(kv_codes)
        q_dequant = _decode_fp4_rows(q_data, q_scale_words)
        kv_dequant = _decode_fp4_rows(kv_data, kv_scale_words)
        packed = config.head_dim // 2
        kernel_inputs = {
            "sf_kv_map": _tensor_map(
                kv_scale_words.reshape(1, config.seq_len_kv),
                dtype="uint32",
                global_shape=(config.seq_len_kv,),
                global_strides=(),
                box_shape=(config.block_kv,),
                swizzle=None,
            ),
            "kv_map": _tensor_map(
                kv_data,
                dtype="uint8",
                global_shape=(config.head_dim, config.seq_len_kv),
                global_strides=(packed,),
                box_shape=(config.head_dim, config.block_kv),
                swizzle="64B",
                fp4_shared_layout="align8_packed",
            ),
            "sf_q_map": _tensor_map(
                q_scale_words,
                dtype="uint32",
                global_shape=(config.num_heads, config.seq_len),
                global_strides=(config.num_heads * 4,),
                box_shape=(config.num_heads, config.block_q),
                swizzle=None,
            ),
            "q_map": _tensor_map(
                q_data.reshape(config.seq_len * config.num_heads, packed),
                dtype="uint8",
                global_shape=(config.head_dim, config.seq_len * config.num_heads),
                global_strides=(packed,),
                box_shape=(config.head_dim, config.block_q * config.num_heads),
                swizzle="64B",
                fp4_shared_layout="align8_packed",
            ),
        }
    else:
        raise ValueError(f"unknown MQA input format: {input_format}")

    kernel_inputs["weights_map"] = _tensor_map(
        weights,
        dtype="float32",
        global_shape=(config.num_heads, config.seq_len),
        global_strides=(config.num_heads * 4,),
        box_shape=(config.num_heads, config.block_q),
        swizzle=None,
    )
    expected = _encode_logits(
        _dense_mqa_reference(q_dequant, kv_dequant, weights, starts, ends), config.logits_dtype
    )
    max_seqlen_k = int(np.max(ends - starts)) if config.compressed_logits else 0
    logits_stride = (
        _align_up(max_seqlen_k, config.block_kv)
        if config.compressed_logits
        else _align_up(config.seq_len_kv + config.block_kv, 8)
    )
    logits = _empty_logits((config.aligned_seq_len, logits_stride), config.logits_dtype)
    kernel_kwargs = asdict(config)
    if config.compressed_logits:
        kernel_kwargs["logits_stride_override"] = logits_stride
    kernel = get_kernel(**kernel_kwargs)
    # The pinned FP4 MQA package returns a module containing a private helper
    # plus the public ``main`` entry.  NumSim launches one public PrimFunc at a
    # time, so select that entry without modifying the canonical source.
    if type(kernel).__name__ == "IRModule":
        kernel = kernel["main"]
    kernel = _inline_wrelu_reduce(kernel, config.num_heads)
    args = {
        "seq_len": np.uint32(config.seq_len),
        "seq_len_kv": np.uint32(config.seq_len_kv),
        "max_seqlen_k": np.uint32(max_seqlen_k),
        "logits_stride": np.uint32(logits_stride),
        "cu_seq_len_k_start": starts,
        "cu_seq_len_k_end": ends,
        "logits_flat": logits.reshape(-1),
        **kernel_inputs,
    }
    if input_format == "fp8":
        args["num_q_blocks"] = np.uint32(config.aligned_seq_len // config.block_q)

    regions = tuple(
        ComparisonRegion(
            actual=(
                slice(
                    row * logits_stride + (0 if config.compressed_logits else int(start)),
                    row * logits_stride
                    + (int(end - start) if config.compressed_logits else int(end)),
                ),
            ),
            expected=(
                slice(
                    row * config.seq_len_kv + int(start),
                    row * config.seq_len_kv + int(end),
                ),
            ),
        )
        for row, (start, end) in enumerate(zip(starts, ends))
    )

    tolerance = (
        ComparisonSpec(rtol=2e-4, atol=5e-4, regions=regions)
        if config.logits_dtype == "float32"
        else ComparisonSpec(rtol=0.0, atol=0.0, regions=regions)
    )
    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs={"logits": "logits_flat"},
        reference=lambda: {"logits": expected.reshape(-1).copy()},
        comparisons={"logits": tolerance},
    )


_E4M3_NAN_MANTISSA = 7


def float32_to_bfloat16_bits(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) >> np.uint32(16)).astype(np.uint16)


def bfloat16_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def e4m3fn_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8)
    sign = np.where((bits & np.uint8(0x80)) != 0, -1.0, 1.0).astype(np.float32)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (bits & np.uint8(0x7)).astype(np.int16)
    result = np.empty(bits.shape, dtype=np.float32)
    subnormal = exponent == 0
    result[subnormal] = np.ldexp(mantissa[subnormal].astype(np.float32) / 8.0, -6)
    normal = (exponent > 0) & (exponent < 15)
    result[normal] = np.ldexp(
        1.0 + mantissa[normal].astype(np.float32) / 8.0,
        exponent[normal].astype(np.int32) - 7,
    )
    finite_top = (exponent == 15) & (mantissa != _E4M3_NAN_MANTISSA)
    result[finite_top] = np.ldexp(1.0 + mantissa[finite_top].astype(np.float32) / 8.0, 8)
    result[(exponent == 15) & (mantissa == _E4M3_NAN_MANTISSA)] = np.nan
    return result * sign


def _build_e4m3_encoding_table() -> tuple[np.ndarray, np.ndarray]:
    codes = np.arange(256, dtype=np.uint8)
    values = e4m3fn_bits_to_float32(codes)
    entries: dict[float, int] = {}
    for code, value in zip(codes.tolist(), values.tolist()):
        if not np.isfinite(value):
            continue
        previous = entries.get(value)
        if previous is None or ((code & 1) == 0 and (previous & 1) != 0):
            entries[value] = code
    sorted_values = np.array(sorted(entries), dtype=np.float32)
    sorted_codes = np.array([entries[float(value)] for value in sorted_values], dtype=np.uint8)
    return sorted_values, sorted_codes


_E4M3_VALUES, _E4M3_CODES = _build_e4m3_encoding_table()


def float32_to_e4m3fn_bits(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    clipped = np.clip(values, _E4M3_VALUES[0], _E4M3_VALUES[-1])
    right = np.searchsorted(_E4M3_VALUES, clipped, side="left")
    right = np.clip(right, 0, len(_E4M3_VALUES) - 1)
    left = np.maximum(right - 1, 0)
    left_distance = np.abs(clipped - _E4M3_VALUES[left])
    right_distance = np.abs(_E4M3_VALUES[right] - clipped)
    choose_right = right_distance < left_distance
    ties = right_distance == left_distance
    right_even = (_E4M3_CODES[right] & np.uint8(1)) == 0
    left_even = (_E4M3_CODES[left] & np.uint8(1)) == 0
    choose_right |= ties & right_even & ~left_even
    selected = np.where(choose_right, right, left)
    result = _E4M3_CODES[selected]
    result = np.where(np.isnan(values), np.uint8(0x7F), result)
    return result.astype(np.uint8)


def _round_float32_to_tf32(values: Any) -> np.ndarray:
    """Round finite float32 operands to the 10-bit TF32 fraction."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    exponent = bits & np.uint32(0x7F800000)
    finite = exponent != np.uint32(0x7F800000)
    rounded = bits.copy()
    bias = np.uint32(0x00000FFF) + ((bits >> np.uint32(13)) & np.uint32(1))
    rounded[finite] = (bits[finite] + bias[finite]) & np.uint32(0xFFFFE000)
    return rounded.view(np.float32)


def _numpy_tf32_hc_reference(
    a_bits: np.ndarray, b: np.ndarray, *, num_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    a = bfloat16_bits_to_float32(a_bits)
    b = np.asarray(b, dtype=np.float32)
    m, k = a.shape
    n = b.shape[0]
    if b.shape[1] != k or k % 64 != 0:
        raise ValueError("TF32 HC reference requires matching K divisible by 64")
    num_k_blocks = k // 64
    blocks_per_split, remainder = divmod(num_k_blocks, num_splits)
    partial_d = np.zeros((num_splits, m, n), dtype=np.float32)
    partial_sqr = np.zeros((num_splits, m), dtype=np.float32)
    block_offset = 0
    for split in range(num_splits):
        split_blocks = blocks_per_split + int(split < remainder)
        start = block_offset * 64
        stop = start + split_blocks * 64
        a_slice = _round_float32_to_tf32(a[:, start:stop])
        b_slice = _round_float32_to_tf32(b[:, start:stop])
        partial_d[split] = a_slice @ b_slice.T
        partial_sqr[split] = np.sum(a[:, start:stop] * a[:, start:stop], axis=-1)
        block_offset += split_blocks
    if num_splits == 1:
        return partial_d[0], partial_sqr[0]
    return partial_d, partial_sqr


def prepare_tf32_hc_case(**kwargs: Any):
    """Build a deterministic CUDA-free TF32 HC prenorm GEMM launch."""
    tf32_hc_prenorm_gemm = load_tirx_kernel("deepgemm_sm100_tf32_hc_prenorm_gemm")

    from tirx_harness.numsim.cases import (
        ComparisonSpec,
        NumSimCase,
        TensorMap,
    )

    config = tf32_hc_prenorm_gemm._make_config(**kwargs)
    rng = np.random.default_rng(config.seed)
    a_values = rng.integers(-4, 5, size=(config.m, config.k), dtype=np.int16).astype(np.float32)
    a_values *= np.float32(0.125)
    a_bits = float32_to_bfloat16_bits(a_values)
    b = rng.standard_normal((config.n, config.k), dtype=np.float32) * np.float32(0.125)
    d = np.zeros(config.d_shape, dtype=np.float32)
    sqr_sum = np.zeros((config.num_splits * config.m,), dtype=np.float32)
    a_binding = a_bits.reshape(-1)
    b_binding = b.reshape(-1)
    d_binding = d.reshape(-1)
    sqr_binding = sqr_sum
    expected_d, expected_sqr = _numpy_tf32_hc_reference(a_bits, b, num_splits=config.num_splits)

    def tensor_map(
        base: np.ndarray,
        *,
        dtype: str,
        global_shape: tuple[int, ...],
        global_strides: tuple[int, ...],
        box_shape: tuple[int, ...],
        tma_dtype: str | None = None,
    ) -> np.ndarray:
        return TensorMap(
            base=base,
            dtype=dtype,
            global_shape=global_shape,
            global_strides=global_strides,
            box_shape=box_shape,
            element_strides=(1,) * len(global_shape),
            tma_dtype=tma_dtype,
            swizzle="128B",
            interleave=None,
            fill_mode="none",
        ).numpy()

    block_swizzled_bk = min(config.block_k * 4, 128) // 4
    d_map = tensor_map(
        d_binding,
        dtype="float32",
        global_shape=(config.n, config.m)
        if config.num_splits == 1
        else (config.n, config.m, config.num_splits),
        global_strides=(config.n * 4,)
        if config.num_splits == 1
        else (config.n * 4, config.m * config.n * 4),
        box_shape=(config.block_n, config.block_m)
        if config.num_splits == 1
        else (config.block_n, config.block_m, 1),
    )
    args = {
        "shape_m": np.uint32(config.m),
        "a": a_binding,
        "b": b_binding,
        "d": d_binding,
        "sqr_sum": sqr_binding,
        "a_map": tensor_map(
            a_binding,
            dtype="bfloat16",
            global_shape=(config.k, config.m),
            global_strides=(config.k * 2,),
            box_shape=(config.block_k, config.block_m),
        ),
        "b_map": tensor_map(
            b_binding,
            dtype="float32",
            global_shape=(config.k, config.n),
            global_strides=(config.k * 4,),
            box_shape=(block_swizzled_bk, config.block_n),
            tma_dtype="tf32",
        ),
        "d_map": d_map,
    }
    return NumSimCase(
        kernel=tf32_hc_prenorm_gemm.get_kernel(**kwargs),
        args=args,
        outputs={"D": "d", "sqr_sum": "sqr_sum"},
        reference=lambda: {
            "D": expected_d.reshape(-1).copy(),
            "sqr_sum": expected_sqr.reshape(-1).copy(),
        },
        comparisons={
            "D": ComparisonSpec(rtol=2e-4, atol=2e-4),
            "sqr_sum": ComparisonSpec(rtol=1e-6, atol=1e-6),
        },
    )


def prepare_fp4_mqa_case(**kwargs: Any):
    mqa_logits_fp4 = load_tirx_kernel("deepgemm_sm100_fp4_mqa_logits")

    return prepare_dense_mqa_case(
        mqa_logits_fp4._make_config(**kwargs), mqa_logits_fp4.get_kernel, input_format="fp4"
    )


def prepare_fp8_mqa_case(**kwargs: Any):
    mqa_logits_fp8 = load_tirx_kernel("deepgemm_sm100_fp8_mqa_logits")

    return prepare_dense_mqa_case(
        mqa_logits_fp8._make_config(**kwargs), mqa_logits_fp8.get_kernel, input_format="fp8"
    )


def _paged_mqa_reference(q: np.ndarray, kv: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reference weighted multi-head dot products in logical tensor order."""

    return np.einsum(
        "hd,td,h->t",
        np.asarray(q, dtype=np.float64),
        np.asarray(kv, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        optimize=False,
    ).astype(np.float32)


def prepare_paged_mqa_case(*, input_format: str):
    """Build a complete paged-MQA schedule with an independent numerical oracle."""
    from tirx_harness.numsim.cases import (
        ComparisonRegion,
        ComparisonSpec,
        NumSimCase,
    )

    if input_format not in {"fp4", "fp8"}:
        raise ValueError(f"unsupported paged MQA input format: {input_format}")
    module = load_tirx_kernel(f"deepgemm_sm100_{input_format}_paged_mqa_logits")
    kernel_kwargs = {
        "batch_size": 1,
        "next_n": 1,
        "max_num_pages": 1,
        "num_pages": 1,
        "num_heads": 64,
        "head_dim": 128,
        "page_size": 64,
        "logits_dtype": "float32",
        "num_sms": 2,
        "varlen": False,
    }
    if input_format == "fp8":
        kernel_kwargs["context_pattern"] = "random_2d"
    kernel = module.get_kernel(**kernel_kwargs)

    context_lens = np.asarray([[64]], dtype=np.int32)
    logits = np.full((1, 256), -np.inf, dtype=np.float32)
    block_table = np.asarray([[0]], dtype=np.int32)
    indices = np.asarray([0], dtype=np.int32)
    # One real task followed by the terminal cursor used by both persistent CTAs.
    schedule_meta = np.asarray([[0, 0], [1, 0], [1, 0]], dtype=np.int32)
    weights = np.resize(
        np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        (1, 64),
    )
    tensor_maps: dict[str, Any]
    if input_format == "fp8":
        exact_codes = np.asarray([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8)
        head = np.arange(64, dtype=np.intp)[:, None]
        dim = np.arange(128, dtype=np.intp)[None, :]
        token = np.arange(64, dtype=np.intp)[:, None]
        q = exact_codes[(head * 3 + dim) % exact_codes.size]
        kv_logical = exact_codes[(token + dim * 3 + 1) % exact_codes.size]
        kv = kv_logical[None, ...]
        kv_scales = np.ldexp(
            np.ones((1, 64), dtype=np.float32),
            (np.arange(64, dtype=np.int16) % 3) - 1,
        )
        q_decoded = _e4m3fn_bits_to_float32(q)
        kv_decoded = _e4m3fn_bits_to_float32(kv_logical) * kv_scales[0, :, None]
        tensor_maps = {
            "tensor_map_q": _tensor_map(
                q,
                dtype="float8_e4m3fn",
                global_shape=(128, 64),
                global_strides=(128,),
                box_shape=(128, 64),
                swizzle="128B",
            ),
            "tensor_map_kv": _tensor_map(
                kv,
                dtype="float8_e4m3fn",
                global_shape=(128, 64, 1),
                global_strides=(128, 8192),
                box_shape=(128, 64, 1),
                swizzle="128B",
            ),
            "tensor_map_kv_scales": _tensor_map(
                kv_scales,
                dtype="float32",
                global_shape=(64, 1),
                global_strides=(256,),
                box_shape=(64, 1),
                swizzle=None,
            ),
        }
    else:
        exact_codes = np.asarray([0x1, 0x2, 0x3, 0x4], dtype=np.uint8)
        head = np.arange(64, dtype=np.intp)[:, None]
        dim = np.arange(128, dtype=np.intp)[None, :]
        token = np.arange(64, dtype=np.intp)[:, None]
        q_codes = exact_codes[(head * 3 + dim) % exact_codes.size]
        kv_codes = exact_codes[(token + dim * 3 + 1) % exact_codes.size]
        q = _pack_e2m1(q_codes)
        kv = _pack_e2m1(kv_codes)[None, ...]
        block = np.arange(4, dtype=np.int16)[None, :]
        q_exponents = ((np.arange(64, dtype=np.int16)[:, None] + block) % 3) - 1
        kv_exponents = ((np.arange(64, dtype=np.int16)[:, None] * 2 + block) % 3) - 1
        q_scales = _pack_e8m0_words(q_exponents)[None, :]
        kv_scales = _pack_e8m0_words(kv_exponents)[None, :]
        q_decoded = (
            _e2m1_bits_to_float32(q_codes).reshape(64, 4, 32)
            * np.ldexp(np.ones((64, 4, 1), dtype=np.float32), q_exponents[..., None])
        ).reshape(64, 128)
        kv_decoded = (
            _e2m1_bits_to_float32(kv_codes).reshape(64, 4, 32)
            * np.ldexp(np.ones((64, 4, 1), dtype=np.float32), kv_exponents[..., None])
        ).reshape(64, 128)
        tensor_maps = {
            "tensor_map_q": _tensor_map(
                q,
                dtype="uint8",
                global_shape=(128, 64),
                global_strides=(64,),
                box_shape=(128, 64),
                swizzle="64B",
                fp4_shared_layout="align8_packed",
            ),
            "tensor_map_sf_q": _tensor_map(
                q_scales,
                dtype="uint32",
                global_shape=(64, 1),
                global_strides=(256,),
                box_shape=(64, 1),
                swizzle=None,
            ),
            "tensor_map_kv": _tensor_map(
                kv,
                dtype="uint8",
                global_shape=(128, 64, 1),
                global_strides=(64, 4096),
                box_shape=(128, 64, 1),
                swizzle="64B",
                fp4_shared_layout="align8_packed",
            ),
            "tensor_map_sf_kv": _tensor_map(
                kv_scales,
                dtype="uint32",
                global_shape=(64, 1),
                global_strides=(256,),
                box_shape=(64, 1),
                swizzle=None,
            ),
        }
    tensor_maps["tensor_map_weights"] = _tensor_map(
        weights,
        dtype="float32",
        global_shape=(64, 1),
        global_strides=(256,),
        box_shape=(64, 1),
        swizzle=None,
    )
    args = {
        "batch_size": np.uint32(1),
        "logits_stride": np.uint32(256),
        "block_table_stride": np.uint32(1),
        "context_lens_flat": context_lens.reshape(-1),
        "logits_flat": logits.reshape(-1),
        "block_table_flat": block_table.reshape(-1),
        "indices": indices,
        "schedule_meta_flat": schedule_meta.reshape(-1),
        **tensor_maps,
    }
    expected_logits = np.full_like(logits, -np.inf)
    expected_logits[0, :64] = _paged_mqa_reference(q_decoded, kv_decoded, weights[0])
    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs={"logits": "logits_flat"},
        reference=lambda: {"logits": expected_logits.reshape(-1).copy()},
        comparisons={
            # Paged MQA only defines logits below context_lens.  The 128-row
            # MMA tile may store padded rows, matching the canonical CUDA
            # harness which excludes reference -inf positions from its error.
            "logits": ComparisonSpec(
                rtol=0,
                atol=0,
                regions=(
                    ComparisonRegion(
                        actual=(slice(0, 64),),
                        expected=(slice(0, 64),),
                    ),
                ),
            )
        },
    )


def _mega_moe_interleave_l1_rows(values: np.ndarray) -> np.ndarray:
    """Encode DeepGEMM's eight-row gate/up interleave."""
    values = np.asarray(values)
    num_experts, num_rows, *tail = values.shape
    half = num_rows // 2
    gate = values[:, :half].reshape(num_experts, half // 8, 8, *tail)
    up = values[:, half:].reshape(num_experts, half // 8, 8, *tail)
    result = _aligned_empty(values.shape, dtype=values.dtype)
    np.stack((gate, up), axis=2, out=result.reshape(num_experts, half // 8, 2, 8, *tail))
    return result


def _mega_moe_scale_storage(exponents: np.ndarray, *, interleave_l1: bool) -> np.ndarray:
    """Pack natural per-32 UE8M0 scales into the public UTCCP/TMA ABI."""
    exponents = np.asarray(exponents, dtype=np.int16)
    num_experts, num_rows, num_scale_groups = exponents.shape
    if num_rows % 128 != 0 or num_scale_groups % 4 != 0:
        raise ValueError("MegaMoE scale storage requires 128 rows and four K groups per tile")
    packed_k = num_scale_groups // 4
    packed = _pack_e8m0_words(exponents.reshape(num_experts, num_rows, packed_k, 4))
    if interleave_l1:
        packed = _mega_moe_interleave_l1_rows(packed)
    packed = (
        packed.reshape(num_experts, num_rows // 128, 4, 32, packed_k)
        .swapaxes(2, 3)
        .reshape(num_experts, num_rows, packed_k)
    )
    return packed.transpose(0, 2, 1).copy().reshape(num_experts * packed_k, num_rows)


def _mega_moe_decode_weights(packed: np.ndarray, scale_exponents: np.ndarray) -> np.ndarray:
    codes = _unpack_e2m1(np.asarray(packed, dtype=np.uint8))
    scales = np.exp2(np.asarray(scale_exponents, dtype=np.float32)).astype(np.float32)
    return (
        _e2m1_bits_to_float32(codes).reshape(*codes.shape[:-1], -1, 32) * scales[..., None]
    ).reshape(codes.shape)


def _mega_moe_decode_fp8_weights(codes: np.ndarray, scale_exponents: np.ndarray) -> np.ndarray:
    """Decode natural shared-expert E4M3 weights with per-32 UE8M0 scales."""
    codes = np.asarray(codes, dtype=np.uint8)
    scales = np.exp2(np.asarray(scale_exponents, dtype=np.float32)).astype(np.float32)
    return (
        e4m3fn_bits_to_float32(codes).reshape(*codes.shape[:-1], -1, 32) * scales[..., None]
    ).reshape(codes.shape)


def _mega_moe_requantize_e4m3(values: np.ndarray) -> np.ndarray:
    """Mirror the observable per-32 UE8M0 requantization boundary."""
    groups = np.asarray(values, dtype=np.float32).reshape(-1, 32)
    amax = np.max(np.abs(groups), axis=1).astype(np.float32)
    scaled = (amax * np.float32(1.0 / 448.0)).astype(np.float32)
    bits = scaled.view(np.uint32)
    exponents = (
        ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
        - np.int32(127)
        + ((bits & np.uint32((1 << 23) - 1)) != 0).astype(np.int32)
    )
    scales = np.ldexp(np.ones(exponents.shape, dtype=np.float32), exponents)
    inverses = np.ldexp(np.ones(exponents.shape, dtype=np.float32), -exponents)
    quantized = float32_to_e4m3fn_bits(groups * inverses[:, None])
    return (e4m3fn_bits_to_float32(quantized) * scales[:, None]).reshape(values.shape)


def _mega_moe_reference(
    *,
    input_codes: np.ndarray,
    input_scale_exponents: np.ndarray,
    l1_packed: np.ndarray,
    l1_scale_exponents: np.ndarray,
    l2_packed: np.ndarray,
    l2_scale_exponents: np.ndarray,
    topk_idx: np.ndarray,
    topk_weights: np.ndarray,
    activation_clamp: float,
    shared_l1_codes: np.ndarray,
    shared_l1_scale_exponents: np.ndarray,
    shared_l2_codes: np.ndarray,
    shared_l2_scale_exponents: np.ndarray,
) -> np.ndarray:
    """Independent TP1 routed-FP4 plus shared-FP8 MoE reference."""
    input_values = (
        e4m3fn_bits_to_float32(input_codes).reshape(*input_codes.shape[:-1], -1, 32)
        * np.exp2(np.asarray(input_scale_exponents, dtype=np.float32))[..., None]
    )
    input_values = input_values.reshape(input_codes.shape).astype(np.float32)
    l1_weights = _mega_moe_decode_weights(l1_packed, l1_scale_exponents)
    l2_weights = _mega_moe_decode_weights(l2_packed, l2_scale_exponents)
    intermediate = l2_weights.shape[-1]
    clamp = _round_to_bfloat16_float32(np.asarray(activation_clamp, dtype=np.float32))
    combined = np.zeros((input_codes.shape[0], l2_weights.shape[1]), dtype=np.float32)

    for token_idx in range(input_codes.shape[0]):
        for topk_slot in range(topk_idx.shape[1]):
            expert_idx = int(topk_idx[token_idx, topk_slot])
            l1 = np.sum(
                l1_weights[expert_idx] * input_values[token_idx][None, :],
                axis=1,
                dtype=np.float32,
            )
            gate = np.minimum(_round_to_bfloat16_float32(l1[:intermediate]), clamp)
            up = np.clip(_round_to_bfloat16_float32(l1[intermediate:]), -clamp, clamp)
            denom = (np.float32(1.0) + np.exp(-gate)).astype(np.float32)
            activated = (gate * (np.float32(1.0) / denom)).astype(np.float32)
            activated = (activated * up).astype(np.float32)
            activated = (activated * np.float32(topk_weights[token_idx, topk_slot])).astype(
                np.float32
            )
            l2_input = _mega_moe_requantize_e4m3(activated)
            l2 = np.sum(
                l2_weights[expert_idx] * l2_input[None, :],
                axis=1,
                dtype=np.float32,
            )
            combined[token_idx] = (combined[token_idx] + _round_to_bfloat16_float32(l2)).astype(
                np.float32
            )

    shared_l1_weights = _mega_moe_decode_fp8_weights(shared_l1_codes, shared_l1_scale_exponents)
    shared_l2_weights = _mega_moe_decode_fp8_weights(shared_l2_codes, shared_l2_scale_exponents)
    shared_intermediate = shared_l2_weights.shape[-1]
    for token_idx in range(input_codes.shape[0]):
        l1 = np.sum(
            shared_l1_weights * input_values[token_idx][None, :],
            axis=1,
            dtype=np.float32,
        )
        gate = np.minimum(_round_to_bfloat16_float32(l1[:shared_intermediate]), clamp)
        up = np.clip(_round_to_bfloat16_float32(l1[shared_intermediate:]), -clamp, clamp)
        denom = (np.float32(1.0) + np.exp(-gate)).astype(np.float32)
        activated = (gate * (np.float32(1.0) / denom)).astype(np.float32)
        activated = (activated * up).astype(np.float32)
        l2_input = _mega_moe_requantize_e4m3(activated)
        l2 = np.sum(shared_l2_weights * l2_input[None, :], axis=1, dtype=np.float32)
        combined[token_idx] = (combined[token_idx] + _round_to_bfloat16_float32(l2)).astype(
            np.float32
        )
    return _round_to_bfloat16_float32(combined)


def _aligned_empty(shape: tuple[int, ...], *, dtype: np.dtype[Any]) -> np.ndarray:
    # CUDA's 16U4_ALIGN16B TensorMaps require a 32-byte global address.
    # Keep the actual NumPy backing aligned, not a fabricated descriptor address.
    byte_len = int(np.prod(shape)) * dtype.itemsize
    owner = np.empty(byte_len + 31, dtype=np.uint8)
    offset = -owner.ctypes.data % 32
    return owner[offset : offset + byte_len].view(dtype).reshape(shape)


def _filled_array(shape: tuple[int, ...], *, dtype: np.dtype[Any], byte: int) -> np.ndarray:
    result = _aligned_empty(shape, dtype=dtype)
    result.view(np.uint8).reshape(-1).fill(byte)
    return result


def prepare_mega_moe_case(config_entry: Mapping[str, Any] | None = None):
    """Run one dense TP1 config through dispatch, both GEMMs, activation, and combine.

    Registry configurations exercise every route with dense nonzero expert
    weights. Exact-zero inputs keep their independent output oracle cheap even
    for the maximum public shape; the default numerical case remains nonzero.
    """
    from tirx_kernels.ported.deepgemm._sm100_fp8_fp4_mega_moe import spec as mega_moe_spec

    from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase, TensorMap

    module = load_tirx_kernel("sm100_fp8_fp4_mega_moe")
    kwargs = (
        {
            "num_processes": 1,
            "num_max_tokens_per_rank": 4,
            "num_tokens": 2,
            "hidden": 256,
            "intermediate_hidden": 256,
            "num_experts": 2,
            "num_topk": 2,
            "num_shared_experts": 1,
            "activation_clamp": 1.0,
            "fast_math": 1,
        }
        if config_entry is None
        else config_params(config_entry)
    )
    if kwargs["num_processes"] != 1:
        raise ValueError("the MegaMoE NumSim corpus currently covers TP=1 only")
    launch_environment = (
        patch.dict(os.environ, {"TIRX_DEEPGEMM_NUM_SMS_OVERRIDE": "4"})
        if config_entry is None
        else nullcontext()
    )
    with launch_environment:
        config = module.MegaMoeConfig(**kwargs)
        launch = mega_moe_spec.get_deepgemm_launch_config(config)
        workspace = mega_moe_spec.get_deepgemm_workspace_layout(config)
        layout = mega_moe_spec.get_deepgemm_symm_buffer_layout(config)
        kernel = module.get_kernel(
            **kwargs,
            collect_stats=True,
            emit_nvl_barrier_timeout_printf=False,
        )
    symm = np.zeros((layout.total_bytes,), dtype=np.int8)
    symm_u8 = symm.view(np.uint8)

    rng = np.random.default_rng(4700)
    fp8_codes = np.array([0x20, 0x28, 0x30, 0x34, 0x38, 0xA0, 0xA8, 0xB0, 0xB4, 0xB8])
    if config_entry is None:
        input_codes = rng.choice(fp8_codes, size=(config.num_tokens, config.hidden)).astype(
            np.uint8
        )
        input_scale_exponents = rng.integers(
            -1, 2, size=(config.num_tokens, config.hidden // 32), dtype=np.int16
        )
    else:
        input_codes = np.zeros((config.num_tokens, config.hidden), dtype=np.uint8)
        input_scale_exponents = np.zeros((config.num_tokens, config.hidden // 32), dtype=np.int16)
    input_bytes = workspace.num_max_tokens_per_rank * config.hidden
    input_region = symm_u8[layout.input_token_offset : layout.input_token_offset + input_bytes]
    input_region.reshape(workspace.num_max_tokens_per_rank, config.hidden)[: config.num_tokens] = (
        input_codes
    )
    input_sf_bytes = workspace.num_max_tokens_per_rank * (config.hidden // 32)
    input_sf_region = symm_u8[
        layout.input_sf_offset : layout.input_sf_offset + input_sf_bytes
    ].reshape(workspace.num_max_tokens_per_rank, config.hidden // 32)
    input_sf_region[: config.num_tokens] = (input_scale_exponents + 127).astype(np.uint8)

    if config_entry is None:
        routes = np.array([[0, 1], [1, 0]], dtype=np.int64)
        route_weights = np.array([[0.5, 1.25], [0.75, 0.25]], dtype=np.float32)
    else:
        route_count = config.num_tokens * config.num_topk
        routes = np.arange(route_count, dtype=np.int64).reshape(config.num_tokens, config.num_topk)
        routes %= config.num_experts_per_rank
        routes[-1, -1] = config.num_experts_per_rank - 1
        route_weights = (
            (np.arange(route_count, dtype=np.float32) % np.float32(5.0)) + np.float32(1.0)
        ).reshape(config.num_tokens, config.num_topk) * np.float32(0.25)
    topk_idx = (
        symm_u8[
            layout.input_topk_idx_offset : layout.input_topk_idx_offset
            + workspace.num_max_tokens_per_rank * config.num_topk * 8
        ]
        .view(np.int64)
        .reshape(workspace.num_max_tokens_per_rank, config.num_topk)
    )
    topk_idx[: config.num_tokens] = routes
    topk_weights = (
        symm_u8[
            layout.input_topk_weights_offset : layout.input_topk_weights_offset
            + workspace.num_max_tokens_per_rank * config.num_topk * 4
        ]
        .view(np.float32)
        .reshape(workspace.num_max_tokens_per_rank, config.num_topk)
    )
    topk_weights[: config.num_tokens] = route_weights

    def symm_view(offset: int, size: int, *, dtype: str) -> np.ndarray:
        array = symm_u8[offset : offset + size]
        if dtype == "int32":
            array = array.view(np.int32)
        return array

    def tensor_map(
        base: np.ndarray,
        *,
        dtype: str,
        global_shape: tuple[int, ...],
        global_strides: tuple[int, ...],
        box_shape: tuple[int, ...],
        swizzle: str | None,
        fp4_shared_layout: str | None = None,
    ) -> np.ndarray:
        return TensorMap(
            base=base,
            dtype="float4_e2m1fn" if fp4_shared_layout is not None else dtype,
            global_shape=global_shape,
            global_strides=global_strides,
            box_shape=box_shape,
            element_strides=(1,) * len(global_shape),
            fp4_shared_layout=fp4_shared_layout,
            swizzle=swizzle,
            interleave=None,
            fill_mode="none",
        ).numpy()

    sf_block_m = _align_up(launch.block_m, 128)
    l1_acts_size = workspace.num_ring_tokens * config.hidden
    l1_acts_sf_size = workspace.num_sf_ring_tokens * (config.hidden // 32)
    l2_acts_size = workspace.num_ring_tokens * config.intermediate_hidden
    l2_acts_sf_size = workspace.num_sf_ring_tokens * (config.intermediate_hidden // 32)
    if config_entry is None:
        fp4_codes = np.array([0x1, 0x2, 0x3, 0x9, 0xA, 0xB], dtype=np.uint8)
        l1_codes = rng.choice(
            fp4_codes,
            size=(config.num_experts_per_rank, config.intermediate_hidden * 2, config.hidden),
        ).astype(np.uint8)
        l2_codes = rng.choice(
            fp4_codes,
            size=(config.num_experts_per_rank, config.hidden, config.intermediate_hidden),
        ).astype(np.uint8)
        l1_scale_exponents = rng.integers(
            -3,
            0,
            size=(
                config.num_experts_per_rank,
                config.intermediate_hidden * 2,
                config.hidden // 32,
            ),
            dtype=np.int16,
        )
        l2_scale_exponents = rng.integers(
            -4,
            -1,
            size=(
                config.num_experts_per_rank,
                config.hidden,
                config.intermediate_hidden // 32,
            ),
            dtype=np.int16,
        )
        l1_packed = _pack_e2m1(l1_codes)
        l2_packed = _pack_e2m1(l2_codes)
        l1_weights = _mega_moe_interleave_l1_rows(l1_packed)
        l2_weights = _aligned_empty(l2_packed.shape, dtype=l2_packed.dtype)
        np.copyto(l2_weights, l2_packed)
        l1_weight_sf = _mega_moe_scale_storage(l1_scale_exponents, interleave_l1=True)
        l2_weight_sf = _mega_moe_scale_storage(l2_scale_exponents, interleave_l1=False)
    else:
        # Packed 0x1 E2M1 values and an exponent of -1 make every logical
        # expert weight nonzero without allocating an unpacked FP4 image.
        l1_weights = _filled_array(
            (
                config.num_experts_per_rank,
                config.intermediate_hidden * 2,
                config.hidden // 2,
            ),
            dtype=np.dtype(np.uint8),
            byte=0x11,
        )
        l2_weights = _filled_array(
            (
                config.num_experts_per_rank,
                config.hidden,
                config.intermediate_hidden // 2,
            ),
            dtype=np.dtype(np.uint8),
            byte=0x11,
        )
        l1_weight_sf = _filled_array(
            (
                config.num_experts_per_rank * (config.hidden // 128),
                config.intermediate_hidden * 2,
            ),
            dtype=np.dtype(np.uint32),
            byte=0x7E,
        )
        l2_weight_sf = _filled_array(
            (
                config.num_experts_per_rank * (config.intermediate_hidden // 128),
                config.hidden,
            ),
            dtype=np.dtype(np.uint32),
            byte=0x7E,
        )
    shared_intermediate = config.shared_intermediate_hidden
    shared_l1_codes = rng.choice(
        fp8_codes,
        size=(shared_intermediate * 2, config.hidden),
    ).astype(np.uint8)
    shared_l2_codes = rng.choice(
        fp8_codes,
        size=(config.hidden, shared_intermediate),
    ).astype(np.uint8)
    shared_l1_scale_exponents = rng.integers(
        -4,
        -1,
        size=(shared_intermediate * 2, config.hidden // 32),
        dtype=np.int16,
    )
    shared_l2_scale_exponents = rng.integers(
        -4,
        -1,
        size=(config.hidden, shared_intermediate // 32),
        dtype=np.int16,
    )
    shared_l1_weights = _mega_moe_interleave_l1_rows(shared_l1_codes[None, ...])[0]
    shared_l2_weights = shared_l2_codes.copy()
    shared_l1_weight_sf = _mega_moe_scale_storage(
        shared_l1_scale_exponents[None, ...], interleave_l1=True
    )
    shared_l2_weight_sf = _mega_moe_scale_storage(
        shared_l2_scale_exponents[None, ...], interleave_l1=False
    )

    if config.num_shared_experts > 0:
        shared_l1_sf_size = layout.num_max_shared_sf_tokens * (config.hidden // 32)
        shared_l1_sf = (
            symm_u8[layout.shared_l1_sf_offset : layout.shared_l1_sf_offset + shared_l1_sf_size]
            .view(np.uint32)
            .reshape(config.hidden // 128, layout.num_max_shared_sf_tokens)
        )
        input_scale_words = _pack_e8m0_words(
            input_scale_exponents.reshape(config.num_tokens, -1, 4)
        )
        for token_idx in range(config.num_tokens):
            token_in_block = token_idx % launch.block_m
            transposed_token = (
                (token_in_block // 128) * 128
                + (token_in_block % 32) * 4
                + (token_in_block % 128) // 32
            )
            shared_l1_sf[:, transposed_token] = input_scale_words[token_idx]

        shared_l2_acts_size = workspace.num_max_tokens_per_rank * shared_intermediate
        shared_l2_acts_sf_size = layout.num_max_shared_sf_tokens * (shared_intermediate // 32)
    tensor_maps = {
        "tensor_map_l1_acts": tensor_map(
            symm_view(layout.l1_token_offset, l1_acts_size, dtype="float8_e4m3fn"),
            dtype="float8_e4m3fn",
            global_shape=(config.hidden, workspace.num_ring_tokens),
            global_strides=(config.hidden,),
            box_shape=(128, launch.load_block_m),
            swizzle="128B",
        ),
        "tensor_map_l1_acts_sf": tensor_map(
            symm_view(layout.l1_sf_offset, l1_acts_sf_size, dtype="int32"),
            dtype="int32",
            global_shape=(workspace.num_sf_ring_tokens, config.hidden // 128),
            global_strides=(workspace.num_sf_ring_tokens * 4,),
            box_shape=(sf_block_m, launch.block_k // 128),
            swizzle=None,
        ),
        "tensor_map_l1_weights": tensor_map(
            l1_weights,
            dtype="uint8",
            global_shape=(
                config.hidden,
                config.num_experts_per_rank * config.intermediate_hidden * 2,
            ),
            global_strides=(config.hidden // 2,),
            box_shape=(128, launch.load_block_n),
            swizzle="128B",
            fp4_shared_layout="align16_padded",
        ),
        "tensor_map_l1_weights_sf": tensor_map(
            l1_weight_sf,
            dtype="int32",
            global_shape=(
                config.intermediate_hidden * 2,
                config.num_experts_per_rank * (config.hidden // 128),
            ),
            global_strides=(config.intermediate_hidden * 2 * 4,),
            box_shape=(launch.block_n, launch.block_k // 128),
            swizzle=None,
        ),
        "tensor_map_l1_output": tensor_map(
            symm_view(layout.l2_token_offset, l2_acts_size, dtype="float8_e4m3fn"),
            dtype="float8_e4m3fn",
            global_shape=(config.intermediate_hidden, workspace.num_ring_tokens),
            global_strides=(config.intermediate_hidden,),
            box_shape=(64, launch.store_block_m),
            swizzle="64B",
        ),
        "tensor_map_l2_acts": tensor_map(
            symm_view(layout.l2_token_offset, l2_acts_size, dtype="float8_e4m3fn"),
            dtype="float8_e4m3fn",
            global_shape=(config.intermediate_hidden, workspace.num_ring_tokens),
            global_strides=(config.intermediate_hidden,),
            box_shape=(128, launch.load_block_m),
            swizzle="128B",
        ),
        "tensor_map_l2_acts_sf": tensor_map(
            symm_view(layout.l2_sf_offset, l2_acts_sf_size, dtype="int32"),
            dtype="int32",
            global_shape=(workspace.num_sf_ring_tokens, config.intermediate_hidden // 128),
            global_strides=(workspace.num_sf_ring_tokens * 4,),
            box_shape=(sf_block_m, launch.block_k // 128),
            swizzle=None,
        ),
        "tensor_map_l2_weights": tensor_map(
            l2_weights,
            dtype="uint8",
            global_shape=(config.intermediate_hidden, config.num_experts_per_rank * config.hidden),
            global_strides=(config.intermediate_hidden // 2,),
            box_shape=(128, launch.load_block_n),
            swizzle="128B",
            fp4_shared_layout="align16_padded",
        ),
        "tensor_map_l2_weights_sf": tensor_map(
            l2_weight_sf,
            dtype="int32",
            global_shape=(
                config.hidden,
                config.num_experts_per_rank * (config.intermediate_hidden // 128),
            ),
            global_strides=(config.hidden * 4,),
            box_shape=(launch.block_n, launch.block_k // 128),
            swizzle=None,
        ),
    }
    if config.num_shared_experts > 0:
        tensor_maps.update(
            {
                "tensor_map_shared_l1_acts": tensor_map(
                    symm_view(layout.shared_l1_token_offset, input_bytes, dtype="float8_e4m3fn"),
                    dtype="float8_e4m3fn",
                    global_shape=(config.hidden, workspace.num_max_tokens_per_rank),
                    global_strides=(config.hidden,),
                    box_shape=(128, launch.load_block_m),
                    swizzle="128B",
                ),
                "tensor_map_shared_l1_acts_sf": tensor_map(
                    symm_view(layout.shared_l1_sf_offset, shared_l1_sf_size, dtype="int32"),
                    dtype="int32",
                    global_shape=(layout.num_max_shared_sf_tokens, config.hidden // 128),
                    global_strides=(layout.num_max_shared_sf_tokens * 4,),
                    box_shape=(sf_block_m, launch.block_k // 128),
                    swizzle=None,
                ),
                "tensor_map_shared_l1_weights": tensor_map(
                    shared_l1_weights,
                    dtype="uint8",
                    global_shape=(config.hidden, shared_intermediate * 2),
                    global_strides=(config.hidden,),
                    box_shape=(128, launch.load_block_n),
                    swizzle="128B",
                ),
                "tensor_map_shared_l1_weights_sf": tensor_map(
                    shared_l1_weight_sf,
                    dtype="int32",
                    global_shape=(shared_intermediate * 2, config.hidden // 128),
                    global_strides=(shared_intermediate * 2 * 4,),
                    box_shape=(launch.block_n, launch.block_k // 128),
                    swizzle=None,
                ),
                "tensor_map_shared_l1_output": tensor_map(
                    symm_view(
                        layout.shared_l2_token_offset,
                        shared_l2_acts_size,
                        dtype="float8_e4m3fn",
                    ),
                    dtype="float8_e4m3fn",
                    global_shape=(shared_intermediate, workspace.num_max_tokens_per_rank),
                    global_strides=(shared_intermediate,),
                    box_shape=(64, launch.store_block_m),
                    swizzle="64B",
                ),
                "tensor_map_shared_l2_acts": tensor_map(
                    symm_view(
                        layout.shared_l2_token_offset,
                        shared_l2_acts_size,
                        dtype="float8_e4m3fn",
                    ),
                    dtype="float8_e4m3fn",
                    global_shape=(shared_intermediate, workspace.num_max_tokens_per_rank),
                    global_strides=(shared_intermediate,),
                    box_shape=(128, launch.load_block_m),
                    swizzle="128B",
                ),
                "tensor_map_shared_l2_acts_sf": tensor_map(
                    symm_view(
                        layout.shared_l2_sf_offset,
                        shared_l2_acts_sf_size,
                        dtype="int32",
                    ),
                    dtype="int32",
                    global_shape=(layout.num_max_shared_sf_tokens, shared_intermediate // 128),
                    global_strides=(layout.num_max_shared_sf_tokens * 4,),
                    box_shape=(sf_block_m, launch.block_k // 128),
                    swizzle=None,
                ),
                "tensor_map_shared_l2_weights": tensor_map(
                    shared_l2_weights,
                    dtype="uint8",
                    global_shape=(shared_intermediate, config.hidden),
                    global_strides=(shared_intermediate,),
                    box_shape=(128, launch.load_block_n),
                    swizzle="128B",
                ),
                "tensor_map_shared_l2_weights_sf": tensor_map(
                    shared_l2_weight_sf,
                    dtype="int32",
                    global_shape=(config.hidden, shared_intermediate // 128),
                    global_strides=(config.hidden * 4,),
                    box_shape=(launch.block_n, launch.block_k // 128),
                    swizzle=None,
                ),
            }
        )
    else:
        # Mirror the launcher ABI: at S == 0 the nine shared slots carry the
        # matching routed descriptor, so the signature is independent of S.
        for name in (
            "tensor_map_shared_l1_acts",
            "tensor_map_shared_l1_acts_sf",
            "tensor_map_shared_l1_weights",
            "tensor_map_shared_l1_weights_sf",
            "tensor_map_shared_l1_output",
            "tensor_map_shared_l2_acts",
            "tensor_map_shared_l2_acts_sf",
            "tensor_map_shared_l2_weights",
            "tensor_map_shared_l2_weights_sf",
        ):
            tensor_maps[name] = tensor_maps[name.replace("_shared", "", 1)]
    y = np.full((config.num_tokens, config.hidden), np.uint16(0x7FC0), dtype=np.uint16)
    stats = (
        np.array([7, 11], dtype=np.int32)
        if config_entry is None
        else np.arange(config.num_experts_per_rank, dtype=np.int32) + np.int32(7)
    )
    expected_stats = stats + np.bincount(routes.ravel(), minlength=config.num_experts_per_rank)
    expected_y = (
        _mega_moe_reference(
            input_codes=input_codes,
            input_scale_exponents=input_scale_exponents,
            l1_packed=l1_packed,
            l1_scale_exponents=l1_scale_exponents,
            l2_packed=l2_packed,
            l2_scale_exponents=l2_scale_exponents,
            topk_idx=routes,
            topk_weights=route_weights,
            activation_clamp=config.activation_clamp,
            shared_l1_codes=shared_l1_codes,
            shared_l1_scale_exponents=shared_l1_scale_exponents,
            shared_l2_codes=shared_l2_codes,
            shared_l2_scale_exponents=shared_l2_scale_exponents,
        )
        if config_entry is None
        else np.zeros((config.num_tokens, config.hidden), dtype=np.float32)
    )
    args: dict[str, Any] = {
        "y": y.reshape(-1),
        "cumulative_local_expert_recv_stats": stats,
        "symm_buffer": symm,
        **{f"symm_rank_offset_{rank}": np.int64(0) for rank in range(72)},
        **tensor_maps,
        "num_tokens": np.int32(config.num_tokens),
        "rank_idx": np.int32(0),
    }
    return NumSimCase(
        kernel=kernel,
        args=args,
        outputs=("y", "cumulative_local_expert_recv_stats"),
        reference=lambda: {
            "y": expected_y.reshape(-1).copy(),
            "cumulative_local_expert_recv_stats": expected_stats.astype(np.int32, copy=True),
        },
        comparisons={
            "y": ComparisonSpec(rtol=0, atol=0, actual_encoding="bfloat16"),
            "cumulative_local_expert_recv_stats": ComparisonSpec(rtol=0, atol=0),
        },
    )


_FP4_MQA_CONFIGS = (
    {
        "seq_len": 32,
        "seq_len_kv": 256,
        "num_heads": 64,
        "head_dim": 128,
        "logits_dtype": "float32",
        "compressed_logits": False,
        "disable_cp": True,
        "seed": 4100,
        "label": "s32_skv256_h64_d128_f32_dense_nocp",
        "num_sms": 2,
    },
    {
        "seq_len": 32,
        "seq_len_kv": 256,
        "num_heads": 64,
        "head_dim": 128,
        "logits_dtype": "bfloat16",
        "compressed_logits": True,
        "disable_cp": True,
        "seed": 4101,
        "label": "s32_skv256_h64_d128_bf16_compressed_nocp",
        "num_sms": 2,
    },
    {
        "seq_len": 32,
        "seq_len_kv": 256,
        "num_heads": 64,
        "head_dim": 128,
        "logits_dtype": "float32",
        "compressed_logits": True,
        "disable_cp": False,
        "seed": 4102,
        "label": "s32_skv256_h64_d128_f32_compressed_cp",
        "num_sms": 2,
    },
    {
        "seq_len": 32,
        "seq_len_kv": 256,
        "num_heads": 64,
        "head_dim": 128,
        "logits_dtype": "float32",
        "compressed_logits": False,
        "disable_cp": False,
        "seed": 4103,
        "label": "s32_skv256_h64_d128_f32_dense_cp",
        "num_sms": 2,
    },
)

_FP8_MQA_CONFIGS = tuple({**config, "seed": config["seed"] + 100} for config in _FP4_MQA_CONFIGS)


_TF32_HC_CONFIGS = (
    {"m": 13, "n": 24, "k": 128, "num_splits": 1, "seed": 3200, "label": "m13_n24_k128_s1"},
    {"m": 13, "n": 24, "k": 2048, "num_splits": 16, "seed": 3201, "label": "m13_n24_k2048_s16"},
    {"m": 65, "n": 24, "k": 512, "num_splits": 4, "seed": 3202, "label": "m65_n24_k512_s4"},
)

FP4_MQA_CONFIGS = _FP4_MQA_CONFIGS
FP8_MQA_CONFIGS = _FP8_MQA_CONFIGS
TF32_HC_CONFIGS = _TF32_HC_CONFIGS


__all__ = [
    "FP4_MQA_CONFIGS",
    "FP8_MQA_CONFIGS",
    "TF32_HC_CONFIGS",
    "_dense_mqa_reference",
    "_e2m1_bits_to_float32",
    "_float32_to_bfloat16_bits",
    "_numpy_tf32_hc_reference",
    "_pack_e2m1",
    "_pack_e8m0_words",
    "_round_float32_to_tf32",
    "_unpack_e2m1",
    "_unpack_e8m0_words",
    "bfloat16_bits_to_float32",
    "float32_to_bfloat16_bits",
    "float32_to_e4m3fn_bits",
    "prepare_fp4_mqa_case",
    "prepare_fp8_mqa_case",
    "prepare_mega_moe_case",
    "prepare_paged_mqa_case",
    "prepare_tf32_hc_case",
]
