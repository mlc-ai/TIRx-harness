from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tests.numsim.corpus.kernels.deepgemm import (
    bfloat16_bits_to_float32,
    float32_to_bfloat16_bits,
    float32_to_e4m3fn_bits,
)
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm.script import tirx as T


@T.prim_func
def mega_scalar_helpers(
    packed_lhs: T.Buffer((32,), "uint32"),
    packed_rhs: T.Buffer((32,), "uint32"),
    fp8_values: T.Buffer((32, 4), "float32"),
    masks: T.Buffer((32,), "uint32"),
    bases: T.Buffer((32,), "uint32"),
    offsets: T.Buffer((32,), "int32"),
    accum: T.Buffer((32,), "float32"),
    bf16_values: T.Buffer((32,), "uint16"),
    packed_output: T.Buffer((32, 4), "uint32"),
    float_output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed_output[lane, 0] = T.cuda.hmin2(packed_lhs[lane], packed_rhs[lane])
    packed_output[lane, 1] = T.cuda.hmax2(packed_lhs[lane], packed_rhs[lane])
    packed_output[lane, 2] = T.cuda.fp8x4_e4m3_from_float4(
        fp8_values[lane, 0], fp8_values[lane, 1], fp8_values[lane, 2], fp8_values[lane, 3]
    )
    T.ptx.fns.b32(packed_output[lane, 3], masks[lane], bases[lane], offsets[lane])
    T.ptx.add.rn.f32.bf16(float_output[lane], bf16_values[lane], accum[lane])


def _pack_bf16_pairs(values: np.ndarray) -> np.ndarray:
    bits = float32_to_bfloat16_bits(values)
    return bits[:, 0].astype(np.uint32) | (bits[:, 1].astype(np.uint32) << np.uint32(16))


def _fns(mask: int, base: int, offset: int) -> np.uint32:
    if offset == 0:
        return np.uint32(base if mask & (1 << base) else 0xFFFFFFFF)
    position = base
    remaining = abs(offset) - 1
    increment = 1 if offset > 0 else -1
    while 0 <= position < 32:
        if mask & (1 << position):
            if remaining == 0:
                return np.uint32(position)
            remaining -= 1
        position += increment
    return np.uint32(0xFFFFFFFF)


def test_mega_scalar_helpers_have_exact_typed_registry_entries():
    assert analyze(mega_scalar_helpers).unsupported == ()


def test_mega_scalar_helpers_match_packed_numeric_semantics(tmp_path):
    lanes = np.arange(32, dtype=np.int32)
    lhs_values = np.stack((lanes - 13, 7 - lanes), axis=1).astype(np.float32) / 4
    rhs_values = np.stack((5 - lanes, lanes - 19), axis=1).astype(np.float32) / 8
    packed_lhs = _pack_bf16_pairs(lhs_values)
    packed_rhs = _pack_bf16_pairs(rhs_values)
    fp8_values = np.stack(
        (
            lanes / 8,
            -lanes / 16,
            np.full(32, 448.0, dtype=np.float32),
            np.full(32, -0.5, dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    masks = np.array([0xA5A55A5A ^ (1 << (int(lane) % 31)) for lane in lanes], dtype=np.uint32)
    bases = (lanes % 32).astype(np.uint32)
    offsets = np.where(lanes % 3 == 0, 0, np.where(lanes % 2 == 0, 2, -2)).astype(np.int32)
    accum = np.linspace(-3.0, 4.0, 32, dtype=np.float32)
    bf16_values = float32_to_bfloat16_bits(np.linspace(-1.0, 2.0, 32, dtype=np.float32))
    packed_output = np.zeros((32, 4), dtype=np.uint32)
    float_output = np.zeros(32, dtype=np.float32)

    module = numsim.transpile(mega_scalar_helpers, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "packed_lhs": packed_lhs,
            "packed_rhs": packed_rhs,
            "fp8_values": fp8_values,
            "masks": masks,
            "bases": bases,
            "offsets": offsets,
            "accum": accum,
            "bf16_values": bf16_values,
            "packed_output": packed_output,
            "float_output": float_output,
        },
    )

    expected_min = _pack_bf16_pairs(np.minimum(lhs_values, rhs_values))
    expected_max = _pack_bf16_pairs(np.maximum(lhs_values, rhs_values))
    fp8_bits = float32_to_e4m3fn_bits(fp8_values).astype(np.uint32)
    expected_fp8 = (
        fp8_bits[:, 0]
        | (fp8_bits[:, 1] << np.uint32(8))
        | (fp8_bits[:, 2] << np.uint32(16))
        | (fp8_bits[:, 3] << np.uint32(24))
    )
    expected_fns = np.array(
        [
            _fns(int(mask), int(base), int(offset))
            for mask, base, offset in zip(masks, bases, offsets)
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(result.outputs["packed_output"][:, 0], expected_min)
    np.testing.assert_array_equal(result.outputs["packed_output"][:, 1], expected_max)
    np.testing.assert_array_equal(result.outputs["packed_output"][:, 2], expected_fp8)
    np.testing.assert_array_equal(result.outputs["packed_output"][:, 3], expected_fns)
    np.testing.assert_array_equal(
        result.outputs["float_output"], accum + bfloat16_bits_to_float32(bf16_values)
    )
