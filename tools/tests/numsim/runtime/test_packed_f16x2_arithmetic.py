from __future__ import annotations

import numpy as np
import pytest
import ml_dtypes

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def scalar_f16_add(
    lhs: T.Buffer((32,), "uint16"),
    rhs: T.Buffer((32,), "uint16"),
    output: T.Buffer((32,), "uint16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.add.f16(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def packed_f16x2_subtract(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.sub.f16x2(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def packed_f16x2_multiply(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.mul.f16x2(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def packed_f16x2_subtract_rn(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.sub.rn.f16x2(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def packed_f16x2_subtract_ftz(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.sub.ftz.f16x2(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def packed_f16x2_subtract_sat(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.sub.sat.f16x2(output[lane], lhs[lane], rhs[lane])


@T.prim_func
def packed_bf16x2_subtract(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.sub.bf16x2(output[lane], lhs[lane], rhs[lane])


def _pack_f16x2(values: np.ndarray) -> np.ndarray:
    halves = np.ascontiguousarray(values, dtype=np.float16)
    return halves.view(np.uint32).reshape(halves.shape[0])


def _pack_bf16x2(values: np.ndarray) -> np.ndarray:
    halves = np.ascontiguousarray(values, dtype=ml_dtypes.bfloat16)
    return halves.view(np.uint32).reshape(halves.shape[0])


def test_scalar_f16_add_rounds_once_to_binary16(tmp_path):
    lane = np.arange(32, dtype=np.float32)
    lhs = (lane / 7.0 - 1.75).astype(np.float16)
    rhs = ((lane % 5) / 9.0 - 0.25).astype(np.float16)
    expected = (lhs.astype(np.float32) + rhs.astype(np.float32)).astype(np.float16)

    module = numsim.transpile(scalar_f16_add, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "lhs": lhs.view(np.uint16),
            "rhs": rhs.view(np.uint16),
            "output": np.zeros(32, dtype=np.uint16),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], expected.view(np.uint16))


def test_packed_f16x2_subtracts_both_halves_with_one_rounding_step(tmp_path):
    lane = np.arange(32, dtype=np.float32)
    lhs_halves = np.stack((lane / 7.0 - 1.75, lane / 11.0 + 0.375), axis=1).astype(np.float16)
    rhs_halves = np.stack(((lane % 5) / 9.0, (lane % 7) / 13.0 - 0.25), axis=1).astype(np.float16)
    expected_halves = (lhs_halves.astype(np.float32) - rhs_halves.astype(np.float32)).astype(
        np.float16
    )

    module = numsim.transpile(packed_f16x2_subtract, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "lhs": _pack_f16x2(lhs_halves),
            "rhs": _pack_f16x2(rhs_halves),
            "output": np.zeros(32, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], _pack_f16x2(expected_halves))


def test_packed_bf16x2_subtracts_both_halves_with_one_rounding_step(tmp_path):
    lane = np.arange(32, dtype=np.float32)
    lhs_halves = np.asarray(
        np.stack((lane / 7.0 - 1.75, lane / 11.0 + 0.375), axis=1),
        dtype=ml_dtypes.bfloat16,
    )
    rhs_halves = np.asarray(
        np.stack(((lane % 5) / 9.0, (lane % 7) / 13.0 - 0.25), axis=1),
        dtype=ml_dtypes.bfloat16,
    )
    expected_halves = np.asarray(
        lhs_halves.astype(np.float32) - rhs_halves.astype(np.float32),
        dtype=ml_dtypes.bfloat16,
    )

    module = numsim.transpile(packed_bf16x2_subtract, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "lhs": _pack_bf16x2(lhs_halves),
            "rhs": _pack_bf16x2(rhs_halves),
            "output": np.zeros(32, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], _pack_bf16x2(expected_halves))


def test_packed_f16x2_multiplies_both_halves_with_one_rounding_step(tmp_path):
    lane = np.arange(32, dtype=np.float32)
    lhs_halves = np.stack((lane / 7.0 - 1.75, lane / 11.0 + 0.375), axis=1).astype(np.float16)
    rhs_halves = np.stack(((lane % 5) / 9.0, (lane % 7) / 13.0 - 0.25), axis=1).astype(np.float16)
    expected_halves = (lhs_halves.astype(np.float32) * rhs_halves.astype(np.float32)).astype(
        np.float16
    )
    # Multiplication preserves the XOR sign even for an exact zero product.

    module = numsim.transpile(packed_f16x2_multiply, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "lhs": _pack_f16x2(lhs_halves),
            "rhs": _pack_f16x2(rhs_halves),
            "output": np.zeros(32, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], _pack_f16x2(expected_halves))


@pytest.mark.parametrize(
    "kernel, expected",
    (
        (packed_f16x2_subtract_rn, [1, 0x8000, 0x3C00, 0xC000, 0x7FFF, 0x4000]),
        (packed_f16x2_subtract_ftz, [0, 0x8000, 0x3C00, 0xC000, 0x7FFF, 0x4000]),
        (packed_f16x2_subtract_sat, [1, 0, 0x3C00, 0, 0, 0x3C00]),
    ),
)
def test_half_subtract_modifiers(kernel, expected, tmp_path):
    def pack(bits):
        return np.resize(np.asarray(bits, np.uint16), 64).view(np.uint32)

    result = numsim.Engine().run(
        numsim.transpile(kernel, cache_dir=tmp_path),
        {
            "lhs": pack([1, 0x8000, 0x3C00, 0xBC00, 0x7FFF, 0x4000]),
            "rhs": pack([0, 0, 1, 0x3C00, 0x3C00, 0]),
            "output": np.zeros(32, np.uint32),
        },
    )
    np.testing.assert_array_equal(result.outputs["output"], pack(expected))
