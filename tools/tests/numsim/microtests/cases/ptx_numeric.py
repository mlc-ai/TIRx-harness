from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from tvm.script import tirx as T

from tests.numsim.runtime.test_scalar_control import (
    cuda_packed_nan_semantics,
    ptx_dps_arithmetic,
    ptx_dps_modifier_forms,
)
from tests.numsim.runtime.test_approximate_f32_contract import (
    approximate_f32_calls,
    f16x2_exp2_calls,
)


@T.prim_func
def ptx_directed_rounding_forms(
    output_scalar: T.Buffer((4, 4), "float32"),
    output_packed: T.Buffer((3, 4), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        one = T.cuda.uint_as_float(T.uint32(0x3F800000))
        half_ulp = T.cuda.uint_as_float(T.uint32(0x33800000))
        minus_one = T.cuda.uint_as_float(T.uint32(0xBF800000))
        minus_half_ulp = T.cuda.uint_as_float(T.uint32(0xB3800000))
        largest = T.cuda.uint_as_float(T.uint32(0x7F7FFFFF))
        minus_largest = T.cuda.uint_as_float(T.uint32(0xFF7FFFFF))
        two = T.cuda.uint_as_float(T.uint32(0x40000000))
        one_plus_2m12 = T.cuda.uint_as_float(T.uint32(0x3F800800))
        one_minus_2m12 = T.cuda.uint_as_float(T.uint32(0x3F7FF000))

        T.ptx.add.rn.f32(output_scalar[0, 0], one, half_ulp)
        T.ptx.add.rz.f32(output_scalar[0, 1], one, half_ulp)
        T.ptx.add.rm.f32(output_scalar[0, 2], one, half_ulp)
        T.ptx.add.rp.f32(output_scalar[0, 3], one, half_ulp)
        T.ptx.add.rn.f32(output_scalar[1, 0], minus_one, minus_half_ulp)
        T.ptx.add.rz.f32(output_scalar[1, 1], minus_one, minus_half_ulp)
        T.ptx.add.rm.f32(output_scalar[1, 2], minus_one, minus_half_ulp)
        T.ptx.add.rp.f32(output_scalar[1, 3], minus_one, minus_half_ulp)
        T.ptx.mul.rn.f32(output_scalar[2, 0], largest, two)
        T.ptx.mul.rz.f32(output_scalar[2, 1], largest, two)
        T.ptx.mul.rm.f32(output_scalar[2, 2], largest, two)
        T.ptx.mul.rp.f32(output_scalar[2, 3], largest, two)
        T.ptx.fma.rn.f32(output_scalar[3, 0], one_plus_2m12, one_minus_2m12, minus_one)
        T.ptx.fma.rz.f32(output_scalar[3, 1], one_plus_2m12, one_minus_2m12, minus_one)
        T.ptx.fma.rm.f32(output_scalar[3, 2], one_plus_2m12, one_minus_2m12, minus_one)
        T.ptx.fma.rp.f32(output_scalar[3, 3], one_plus_2m12, one_minus_2m12, minus_one)

        add_lhs = T.cuda.make_float2(one, minus_one)
        add_rhs = T.cuda.make_float2(half_ulp, minus_half_ulp)
        mul_lhs = T.cuda.make_float2(largest, minus_largest)
        mul_rhs = T.cuda.make_float2(two, two)
        fma_lhs = T.cuda.make_float2(one_plus_2m12, one_plus_2m12)
        fma_rhs = T.cuda.make_float2(one_minus_2m12, one_minus_2m12)
        fma_addend = T.cuda.make_float2(minus_one, minus_one)
        T.ptx.add.rn.f32x2(output_packed[0, 0], add_lhs, add_rhs)
        T.ptx.add.rz.f32x2(output_packed[0, 1], add_lhs, add_rhs)
        T.ptx.add.rm.f32x2(output_packed[0, 2], add_lhs, add_rhs)
        T.ptx.add.rp.f32x2(output_packed[0, 3], add_lhs, add_rhs)
        T.ptx.mul.rn.f32x2(output_packed[1, 0], mul_lhs, mul_rhs)
        T.ptx.mul.rz.f32x2(output_packed[1, 1], mul_lhs, mul_rhs)
        T.ptx.mul.rm.f32x2(output_packed[1, 2], mul_lhs, mul_rhs)
        T.ptx.mul.rp.f32x2(output_packed[1, 3], mul_lhs, mul_rhs)
        T.ptx.fma.rn.f32x2(output_packed[2, 0], fma_lhs, fma_rhs, fma_addend)
        T.ptx.fma.rz.f32x2(output_packed[2, 1], fma_lhs, fma_rhs, fma_addend)
        T.ptx.fma.rm.f32x2(output_packed[2, 2], fma_lhs, fma_rhs, fma_addend)
        T.ptx.fma.rp.f32x2(output_packed[2, 3], fma_lhs, fma_rhs, fma_addend)


@dataclass(frozen=True)
class PtxNumericCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...]
    max_ulp: int | Mapping[str, int] = 0


def _approximate_arguments() -> Mapping[str, Any]:
    raw_inputs = np.asarray(
        [
            0x3DCCCCCD,
            0xC0700000,
            0x41890000,
            0x3F9E064B,
            0x00000001,
            0x80000001,
            0xC3150000,
            0x7F7FFFFF,
        ],
        dtype=np.uint32,
    ).view(np.float32)
    source = np.tile(raw_inputs, 4)
    return {
        "source": source,
        "exponentials": np.zeros(32, dtype=np.float32),
        "reciprocals": np.zeros(32, dtype=np.float32),
    }


def _f16x2_exp2_arguments() -> Mapping[str, Any]:
    source = np.resize(
        np.array(
            [
                0x3C00_0000,
                0x0001_FC00,
                0xCE00_CB00,
                0x7C00_CE40,
            ],
            dtype=np.uint32,
        ),
        32,
    )
    return {
        "source": source,
        "exponentials": np.zeros(32, dtype=np.uint32),
    }


PTX_NUMERIC_CASES = (
    PtxNumericCase(
        "directed_rounding_all_scalar_and_packed_forms",
        ptx_directed_rounding_forms,
        lambda: {
            "output_scalar": np.zeros((4, 4), dtype=np.float32),
            "output_packed": np.zeros((3, 4), dtype=np.uint64),
        },
        ("output_scalar", "output_packed"),
    ),
    PtxNumericCase(
        "dps_f32x2_f64_arithmetic",
        ptx_dps_arithmetic,
        lambda: {
            "output_f32": np.zeros(32, dtype=np.float32),
            "output_f32x2": np.zeros(32, dtype=np.uint64),
            "output_f64": np.zeros((32, 4), dtype=np.float64),
        },
        ("output_f32", "output_f32x2", "output_f64"),
    ),
    PtxNumericCase(
        "dps_directed_rounding_ftz_sat",
        ptx_dps_modifier_forms,
        lambda: {
            "output_f32": np.zeros((32, 4), dtype=np.float32),
            "output_f32x2": np.zeros((32, 4), dtype=np.uint64),
        },
        ("output_f32", "output_f32x2"),
    ),
    PtxNumericCase(
        "cuda_packed_nan_and_bf16_conversion",
        cuda_packed_nan_semantics,
        lambda: {
            "output_u64": np.zeros((32, 2), dtype=np.uint64),
            "output_u32": np.zeros((32, 4), dtype=np.uint32),
        },
        ("output_u64", "output_u32"),
    ),
    PtxNumericCase(
        "approximate_exp2_and_reciprocal",
        approximate_f32_calls,
        _approximate_arguments,
        ("exponentials", "reciprocals"),
        {"exponentials": 2, "reciprocals": 1},
    ),
    PtxNumericCase(
        "packed_f16x2_approximate_exp2",
        f16x2_exp2_calls,
        _f16x2_exp2_arguments,
        ("exponentials",),
    ),
)


__all__ = ["PTX_NUMERIC_CASES", "PtxNumericCase"]
