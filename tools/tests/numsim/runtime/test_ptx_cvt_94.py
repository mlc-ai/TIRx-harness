"""Bit-exact runtime coverage for the PTX 9.4 ``cvt`` families."""

from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def ptx94_cvt_forms(
    output_u16: T.Buffer((32, 6), "uint16"),
    output_u32: T.Buffer((32, 2), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    T.ptx["cvt.rz.satfinite.scaled::n1::ue8m0.e2m3x2.f32"](
        output_u16[lane, 0], T.float32(2.125), T.float32(-2.375), T.uint8(128)
    )
    T.ptx["cvt.rz.satfinite.e3m2x2.bf16x2"](output_u16[lane, 1], T.uint32(0x3F88BF98))
    T.ptx["cvt.rp.satfinite.ue5m3x2.f32"](output_u16[lane, 2], T.float32(1.0625), T.float32(1.1875))
    T.ptx["cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32"](
        output_u16[lane, 3], T.float32(2.125), T.float32(2.375), T.uint8(128)
    )
    T.ptx["cvt.rz.satfinite.ue5m3x2.f16x2"](output_u16[lane, 4], T.uint32(0x3C403CC0))
    T.ptx["cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.bf16x2"](
        output_u16[lane, 5], T.uint32(0x40084018), T.uint8(128)
    )

    T.ptx["cvt.rn.f16x2.ue5m3x2"](output_u32[lane, 0], T.uint16(0x0178))
    T.ptx["cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.ue5m3x2"](
        output_u32[lane, 1], T.uint16(0xFE01), T.uint16(0x707F)
    )


@T.prim_func
def ptx94_pzo_forms(output: T.Buffer((32, 4), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["cvt.rz.satfinite.e4m3x2.f32"](output[lane, 0], T.float32(-1e-30), T.float32(-0.0))
    T.ptx["cvt.rz.satfinite.pzo.e4m3x2.f32"](output[lane, 1], T.float32(-1e-30), T.float32(-0.0))
    T.ptx["cvt.rn.satfinite.scaled::n1::ue8m0.e2m3x2.bf16x2"](
        output[lane, 2], T.uint32(0x80008000), T.uint8(127)
    )
    T.ptx["cvt.rn.satfinite.pzo.scaled::n1::ue8m0.e2m3x2.bf16x2"](
        output[lane, 3], T.uint32(0x80008000), T.uint8(127)
    )


def test_ptx94_non_pzo_cvt_families_match_independent_bit_oracle(tmp_path):
    result = numsim.Engine().run(
        numsim.transpile(ptx94_cvt_forms, cache_dir=tmp_path),
        {
            "output_u16": np.zeros((32, 6), np.uint16),
            "output_u32": np.zeros((32, 2), np.uint32),
        },
    )
    # These literals are derived directly from the format grids, not from the
    # engine codec: E2M3 encodes 1.0/-1.125 as 0x08/0x29; E3M2 encodes
    # 1.0/-1.0 as 0x0c/0x2c; UE5M3 encodes 1.0/1.125/1.25 as 0x78/0x79/0x7a.
    expected_u16 = np.asarray([0x0829, 0x0C2C, 0x797A, 0x787A, 0x7879, 0x787A], dtype=np.uint16)
    # UE5M3 0x01 is 2^-17 and 0x78 is 1.0.  For the scaled bfloat16 form,
    # 0xfe is 114688 and scale 0x70 is 2^-15, giving exactly 3.5; the lower
    # 0x01 uses unit scale.  The expected packed IEEE encodings are explicit.
    expected_u32 = np.asarray([0x00803C00, 0x40603700], dtype=np.uint32)
    expected = {"output_u16": expected_u16, "output_u32": expected_u32}
    for name, values in expected.items():
        np.testing.assert_array_equal(
            result.outputs[name], np.broadcast_to(values, (32, len(values))), err_msg=name
        )


def test_ptx94_pzo_normalizes_each_negative_zero_after_narrow_conversion(tmp_path):
    result = numsim.Engine().run(
        numsim.transpile(ptx94_pzo_forms, cache_dir=tmp_path),
        {"output": np.zeros((32, 4), np.uint16)},
    )
    expected = np.asarray([0x8080, 0, 0x2020, 0], dtype=np.uint16)
    np.testing.assert_array_equal(result.outputs["output"], np.broadcast_to(expected, (32, 4)))
