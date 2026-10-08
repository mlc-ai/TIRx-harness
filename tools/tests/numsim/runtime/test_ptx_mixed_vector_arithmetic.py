from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def ptx_mixed_vector_all_forms(
    low_f16: T.Buffer((32,), "uint32"),
    low_bf16: T.Buffer((32,), "uint32"),
    wide_a: T.Buffer((32,), "uint64"),
    wide_b: T.Buffer((32,), "uint64"),
    wide_c: T.Buffer((32,), "uint64"),
    output_wide: T.Buffer((28, 32), "uint64"),
    output_low: T.Buffer((8, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    T.ptx["add.f32x2.f16x2.f32x2"](output_wide[0, lane], low_f16[lane], wide_c[lane])
    T.ptx["add.rn.f32x2.f16x2.f32x2"](output_wide[1, lane], low_f16[lane], wide_c[lane])
    T.ptx["add.rz.f32x2.f16x2.f32x2"](output_wide[2, lane], low_f16[lane], wide_c[lane])
    T.ptx["add.rm.f32x2.f16x2.f32x2"](output_wide[3, lane], low_f16[lane], wide_c[lane])
    T.ptx["add.rp.f32x2.f16x2.f32x2"](output_wide[4, lane], low_f16[lane], wide_c[lane])
    T.ptx["add.f32x2.bf16x2.f32x2"](output_wide[5, lane], low_bf16[lane], wide_c[lane])
    T.ptx["add.rn.f32x2.bf16x2.f32x2"](output_wide[6, lane], low_bf16[lane], wide_c[lane])
    T.ptx["add.rz.f32x2.bf16x2.f32x2"](output_wide[7, lane], low_bf16[lane], wide_c[lane])
    T.ptx["add.rm.f32x2.bf16x2.f32x2"](output_wide[8, lane], low_bf16[lane], wide_c[lane])
    T.ptx["add.rp.f32x2.bf16x2.f32x2"](output_wide[9, lane], low_bf16[lane], wide_c[lane])

    T.ptx["sub.f32x2.f16x2.f32x2"](output_wide[10, lane], low_f16[lane], wide_c[lane])
    T.ptx["sub.rn.f32x2.f16x2.f32x2"](output_wide[11, lane], low_f16[lane], wide_c[lane])
    T.ptx["sub.rz.f32x2.f16x2.f32x2"](output_wide[12, lane], low_f16[lane], wide_c[lane])
    T.ptx["sub.rm.f32x2.f16x2.f32x2"](output_wide[13, lane], low_f16[lane], wide_c[lane])
    T.ptx["sub.rp.f32x2.f16x2.f32x2"](output_wide[14, lane], low_f16[lane], wide_c[lane])
    T.ptx["sub.f32x2.bf16x2.f32x2"](output_wide[15, lane], low_bf16[lane], wide_c[lane])
    T.ptx["sub.rn.f32x2.bf16x2.f32x2"](output_wide[16, lane], low_bf16[lane], wide_c[lane])
    T.ptx["sub.rz.f32x2.bf16x2.f32x2"](output_wide[17, lane], low_bf16[lane], wide_c[lane])
    T.ptx["sub.rm.f32x2.bf16x2.f32x2"](output_wide[18, lane], low_bf16[lane], wide_c[lane])
    T.ptx["sub.rp.f32x2.bf16x2.f32x2"](output_wide[19, lane], low_bf16[lane], wide_c[lane])

    T.ptx["fma.rn.f32x2.f16x2.f32x2.f32x2"](
        output_wide[20, lane], low_f16[lane], wide_b[lane], wide_c[lane]
    )
    T.ptx["fma.rz.f32x2.f16x2.f32x2.f32x2"](
        output_wide[21, lane], low_f16[lane], wide_b[lane], wide_c[lane]
    )
    T.ptx["fma.rm.f32x2.f16x2.f32x2.f32x2"](
        output_wide[22, lane], low_f16[lane], wide_b[lane], wide_c[lane]
    )
    T.ptx["fma.rp.f32x2.f16x2.f32x2.f32x2"](
        output_wide[23, lane], low_f16[lane], wide_b[lane], wide_c[lane]
    )
    T.ptx["fma.rn.f32x2.bf16x2.f32x2.f32x2"](
        output_wide[24, lane], low_bf16[lane], wide_b[lane], wide_c[lane]
    )
    T.ptx["fma.rz.f32x2.bf16x2.f32x2.f32x2"](
        output_wide[25, lane], low_bf16[lane], wide_b[lane], wide_c[lane]
    )
    T.ptx["fma.rm.f32x2.bf16x2.f32x2.f32x2"](
        output_wide[26, lane], low_bf16[lane], wide_b[lane], wide_c[lane]
    )
    T.ptx["fma.rp.f32x2.bf16x2.f32x2.f32x2"](
        output_wide[27, lane], low_bf16[lane], wide_b[lane], wide_c[lane]
    )

    T.ptx["add.rz.ftz.f16x2.f32x2.f32x2"](output_low[0, lane], wide_a[lane], wide_c[lane])
    T.ptx["add.rz.bf16x2.f32x2.f32x2"](output_low[1, lane], wide_a[lane], wide_c[lane])
    T.ptx["sub.rz.ftz.f16x2.f32x2.f32x2"](output_low[2, lane], wide_a[lane], wide_c[lane])
    T.ptx["sub.rz.bf16x2.f32x2.f32x2"](output_low[3, lane], wide_a[lane], wide_c[lane])
    T.ptx["mul.ftz.rz.f16x2.f32x2.f32x2"](output_low[4, lane], wide_a[lane], wide_c[lane])
    T.ptx["mul.rz.bf16x2.f32x2.f32x2"](output_low[5, lane], wide_a[lane], wide_c[lane])
    T.ptx["mul.bf16x2.bf16x2.f16x2"](output_low[6, lane], low_bf16[lane], low_f16[lane])
    T.ptx["mul.f16x2.f16x2.bf16x2"](output_low[7, lane], low_f16[lane], low_bf16[lane])


def _pack_f32x2(low: float, high: float) -> np.uint64:
    low_bits = int(np.asarray(low, dtype=np.float32).view(np.uint32))
    high_bits = int(np.asarray(high, dtype=np.float32).view(np.uint32))
    return np.uint64(low_bits | (high_bits << 32))


def test_all_36_mixed_vector_arithmetic_forms_execute_with_packed_lane_semantics(tmp_path):
    low_f16 = np.full(32, np.uint32(0x4000_3C00), dtype=np.uint32)
    low_bf16 = np.full(32, np.uint32(0x4000_3F80), dtype=np.uint32)
    wide_a = np.full(32, _pack_f32x2(4.0, 8.0), dtype=np.uint64)
    wide_b = np.full(32, _pack_f32x2(3.0, 4.0), dtype=np.uint64)
    wide_c = np.full(32, _pack_f32x2(0.5, 1.0), dtype=np.uint64)
    arguments = {
        "low_f16": low_f16,
        "low_bf16": low_bf16,
        "wide_a": wide_a,
        "wide_b": wide_b,
        "wide_c": wide_c,
        "output_wide": np.full((28, 32), np.uint64(0xDEAD_BEEF), dtype=np.uint64),
        "output_low": np.full((8, 32), np.uint32(0xDEAD_BEEF), dtype=np.uint32),
    }

    result = numsim.Engine().run(
        numsim.transpile(ptx_mixed_vector_all_forms, cache_dir=tmp_path),
        arguments,
        outputs=("output_wide", "output_low"),
    )
    expected_wide = np.asarray(
        [
            *([_pack_f32x2(1.5, 3.0)] * 10),
            *([_pack_f32x2(0.5, 1.0)] * 10),
            *([_pack_f32x2(3.5, 9.0)] * 8),
        ],
        dtype=np.uint64,
    )[:, None]
    expected_wide = np.broadcast_to(expected_wide, (28, 32))

    expected_low = np.asarray(
        [
            0x4880_4480,
            0x4110_4090,
            0x4700_4300,
            0x40E0_4060,
            0x4800_4000,
            0x4100_4000,
            0x4080_3F80,
            0x4400_3C00,
        ],
        dtype=np.uint32,
    )[:, None]
    expected_low = np.broadcast_to(expected_low, (8, 32))
    expected = {"output_wide": expected_wide, "output_low": expected_low}
    for name, values in expected.items():
        np.testing.assert_array_equal(result.outputs[name], values, err_msg=name)


__all__ = ["ptx_mixed_vector_all_forms"]
