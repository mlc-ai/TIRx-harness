from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def ptx_rm_zero_sign(
    scalar_one: T.Buffer((32,), "float32"),
    scalar_negative_one: T.Buffer((32,), "float32"),
    packed_f16_one: T.Buffer((32,), "uint32"),
    wide_one: T.Buffer((32,), "uint64"),
    wide_negative_one: T.Buffer((32,), "uint64"),
    scalar_output: T.Buffer((4, 32), "float32"),
    mixed_output: T.Buffer((12, 32), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    T.ptx["mad.rn.f32"](
        scalar_output[0, lane], scalar_one[lane], scalar_one[lane], scalar_negative_one[lane]
    )
    T.ptx["mad.rz.f32"](
        scalar_output[1, lane], scalar_one[lane], scalar_one[lane], scalar_negative_one[lane]
    )
    T.ptx["mad.rm.f32"](
        scalar_output[2, lane], scalar_one[lane], scalar_one[lane], scalar_negative_one[lane]
    )
    T.ptx["mad.rp.f32"](
        scalar_output[3, lane], scalar_one[lane], scalar_one[lane], scalar_negative_one[lane]
    )

    T.ptx["add.rn.f32x2.f16x2.f32x2"](
        mixed_output[0, lane], packed_f16_one[lane], wide_negative_one[lane]
    )
    T.ptx["add.rz.f32x2.f16x2.f32x2"](
        mixed_output[1, lane], packed_f16_one[lane], wide_negative_one[lane]
    )
    T.ptx["add.rm.f32x2.f16x2.f32x2"](
        mixed_output[2, lane], packed_f16_one[lane], wide_negative_one[lane]
    )
    T.ptx["add.rp.f32x2.f16x2.f32x2"](
        mixed_output[3, lane], packed_f16_one[lane], wide_negative_one[lane]
    )

    T.ptx["sub.rn.f32x2.f16x2.f32x2"](mixed_output[4, lane], packed_f16_one[lane], wide_one[lane])
    T.ptx["sub.rz.f32x2.f16x2.f32x2"](mixed_output[5, lane], packed_f16_one[lane], wide_one[lane])
    T.ptx["sub.rm.f32x2.f16x2.f32x2"](mixed_output[6, lane], packed_f16_one[lane], wide_one[lane])
    T.ptx["sub.rp.f32x2.f16x2.f32x2"](mixed_output[7, lane], packed_f16_one[lane], wide_one[lane])

    T.ptx["fma.rn.f32x2.f16x2.f32x2.f32x2"](
        mixed_output[8, lane],
        packed_f16_one[lane],
        wide_one[lane],
        wide_negative_one[lane],
    )
    T.ptx["fma.rz.f32x2.f16x2.f32x2.f32x2"](
        mixed_output[9, lane],
        packed_f16_one[lane],
        wide_one[lane],
        wide_negative_one[lane],
    )
    T.ptx["fma.rm.f32x2.f16x2.f32x2.f32x2"](
        mixed_output[10, lane],
        packed_f16_one[lane],
        wide_one[lane],
        wide_negative_one[lane],
    )
    T.ptx["fma.rp.f32x2.f16x2.f32x2.f32x2"](
        mixed_output[11, lane],
        packed_f16_one[lane],
        wide_one[lane],
        wide_negative_one[lane],
    )


def _pack_f32x2(value: float) -> np.uint64:
    bits = int(np.asarray(value, dtype=np.float32).view(np.uint32))
    return np.uint64(bits | (bits << 32))


def test_ptx_round_down_exact_cancellation_has_negative_zero_bits(tmp_path):
    arguments = {
        "scalar_one": np.ones(32, dtype=np.float32),
        "scalar_negative_one": np.full(32, -1.0, dtype=np.float32),
        "packed_f16_one": np.full(32, np.uint32(0x3C00_3C00), dtype=np.uint32),
        "wide_one": np.full(32, _pack_f32x2(1.0), dtype=np.uint64),
        "wide_negative_one": np.full(32, _pack_f32x2(-1.0), dtype=np.uint64),
        "scalar_output": np.full((4, 32), np.nan, dtype=np.float32),
        "mixed_output": np.full((12, 32), np.uint64(0xDEAD_BEEF), dtype=np.uint64),
    }
    result = numsim.Engine().run(
        numsim.transpile(ptx_rm_zero_sign, cache_dir=tmp_path / "artifact"),
        arguments,
        outputs=("scalar_output", "mixed_output"),
    )

    expected_scalar = np.asarray(
        [0x0000_0000, 0x0000_0000, 0x8000_0000, 0x0000_0000], dtype=np.uint32
    )[:, None]
    np.testing.assert_array_equal(
        result.outputs["scalar_output"].view(np.uint32),
        np.broadcast_to(expected_scalar, (4, 32)),
    )

    positive_zero_pair = np.uint64(0x0000_0000_0000_0000)
    negative_zero_pair = np.uint64(0x8000_0000_8000_0000)
    expected_mixed = np.asarray(
        [positive_zero_pair, positive_zero_pair, negative_zero_pair, positive_zero_pair] * 3,
        dtype=np.uint64,
    )[:, None]
    np.testing.assert_array_equal(
        result.outputs["mixed_output"],
        np.broadcast_to(expected_mixed, (12, 32)),
    )


__all__ = ["ptx_rm_zero_sign"]
