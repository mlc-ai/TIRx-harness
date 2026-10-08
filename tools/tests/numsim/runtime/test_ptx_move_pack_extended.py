from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim


@T.prim_func
def cvt_pack_u16_s16(
    a: T.Buffer((32,), "int32"),
    b: T.Buffer((32,), "int32"),
    output: T.Buffer((2, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.cvt_pack.sat.u16.s32(output[0, lane], a[lane], b[lane])
    T.ptx.cvt_pack.sat.s16.s32(output[1, lane], a[lane], b[lane])


@T.prim_func
def cvt_pack_subbyte(
    a: T.Buffer((32,), "int32"),
    b: T.Buffer((32,), "int32"),
    c: T.Buffer((32,), "float32"),
    output: T.Buffer((6, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.cvt_pack.sat.u2.s32.b32(output[0, lane], a[lane], b[lane], c[lane])
    T.ptx.cvt_pack.sat.s2.s32.b32(output[1, lane], a[lane], b[lane], c[lane])
    T.ptx.cvt_pack.sat.u4.s32.b32(output[2, lane], a[lane], b[lane], c[lane])
    T.ptx.cvt_pack.sat.s4.s32.b32(output[3, lane], a[lane], b[lane], c[lane])
    T.ptx.cvt_pack.sat.u8.s32.b32(output[4, lane], a[lane], b[lane], c[lane])
    T.ptx.cvt_pack.sat.s8.s32.b32(output[5, lane], a[lane], b[lane], c[lane])


@T.prim_func
def mov_pack_b16x4(
    source: T.Buffer((32, 4), "int16"),
    output: T.Buffer((32,), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.mov.b64(
        output[lane],
        source[lane, 0],
        source[lane, 1],
        source[lane, 2],
        source[lane, 3],
    )


@T.prim_func
def mov_unpack_b16x4(
    source: T.Buffer((32,), "float64"),
    output: T.Buffer((32, 4), "int16"),
    sink_output: T.Buffer((32, 2), "int16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.mov.b64(
        output[lane, 0],
        output[lane, 1],
        output[lane, 2],
        output[lane, 3],
        source[lane],
    )
    T.ptx.mov.b64(
        T.ptx.SINK,
        sink_output[lane, 0],
        T.ptx.SINK,
        sink_output[lane, 1],
        source[lane],
    )


@T.prim_func
def mov_pack_b32x4(
    source: T.Buffer((32, 4), "float32"),
    output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed = T.alloc_local((1,), "uint128")
    T.ptx.mov.b128(
        packed[0],
        source[lane, 0],
        source[lane, 1],
        source[lane, 2],
        source[lane, 3],
    )
    T.ptx["st.global.b128"](output.ptr_to([lane, 0]), packed[0])


@T.prim_func
def mov_unpack_b32x4(
    source: T.Buffer((32, 2), "uint64"),
    output: T.Buffer((32, 4), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed = T.alloc_local((1,), "uint128")
    T.ptx.mov.b128(packed[0], source[lane, 0], source[lane, 1])
    T.ptx.mov.b128(
        output[lane, 0],
        output[lane, 1],
        output[lane, 2],
        output[lane, 3],
        packed[0],
    )


@T.prim_func
def mov_unpack_b64x2(
    source: T.Buffer((32, 2), "uint64"),
    output: T.Buffer((32, 2), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed = T.alloc_local((1,), "uint128")
    T.ptx.mov.b128(packed[0], source[lane, 0], source[lane, 1])
    T.ptx.mov.b128(output[lane, 0], output[lane, 1], packed[0])


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    lane = np.arange(32, dtype=np.int64)
    a = (lane * 7919 - 90_000).astype(np.int32)
    b = (80_000 - lane * 6151).astype(np.int32)
    return a, b


def _cvt_pack_oracle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    bits: int,
    signed: bool,
) -> np.ndarray:
    if signed:
        minimum = -(1 << (bits - 1))
        maximum = (1 << (bits - 1)) - 1
    else:
        minimum = 0
        maximum = (1 << bits) - 1
    mask = np.uint64((1 << bits) - 1)
    a_field = np.clip(a.astype(np.int64), minimum, maximum).astype(np.uint64) & mask
    b_field = np.clip(b.astype(np.int64), minimum, maximum).astype(np.uint64) & mask
    result = b_field | (a_field << np.uint64(bits))
    if bits < 16:
        result |= c.astype(np.uint64) << np.uint64(2 * bits)
    return result.astype(np.uint32)


def _move_words() -> np.ndarray:
    lane = np.arange(32, dtype=np.uint64)[:, None]
    seeds = np.array([0x0123_4567, 0x89AB_CDEF, 0xFEDC_BA98, 0x7654_3210], dtype=np.uint64)[None, :]
    return ((lane * np.uint64(0x1020_3041) + seeds) & np.uint64(0xFFFF_FFFF)).astype(np.uint32)


def _make_mov_b16x4_roundtrip(dtype: str):
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def mov_b16x4_roundtrip(",
                f'    source: T.Buffer((32, 4), "{dtype}"),',
                f'    output: T.Buffer((32, 4), "{dtype}"),',
                "):",
                "    T.device_entry()",
                "    _warp = T.warp_id([1])",
                "    lane = T.lane_id([32])",
                '    packed = T.alloc_local((1,), "uint64")',
                "    T.ptx.mov.b64(",
                "        packed[0],",
                "        source[lane, 0],",
                "        source[lane, 1],",
                "        source[lane, 2],",
                "        source[lane, 3],",
                "    )",
                "    T.ptx.mov.b64(",
                "        output[lane, 0],",
                "        output[lane, 1],",
                "        output[lane, 2],",
                "        output[lane, 3],",
                "        packed[0],",
                "    )",
            )
        ),
        {"T": T},
    )


def test_cvt_pack_matches_saturating_two_field_oracle(tmp_path):
    a, b = _inputs()
    result = numsim.Engine().run(
        numsim.transpile(cvt_pack_u16_s16, cache_dir=tmp_path),
        {"a": a, "b": b, "output": np.zeros((2, 32), dtype=np.uint32)},
    )
    zeros = np.zeros(32, dtype=np.uint32)
    np.testing.assert_array_equal(
        result.outputs["output"][0], _cvt_pack_oracle(a, b, zeros, bits=16, signed=False)
    )
    np.testing.assert_array_equal(
        result.outputs["output"][1], _cvt_pack_oracle(a, b, zeros, bits=16, signed=True)
    )


def test_cvt_pack_c_matches_all_six_subbyte_forms_and_b32_carrier_bits(tmp_path):
    a, b = _inputs()
    c_bits = (_move_words()[:, 0] ^ np.uint32(0x7F80_0001)).astype(np.uint32)
    c = c_bits.view(np.float32)
    result = numsim.Engine().run(
        numsim.transpile(cvt_pack_subbyte, cache_dir=tmp_path),
        {
            "a": a,
            "b": b,
            "c": c,
            "output": np.zeros((6, 32), dtype=np.uint32),
        },
    )
    for row, (bits, signed) in enumerate(
        ((2, False), (2, True), (4, False), (4, True), (8, False), (8, True))
    ):
        np.testing.assert_array_equal(
            result.outputs["output"][row],
            _cvt_pack_oracle(a, b, c_bits, bits=bits, signed=signed),
        )


def test_mov_pack_b16x4_places_first_integer_carrier_in_low_bits(tmp_path):
    words = np.ascontiguousarray(_move_words().view(np.uint16).reshape(32, 8)[:, :4])
    source = words.view(np.int16)
    result = numsim.Engine().run(
        numsim.transpile(mov_pack_b16x4, cache_dir=tmp_path),
        {"source": source, "output": np.zeros(32, dtype=np.float64)},
    )
    expected = sum(words[:, index].astype(np.uint64) << np.uint64(16 * index) for index in range(4))
    np.testing.assert_array_equal(result.outputs["output"].view(np.uint64), expected)


def test_mov_unpack_b16x4_preserves_lane_order_and_explicit_sinks(tmp_path):
    words = np.ascontiguousarray(_move_words().view(np.uint16).reshape(32, 8)[:, :4])
    packed = sum(words[:, index].astype(np.uint64) << np.uint64(16 * index) for index in range(4))
    result = numsim.Engine().run(
        numsim.transpile(mov_unpack_b16x4, cache_dir=tmp_path),
        {
            "source": packed.view(np.float64),
            "output": np.zeros((32, 4), dtype=np.int16),
            "sink_output": np.zeros((32, 2), dtype=np.int16),
        },
    )
    np.testing.assert_array_equal(result.outputs["output"].view(np.uint16), words)
    np.testing.assert_array_equal(result.outputs["sink_output"].view(np.uint16), words[:, (1, 3)])


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    (("float16", np.float16), ("bfloat16", ml_dtypes.bfloat16)),
    ids=("float16", "bfloat16"),
)
def test_mov_b16x4_roundtrip_preserves_every_low_precision_payload_bit(
    tmp_path, dtype, numpy_dtype
):
    edge_bits = np.asarray(
        [
            0x0000,
            0x0001,
            0x03FF,
            0x0400,
            0x3C00,
            0x7BFF,
            0x7C00,
            0x7C01,
            0x7DFF,
            0x7E00,
            0x7FFF,
            0x8000,
            0xFC00,
            0xFC01,
            0xFE00,
            0xFFFF,
            0x7F80,
            0x7F81,
            0x7FBF,
            0x7FC0,
            0xFF80,
            0xFF81,
            0xFFBF,
            0xFFC0,
            0x1234,
            0x2345,
            0x4567,
            0x6789,
            0x89AB,
            0xABCD,
            0xCDEF,
            0xDEAD,
        ],
        dtype=np.uint16,
    )
    bits = np.tile(edge_bits, 4).reshape(32, 4)
    source = bits.view(numpy_dtype)
    result = numsim.Engine().run(
        numsim.transpile(_make_mov_b16x4_roundtrip(dtype), cache_dir=tmp_path),
        {"source": source, "output": np.zeros((32, 4), dtype=numpy_dtype)},
    )
    np.testing.assert_array_equal(result.outputs["output"].view(np.uint16), bits)


def test_mov_pack_b32x4_preserves_float_carrier_payloads_and_lane_order(tmp_path):
    words = _move_words()
    result = numsim.Engine().run(
        numsim.transpile(mov_pack_b32x4, cache_dir=tmp_path),
        {
            "source": words.view(np.float32),
            "output": np.zeros((32, 4), dtype=np.uint32),
        },
    )
    np.testing.assert_array_equal(result.outputs["output"], words)


def test_mov_unpack_b32x4_preserves_b128_halves_as_float_carrier_bits(tmp_path):
    words = _move_words()
    source = words.view(np.uint64).reshape(32, 2)
    result = numsim.Engine().run(
        numsim.transpile(mov_unpack_b32x4, cache_dir=tmp_path),
        {"source": source, "output": np.zeros((32, 4), dtype=np.float32)},
    )
    np.testing.assert_array_equal(result.outputs["output"].view(np.uint32), words)


def test_mov_unpack_b64x2_preserves_low_then_high_halves_as_float_bits(tmp_path):
    source = _move_words().view(np.uint64).reshape(32, 2)
    result = numsim.Engine().run(
        numsim.transpile(mov_unpack_b64x2, cache_dir=tmp_path),
        {"source": source, "output": np.zeros((32, 2), dtype=np.float64)},
    )
    np.testing.assert_array_equal(result.outputs["output"].view(np.uint64), source)
