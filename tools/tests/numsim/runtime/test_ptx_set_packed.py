from __future__ import annotations

import operator

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


_PACKED_FORMS = (
    *(("u8x4", cmp) for cmp in ("eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs")),
    *(("s8x4", cmp) for cmp in ("eq", "ne", "lt", "le", "gt", "ge")),
    *(("u16x2", cmp) for cmp in ("eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs")),
    *(("s16x2", cmp) for cmp in ("eq", "ne", "lt", "le", "gt", "ge")),
)


@T.prim_func
def ptx_set_packed_semantics(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    output: T.Buffer((32, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["set.eq.u8x4"](output[0, lane], lhs[lane], rhs[lane])
    T.ptx["set.ne.u8x4"](output[1, lane], lhs[lane], rhs[lane])
    T.ptx["set.lt.u8x4"](output[2, lane], lhs[lane], rhs[lane])
    T.ptx["set.le.u8x4"](output[3, lane], lhs[lane], rhs[lane])
    T.ptx["set.gt.u8x4"](output[4, lane], lhs[lane], rhs[lane])
    T.ptx["set.ge.u8x4"](output[5, lane], lhs[lane], rhs[lane])
    T.ptx["set.lo.u8x4"](output[6, lane], lhs[lane], rhs[lane])
    T.ptx["set.ls.u8x4"](output[7, lane], lhs[lane], rhs[lane])
    T.ptx["set.hi.u8x4"](output[8, lane], lhs[lane], rhs[lane])
    T.ptx["set.hs.u8x4"](output[9, lane], lhs[lane], rhs[lane])
    T.ptx["set.eq.s8x4"](output[10, lane], lhs[lane], rhs[lane])
    T.ptx["set.ne.s8x4"](output[11, lane], lhs[lane], rhs[lane])
    T.ptx["set.lt.s8x4"](output[12, lane], lhs[lane], rhs[lane])
    T.ptx["set.le.s8x4"](output[13, lane], lhs[lane], rhs[lane])
    T.ptx["set.gt.s8x4"](output[14, lane], lhs[lane], rhs[lane])
    T.ptx["set.ge.s8x4"](output[15, lane], lhs[lane], rhs[lane])
    T.ptx["set.eq.u16x2"](output[16, lane], lhs[lane], rhs[lane])
    T.ptx["set.ne.u16x2"](output[17, lane], lhs[lane], rhs[lane])
    T.ptx["set.lt.u16x2"](output[18, lane], lhs[lane], rhs[lane])
    T.ptx["set.le.u16x2"](output[19, lane], lhs[lane], rhs[lane])
    T.ptx["set.gt.u16x2"](output[20, lane], lhs[lane], rhs[lane])
    T.ptx["set.ge.u16x2"](output[21, lane], lhs[lane], rhs[lane])
    T.ptx["set.lo.u16x2"](output[22, lane], lhs[lane], rhs[lane])
    T.ptx["set.ls.u16x2"](output[23, lane], lhs[lane], rhs[lane])
    T.ptx["set.hi.u16x2"](output[24, lane], lhs[lane], rhs[lane])
    T.ptx["set.hs.u16x2"](output[25, lane], lhs[lane], rhs[lane])
    T.ptx["set.eq.s16x2"](output[26, lane], lhs[lane], rhs[lane])
    T.ptx["set.ne.s16x2"](output[27, lane], lhs[lane], rhs[lane])
    T.ptx["set.lt.s16x2"](output[28, lane], lhs[lane], rhs[lane])
    T.ptx["set.le.s16x2"](output[29, lane], lhs[lane], rhs[lane])
    T.ptx["set.gt.s16x2"](output[30, lane], lhs[lane], rhs[lane])
    T.ptx["set.ge.s16x2"](output[31, lane], lhs[lane], rhs[lane])


@T.prim_func
def ptx_set_packed_b32_carriers(
    lhs_u32: T.Buffer((32,), "uint32"),
    lhs_i32: T.Buffer((32,), "int32"),
    lhs_f32: T.Buffer((32,), "float32"),
    rhs_u32: T.Buffer((32,), "uint32"),
    rhs_i32: T.Buffer((32,), "int32"),
    rhs_f32: T.Buffer((32,), "float32"),
    output_u32: T.Buffer((9, 32), "uint32"),
    output_i32: T.Buffer((9, 32), "int32"),
    output_f32: T.Buffer((9, 32), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    T.ptx["set.lt.u8x4"](output_u32[0, lane], lhs_u32[lane], rhs_u32[lane])
    T.ptx["set.lt.u8x4"](output_u32[1, lane], lhs_u32[lane], rhs_i32[lane])
    T.ptx["set.lt.u8x4"](output_u32[2, lane], lhs_u32[lane], rhs_f32[lane])
    T.ptx["set.lt.u8x4"](output_u32[3, lane], lhs_i32[lane], rhs_u32[lane])
    T.ptx["set.lt.u8x4"](output_u32[4, lane], lhs_i32[lane], rhs_i32[lane])
    T.ptx["set.lt.u8x4"](output_u32[5, lane], lhs_i32[lane], rhs_f32[lane])
    T.ptx["set.lt.u8x4"](output_u32[6, lane], lhs_f32[lane], rhs_u32[lane])
    T.ptx["set.lt.u8x4"](output_u32[7, lane], lhs_f32[lane], rhs_i32[lane])
    T.ptx["set.lt.u8x4"](output_u32[8, lane], lhs_f32[lane], rhs_f32[lane])

    T.ptx["set.lt.u8x4"](output_i32[0, lane], lhs_u32[lane], rhs_u32[lane])
    T.ptx["set.lt.u8x4"](output_i32[1, lane], lhs_u32[lane], rhs_i32[lane])
    T.ptx["set.lt.u8x4"](output_i32[2, lane], lhs_u32[lane], rhs_f32[lane])
    T.ptx["set.lt.u8x4"](output_i32[3, lane], lhs_i32[lane], rhs_u32[lane])
    T.ptx["set.lt.u8x4"](output_i32[4, lane], lhs_i32[lane], rhs_i32[lane])
    T.ptx["set.lt.u8x4"](output_i32[5, lane], lhs_i32[lane], rhs_f32[lane])
    T.ptx["set.lt.u8x4"](output_i32[6, lane], lhs_f32[lane], rhs_u32[lane])
    T.ptx["set.lt.u8x4"](output_i32[7, lane], lhs_f32[lane], rhs_i32[lane])
    T.ptx["set.lt.u8x4"](output_i32[8, lane], lhs_f32[lane], rhs_f32[lane])

    T.ptx["set.lt.u8x4"](output_f32[0, lane], lhs_u32[lane], rhs_u32[lane])
    T.ptx["set.lt.u8x4"](output_f32[1, lane], lhs_u32[lane], rhs_i32[lane])
    T.ptx["set.lt.u8x4"](output_f32[2, lane], lhs_u32[lane], rhs_f32[lane])
    T.ptx["set.lt.u8x4"](output_f32[3, lane], lhs_i32[lane], rhs_u32[lane])
    T.ptx["set.lt.u8x4"](output_f32[4, lane], lhs_i32[lane], rhs_i32[lane])
    T.ptx["set.lt.u8x4"](output_f32[5, lane], lhs_i32[lane], rhs_f32[lane])
    T.ptx["set.lt.u8x4"](output_f32[6, lane], lhs_f32[lane], rhs_u32[lane])
    T.ptx["set.lt.u8x4"](output_f32[7, lane], lhs_f32[lane], rhs_i32[lane])
    T.ptx["set.lt.u8x4"](output_f32[8, lane], lhs_f32[lane], rhs_f32[lane])


_COMPARISONS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "lo": operator.lt,
    "ls": operator.le,
    "hi": operator.gt,
    "hs": operator.ge,
}


def _lane_value(word: int, lane: int, width: int, *, signed: bool) -> int:
    value = (word >> (lane * width)) & ((1 << width) - 1)
    if signed and value & (1 << (width - 1)):
        value -= 1 << width
    return value


def _set_packed_reference(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    ptx_type: str,
    comparison: str,
) -> np.ndarray:
    width = 8 if ptx_type.endswith("8x4") else 16
    lanes = 32 // width
    lane_mask = (1 << width) - 1
    signed = ptx_type.startswith("s")
    compare = _COMPARISONS[comparison]
    results: list[int] = []
    for lhs_word, rhs_word in zip(lhs, rhs, strict=True):
        packed = 0
        for lane in range(lanes):
            lhs_value = _lane_value(int(lhs_word), lane, width, signed=signed)
            rhs_value = _lane_value(int(rhs_word), lane, width, signed=signed)
            if compare(lhs_value, rhs_value):
                packed |= lane_mask << (lane * width)
        results.append(packed)
    return np.asarray(results, dtype=np.uint32)


def _semantic_inputs() -> tuple[np.ndarray, np.ndarray]:
    edge_lhs = (
        0x8000_7FFF,
        0x7FFF_8000,
        0xFFFF_0000,
        0x0000_FFFF,
        0x80FF_7F00,
        0x7F00_80FF,
        0x1234_5678,
        0xFEDC_BA98,
    )
    edge_rhs = (
        0x7FFF_8000,
        0x8000_7FFF,
        0x0000_FFFF,
        0xFFFF_0000,
        0x7F00_80FF,
        0x80FF_7F00,
        0x1234_9ABC,
        0x7654_BA98,
    )
    lhs = np.asarray(
        (*edge_lhs, *((0x9E37_79B9 * lane + 0x80FF_7F00) & 0xFFFF_FFFF for lane in range(8, 32))),
        dtype=np.uint32,
    )
    rhs = np.asarray(
        (
            *edge_rhs,
            *((0x7F4A_7C15 * (lane + 3) ^ 0x7FFF_8000) & 0xFFFF_FFFF for lane in range(8, 32)),
        ),
        dtype=np.uint32,
    )
    return lhs, rhs


def test_ptx_set_packed_all_legal_forms_match_independent_lane_oracle(tmp_path):
    lhs, rhs = _semantic_inputs()
    result = numsim.Engine().run(
        numsim.transpile(ptx_set_packed_semantics, cache_dir=tmp_path),
        {
            "lhs": lhs,
            "rhs": rhs,
            "output": np.full((32, 32), np.uint32(0xDEAD_BEEF), dtype=np.uint32),
        },
        outputs=("output",),
    )
    expected = np.stack(
        [
            _set_packed_reference(lhs, rhs, ptx_type=ptx_type, comparison=comparison)
            for ptx_type, comparison in _PACKED_FORMS
        ]
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_ptx_set_packed_preserves_every_b32_carrier_cross_product(tmp_path):
    base_lhs, base_rhs = _semantic_inputs()
    lhs_words = (
        base_lhs,
        np.bitwise_xor(base_lhs, np.uint32(0x00FF_8001)),
        np.bitwise_xor(base_lhs, np.uint32(0x3F00_007F)),
    )
    rhs_words = (
        base_rhs,
        np.bitwise_xor(base_rhs, np.uint32(0x7F00_FF80)),
        np.bitwise_xor(base_rhs, np.uint32(0x0080_3F00)),
    )
    result = numsim.Engine().run(
        numsim.transpile(ptx_set_packed_b32_carriers, cache_dir=tmp_path),
        {
            "lhs_u32": lhs_words[0],
            "lhs_i32": lhs_words[1].view(np.int32),
            "lhs_f32": lhs_words[2].view(np.float32),
            "rhs_u32": rhs_words[0],
            "rhs_i32": rhs_words[1].view(np.int32),
            "rhs_f32": rhs_words[2].view(np.float32),
            "output_u32": np.full((9, 32), np.uint32(0xDEAD_BEEF), dtype=np.uint32),
            "output_i32": np.full((9, 32), np.uint32(0xDEAD_BEEF), dtype=np.uint32).view(np.int32),
            "output_f32": np.full((9, 32), np.uint32(0xDEAD_BEEF), dtype=np.uint32).view(
                np.float32
            ),
        },
        outputs=("output_u32", "output_i32", "output_f32"),
    )
    expected = np.stack(
        [
            _set_packed_reference(lhs, rhs, ptx_type="u8x4", comparison="lt")
            for lhs in lhs_words
            for rhs in rhs_words
        ]
    )
    np.testing.assert_array_equal(result.outputs["output_u32"], expected)
    np.testing.assert_array_equal(result.outputs["output_i32"].view(np.uint32), expected)
    np.testing.assert_array_equal(result.outputs["output_f32"].view(np.uint32), expected)


__all__ = ["ptx_set_packed_b32_carriers", "ptx_set_packed_semantics"]
