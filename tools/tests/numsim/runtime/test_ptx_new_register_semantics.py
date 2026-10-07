from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def ptx_new_register_semantics(
    lhs_i32: T.Buffer((32,), "int32"),
    rhs_i32: T.Buffer((32,), "int32"),
    mul_i32: T.Buffer((32,), "int32"),
    sub_i32: T.Buffer((32,), "int32"),
    f16_lhs: T.Buffer((32,), "uint32"),
    f16_rhs: T.Buffer((32,), "uint32"),
    f16_addend: T.Buffer((32,), "uint32"),
    f16_out: T.Buffer((32,), "uint32"),
    bf16_lhs: T.Buffer((32,), "uint32"),
    bf16_rhs: T.Buffer((32,), "uint32"),
    bf16_addend: T.Buffer((32,), "uint32"),
    bf16_out: T.Buffer((32,), "uint32"),
    bit_lhs: T.Buffer((32,), "uint32"),
    bit_rhs: T.Buffer((32,), "uint32"),
    bit_or: T.Buffer((32,), "uint32"),
    prmt_a: T.Buffer((32,), "uint32"),
    prmt_b: T.Buffer((32,), "uint32"),
    prmt_selector: T.Buffer((32,), "uint32"),
    prmt_out: T.Buffer((32,), "uint32"),
    mul_hi_lhs: T.Buffer((32,), "uint32"),
    mul_hi_rhs: T.Buffer((32,), "uint32"),
    mul_hi_out: T.Buffer((32,), "uint32"),
    xor_lhs: T.Buffer((32,), "int32"),
    xor_rhs: T.Buffer((32,), "int32"),
    xor_out: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.mul.lo.s32(mul_i32[lane], lhs_i32[lane], rhs_i32[lane])
    T.ptx.sub.s32(sub_i32[lane], lhs_i32[lane], rhs_i32[lane])
    T.ptx.fma.rn.f16x2(f16_out[lane], f16_lhs[lane], f16_rhs[lane], f16_addend[lane])
    T.ptx.fma.rn.bf16x2(bf16_out[lane], bf16_lhs[lane], bf16_rhs[lane], bf16_addend[lane])
    T.ptx.or_.b32(bit_or[lane], bit_lhs[lane], bit_rhs[lane])
    T.ptx.prmt.b32(prmt_out[lane], prmt_a[lane], prmt_b[lane], prmt_selector[lane])
    T.ptx.mul.hi.u32(mul_hi_out[lane], mul_hi_lhs[lane], mul_hi_rhs[lane])
    T.ptx.xor.b32(xor_out[lane], xor_lhs[lane], xor_rhs[lane])


@T.prim_func
def ptx_scalar_moves_and_mad_lo_s64(
    predicates: T.Buffer((32,), "bool"),
    pred_out: T.Buffer((32,), "uint32"),
    b16_source: T.Buffer((32,), "uint16"),
    b16_out: T.Buffer((32,), "uint16"),
    b64_source: T.Buffer((32,), "uint64"),
    b64_out: T.Buffer((32,), "uint64"),
    mad_a: T.Buffer((32,), "int64"),
    mad_b: T.Buffer((32,), "int64"),
    mad_c: T.Buffer((32,), "int64"),
    mad_out: T.Buffer((32,), "int64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.mov.pred(pred_out[lane], predicates[lane])
    T.ptx.mov.b16(b16_out[lane], b16_source[lane])
    T.ptx.mov.b64(b64_out[lane], b64_source[lane])
    T.ptx.mad.lo.s64(mad_out[lane], mad_a[lane], mad_b[lane], mad_c[lane])


def _pack16x2(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float16).view(np.uint32).reshape(32)


def _pack_bf16x2(values: np.ndarray) -> np.ndarray:
    f32 = np.asarray(values, dtype=np.float32)
    bits = ((f32.view(np.uint32) + np.uint32(0x7FFF)) >> np.uint32(16)).astype(np.uint16)
    return bits.view(np.uint32).reshape(32)


def _unpack16x2(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.uint32).view(np.uint16).reshape(32, 2)


def _prmt_reference(a: np.ndarray, b: np.ndarray, selector: np.ndarray) -> np.ndarray:
    """Independent generic PTX ``prmt.b32`` oracle, including sign extension."""

    output = np.zeros_like(selector, dtype=np.uint32)
    for lane, (a_value, b_value, control) in enumerate(zip(a, b, selector, strict=True)):
        a_value = int(a_value)
        b_value = int(b_value)
        control = int(control)
        result = 0
        for output_byte in range(4):
            select = (control >> (output_byte * 4)) & 0xF
            source_byte = select & 0x7
            source = a_value if source_byte < 4 else b_value
            byte = (source >> ((source_byte & 0x3) * 8)) & 0xFF
            if select & 0x8:
                byte = 0xFF if byte & 0x80 else 0
            result |= byte << (output_byte * 8)
        output[lane] = result
    return output


def test_ptx_new_register_semantics_match_independent_oracles(tmp_path):
    lane = np.arange(32, dtype=np.float32)
    lhs_i32 = (lane.astype(np.int32) - 11).astype(np.int32)
    rhs_i32 = (lane.astype(np.int32) % 7 - 3).astype(np.int32)

    f16_lhs = np.stack((lane / 7.0 - 1.5, lane / 11.0 + 0.25), axis=1).astype(np.float16)
    f16_rhs = np.stack(((lane % 5) / 9.0, (lane % 7) / 13.0 - 0.2), axis=1).astype(np.float16)
    f16_addend = np.stack((lane / 17.0, -lane / 19.0), axis=1).astype(np.float16)
    f16_expected = (
        f16_lhs.astype(np.float32) * f16_rhs.astype(np.float32) + f16_addend.astype(np.float32)
    ).astype(np.float16)

    bf16_lhs = np.stack((lane / 7.0 - 1.5, lane / 11.0 + 0.25), axis=1)
    bf16_rhs = np.stack(((lane % 5) / 9.0, (lane % 7) / 13.0 - 0.2), axis=1)
    bf16_addend = np.stack((lane / 17.0, -lane / 19.0), axis=1)
    bf16_lhs_bits = _unpack16x2(_pack_bf16x2(bf16_lhs))
    bf16_rhs_bits = _unpack16x2(_pack_bf16x2(bf16_rhs))
    bf16_addend_bits = _unpack16x2(_pack_bf16x2(bf16_addend))
    bf16_lhs_narrow = (bf16_lhs_bits.astype(np.uint32) << 16).view(np.float32)
    bf16_rhs_narrow = (bf16_rhs_bits.astype(np.uint32) << 16).view(np.float32)
    bf16_addend_narrow = (bf16_addend_bits.astype(np.uint32) << 16).view(np.float32)
    bf16_expected_f32 = (bf16_lhs_narrow * bf16_rhs_narrow + bf16_addend_narrow).astype(np.float32)
    bf16_expected = _pack_bf16x2(bf16_expected_f32)

    bit_lhs = (np.uint32(0x13579BDF) + lane.astype(np.uint32) * np.uint32(0x10101)).astype(
        np.uint32
    )
    bit_rhs = np.uint32(0xA5A5_5A5A) ^ lane.astype(np.uint32)
    prmt_a = np.full(32, np.uint32(0x11223344), dtype=np.uint32)
    prmt_b = np.full(32, np.uint32(0xAABBCCDD), dtype=np.uint32)
    selectors = np.resize(np.array([0x1302, 0x8FED, 0x5410, 0x7654], dtype=np.uint32), 32)
    prmt_expected = _prmt_reference(prmt_a, prmt_b, selectors)
    mul_hi_lhs = np.uint32(0xFFFF_FF00) - lane.astype(np.uint32) * np.uint32(0x10101)
    mul_hi_rhs = np.uint32(0x9E37_79B9) + lane.astype(np.uint32) * np.uint32(17)
    mul_hi_expected = (
        (mul_hi_lhs.astype(np.uint64) * mul_hi_rhs.astype(np.uint64)) >> np.uint64(32)
    ).astype(np.uint32)
    xor_lhs = (np.uint32(0x8000_0000) | lane.astype(np.uint32)).view(np.int32)
    xor_rhs = (np.uint32(0x5A5A_0000) | (lane.astype(np.uint32) << np.uint32(8))).view(np.int32)

    module = numsim.transpile(ptx_new_register_semantics, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "lhs_i32": lhs_i32,
            "rhs_i32": rhs_i32,
            "mul_i32": np.zeros(32, dtype=np.int32),
            "sub_i32": np.zeros(32, dtype=np.int32),
            "f16_lhs": _pack16x2(f16_lhs),
            "f16_rhs": _pack16x2(f16_rhs),
            "f16_addend": _pack16x2(f16_addend),
            "f16_out": np.zeros(32, dtype=np.uint32),
            "bf16_lhs": _pack_bf16x2(bf16_lhs),
            "bf16_rhs": _pack_bf16x2(bf16_rhs),
            "bf16_addend": _pack_bf16x2(bf16_addend),
            "bf16_out": np.zeros(32, dtype=np.uint32),
            "bit_lhs": bit_lhs,
            "bit_rhs": bit_rhs,
            "bit_or": np.zeros(32, dtype=np.uint32),
            "prmt_a": prmt_a,
            "prmt_b": prmt_b,
            "prmt_selector": selectors,
            "prmt_out": np.zeros(32, dtype=np.uint32),
            "mul_hi_lhs": mul_hi_lhs,
            "mul_hi_rhs": mul_hi_rhs,
            "mul_hi_out": np.zeros(32, dtype=np.uint32),
            "xor_lhs": xor_lhs,
            "xor_rhs": xor_rhs,
            "xor_out": np.zeros(32, dtype=np.int32),
        },
    )
    np.testing.assert_array_equal(result.outputs["mul_i32"], lhs_i32 * rhs_i32)
    np.testing.assert_array_equal(result.outputs["sub_i32"], lhs_i32 - rhs_i32)
    np.testing.assert_array_equal(result.outputs["f16_out"], _pack16x2(f16_expected))
    np.testing.assert_array_equal(result.outputs["bf16_out"], bf16_expected)
    np.testing.assert_array_equal(result.outputs["bit_or"], bit_lhs | bit_rhs)
    np.testing.assert_array_equal(result.outputs["prmt_out"], prmt_expected)
    np.testing.assert_array_equal(result.outputs["mul_hi_out"], mul_hi_expected)
    np.testing.assert_array_equal(result.outputs["xor_out"], np.bitwise_xor(xor_lhs, xor_rhs))


def test_ptx_scalar_moves_and_mad_lo_s64_match_bit_and_wrap_oracles(tmp_path):
    lanes = np.arange(32, dtype=np.uint64)
    predicates = lanes % np.uint64(5) < np.uint64(2)
    b16_source = (np.uint16(0x8000) | lanes.astype(np.uint16) * np.uint16(0x0101)).astype(np.uint16)
    b64_source = np.uint64(0x8000_0000_0000_0000) | (lanes * np.uint64(0x0102_0408_1020_4081))
    mad_a = (np.uint64(0xFFFF_FFFF_FFFF_FF00) - lanes * np.uint64(0x0101)).view(np.int64)
    mad_b = (np.uint64(0x4000_0000_0000_0001) + lanes * np.uint64(17)).view(np.int64)
    mad_c = (np.uint64(0x7FFF_FFFF_FFFF_F000) + lanes * np.uint64(0x101)).view(np.int64)
    mad_expected = np.array(
        [
            ((int(a) * int(b) + int(c)) & 0xFFFF_FFFF_FFFF_FFFF) - (1 << 64)
            if ((int(a) * int(b) + int(c)) & (1 << 63))
            else ((int(a) * int(b) + int(c)) & 0xFFFF_FFFF_FFFF_FFFF)
            for a, b, c in zip(mad_a, mad_b, mad_c, strict=True)
        ],
        dtype=np.int64,
    )

    result = numsim.Engine().run(
        numsim.transpile(ptx_scalar_moves_and_mad_lo_s64, cache_dir=tmp_path),
        {
            "predicates": predicates,
            "pred_out": np.zeros(32, dtype=np.uint32),
            "b16_source": b16_source,
            "b16_out": np.zeros(32, dtype=np.uint16),
            "b64_source": b64_source,
            "b64_out": np.zeros(32, dtype=np.uint64),
            "mad_a": mad_a,
            "mad_b": mad_b,
            "mad_c": mad_c,
            "mad_out": np.zeros(32, dtype=np.int64),
        },
    )

    np.testing.assert_array_equal(result.outputs["pred_out"], predicates.astype(np.uint32))
    np.testing.assert_array_equal(result.outputs["b16_out"], b16_source)
    np.testing.assert_array_equal(result.outputs["b64_out"], b64_source)
    np.testing.assert_array_equal(result.outputs["mad_out"], mad_expected)
