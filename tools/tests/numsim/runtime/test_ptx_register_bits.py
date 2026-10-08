from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def ptx_mov_b128_pack_store(
    low: T.Buffer((32,), "uint64"),
    high: T.Buffer((32,), "uint64"),
    output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed = T.alloc_local((1,), "uint128")
    T.ptx.mov.b128(packed[0], low[lane], high[lane])
    T.ptx["st.global.b128"](output.ptr_to([lane, 0]), packed[0])


@T.prim_func
def ptx_register_bits(
    input_u32: T.Buffer((32, 2), "uint32"),
    input_i32: T.Buffer((32, 2), "int32"),
    input_u64: T.Buffer((32, 2), "uint64"),
    input_f32: T.Buffer((32, 3), "float32"),
    output_u32: T.Buffer((32, 3), "uint32"),
    output_i32: T.Buffer((32,), "int32"),
    output_u64: T.Buffer((32, 3), "uint64"),
    output_f32: T.Buffer((32, 3), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed_u32 = T.alloc_local((1,), "uint64")
    packed_f32 = T.alloc_local((1,), "uint64")

    T.evaluate(T.ptx.mov.b64(packed_u32[0], input_u32[lane, 0], input_u32[lane, 1]))
    output_u64[lane, 0] = packed_u32[0]
    T.evaluate(T.ptx.mov.b64(output_u32[lane, 0], output_u32[lane, 1], packed_u32[0]))

    T.evaluate(T.ptx.mov.b64(packed_f32[0], input_f32[lane, 0], input_f32[lane, 1]))
    output_u64[lane, 1] = packed_f32[0]
    T.evaluate(T.ptx.mov.b64(output_f32[lane, 0], output_f32[lane, 1], packed_f32[0]))

    T.evaluate(T.ptx.add.u32(output_u32[lane, 2], input_u32[lane, 0], input_u32[lane, 1]))
    T.evaluate(T.ptx.add.s32(output_i32[lane], input_i32[lane, 0], input_i32[lane, 1]))
    T.evaluate(T.ptx.xor.b64(output_u64[lane, 2], input_u64[lane, 0], input_u64[lane, 1]))
    T.evaluate(T.ptx.abs.f32(output_f32[lane, 2], input_f32[lane, 2]))


@T.prim_func
def ptx_scalar_move_and_shift_bits(
    input_f32: T.Buffer((32,), "float32"),
    input_i32: T.Buffer((32,), "int32"),
    input_u32: T.Buffer((32,), "uint32"),
    shifts: T.Buffer((32,), "uint32"),
    moved_i32: T.Buffer((32,), "int32"),
    roundtrip_f32: T.Buffer((32,), "float32"),
    shifted_i32: T.Buffer((32,), "int32"),
    shifted_u32: T.Buffer((32,), "uint32"),
    shifted_f32: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    moved = T.alloc_local((1,), "int32")

    T.evaluate(T.ptx.mov.b32(moved[0], input_f32[lane]))
    moved_i32[lane] = moved[0]
    T.evaluate(T.ptx.mov.b32(roundtrip_f32[lane], moved[0]))
    T.evaluate(T.ptx.shl.b32(shifted_i32[lane], input_i32[lane], shifts[lane]))
    T.evaluate(T.ptx.shl.b32(shifted_u32[lane], input_u32[lane], shifts[lane]))
    T.evaluate(T.ptx.shl.b32(shifted_f32[lane], input_f32[lane], shifts[lane]))


@T.prim_func
def ptx_scalar_shift_right_bits(
    input_i32: T.Buffer((32,), "int32"),
    shifts: T.Buffer((32,), "uint32"),
    shifted_i32: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.ptx.shr.s32(shifted_i32[lane], input_i32[lane], shifts[lane]))


@T.prim_func
def ptx_packed_f16_minmax(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    maximum: T.Buffer((32,), "uint32"),
    minimum: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.max.f16x2(maximum[lane], lhs[lane], rhs[lane])
    T.ptx.min.f16x2(minimum[lane], lhs[lane], rhs[lane])


@T.prim_func
def ptx_predicate_data_path(
    lhs_f32: T.Buffer((32,), "float32"),
    lhs_u32: T.Buffer((32,), "uint32"),
    selected_u32: T.Buffer((32,), "uint32"),
    selected_f32: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    p_float = T.alloc_local((1,), "uint32")
    p_integer = T.alloc_local((1,), "uint32")
    p_both = T.alloc_local((1,), "uint32")
    T.ptx.setp.le.f32(p_float[0], lhs_f32[lane], T.float32(0))
    T.ptx.setp.ne.u32(p_integer[0], lhs_u32[lane], T.uint32(0))
    T.ptx.and_.pred(p_both[0], T.ptx.pred(p_float[0]), T.ptx.pred(p_integer[0]))
    T.ptx.selp.u32(selected_u32[lane], lhs_u32[lane], T.uint32(99), T.ptx.pred(p_both[0]))
    T.ptx.selp.f32(selected_f32[lane], lhs_f32[lane], T.float32(7), T.ptx.pred(p_both[0]))


@T.prim_func
def ptx_predicate_or(
    lhs: T.Buffer((32,), "uint32"),
    rhs: T.Buffer((32,), "uint32"),
    result: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.or_.pred(result[lane], T.ptx.pred(lhs[lane]), T.ptx.pred(rhs[lane]))


@T.prim_func
def ptx_setp_nan_f32(
    lhs: T.Buffer((32,), "float32"),
    rhs: T.Buffer((32,), "float32"),
    nan_result: T.Buffer((32,), "uint32"),
    equ_result: T.Buffer((32,), "uint32"),
    neu_result: T.Buffer((32,), "uint32"),
    ne_result: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.setp.nan.f32(nan_result[lane], lhs[lane], rhs[lane])
    T.ptx.setp.equ.f32(equ_result[lane], lhs[lane], rhs[lane])
    T.ptx.setp.neu.f32(neu_result[lane], lhs[lane], rhs[lane])
    T.ptx.setp.ne.f32(ne_result[lane], lhs[lane], rhs[lane])


@T.prim_func
def ptx_selp_b16_bits(
    predicate: T.Buffer((32,), "uint32"),
    on_true: T.Buffer((32,), "uint16"),
    on_false: T.Buffer((32,), "uint16"),
    selected: T.Buffer((32,), "uint16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.selp.b16(selected[lane], on_true[lane], on_false[lane], T.ptx.pred(predicate[lane]))


@T.prim_func
def ptx_selp_b32_signed_bits(
    predicate: T.Buffer((32,), "uint32"),
    on_true: T.Buffer((32,), "int32"),
    on_false: T.Buffer((32,), "int32"),
    selected: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.selp.b32(selected[lane], on_true[lane], on_false[lane], T.ptx.pred(predicate[lane]))


@T.prim_func
def ptx_half_abs_and_setp(
    packed: T.Buffer((32,), "uint32"),
    lhs: T.Buffer((32,), "uint16"),
    rhs: T.Buffer((32,), "uint16"),
    absolute: T.Buffer((32,), "uint32"),
    greater: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.abs.f16x2(absolute[lane], packed[lane])
    T.ptx.setp.gt.f16(greater[lane], lhs[lane], rhs[lane])


@T.prim_func
def ptx_typed_move_and_b16_pack(
    low: T.Buffer((32,), "uint16"),
    high: T.Buffer((32,), "uint16"),
    source_i32: T.Buffer((32,), "int32"),
    packed: T.Buffer((32,), "uint32"),
    unpacked_low: T.Buffer((32,), "uint16"),
    unpacked_high: T.Buffer((32,), "uint16"),
    moved_i32: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.mov.b32(packed[lane], low[lane], high[lane])
    T.ptx.mov.b32(unpacked_low[lane], unpacked_high[lane], packed[lane])
    T.ptx.mov.s32(moved_i32[lane], source_i32[lane])


@T.prim_func
def ptx_latest_canonical_scalar_forms(
    lhs_i32: T.Buffer((32,), "int32"),
    rhs_i32: T.Buffer((32,), "int32"),
    lhs_bf16: T.Buffer((32,), "uint16"),
    rhs_bf16: T.Buffer((32,), "uint16"),
    addend: T.Buffer((32,), "float32"),
    unary_input: T.Buffer((32,), "float32"),
    divisor: T.Buffer((32,), "float32"),
    maximum: T.Buffer((32,), "int32"),
    mixed_add: T.Buffer((32,), "float32"),
    mixed_sub: T.Buffer((32,), "float32"),
    mixed_fma: T.Buffer((32,), "float32"),
    negated: T.Buffer((32,), "float32"),
    quotient: T.Buffer((32,), "float32"),
    quotient_rn: T.Buffer((32,), "float32"),
    reciprocal: T.Buffer((32,), "float32"),
    inverse_sqrt: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["max.s32"](maximum[lane], lhs_i32[lane], rhs_i32[lane])
    T.ptx["add.rn.f32.bf16"](mixed_add[lane], lhs_bf16[lane], addend[lane])
    T.ptx["sub.rn.f32.bf16"](mixed_sub[lane], lhs_bf16[lane], addend[lane])
    T.ptx["fma.rn.f32.bf16"](mixed_fma[lane], lhs_bf16[lane], rhs_bf16[lane], addend[lane])
    T.ptx["neg.ftz.f32"](negated[lane], unary_input[lane])
    T.ptx["div.approx.ftz.f32"](quotient[lane], unary_input[lane], divisor[lane])
    T.ptx["div.rn.f32"](quotient_rn[lane], unary_input[lane], divisor[lane])
    T.ptx["rcp.rn.f32"](reciprocal[lane], divisor[lane])
    T.ptx["rsqrt.approx.f32"](inverse_sqrt[lane], divisor[lane])


def test_ptx_selp_b16_selects_raw_halfword_bits(tmp_path):
    lanes = np.arange(32, dtype=np.uint16)
    predicate = (lanes & np.uint16(1)).astype(np.uint32)
    on_true = lanes ^ np.uint16(0x8000)
    on_false = lanes ^ np.uint16(0x7C00)
    module = numsim.transpile(ptx_selp_b16_bits, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "predicate": predicate,
            "on_true": on_true,
            "on_false": on_false,
            "selected": np.zeros(32, dtype=np.uint16),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["selected"], np.where(predicate != 0, on_true, on_false)
    )


def test_ptx_selp_b32_selects_signed_carrier_bits(tmp_path):
    lanes = np.arange(32, dtype=np.int64)
    predicate = (lanes & 1).astype(np.uint32)
    on_true = (lanes * 0x1020304 - 0x76543210).astype(np.int32)
    on_false = (0x6A5B4C3D - lanes * 0x01020304).astype(np.int32)

    result = numsim.Engine().run(
        numsim.transpile(ptx_selp_b32_signed_bits, cache_dir=tmp_path),
        {
            "predicate": predicate,
            "on_true": on_true,
            "on_false": on_false,
            "selected": np.zeros(32, dtype=np.int32),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["selected"], np.where(predicate != 0, on_true, on_false)
    )


def test_ptx_half_abs_and_setp_match_bitwise_numpy_oracle(tmp_path):
    low = np.linspace(-4.0, 3.75, 32, dtype=np.float16)
    high = np.linspace(7.75, -7.75, 32, dtype=np.float16)
    low_bits = low.view(np.uint16)
    high_bits = high.view(np.uint16)
    packed = low_bits.astype(np.uint32) | (high_bits.astype(np.uint32) << np.uint32(16))
    rhs = np.linspace(-2.0, 2.0, 32, dtype=np.float16)
    module = numsim.transpile(ptx_half_abs_and_setp, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {
            "packed": packed,
            "lhs": low_bits,
            "rhs": rhs.view(np.uint16),
            "absolute": np.zeros(32, dtype=np.uint32),
            "greater": np.zeros(32, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["absolute"], packed & np.uint32(0x7FFF7FFF))
    np.testing.assert_array_equal(result.outputs["greater"], (low > rhs).astype(np.uint32))


def test_ptx_register_bit_ops_match_ptx_value_and_bit_semantics(tmp_path):
    lanes = np.arange(32, dtype=np.uint32)
    input_u32 = np.stack(
        (
            lanes * np.uint32(0x1020304) + np.uint32(0x89ABCDEF),
            np.uint32(0xFEDCBA98) - lanes * np.uint32(0x01010101),
        ),
        axis=1,
    )
    input_i32 = (
        np.stack(
            (
                np.arange(np.iinfo(np.int32).max - 15, np.iinfo(np.int32).max + 17),
                np.arange(32) * 17 - 23,
            ),
            axis=1,
        )
        .astype(np.int64)
        .astype(np.int32)
    )
    input_u64 = np.stack(
        (
            lanes.astype(np.uint64) * np.uint64(0x0102030405060708),
            np.uint64(0xFEDCBA9876543210) - lanes.astype(np.uint64),
        ),
        axis=1,
    )
    input_f32_bits = np.stack(
        (
            lanes * np.uint32(0x01010101) ^ np.uint32(0x80000000),
            lanes * np.uint32(0x00110011) ^ np.uint32(0x7FC00000),
            lanes * np.uint32(0x00010001) ^ np.uint32(0xFFC00000),
        ),
        axis=1,
    )
    input_f32 = input_f32_bits.view(np.float32)

    module = numsim.transpile(ptx_register_bits, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "input_u32": input_u32,
            "input_i32": input_i32,
            "input_u64": input_u64,
            "input_f32": input_f32,
            "output_u32": np.zeros((32, 3), dtype=np.uint32),
            "output_i32": np.zeros(32, dtype=np.int32),
            "output_u64": np.zeros((32, 3), dtype=np.uint64),
            "output_f32": np.zeros((32, 3), dtype=np.float32),
        },
    )

    expected_packed_u32 = input_u32[:, 0].astype(np.uint64) | (
        input_u32[:, 1].astype(np.uint64) << np.uint64(32)
    )
    expected_packed_f32 = input_f32_bits[:, 0].astype(np.uint64) | (
        input_f32_bits[:, 1].astype(np.uint64) << np.uint64(32)
    )
    expected_i32 = (input_i32[:, 0].astype(np.uint32) + input_i32[:, 1].astype(np.uint32)).astype(
        np.int32
    )

    np.testing.assert_array_equal(result.outputs["output_u64"][:, 0], expected_packed_u32)
    np.testing.assert_array_equal(result.outputs["output_u64"][:, 1], expected_packed_f32)
    np.testing.assert_array_equal(
        result.outputs["output_u64"][:, 2],
        np.bitwise_xor(input_u64[:, 0], input_u64[:, 1]),
    )
    np.testing.assert_array_equal(result.outputs["output_u32"][:, :2], input_u32)
    np.testing.assert_array_equal(
        result.outputs["output_u32"][:, 2], input_u32[:, 0] + input_u32[:, 1]
    )
    np.testing.assert_array_equal(result.outputs["output_i32"], expected_i32)
    np.testing.assert_array_equal(
        result.outputs["output_f32"][:, :2].view(np.uint32), input_f32_bits[:, :2]
    )
    np.testing.assert_array_equal(
        result.outputs["output_f32"][:, 2].view(np.uint32),
        input_f32_bits[:, 2] & np.uint32(0x7FFFFFFF),
    )


def test_ptx_b128_move_packs_two_uint64_halves_without_truncation(tmp_path):
    lanes = np.arange(32, dtype=np.uint64)
    low = (lanes * np.uint64(0x0102030405060708)) ^ np.uint64(0x0123456789ABCDEF)
    high = (lanes * np.uint64(0x1020304050607080)) ^ np.uint64(0xFEDCBA9876543210)

    module = numsim.transpile(ptx_mov_b128_pack_store, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"low": low, "high": high, "output": np.zeros((32, 4), dtype=np.uint32)},
    )

    words = result.outputs["output"].view(np.uint64).reshape(32, 2)
    np.testing.assert_array_equal(words[:, 0], low)
    np.testing.assert_array_equal(words[:, 1], high)


def test_ptx_scalar_move_and_shift_match_bitwise_numpy_oracle(tmp_path):
    lanes = np.arange(32, dtype=np.uint32)
    input_f32_bits = lanes * np.uint32(0x01010101) ^ np.uint32(0xFFC00000)
    input_f32 = input_f32_bits.view(np.float32)
    input_i32_bits = lanes * np.uint32(0x1020304) ^ np.uint32(0x89ABCDEF)
    input_i32 = input_i32_bits.view(np.int32)
    input_u32 = lanes * np.uint32(0x11111111) ^ np.uint32(0xFEDCBA98)
    shifts = np.resize(np.array([0, 1, 23, 31, 32, 33, 63], dtype=np.uint32), 32)

    module = numsim.transpile(ptx_scalar_move_and_shift_bits, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "input_f32": input_f32,
            "input_i32": input_i32,
            "input_u32": input_u32,
            "shifts": shifts,
            "moved_i32": np.zeros(32, dtype=np.int32),
            "roundtrip_f32": np.zeros(32, dtype=np.float32),
            "shifted_i32": np.zeros(32, dtype=np.int32),
            "shifted_u32": np.zeros(32, dtype=np.uint32),
            "shifted_f32": np.zeros(32, dtype=np.float32),
        },
    )

    def shl_b32(values: np.ndarray) -> np.ndarray:
        expected = np.zeros(32, dtype=np.uint32)
        in_range = shifts < np.uint32(32)
        expected[in_range] = (
            values[in_range].astype(np.uint64) << shifts[in_range].astype(np.uint64)
        ).astype(np.uint32)
        return expected

    np.testing.assert_array_equal(result.outputs["moved_i32"].view(np.uint32), input_f32_bits)
    np.testing.assert_array_equal(result.outputs["roundtrip_f32"].view(np.uint32), input_f32_bits)
    np.testing.assert_array_equal(
        result.outputs["shifted_i32"].view(np.uint32), shl_b32(input_i32_bits)
    )
    np.testing.assert_array_equal(result.outputs["shifted_u32"], shl_b32(input_u32))
    np.testing.assert_array_equal(
        result.outputs["shifted_f32"].view(np.uint32), shl_b32(input_f32_bits)
    )


def test_ptx_scalar_shift_right_matches_signed_numpy_oracle(tmp_path):
    values = np.array([0, 1, -1, 7, -7, 0x40000000, -0x40000000, -0x80000000], dtype=np.int32)
    values = np.resize(values, 32)
    shifts = np.resize(np.array([0, 1, 3, 31, 32, 33], dtype=np.uint32), 32)
    module = numsim.transpile(ptx_scalar_shift_right_bits, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "input_i32": values,
            "shifts": shifts,
            "shifted_i32": np.zeros(32, dtype=np.int32),
        },
    )

    expected = np.where(shifts < 32, values >> shifts, values >> 31).astype(np.int32)
    np.testing.assert_array_equal(result.outputs["shifted_i32"], expected)


def test_ptx_packed_f16_minmax_matches_componentwise_numpy_oracle(tmp_path):
    lanes = np.arange(32, dtype=np.float16)
    lhs_values = np.stack((lanes - np.float16(20), lanes / np.float16(4) + 1), axis=1)
    rhs_values = np.stack((lanes / np.float16(2) - 10, 20 - lanes), axis=1)

    def pack(values: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(values).view(np.uint16).reshape(32, 2)
        return bits[:, 0].astype(np.uint32) | (bits[:, 1].astype(np.uint32) << np.uint32(16))

    lhs = pack(lhs_values)
    rhs = pack(rhs_values)
    result = numsim.Engine().run(
        numsim.transpile(ptx_packed_f16_minmax, cache_dir=tmp_path),
        {
            "lhs": lhs,
            "rhs": rhs,
            "maximum": np.zeros(32, dtype=np.uint32),
            "minimum": np.zeros(32, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["maximum"], pack(np.maximum(lhs_values, rhs_values))
    )
    np.testing.assert_array_equal(
        result.outputs["minimum"], pack(np.minimum(lhs_values, rhs_values))
    )


def test_ptx_setp_and_predicate_selp_match_numpy_control_oracle(tmp_path):
    lhs_f32 = np.arange(32, dtype=np.float32) - np.float32(16)
    lhs_u32 = (np.arange(32, dtype=np.uint32) % np.uint32(3)).astype(np.uint32)
    predicate = (lhs_f32 <= 0) & (lhs_u32 != 0)
    result = numsim.Engine().run(
        numsim.transpile(ptx_predicate_data_path, cache_dir=tmp_path),
        {
            "lhs_f32": lhs_f32,
            "lhs_u32": lhs_u32,
            "selected_u32": np.zeros(32, dtype=np.uint32),
            "selected_f32": np.zeros(32, dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(result.outputs["selected_u32"], np.where(predicate, lhs_u32, 99))
    np.testing.assert_array_equal(result.outputs["selected_f32"], np.where(predicate, lhs_f32, 7))


def test_ptx_or_pred_matches_boolean_truth_table(tmp_path):
    lanes = np.arange(32, dtype=np.uint32)
    lhs = (lanes & np.uint32(1)).astype(np.uint32)
    rhs = ((lanes >> np.uint32(1)) & np.uint32(1)).astype(np.uint32)

    result = numsim.Engine().run(
        numsim.transpile(ptx_predicate_or, cache_dir=tmp_path),
        {
            "lhs": lhs,
            "rhs": rhs,
            "result": np.zeros(32, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["result"], np.logical_or(lhs != 0, rhs != 0).astype(np.uint32)
    )


def test_ptx_setp_f32_nan_and_unordered_relations_match_ptx(tmp_path):
    lhs = np.linspace(-4.0, 4.0, 32, dtype=np.float32)
    rhs = np.linspace(8.0, -8.0, 32, dtype=np.float32)
    lhs[[1, 7, 19]] = np.float32(np.nan)
    rhs[[2, 7, 23]] = np.float32(np.nan)
    lhs[3] = np.float32(np.inf)
    rhs[4] = np.float32(-np.inf)

    result = numsim.Engine().run(
        numsim.transpile(ptx_setp_nan_f32, cache_dir=tmp_path),
        {
            "lhs": lhs,
            "rhs": rhs,
            "nan_result": np.zeros(32, dtype=np.uint32),
            "equ_result": np.zeros(32, dtype=np.uint32),
            "neu_result": np.zeros(32, dtype=np.uint32),
            "ne_result": np.zeros(32, dtype=np.uint32),
        },
    )

    unordered = np.logical_or(np.isnan(lhs), np.isnan(rhs))
    np.testing.assert_array_equal(result.outputs["nan_result"], unordered.astype(np.uint32))
    np.testing.assert_array_equal(
        result.outputs["equ_result"], np.logical_or(unordered, lhs == rhs).astype(np.uint32)
    )
    np.testing.assert_array_equal(
        result.outputs["neu_result"], np.logical_or(unordered, lhs != rhs).astype(np.uint32)
    )
    np.testing.assert_array_equal(
        result.outputs["ne_result"], np.logical_and(~unordered, lhs != rhs).astype(np.uint32)
    )


def test_ptx_typed_move_and_b16_pack_preserve_source_bits(tmp_path):
    lanes = np.arange(32, dtype=np.uint16)
    low = lanes * np.uint16(257) + np.uint16(3)
    high = np.uint16(0xFFFF) - lanes * np.uint16(131)
    source_i32 = (np.arange(32, dtype=np.int64) * 0x1020304 - 0x76543210).astype(np.int32)
    result = numsim.Engine().run(
        numsim.transpile(ptx_typed_move_and_b16_pack, cache_dir=tmp_path),
        {
            "low": low,
            "high": high,
            "source_i32": source_i32,
            "packed": np.zeros(32, dtype=np.uint32),
            "unpacked_low": np.zeros(32, dtype=np.uint16),
            "unpacked_high": np.zeros(32, dtype=np.uint16),
            "moved_i32": np.zeros(32, dtype=np.int32),
        },
    )

    expected = low.astype(np.uint32) | (high.astype(np.uint32) << np.uint32(16))
    np.testing.assert_array_equal(result.outputs["packed"], expected)
    np.testing.assert_array_equal(result.outputs["unpacked_low"], low)
    np.testing.assert_array_equal(result.outputs["unpacked_high"], high)
    np.testing.assert_array_equal(result.outputs["moved_i32"], source_i32)


def test_latest_canonical_scalar_forms_match_independent_numpy_oracle(tmp_path):
    lane = np.arange(32, dtype=np.int32)
    lhs_i32 = lane * np.int32(11) - np.int32(91)
    rhs_i32 = np.int32(73) - lane * np.int32(7)

    lhs_f32 = ((lane % 9) - 4).astype(np.float32)
    rhs_f32 = ((lane % 7) - 3).astype(np.float32) * np.float32(0.5)
    lhs_bf16 = (lhs_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    rhs_bf16 = (rhs_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    addend = (lane.astype(np.float32) - np.float32(12)) * np.float32(0.25)

    unary_bits = (lane.astype(np.float32) + np.float32(1)).view(np.uint32).copy()
    unary_bits[0] = np.uint32(0x00000001)
    unary_bits[1] = np.uint32(0x80000001)
    unary_input = unary_bits.view(np.float32)
    divisor = np.resize(np.array([2.0, 2.0, 0.25, 1.0, 4.0, 16.0, 64.0], dtype=np.float32), 32)

    outputs = {
        "maximum": np.zeros(32, dtype=np.int32),
        "mixed_add": np.zeros(32, dtype=np.float32),
        "mixed_sub": np.zeros(32, dtype=np.float32),
        "mixed_fma": np.zeros(32, dtype=np.float32),
        "negated": np.zeros(32, dtype=np.float32),
        "quotient": np.zeros(32, dtype=np.float32),
        "quotient_rn": np.zeros(32, dtype=np.float32),
        "reciprocal": np.zeros(32, dtype=np.float32),
        "inverse_sqrt": np.zeros(32, dtype=np.float32),
    }
    result = numsim.Engine().run(
        numsim.transpile(ptx_latest_canonical_scalar_forms, cache_dir=tmp_path),
        {
            "lhs_i32": lhs_i32,
            "rhs_i32": rhs_i32,
            "lhs_bf16": lhs_bf16,
            "rhs_bf16": rhs_bf16,
            "addend": addend,
            "unary_input": unary_input,
            "divisor": divisor,
            **outputs,
        },
    )

    np.testing.assert_array_equal(result.outputs["maximum"], np.maximum(lhs_i32, rhs_i32))
    np.testing.assert_array_equal(result.outputs["mixed_add"], lhs_f32 + addend)
    np.testing.assert_array_equal(result.outputs["mixed_sub"], lhs_f32 - addend)
    np.testing.assert_array_equal(result.outputs["mixed_fma"], lhs_f32 * rhs_f32 + addend)

    expected_neg_bits = unary_bits ^ np.uint32(0x80000000)
    expected_neg_bits[:2] = np.array([0x80000000, 0x00000000], dtype=np.uint32)
    np.testing.assert_array_equal(result.outputs["negated"].view(np.uint32), expected_neg_bits)
    expected_quotient = unary_input / divisor
    expected_quotient_bits = expected_quotient.view(np.uint32)
    expected_quotient_bits[:2] = np.array([0x00000000, 0x80000000], dtype=np.uint32)
    np.testing.assert_array_equal(
        result.outputs["quotient"].view(np.uint32), expected_quotient_bits
    )
    np.testing.assert_array_equal(
        result.outputs["quotient_rn"].view(np.uint32),
        expected_quotient.view(np.uint32),
    )
    np.testing.assert_array_equal(
        result.outputs["reciprocal"].view(np.uint32),
        (np.float32(1) / divisor).view(np.uint32),
    )
    np.testing.assert_array_equal(
        result.outputs["inverse_sqrt"].view(np.uint32),
        (np.float32(1) / np.sqrt(divisor, dtype=np.float32)).view(np.uint32),
    )
