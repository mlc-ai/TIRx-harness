from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
import tvm
from tvm_ffi import structural_walk
from tvm.script import tirx as T

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.ptx_dialect import PtxCallDecodeError, decode_ptx_call


_FLOAT_COMPARISONS = (
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "equ",
    "neu",
    "ltu",
    "leu",
    "gtu",
    "geu",
    "num",
    "nan",
)
_BOOL_OPS = ("and", "or", "xor")
_CLASSES = ("finite", "infinite", "number", "notanumber", "normal", "subnormal")


@T.prim_func
def ptx_setp_bit_carriers(
    a16: T.Buffer((32,), "int16"),
    b16: T.Buffer((32,), "uint16"),
    a32: T.Buffer((32,), "float32"),
    b32: T.Buffer((32,), "int32"),
    a64: T.Buffer((32,), "float64"),
    b64: T.Buffer((32,), "uint64"),
    output: T.Buffer((9, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.setp.eq.b16(output[0, lane], a16[lane], b16[lane])
    T.ptx.setp.ne.b16(output[1, lane], a16[lane], b16[lane])
    T.ptx.setp.eq.b32(output[2, lane], a32[lane], b32[lane])
    T.ptx.setp.ne.b32(output[3, lane], a32[lane], b32[lane])
    T.ptx.setp.eq.b64(output[4, lane], a64[lane], b64[lane])
    T.ptx.setp.ne.b64(output[5, lane], a64[lane], b64[lane])
    T.ptx.setp.eq.ftz.f32(output[6, lane], a32[lane], T.float32(0))
    T.ptx.setp.eq.f32(output[7, lane], a32[lane], T.float32(0))
    T.ptx.setp.eq.ftz.f16(output[8, lane], b16[lane], T.uint16(0))


def test_setp_bit_size_eq_ne_preserves_mixed_carrier_payloads(tmp_path):
    inputs, expected = {}, []
    for width, carrier in ((16, np.int16), (32, np.float32), (64, np.float64)):
        sign = 1 << (width - 1)
        nan = {16: 0x7E01, 32: 0x7FC00001, 64: 0x7FF8000000000001}[width]
        # Signed zero and NaN payloads must compare as bits, not floating values.
        left = np.resize(np.array([0, sign, nan, nan, sign + 1, 1], dtype=f"uint{width}"), 32)
        right = np.resize(np.array([sign, sign, nan, nan + 1, sign, 1], dtype=left.dtype), 32)
        inputs[f"a{width}"] = left.view(carrier)
        inputs[f"b{width}"] = right.view(np.int32) if width == 32 else right
        expected.extend(((left == right).astype(np.uint32), (left != right).astype(np.uint32)))
    expected.extend(
        (
            (np.abs(inputs["a32"]) < np.finfo(np.float32).tiny).astype(np.uint32),
            (inputs["a32"] == 0).astype(np.uint32),
            (np.abs(inputs["b16"].view(np.float16)) < np.finfo(np.float16).tiny).astype(np.uint32),
        )
    )
    inputs["output"] = np.full((9, 32), 0xFFFFFFFF, dtype=np.uint32)
    result = numsim.Engine().run(
        numsim.transpile(ptx_setp_bit_carriers, cache_dir=tmp_path), inputs, outputs=("output",)
    )
    np.testing.assert_array_equal(result.outputs["output"], np.asarray(expected))
    assert result.verdict == "clean"


def test_setp_bit_size_relational_comparison_remains_rejected():
    calls = []
    structural_walk(
        ptx_setp_bit_carriers.body,
        lambda node: (
            calls.append(node)
            if isinstance(node, tvm.ir.Call) and str(node.op.name) == "tirx.ptx.setp"
            else None
        ),
    )
    assert len(calls) == 8
    for call in calls[:6]:
        args = [
            tvm.ir.StringImm("lt")
            if isinstance(arg, tvm.ir.StringImm) and arg.value in {"eq", "ne"}
            else arg
            for arg in call.args
        ]
        invalid = tvm.ir.Call(call.op, args, attrs=call.attrs, span=call.span, ret_ty=call.ty)
        with pytest.raises(PtxCallDecodeError, match="bit-size.*compares only with eq/ne"):
            decode_ptx_call(invalid)
    for ptx_type in ("u32", "s32", "f64"):
        call = calls[6]
        args = [
            tvm.ir.StringImm(ptx_type)
            if isinstance(arg, tvm.ir.StringImm) and arg.value == "f32"
            else arg
            for arg in call.args
        ]
        invalid = tvm.ir.Call(call.op, args, attrs=call.attrs, span=call.span, ret_ty=call.ty)
        with pytest.raises(PtxCallDecodeError, match="ftz"):
            decode_ptx_call(invalid)


def _comparison_kernel():
    parameters = (
        'lhs_f32: T.Buffer((32,), "float32")',
        'rhs_f32: T.Buffer((32,), "float32")',
        'lhs_f64: T.Buffer((32,), "float64")',
        'rhs_f64: T.Buffer((32,), "float64")',
        'lhs_f16: T.Buffer((32,), "uint16")',
        'rhs_f16: T.Buffer((32,), "uint16")',
        'lhs_bf16: T.Buffer((32,), "uint16")',
        'rhs_bf16: T.Buffer((32,), "uint16")',
        'lhs_f16x2: T.Buffer((32,), "uint32")',
        'rhs_f16x2: T.Buffer((32,), "uint32")',
        'lhs_bf16x2: T.Buffer((32,), "uint32")',
        'rhs_bf16x2: T.Buffer((32,), "uint32")',
        'lhs_i32: T.Buffer((32,), "int32")',
        'rhs_i32: T.Buffer((32,), "int32")',
        'lhs_i64: T.Buffer((32,), "int64")',
        'rhs_i64: T.Buffer((32,), "int64")',
        'lhs_u64: T.Buffer((32,), "uint64")',
        'rhs_u64: T.Buffer((32,), "uint64")',
        'predicate: T.Buffer((32,), "uint32")',
        'regular_set: T.Buffer((17, 32), "uint32")',
        'integer_set: T.Buffer((32,), "int32")',
        'float_set: T.Buffer((32,), "float32")',
        'double_set: T.Buffer((32,), "uint32")',
        'half_set: T.Buffer((17, 32), "uint32")',
        'half_scalar_set: T.Buffer((4, 32), "uint16")',
        'predicate_set: T.Buffer((12, 32), "uint32")',
        'classes: T.Buffer((12, 32), "uint32")',
    )
    lines = [
        "@T.prim_func",
        "def ptx_compare_semantics(",
        *(f"    {parameter}," for parameter in parameters),
        "):",
        "    T.device_entry()",
        "    _warp = T.warp_id([1])",
        "    lane = T.lane_id([32])",
    ]
    for row, comparison in enumerate(_FLOAT_COMPARISONS):
        lines.append(
            f'    T.ptx["set.{comparison}.u32.f32"]('
            f"regular_set[{row}, lane], lhs_f32[lane], rhs_f32[lane])"
        )
        lines.append(
            f'    T.ptx["set.{comparison}.u32.f16x2"]('
            f"half_set[{row}, lane], lhs_f16x2[lane], rhs_f16x2[lane])"
        )
    for offset, bool_op in enumerate(_BOOL_OPS):
        lines.append(
            f'    T.ptx["set.eq.{bool_op}.u32.f32"]('
            f"regular_set[{14 + offset}, lane], lhs_f32[lane], rhs_f32[lane], "
            "T.ptx.pred(predicate[lane]))"
        )
        lines.append(
            f'    T.ptx["set.ne.{bool_op}.u32.f16x2"]('
            f"half_set[{14 + offset}, lane], lhs_f16x2[lane], rhs_f16x2[lane], "
            "T.ptx.pred(predicate[lane]))"
        )
    lines.extend(
        (
            '    T.ptx["set.lt.s32.s64"](integer_set[lane], lhs_i64[lane], rhs_i64[lane])',
            '    T.ptx["set.hi.f32.u64"](float_set[lane], lhs_u64[lane], rhs_u64[lane])',
            '    T.ptx["set.nan.u32.f64"](double_set[lane], lhs_f64[lane], rhs_f64[lane])',
            '    T.ptx["set.lt.f16.s32"](half_scalar_set[0, lane], lhs_i32[lane], rhs_i32[lane])',
            '    T.ptx["set.gt.u16.bf16"](half_scalar_set[1, lane], lhs_bf16[lane], rhs_bf16[lane])',
            '    T.ptx["set.eq.bf16.b16"](half_scalar_set[2, lane], lhs_f16[lane], rhs_f16[lane])',
            '    T.ptx["set.eq.ftz.f16.f16"](half_scalar_set[3, lane], lhs_f16[lane], rhs_f16[lane])',
            '    T.ptx["setp.ne.and.f32"](predicate_set[0, lane], lhs_f32[lane], rhs_f32[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.eq.or.f32"](predicate_set[1, lane], predicate_set[2, lane], lhs_f32[lane], rhs_f32[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.hi.u64"](predicate_set[3, lane], predicate_set[4, lane], lhs_u64[lane], rhs_u64[lane])',
            '    T.ptx["setp.ltu.and.ftz.f16"](predicate_set[5, lane], lhs_f16[lane], rhs_f16[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.ltu.ftz.f16x2"](predicate_set[6, lane], predicate_set[7, lane], lhs_f16x2[lane], rhs_f16x2[lane])',
            '    T.ptx["setp.geu.xor.bf16x2"](predicate_set[8, lane], predicate_set[9, lane], lhs_bf16x2[lane], rhs_bf16x2[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.nan.or.bf16"](predicate_set[10, lane], lhs_bf16[lane], rhs_bf16[lane], T.ptx.pred(predicate[lane]))',
            '    T.ptx["setp.ne.xor.s32"](predicate_set[11, lane], lhs_i32[lane], rhs_i32[lane], T.ptx.pred(predicate[lane]))',
        )
    )
    for offset, classification in enumerate(_CLASSES):
        lines.append(
            f'    T.ptx["testp.{classification}.f32"](classes[{offset}, lane], lhs_f32[lane])'
        )
        lines.append(
            f'    T.ptx["testp.{classification}.f64"](classes[{6 + offset}, lane], lhs_f64[lane])'
        )
    return tvm.script.from_source("\n".join(lines), {"T": T})


def _slct_kernel():
    return tvm.script.from_source(
        """
@T.prim_func
def ptx_slct_semantics(
    a16: T.Buffer((32,), "bfloat16"),
    b16: T.Buffer((32,), "int16"),
    c_i32: T.Buffer((32,), "int32"),
    a32: T.Buffer((32,), "float32"),
    b32: T.Buffer((32,), "int32"),
    c_f32: T.Buffer((32,), "float32"),
    a64: T.Buffer((32,), "uint64"),
    b64: T.Buffer((32,), "int64"),
    out16: T.Buffer((32,), "float16"),
    out32: T.Buffer((32,), "uint32"),
    out32_ftz: T.Buffer((32,), "int32"),
    out64: T.Buffer((32,), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["slct.b16.s32"](out16[lane], a16[lane], b16[lane], c_i32[lane])
    T.ptx["slct.f32.f32"](out32[lane], a32[lane], b32[lane], c_f32[lane])
    T.ptx["slct.ftz.f32.f32"](out32_ftz[lane], out32[lane], a32[lane], c_f32[lane])
    T.ptx["slct.b64.s32"](out64[lane], a64[lane], b64[lane], c_i32[lane])
""",
        {"T": T},
    )


ptx_compare_semantics = _comparison_kernel()
ptx_slct_semantics = _slct_kernel()


def _compare(lhs: np.ndarray, rhs: np.ndarray, relation: str) -> np.ndarray:
    unordered = np.isnan(lhs) | np.isnan(rhs)
    ordered = {
        "eq": lhs == rhs,
        "ne": lhs != rhs,
        "lt": lhs < rhs,
        "le": lhs <= rhs,
        "gt": lhs > rhs,
        "ge": lhs >= rhs,
    }
    if relation in ordered:
        return ~unordered & ordered[relation]
    if relation in {"equ", "neu", "ltu", "leu", "gtu", "geu"}:
        return unordered | ordered[relation[:-1]]
    if relation == "num":
        return ~unordered
    if relation == "nan":
        return unordered
    raise AssertionError(relation)


def _bool_op(value: np.ndarray, predicate: np.ndarray, op: str) -> np.ndarray:
    return {"and": value & predicate, "or": value | predicate, "xor": value ^ predicate}[op]


def _f16_lanes(bits: np.ndarray) -> np.ndarray:
    return bits.view(np.uint16).reshape(32, 2).view(np.float16).astype(np.float32)


def _f16_lanes_ftz(bits: np.ndarray) -> np.ndarray:
    lanes = bits.view(np.uint16).reshape(32, 2).copy()
    subnormal = (lanes & np.uint16(0x7C00) == 0) & (lanes & np.uint16(0x03FF) != 0)
    lanes[subnormal] &= np.uint16(0x8000)
    return lanes.view(np.float16).astype(np.float32)


def _bf16(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _classify(values: np.ndarray, classification: str) -> np.ndarray:
    if classification == "finite":
        return np.isfinite(values)
    if classification == "infinite":
        return np.isinf(values)
    if classification == "number":
        return ~np.isnan(values)
    if classification == "notanumber":
        return np.isnan(values)
    if classification == "normal":
        tiny = np.finfo(values.dtype).tiny
        return (values == 0) | (np.isfinite(values) & (np.abs(values) >= tiny))
    if classification == "subnormal":
        tiny = np.finfo(values.dtype).tiny
        return np.isfinite(values) & (values != 0) & (np.abs(values) < tiny)
    raise AssertionError(classification)


def _inputs() -> dict[str, np.ndarray]:
    f32_bits = np.resize(
        np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x0080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC0_0001,
                0xFFC1_2345,
            ],
            dtype=np.uint32,
        ),
        32,
    )
    rhs_f32_bits = np.roll(f32_bits, 3) ^ np.uint32(0x0000_0001)
    f64_bits = np.resize(
        np.asarray(
            [
                0x0000_0000_0000_0000,
                0x8000_0000_0000_0000,
                0x0000_0000_0000_0001,
                0x0010_0000_0000_0000,
                0x3FF0_0000_0000_0000,
                0xBFF0_0000_0000_0000,
                0x7FF0_0000_0000_0000,
                0xFFF0_0000_0000_0000,
                0x7FF8_0000_0000_0001,
            ],
            dtype=np.uint64,
        ),
        32,
    )
    half_lanes = np.resize(
        np.asarray(
            [0x0000, 0x8000, 0x0001, 0x8001, 0x3C00, 0xBC00, 0x7C00, 0x7E01], dtype=np.uint16
        ),
        64,
    ).reshape(32, 2)
    rhs_half_lanes = np.roll(half_lanes, 5, axis=None).reshape(32, 2)
    bf16 = np.resize(
        np.asarray(
            [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x7F80, 0x7FC1], dtype=np.uint16
        ),
        32,
    )
    lanes = np.arange(32, dtype=np.int64)
    return {
        "lhs_f32": f32_bits.view(np.float32),
        "rhs_f32": rhs_f32_bits.view(np.float32),
        "lhs_f64": f64_bits.view(np.float64),
        "rhs_f64": np.roll(f64_bits, 2).view(np.float64),
        "lhs_f16": half_lanes[:, 0].copy(),
        "rhs_f16": rhs_half_lanes[:, 0].copy(),
        "lhs_bf16": bf16,
        "rhs_bf16": np.roll(bf16, 3),
        "lhs_f16x2": half_lanes.view(np.uint32).reshape(32),
        "rhs_f16x2": rhs_half_lanes.view(np.uint32).reshape(32),
        "lhs_bf16x2": np.column_stack((bf16, np.roll(bf16, 1))).view(np.uint32).reshape(32),
        "rhs_bf16x2": np.column_stack((np.roll(bf16, 3), np.roll(bf16, 4)))
        .view(np.uint32)
        .reshape(32),
        "lhs_i32": (lanes - 15).astype(np.int32),
        "rhs_i32": (8 - lanes).astype(np.int32),
        "lhs_i64": (lanes * 0x101 - 0x800).astype(np.int64),
        "rhs_i64": (0x400 - lanes * 0x31).astype(np.int64),
        "lhs_u64": (lanes.astype(np.uint64) * np.uint64(0x1020_4081))
        ^ np.uint64(0x8000_0000_0000_0000),
        "rhs_u64": np.roll(lanes.astype(np.uint64), 7) * np.uint64(0x0101_0101),
        "predicate": (lanes % 3 == 0).astype(np.uint32),
    }


def test_compare_and_classify_match_independent_bit_oracle(tmp_path):
    inputs = _inputs()
    outputs = {
        "regular_set": np.zeros((17, 32), dtype=np.uint32),
        "integer_set": np.zeros(32, dtype=np.int32),
        "float_set": np.zeros(32, dtype=np.float32),
        "double_set": np.zeros(32, dtype=np.uint32),
        "half_set": np.zeros((17, 32), dtype=np.uint32),
        "half_scalar_set": np.zeros((4, 32), dtype=np.uint16),
        "predicate_set": np.zeros((12, 32), dtype=np.uint32),
        "classes": np.zeros((12, 32), dtype=np.uint32),
    }
    result = numsim.Engine().run(
        numsim.transpile(ptx_compare_semantics, cache_dir=tmp_path),
        {**inputs, **outputs},
        outputs=tuple(outputs),
    )
    predicate = inputs["predicate"].astype(bool)
    for row, comparison in enumerate(_FLOAT_COMPARISONS):
        expected = _compare(inputs["lhs_f32"], inputs["rhs_f32"], comparison)
        np.testing.assert_array_equal(
            result.outputs["regular_set"][row],
            expected.astype(np.uint32) * np.uint32(0xFFFF_FFFF),
        )
        half_expected = _compare(
            _f16_lanes(inputs["lhs_f16x2"]), _f16_lanes(inputs["rhs_f16x2"]), comparison
        )
        packed = half_expected.astype(np.uint32) * np.uint32(0xFFFF)
        packed = packed[:, 0] | (packed[:, 1] << np.uint32(16))
        np.testing.assert_array_equal(result.outputs["half_set"][row], packed)
    equality = _compare(inputs["lhs_f32"], inputs["rhs_f32"], "eq")
    half_ne = _compare(_f16_lanes(inputs["lhs_f16x2"]), _f16_lanes(inputs["rhs_f16x2"]), "ne")
    for offset, bool_op in enumerate(_BOOL_OPS):
        expected = _bool_op(equality, predicate, bool_op).astype(np.uint32) * np.uint32(0xFFFF_FFFF)
        np.testing.assert_array_equal(result.outputs["regular_set"][14 + offset], expected)
        expected_half = _bool_op(half_ne, predicate[:, None], bool_op).astype(
            np.uint32
        ) * np.uint32(0xFFFF)
        expected_half = expected_half[:, 0] | (expected_half[:, 1] << np.uint32(16))
        np.testing.assert_array_equal(result.outputs["half_set"][14 + offset], expected_half)

    np.testing.assert_array_equal(
        result.outputs["integer_set"],
        np.where(inputs["lhs_i64"] < inputs["rhs_i64"], -1, 0).astype(np.int32),
    )
    np.testing.assert_array_equal(
        result.outputs["float_set"], (inputs["lhs_u64"] > inputs["rhs_u64"]).astype(np.float32)
    )
    np.testing.assert_array_equal(
        result.outputs["double_set"],
        np.isnan(inputs["lhs_f64"]).astype(np.uint32) * np.uint32(0xFFFF_FFFF)
        | np.isnan(inputs["rhs_f64"]).astype(np.uint32) * np.uint32(0xFFFF_FFFF),
    )
    half_scalar = result.outputs["half_scalar_set"]
    np.testing.assert_array_equal(
        half_scalar[0], np.where(inputs["lhs_i32"] < inputs["rhs_i32"], 0x3C00, 0).astype(np.uint16)
    )
    np.testing.assert_array_equal(
        half_scalar[1],
        (_bf16(inputs["lhs_bf16"]) > _bf16(inputs["rhs_bf16"])).astype(np.uint16)
        * np.uint16(0xFFFF),
    )
    np.testing.assert_array_equal(
        half_scalar[2],
        np.where(inputs["lhs_f16"] == inputs["rhs_f16"], 0x3F80, 0).astype(np.uint16),
    )
    lhs_ftz = inputs["lhs_f16"].copy()
    rhs_ftz = inputs["rhs_f16"].copy()
    for values in (lhs_ftz, rhs_ftz):
        subnormal = (values & np.uint16(0x7C00) == 0) & (values & np.uint16(0x03FF) != 0)
        values[subnormal] &= np.uint16(0x8000)
    np.testing.assert_array_equal(
        half_scalar[3],
        np.where(lhs_ftz.view(np.float16) == rhs_ftz.view(np.float16), 0x3C00, 0).astype(np.uint16),
    )

    p = result.outputs["predicate_set"]
    ne = _compare(inputs["lhs_f32"], inputs["rhs_f32"], "ne")
    np.testing.assert_array_equal(p[0], ne & predicate)
    eq = _compare(inputs["lhs_f32"], inputs["rhs_f32"], "eq")
    np.testing.assert_array_equal(p[1], eq | predicate)
    np.testing.assert_array_equal(p[2], (~eq) | predicate)
    hi = inputs["lhs_u64"] > inputs["rhs_u64"]
    np.testing.assert_array_equal(p[3], hi)
    np.testing.assert_array_equal(p[4], ~hi)
    f16_ltu = _compare(
        _f16_lanes_ftz(inputs["lhs_f16x2"]),
        _f16_lanes_ftz(inputs["rhs_f16x2"]),
        "ltu",
    )
    scalar_lhs_ftz = lhs_ftz.view(np.float16).astype(np.float32)
    scalar_rhs_ftz = rhs_ftz.view(np.float16).astype(np.float32)
    np.testing.assert_array_equal(p[5], _compare(scalar_lhs_ftz, scalar_rhs_ftz, "ltu") & predicate)
    np.testing.assert_array_equal(p[6], f16_ltu[:, 0])
    np.testing.assert_array_equal(p[7], f16_ltu[:, 1])
    bf16_lhs = np.column_stack(
        (
            _bf16(inputs["lhs_bf16x2"].view(np.uint16).reshape(32, 2)[:, 0]),
            _bf16(inputs["lhs_bf16x2"].view(np.uint16).reshape(32, 2)[:, 1]),
        )
    )
    bf16_rhs = np.column_stack(
        (
            _bf16(inputs["rhs_bf16x2"].view(np.uint16).reshape(32, 2)[:, 0]),
            _bf16(inputs["rhs_bf16x2"].view(np.uint16).reshape(32, 2)[:, 1]),
        )
    )
    bf16_geu = _compare(bf16_lhs, bf16_rhs, "geu")
    np.testing.assert_array_equal(p[8], bf16_geu[:, 0] ^ predicate)
    np.testing.assert_array_equal(p[9], bf16_geu[:, 1] ^ predicate)
    bf16_nan = _compare(_bf16(inputs["lhs_bf16"]), _bf16(inputs["rhs_bf16"]), "nan")
    np.testing.assert_array_equal(p[10], bf16_nan | predicate)
    np.testing.assert_array_equal(p[11], (inputs["lhs_i32"] != inputs["rhs_i32"]) ^ predicate)

    for offset, classification in enumerate(_CLASSES):
        np.testing.assert_array_equal(
            result.outputs["classes"][offset], _classify(inputs["lhs_f32"], classification)
        )
        np.testing.assert_array_equal(
            result.outputs["classes"][6 + offset], _classify(inputs["lhs_f64"], classification)
        )


def test_slct_preserves_payload_bits_and_defines_nan_zero_and_ftz(tmp_path):
    lanes = np.arange(32, dtype=np.int64)
    a16_bits = (np.uint16(0x7C01) + lanes.astype(np.uint16) * np.uint16(0x0111)).astype(np.uint16)
    b16_bits = (np.uint16(0x8001) ^ lanes.astype(np.uint16) * np.uint16(0x101)).astype(np.uint16)
    c_i32 = (lanes - 15).astype(np.int32)
    a32_bits = (np.uint32(0x7FC0_0001) + lanes.astype(np.uint32) * np.uint32(0x10001)).astype(
        np.uint32
    )
    b32_bits = (np.uint32(0x8000_0001) ^ lanes.astype(np.uint32) * np.uint32(0x10101)).astype(
        np.uint32
    )
    c_f32_bits = np.resize(
        np.asarray(
            [0x7FC0_0001, 0x8000_0000, 0x0000_0001, 0x8000_0001, 0x3F80_0000, 0xBF80_0000],
            dtype=np.uint32,
        ),
        32,
    )
    a64_bits = np.uint64(0x7FF0_0000_0000_0001) + lanes.astype(np.uint64) * np.uint64(0x100000001)
    b64_bits = np.uint64(0x8000_0000_0000_0001) ^ lanes.astype(np.uint64) * np.uint64(0x101010101)
    result = numsim.Engine().run(
        numsim.transpile(ptx_slct_semantics, cache_dir=tmp_path),
        {
            "a16": a16_bits.view(ml_dtypes.bfloat16),
            "b16": b16_bits.view(np.int16),
            "c_i32": c_i32,
            "a32": a32_bits.view(np.float32),
            "b32": b32_bits.view(np.int32),
            "c_f32": c_f32_bits.view(np.float32),
            "a64": a64_bits,
            "b64": b64_bits.view(np.int64),
            "out16": np.zeros(32, dtype=np.float16),
            "out32": np.zeros(32, dtype=np.uint32),
            "out32_ftz": np.zeros(32, dtype=np.int32),
            "out64": np.zeros(32, dtype=np.float64),
        },
        outputs=("out16", "out32", "out32_ftz", "out64"),
    )
    np.testing.assert_array_equal(
        result.outputs["out16"].view(np.uint16), np.where(c_i32 >= 0, a16_bits, b16_bits)
    )
    c_f32 = c_f32_bits.view(np.float32)
    select_a = ~np.isnan(c_f32) & (c_f32 >= 0)
    np.testing.assert_array_equal(result.outputs["out32"], np.where(select_a, a32_bits, b32_bits))
    # NaN remains ordered-false; both signs of subnormal flush to zero and select a.
    ftz_select_a = select_a | (c_f32_bits & np.uint32(0x7F80_0000) == 0)
    np.testing.assert_array_equal(
        result.outputs["out32_ftz"].view(np.uint32),
        np.where(ftz_select_a, np.where(select_a, a32_bits, b32_bits), a32_bits),
    )
    np.testing.assert_array_equal(
        result.outputs["out64"].view(np.uint64), np.where(c_i32 >= 0, a64_bits, b64_bits)
    )


__all__ = []
