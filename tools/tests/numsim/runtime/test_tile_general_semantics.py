from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import tmem_datapath_layout
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout


@T.prim_func
def right_aligned_elementwise_broadcast(
    source: T.Buffer((32, 2, 4), "float32"),
    column: T.Buffer((32, 4), "float32"),
    row: T.Buffer((32, 2), "float32"),
    half_column: T.Buffer((32, 4), "float16"),
    output: T.Buffer((32, 4, 2, 4), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_local = T.alloc_buffer((2, 4), "float32", scope="local")
    column_local = T.alloc_buffer((4,), "float32", scope="local")
    row_local = T.alloc_buffer((2, 1), "float32", scope="local")
    half_column_local = T.alloc_buffer((4,), "float16", scope="local")
    add_result = T.alloc_buffer((2, 4), "float32", scope="local")
    sub_result = T.alloc_buffer((2, 4), "float32", scope="local")
    mul_result = T.alloc_buffer((2, 4), "float32", scope="local")
    cast_result = T.alloc_buffer((2, 4), "float32", scope="local")
    for row_index in T.serial(2):
        row_local[row_index, 0] = row[lane, row_index]
        for column_index in T.serial(4):
            source_local[row_index, column_index] = source[lane, row_index, column_index]
    for column_index in T.serial(4):
        column_local[column_index] = column[lane, column_index]
        half_column_local[column_index] = half_column[lane, column_index]

    Tx.add(add_result[:, :], source_local[:, :], column_local[:])
    Tx.sub(sub_result[:, :], source_local[:, :], row_local[:, :])
    Tx.mul(mul_result[:, :], source_local[:, :], row_local[:, :])
    Tx.cast(cast_result[:, :], half_column_local[:])

    for row_index in T.serial(2):
        for column_index in T.serial(4):
            output[lane, 0, row_index, column_index] = add_result[row_index, column_index]
            output[lane, 1, row_index, column_index] = sub_result[row_index, column_index]
            output[lane, 2, row_index, column_index] = mul_result[row_index, column_index]
            output[lane, 3, row_index, column_index] = cast_result[row_index, column_index]


@T.prim_func
def fdiv_rounding_config_uses_cuda_scalar_fallback(
    source: T.Buffer((32, 2), "float32"), output: T.Buffer((32, 4), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs: T.f32[1]
    rhs: T.f32[1]
    result: T.f32[1]
    lhs[0] = source[lane, 0]
    rhs[0] = source[lane, 1]
    Tx.fdiv(result, lhs, rhs, rounding_mode="rn")
    output[lane, 0] = result[0]
    Tx.fdiv(result, lhs, rhs, rounding_mode="rm")
    output[lane, 1] = result[0]
    Tx.fdiv(result, lhs, rhs, rounding_mode="rp")
    output[lane, 2] = result[0]
    Tx.fdiv(result, lhs, rhs, rounding_mode="rz")
    output[lane, 3] = result[0]


@T.prim_func
def elementwise_maximum(
    lhs: T.Buffer((32, 4), "float32"),
    rhs: T.Buffer((32, 4), "float32"),
    output: T.Buffer((32, 4), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs_local: T.f32[4]
    rhs_local: T.f32[4]
    result: T.f32[4]
    for index in T.serial(4):
        lhs_local[index] = lhs[lane, index]
        rhs_local[index] = rhs[lane, index]
    Tx.maximum(result, lhs_local, rhs_local)
    for index in T.serial(4):
        output[lane, index] = result[index]


@T.prim_func
def float64_elementwise_forms(
    lhs: T.Buffer((32, 4), "float64"),
    rhs: T.Buffer((32, 4), "float64"),
    output: T.Buffer((32, 10, 4), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs_local: T.f64[4]
    rhs_local: T.f64[4]
    result: T.f64[4]
    for index in T.serial(4):
        lhs_local[index] = lhs[lane, index]
        rhs_local[index] = rhs[lane, index]

    Tx.add(result, lhs_local, rhs_local)
    for index in T.serial(4):
        output[lane, 0, index] = result[index]
    Tx.sub(result, lhs_local, rhs_local)
    for index in T.serial(4):
        output[lane, 1, index] = result[index]
    Tx.mul(result, lhs_local, rhs_local)
    for index in T.serial(4):
        output[lane, 2, index] = result[index]
    Tx.fdiv(result, lhs_local, rhs_local)
    for index in T.serial(4):
        output[lane, 3, index] = result[index]
    Tx.fma(result, lhs_local, rhs_local, T.float64(0.25))
    for index in T.serial(4):
        output[lane, 4, index] = result[index]
    Tx.sqrt(result, lhs_local)
    for index in T.serial(4):
        output[lane, 5, index] = result[index]
    Tx.exp(result, lhs_local)
    for index in T.serial(4):
        output[lane, 6, index] = result[index]
    Tx.exp2(result, lhs_local)
    for index in T.serial(4):
        output[lane, 7, index] = result[index]
    Tx.reciprocal(result, lhs_local)
    for index in T.serial(4):
        output[lane, 8, index] = result[index]
    Tx.silu(result, lhs_local)
    for index in T.serial(4):
        output[lane, 9, index] = result[index]


@T.prim_func
def float64_directed_rounding_is_unsupported(
    lhs: T.Buffer((32,), "float64"),
    rhs: T.Buffer((32,), "float64"),
    output: T.Buffer((32,), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs_local: T.f64[1]
    rhs_local: T.f64[1]
    result: T.f64[1]
    lhs_local[0] = lhs[lane]
    rhs_local[0] = rhs[lane]
    Tx.add(result, lhs_local, rhs_local, rounding_mode="rp")
    output[lane] = result[0]


@T.prim_func
def cast_uses_general_scalar_conversion(
    integers: T.Buffer((32, 4), "int32"),
    floats: T.Buffer((32, 4), "float32"),
    output_float: T.Buffer((32, 4), "float64"),
    output_byte: T.Buffer((32, 4), "uint8"),
    output_bool: T.Buffer((32, 4), "bool"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    integer_local = T.alloc_buffer((4,), "int32", scope="local")
    float_local = T.alloc_buffer((4,), "float32", scope="local")
    double_local = T.alloc_buffer((4,), "float64", scope="local")
    byte_local = T.alloc_buffer((4,), "uint8", scope="local")
    bool_local = T.alloc_buffer((4,), "bool", scope="local")
    for index in T.serial(4):
        integer_local[index] = integers[lane, index]
        float_local[index] = floats[lane, index]
    Tx.cast(double_local[:], integer_local[:])
    Tx.cast(byte_local[:], float_local[:])
    Tx.cast(bool_local[:], integer_local[:])
    for index in T.serial(4):
        output_float[lane, index] = double_local[index]
        output_byte[lane, index] = byte_local[index]
        output_bool[lane, index] = bool_local[index]


def _integer_elementwise_kernel(dtype: str):
    @T.prim_func
    def kernel(
        lhs: T.Buffer((32, 4), dtype),
        rhs: T.Buffer((32, 4), dtype),
        output: T.Buffer((32, 3, 4), dtype),
    ):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        lhs_local = T.alloc_buffer((4,), dtype, scope="local")
        rhs_local = T.alloc_buffer((4,), dtype, scope="local")
        result = T.alloc_buffer((4,), dtype, scope="local")
        for index in T.serial(4):
            lhs_local[index] = lhs[lane, index]
            rhs_local[index] = rhs[lane, index]
        Tx.add(result[:], lhs_local[:], rhs_local[:])
        for index in T.serial(4):
            output[lane, 0, index] = result[index]
        Tx.sub(result[:], lhs_local[:], rhs_local[:])
        for index in T.serial(4):
            output[lane, 1, index] = result[index]
        Tx.mul(result[:], lhs_local[:], rhs_local[:])
        for index in T.serial(4):
            output[lane, 2, index] = result[index]

    return kernel


# Stable source-backed representative; runtime tests instantiate every supported
# integer width through the factory.
integer_elementwise_int32 = _integer_elementwise_kernel("int32")


_PACKED_FP4_MMA_128X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32))
_PACKED_FP4_MMA_8X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 32))


@T.prim_func
def block_scaled_mxfp4_gemm(
    left_packed: T.Buffer((128, 32), "uint8"),
    right_packed: T.Buffer((8, 32), "uint8"),
    scale_a: T.Buffer((128, 2), "float8_e8m0fnu"),
    scale_b: T.Buffer((8, 2), "float8_e8m0fnu"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32
    )
    right_shared_packed = T.alloc_buffer(
        (8, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_8X32
    )
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 2),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=2),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 2),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=2),
        allocated_addr=24,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[:, :])
        Tx.copy(right_shared_packed[:, :], right_packed[:, :])
        for row_index in T.serial(128):
            for scale_index in T.serial(2):
                scale_a_tmem[row_index, scale_index] = scale_a[row_index, scale_index]
        for row_index in T.serial(8):
            for scale_index in T.serial(2):
                scale_b_tmem[row_index, scale_index] = scale_b[row_index, scale_index]
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row_index in T.serial(128):
            for column_index in T.serial(8):
                output[row_index, column_index] = accumulator[row_index, column_index]


def _decode_e2m1(bits: np.ndarray) -> np.ndarray:
    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    bits = np.asarray(bits, dtype=np.uint8) & np.uint8(0xF)
    magnitude = values[(bits & np.uint8(0x7)).astype(np.intp)]
    return np.where((bits & np.uint8(0x8)) != 0, -magnitude, magnitude).astype(np.float32)


def _unpack_e2m1(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    result = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    result[..., 0::2] = packed & np.uint8(0xF)
    result[..., 1::2] = packed >> np.uint8(4)
    return result


def test_right_aligned_buffer_broadcast_maps_destination_coordinates(tmp_path):
    source = np.arange(32 * 2 * 4, dtype=np.float32).reshape(32, 2, 4) / np.float32(7)
    column = np.arange(32 * 4, dtype=np.float32).reshape(32, 4) / np.float32(5)
    row = (np.arange(32 * 2, dtype=np.float32).reshape(32, 2) - 11) / np.float32(3)
    half_column = (column - np.float32(4.25)).astype(np.float16)
    output = np.zeros((32, 4, 2, 4), dtype=np.float32)

    module = numsim.transpile(right_aligned_elementwise_broadcast, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "column": column,
            "row": row,
            "half_column": half_column,
            "output": output,
        },
    )

    expected = np.stack(
        (
            source + column[:, None, :],
            source - row[:, :, None],
            source * row[:, :, None],
            np.broadcast_to(half_column[:, None, :], source.shape),
        ),
        axis=1,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    [
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
        ("int64", np.int64),
        ("uint8", np.uint8),
        ("uint16", np.uint16),
        ("uint32", np.uint32),
        ("uint64", np.uint64),
    ],
)
def test_integer_add_sub_mul_use_wrapping_arithmetic(dtype, numpy_dtype, tmp_path):
    info = np.iinfo(numpy_dtype)
    lhs = np.resize(np.array([info.max, info.min, 7, 13], dtype=numpy_dtype), (32, 4)).copy()
    rhs = np.resize(np.array([2, 3, info.max, 11], dtype=numpy_dtype), (32, 4)).copy()
    output = np.zeros((32, 3, 4), dtype=numpy_dtype)

    module = numsim.transpile(_integer_elementwise_kernel(dtype), cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"lhs": lhs, "rhs": rhs, "output": output})

    with np.errstate(over="ignore"):
        expected = np.stack(
            (
                np.add(lhs, rhs, dtype=numpy_dtype),
                np.subtract(lhs, rhs, dtype=numpy_dtype),
                np.multiply(lhs, rhs, dtype=numpy_dtype),
            ),
            axis=1,
        )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_fdiv_rounding_config_matches_current_cuda_scalar_fallback(tmp_path):
    source = np.stack(
        (
            np.linspace(-7.0, 9.0, 32, dtype=np.float32),
            np.linspace(0.75, 3.25, 32, dtype=np.float32),
        ),
        axis=1,
    )
    output = np.zeros((32, 4), dtype=np.float32)

    module = numsim.transpile(fdiv_rounding_config_uses_cuda_scalar_fallback, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.broadcast_to((source[:, 0] / source[:, 1])[:, None], output.shape)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_elementwise_maximum_matches_pairwise_values(tmp_path):
    lhs = np.linspace(-5.0, 7.0, 32 * 4, dtype=np.float32).reshape(32, 4)
    rhs = np.linspace(3.0, -9.0, 32 * 4, dtype=np.float32).reshape(32, 4)
    output = np.zeros((32, 4), dtype=np.float32)

    module = numsim.transpile(elementwise_maximum, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"lhs": lhs, "rhs": rhs, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.maximum(lhs, rhs))


def test_float64_elementwise_forms_use_native_double_semantics(tmp_path):
    lhs = np.linspace(0.125, 2.0, 32 * 4, dtype=np.float64).reshape(32, 4)
    rhs = np.linspace(0.75, 1.5, 32 * 4, dtype=np.float64).reshape(32, 4)
    output = np.zeros((32, 10, 4), dtype=np.float64)

    module = numsim.transpile(float64_elementwise_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"lhs": lhs, "rhs": rhs, "output": output})

    expected = np.stack(
        (
            lhs + rhs,
            lhs - rhs,
            lhs * rhs,
            lhs / rhs,
            lhs * rhs + np.float64(0.25),
            np.sqrt(lhs),
            np.exp(lhs),
            np.exp2(lhs),
            np.float64(1) / lhs,
            lhs / (np.float64(1) + np.exp(-lhs)),
        ),
        axis=1,
    )
    np.testing.assert_allclose(result.outputs["output"], expected, rtol=2e-15, atol=0.0)


def test_float64_directed_rounding_fails_closed(tmp_path):
    with pytest.raises(numsim.UnsupportedTIRxError, match="directed float64 rounding"):
        numsim.transpile(float64_directed_rounding_is_unsupported, cache_dir=tmp_path)


def test_cast_uses_general_scalar_conversion_without_dispatch_semantics(tmp_path):
    integers = np.resize(np.array([-3, 0, 17, 255], dtype=np.int32), (32, 4)).copy()
    floats = np.resize(np.array([0.0, 1.9, 17.25, 255.75], dtype=np.float32), (32, 4)).copy()
    output_float = np.zeros((32, 4), dtype=np.float64)
    output_byte = np.zeros((32, 4), dtype=np.uint8)
    output_bool = np.zeros((32, 4), dtype=np.bool_)

    module = numsim.transpile(cast_uses_general_scalar_conversion, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "integers": integers,
            "floats": floats,
            "output_float": output_float,
            "output_byte": output_byte,
            "output_bool": output_bool,
        },
    )

    np.testing.assert_array_equal(result.outputs["output_float"], integers.astype(np.float64))
    np.testing.assert_array_equal(result.outputs["output_byte"], floats.astype(np.uint8))
    np.testing.assert_array_equal(result.outputs["output_bool"], integers.astype(np.bool_))


def test_mxfp4_uses_ue8m0_scales_over_32_element_vectors(tmp_path):
    left_codes = np.resize(
        np.array([0x0, 0x1, 0x2, 0x3, 0x7, 0x9, 0xA, 0xF], dtype=np.uint8), (128, 64)
    )
    right_codes = np.resize(np.array([0x1, 0x2, 0x4, 0x7, 0x9, 0xB], dtype=np.uint8), (8, 64))
    left_packed = (left_codes[:, 0::2] | (left_codes[:, 1::2] << np.uint8(4))).astype(np.uint8)
    right_packed = (right_codes[:, 0::2] | (right_codes[:, 1::2] << np.uint8(4))).astype(np.uint8)
    scale_a = np.resize(np.array([126, 128], dtype=np.uint8), (128, 2))
    scale_b = np.resize(np.array([[127, 129], [128, 126]], dtype=np.uint8), (8, 2))
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(block_scaled_mxfp4_gemm, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left_packed": left_packed,
            "right_packed": right_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": output,
        },
    )

    decoded_a = _decode_e2m1(_unpack_e2m1(left_packed)).reshape(128, 2, 32)
    decoded_b = _decode_e2m1(_unpack_e2m1(right_packed)).reshape(8, 2, 32)
    a_scales = np.exp2(scale_a.astype(np.int16) - 127).astype(np.float32)
    b_scales = np.exp2(scale_b.astype(np.int16) - 127).astype(np.float32)
    expected = (decoded_a * a_scales[:, :, None]).reshape(128, 64) @ (
        decoded_b * b_scales[:, :, None]
    ).reshape(8, 64).T
    np.testing.assert_array_equal(result.outputs["output"], expected)
