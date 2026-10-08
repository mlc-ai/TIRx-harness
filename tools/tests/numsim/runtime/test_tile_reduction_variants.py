from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from tirx_harness import numsim
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import R, S, TileLayout, laneid


@T.prim_func
def shared_cta_f16_reductions(
    source: T.Buffer((64, 4), "float16"), output: T.Buffer((3, 64), "float16")
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    thread = warp * 32 + lane
    source_shared = T.alloc_buffer(
        (64, 4), "float16", scope="shared", layout=TileLayout(S[(64, 4)])
    )
    result_shared = T.alloc_buffer((64,), "float16", scope="shared", layout=TileLayout(S[64]))
    for column in T.serial(4):
        source_shared[thread, column] = source[thread, column]
    T.cuda.cta_sync()

    Tx.cta.sum(result_shared, source_shared, axes=[1], dispatch="shared")
    output[0, thread] = result_shared[thread]
    T.cuda.cta_sync()
    Tx.cta.max(result_shared, source_shared, axes=[1], dispatch="shared")
    output[1, thread] = result_shared[thread]
    T.cuda.cta_sync()
    Tx.cta.min(result_shared, source_shared, axes=[1], dispatch="shared")
    output[2, thread] = result_shared[thread]


@T.prim_func
def local_bf16_reductions(
    source: T.Buffer((32, 4), "bfloat16"), output: T.Buffer((32, 3), "bfloat16")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    values: T.bf16[4]
    result: T.bf16[1]
    for column in T.serial(4):
        values[column] = source[lane, column]
    Tx.sum(result, values, dispatch="local")
    output[lane, 0] = result[0]
    Tx.max(result, values, dispatch="local")
    output[lane, 1] = result[0]
    Tx.min(result, values, dispatch="local")
    output[lane, 2] = result[0]


@T.prim_func
def shared_empty_axis_reductions(
    source: T.Buffer((32,), "float32"), output: T.Buffer((3, 32), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    values = T.alloc_buffer((32,), "float32", scope="shared", layout=TileLayout(S[32]))
    result = T.alloc_buffer((32,), "float32", scope="shared", layout=TileLayout(S[32]))
    values[lane] = source[lane]
    T.cuda.warp_sync()
    Tx.warp.sum(result[:], values[:], axes=[], dispatch="shared")
    output[0, lane] = result[lane]
    Tx.warp.max(result[:], values[:], axes=[], dispatch="shared")
    output[1, lane] = result[lane]
    Tx.warp.min(result[:], values[:], axes=[], dispatch="shared")
    output[2, lane] = result[lane]


@T.prim_func
def warp_collective_reductions(
    source: T.Buffer((32, 2), "float32"), output: T.Buffer((3, 32, 2), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_local: T.f32[2]
    result_local: T.f32[2]
    for column in T.serial(2):
        source_local[column] = source[lane, column]
    source_view = source_local.view(32, 2, layout=TileLayout(S[(32, 2) : (1 @ laneid, 1)]))
    result_view = result_local.view(2, layout=TileLayout(S[2:1] + R[32 : 1 @ laneid]))
    Tx.warp.sum(result_view, source_view, axes=[0], dispatch="local")
    for column in T.serial(2):
        output[0, lane, column] = result_local[column]
    Tx.warp.max(result_view, source_view, axes=[0], dispatch="local")
    for column in T.serial(2):
        output[1, lane, column] = result_local[column]
    Tx.warp.min(result_view, source_view, axes=[0], dispatch="local")
    for column in T.serial(2):
        output[2, lane, column] = result_local[column]


def _make_partial_warp_sum(reduce_width: int):
    @T.prim_func
    def partial_warp_sum(
        source: T.Buffer((32, 2), "float32"), output: T.Buffer((32, 2), "float32")
    ):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        source_local: T.f32[2]
        result_local: T.f32[2]
        for column in T.serial(2):
            source_local[column] = source[lane, column]
        source_view = source_local.view(32, 2, layout=TileLayout(S[(32, 2) : (1 @ laneid, 1)]))
        result_view = result_local.view(2, layout=TileLayout(S[2:1] + R[reduce_width : 1 @ laneid]))
        Tx.warp.sum(result_view, source_view, axes=[0], dispatch="local")
        for column in T.serial(2):
            output[lane, column] = result_local[column]

    return partial_warp_sum


@T.prim_func
def shared_cta_accum_sum(
    source: T.Buffer((64, 4), "float32"),
    initial: T.Buffer((64,), "float32"),
    output: T.Buffer((64,), "float32"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    thread = warp * 32 + lane
    source_shared = T.alloc_buffer(
        (64, 4), "float32", scope="shared", layout=TileLayout(S[(64, 4)])
    )
    result_shared = T.alloc_buffer((64,), "float32", scope="shared", layout=TileLayout(S[64]))
    for column in T.serial(4):
        source_shared[thread, column] = source[thread, column]
    result_shared[thread] = initial[thread]
    T.cuda.cta_sync()

    Tx.cta.sum(result_shared, source_shared, axes=[1], accum=True, dispatch="shared")
    output[thread] = result_shared[thread]


@T.prim_func
def three_input_maxmin_order(
    source: T.Buffer((2, 8), "float32"), output: T.Buffer((4, 32), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    values: T.f32[8]
    result: T.f32[1]
    for row in T.serial(2):
        for index in T.serial(8):
            values[index] = source[row, index]
        Tx.max(result, values, dispatch="3input_maxmin")
        output[row * 2, lane] = result[0]
        Tx.min(result, values, dispatch="3input_maxmin")
        output[row * 2 + 1, lane] = result[0]


def _make_local_typed_reductions(dtype: str):
    @T.prim_func
    def local_typed_reductions(source: T.Buffer((32, 4), dtype), output: T.Buffer((32, 3), dtype)):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        values = T.alloc_buffer((4,), dtype, scope="local")
        result = T.alloc_buffer((1,), dtype, scope="local")
        for index in T.serial(4):
            values[index] = source[lane, index]
        Tx.sum(result, values)
        output[lane, 0] = result[0]
        Tx.max(result, values)
        output[lane, 1] = result[0]
        Tx.min(result, values)
        output[lane, 2] = result[0]

    return local_typed_reductions


def _stepwise_low_precision_sum(
    values: np.ndarray, dtype, initial: np.ndarray | None = None
) -> np.ndarray:
    result = []
    for row_index, row in enumerate(values):
        accumulator = dtype(0 if initial is None else initial[row_index])
        for value in row:
            accumulator = dtype(dtype(accumulator) + dtype(value))
        result.append(accumulator)
    return np.asarray(result, dtype=dtype)


def _grouped_lexicographic_reduce(values: np.ndarray, width: int, dtype, reducer) -> np.ndarray:
    result = np.empty_like(values, dtype=dtype)
    for group_start in range(0, values.shape[0], width):
        group = values[group_start : group_start + width]
        reduced = []
        for column in range(values.shape[1]):
            accumulator = dtype(group[0, column])
            for row in range(1, len(group)):
                accumulator = dtype(reducer(accumulator, dtype(group[row, column])))
            reduced.append(accumulator)
        result[group_start : group_start + width] = np.asarray(reduced, dtype=dtype)
    return result


def test_shared_cta_reductions_support_float16_and_scope_completion(tmp_path):
    source = np.linspace(-2, 3, 64 * 4, dtype=np.float32).reshape(64, 4).astype(np.float16)
    output = np.zeros((3, 64), dtype=np.float16)

    module = numsim.transpile(shared_cta_f16_reductions, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.stack(
        [
            _stepwise_low_precision_sum(source, np.float16),
            np.max(source, axis=1),
            np.min(source, axis=1),
        ]
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_local_reductions_round_each_bfloat16_sum_step(tmp_path):
    dtype = ml_dtypes.bfloat16
    source = np.linspace(-1, 2, 32 * 4, dtype=np.float32).reshape(32, 4).astype(dtype)
    output = np.zeros((32, 3), dtype=dtype)

    module = numsim.transpile(local_bf16_reductions, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.stack(
        [
            _stepwise_low_precision_sum(source, dtype),
            np.max(source, axis=1),
            np.min(source, axis=1),
        ],
        axis=1,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_local_float64_reductions_match_b200_edge_bits(tmp_path):
    nan = np.asarray([0x7FF8_0000_0000_1234], dtype=np.uint64).view(np.float64)[0]
    rows = np.asarray(
        [
            [nan, -0.0, 0.0, 3.0],
            [nan, nan, nan, nan],
            [-0.0, 0.0, -0.0, 0.0],
            [np.inf, -np.inf, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    source = np.resize(rows, (32, 4)).copy()
    output = np.zeros((32, 3), dtype=np.float64)

    module = numsim.transpile(_make_local_typed_reductions("float64"), cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected_bits = np.asarray(
        [
            [0x7FF8_0000_0000_1234, 0x4008_0000_0000_0000, 0x8000_0000_0000_0000],
            [0x7FF8_0000_0000_1234, 0xFFEF_FFFF_FFFF_FFFF, 0x7FEF_FFFF_FFFF_FFFF],
            [0x0000_0000_0000_0000, 0x0000_0000_0000_0000, 0x8000_0000_0000_0000],
            [0xFFF8_0000_0000_0000, 0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000],
        ],
        dtype=np.uint64,
    )
    actual_bits = result.outputs["output"][:4].view(np.uint64)
    np.testing.assert_array_equal(actual_bits, expected_bits)


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
def test_local_integer_reductions_wrap_and_preserve_order(dtype, numpy_dtype, tmp_path):
    info = np.iinfo(numpy_dtype)
    row = np.asarray([info.max, 1, 2, info.min], dtype=numpy_dtype)
    source = np.resize(row, (32, 4)).copy()
    output = np.zeros((32, 3), dtype=numpy_dtype)

    module = numsim.transpile(_make_local_typed_reductions(dtype), cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    accumulator = numpy_dtype(0)
    with np.errstate(over="ignore"):
        for value in row:
            accumulator = np.add(accumulator, value, dtype=numpy_dtype)
    expected = np.asarray([accumulator, row.max(), row.min()], dtype=numpy_dtype)
    np.testing.assert_array_equal(result.outputs["output"], np.broadcast_to(expected, output.shape))


def test_warp_collective_reduction_follows_physical_lane_ownership(tmp_path):
    source = np.linspace(-4, 3, 64, dtype=np.float32).reshape(32, 2)
    output = np.zeros((3, 32, 2), dtype=np.float32)

    module = numsim.transpile(warp_collective_reductions, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected_sum = _grouped_lexicographic_reduce(
        source, 32, np.float32, lambda lhs, rhs: np.float32(lhs + rhs)
    )
    expected = np.stack(
        [
            expected_sum,
            _grouped_lexicographic_reduce(source, 32, np.float32, max),
            _grouped_lexicographic_reduce(source, 32, np.float32, min),
        ]
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_partial_warp_collective_reduction_preserves_lane_groups(tmp_path):
    source = np.arange(32 * 2, dtype=np.float32).reshape(32, 2) + np.float32(0.25)

    for reduce_width in (1, 2, 4, 8, 16, 32):
        output = np.zeros((32, 2), dtype=np.float32)
        module = numsim.transpile(
            _make_partial_warp_sum(reduce_width), cache_dir=tmp_path / str(reduce_width)
        )
        result = numsim.Engine().run(module, {"source": source, "output": output})

        expected = np.empty_like(source)
        for group_start in range(0, 32, reduce_width):
            group_sum = _grouped_lexicographic_reduce(
                source[group_start : group_start + reduce_width],
                reduce_width,
                np.float32,
                lambda lhs, rhs: np.float32(lhs + rhs),
            )
            expected[group_start : group_start + reduce_width] = group_sum
        np.testing.assert_array_equal(result.outputs["output"], expected)


def test_shared_cta_accum_reduction_writes_each_output_once(tmp_path):
    source = np.linspace(-1.5, 2.0, 64 * 4, dtype=np.float32).reshape(64, 4)
    initial = np.linspace(3.0, 5.0, 64, dtype=np.float32)
    output = np.zeros((64,), dtype=np.float32)

    module = numsim.transpile(shared_cta_accum_sum, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "initial": initial, "output": output})

    expected = _stepwise_low_precision_sum(source, np.float32, initial)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_shared_reduction_uses_lexicographic_order(tmp_path):
    source = np.zeros((64, 4), dtype=np.float16)
    source[:] = np.asarray([10000.0, 1.0, -10000.0, 1.0], dtype=np.float16)
    output = np.zeros((3, 64), dtype=np.float16)

    result = numsim.Engine().run(
        numsim.transpile(shared_cta_f16_reductions, cache_dir=tmp_path),
        {"source": source, "output": output},
    )

    np.testing.assert_array_equal(result.outputs["output"][0], np.ones(64, dtype=np.float16))


def test_empty_reduction_axes_follow_identity_reduction_semantics(tmp_path):
    source = np.linspace(-3, 5, 32, dtype=np.float32)
    output = np.zeros((3, 32), dtype=np.float32)

    result = numsim.Engine().run(
        numsim.transpile(shared_empty_axis_reductions, cache_dir=tmp_path),
        {"source": source, "output": output},
    )

    np.testing.assert_array_equal(result.outputs["output"], np.broadcast_to(source, output.shape))


def test_local_collective_uses_lexicographic_order(tmp_path):
    source = np.zeros((32, 2), dtype=np.float32)
    source[:4, 0] = np.asarray([1.0e20, 1.0, -1.0e20, 1.0], dtype=np.float32)
    output = np.zeros((3, 32, 2), dtype=np.float32)

    result = numsim.Engine().run(
        numsim.transpile(warp_collective_reductions, cache_dir=tmp_path),
        {"source": source, "output": output},
    )

    np.testing.assert_array_equal(
        result.outputs["output"][0, :, 0], np.full(32, 1.0, dtype=np.float32)
    )


def test_maxmin_uses_canonical_lexicographic_nan_and_signed_zero_order(tmp_path):
    nan = np.asarray([0x7FC0_1234], dtype=np.uint32).view(np.float32)[0]
    source = np.asarray(
        [
            [nan, -0.0, 0.0, nan, -np.inf, -np.inf, nan, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan],
        ],
        dtype=np.float32,
    )
    output = np.zeros((4, 32), dtype=np.float32)

    result = (
        numsim.Engine()
        .run(
            numsim.transpile(three_input_maxmin_order, cache_dir=tmp_path),
            {"source": source, "output": output},
        )
        .outputs["output"]
    )

    bits = result.view(np.uint32)
    assert np.all(bits[0] == np.uint32(0x0000_0000))
    assert np.all(bits[1] == np.uint32(0xFF80_0000))
    assert np.all(bits[2] == np.uint32(0xFF7F_FFFF))
    assert np.all(bits[3] == np.uint32(0x7F7F_FFFF))
