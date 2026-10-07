from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm.script import tirx as T


@T.prim_func
def cuda_float16_reductions(output: T.Buffer((2, 32, 6), "float32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    sum_scratch = T.alloc_buffer((2,), "float16", scope="shared")
    max_scratch = T.alloc_buffer((2,), "float16", scope="shared")
    min_scratch = T.alloc_buffer((2,), "float16", scope="shared")
    raw: T.float32 = T.if_then_else(
        lane == 0,
        T.float32(1.0) + T.cast(warp, "float32") * T.float32(0.5),
        T.float32(0.0006) * (T.cast(warp, "float32") + T.float32(1.0)),
    )
    value: T.float16 = T.cast(raw, "float16")
    output[warp, lane, 0] = T.cast(T.cuda.warp_sum(value, width=8), "float32")
    output[warp, lane, 1] = T.cast(T.cuda.warp_max(value, width=8), "float32")
    output[warp, lane, 2] = T.cast(T.cuda.warp_min(value, width=8), "float32")
    output[warp, lane, 3] = T.cast(T.cuda.cta_sum(value, 2, sum_scratch.ptr_to([0])), "float32")
    output[warp, lane, 4] = T.cast(T.cuda.cta_max(value, 2, max_scratch.ptr_to([0])), "float32")
    output[warp, lane, 5] = T.cast(T.cuda.cta_min(value, 2, min_scratch.ptr_to([0])), "float32")


@T.prim_func
def cuda_bfloat16_reductions(output: T.Buffer((2, 32, 6), "float32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    sum_scratch = T.alloc_buffer((2,), "bfloat16", scope="shared")
    max_scratch = T.alloc_buffer((2,), "bfloat16", scope="shared")
    min_scratch = T.alloc_buffer((2,), "bfloat16", scope="shared")
    raw: T.float32 = T.if_then_else(
        lane == 0,
        T.float32(1.0) + T.cast(warp, "float32") * T.float32(0.5),
        T.float32(0.0006) * (T.cast(warp, "float32") + T.float32(1.0)),
    )
    value: T.bfloat16 = T.cast(raw, "bfloat16")
    output[warp, lane, 0] = T.cast(T.cuda.warp_sum(value, width=8), "float32")
    output[warp, lane, 1] = T.cast(T.cuda.warp_max(value, width=8), "float32")
    output[warp, lane, 2] = T.cast(T.cuda.warp_min(value, width=8), "float32")
    output[warp, lane, 3] = T.cast(T.cuda.cta_sum(value, 2, sum_scratch.ptr_to([0])), "float32")
    output[warp, lane, 4] = T.cast(T.cuda.cta_max(value, 2, max_scratch.ptr_to([0])), "float32")
    output[warp, lane, 5] = T.cast(T.cuda.cta_min(value, 2, min_scratch.ptr_to([0])), "float32")


@T.prim_func
def cuda_float16_zero_reductions(output: T.Buffer((32, 2), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    raw: T.float32 = T.if_then_else(lane % 2 == 0, T.float32(0.0), T.float32(-0.0))
    value: T.float16 = T.cast(raw, "float16")
    output[lane, 0] = T.cast(T.cuda.warp_max(value, width=2), "float32")
    output[lane, 1] = T.cast(T.cuda.warp_min(value, width=2), "float32")


@T.prim_func
def cuda_bfloat16_zero_reductions(output: T.Buffer((32, 2), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    raw: T.float32 = T.if_then_else(lane % 2 == 0, T.float32(0.0), T.float32(-0.0))
    value: T.bfloat16 = T.cast(raw, "bfloat16")
    output[lane, 0] = T.cast(T.cuda.warp_max(value, width=2), "float32")
    output[lane, 1] = T.cast(T.cuda.warp_min(value, width=2), "float32")


def _round_fp16(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).astype(np.float32)


def _round_bf16(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((rounded >> np.uint32(16)) << np.uint32(16)).view(np.float32)


def _butterfly_reduce(
    values: np.ndarray,
    width: int,
    operation: str,
    round_value: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    result = round_value(np.asarray(values, dtype=np.float32))
    delta = width // 2
    while delta:
        previous = result.copy()
        source = np.arange(result.size) ^ delta
        if operation == "sum":
            combined = previous + previous[source]
        elif operation == "max":
            combined = np.where(previous > previous[source], previous, previous[source])
        else:
            combined = np.where(previous < previous[source], previous, previous[source])
        result = round_value(combined)
        delta //= 2
    return result


def _expected_low_precision_reductions(
    round_value: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    raw = np.full((2, 32), 0.0006, dtype=np.float32)
    raw[1] *= np.float32(2.0)
    raw[0, 0] = np.float32(1.0)
    raw[1, 0] = np.float32(1.5)
    values = round_value(raw)
    expected = np.empty((2, 32, 6), dtype=np.float32)

    for operation_index, operation in enumerate(("sum", "max", "min")):
        for warp in range(2):
            expected[warp, :, operation_index] = _butterfly_reduce(
                values[warp], 8, operation, round_value
            )

        identity = {"sum": 0.0, "max": -np.inf, "min": np.inf}[operation]
        cross_warp = np.full(32, identity, dtype=np.float32)
        for warp in range(2):
            cross_warp[warp] = _butterfly_reduce(values[warp], 32, operation, round_value)[0]
        cta_result = _butterfly_reduce(cross_warp, 32, operation, round_value)[0]
        expected[:, :, operation_index + 3] = cta_result

    return expected


@pytest.mark.parametrize(
    ("kernel", "dtype_name", "round_value"),
    [
        (cuda_float16_reductions, "fp16", _round_fp16),
        (cuda_bfloat16_reductions, "bf16", _round_bf16),
    ],
)
def test_cuda_low_precision_reductions_follow_typed_butterfly_semantics(
    tmp_path, kernel, dtype_name, round_value
):
    spec = analyze(kernel)
    assert spec.unsupported == ()
    module = numsim.transpile(kernel, cache_dir=tmp_path / dtype_name)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((2, 32, 6), dtype=np.float32)},
    )

    np.testing.assert_array_equal(
        result.outputs["output"],
        _expected_low_precision_reductions(round_value),
    )


@pytest.mark.parametrize(
    ("kernel", "dtype_name"),
    [
        (cuda_float16_zero_reductions, "fp16"),
        (cuda_bfloat16_zero_reductions, "bf16"),
    ],
)
def test_cuda_low_precision_max_min_select_the_rhs_for_equal_zeros(tmp_path, kernel, dtype_name):
    module = numsim.transpile(kernel, cache_dir=tmp_path / dtype_name)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((32, 2), dtype=np.float32)},
    )

    bits = result.outputs["output"].view(np.uint32)
    np.testing.assert_array_equal(
        bits[:2],
        np.array([[0x80000000, 0x80000000], [0x00000000, 0x00000000]], dtype=np.uint32),
    )
