from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm_tirx.script import tirx as T


@T.function
def warp_float64_reductions(output: T.Tensor((32, 3), "float64")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    value: T.let = T.cast(lane, "float64") * T.float64(0.5) - T.float64(8)
    output[lane, 0] = T.cuda.warp_sum(value, width=8)
    output[lane, 1] = T.cuda.warp_max(value, width=8)
    output[lane, 2] = T.cuda.warp_min(value, width=8)


@T.function
def cta_int32_reductions(output: T.Tensor((3, 3), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    sum_scratch = T.alloc_shared((2,), "int32")
    max_scratch = T.alloc_shared((2,), "int32")
    min_scratch = T.alloc_shared((2,), "int32")
    value: T.let = warp * 100 + lane - 50
    sum_result: T.let = T.cuda.cta_sum(value, 2, sum_scratch.ptr_to([0]))
    max_result: T.let = T.cuda.cta_max(value, 2, max_scratch.ptr_to([0]))
    min_result: T.let = T.cuda.cta_min(value, 2, min_scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        output[0, 0] = sum_result
        output[0, 1] = sum_scratch[0]
        output[0, 2] = sum_scratch[1]
        output[1, 0] = max_result
        output[1, 1] = max_scratch[0]
        output[1, 2] = max_scratch[1]
        output[2, 0] = min_result
        output[2, 1] = min_scratch[0]
        output[2, 2] = min_scratch[1]


def test_warp_float64_reductions_use_generic_typed_runtime(tmp_path):
    spec = analyze(warp_float64_reductions)
    assert spec.unsupported == ()
    module = numsim.transpile(warp_float64_reductions, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((32, 3), dtype=np.float64)})

    values = np.arange(32, dtype=np.float64) * 0.5 - 8.0
    expected = np.empty((32, 3), dtype=np.float64)
    for group_start in range(0, 32, 8):
        group = values[group_start : group_start + 8]
        expected[group_start : group_start + 8, 0] = group.sum()
        expected[group_start : group_start + 8, 1] = group.max()
        expected[group_start : group_start + 8, 2] = group.min()
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cta_int32_reductions_preserve_typed_scratch(tmp_path):
    spec = analyze(cta_int32_reductions)
    assert spec.unsupported == ()
    module = numsim.transpile(cta_int32_reductions, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((3, 3), dtype=np.int32)})

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.array([[992, 992, 2096], [81, 81, 81], [-50, -50, 50]], dtype=np.int32),
    )
