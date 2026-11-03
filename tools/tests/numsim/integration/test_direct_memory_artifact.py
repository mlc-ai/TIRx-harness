from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm_tirx.script import tirx as T


@T.function
def raw_cp_async_zero_fill(
    source: T.Tensor((128,), "float32"), output: T.Tensor((128,), "float32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "float32")
    T.evaluate(
        T.ptx["cp.async.cg.shared.global.L2::128B"](
            T.address_of(shared[lane * 4]),
            T.address_of(source[lane * 4]),
            16,
            T.cast(T.if_then_else(lane < 16, 16, 0), "uint32"),
        )
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


@T.function
def pass_emitted_cp_async_raw(
    source: T.Tensor((128,), "float32"), output: T.Tensor((128,), "float32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "float32")
    T.ptx["cp.async.cg.shared.global"](shared.ptr_to([lane * 4]), source.ptr_to([lane * 4]), 16)
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


def test_raw_cp_async_issues_nonbulk_group_for_zero_fill_lanes(tmp_path):
    source = np.arange(128, dtype=np.float32) + np.float32(0.25)
    output = np.full_like(source, np.float32(-1))

    module = numsim.transpile(raw_cp_async_zero_fill, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = source.copy()
    expected[64:] = np.float32(0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_pass_emitted_cp_async_raw_restores_element_offsets(tmp_path):
    source = np.arange(128, dtype=np.float32) + np.float32(0.375)
    output = np.zeros_like(source)

    module = numsim.transpile(pass_emitted_cp_async_raw, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)
