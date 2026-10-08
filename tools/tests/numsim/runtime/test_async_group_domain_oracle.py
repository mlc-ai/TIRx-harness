from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim
from tests.numsim.support.manifest import evaluated_kernel, resolved_kernel


@T.prim_func
def async_group_wait_counts(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(8)
    T.ptx.cp.async_.wait_group(0)
    T.ptx.cp.async_.bulk.commit_group()
    T.ptx.cp.async_.bulk.wait_group(64)
    T.ptx.cp.async_.bulk.wait_group.read(255)
    T.ptx.cp.async_.bulk.wait_group.read(2147483647)
    T.ptx.cp.async_.bulk.wait_group(0)
    if lane == 0:
        output[0] = 1


def test_async_group_wait_counts_complete_on_an_empty_queue(tmp_path):
    module = numsim.transpile(async_group_wait_counts, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "output": np.zeros(1, dtype=np.int32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.int32))
    assert result.stats["task_count"] == 1
    assert result.stats["completed_task_count"] == 1


@pytest.mark.parametrize("opcode", ("cp.async.wait_group", "cp.async.bulk.wait_group.read"))
def test_async_group_wait_count_rejects_negative_values(opcode):
    call = T.ptx[opcode](-1)
    with pytest.raises(numsim.UnsupportedTIRxError, match="must be a non-negative int64"):
        resolved_kernel(evaluated_kernel(call))


def test_async_group_wait_count_rejects_runtime_values():
    pending = tvm.tirx.Var("pending", "int32")
    call = T.ptx.cp.async_.wait_group(pending)
    with pytest.raises(numsim.UnsupportedTIRxError, match="must be static"):
        resolved_kernel(evaluated_kernel(call, (pending,)))
