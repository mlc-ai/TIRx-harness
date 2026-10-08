"""Readonly is a kernel-lifetime memory contract, not just a cache hint."""

import numpy as np
import pytest
from tvm.script import tirx as T

from tests.numsim.support.execution import assert_rejected, run_checked


def readonly_kernel(*, before=False, after=False, disjoint=False, predicate=True):
    write_offset = 32 if disjoint else 0

    @T.prim_func
    def kernel(data: T.Buffer((64,), "uint32"), output: T.Buffer((32,), "uint32")):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        value = T.alloc_local((1,), "uint32")
        value[0] = T.uint32(123)
        if before:
            data[lane + write_offset] = T.cast(lane, "uint32")
        T.ptx["ld.global.u32.proxy::readonly"](value[0], data.ptr_to([lane]), pred=predicate, preserve_dst=True)
        if after:
            data[lane + write_offset] = T.cast(lane, "uint32")
        output[lane] = value[0]

    return kernel


@pytest.mark.parametrize(
    "options",
    [{}, {"before": True, "after": True, "disjoint": True}, {"after": True, "predicate": False}],
)
def test_readonly_proxy_values_and_disjoint_writes(options, tmp_path):
    kernel = readonly_kernel(**options)
    args = {"data": np.arange(64, dtype=np.uint32), "output": np.zeros(32, np.uint32)}
    result = run_checked(kernel, args, cache_dir=tmp_path)
    expected = np.arange(32, dtype=np.uint32) if options.get("predicate", True) else 123
    np.testing.assert_array_equal(result.outputs["output"], np.broadcast_to(expected, (32,)))


@pytest.mark.parametrize("order", ["before", "after"])
def test_readonly_proxy_rejects_same_value_write_in_either_order(order, tmp_path):
    kernel = readonly_kernel(**{order: True})
    args = {"data": np.arange(64, dtype=np.uint32), "output": np.zeros(32, np.uint32)}
    assert_rejected(kernel, args, "write overlaps readonly bytes", cache_dir=tmp_path)
