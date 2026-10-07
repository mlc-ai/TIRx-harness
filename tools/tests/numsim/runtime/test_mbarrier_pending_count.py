"""The noComplete token retains a pre-arrival count after later arrivals."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked


def pending_count_kernel(*, drop=False):
    action = "arrive_drop" if drop else "arrive"
    return tvm.script.from_source(
        f"""
@T.prim_func
def pending_counts(out: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    saved = T.alloc_local((1,), "uint64")
    scratch = T.alloc_local((1,), "uint64")
    count = T.alloc_local((1,), "uint32")
    ready = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 5)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane < 2:
        T.ptx["mbarrier.{action}.noComplete.shared.b64"](
            saved[0], barrier.ptr_to([0]), T.uint32(lane + 1))
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(scratch[0], barrier.ptr_to([0]), T.uint32(2))
    if lane < 2:
        T.ptx["mbarrier.pending_count.layout::v0.b64"](count[0], saved[0])
        out[lane] = count[0]
        T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), saved[0])
        out[lane + 2] = ready[0]
""",
        {"T": T},
    )


@pytest.mark.parametrize("drop", [False, True])
def test_pending_count_snapshot(drop, tmp_path):
    kernel = pending_count_kernel(drop=drop)
    inputs = {"out": np.zeros(4, np.uint32)}
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], [5, 4, 1, 1])
