"""Discard invalidates only its 128-byte range and is not a prefetch hint."""

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness import racecheck, synccheck
from tests.numsim.support.execution import run_checked


def discard_kernel(*, restore=True, offset=0, predicate=True):
    @T.prim_func
    def kernel(data: T.Buffer((64,), "uint32"), output: T.Buffer((2,), "uint32")):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        if lane == 0:
            T.ptx.discard.global_.L2(data.ptr_to([offset]), pred=predicate)
            if restore:
                data[0] = T.uint32(91)
            output[0] = data[0]
            output[1] = data[32]

    return kernel


@pytest.mark.parametrize("restore,predicate,expected", [(True, True, 91), (False, False, 0)])
def test_discard_range_and_predication(restore, predicate, expected, tmp_path):
    kernel = discard_kernel(restore=restore, predicate=predicate)
    args = {"data": np.arange(64, dtype=np.uint32), "output": np.zeros(2, np.uint32)}
    result = run_checked(kernel, args, cache_dir=tmp_path, outputs=("output",))
    np.testing.assert_array_equal(result.outputs["output"], [expected, 32])


def test_discard_indeterminate_read_and_alignment():
    args = {"data": np.arange(64, dtype=np.uint32), "output": np.zeros(2, np.uint32)}
    for checker in (synccheck, racecheck):
        report = checker(discard_kernel(restore=False), args)
        assert report.verdict == "review"
        assert any(f.status == "review" and "uninitialized" in f.kind for f in report.findings)
        report = checker(discard_kernel(offset=1), args)
        assert report.verdict == "error"
        assert "128-byte aligned" in str(report.to_dict())
