"""Remaining sync predicates reuse the existing lane mask and register contract."""

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


def pending_predicate_kernel(*, drop=False, valid=True):
    action = "arrive_drop" if drop else "arrive"
    return tvm.script.from_source(
        f"""@T.prim_func
def pending(output: T.Buffer((3, 32), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_shared((1,), "uint64", align=8)
    state = T.alloc_local((1,), "uint64")
    result = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 2)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["mbarrier.{action}{".noComplete" if valid else ""}.shared.b64"](
            state[0], barrier.ptr_to([0]), T.uint32(1))
    result[0] = T.uint32(123)
    T.ptx.mbarrier.pending_count.b64(result[0], state[0], pred=lane == 0, preserve_dst=True)
    output[0, lane] = result[0]
    result[0] = T.uint32(123)
    T.ptx.mbarrier.pending_count.b64(result[0], state[0], pred=lane == 0)
    output[1, lane] = result[0]
    result[0] = T.uint32(123)
    T.ptx.mbarrier.pending_count.b64(result[0], state[0], pred=False, preserve_dst=True)
    output[2, lane] = result[0]
""",
        {"T": T},
    )


def pending_predicate_expected():
    expected = np.full((3, 32), 123, np.uint32)
    expected[1] = 0  # Only lane 0 is defined on GPU for the non-preserving call.
    expected[:2, 0] = 2
    return expected


def test_pending_count_instruction_predicates(tmp_path):
    for drop in (False, True):
        kernel = pending_predicate_kernel(drop=drop)
        inputs = {"output": np.zeros((3, 32), np.uint32)}
        for checker in (synccheck, racecheck):
            checker(kernel, inputs).require_clean()
            invalid = checker(pending_predicate_kernel(drop=drop, valid=False), inputs)
            assert invalid.verdict == "error", invalid.format()
            assert "noComplete" in str(invalid.to_dict())
        actual = numsim.Engine().run(numsim.transpile(kernel, cache_dir=tmp_path), inputs)
        np.testing.assert_array_equal(actual.outputs["output"], pending_predicate_expected())
