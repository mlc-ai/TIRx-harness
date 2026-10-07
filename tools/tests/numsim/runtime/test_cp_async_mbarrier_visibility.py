"""A legacy copy arrival publishes copies, including through relaxed queries."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


def copy_arrival_case(
    *,
    unrelated=False,
    relaxed=False,
    committed=False,
    waited=False,
    extra=False,
    reuse_source=False,
):
    wait = (
        """ready[0] = T.uint32(0)
            while ready[0] == 0:
                T.ptx.mbarrier.test_wait.parity.relaxed.cta.shared.b64(
                    ready[0], barrier.ptr_to([0]), T.uint32(0))"""
        if relaxed
        else """T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)"""
    )
    extra_copy = (
        """T.ptx["cp.async.ca.shared.global"](
                shared.ptr_to([8]), source.ptr_to([0]), 16)"""
        if extra
        else "T.evaluate(0)"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(source: T.Buffer((4,), "uint32"), output: T.Buffer((1,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_shared((12,), "uint32", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=16)
    ready = T.alloc_local((1,), "uint32")
    if warp == 1 and lane == 0:
        shared[4] = T.uint32(0)
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        if warp == 0:
            shared[4] = T.uint32(42)
            T.ptx["cp.async.ca.shared.global"](shared.ptr_to([0]), source.ptr_to([0]), 16)
            {"T.ptx.cp.async_.commit_group()" if committed else "T.evaluate(0)"}
            {"T.ptx.cp.async_.wait_group(0)" if waited else "T.evaluate(0)"}
            {extra_copy}
            T.ptx.cp.async_.mbarrier.arrive.noinc.shared.b64(barrier.ptr_to([0]))
        else:
            {wait}
            {"source[0] = T.uint32(42)" if reuse_source else "T.evaluate(0)"}
            output[0] = {"shared[4]" if unrelated else "shared[0]"}
""",
        {"T": T},
    )


def inputs():
    return {"source": np.arange(7, 11, dtype=np.uint32), "output": np.zeros(1, np.uint32)}


def arrived_open_batch_case(
    *, commit=False, pending=0, empty_tail=False, observe_barrier=False, overwrite_after_wait=False
):
    overwrite = (
        """T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        shared[0] = T.uint32(42)"""
        if overwrite_after_wait
        else "T.evaluate(0)"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def kernel(source: T.Buffer((4,), "uint32"), output: T.Buffer((1,), "uint32")):
    T.device_entry()
    warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_shared((4,), "uint32", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=16)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx["cp.async.ca.shared.global"](shared.ptr_to([0]), source.ptr_to([0]), 16)
        T.ptx.cp.async_.mbarrier.arrive.noinc.shared.b64(barrier.ptr_to([0]))
        {overwrite}
        {"T.ptx.cp.async_.commit_group()" if commit else "T.evaluate(0)"}
        {"T.ptx.cp.async_.commit_group()" if empty_tail else "T.evaluate(0)"}
        T.ptx.cp.async_.wait_group({pending})
        {"T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)" if observe_barrier else "T.evaluate(0)"}
        output[0] = shared[0]
""",
        {"T": T},
    )


def test_copy_arrival_is_not_an_explicit_commit(tmp_path):
    for options, clean in (
        ({}, False),
        ({"commit": True, "pending": 1}, False),
        ({"commit": True}, True),
        ({"commit": True, "empty_tail": True, "pending": 1}, True),
        ({"observe_barrier": True}, True),
        ({"commit": True, "overwrite_after_wait": True}, True),
    ):
        kernel = arrived_open_batch_case(**options)
        synccheck(kernel, inputs()).require_clean()
        report = racecheck(kernel, inputs())
        if clean:
            report.require_clean()
            result = numsim.Engine().run(numsim.transpile(kernel, cache_dir=tmp_path), inputs())
            np.testing.assert_array_equal(
                result.outputs["output"], [42 if options.get("overwrite_after_wait") else 7]
            )
        else:
            assert report.verdict == "error", report.format()
            assert any(f.details["access_pair"] in ("write_read", "read_write") for f in report.findings), (
                report.format()
            )


@pytest.mark.parametrize("history", ["waited", "committed-and-open"])
def test_copy_arrival_retains_earlier_copy_history(history, tmp_path):
    for relaxed in (False, True):
        for unrelated, reuse_source in ((False, False), (True, False), (False, True)):
            kernel = copy_arrival_case(
                committed=True,
                relaxed=relaxed,
                waited=history == "waited",
                extra=history != "waited",
                unrelated=unrelated,
                reuse_source=reuse_source,
            )
            synccheck(kernel, inputs()).require_clean()
            report = racecheck(kernel, inputs())
            if unrelated:
                assert report.verdict == "error", report.format()
                assert any(f.details["access_pair"] == "write_read" for f in report.findings), report.format()
            else:
                report.require_clean()
                result = numsim.Engine().run(numsim.transpile(kernel, cache_dir=tmp_path), inputs())
                np.testing.assert_array_equal(result.outputs["output"], [7])
