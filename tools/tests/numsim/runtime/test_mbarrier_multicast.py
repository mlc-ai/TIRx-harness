"""32-bit masks route the same barrier transition to each selected CTA."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import racecheck, synccheck

MULTICAST_FORMS = (
    "arrive",
    "arrive_nocount",
    "arrive_drop",
    "arrive_drop_nocount",
    "arrive_expect_tx",
    "arrive_drop_expect_tx",
    "expect_tx",
    "complete_tx",
)


def multicast_barrier_kernel(form, *, mask=None, ctas=20, lane_varying=False):
    arrival = form.startswith("arrive")
    drop = "drop" in form
    transactions = "tx" in form
    initial = 2 if arrival else 1
    action = form.replace("_nocount", "").replace("_expect_tx", ".expect_tx")
    instruction = f"mbarrier.{action}.shared::cluster.multicast::cluster::32b.b64"
    args = ["barrier.ptr_to([0])"]
    if not form.endswith("_nocount"):
        args.append(f"T.uint32({16 if transactions else 1})")
    args.append(
        f"T.uint32(1) << (lane * {ctas - 1})"
        if lane_varying
        else f"T.uint32({(1 << (ctas - 1)) | 1 if mask is None else mask})"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def multicast_barrier(out: T.Buffer(({ctas}, 2), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([{ctas}])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    state = T.alloc_local((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), {initial})
        {"T.ptx.mbarrier.expect_tx.shared.b64(barrier.ptr_to([0]), 16)" if form == "complete_tx" else "T.evaluate(0)"}
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if cta == 0 and lane < {2 if lane_varying else 1}:
        T.ptx["{instruction}"]({", ".join(args)})
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0 or cta == {ctas - 1}:
            {"T.ptx.mbarrier.complete_tx.relaxed.cta.shared.b64(barrier.ptr_to([0]), 16)" if transactions and form != "complete_tx" else "T.evaluate(0)"}
            T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]), T.uint32(1))
        else:
            {"T.ptx.mbarrier.complete_tx.relaxed.cta.shared.b64(barrier.ptr_to([0]), 16)" if form == "complete_tx" else "T.evaluate(0)"}
            T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]), T.uint32({initial}))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), state[0])
        out[cta, 0] = ready[0]
        T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]),
            T.uint32({1 if drop else initial}) if cta == 0 or cta == {ctas - 1} else T.uint32({initial}))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 1)
        T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), state[0])
        out[cta, 1] = ready[0]
    T.cuda.cluster_sync()
""",
        {"T": T},
    )


@pytest.mark.parametrize("form", MULTICAST_FORMS)
def test_mbarrier_multicast32(form, tmp_path):
    kernel = multicast_barrier_kernel(form)
    inputs = {"out": np.zeros((20, 2), np.uint32)}
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], np.ones((20, 2), np.uint32))


def test_mbarrier_multicast_lane_masks(tmp_path):
    kernel = multicast_barrier_kernel("arrive_drop", lane_varying=True)
    inputs = {"out": np.zeros((20, 2), np.uint32)}
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], 1)


def test_mbarrier_multicast_outside_cluster():
    kernel = multicast_barrier_kernel("arrive", mask=1 << 20)
    for checker in (synccheck, racecheck):
        report = checker(kernel, {"out": np.zeros((20, 2), np.uint32)})
        assert report.verdict == "error", report.format()
        assert "outside the cluster" in report.format()
