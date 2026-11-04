"""End-to-end native Racecheck coverage for generic/async proxy ordering."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tvm_tirx.script import tirx as T


def _mapa_u64(ptr, rank):
    mapped = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


@T.inline
def _query_cancel_first_ctaid_x_without_reuse_fence(first_ctaid_x, handle):
    response = T.local_scalar("uint128")
    canceled = T.local_scalar("uint32")

    T.ptx["ld.acquire.cta.shared.b128"](response, handle)
    T.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, response)
    T.ptx.mov.b32(first_ctaid_x, T.uint32(0xFFFFFFFF))
    T.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
        first_ctaid_x, response, pred=canceled
    )


@T.function
def proxy_async_shared_cta_g2a(
    fence_mode: T.int32,
    destination: T.Tensor((16,), "uint8"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((16,), "uint8")
    if lane == 0:
        if fence_mode == 4:
            T.ptx.fence.proxy.async_.shared__cta()
        for index in T.serial(16):
            shared[index] = T.cast(index, "uint8")
        if fence_mode == 1:
            T.ptx.fence.proxy.async_.shared__cta()
        elif fence_mode == 2:
            T.ptx.fence.proxy.async_.shared__cluster()
        elif fence_mode == 3:
            T.ptx.fence.proxy.async_()
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
            destination.ptr_to([0]), shared.ptr_to([0]), T.cast(16, "uint32")
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
        destination[0] = destination[0]


@T.function
def proxy_async_fence_active_lane(
    destination: T.Tensor((16,), "uint8"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((16,), "uint8")
    barrier = T.alloc_shared((1,), "uint64")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0:
        if lane == 0:
            for index in T.serial(16):
                shared[index] = T.cast(index, "uint8")
            T.ptx.mbarrier.arrive.shared.b64(barrier.ptr_to([0]))
    elif warp == 1:
        if lane == 1:
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        if lane == 0:
            T.ptx.fence.proxy.async_.shared__cta()
        if lane == 1:
            T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
                destination.ptr_to([0]), shared.ptr_to([0]), T.cast(16, "uint32")
            )
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)


@T.function
def clc_response_reuse_proxy_ordering(
    fence_before_reuse: T.int32,
    output: T.Tensor((2,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, cluster=1))
    lane = T.cuda.lane_id()
    response = T.alloc_shared((4,), "uint32", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=8)
    first_ctaid_x = T.local_scalar("uint32")

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx[
            "clusterlaunchcontrol.try_cancel.async.shared::cta"
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
        ](response.ptr_to([0]), barrier.ptr_to([0]))
        T.cuda.mbarrier_wait_acquire_cluster(barrier.ptr_to([0]), 0)
        _query_cancel_first_ctaid_x_without_reuse_fence(first_ctaid_x, response.ptr_to([0]))
        output[0] = first_ctaid_x

        if fence_before_reuse != 0:
            T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx[
            "clusterlaunchcontrol.try_cancel.async.shared::cta"
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
        ](response.ptr_to([0]), barrier.ptr_to([0]))
        T.cuda.mbarrier_wait_acquire_cluster(barrier.ptr_to([0]), 1)
        _query_cancel_first_ctaid_x_without_reuse_fence(first_ctaid_x, response.ptr_to([0]))
        output[1] = first_ctaid_x


@pytest.fixture(scope="module")
def proxy_async_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-proxy-async")


def _assert_single_finding(
    report,
    *,
    kind: str,
    byte_len: int,
):
    native = report.native_payload
    assert native["incomplete"] == []
    assert len(native["findings"]) == 1
    finding = native["findings"][0]
    assert finding["kind"] == "data_race"
    assert finding["access_pair"] == kind
    assert finding["overlap"]["byte_len"] == byte_len
    return finding


@pytest.mark.parametrize(
    ("fence_mode", "expected"),
    [(0, "error"), (1, "clean"), (2, "error"), (3, "clean"), (4, "error")],
)
def test_shared_cta_proxy_fence_qualifier_is_exact(proxy_async_cache, fence_mode, expected):
    report = racecheck(
        proxy_async_shared_cta_g2a,
        inputs={
            "fence_mode": np.int32(fence_mode),
            "destination": np.zeros(16, dtype=np.uint8),
        },
        cache_dir=proxy_async_cache,
    )

    assert report.verdict == expected
    assert report.native_payload["incomplete"] == []
    if expected == "error":
        _assert_single_finding(
            report,
            kind="write_read",
            byte_len=1,
        )
    else:
        assert report.native_payload["findings"] == []


@pytest.mark.parametrize("max_workers", [1, 2, 4, 32])
def test_proxy_fence_does_not_escape_its_active_lane(proxy_async_cache, max_workers):
    report = racecheck(
        proxy_async_fence_active_lane,
        inputs={"destination": np.zeros(16, dtype=np.uint8)},
        cache_dir=proxy_async_cache,
        max_workers=max_workers,
    )

    assert report.verdict == "error", report.native_payload
    finding = _assert_single_finding(
        report,
        kind="write_read",
        byte_len=1,
    )
    assert finding["prior"]["operation"]["global_warp_id"] == 0
    assert finding["prior"]["lane"] == 0
    assert finding["current"]["operation"]["global_warp_id"] == 1
    assert finding["current"]["lane"] == 1


@pytest.mark.parametrize(
    ("fence_before_reuse", "expected"),
    [(False, "error"), (True, "clean")],
    ids=["missing-ld-to-clc-fence", "fenced-ld-to-clc-reuse"],
)
def test_clc_completion_orders_query_but_query_requires_a_fence_before_reuse(
    proxy_async_cache, fence_before_reuse, expected
):
    report = racecheck(
        clc_response_reuse_proxy_ordering,
        inputs={
            "fence_before_reuse": np.int32(fence_before_reuse),
            "output": np.zeros(2, dtype=np.uint32),
        },
        cache_dir=proxy_async_cache,
        max_workers=2,
    )

    assert report.verdict == expected, report.native_payload
    assert report.native_payload["incomplete"] == []
    if expected == "clean":
        assert report.native_payload["findings"] == []
    else:
        _assert_single_finding(report, kind="read_write", byte_len=8)
