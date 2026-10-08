"""End-to-end native Racecheck coverage for generic/async proxy ordering."""

from __future__ import annotations

import numpy as np
import pytest
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

from tirx_harness.numsim.checkers import _run_racecheck as racecheck


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
    first_ctaid_x = T.uint32(0xFFFFFFFF)
    T.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
        first_ctaid_x, response, pred=canceled
    )


@T.prim_func
def proxy_async_global_g2a(
    fence_mode: T.int32,
    source: T.Buffer((4,), "float32"),
    sink: T.Buffer((1,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        source[0] = T.float32(7)
        if fence_mode == 1:
            T.ptx.fence.proxy.async_.global_()
        elif fence_mode == 2:
            T.ptx.fence.proxy.async_.shared__cta()
        elif fence_mode == 3:
            T.ptx.fence.proxy.async_()
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=barrier.ptr_to([0]),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        sink[0] = shared[0]


@T.prim_func
def proxy_async_shared_cta_g2a(
    fence_mode: T.int32,
    destination: T.Buffer((16,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
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


@T.prim_func
def proxy_async_shared_cluster_g2a(
    fence_mode: T.int32,
    source: T.Buffer((4,), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    remote_ptr: T.let[
        T.Var(
            name="proxy_async_remote_shared",
            ty=PointerType(PrimType("float32"), "shared"),
        )
    ] = T.reinterpret(
        PointerType(PrimType("float32"), "shared"),
        _mapa_u64(shared.ptr_to([0]), 0),
    )
    remote = T.decl_buffer((4,), "float32", scope="shared", data=remote_ptr)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        remote[0] = T.float32(9)
        if fence_mode == 1:
            T.ptx.fence.proxy.async_.shared__cluster()
        elif fence_mode == 2:
            T.ptx.fence.proxy.async_.shared__cta()
        elif fence_mode == 3:
            T.ptx.fence.proxy.async_()
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=barrier.ptr_to([0]),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)


@T.prim_func
def proxy_async_fence_active_lane(
    destination: T.Buffer((16,), "uint8"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
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


@T.prim_func
def proxy_async_completion_acquire_lane(
    wait_lane: T.int32,
    source: T.Buffer((4,), "float32"),
    output: T.Buffer((1,), "float32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0:
        if lane == 0:
            Tx.copy_async(
                shared[:],
                source[:],
                dispatch="tma_auto",
                mbar=barrier.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
    elif warp == 1:
        if lane == wait_lane:
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        if lane == 1:
            output[0] = shared[0]


@T.prim_func
def proxy_async_raw_try_wait_acquire(
    acquire: T.int32,
    read_ordinary: T.int32,
    source: T.Buffer((4,), "float32"),
    ordinary: T.Buffer((1,), "float32"),
    output: T.Buffer((1,), "float32"),
):
    """Copy completion is observable independently of ordinary release history."""

    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        ordinary[0] = T.float32(42)
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=barrier.ptr_to([0]),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
    elif (warp == 1) and (lane == 0):
        ready: T.uint32
        ready = T.uint32(0)
        while ready == T.uint32(0):
            if acquire == 1:
                T.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
                    ready, barrier.ptr_to([0]), T.uint32(0), T.uint32(0)
                )
            elif acquire == 0:
                T.ptx.mbarrier.try_wait.parity.relaxed.cta.shared__cta.b64(
                    ready, barrier.ptr_to([0]), T.uint32(0), T.uint32(0)
                )
            elif acquire == 2:
                T.ptx[
                    "mbarrier.try_wait.parity.phase_type::conditional.relaxed.cta.shared::cta.b64"
                ](ready, barrier.ptr_to([0]), T.uint32(0), T.uint32(0))
            else:
                T.ptx[
                    "mbarrier.try_wait.parity.phase_type::conditional.acquire.cta.shared::cta.b64"
                ](ready, barrier.ptr_to([0]), T.uint32(0), T.uint32(0))
        if read_ordinary != 0:
            output[0] = ordinary[0]
        else:
            output[0] = shared[0]


@T.prim_func
def clc_response_reuse_proxy_ordering(
    fence_before_reuse: T.int32,
    output: T.Buffer((2,), "uint32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    response = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
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


@T.prim_func
def proxy_async_v30_qv_isolation(
    reuse_kind: T.int32,
    source: T.Buffer((4,), "float32"),
    sink: T.Buffer((1,), "float32"),
):
    """Isolate the v30_2 h_smem anti-dependency for its Q and V aliases."""

    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    pool = T.SMEMPool()
    alias_base = T.meta_var(pool.offset)
    h_smem = pool.alloc((4,), "float32")
    pool.move_base_to(alias_base)
    q_smem = pool.alloc((4,), "float32")
    pool.move_base_to(alias_base)
    v_smem = pool.alloc((4,), "float32")
    pool.commit()
    init_done = T.alloc_buffer((1,), "uint64", scope="shared")
    read_done = T.alloc_buffer((1,), "uint64", scope="shared")
    tma_done = T.alloc_buffer((1,), "uint64", scope="shared")

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(init_done.ptr_to([0]), 1)
        T.ptx.mbarrier.init.shared.b64(read_done.ptr_to([0]), 1)
        T.ptx.mbarrier.init.shared.b64(tma_done.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        h_smem[0] = T.float32(1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.mbarrier.arrive.shared.b64(init_done.ptr_to([0]))
        T.cuda.mbarrier_wait(read_done.ptr_to([0]), 0)
        if reuse_kind == 0:
            Tx.copy_async(
                q_smem[:],
                source[:],
                dispatch="tma_auto",
                mbar=tma_done.ptr_to([0]),
            )
        else:
            Tx.copy_async(
                v_smem[:],
                source[:],
                dispatch="tma_auto",
                mbar=tma_done.ptr_to([0]),
            )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(tma_done.ptr_to([0]), 16)
        T.cuda.mbarrier_wait(tma_done.ptr_to([0]), 0)
    elif (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(init_done.ptr_to([0]), 0)
        sink[0] = h_smem[0]
        T.ptx.mbarrier.arrive.shared.b64(read_done.ptr_to([0]))


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
    [(0, "error"), (1, "clean"), (2, "error"), (3, "clean")],
)
def test_global_proxy_fence_qualifier_is_exact(proxy_async_cache, fence_mode, expected):
    report = racecheck(
        proxy_async_global_g2a,
        inputs={
            "fence_mode": np.int32(fence_mode),
            "source": np.zeros(4, dtype=np.float32),
            "sink": np.zeros(1, dtype=np.float32),
        },
        cache_dir=proxy_async_cache,
    )

    assert report.verdict == expected
    assert report.native_payload["incomplete"] == []
    if expected == "error":
        _assert_single_finding(
            report,
            kind="write_read",
            byte_len=4,
        )
    else:
        assert report.native_payload["findings"] == []


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


@pytest.mark.parametrize(
    ("fence_mode", "expected"),
    [(0, "error"), (1, "error"), (2, "clean"), (3, "clean")],
)
def test_same_rank_mapa_address_uses_its_final_cta_local_bits(
    proxy_async_cache, fence_mode, expected
):
    report = racecheck(
        proxy_async_shared_cluster_g2a,
        inputs={
            "fence_mode": np.int32(fence_mode),
            "source": np.zeros(4, dtype=np.float32),
        },
        cache_dir=proxy_async_cache,
    )

    assert report.verdict == expected
    assert report.native_payload["incomplete"] == []
    if expected == "error":
        _assert_single_finding(
            report,
            kind="write_write",
            byte_len=4,
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


@pytest.mark.parametrize("max_workers", [1, 2, 4, 32])
def test_mbarrier_completion_acquire_does_not_escape_its_active_lane(
    proxy_async_cache, max_workers
):
    inputs = {
        "source": np.arange(4, dtype=np.float32),
        "output": np.zeros(1, dtype=np.float32),
    }
    wrong_lane = racecheck(
        proxy_async_completion_acquire_lane,
        inputs={"wait_lane": np.int32(0), **inputs},
        cache_dir=proxy_async_cache,
        max_workers=max_workers,
    )
    acquiring_lane = racecheck(
        proxy_async_completion_acquire_lane,
        inputs={"wait_lane": np.int32(1), **inputs},
        cache_dir=proxy_async_cache,
        max_workers=max_workers,
    )

    assert wrong_lane.verdict == "error"
    finding = _assert_single_finding(
        wrong_lane,
        kind="write_read",
        byte_len=4,
    )
    assert finding["prior"]["operation"]["global_warp_id"] == 0
    assert finding["prior"]["lane"] == 0
    assert finding["current"]["operation"]["global_warp_id"] == 1
    assert finding["current"]["lane"] == 1

    acquiring_lane.require_clean()
    assert acquiring_lane.native_payload["incomplete"] == []
    assert acquiring_lane.native_payload["findings"] == []


@pytest.mark.parametrize(
    "acquire",
    [1, 0, 2, 3],
    ids=["acquire-on-true", "explicit-relaxed", "conditional-relaxed", "conditional-acquire"],
)
def test_raw_try_wait_observes_copy_but_only_acquire_orders_ordinary_memory(
    proxy_async_cache, acquire
):
    # PTX async-proxy completion visibility applies even to relaxed queries.
    # Only an acquiring query also orders the producer's unrelated ordinary store.
    for read_ordinary in (False, True):
        report = racecheck(
            proxy_async_raw_try_wait_acquire,
            inputs={
                "acquire": np.int32(acquire),
                "read_ordinary": np.int32(read_ordinary),
                "source": np.arange(4, dtype=np.float32),
                "ordinary": np.zeros(1, dtype=np.float32),
                "output": np.zeros(1, dtype=np.float32),
            },
            cache_dir=proxy_async_cache,
            max_workers=2,
        )
        expected = "error" if read_ordinary and acquire in (0, 2) else "clean"
        assert report.verdict == expected, report.native_payload
        assert report.native_payload["incomplete"] == []
        if expected == "clean":
            assert report.native_payload["findings"] == []
        else:
            _assert_single_finding(report, kind="write_read", byte_len=4)


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


@pytest.mark.parametrize("reuse_kind", [0, 1], ids=["q", "v"])
def test_v30_q_and_v_aliases_have_isolated_dynamic_witnesses(proxy_async_cache, reuse_kind):
    report = racecheck(
        proxy_async_v30_qv_isolation,
        inputs={
            "reuse_kind": np.int32(reuse_kind),
            "source": np.arange(4, dtype=np.float32),
            "sink": np.zeros(1, dtype=np.float32),
        },
        cache_dir=proxy_async_cache,
        max_workers=4,
    )

    assert report.verdict == "error", report.native_payload
    finding = _assert_single_finding(
        report,
        kind="read_write",
        byte_len=4,
    )
    assert finding["prior"]["operation"]["global_warp_id"] == 1
    assert finding["current"]["operation"]["global_warp_id"] == 0
