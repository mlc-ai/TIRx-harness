"""PTX tcgen05 thread-fence ordering, exercised through public Racecheck."""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import (
    ComposeLayout,
    R,
    S,
    TCol,
    TileLayout,
    TLane,
    tmem_datapath_layout,
    wg_local_layout,
)

from tirx_harness.numsim.checkers import _run_racecheck as racecheck


_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_REPLICATED_CP_LAYOUT = TileLayout(S[(32, 4) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])


@T.prim_func
def tcgen_cp_to_mma_handoff(
    cp_source: T.Buffer((32, 4), "float32"),
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    flag: T.Buffer((1,), "int32"),
    with_before: T.int32,
    with_after: T.int32,
    handoff: T.int32,
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    observed = T.local_scalar("int32")
    query_ready = T.local_scalar("uint32")
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barriers = T.alloc_buffer((3,), "uint64", scope="shared")
    shared_cp = T.alloc_buffer((32, 4), "float32", scope="shared")
    shared_a = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    shared_b = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    copied = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=_REPLICATED_CP_LAYOUT,
        allocated_addr=address[0],
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=address[0],
    )

    if (warp == 0) and (lane == 0):
        Tx.copy(shared_cp[:, :], cp_source[:, :])
        Tx.copy(shared_a[:, :], left[:, :])
        Tx.copy(shared_b[:, :], right[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[2]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        Tx.copy_async(
            copied[:, :],
            shared_cp[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
        T.ptx.tcgen05.fence__before_thread_sync(pred=with_before != 0)
        if handoff == 1:
            T.ptx.st.relaxed.cta.global_.s32(flag.ptr_to([0]), T.int32(1))
        elif handoff == 4:
            T.ptx.mbarrier.arrive.relaxed.cluster.shared.b64(T.address_of(barriers[2]))
        elif handoff >= 2:
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[2]))
    if handoff == 0:
        T.cuda.cta_sync()

    if (warp == 1) and (lane == 0):
        if handoff == 1:
            observed = T.int32(0)
            T.cuda.wait_until(
                observed, flag.ptr_to([0]), observed != T.int32(0), "cta", "global"
            )
        elif handoff == 2:
            T.cuda.mbarrier_wait(T.address_of(barriers[2]), 0)
        elif handoff >= 3:
            query_ready = T.uint32(0)
            while query_ready == T.uint32(0):
                T.ptx.mbarrier.test_wait.parity.relaxed.cta.shared.b64(
                    query_ready, T.address_of(barriers[2]), T.uint32(0)
                )
        T.ptx.tcgen05.fence__after_thread_sync(pred=with_after != 0)
        Tx.gemm_async(
            accumulator[:, :],
            shared_a[:, :],
            shared_b[:, :],
            accum=False,
        )
    T.cuda.cta_sync()

    if warp == 0:
        if lane == 0:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[0])
            )
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    if warp == 1:
        if lane == 0:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[1])
            )
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def repeated_tcgen_fence_handoff(
    cp_source: T.Buffer((32, 4), "float32"),
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    rounds: T.int32,
):
    """Repeated full-frontier publication must stay bounded and ordered."""

    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    shared_cp = T.alloc_buffer((32, 4), "float32", scope="shared")
    shared_a = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    shared_b = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    copied = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=_REPLICATED_CP_LAYOUT,
        allocated_addr=address[0],
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=address[0],
    )

    if (warp == 0) and (lane == 0):
        Tx.copy(shared_cp[:, :], cp_source[:, :])
        Tx.copy(shared_a[:, :], left[:, :])
        Tx.copy(shared_b[:, :], right[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        Tx.copy_async(
            copied[:, :],
            shared_cp[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
    for _ in T.serial(rounds):
        if (warp == 0) and (lane == 0):
            T.ptx.tcgen05.fence__before_thread_sync()
        T.cuda.cta_sync()
        if (warp == 1) and (lane == 0):
            T.ptx.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

    if (warp == 1) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            shared_a[:, :],
            shared_b[:, :],
            accum=False,
        )
    T.cuda.cta_sync()

    if warp == 0:
        if lane == 0:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[0])
            )
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    if warp == 1:
        if lane == 0:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[1])
            )
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def same_thread_nonpipelined_copies(
    first_source: T.Buffer((32, 4), "float32"),
    second_source: T.Buffer((32, 4), "float32"),
    ordering: T.int32,
):
    """Two overlapping CP operations, which are not an implicit PTX pair."""

    T.device_entry()
    warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    first_shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    second_shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    copied = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=_REPLICATED_CP_LAYOUT,
        allocated_addr=address[0],
    )

    if lane == 0:
        Tx.copy(first_shared[:, :], first_source[:, :])
        Tx.copy(second_shared[:, :], second_source[:, :])
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if lane == 0:
        Tx.copy_async(
            copied[:, :],
            first_shared[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
        if ordering == 1:
            T.ptx.tcgen05.fence__before_thread_sync()
        elif ordering == 2:
            T.ptx.tcgen05.fence__after_thread_sync()
        elif ordering == 3:
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[0])
            )
        elif (ordering == 4) or (ordering == 5):
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[0])
            )
    if (ordering == 4) or (ordering == 5):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        if ordering == 4:
            T.ptx.tcgen05.fence__after_thread_sync()
    if lane == 0:
        Tx.copy_async(
            copied[:, :],
            second_shared[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[1])
        )
    if ordering == 3:
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()

    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def cross_thread_nonpipelined_load_store(with_wait: T.int32):
    """LD and ST need producer completion in addition to thread fences."""

    T.device_entry()
    warpgroup = T.warpgroup_id([2])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 32),
        allocated_addr=0,
    )
    registers = T.alloc_local((32,), "float32")
    register_tile = registers.view(128, 32, layout=wg_local_layout(32))

    if warpgroup == 0:
        for col in T.serial(32):
            tmem[row, col] = T.cast(row * 32 + col, "float32")
    else:
        for col in T.serial(32):
            registers[col] = T.cast(row * 32 + col, "float32")
    T.cuda.cta_sync()

    if warpgroup == 0:
        Tx.wg.copy_async(register_tile[:, :], tmem[:, :])
        if with_wait != 0:
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    if warpgroup == 1:
        T.ptx.tcgen05.fence__after_thread_sync()
        Tx.wg.copy_async(tmem[:, :], register_tile[:, :])
        T.ptx.tcgen05.wait__st.sync.aligned()


@T.prim_func
def commit_forwards_only_issued_work(
    first_source: T.Buffer((32, 4), "float32"),
    second_source: T.Buffer((32, 4), "float32"),
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    with_local_work: T.int32,
):
    """Commit publishes local work and its causes, but not bare imported work."""

    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barriers = T.alloc_buffer((4,), "uint64", scope="shared")
    first_shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    second_shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    shared_a = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    shared_b = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    copied = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=_REPLICATED_CP_LAYOUT,
        allocated_addr=address[0],
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=address[0] + T.uint32(16),
    )

    if (warp == 0) and (lane == 0):
        Tx.copy(first_shared[:, :], first_source[:, :])
        Tx.copy(second_shared[:, :], second_source[:, :])
        Tx.copy(shared_a[:, :], left[:, :])
        Tx.copy(shared_b[:, :], right[:, :])
        for i in T.unroll(4):
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[i]), 1)
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        Tx.copy_async(
            copied[:, :],
            first_shared[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
        T.ptx.tcgen05.fence__before_thread_sync()
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[2])
        )

    if warp == 1:
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        if lane == 0:
            T.ptx.tcgen05.fence__after_thread_sync()
            if with_local_work != 0:
                Tx.gemm_async(
                    accumulator[:, :],
                    shared_a[:, :],
                    shared_b[:, :],
                    accum=False,
                )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[1])
            )

    if warp == 2:
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
        if lane == 0:
            T.ptx.tcgen05.fence__after_thread_sync()
            Tx.copy_async(
                copied[:, :],
                second_shared[:, :],
                dispatch="smem->tmem",
                shape="32x128b",
                multicast="warpx4",
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barriers[3])
            )

    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[2]), 0)
    if warp == 2:
        T.cuda.mbarrier_wait(T.address_of(barriers[3]), 0)
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-tcgen-thread-fence")


def _run(*, with_before: bool, with_after: bool, handoff: int, cache_dir):
    return racecheck(
        tcgen_cp_to_mma_handoff,
        inputs={
            "cp_source": np.zeros((32, 4), dtype=np.float32),
            "left": np.zeros((128, 16), dtype=np.float16),
            "right": np.zeros((8, 16), dtype=np.float16),
            "flag": np.zeros((1,), dtype=np.int32),
            "with_before": np.int32(with_before),
            "with_after": np.int32(with_after),
            "handoff": np.int32(handoff),
        },
        cache_dir=cache_dir,
        max_workers=1,
    )


@pytest.mark.parametrize(
    "handoff",
    [0, 1, 2, 3, 4],
    ids=["cta_barrier", "relaxed_flag", "mbarrier", "relaxed_wait", "relaxed_arrive_wait"],
)
def test_cross_thread_cp_to_mma_requires_both_thread_fences(
    handoff: int,
    native_cache_dir,
) -> None:
    complete = _run(
        with_before=True,
        with_after=True,
        handoff=handoff,
        cache_dir=native_cache_dir,
    )
    complete.require_clean()

    for missing in ("after", "before"):
        report = _run(
            with_before=missing != "before",
            with_after=missing != "after",
            handoff=handoff,
            cache_dir=native_cache_dir,
        )
        assert report.verdict != "clean", (missing, report.format())


@pytest.mark.parametrize("ordering", [4, 5], ids=["completion_and_after", "completion_only"])
def test_completion_orders_nonimplicit_same_thread_pair(
    ordering: int,
    native_cache_dir,
) -> None:
    report = racecheck(
        same_thread_nonpipelined_copies,
        inputs={
            "first_source": np.zeros((32, 4), dtype=np.float32),
            "second_source": np.ones((32, 4), dtype=np.float32),
            "ordering": np.int32(ordering),
        },
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    report.require_clean()


@pytest.mark.parametrize("ordering", [1, 3], ids=["before_fence", "implicit_commit_fence"])
def test_before_fence_orders_nonimplicit_same_thread_pair(
    ordering: int,
    native_cache_dir,
) -> None:
    report = racecheck(
        same_thread_nonpipelined_copies,
        inputs={
            "first_source": np.zeros((32, 4), dtype=np.float32),
            "second_source": np.ones((32, 4), dtype=np.float32),
            "ordering": np.int32(ordering),
        },
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    report.require_clean()


@pytest.mark.parametrize(
    "ordering",
    [0, 2],
    ids=[
        "none",
        "after_only",
    ],
)
def test_nonimplicit_same_thread_pair_without_before_fence_is_unordered(
    ordering: int,
    native_cache_dir,
) -> None:
    report = racecheck(
        same_thread_nonpipelined_copies,
        inputs={
            "first_source": np.zeros((32, 4), dtype=np.float32),
            "second_source": np.ones((32, 4), dtype=np.float32),
            "ordering": np.int32(ordering),
        },
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    assert report.verdict != "clean", report.format()


@pytest.mark.parametrize("with_wait", [False, True], ids=["no_wait", "wait_ld"])
def test_nonpipelined_cross_thread_ld_st_requires_producer_wait(
    with_wait: bool,
    native_cache_dir,
) -> None:
    report = racecheck(
        cross_thread_nonpipelined_load_store,
        inputs={"with_wait": np.int32(with_wait)},
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    if with_wait:
        report.require_clean()
    else:
        assert report.verdict != "clean", report.format()


def test_empty_commit_does_not_republish_tcgen_imported_from_another_thread(
    native_cache_dir,
) -> None:
    report = racecheck(
        commit_forwards_only_issued_work,
        inputs={
            "first_source": np.zeros((32, 4), dtype=np.float32),
            "second_source": np.ones((32, 4), dtype=np.float32),
            "left": np.zeros((128, 16), dtype=np.float16),
            "right": np.zeros((8, 16), dtype=np.float16),
            "with_local_work": np.int32(0),
        },
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    assert report.verdict != "clean", report.format()


def test_commit_republishes_causal_predecessors_of_local_work(native_cache_dir) -> None:
    report = racecheck(
        commit_forwards_only_issued_work,
        inputs={
            "first_source": np.zeros((32, 4), dtype=np.float32),
            "second_source": np.ones((32, 4), dtype=np.float32),
            "left": np.zeros((128, 16), dtype=np.float16),
            "right": np.zeros((8, 16), dtype=np.float16),
            "with_local_work": np.int32(1),
        },
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    report.require_clean()


def test_repeated_thread_fence_handoff_keeps_full_frontier_ordered(native_cache_dir) -> None:
    report = racecheck(
        repeated_tcgen_fence_handoff,
        inputs={
            "cp_source": np.zeros((32, 4), dtype=np.float32),
            "left": np.zeros((128, 16), dtype=np.float16),
            "right": np.zeros((8, 16), dtype=np.float16),
            "rounds": np.int32(128),
        },
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    report.require_clean()
