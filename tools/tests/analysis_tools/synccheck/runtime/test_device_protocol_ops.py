"""Per-operation runtime evidence for the transpile-backed native Synccheck path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.numsim.runtime.test_sync_runtime_domain_oracle import (
    clc_no_work_cluster_acquire_wait,
)
from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


@T.prim_func
def collective_protocol_ops():
    T.device_entry()
    _warp = T.warp_id([4])
    _lane = T.lane_id([32])

    T.cuda.warp_sync()
    T.cuda.warpgroup_sync(7)
    T.cuda.cta_sync()
    T.cuda.cluster_sync()
    T.cuda.grid_sync()


@T.prim_func
def named_barrier_protocol_ops():
    T.device_entry()
    warp = T.warp_id([2])
    _lane = T.lane_id([32])

    if warp == 0:
        T.ptx.bar.arrive(T.uint32(3), T.uint32(64))
    else:
        T.ptx.bar.sync(T.uint32(3), T.uint32(64))
    if warp == 0:
        T.ptx.bar.arrive(T.uint32(4), T.uint32(64))
    else:
        T.ptx.barrier.sync(T.uint32(4), T.uint32(64))
    T.ptx.bar.sync(T.uint32(5))
    T.ptx.barrier.sync(T.uint32(6))
    if warp == 0:
        T.ptx.barrier.arrive(T.uint32(7), T.uint32(64))
    else:
        T.ptx.barrier.sync(T.uint32(7), T.uint32(64))
    T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))


@T.prim_func
def divergent_unaligned_named_barrier_op():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    if lane == 0:
        T.ptx.barrier.sync(T.uint32(5), T.uint32(32))
    else:
        T.ptx.barrier.sync(T.uint32(5), T.uint32(32))


@T.prim_func
def split_cluster_barrier_protocol_ops():
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])

    T.ptx.barrier.cluster.arrive()
    T.ptx.barrier.cluster.wait()


@T.prim_func
def mbarrier_protocol_ops():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((6,), "uint64", scope="shared", align=8)
    state = T.local_scalar("uint64")

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[2]), 2)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[3]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[4]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[5]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[4]), T.uint32(1))
        T.cuda.mbarrier_wait(T.address_of(barriers[4]), 0)
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[1]), 0)
        T.cuda.mbarrier_wait_acquire_cluster(T.address_of(barriers[1]), 0)
        T.ptx.mbarrier.arrive.noComplete.shared.b64(state, T.address_of(barriers[2]), T.uint32(1))
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[2]))
        T.cuda.mbarrier_wait(T.address_of(barriers[2]), 0)
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[3]), 16)
        T.ptx.mbarrier.complete_tx.relaxed.cta.shared__cta.b64(
            T.address_of(barriers[3]),
            T.uint32(16),
            pred=T.uint32(1),
        )
        T.cuda.mbarrier_wait(T.address_of(barriers[3]), 0)
        T.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(
            T.address_of(barriers[5]), T.uint32(16)
        )
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[5]))
        T.ptx.mbarrier.complete_tx.relaxed.cta.shared__cta.b64(
            T.address_of(barriers[5]), T.uint32(16)
        )
        T.cuda.mbarrier_wait(T.address_of(barriers[5]), 0)


@T.prim_func
def clc_protocol_op():
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    response = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.ptx[
            "clusterlaunchcontrol.try_cancel.async.shared::cta"
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
        ](T.address_of(response[0]), T.address_of(barrier[0]))
        T.cuda.mbarrier_wait_acquire_cluster(T.address_of(barrier[0]), 0)


@T.prim_func
def ordering_protocol_ops():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])

    T.cuda.thread_fence()
    T.cuda.nano_sleep(0)
    T.cuda.printf("native synccheck protocol runtime %d", 7)
    T.ptx.fence.sc.cta()
    T.ptx.griddepcontrol.launch_dependents()
    T.ptx.griddepcontrol.wait()
    T.cuda.warp_sync()


@T.prim_func
def setmaxnreg_protocol_op():
    T.device_entry()
    _wg = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])

    T.ptx.setmaxnreg.dec.sync.aligned.u32(88)


@T.prim_func
def tcgen_lifecycle_protocol_ops():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared", align=4)

    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.warp_sync()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tcgen_commit_protocol_op():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)


@T.prim_func
def tcgen_ordering_protocol_ops(output: T.Buffer((4, 32), "uint32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    value = T.alloc_local((1,), "uint32")

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.cta_sync()

    value[0] = T.cast(1000 + warp * 32 + lane, "uint32")
    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0], value[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    value[0] = T.uint32(0)
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[warp, lane] = value[0]
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def cp_async_group_protocol_ops(
    source: T.Buffer((128,), "uint8"), output: T.Buffer((128,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")

    T.ptx["cp.async.ca.shared.global.L2::64B"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        4,
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


@T.prim_func
def cp_async_predicated_group_protocol_ops(source: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")

    T.ptx["cp.async.ca.shared.global"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        4,
        pred=lane < 16,
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)


@T.prim_func
def cp_async_zero_fill_group_protocol_ops(source: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")

    T.ptx["cp.async.ca.shared.global"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        4,
        T.cast(T.if_then_else(lane < 16, 4, 0), "uint32"),
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)


@T.prim_func
def ldgsts_cp_async_group_protocol_ops(
    source: T.Buffer((128,), "uint8"), output: T.Buffer((128,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")

    Tx.copy_async(shared[:], source[:], dispatch="ldgsts")
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


@T.prim_func
def cp_async_uncommitted_protocol_op(source: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")

    T.ptx["cp.async.ca.shared.global"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        4,
    )


@T.prim_func
def bulk_async_group_protocol_ops(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")

    if lane < 16:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
            output.ptr_to([0]), shared.ptr_to([0]), T.cast(16, "uint32")
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
    T.cuda.warp_sync()


@T.prim_func
def pure_warp_sync_source_ops(output: T.Buffer((32, 10), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    full: T.let = T.uint32(0xFFFFFFFF)
    value: T.let = T.cast(lane + 1, "uint32")

    output[lane, 0] = T.cuda.__shfl_sync(full, value, T.cast(31 - lane, "uint32"), 32)
    output[lane, 1] = T.cuda.__shfl_up_sync(full, value, 1, 32)
    output[lane, 2] = T.cuda.__shfl_down_sync(full, value, 1, 32)
    output[lane, 3] = T.cuda.__shfl_xor_sync(full, value, 1, 32)
    output[lane, 4] = T.cuda.ballot_sync(full, lane < 16)
    output[lane, 5] = T.cuda.reduce_add_sync_u32(full, value)
    output[lane, 6] = T.cuda.reduce_min_sync_u32(full, value)
    output[lane, 7] = T.cast(T.cuda.any_sync(full, lane == 31), "uint32")
    output[lane, 8] = T.cuda.warp_sum(value, width=8)
    output[lane, 9] = T.cuda.elect_sync()


@T.prim_func
def pure_cta_sync_source_ops(output: T.Buffer((2, 32, 3), "int64")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    scratch = T.alloc_buffer((2,), "int32", scope="shared")
    value: T.let = warp * 32 + lane + 1

    output[warp, lane, 0] = T.cast(T.cuda.cta_sum(value, 2, scratch.ptr_to([0])), "int64")
    output[warp, lane, 1] = T.cuda.syncthreads_and(value > 0)
    output[warp, lane, 2] = T.cuda.syncthreads_or(lane == 31)


@T.prim_func
def mbarrier_query_sync_source_ops(output: T.Buffer((32, 4), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)

    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    T.ptx.mbarrier.test_wait.parity.shared.b64(
        output[lane, 0], T.address_of(barrier[0]), T.uint32(1)
    )
    T.ptx.mbarrier.try_wait.parity.shared.b64(
        output[lane, 1], T.address_of(barrier[0]), T.uint32(0), T.uint32(1)
    )
    T.ptx.mbarrier.try_wait.parity.shared.b64(
        output[lane, 2], T.address_of(barrier[0]), T.uint32(0)
    )
    if lane == 0:
        token = T.alloc_buffer((1,), "uint64", scope="local")
        barrier_address: T.let = T.cuda.cvta_generic_to_shared(T.address_of(barrier[0]))
        T.ptx.mbarrier.arrive.shared__cta.b64(token[0], barrier_address, T.uint32(1))
        T.ptx.mbarrier.try_wait.shared__cta.b64(output[lane, 3], barrier_address, token[0])


_KERNELS = (
    collective_protocol_ops,
    named_barrier_protocol_ops,
    divergent_unaligned_named_barrier_op,
    split_cluster_barrier_protocol_ops,
    mbarrier_protocol_ops,
    clc_protocol_op,
    clc_no_work_cluster_acquire_wait,
    ordering_protocol_ops,
    setmaxnreg_protocol_op,
    tcgen_lifecycle_protocol_ops,
    tcgen_commit_protocol_op,
    tcgen_ordering_protocol_ops,
    cp_async_group_protocol_ops,
    cp_async_predicated_group_protocol_ops,
    cp_async_zero_fill_group_protocol_ops,
    ldgsts_cp_async_group_protocol_ops,
    cp_async_uncommitted_protocol_op,
    bulk_async_group_protocol_ops,
    pure_warp_sync_source_ops,
    pure_cta_sync_source_ops,
    mbarrier_query_sync_source_ops,
)

_KERNEL_BY_OP = {
    "tirx.cuda.__shfl_down_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.__shfl_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.__shfl_up_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.__shfl_xor_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.ballot_sync": "pure_warp_sync_source_ops",
    "tirx.cuda.cluster_sync": "collective_protocol_ops",
    "tirx.cuda.cta_reduce": "pure_cta_sync_source_ops",
    "tirx.cuda.cta_sync": "collective_protocol_ops",
    "tirx.cuda.grid_sync": "collective_protocol_ops",
    "tirx.cuda.nano_sleep": "ordering_protocol_ops",
    "tirx.cuda.printf": "ordering_protocol_ops",
    "tirx.cuda.reduce_add_sync_u32": "pure_warp_sync_source_ops",
    "tirx.cuda.reduce_min_sync_u32": "pure_warp_sync_source_ops",
    "tirx.cuda.syncthreads_and": "pure_cta_sync_source_ops",
    "tirx.cuda.syncthreads_or": "pure_cta_sync_source_ops",
    "tirx.cuda.thread_fence": "ordering_protocol_ops",
    "tirx.cuda.warp_reduce": "pure_warp_sync_source_ops",
    "tirx.cuda.warp_sync": "collective_protocol_ops",
    "tirx.cuda.warpgroup_sync": "collective_protocol_ops",
    "tirx.ptx.bar_arrive": "named_barrier_protocol_ops",
    "tirx.ptx.bar_sync": "named_barrier_protocol_ops",
    "tirx.ptx.bar_sync_count": "named_barrier_protocol_ops",
    "tirx.ptx.bar_warp_sync": "named_barrier_protocol_ops",
    "tirx.ptx.barrier_arrive": "named_barrier_protocol_ops",
    "tirx.ptx.barrier_sync": "named_barrier_protocol_ops",
    "tirx.ptx.barrier_sync_count": "named_barrier_protocol_ops",
    "tirx.ptx.barrier_cluster_arrive": "split_cluster_barrier_protocol_ops",
    "tirx.ptx.barrier_cluster_wait": "split_cluster_barrier_protocol_ops",
    "tirx.cuda.any_sync": "pure_warp_sync_source_ops",
    "tirx.ptx.clusterlaunchcontrol_query_cancel_get_first_ctaid": (
        "clc_no_work_cluster_acquire_wait"
    ),
    "tirx.ptx.clusterlaunchcontrol_query_cancel_is_canceled": ("clc_no_work_cluster_acquire_wait"),
    "tirx.ptx.clusterlaunchcontrol_try_cancel": "clc_protocol_op",
    "tirx.ptx.cp_async_bulk_commit_group": "bulk_async_group_protocol_ops",
    "tirx.ptx.cp_async_bulk_wait_group": "bulk_async_group_protocol_ops",
    "tirx.ptx.cp_async_ca": "cp_async_group_protocol_ops",
    "tirx.ptx.cp_async_ca_src_size": "cp_async_zero_fill_group_protocol_ops",
    "tirx.ptx.cp_async_commit_group": "cp_async_group_protocol_ops",
    "tirx.ptx.cp_async_wait_group": "cp_async_group_protocol_ops",
    "tirx.cuda.elect_sync": "pure_warp_sync_source_ops",
    "tirx.ptx.fence": "ordering_protocol_ops",
    "tirx.ptx.fence_mbarrier_init": "mbarrier_protocol_ops",
    "tirx.ptx.fence_proxy": "mbarrier_protocol_ops",
    "tirx.ptx.griddepcontrol": "ordering_protocol_ops",
    "tirx.ptx.mbarrier_arrive": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_arrive_count_state": "mbarrier_query_sync_source_ops",
    "tirx.ptx.mbarrier_arrive_nocount": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_arrive_no_complete": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_arrive_expect_tx": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_expect_tx": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_complete_tx": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_init": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_test_wait_parity": "mbarrier_query_sync_source_ops",
    "tirx.cuda.mbarrier_wait": "mbarrier_protocol_ops",
    "tirx.cuda.mbarrier_wait_acquire_cluster": "mbarrier_protocol_ops",
    "tirx.ptx.mbarrier_try_wait_parity": "mbarrier_query_sync_source_ops",
    "tirx.ptx.mbarrier_try_wait_parity_no_hint": "mbarrier_query_sync_source_ops",
    "tirx.ptx.mbarrier_try_wait": "mbarrier_query_sync_source_ops",
    "tirx.ptx.setmaxnreg": "setmaxnreg_protocol_op",
    "tirx.ptx.tcgen05_alloc": "tcgen_lifecycle_protocol_ops",
    "tirx.ptx.tcgen05_commit": "tcgen_commit_protocol_op",
    "tirx.ptx.tcgen05_dealloc": "tcgen_lifecycle_protocol_ops",
    "tirx.ptx.tcgen05_fence": "tcgen_ordering_protocol_ops",
    "tirx.ptx.tcgen05_ld": "tcgen_ordering_protocol_ops",
    "tirx.ptx.tcgen05_relinquish_alloc_permit": "tcgen_lifecycle_protocol_ops",
    "tirx.ptx.tcgen05_wait": "tcgen_ordering_protocol_ops",
    "tirx.ptx.tcgen05_st": "tcgen_ordering_protocol_ops",
}

_EFFECTS_BY_OP = {
    "tirx.cuda.cluster_sync": {
        "barrier.cluster.arrive",
        "barrier.cluster.wait.register",
        "barrier.cluster.wait.resume",
    },
    "tirx.cuda.cta_sync": {"bar.sync.register", "bar.sync.resume"},
    "tirx.cuda.cta_reduce": {"bar.sync.register", "bar.sync.resume"},
    "tirx.cuda.warpgroup_sync": {"bar.sync.register", "bar.sync.resume"},
    "tirx.ptx.bar_arrive": {"bar.arrive.register"},
    "tirx.ptx.bar_sync": {"bar.sync.register", "bar.sync.resume"},
    "tirx.ptx.barrier_sync": {"bar.sync.register", "bar.sync.resume"},
    "tirx.ptx.barrier_cluster_arrive": {"barrier.cluster.arrive"},
    "tirx.ptx.barrier_cluster_wait": {
        "barrier.cluster.wait.register",
        "barrier.cluster.wait.resume",
    },
    "tirx.ptx.clusterlaunchcontrol_try_cancel": {"mbarrier.completion_issue"},
    "tirx.ptx.mbarrier_arrive": {"mbarrier.arrive"},
    "tirx.ptx.mbarrier_arrive_count_state": {"mbarrier.arrive"},
    "tirx.ptx.mbarrier_arrive_nocount": {"mbarrier.arrive"},
    "tirx.ptx.mbarrier_arrive_no_complete": {"mbarrier.arrive"},
    "tirx.ptx.mbarrier_arrive_expect_tx": {"mbarrier.arrive"},
    "tirx.ptx.mbarrier_expect_tx": {"mbarrier.expect_tx"},
    "tirx.ptx.mbarrier_complete_tx": {"mbarrier.completion_issue"},
    "tirx.ptx.mbarrier_init": {"mbarrier.init"},
    "tirx.ptx.mbarrier_try_wait": {"mbarrier.wait"},
    "tirx.cuda.mbarrier_wait": {"mbarrier.wait"},
    "tirx.cuda.mbarrier_wait_acquire_cluster": {"mbarrier.wait"},
    "tirx.ptx.setmaxnreg": {"setmaxnreg.register", "setmaxnreg.resume"},
    "tirx.ptx.tcgen05_alloc": {"tcgen05.alloc.register", "tcgen05.alloc.resume"},
    "tirx.ptx.tcgen05_commit": {"tcgen05.commit.issue"},
    "tirx.ptx.tcgen05_dealloc": {
        "tcgen05.dealloc.register",
        "tcgen05.dealloc.resume",
    },
    "tirx.ptx.tcgen05_relinquish_alloc_permit": {
        "tcgen05.relinquish_alloc_permit.register",
        "tcgen05.relinquish_alloc_permit.resume",
    },
}

_CP_ASYNC_COMPLETION_OPS = frozenset(
    {
        "tirx.ptx.cp_async_ca",
        "tirx.ptx.cp_async_ca_src_size",
        "tirx.ptx.cp_async_commit_group",
        "tirx.ptx.cp_async_wait_group",
    }
)
_BULK_ASYNC_GROUP_OPS = frozenset(
    {
        "tirx.ptx.cp_async_bulk_commit_group",
        "tirx.ptx.cp_async_bulk_wait_group",
    }
)


@dataclass(frozen=True)
class NativeProtocolArtifact:
    module: Any
    phase_by_name: dict[str, int]
    source_op_ids: dict[tuple[str, str], tuple[int, ...]]

    def inputs(self) -> dict[str, np.ndarray]:
        bindings: dict[str, np.ndarray] = {}
        tcgen_phase = self.phase_by_name["tcgen_ordering_protocol_ops"]
        bindings[f"k{tcgen_phase}:output"] = np.zeros((4, 32), dtype=np.uint32)
        for kernel_name, size in (
            ("cp_async_group_protocol_ops", 128),
            ("cp_async_predicated_group_protocol_ops", 128),
            ("cp_async_zero_fill_group_protocol_ops", 128),
            ("ldgsts_cp_async_group_protocol_ops", 128),
            ("cp_async_uncommitted_protocol_op", 128),
            ("bulk_async_group_protocol_ops", 16),
        ):
            phase_index = self.phase_by_name[kernel_name]
            bindings[f"k{phase_index}:source"] = np.arange(size, dtype=np.uint8)
            if kernel_name in {
                "cp_async_group_protocol_ops",
                "ldgsts_cp_async_group_protocol_ops",
                "bulk_async_group_protocol_ops",
            }:
                bindings[f"k{phase_index}:output"] = np.zeros(size, dtype=np.uint8)
        for kernel_name, shape, dtype in (
            ("clc_no_work_cluster_acquire_wait", (1,), np.uint32),
            ("pure_warp_sync_source_ops", (32, 10), np.uint32),
            ("pure_cta_sync_source_ops", (2, 32, 3), np.int64),
            ("mbarrier_query_sync_source_ops", (32, 4), np.uint32),
        ):
            phase_index = self.phase_by_name[kernel_name]
            bindings[f"k{phase_index}:output"] = np.zeros(shape, dtype=dtype)
        return bindings


@pytest.fixture(scope="module")
def native_protocol_artifact(tmp_path_factory) -> NativeProtocolArtifact:
    cache_dir: Path = tmp_path_factory.mktemp("native-synccheck-device-protocol")
    spec = analyze(_KERNELS)
    module = numsim.transpile(_KERNELS, cache_dir=cache_dir, _analysis_capable=True)
    assert module.spec.to_manifest(include_source_spans=False) == spec.to_manifest(include_source_spans=False)
    return NativeProtocolArtifact(
        module=module,
        phase_by_name={kernel.name: index for index, kernel in enumerate(module.spec.kernels)},
        source_op_ids={
            (kernel.name, op_name): tuple(
                entry.op_id
                for entry in kernel.source_map
                if entry.kind == "Call" and str(getattr(entry.node.op, "name", "")) == op_name
            )
            for kernel in spec.kernels
            for op_name in _KERNEL_BY_OP
        },
    )


def _source_op_ids(artifact: NativeProtocolArtifact, op_name: str) -> tuple[int, ...]:
    kernel_name = _KERNEL_BY_OP[op_name]
    return artifact.source_op_ids[kernel_name, op_name]


def _run_named_protocol_phase(artifact: NativeProtocolArtifact, kernel_name: str) -> dict:
    return (
        numsim.Engine(
            max_workers=1,
            native_loop_iteration_budget=1_000,
            native_loop_reschedule_quantum=16,
        )
        .run_synccheck_phase(
            artifact.module,
            artifact.inputs(),
            phase_index=artifact.phase_by_name[kernel_name],
            coverage_bounds=numsim.CoverageBounds(0, 0),
            resource_limits=numsim.ResourceLimits(
                max_schedules=100,
                max_backtrack_nodes=10_000,
                max_events_per_run=10_000,
                max_total_events=100_000,
                max_loop_steps=100_000,
                max_wall_time_ms=30_000,
                max_diagnostic_bytes=1_000_000,
            ),
            max_polls=10_000,
            max_transitions=10_000,
        )
        .to_dict()
    )


def _assert_runtime_case(artifact: NativeProtocolArtifact, op_name: str, result: dict) -> None:
    source_op_ids = _source_op_ids(artifact, op_name)
    assert source_op_ids, f"{op_name} is absent from its assigned runtime kernel"

    assert result["phase"]["name"] == _KERNEL_BY_OP[op_name]
    assert result["execution_error"] is None
    assert result["findings"] == []
    assert result["verdict"] == "clean"
    assert result["incomplete"] == []
    assert result["coverage"]["eligible_for_clean"] is True
    assert result["search"]["algorithm"] == "fixed_sync_state"
    assert result["stats"]["task_count"] > 0
    assert result["stats"]["completed_task_count"] == result["stats"]["task_count"]
    if op_name in _CP_ASYNC_COMPLETION_OPS | _BULK_ASYNC_GROUP_OPS:
        assert result["stats"]["completion_operation_count"] > 0

    expected_effects = _EFFECTS_BY_OP.get(op_name)
    if expected_effects is not None:
        observed = {
            effect["effect"]
            for effect in result["effects"]
            if effect["operation"]["source_op_id"] in source_op_ids
        }
        assert expected_effects <= observed, (op_name, observed, expected_effects)


def test_divergent_unaligned_named_barrier_runtime(native_protocol_artifact):
    result = _run_named_protocol_phase(
        native_protocol_artifact, "divergent_unaligned_named_barrier_op"
    )

    assert result["verdict"] == "clean"
    assert result["execution_error"] is None
    assert result["findings"] == []
    assert result["incomplete"] == []


def test_cp_async_predicate_tracks_only_issuing_lanes(native_protocol_artifact):
    result = _run_named_protocol_phase(
        native_protocol_artifact, "cp_async_predicated_group_protocol_ops"
    )

    assert result["verdict"] == "clean"
    assert result["execution_error"] is None
    assert result["findings"] == []
    assert result["incomplete"] == []
    # Sixteen issuing lanes, with source-read and full-completion milestones
    # for each non-empty per-lane group.  The other lanes commit empty groups.
    assert result["stats"]["completion_operation_count"] == 32


def test_cp_async_zero_fill_keeps_all_lanes_in_their_groups(native_protocol_artifact):
    result = _run_named_protocol_phase(
        native_protocol_artifact, "cp_async_zero_fill_group_protocol_ops"
    )

    assert result["verdict"] == "clean"
    assert result["execution_error"] is None
    assert result["findings"] == []
    assert result["incomplete"] == []
    # src-size=0 still issues cp.async, so all 32 lanes own a non-empty group.
    assert result["stats"]["completion_operation_count"] == 64


def test_ldgsts_cp_async_uses_the_same_thread_local_group_protocol(native_protocol_artifact):
    result = _run_named_protocol_phase(
        native_protocol_artifact, "ldgsts_cp_async_group_protocol_ops"
    )

    assert result["verdict"] == "clean"
    assert result["execution_error"] is None
    assert result["findings"] == []
    assert result["incomplete"] == []
    assert result["stats"]["completion_operation_count"] == 64


def test_uncommitted_cp_async_is_a_protocol_error(native_protocol_artifact):
    result = _run_named_protocol_phase(native_protocol_artifact, "cp_async_uncommitted_protocol_op")

    assert result["verdict"] == "error"
    assert result["execution_error"] is not None
    assert "uncommitted issue" in result["execution_error"]["message"]
    assert result["findings"] == []
    assert result["incomplete"] == []


@pytest.mark.parametrize("kernel_name", sorted(set(_KERNEL_BY_OP.values())))
def test_protocol_runtime(native_protocol_artifact, kernel_name):
    result = _run_named_protocol_phase(native_protocol_artifact, kernel_name)
    for op_name, owner in _KERNEL_BY_OP.items():
        if owner == kernel_name:
            _assert_runtime_case(native_protocol_artifact, op_name, result)
