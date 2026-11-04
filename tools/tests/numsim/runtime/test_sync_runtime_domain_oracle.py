from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.api import ExecutionSubset
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tvm_tirx.backend.cuda.lang.clc import query_cancel_first_ctaid_x
from tvm_tirx.script import tirx as T


@T.function
def tvm_storage_sync_roundtrip(
    source: T.Tensor((32,), "uint32"), output: T.Tensor((32,), "uint32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((32,), "uint32")
    shared[lane] = source[lane] ^ T.uint32(0xA5A55A5A)
    T.gpu_storage_sync("shared")
    output[lane] = shared[lane]


@T.function
def mbarrier_query_modifier_domain(output: T.Tensor((32, 2), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    barrier = T.alloc_shared((1,), "uint64")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    T.ptx.mbarrier.test_wait.parity.acquire.cluster.shared.b64(
        output[lane, 0],
        T.address_of(barrier[0]),
        T.uint32(1),
    )
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))
    T.ptx.mbarrier.test_wait.parity.relaxed.cta.shared.b64(
        output[lane, 1],
        T.address_of(barrier[0]),
        T.uint32(0),
    )


@T.function
def ordering_and_cp_async_mbarrier_modifier_forms(
    source: T.Tensor((32,), "uint32"), output: T.Tensor((32,), "uint32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((2,), "uint64")
    value: T.let = source[lane] ^ T.uint32(0x5A5AA5A5)
    T.ptx.fence.proxy.async_()
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.cp.async_.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        T.ptx.cp.async_.mbarrier.arrive.noinc.shared__cta.b64(T.address_of(barriers[1]))
    T.ptx.fence.proxy.async_.global_()
    output[lane] = value


@T.function
def cp_async_mbarrier_noinc_mixed_immediate_and_deferred_lanes(
    source: T.Tensor((128,), "uint8"), output: T.Tensor((128,), "uint8")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "uint8")
    barrier = T.alloc_shared((1,), "uint64")
    for element in T.unroll(4):
        shared[lane * 4 + element] = T.uint8(0xA5)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 32)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    T.ptx["cp.async.ca.shared.global"](
        T.address_of(shared[lane * 4]),
        T.address_of(source[lane * 4]),
        4,
        pred=lane < 16,
    )
    T.ptx.cp.async_.mbarrier.arrive.noinc.shared__cta.b64(T.address_of(barrier[0]))
    if lane == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    for element in T.unroll(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


@T.function
def cp_async_mbarrier_noinc_raw_shared_offset(
    barrier_index: T.int32, output: T.Tensor((1,), "uint32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((2,), "uint64")
    barrier_base: T.uint32 = T.cuda.cvta_generic_to_shared(barriers.ptr_to([0]))
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[barrier_index]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.cp.async_.mbarrier.arrive.noinc.shared__cta.b64(
            barrier_base + T.cast(barrier_index * 8, "uint32")
        )
        T.cuda.mbarrier_wait(T.address_of(barriers[barrier_index]), 0)
        output[0] = T.uint32(1)


@T.function
def remote_expect_tx_completes_target_barrier(output: T.Tensor((2,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    barrier = T.alloc_shared((1,), "uint64")
    remote_barrier = T.alloc_local((1,), "uint64")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0:
            T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        else:
            T.ptx.mapa.shared__cluster.u64(remote_barrier[0], T.address_of(barrier[0]), T.uint32(0))
            T.ptx.mbarrier.arrive.expect_tx.b64(
                remote_barrier[0],
                T.uint32(0),
                pred=True,
            )
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1


@T.function
def clc_no_work_cluster_acquire_wait(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, cluster=1))
    lane = T.cuda.lane_id()
    response = T.alloc_shared((4,), "uint32", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=8)
    first_ctaid_x = T.local_scalar("uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), T.uint32(16))
        T.ptx[
            "clusterlaunchcontrol.try_cancel.async.shared::cta"
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
        ](T.address_of(response[0]), T.address_of(barrier[0]))
        T.cuda.mbarrier_wait_acquire_cluster(T.address_of(barrier[0]), 0)
        query_cancel_first_ctaid_x(first_ctaid_x, T.address_of(response[0]))
        output[0] = first_ctaid_x


@T.function
def clc_claims_nonresident_cluster(output: T.Tensor((2,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=4, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    response = T.alloc_shared((4,), "uint32", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=8)
    first_ctaid_x = T.local_scalar("uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), T.uint32(16))
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        T.ptx[
            "clusterlaunchcontrol.try_cancel.async.shared::cta"
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
        ](T.address_of(response[0]), T.address_of(barrier[0]))
    if lane == 0:
        T.cuda.mbarrier_wait_acquire_cluster(T.address_of(barrier[0]), 0)
        query_cancel_first_ctaid_x(first_ctaid_x, T.address_of(response[0]), use_ld_acquire=True)
        output[cta] = first_ctaid_x


@T.function
def clc_claims_nonresident_cluster_yz(output: T.Tensor((2, 6), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=4, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    response = T.alloc_shared((4,), "uint32", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=8)
    response_value = T.local_scalar("uint128")
    canceled = T.local_scalar("uint32")
    first_ctaid_y = T.local_scalar("uint32")
    first_ctaid_z = T.local_scalar("uint32")
    first_ctaid_x = T.local_scalar("uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), T.uint32(16))
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        T.ptx[
            "clusterlaunchcontrol.try_cancel.async.shared::cta"
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
        ](T.address_of(response[0]), T.address_of(barrier[0]))
    if lane == 0:
        T.cuda.mbarrier_wait_acquire_cluster(T.address_of(barrier[0]), 0)
        T.ptx["ld.acquire.cta.shared.b128"](response_value, T.address_of(response[0]))
        T.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, response_value)
        first_ctaid_y = T.uint32(0xFFFFFFFF)
        first_ctaid_z = T.uint32(0xFFFFFFFF)
        T.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__y.b32.b128(
            first_ctaid_y, response_value, pred=canceled
        )
        T.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__z.b32.b128(
            first_ctaid_z, response_value, pred=canceled
        )
        output[cta, 0] = first_ctaid_y
        output[cta, 1] = first_ctaid_z
        first_ctaid_x = T.uint32(0xFFFFFFFF)
        first_ctaid_y = T.uint32(1)
        first_ctaid_z = T.uint32(0xFFFFFFFF)
        T.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128(
            first_ctaid_x,
            first_ctaid_y,
            first_ctaid_z,
            T.ptx.SINK,
            response_value,
            pred=first_ctaid_y,
        )
        output[cta, 2] = first_ctaid_x
        output[cta, 3] = first_ctaid_y
        output[cta, 4] = first_ctaid_z
        first_ctaid_x = T.uint32(99)
        T.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128(
            first_ctaid_x, T.ptx.SINK, T.ptx.SINK, T.ptx.SINK, response_value, pred=T.uint32(0)
        )
        output[cta, 5] = first_ctaid_x


def test_mbarrier_query_modifier_domain_matches_barrier_phase(tmp_path):
    module = numsim.transpile(mbarrier_query_modifier_domain, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((32, 2), dtype=np.uint32)})
    np.testing.assert_array_equal(result.outputs["output"], np.ones((32, 2), dtype=np.uint32))


def test_ordering_and_cp_async_mbarrier_modifier_forms_preserve_values(tmp_path):
    source = np.arange(32, dtype=np.uint32) * np.uint32(0x01020304)
    module = numsim.transpile(ordering_and_cp_async_mbarrier_modifier_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros_like(source)})
    expected = source ^ np.uint32(0x5A5AA5A5)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tvm_storage_sync_roundtrip_preserves_shared_values(tmp_path):
    source = np.arange(32, dtype=np.uint32) * np.uint32(13) + np.uint32(7)
    module = numsim.transpile(tvm_storage_sync_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros_like(source)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source ^ np.uint32(0xA5A55A5A))


@T.function
def global_storage_sync(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    T.gpu_storage_sync("global")
    output[T.cuda.lane_id()] = T.uint32(1)


def test_storage_sync_outside_shared_memory_is_rejected(tmp_path):
    with pytest.raises(
        UnsupportedTIRxError, match='only for the shared storage scope, got "global"'
    ):
        numsim.transpile(global_storage_sync, cache_dir=tmp_path)


def test_cp_async_mbarrier_noinc_arrives_immediately_only_for_lanes_without_prior_work(
    tmp_path,
):
    source = np.arange(128, dtype=np.uint8)
    module = numsim.transpile(
        cp_async_mbarrier_noinc_mixed_immediate_and_deferred_lanes,
        cache_dir=tmp_path,
    )
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros_like(source)},
    )

    expected = np.full(128, np.uint8(0xA5))
    expected[:64] = source[:64]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cp_async_mbarrier_noinc_accepts_bound_raw_shared_offset(tmp_path):
    module = numsim.transpile(cp_async_mbarrier_noinc_raw_shared_offset, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"barrier_index": 1, "output": np.zeros(1, dtype=np.uint32)},
    )

    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.uint32))


def test_remote_expect_tx_completes_the_named_ctas_barrier(tmp_path):
    module = numsim.transpile(remote_expect_tx_completes_target_barrier, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(2, dtype=np.int32)})
    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))


def test_clc_no_work_completion_and_cluster_acquire_wait(tmp_path):
    module = numsim.transpile(clc_no_work_cluster_acquire_wait, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.uint32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([0xFFFFFFFF], np.uint32))


def test_clc_claims_the_nonresident_logical_cluster(tmp_path):
    module = numsim.transpile(clc_claims_nonresident_cluster, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros(2, dtype=np.uint32)},
        subset=ExecutionSubset(cluster_ids=[0]),
    )

    np.testing.assert_array_equal(result.outputs["output"], np.array([2, 2], np.uint32))


def test_clc_first_ctaid_y_and_z_are_zero_in_linear_launch_model(tmp_path):
    module = numsim.transpile(clc_claims_nonresident_cluster_yz, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((2, 6), dtype=np.uint32)},
        subset=ExecutionSubset(cluster_ids=[0]),
    )

    np.testing.assert_array_equal(result.outputs["output"], [[0, 0, 2, 0, 0, 99]] * 2)
