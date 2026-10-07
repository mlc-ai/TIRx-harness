from __future__ import annotations

import numpy as np
from tvm.backend.cuda.lang.clc import query_cancel_first_ctaid_x
from tvm.script import tirx as T

from tirx_harness import numsim
from tirx_harness.numsim.api import ExecutionSubset


@T.prim_func
def tvm_storage_sync_roundtrip(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    shared[lane] = source[lane] ^ T.uint32(0xA5A55A5A)
    T.tvm_storage_sync("shared")
    output[lane] = shared[lane]


@T.prim_func
def mbarrier_query_modifier_domain(output: T.Buffer((32, 2), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
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


@T.prim_func
def ordering_and_cp_async_mbarrier_modifier_forms(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
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


@T.prim_func
def cp_async_mbarrier_noinc_mixed_immediate_and_deferred_lanes(
    source: T.Buffer((128,), "uint8"), output: T.Buffer((128,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
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


@T.prim_func
def cp_async_mbarrier_noinc_raw_shared_offset(
    barrier_index: T.int32, output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
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


@T.prim_func
def remote_expect_tx_completes_target_barrier(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    remote_barrier = T.alloc_buffer((1,), "uint64", scope="local")
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


@T.prim_func
def clc_no_work_cluster_acquire_wait(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    response = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
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


@T.prim_func
def clc_claims_nonresident_cluster(output: T.Buffer((2,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([2])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    response = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
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


@T.prim_func
def clc_claims_nonresident_cluster_yz(output: T.Buffer((2, 6), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([2])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    response = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
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


@T.prim_func
def sm100_pair_base_handle_domain(
    input_map_even: T.TensorMap(),
    input_map_odd: T.TensorMap(),
    output: T.Buffer((2, 4), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        T.evaluate(
            T.call_intrin(
                "uint32",
                "tirx.cuda.sm100_2sm_leader_smem_addr",
                T.address_of(barrier[0]),
            )
        )
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
            ](
                T.address_of(shared[0]),
                T.Select(cta == 0, T.address_of(input_map_even), T.address_of(input_map_odd)),
                0,
                0,
                T.call_intrin(
                    "uint32",
                    "tirx.cuda.sm100_2sm_leader_smem_addr",
                    T.address_of(barrier[0]),
                ),
            )
        )
        if cta == 0:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), T.uint32(32))
            T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


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


def test_sm100_pair_base_handle_form_routes_both_ctas(tmp_path):
    even = np.arange(4, dtype=np.float32).reshape(1, 4) + np.float32(10)
    odd = np.arange(4, dtype=np.float32).reshape(1, 4) + np.float32(20)

    def tensor_map(array: np.ndarray) -> np.ndarray:
        return numsim.TensorMap(
            base=array,
            global_shape=(4, 1),
            global_strides=(16,),
            box_shape=(4, 1),
            element_strides=(1, 1),
        ).numpy()

    module = numsim.transpile(sm100_pair_base_handle_domain, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "input_map_even": tensor_map(even),
            "input_map_odd": tensor_map(odd),
            "output": np.zeros((2, 4), dtype=np.float32),
        },
    )
    np.testing.assert_array_equal(result.outputs["output"], np.concatenate([even, odd], axis=0))
