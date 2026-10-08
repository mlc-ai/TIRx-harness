"""Native Racecheck integration coverage."""

from __future__ import annotations

import numpy as np
from tirx_harness import numsim
from tirx_harness.numsim.api import ExecutionSubset
from tirx_harness.numsim.checker_report import RaceReport
from tirx_harness.numsim.checkers import _run_racecheck as internal_racecheck
from tirx_harness.numsim.bindings import prepare_bindings
from tests.numsim.support.remote_mbarrier import mapped_remote_mbarrier_cluster_view
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout


_RACECHECK_PERMUTED_SHARED_LAYOUT = TileLayout(S[(4, 32) : (1, 4)])


@T.prim_func
def native_racecheck_write_write():
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    if lane == 0:
        shared[0] = warp + 1


@T.prim_func
def native_racecheck_fully_predicated_shared_pointer_load(
    output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "float32", scope="shared")
    if lane < 0:
        T.ptx.ld.shared.f32(output[lane], shared.ptr_to([lane]))


@T.prim_func
def native_racecheck_tile_region_shared_index(
    source: T.Buffer((1,), "float32"), output: T.Buffer((32,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    index = T.alloc_buffer((1,), "int32", scope="shared")
    value: T.f32[1]
    if lane == 0:
        index[0] = 0
    T.cuda.warp_sync()
    Tx.copy(value[:], source[index[0] : index[0] + 1])
    output[lane] = value[0]


@T.prim_func
def native_racecheck_lane_selected_oob(
    active_lane: T.int32, source: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == active_lane:
        output[0] = source[lane]


@T.prim_func
def native_racecheck_mbarrier_ordered(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        shared[0] = 7
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
    if (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        output[0] = shared[0]


@T.prim_func
def native_racecheck_cta_reduce_hidden_scratch_waw(output: T.Buffer((1,), "float32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    scratch = T.alloc_buffer((2,), "float32", scope="shared")
    if (warp == 1) and (lane == 0):
        scratch[0] = T.float32(7)
    reduced: T.let = T.cuda.cta_sum(T.Cast("float32", lane), 2, scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        output[0] = reduced


@T.prim_func
def native_racecheck_cta_reduce_scratch_clean(output: T.Buffer((1,), "float32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    scratch = T.alloc_buffer((2,), "float32", scope="shared")
    reduced: T.let = T.cuda.cta_sum(T.Cast("float32", warp * 32 + lane), 2, scratch.ptr_to([0]))
    if (warp == 0) and (lane == 0):
        output[0] = reduced


@T.prim_func
def native_racecheck_single_warp_cta_reduce(output: T.Buffer((1,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    scratch = T.alloc_buffer((1,), "float32", scope="shared")
    reduced: T.let = T.cuda.cta_sum(T.Cast("float32", lane), 1, scratch.ptr_to([0]))
    if lane == 0:
        output[0] = reduced


@T.prim_func
def native_racecheck_schedule_sensitive_atomic(counter: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    if lane == 0:
        ticket: T.let = T.cuda.atomic_add(counter.ptr_to([0]), T.int32(1))
        if warp == 0:
            shared[0] = 1
        else:
            if ticket == 0:
                shared[0] = 2


@T.prim_func
def native_racecheck_named_barrier_ordered(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    if (warp == 0) and (lane == 0):
        shared[0] = 7
    T.ptx.bar.sync(T.uint32(5), T.uint32(64))
    if (warp == 1) and (lane == 0):
        output[0] = shared[0]


@T.prim_func
def native_racecheck_two_ctas():
    T.device_entry()
    _cta = T.cta_id([2])
    _warp = T.warp_id([1])


@T.prim_func
def native_racecheck_cross_cluster_global_waw(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = 1


@T.prim_func
def native_racecheck_same_cluster_global_waw(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([2])
    lane = T.lane_id([32])
    if lane == 0:
        output[0] = 1


@T.prim_func
def native_racecheck_cross_cluster_tile_copy_waw(
    source: T.Buffer((128,), "float32"), output: T.Buffer((128,), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([2])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    staged: T.f32[128]
    if lane == 0:
        Tx.copy(staged[:], source[:])
        Tx.copy(output[:], staged[:])


@T.prim_func
def native_racecheck_shared_permute_snapshot_waw():
    T.device_entry()
    _warp = T.warp_id([2])
    _lane = T.lane_id([32])
    source = T.alloc_buffer((128,), "uint32", scope="shared")
    destination = T.alloc_buffer(
        (128,), "uint32", scope="shared", layout=_RACECHECK_PERMUTED_SHARED_LAYOUT
    )
    Tx.warp.permute_layout(destination[:], source[:])


@T.prim_func
def native_racecheck_shared_permute_snapshot_source_race():
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    source = T.alloc_buffer((128,), "uint32", scope="shared")
    destination = T.alloc_buffer(
        (128,), "uint32", scope="shared", layout=_RACECHECK_PERMUTED_SHARED_LAYOUT
    )
    if warp == 0:
        if lane == 0:
            source[0] = 1
    else:
        Tx.warp.permute_layout(destination[:], source[:])


@T.prim_func
def native_racecheck_cross_cluster_global_atomic(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        _old: T.let = T.cuda.atomic_add(output.ptr_to([0]), T.int32(1))


@T.prim_func
def native_racecheck_bulk_active_lane(active_lane: T.int32, source: T.Buffer((32,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == active_lane:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == active_lane:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 32)
        T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), source.ptr_to([0]), T.cast(32, "uint32"), barrier.ptr_to([0])
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)


@T.prim_func
def native_racecheck_bulk_write_write(source: T.Buffer((32,), "uint8")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint8", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    if lane == 0:
        if warp == 0:
            T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([0]), 1)
        else:
            T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([1]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        if warp == 0:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([0]), 32)
            T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
                shared.ptr_to([0]), source.ptr_to([0]), T.cast(32, "uint32"), barriers.ptr_to([0])
            )
            T.cuda.mbarrier_wait(barriers.ptr_to([0]), 0)
        else:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([1]), 32)
            T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
                shared.ptr_to([0]), source.ptr_to([0]), T.cast(32, "uint32"), barriers.ptr_to([1])
            )
            T.cuda.mbarrier_wait(barriers.ptr_to([1]), 0)


@T.prim_func
def native_racecheck_typed_tma_multicast(source: T.Buffer((4,), "float32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=barrier.ptr_to([0]),
            cta_group=1,
            cta_mask=T.int32(3),
        )
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)


@T.prim_func
def native_racecheck_typed_tma_write_write(source: T.Buffer((4,), "float32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([warp]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=barriers.ptr_to([warp]),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([warp]), 16)
        T.cuda.mbarrier_wait(barriers.ptr_to([warp]), 0)


@T.prim_func
def native_racecheck_typed_tma_disjoint_global_subregion(
    source: T.Buffer((2, 4, 8), "float32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 8), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        if warp == 0:
            Tx.copy_async(
                shared[:, :],
                source[1, :, :],
                dispatch="tma_auto",
                mbar=barrier.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 128)
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        else:
            source[0, 0, 0] = T.float32(7)


@T.prim_func
def native_racecheck_raw_tma_clean(input_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0]), T.address_of(input_map), 0, T.address_of(barrier[0]))
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)


@T.prim_func
def native_racecheck_raw_tma_write_write(input_map: T.TensorMap()):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([warp]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0]), T.address_of(input_map), 0, T.address_of(barriers[warp]))
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[warp]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[warp]), 0)


def _run_native_phase(module, inputs):
    return module.load()._native_racecheck_phase(
        inputs=prepare_bindings(inputs).to_payload(),
        phase_index=0,
        subset=None,
        inspect_accesses=True,
        max_polls=200,
        max_transitions=200,
        native_loop_iteration_budget=1_000,
        native_loop_reschedule_quantum=16,
    )


def _float4_tensor_map(array: np.ndarray) -> np.ndarray:
    return numsim.TensorMap(
        base=array,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
        element_strides=(1,),
    ).numpy()


def test_native_racecheck_reports_exact_cross_warp_shared_write_race(tmp_path):
    module = numsim.transpile(native_racecheck_write_write, cache_dir=tmp_path)

    result = _run_native_phase(module, {})

    assert result["verdict"] == "error"
    assert result["sync"] == {"verdict": "clean", "findings": [], "incomplete": [], "effects": []}
    assert result["incomplete"] == []
    assert result["access_count"] == 1
    assert len(result["accesses"]) == 1
    assert result["accesses"][0]["access_kind"] == "write"
    assert result["accesses"][0]["space"] == "shared"
    assert result["accesses"][0]["active_lane_count"] == 1
    assert result["accesses"][0]["lanes"][0]["lane"] == 0

    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["access_pair"] == "write_write"
    assert finding["prior"]["operation"]["global_warp_id"] == 0
    assert finding["prior"]["lane"] == 0
    assert finding["prior"]["access_kind"] == "write"
    assert finding["prior"]["space"] == "shared"
    assert finding["current"]["operation"]["global_warp_id"] == 1
    assert finding["current"]["lane"] == 0
    assert finding["current"]["access_kind"] == "write"
    assert finding["current"]["space"] == "shared"
    assert finding["prior"]["span"] == finding["current"]["span"]
    assert finding["overlap"] == finding["prior"]["span"]
    assert finding["overlap"]["byte_len"] == 4
    assert result["stats"]["available"] is True
    assert result["execution_error"]["kind"] == "engine_error"
    report = RaceReport.from_native(result)
    assert report.verdict == "error"
    assert [item.details["access_pair"] for item in report.findings] == ["write_write"]


def test_native_racecheck_ignores_fully_predicated_shared_pointer_load(tmp_path):
    module = numsim.transpile(
        native_racecheck_fully_predicated_shared_pointer_load,
        cache_dir=tmp_path,
    )

    result = _run_native_phase(module, {"output": np.zeros(32, dtype=np.float32)})

    assert result["verdict"] == "clean"
    assert result["access_count"] == 0
    assert result["findings"] == []
    assert result["execution_error"] is None


def test_native_racecheck_observes_shared_load_hidden_in_tile_region_index(tmp_path):
    module = numsim.transpile(native_racecheck_tile_region_shared_index, cache_dir=tmp_path)
    result = _run_native_phase(
        module,
        {
            "source": np.array([3.0], dtype=np.float32),
            "output": np.zeros(32, dtype=np.float32),
        },
    )

    tile_op_id = next(
        entry.op_id
        for entry in module.spec.kernels[0].source_map
        if entry.kind == "TilePrimitiveCall"
    )
    shared_index_reads = [
        access
        for access in result["accesses"]
        if access["access_kind"] == "read"
        and access["space"] == "shared"
        and access["logical_buffer"] == "index"
    ]

    assert result["verdict"] == "clean"
    assert len(shared_index_reads) == 1
    assert shared_index_reads[0]["operation"]["source_op_id"] == tile_op_id
    assert shared_index_reads[0]["active_lane_count"] == 32


def test_native_racecheck_oob_is_exact_error_or_clean_from_concrete_mask(tmp_path):
    module = numsim.transpile(native_racecheck_lane_selected_oob, cache_dir=tmp_path)
    source = np.array([17], dtype=np.int32)

    active_oob = _run_native_phase(
        module,
        {"active_lane": np.int32(1), "source": source, "output": np.zeros(1, dtype=np.int32)},
    )
    assert active_oob["verdict"] == "error"
    assert active_oob["findings"] == []
    assert active_oob["incomplete"] == []
    assert active_oob["access_count"] == 0
    assert active_oob["stats"]["available"] is True
    assert active_oob["execution_error"]["kind"] == "oob"
    active_report = RaceReport.from_native(active_oob)
    assert active_report.verdict == "error"
    assert all(item.status != "review" for item in active_report.findings)

    masked_clean = _run_native_phase(
        module,
        {"active_lane": np.int32(0), "source": source, "output": np.zeros(1, dtype=np.int32)},
    )
    assert masked_clean["verdict"] == "clean"
    assert masked_clean["findings"] == []
    assert masked_clean["incomplete"] == []
    assert masked_clean["execution_error"] is None
    assert masked_clean["access_count"] == 2
    assert [access["active_lane_count"] for access in masked_clean["accesses"]] == [1, 1]
    assert [access["lanes"][0]["lane"] for access in masked_clean["accesses"]] == [0, 0]
    masked_report = RaceReport.from_native(masked_clean)
    assert masked_report.verdict == "clean"
    assert masked_report.findings == []

    public_active = internal_racecheck(
        native_racecheck_lane_selected_oob,
        inputs={
            "active_lane": np.int32(1),
            "source": source,
            "output": np.zeros(1, dtype=np.int32),
        },
        cache_dir=tmp_path,
    )
    public_masked = internal_racecheck(
        native_racecheck_lane_selected_oob,
        inputs={
            "active_lane": np.int32(0),
            "source": source,
            "output": np.zeros(1, dtype=np.int32),
        },
        cache_dir=tmp_path,
    )
    assert public_active.verdict == "error"
    assert all(finding.status != "review" for finding in public_active.findings)
    assert public_masked.verdict == "clean"
    assert public_masked.findings == []
    assert public_masked.to_dict()["native"]["input"]["digest"]


def test_native_racecheck_raw_bulk_uses_the_concrete_active_issuer_lane(tmp_path):
    module = numsim.transpile(native_racecheck_bulk_active_lane, cache_dir=tmp_path)
    source = np.arange(32, dtype=np.uint8)

    result = _run_native_phase(module, {"active_lane": np.int32(1), "source": source})

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []
    assert result["execution_error"] is None
    assert result["access_count"] == 2
    assert [
        (access["access_kind"], access["space"], access["width"]) for access in result["accesses"]
    ] == [("read", "global", 32), ("write", "shared", 32)]
    assert [access["active_lane_count"] for access in result["accesses"]] == [1, 1]
    assert [access["lanes"][0]["lane"] for access in result["accesses"]] == [1, 1]
    assert [access["lanes"][0]["spans"][0]["byte_len"] for access in result["accesses"]] == [32, 32]


def test_native_racecheck_raw_bulk_rejects_a_true_async_write_race(tmp_path):
    module = numsim.transpile(native_racecheck_bulk_write_write, cache_dir=tmp_path)
    source = np.arange(32, dtype=np.uint8)

    result = _run_native_phase(module, {"source": source})

    assert result["verdict"] == "error"
    assert result["incomplete"] == []
    assert result["access_count"] == 3
    assert [access["access_kind"] for access in result["accesses"]].count("read") == 2
    assert [access["access_kind"] for access in result["accesses"]].count("write") == 1
    assert all(
        access["space"] == "global"
        for access in result["accesses"]
        if access["access_kind"] == "read"
    )
    assert all(
        access["space"] == "shared"
        for access in result["accesses"]
        if access["access_kind"] == "write"
    )
    assert result["execution_error"]["kind"] == "completion_source_operation_failed"
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["access_pair"] == "write_write"
    assert finding["prior"]["space"] == "shared"
    assert finding["current"]["space"] == "shared"
    assert finding["overlap"]["byte_len"] == 32

    public_report = internal_racecheck(
        native_racecheck_bulk_write_write,
        inputs={"source": source},
        cache_dir=tmp_path,
    )
    assert public_report.verdict == "error"
    assert [item.details["access_pair"] for item in public_report.findings] == ["write_write"]
    assert all(item.status != "review" for item in public_report.findings)
    native = public_report.to_dict()["native"]
    assert native["verdict"] == "error"
    assert native["findings"][0]["access_pair"] == "write_write"


def test_native_racecheck_typed_tma_multicast_records_each_target_allocation(tmp_path):
    module = numsim.transpile(native_racecheck_typed_tma_multicast, cache_dir=tmp_path)
    source = np.arange(4, dtype=np.float32)

    result = _run_native_phase(module, {"source": source})

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []
    assert result["execution_error"] is None
    reads = [access for access in result["accesses"] if access["access_kind"] == "read"]
    writes = [access for access in result["accesses"] if access["access_kind"] == "write"]
    assert len(reads) == 4
    assert len(writes) == 8
    assert all(access["space"] == "global" for access in reads)
    assert all(access["space"] == "shared" for access in writes)
    assert all(access["lanes"][0]["lane"] == 0 for access in reads + writes)
    assert all(access["width"] == 4 for access in reads + writes)
    target_allocations = {access["lanes"][0]["spans"][0]["allocation_id"] for access in writes}
    assert len(target_allocations) == 2

    public = internal_racecheck(
        native_racecheck_typed_tma_multicast,
        inputs={"source": source},
        cache_dir=tmp_path,
    )
    public.require_clean()
    compact = public.to_dict()["native"]
    assert compact["access_count"] == 12
    assert compact["accesses"] == []
    assert compact["accesses_complete"] is False


def test_public_native_racecheck_typed_tma_overlapping_destinations_race(tmp_path):
    source = np.arange(4, dtype=np.float32)

    report = internal_racecheck(
        native_racecheck_typed_tma_write_write,
        inputs={"source": source},
        cache_dir=tmp_path,
    )

    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["findings"][0]["access_pair"] == "write_write"
    assert {
        (finding["ordering_domain"], finding["ordering_failure"])
        for finding in native["findings"]
    } == {("execution", "missing_inter_actor_sync")}
    # Direct mode compacts the four adjacent element fragments into the exact
    # full-operation footprint.  Both TMA writes cover all four float32 cells,
    # so their complete physical intersection is 16 bytes.
    assert native["findings"][0]["overlap"]["byte_len"] == 16


def test_public_native_racecheck_typed_tma_uses_exact_global_subregion(tmp_path):
    report = internal_racecheck(
        native_racecheck_typed_tma_disjoint_global_subregion,
        inputs={"source": np.zeros((2, 4, 8), dtype=np.float32)},
        cache_dir=tmp_path,
    )

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["findings"] == []
    assert native["incomplete"] == []


def test_native_racecheck_raw_tma_records_exact_global_and_shared_payload(tmp_path):
    module = numsim.transpile(native_racecheck_raw_tma_clean, cache_dir=tmp_path)
    input_map = _float4_tensor_map(np.arange(4, dtype=np.float32))

    result = _run_native_phase(module, {"input_map": input_map})

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []
    assert result["execution_error"] is None
    assert result["access_count"] == 8
    assert [
        (access["access_kind"], access["space"], access["width"]) for access in result["accesses"]
    ] == [("read", "global", 4)] * 4 + [("write", "shared", 4)] * 4
    assert all(access["lanes"][0]["lane"] == 0 for access in result["accesses"])

    public = internal_racecheck(
        native_racecheck_raw_tma_clean,
        inputs={"input_map": input_map},
        cache_dir=tmp_path,
    )
    public.require_clean()
    compact = public.to_dict()["native"]
    assert compact["access_count"] == 8
    assert compact["accesses"] == []
    assert compact["accesses_complete"] is False


def test_public_native_racecheck_raw_tma_overlapping_destinations_race(tmp_path):
    input_map = _float4_tensor_map(np.arange(4, dtype=np.float32))

    report = internal_racecheck(
        native_racecheck_raw_tma_write_write,
        inputs={"input_map": input_map},
        cache_dir=tmp_path,
    )

    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["findings"][0]["access_pair"] == "write_write"
    assert native["findings"][0]["overlap"]["byte_len"] == 4


def test_native_racecheck_mbarrier_release_acquire_orders_shared_access(tmp_path):
    module = numsim.transpile(native_racecheck_mbarrier_ordered, cache_dir=tmp_path)

    result = _run_native_phase(module, {"output": np.zeros(1, dtype=np.int32)})

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []
    assert result["sync"]["verdict"] == "clean"
    # Cluster-sharded reports are deterministically grouped by dynamic warp
    # identity, not presented as a cross-warp execution trace.  Pin the two
    # program-order projections without inventing a total order between them.
    sync_effects = result["sync"]["effects"]
    assert [
        effect["effect"] for effect in sync_effects if effect["operation"]["global_warp_id"] == 0
    ] == ["mbarrier.init", "bar.sync.register", "bar.sync.resume", "mbarrier.arrive"]
    assert [
        effect["effect"] for effect in sync_effects if effect["operation"]["global_warp_id"] == 1
    ] == ["bar.sync.register", "bar.sync.resume", "mbarrier.wait"]
    assert result["access_count"] == 3
    shared_accesses = [access for access in result["accesses"] if access["space"] == "shared"]
    assert [access["access_kind"] for access in shared_accesses] == ["write", "read"]
    assert result["stats"]["available"] is True
    assert result["execution_error"] is None
    assert RaceReport.from_native(result).verdict == "clean"

    public_report = internal_racecheck(
        native_racecheck_mbarrier_ordered,
        inputs={"output": np.zeros(1, dtype=np.int32)},
        cache_dir=tmp_path,
    )
    assert public_report.verdict == "clean"
    assert public_report.findings == []
    assert public_report.to_dict()["native"]["incomplete"] == []


def test_public_native_racecheck_accepts_cluster_arrive_through_remote_view(tmp_path):
    report = internal_racecheck(
        mapped_remote_mbarrier_cluster_view,
        inputs={"output": np.zeros(2, dtype=np.int32)},
        cache_dir=tmp_path,
        max_workers=1,
    )

    assert report.verdict == "clean", report.to_dict()
    assert report.findings == []
    assert report.to_dict()["native"]["incomplete"] == []


def test_native_racecheck_direct_atomic_order_executes_without_replay(tmp_path):
    module = numsim.transpile(native_racecheck_schedule_sensitive_atomic, cache_dir=tmp_path)
    counter = np.zeros(1, dtype=np.int32)

    result = numsim.Engine().run_racecheck_phase(module, {"counter": counter})

    assert result.verdict == "clean"
    assert result.findings == []
    assert result.incomplete == []
    assert "search" not in result.payload
    assert "counterexample" not in result.payload
    np.testing.assert_array_equal(counter, np.zeros(1, dtype=np.int32))


def test_native_racecheck_direct_named_barrier_hb_is_clean(tmp_path):
    module = numsim.transpile(native_racecheck_named_barrier_ordered, cache_dir=tmp_path)

    result = numsim.Engine().run_racecheck_phase(module, {"output": np.zeros(1, dtype=np.int32)})

    assert result.verdict == "clean"
    assert result.findings == []
    assert result.incomplete == []
    assert "search" not in result.payload
    assert "counterexample" not in result.payload


def test_native_racecheck_subset_is_typed_incomplete(tmp_path):
    module = numsim.transpile(native_racecheck_two_ctas, cache_dir=tmp_path)
    subset = ExecutionSubset(cta_ids=[0])

    result = numsim.Engine().run_racecheck_phase(module, {}, subset=subset)

    assert result.verdict == "incomplete"
    assert result.payload["analysis_scope"] == {
        "kind": "subset",
        "selected_warp_count": 1,
        "total_warp_count": 2,
    }
    assert result.incomplete == [
        {"kind": "analysis_incomplete", "reason": "subset_execution", "selected_warp_count": 1, "total_warp_count": 2}
    ]


def test_native_racecheck_runs_clusters_in_parallel_with_stable_direct_payload(tmp_path):
    module = numsim.transpile(native_racecheck_two_ctas, cache_dir=tmp_path)

    serial = numsim.Engine(max_workers=1).run_racecheck_phase(module, {})
    parallel = numsim.Engine(max_workers=2).run_racecheck_phase(module, {})

    assert serial.verdict == parallel.verdict == "clean"
    assert serial.findings == parallel.findings == []
    assert serial.incomplete == parallel.incomplete == []
    assert serial.payload["checked_memory_spaces"] == ["global", "shared", "tmem"]
    assert parallel.payload["checked_memory_spaces"] == ["global", "shared", "tmem"]
    assert serial.payload["stats"]["task_count"] == 2
    assert parallel.payload["stats"]["task_count"] == 2
    assert serial.payload["access_count"] == parallel.payload["access_count"] == 0
    assert serial.payload["accesses_complete"] is False
    assert parallel.payload["accesses_complete"] is False


def test_native_racecheck_reports_cross_cluster_global_waw_independent_of_workers(tmp_path):
    module = numsim.transpile(native_racecheck_cross_cluster_global_waw, cache_dir=tmp_path)

    def run(max_workers):
        return numsim.Engine(max_workers=max_workers).run_racecheck_phase(
            module, {"output": np.zeros(1, dtype=np.int32)}
        )

    serial = run(1)
    parallel = run(2)

    assert serial.verdict == parallel.verdict == "error"
    assert serial.findings == parallel.findings
    for result in (serial, parallel):
        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding["access_pair"] == "write_write"
        assert finding["prior"]["space"] == finding["current"]["space"] == "global"
        assert result.incomplete == []
        assert result.payload["checked_memory_spaces"] == ["global", "shared", "tmem"]
        # One lane-0 global write per CTA, in parity with the journal below
        # (`len(inspected.payload["accesses"]) == 2`). The `== 0` this test
        # shipped with pinned the global-branch counting regression it was
        # introduced alongside; before the scoped global model these two
        # writes were counted through the uncontrolled counter.
        assert result.payload["access_count"] == 2
        assert result.payload["accesses_complete"] is False

    inspected = numsim.Engine(max_workers=2).run_racecheck_phase(
        module, {"output": np.zeros(1, dtype=np.int32)}, inspect_accesses=True
    )
    assert inspected.verdict == "error"
    assert len(inspected.payload["accesses"]) == 2
    assert all(access["space"] == "global" for access in inspected.payload["accesses"])
    assert all(
        set(access)
        == {
            "operation",
            "access_kind",
            "space",
            "logical_buffer",
            "width",
            "active_lane_count",
            "lanes",
            "memory_order",
            "memory_scope",
            "memory_proxy",
            "memory_access_class",
        }
        for access in inspected.payload["accesses"]
    )


def test_native_racecheck_reports_same_cluster_global_waw(tmp_path):
    module = numsim.transpile(native_racecheck_same_cluster_global_waw, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run_racecheck_phase(
        module, {"output": np.zeros(1, dtype=np.int32)}
    )

    assert result.verdict == "error"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding["access_pair"] == "write_write"
    assert finding["prior"]["space"] == finding["current"]["space"] == "global"
    assert result.incomplete == []
    assert result.payload["checked_memory_spaces"] == ["global", "shared", "tmem"]
    # One lane-0 global write per warp, in parity with the journal. The `== 0`
    # this test shipped with pinned the global-branch counting regression it
    # was introduced alongside.
    assert result.payload["access_count"] == 2
    assert result.payload["accesses_complete"] is False


def test_native_racecheck_reports_cross_cluster_global_tile_copy_conflict(tmp_path):
    module = numsim.transpile(native_racecheck_cross_cluster_tile_copy_waw, cache_dir=tmp_path)
    inputs = {
        "source": np.ones(128, dtype=np.float32),
        "output": np.zeros(128, dtype=np.float32),
    }

    serial = numsim.Engine(max_workers=1).run_racecheck_phase(module, inputs)
    parallel = numsim.Engine(max_workers=2).run_racecheck_phase(module, inputs)

    assert serial.verdict == parallel.verdict == "error"
    assert serial.findings == parallel.findings
    for result in (serial, parallel):
        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding["access_pair"] == "write_write"
        assert finding["prior"]["space"] == finding["current"]["space"] == "global"
        assert result.incomplete == []
        assert result.payload["checked_memory_spaces"] == ["global", "shared", "tmem"]
        assert result.payload["access_count"] > 0


def test_native_racecheck_permute_snapshot_zero_fill_retains_shared_accesses(tmp_path):
    module = numsim.transpile(native_racecheck_shared_permute_snapshot_waw, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run_racecheck_phase(module, {})

    assert result.verdict == "error"
    assert result.incomplete == []
    finding = result.findings[-1]
    assert finding["access_pair"] == "write_write"
    assert finding["prior"]["space"] == finding["current"]["space"] == "shared"


def test_native_racecheck_permute_snapshot_records_zero_fill_source_reads(tmp_path):
    module = numsim.transpile(
        native_racecheck_shared_permute_snapshot_source_race, cache_dir=tmp_path
    )
    result = numsim.Engine(max_workers=1).run_racecheck_phase(module, {})

    assert result.verdict == "error"
    assert result.incomplete == []
    finding = result.findings[-1]
    assert finding["access_pair"] in {"read_write", "write_read"}
    assert finding["prior"]["space"] == finding["current"]["space"] == "shared"


def test_native_racecheck_records_cta_reduce_hidden_scratch_accesses(tmp_path):
    module = numsim.transpile(native_racecheck_cta_reduce_hidden_scratch_waw, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run_racecheck_phase(
        module, {"output": np.zeros(1, dtype=np.float32)}
    )

    assert result.verdict == "error"
    assert result.incomplete == []
    finding = result.findings[-1]
    assert finding["access_pair"] == "write_write"
    assert finding["prior"]["space"] == finding["current"]["space"] == "shared"


def test_native_racecheck_cta_reduce_internal_barriers_order_scratch_accesses(tmp_path):
    module = numsim.transpile(native_racecheck_cta_reduce_scratch_clean, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run_racecheck_phase(
        module, {"output": np.zeros(1, dtype=np.float32)}
    )

    assert result.verdict == "clean"
    assert result.findings == []
    assert result.incomplete == []


def test_native_racecheck_single_warp_cta_reduce_is_clean_and_numeric(tmp_path):
    numeric_module = numsim.transpile(
        native_racecheck_single_warp_cta_reduce,
        cache_dir=tmp_path,
        _analysis_checker=None,
    )
    racecheck_module = numsim.transpile(
        native_racecheck_single_warp_cta_reduce,
        cache_dir=tmp_path,
    )
    output = np.zeros(1, dtype=np.float32)
    numeric = numsim.Engine(max_workers=1).run(numeric_module, {"output": output})
    result = numsim.Engine(max_workers=1).run_racecheck_phase(
        racecheck_module, {"output": np.zeros(1, dtype=np.float32)}
    )

    assert numeric.outputs["output"][0] == np.float32(sum(range(32)))
    assert result.verdict == "clean"
    assert result.findings == []
    assert result.incomplete == []


def test_native_racecheck_cross_cluster_atomic_modification_order_is_clean(tmp_path):
    module = numsim.transpile(native_racecheck_cross_cluster_global_atomic, cache_dir=tmp_path)

    serial = numsim.Engine(max_workers=1).run_racecheck_phase(
        module, {"output": np.zeros(1, dtype=np.int32)}
    )
    parallel = numsim.Engine(max_workers=2).run_racecheck_phase(
        module, {"output": np.zeros(1, dtype=np.int32)}
    )

    assert serial.verdict == parallel.verdict == "clean"
    assert serial.findings == parallel.findings == []
    assert serial.incomplete == parallel.incomplete == []
