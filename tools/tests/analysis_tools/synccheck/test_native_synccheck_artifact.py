"""Native Synccheck integration coverage."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.api import ExecutionSubset
from tirx_harness.numsim.checker_report import SyncCheckReport
from tirx_harness.numsim.checkers import _run_synccheck as internal_synccheck
from tests.numsim.support.kernels import (
    mapped_remote_mbarrier_pointer,
    mbarrier_missing_arrivals,
    tma_copy_transaction_mismatch,
)
from tests.numsim.support.remote_mbarrier import (
    mapped_remote_mbarrier_cluster_view,
    mapped_remote_mbarrier_pointer_expect_tx,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


@T.prim_func
def native_synccheck_clean():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


@T.prim_func
def native_synccheck_cta_sync_publishes_init():
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.cuda.cta_sync()
    if warp == 1 and lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


@T.prim_func
def native_synccheck_use_before_init():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))


@T.prim_func
def native_synccheck_wait_before_init():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


@T.prim_func
def native_synccheck_expect_tx_before_init():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)


@T.prim_func
def native_synccheck_plain_arrival_overflow():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    if lane < 2:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))


@T.prim_func
def native_synccheck_data_dependent_protocol(
    slot: T.int32, base_count: T.uint32, active_lanes: T.int32
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), base_count)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), base_count + 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    if lane < active_lanes:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[slot]))
    if lane == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[slot]), 0)


@T.prim_func
def native_synccheck_tma_two_generations(source: T.Buffer((2, 4), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "int32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        Tx.copy_async(shared[:], source[0, :], dispatch="tma_auto", mbar=T.address_of(barriers[0]))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        Tx.copy_async(shared[:], source[1, :], dispatch="tma_auto", mbar=T.address_of(barriers[0]))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 1)


@T.prim_func
def native_synccheck_late_wait_gates_next_completion(source: T.Buffer((4,), "int32")):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "int32", scope="shared")
    barriers = T.alloc_buffer((3,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[2]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[1]))
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 1)
    elif (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[2]))
    elif lane == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
        T.cuda.mbarrier_wait(T.address_of(barriers[2]), 0)
        Tx.copy_async(shared[:], source[:], dispatch="tma_auto", mbar=T.address_of(barriers[0]))


@T.prim_func
def native_synccheck_tma_completion_before_expect(source: T.Buffer((4,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "int32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        Tx.copy_async(shared[:], source[:], dispatch="tma_auto", mbar=T.address_of(barriers[0]))
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 1)


@T.prim_func
def native_synccheck_generation_reuse(consume: T.int32):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        if consume != 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        if consume != 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 1)


@T.prim_func
def native_synccheck_tma_over_delivery(source: T.Buffer((4,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "int32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
        Tx.copy_async(shared[:], source[:], dispatch="tma_auto", mbar=T.address_of(barriers[0]))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 8)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


@T.prim_func
def native_synccheck_schedule_sensitive_tma(
    source: T.Buffer((4,), "int32"), flag: T.Buffer((1,), "int32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "int32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    if warp == 0:
        if lane == 0:
            flag[0] = 1
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 2)
            T.ptx.fence.mbarrier_init.release.cluster()
            Tx.copy_async(shared[:], source[:], dispatch="tma_auto", mbar=T.address_of(barriers[0]))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    else:
        if lane == 0:
            if flag[0] == 0:
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[1]))
            else:
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))


@T.prim_func
def native_synccheck_nonblocking_arrive_exposes_uninitialized_successor():
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if warp == 0:
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.fence.mbarrier_init.release.cluster()
    elif warp == 1:
        T.ptx.bar.arrive(T.uint32(7), T.uint32(64))
        if lane == 0:
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    else:
        T.ptx.bar.arrive(T.uint32(7), T.uint32(64))


@T.prim_func
def native_synccheck_blocking_wait_exposes_uninitialized_successor():
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.bar.sync(T.uint32(7), T.uint32(96))
    if warp == 0:
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
            T.ptx.fence.mbarrier_init.release.cluster()
    elif warp == 1:
        if lane == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[1]))
    else:
        if lane == 0:
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))


@T.prim_func
def native_synccheck_no_waiter_arrival_exposes_uninitialized_successor():
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    if warp == 0 and lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.bar.sync(T.uint32(7), T.uint32(96))
    if warp == 0:
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
            T.ptx.fence.mbarrier_init.release.cluster()
    elif warp == 1:
        if lane == 0:
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[1]))


@T.prim_func
def native_synccheck_two_ctas():
    T.device_entry()
    _cta = T.cta_id([2])
    _warp = T.warp_id([1])


@T.prim_func
def native_synccheck_second_cluster_error():
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if cta == 0:
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.fence.mbarrier_init.release.cluster()
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    else:
        if lane == 0:
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))


@T.prim_func
def native_synccheck_named_barrier_contract_mismatch():
    T.device_entry()
    warp = T.warp_id([2])
    if warp == 0:
        T.ptx.bar.sync(T.uint32(3), T.uint32(64))
    else:
        T.ptx.bar.sync(T.uint32(3), T.uint32(32))


@T.prim_func
def native_synccheck_order_dependent_tcgen_allocations():
    T.device_entry()
    warp = T.warp_id([2])
    _lane = T.lane_id([32])
    addresses = T.alloc_buffer((2,), "uint32", scope="shared")

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(addresses[0]), 32
        )
    else:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(addresses[1]), 32
        )
    T.cuda.cta_sync()
    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(addresses[0], 32)
    else:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(addresses[1], 32)
    T.cuda.cta_sync()
    if warp == 0:
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


def _run_native_phase(module, inputs=None):
    return (
        numsim.Engine(
            max_workers=1,
            native_loop_iteration_budget=1_000,
            native_loop_reschedule_quantum=16,
        )
        .run_synccheck_phase(
            module,
            {} if inputs is None else inputs,
            phase_index=0,
            coverage_bounds=numsim.CoverageBounds(0, 0),
            resource_limits=_resource_limits(),
            max_polls=100,
            max_transitions=100,
        )
        .to_dict()
    )


def _transpile_analysis(func, *, cache_dir):
    return numsim.transpile(func, cache_dir=cache_dir, _analysis_capable=True)


def _resource_limits(*, max_schedules=100):
    return numsim.ResourceLimits(
        max_schedules=max_schedules,
        max_backtrack_nodes=10_000,
        max_events_per_run=10_000,
        max_total_events=100_000,
        max_loop_steps=100_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=1_000_000,
    )


def test_native_synccheck_generated_artifacts_default_to_o3():
    from tirx_harness.numsim import checker_runner

    assert checker_runner._NATIVE_ANALYSIS_DEFAULT_GENERATED_OPT_LEVEL == 3


def _assert_public_native_protocol_error(
    report: SyncCheckReport, *, kind: str, effect: str
) -> dict:
    assert report.verdict == "error"
    assert {finding.status for finding in report.findings} == {"error"}

    native = report.to_dict()["native"]
    assert native["verdict"] == "error"
    assert native["incomplete"] == []
    assert native["coverage"]["termination"]["kind"] == "finding"
    assert native["counterexample"] is None
    assert len(native["findings"]) == 1
    finding = native["findings"][0]
    assert finding["kind"] == kind
    assert finding["effect"] == effect
    assert finding["operation"]["kernel_index"] == 0
    assert finding["operation"]["global_warp_id"] == 0
    return finding


def test_native_synccheck_private_phase_returns_clean_primitive_trace(tmp_path):
    module = _transpile_analysis(native_synccheck_clean, cache_dir=tmp_path)

    result = _run_native_phase(module)

    assert result["schema_version"] == 3
    assert result["phase"] == {
        "index": 0,
        "name": "native_synccheck_clean",
        "topology": {"clusters": 1, "ctas_per_cluster": 1, "warps_per_cta": 1, "warp_count": 1},
    }
    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []
    assert [effect["effect"] for effect in result["effects"]] == [
        "mbarrier.init",
        "mbarrier.arrive",
        "mbarrier.wait",
    ]
    assert result["effects"][1]["outcome"] == {
        "kind": "mbarrier_arrive",
        "completed_generation": 0,
        "consumed_generation": None,
        "ready_warps": [],
    }
    assert result["effects"][2]["outcome"] == {
        "kind": "mbarrier_wait",
        "staged": {"kind": "ready", "generation": 0, "consumed_now": True},
        "committed": {"kind": "ready", "generation": 0, "consumed_now": True},
    }
    assert "controlled_trace" not in result
    assert result["stats"]["available"] is True
    assert result["stats"]["task_count"] == 1
    assert result["stats"]["completed_task_count"] == 1
    assert result["stats"]["poll_count"] == 1
    assert result["stats"]["normal_poll_count"] == 1
    assert result["stats"]["poll_recheck_poll_count"] == 0
    assert result["stats"]["completion_operation_count"] == 0
    assert result["stats"]["worker_count"] == 1
    assert result["stats"]["scheduling_domain_count"] == 1
    assert result["execution_error"] is None
    assert SyncCheckReport.from_native(result).verdict == "clean"
    assert result["resource_limits"] == {
        "max_polls": 100,
        "max_transitions": 100,
        "native_loop_iteration_budget": 1_000,
        "native_loop_reschedule_quantum": 16,
    }

    compact = numsim.Engine(max_workers=1).run_synccheck_phase(module, {})
    assert compact.verdict == "clean"
    assert "controlled_trace" not in compact.payload
    assert compact.payload["stats"]["poll_count"] == 1
    assert compact.payload["stats"]["normal_poll_count"] == 1
    with pytest.raises(TypeError, match="inspect_trace"):
        numsim.Engine().run_synccheck_phase(module, {}, inspect_trace=1)
    with pytest.raises(TypeError, match="choice_prefix"):
        numsim.Engine().run_synccheck_phase(module, {}, choice_prefix=[])

    bounded = numsim.Engine(max_workers=1).run_synccheck_phase(
        module, {}, max_polls=100, max_transitions=1
    )
    assert bounded.verdict == "clean"
    assert bounded.payload["resource_limits"]["max_transitions"] == 1


def test_public_synccheck_native_path_uses_fixed_sync_state_and_records_input_identity(tmp_path):
    report = internal_synccheck(
        native_synccheck_clean,
        inputs={},
        cache_dir=tmp_path,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "clean"
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["search"]["algorithm"] == "fixed_sync_state"
    assert native["search"]["run_count"] == 1
    assert native["search"]["backtrack_count"] == 0
    assert native["search"]["program_count"] == 1
    assert native["coverage"]["resource_usage"]["schedules"] == 1
    assert native["coverage"]["eligible_for_clean"] is True
    assert native["engine"]["mode"] == "native"
    assert native["engine"]["artifact_key"]
    assert native["input"]["digest"]
    assert native["input"]["bindings"] == []


def test_fixed_sync_state_finds_bad_tcgen_order_after_clean_native_execution(tmp_path):
    report = internal_synccheck(
        native_synccheck_order_dependent_tcgen_allocations,
        inputs={},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=numsim.CoverageBounds(0, 0),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["execution_error"] is None
    assert native["counterexample"] is None
    assert native["incomplete"] == []
    assert native["search"]["algorithm"] == "fixed_sync_state"
    assert native["search"]["run_count"] == 1
    # The fixed verifier checks the CTA barrier and TCGEN lifecycle as separate
    # protocol projections; the latter rejects the alternate allocation order.
    assert native["search"]["program_count"] == 2
    assert native["search"]["visited_state_count"] == 3
    assert native["search"]["explored_transition_count"] == 6
    assert len(native["findings"]) == 1
    finding = native["findings"][0]
    assert finding["kind"] == "fixed_sync_protocol_error"
    assert finding["protocol"] == "TcgenLifecycle"
    assert "allocation result changed" in finding["source"]
    assert finding["witness"]


def test_public_native_synccheck_wait_before_init_is_exact_error(tmp_path):
    report = internal_synccheck(
        native_synccheck_wait_before_init,
        inputs={},
        cache_dir=tmp_path,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    finding = _assert_public_native_protocol_error(
        report, kind="mbarrier_use_before_init", effect="mbarrier.wait"
    )
    assert "mbarrier.try_wait used uninitialized physical mbarrier" in finding["message"]


def test_public_native_synccheck_plain_arrive_before_init_is_exact_error(tmp_path):
    report = internal_synccheck(
        native_synccheck_use_before_init,
        inputs={},
        cache_dir=tmp_path,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    finding = _assert_public_native_protocol_error(
        report, kind="mbarrier_use_before_init", effect="mbarrier.arrive"
    )
    assert "mbarrier.arrive used uninitialized physical mbarrier" in finding["message"]


def test_public_native_synccheck_accepts_cta_sync_as_mbarrier_init_publication(tmp_path):
    report = internal_synccheck(
        native_synccheck_cta_sync_publishes_init,
        inputs={},
        cache_dir=tmp_path,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "clean", report.to_dict()
    assert report.findings == []


def test_public_native_synccheck_rejects_local_arrive_on_mapped_remote_address(tmp_path):
    report = internal_synccheck(
        mapped_remote_mbarrier_pointer,
        inputs={"output": np.zeros(2, dtype=np.int32)},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "error"
    assert [finding.kind for finding in report.findings] == ["mbarrier_local_arrive_remote_address"]
    native = report.to_dict()["native"]
    assert native["execution_error"]["kind"] == "mbarrier_local_arrive_remote_address"
    assert (
        "local-form mbarrier.arrive from global CTA 1 used an address mapped to remote global CTA 0"
    ) in native["execution_error"]["message"]


def test_public_native_synccheck_accepts_cluster_arrive_through_remote_view(tmp_path):
    report = internal_synccheck(
        mapped_remote_mbarrier_cluster_view,
        inputs={"output": np.zeros(2, dtype=np.int32)},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "clean", report.to_dict()
    assert report.findings == []


def test_public_native_synccheck_rejects_local_expect_tx_on_mapped_remote_address(tmp_path):
    report = internal_synccheck(
        mapped_remote_mbarrier_pointer_expect_tx,
        inputs={"output": np.zeros(2, dtype=np.int32)},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "error"
    assert [finding.kind for finding in report.findings] == ["mbarrier_local_arrive_remote_address"]
    native = report.to_dict()["native"]
    assert native["execution_error"]["kind"] == "mbarrier_local_arrive_remote_address"
    assert (
        "local-form mbarrier.arrive from global CTA 1 used an address mapped to remote global CTA 0"
    ) in native["execution_error"]["message"]


def test_public_native_synccheck_arrive_expect_tx_before_init_is_exact_error(tmp_path):
    report = internal_synccheck(
        native_synccheck_expect_tx_before_init,
        inputs={},
        cache_dir=tmp_path,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    finding = _assert_public_native_protocol_error(
        report, kind="mbarrier_use_before_init", effect="mbarrier.arrive"
    )
    assert "mbarrier.arrive.expect_tx used uninitialized physical mbarrier" in finding["message"]


def test_public_native_synccheck_plain_arrival_overflow_is_typed_error(tmp_path):
    report = internal_synccheck(
        native_synccheck_plain_arrival_overflow,
        inputs={},
        cache_dir=tmp_path,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    finding = _assert_public_native_protocol_error(
        report, kind="mbarrier_arrival_overflow", effect="mbarrier.arrive"
    )
    assert finding["operation"]["source_op_id"] > 0
    assert "received 2 arrivals, expected 1" in finding["message"]


def test_public_native_synccheck_executes_data_dependent_protocol_exactly(tmp_path):
    coverage = numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0)
    clean = internal_synccheck(
        native_synccheck_data_dependent_protocol,
        inputs={"slot": 1, "base_count": 1, "active_lanes": 2},
        cache_dir=tmp_path,
        coverage_bounds=coverage,
        resource_limits=_resource_limits(),
    )

    assert clean.verdict == "clean"
    assert clean.findings == []
    clean_native = clean.to_dict()["native"]
    assert clean_native["verdict"] == "clean"
    assert clean_native["findings"] == []
    assert clean_native["incomplete"] == []
    assert clean_native["counterexample"] is None
    assert clean_native["coverage"]["eligible_for_clean"] is True
    assert clean_native["input"]["bindings"] == ["active_lanes", "base_count", "slot"]
    assert clean_native["input"]["scalars"] == {
        "active_lanes": {"dtype": "int32", "value": {"kind": "integer", "decimal": "2"}},
        "base_count": {"dtype": "uint32", "value": {"kind": "integer", "decimal": "1"}},
        "slot": {"dtype": "int32", "value": {"kind": "integer", "decimal": "1"}},
    }

    error = internal_synccheck(
        native_synccheck_data_dependent_protocol,
        inputs={"slot": 0, "base_count": 1, "active_lanes": 2},
        cache_dir=tmp_path,
        coverage_bounds=coverage,
        resource_limits=_resource_limits(),
    )

    finding = _assert_public_native_protocol_error(
        error, kind="mbarrier_arrival_overflow", effect="mbarrier.arrive"
    )
    error_native = error.to_dict()["native"]
    assert error_native["input"]["scalars"] == {
        "active_lanes": {"dtype": "int32", "value": {"kind": "integer", "decimal": "2"}},
        "base_count": {"dtype": "uint32", "value": {"kind": "integer", "decimal": "1"}},
        "slot": {"dtype": "int32", "value": {"kind": "integer", "decimal": "0"}},
    }
    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]
    assert "byte_offset: 0" in finding["message"]
    assert "received 2 arrivals, expected 1" in finding["message"]

    repeated = internal_synccheck(
        native_synccheck_data_dependent_protocol,
        inputs={"slot": 0, "base_count": 1, "active_lanes": 2},
        cache_dir=tmp_path,
        coverage_bounds=coverage,
        resource_limits=_resource_limits(),
    )
    repeated_finding = _assert_public_native_protocol_error(
        repeated, kind="mbarrier_arrival_overflow", effect="mbarrier.arrive"
    )
    assert repeated.findings[0].id == error.findings[0].id
    assert repeated_finding == finding


def test_native_synccheck_private_phase_returns_strict_mbarrier_error(tmp_path):
    module = _transpile_analysis(native_synccheck_use_before_init, cache_dir=tmp_path)

    result = _run_native_phase(module)

    assert result["verdict"] == "error"
    assert result["incomplete"] == []
    assert result["effects"] == []
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["kind"] == "mbarrier_use_before_init"
    assert finding["effect"] == "mbarrier.arrive"
    assert finding["operation"]["kernel_index"] == 0
    assert finding["operation"]["global_warp_id"] == 0
    assert "used uninitialized physical mbarrier" in finding["message"]
    assert result["stats"]["available"] is True
    assert result["stats"]["completed_task_count"] == 0
    assert result["execution_error"]["kind"] == "engine_error"
    assert "synccheck rejected mbarrier.arrive" in result["execution_error"]["message"]
    report = SyncCheckReport.from_native(result)
    assert report.verdict == "error"
    assert [finding.kind for finding in report.findings] == ["mbarrier_use_before_init"]


def test_native_synccheck_deadlock_is_error_even_with_staged_waits(tmp_path):
    module = _transpile_analysis(mbarrier_missing_arrivals, cache_dir=tmp_path)

    result = _run_native_phase(module)

    assert result["verdict"] == "error"
    assert result["execution_error"]["kind"] == "deadlock"
    assert result["stats"]["available"] is True
    assert result["incomplete"] == []
    assert SyncCheckReport.from_native(result).verdict == "error"


def test_public_native_synccheck_handles_completion_issued_before_future_expectation(tmp_path):
    module = _transpile_analysis(native_synccheck_tma_completion_before_expect, cache_dir=tmp_path)
    source = np.arange(4, dtype=np.int32)
    result = _run_native_phase(module, {"source": source})

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []

    effect_names = [effect["effect"] for effect in result["effects"]]
    assert effect_names == [
        "mbarrier.init",
        "mbarrier.arrive",
        "mbarrier.completion_issue",
        "mbarrier.complete_tx",
        "mbarrier.wait",
        "mbarrier.arrive",
        "mbarrier.wait",
    ]
    issue = result["effects"][2]["outcome"]["actions"][0]
    completion = result["effects"][3]["outcome"]
    assert issue["generation"] == completion["generation"] == 1
    assert issue["transactions"] == completion["transactions"] == 16
    assert completion["completed_generation"] == 1
    assert completion["ready_warps"] == [0]
    arrives = [effect for effect in result["effects"] if effect["effect"] == "mbarrier.arrive"]
    waits = [effect for effect in result["effects"] if effect["effect"] == "mbarrier.wait"]
    assert [effect["outcome"]["completed_generation"] for effect in arrives] == [0, None]
    assert [effect["outcome"]["committed"]["generation"] for effect in waits] == [0, 1]

    coverage = numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=1)
    report = internal_synccheck(
        native_synccheck_tma_completion_before_expect,
        inputs={"source": source},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=coverage,
        resource_limits=_resource_limits(),
    )
    report.require_clean()
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["counterexample"] is None
    assert native["coverage"]["eligible_for_clean"] is True
    assert native["coverage"]["maximum_observed_usage"]["completion_schedule_deviations"] == 0


def test_native_synccheck_tracks_completion_tokens_across_future_generation(tmp_path):
    module = _transpile_analysis(native_synccheck_tma_two_generations, cache_dir=tmp_path)
    source = np.arange(8, dtype=np.int32).reshape(2, 4)

    result = _run_native_phase(module, {"source": source})

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []
    issues = [
        effect for effect in result["effects"] if effect["effect"] == "mbarrier.completion_issue"
    ]
    completions = [
        effect for effect in result["effects"] if effect["effect"] == "mbarrier.complete_tx"
    ]
    assert len(issues) == len(completions) == 2
    assert [effect["outcome"]["actions"][0]["generation"] for effect in issues] == [0, 1]
    assert [effect["outcome"]["actions"][0]["transactions"] for effect in issues] == [16, 16]
    assert [effect["outcome"]["generation"] for effect in completions] == [0, 1]
    assert [effect["outcome"]["completed_generation"] for effect in completions] == [0, 1]
    assert [effect["outcome"]["ready_warps"] for effect in completions] == [[0], [0]]
    assert SyncCheckReport.from_native(result).verdict == "clean"

    report = internal_synccheck(
        native_synccheck_tma_two_generations,
        inputs={"source": source},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=1
        ),
        resource_limits=_resource_limits(),
    )
    report.require_clean()
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["counterexample"] is None
    assert native["coverage"]["eligible_for_clean"] is True
    assert native["coverage"]["maximum_observed_usage"] == {
        "warp_preemptions": 0,
        "completion_schedule_deviations": 0,
    }
    assert native["search"]["algorithm"] == "fixed_sync_state"
    assert native["search"]["run_count"] == 1
    assert native["search"]["backtrack_count"] == 0


def test_native_synccheck_accepts_late_wait_ordered_before_next_completion(tmp_path):
    module = _transpile_analysis(
        native_synccheck_late_wait_gates_next_completion,
        cache_dir=tmp_path,
    )
    source = np.arange(4, dtype=np.int32)

    result = _run_native_phase(module, {"source": source})

    assert result["verdict"] == "clean"
    assert result["findings"] == []
    assert result["incomplete"] == []
    assert result["execution_error"] is None


def test_public_native_synccheck_rejects_unconsumed_generation_reuse(tmp_path):
    module = _transpile_analysis(native_synccheck_generation_reuse, cache_dir=tmp_path)
    clean_phase = _run_native_phase(module, {"consume": np.int32(1)})

    assert clean_phase["verdict"] == "clean"
    assert clean_phase["findings"] == []
    assert clean_phase["incomplete"] == []
    arrives = [effect for effect in clean_phase["effects"] if effect["effect"] == "mbarrier.arrive"]
    waits = [effect for effect in clean_phase["effects"] if effect["effect"] == "mbarrier.wait"]
    assert [effect["outcome"]["completed_generation"] for effect in arrives] == [0, 1]
    assert [effect["outcome"]["committed"]["generation"] for effect in waits] == [0, 1]
    assert all(effect["outcome"]["committed"]["consumed_now"] for effect in waits)

    coverage = numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0)
    clean = internal_synccheck(
        native_synccheck_generation_reuse,
        inputs={"consume": np.int32(1)},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=coverage,
        resource_limits=_resource_limits(),
    )
    clean.require_clean()
    clean_native = clean.to_dict()["native"]
    assert clean_native["incomplete"] == []
    assert clean_native["counterexample"] is None
    assert clean_native["coverage"]["eligible_for_clean"] is True

    error = internal_synccheck(
        native_synccheck_generation_reuse,
        inputs={"consume": np.int32(0)},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=coverage,
        resource_limits=_resource_limits(),
    )
    finding = _assert_public_native_protocol_error(
        error, kind="mbarrier_arrive_before_consumption", effect="mbarrier.arrive"
    )
    error_native = error.to_dict()["native"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]
    assert "before generation 0 was consumed" in finding["message"]


def test_public_native_synccheck_transaction_under_delivery_is_exact_error(tmp_path):
    source = np.arange(32, dtype=np.float16).reshape(4, 8)
    output = np.zeros((4, 8), dtype=np.float16)

    report = internal_synccheck(
        tma_copy_transaction_mismatch,
        inputs={"source": source, "output": output},
        cache_dir=tmp_path,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=1, max_completion_schedule_deviations=1
        ),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "error"
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("error", "deadlock")
    ]
    assert "transactions=64/68" in report.findings[0].message
    native = report.to_dict()["native"]
    assert native["verdict"] == "error"
    assert native["incomplete"] == []
    assert native["coverage"]["termination"]["kind"] == "finding"
    assert native["counterexample"] is None
    assert native["execution_error"]["kind"] == "deadlock"
    assert "transactions=64/68" in native["execution_error"]["message"]


def test_public_native_synccheck_transaction_over_delivery_is_exact_error(tmp_path):
    source = np.arange(4, dtype=np.int32)

    report = internal_synccheck(
        native_synccheck_tma_over_delivery,
        inputs={"source": source},
        cache_dir=tmp_path,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=1
        ),
        resource_limits=_resource_limits(),
    )

    finding = _assert_public_native_protocol_error(
        report, kind="mbarrier_transaction_over_delivery", effect="mbarrier.complete_tx"
    )
    assert "completed 16 transaction bytes, expected 8" in finding["message"]


def test_public_native_synccheck_executes_one_racy_control_path_without_replay(tmp_path):
    module = _transpile_analysis(native_synccheck_schedule_sensitive_tma, cache_dir=tmp_path)
    source = np.arange(4, dtype=np.int32)
    flag_array = np.zeros(1, dtype=np.int32)
    flag = flag_array
    engine = numsim.Engine()

    result = engine.run_synccheck_phase(
        module,
        {"source": source, "flag": flag},
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    assert result.verdict == "error"
    assert result.payload["search"]["algorithm"] == "fixed_sync_state"
    assert result.payload["search"]["run_count"] == 1
    assert result.payload["search"]["backtrack_count"] == 0
    assert result.payload["coverage"]["termination"]["kind"] == "finding"
    assert result.counterexample is None
    assert [finding["kind"] for finding in result.findings] == [
        "mbarrier_init_not_happens_before_use"
    ]
    assert result.payload["search"]["runs"] == [
        {
            "prefix": [],
            "warp_preemption_bound": 0,
            "trace_digest": None,
            "trace_digest_hex": None,
            "coverage_usage": {
                "warp_preemptions": 0,
                "completion_schedule_deviations": 0,
            },
            "status": "finding",
        }
    ]

    report = internal_synccheck(
        native_synccheck_schedule_sensitive_tma,
        inputs={"source": source, "flag": flag},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("error", "mbarrier_init_not_happens_before_use")
    ]
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["counterexample"] is None
    assert [finding["kind"] for finding in native["findings"]] == [
        "mbarrier_init_not_happens_before_use"
    ]
    np.testing.assert_array_equal(flag_array, np.zeros(1, dtype=np.int32))

    limited = engine.run_synccheck_phase(
        module,
        {"source": source, "flag": flag},
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(max_schedules=1),
    )
    assert limited.verdict == "error"
    assert limited.counterexample is None
    assert limited.payload["coverage"]["termination"]["kind"] == "finding"
    assert limited.payload["coverage"]["resource_usage"]["schedules"] == 1
    assert [finding["kind"] for finding in limited.findings] == [
        "mbarrier_init_not_happens_before_use"
    ]

    with pytest.raises(numsim.NumSimExecutionError, match="missing required bindings"):
        engine.run_synccheck_phase(
            module,
            {"flag": flag},
            coverage_bounds=numsim.CoverageBounds(
                max_warp_preemptions=0, max_completion_schedule_deviations=0
            ),
            resource_limits=_resource_limits(),
        )
    np.testing.assert_array_equal(flag_array, np.zeros(1, dtype=np.int32))


def test_synccheck_reports_nonblocking_arrival_successor_error_without_replay(tmp_path):
    report = internal_synccheck(
        native_synccheck_nonblocking_arrive_exposes_uninitialized_successor,
        inputs={},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=1, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["counterexample"] is None
    assert native["search"]["run_count"] == 1
    assert native["search"]["program_count"] == 0
    assert native["execution_error"]["kind"] == "engine_error"
    assert "synccheck rejected causal mbarrier.arrive" in native["execution_error"]["message"]
    assert [finding["kind"] for finding in native["findings"]] == [
        "mbarrier_init_not_happens_before_use"
    ]


def test_synccheck_reports_blocking_wait_handoff_successor_error_without_replay(tmp_path):
    report = internal_synccheck(
        native_synccheck_blocking_wait_exposes_uninitialized_successor,
        inputs={},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=2, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["counterexample"] is None
    assert native["search"]["run_count"] == 1
    assert native["search"]["program_count"] == 0
    assert native["execution_error"]["kind"] == "engine_error"
    assert "synccheck rejected causal mbarrier.arrive" in native["execution_error"]["message"]
    assert [finding["kind"] for finding in native["findings"]] == [
        "mbarrier_init_not_happens_before_use"
    ]


def test_synccheck_no_waiter_arrival_checks_its_successor_without_replay(tmp_path):
    report = internal_synccheck(
        native_synccheck_no_waiter_arrival_exposes_uninitialized_successor,
        inputs={},
        cache_dir=tmp_path,
        max_workers=1,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=2, max_completion_schedule_deviations=0
        ),
        resource_limits=_resource_limits(),
    )

    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["counterexample"] is None
    assert native["search"]["run_count"] == 1
    assert native["search"]["program_count"] == 0
    assert native["execution_error"]["kind"] == "engine_error"
    assert [finding["kind"] for finding in native["findings"]] == [
        "mbarrier_init_not_happens_before_use"
    ]


def test_native_synccheck_subset_is_typed_incomplete_for_fixed_run(tmp_path):
    module = _transpile_analysis(native_synccheck_two_ctas, cache_dir=tmp_path)
    engine = numsim.Engine()
    subset = ExecutionSubset(cta_ids=[0])

    single = engine.run_synccheck_phase(module, {}, subset=subset)
    assert single.verdict == "incomplete"
    assert single.payload["analysis_scope"] == {
        "kind": "subset",
        "selected_warp_count": 1,
        "total_warp_count": 2,
    }
    assert single.incomplete == [
        {"kind": "analysis_incomplete", "reason": "subset_execution", "selected_warp_count": 1, "total_warp_count": 2}
    ]
    assert single.payload["coverage"]["status"] == "complete_within_bounds"
    assert single.payload["search"]["algorithm"] == "fixed_sync_state"


def test_native_synccheck_full_launch_is_stable_across_worker_counts(tmp_path):
    module = _transpile_analysis(native_synccheck_two_ctas, cache_dir=tmp_path)

    serial = numsim.Engine(max_workers=1).run_synccheck_phase(module, {})
    parallel = numsim.Engine(max_workers=2).run_synccheck_phase(module, {})

    assert serial.verdict == parallel.verdict == "clean"
    for result, worker_count in ((serial, 1), (parallel, 2)):
        assert "cluster_parallel" not in result.payload
        assert result.payload["analysis_scope"]["kind"] == "full_launch"
        assert result.payload["stats"]["worker_count"] == worker_count
        assert result.payload["stats"]["task_count"] == 2
        assert result.payload["stats"]["completed_task_count"] == 2
        assert result.payload["stats"]["scheduling_domain_count"] == 2
    assert serial.findings == parallel.findings == []
    assert serial.payload["effects"] == parallel.payload["effects"]
    assert serial.payload["search"] == parallel.payload["search"]

    with pytest.raises(TypeError, match="choice_prefix"):
        numsim.Engine(max_workers=2).run_synccheck_phase(
            module,
            {},
            choice_prefix=[],
        )


def test_native_synccheck_fixed_verification_is_stable_across_worker_counts(tmp_path):
    module = _transpile_analysis(native_synccheck_two_ctas, cache_dir=tmp_path)
    bounds = numsim.CoverageBounds(0, 0)
    limits = _resource_limits()

    serial = numsim.Engine(max_workers=1).run_synccheck_phase(
        module, {}, coverage_bounds=bounds, resource_limits=limits
    )
    parallel = numsim.Engine(max_workers=2).run_synccheck_phase(
        module, {}, coverage_bounds=bounds, resource_limits=limits
    )

    assert serial.verdict == parallel.verdict == "clean"
    assert serial.payload["search"] == parallel.payload["search"]
    assert serial.payload["search"]["algorithm"] == "fixed_sync_state"
    assert serial.payload["search"]["run_count"] == 1
    assert serial.payload["search"]["backtrack_count"] == 0
    assert serial.payload["coverage"]["eligible_for_clean"] is True
    assert parallel.payload["coverage"]["eligible_for_clean"] is True
    assert serial.payload["effects"] == parallel.payload["effects"]


def test_native_synccheck_cluster_error_is_independent_of_worker_completion_order(tmp_path):
    module = _transpile_analysis(native_synccheck_second_cluster_error, cache_dir=tmp_path)
    serial = numsim.Engine(max_workers=1).run_synccheck_phase(module, {})
    parallel = numsim.Engine(max_workers=2).run_synccheck_phase(module, {})

    assert serial.verdict == parallel.verdict == "error"
    assert serial.findings == parallel.findings
    assert serial.payload["execution_error"] == parallel.payload["execution_error"]
    assert serial.findings
    assert {finding["operation"]["global_warp_id"] for finding in serial.findings} == {1}
    assert serial.counterexample is parallel.counterexample is None
    assert serial.payload["search"]["algorithm"] == "fixed_sync_state"
    assert serial.payload["search"]["run_count"] == 1


def test_native_synccheck_named_barrier_gateway_reports_execution_contract_error(tmp_path):
    module = _transpile_analysis(
        native_synccheck_named_barrier_contract_mismatch, cache_dir=tmp_path
    )

    result = numsim.Engine().run_synccheck_phase(module, {})

    assert result.verdict == "error"
    assert result.findings == []
    execution_error = result.payload["execution_error"]
    assert execution_error["kind"] == "synchronization_contract_mismatch"
    assert "warp 1 failed" in execution_error["message"]
    assert "participant contract changed" in execution_error["message"]
    assert "named_barrier[3]" in execution_error["message"]
