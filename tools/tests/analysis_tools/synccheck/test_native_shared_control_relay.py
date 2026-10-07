"""Native Synccheck coverage for shared-control relay kernels."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


@T.prim_func
def native_shared_control_relay(selected_branch: T.int32):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    relay = T.alloc_buffer((1,), "int32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    if (warp == 0) and (lane == 0):
        relay[0] = selected_branch
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([0]))
    if (warp == 1) and (lane == 0):
        branch: T.let = relay[0]
        if branch == 0:
            T.cuda.mbarrier_wait(barriers.ptr_to([0]), 0)
        else:
            T.cuda.mbarrier_wait(barriers.ptr_to([1]), 0)


@T.prim_func
def native_alternate_producer_relay(producer_warp: T.int32, selected_branch: T.int32):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    relay = T.alloc_buffer((1,), "int32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    if warp == producer_warp:
        if lane == 0:
            relay[0] = selected_branch
            T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([0]), 1)
            T.ptx.fence.mbarrier_init.release.cluster()
            T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([0]))
    T.cuda.cta_sync()

    if (warp == 2) and (lane == 0):
        branch: T.let = relay[0]
        if branch == 0:
            T.cuda.mbarrier_wait(barriers.ptr_to([0]), 0)
        else:
            T.cuda.mbarrier_wait(barriers.ptr_to([1]), 0)


@T.prim_func
def native_transitive_shared_relay(selected_branch: T.int32):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    first = T.alloc_buffer((1,), "int32", scope="shared")
    second = T.alloc_buffer((1,), "int32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    if (warp == 0) and (lane == 0):
        first[0] = selected_branch
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([0]))
    if (warp == 1) and (lane == 0):
        second[0] = first[0]
    T.cuda.cta_sync()

    if (warp == 2) and (lane == 0):
        branch: T.let = second[0]
        if branch == 0:
            T.cuda.mbarrier_wait(barriers.ptr_to([0]), 0)
        else:
            T.cuda.mbarrier_wait(barriers.ptr_to([1]), 0)


@T.prim_func
def native_conflicting_multi_writer_relay(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    relay = T.alloc_buffer((1,), "int32", scope="shared")

    if (warp < 2) and (lane == 0):
        relay[0] = warp + 2
    T.cuda.cta_sync()

    if (warp == 2) and (lane == 0):
        output[0] = relay[0]


@T.prim_func
def native_atomic_winner_relay(lock: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    relay = T.alloc_buffer((1,), "int32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    if warp == 2:
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([0]), 1)
            T.ptx.fence.mbarrier_init.release.cluster()
            T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([0]))
    T.cuda.cta_sync()

    if (warp < 2) and (lane == 0):
        old: T.let = T.cuda.atomic_cas(lock.ptr_to([0]), T.int32(0), T.int32(1))
        if old == 0:
            relay[0] = warp
    T.cuda.cta_sync()

    if (warp == 2) and (lane == 0):
        branch: T.let = relay[0]
        if branch == 0:
            T.cuda.mbarrier_wait(barriers.ptr_to([0]), 0)
        else:
            T.cuda.mbarrier_wait(barriers.ptr_to([1]), 0)


@T.prim_func
def native_tma_shared_control_relay(source: T.Buffer((4,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    relay = T.alloc_buffer((4,), "int32", scope="shared")
    barriers = T.alloc_buffer((3,), "uint64", scope="shared")

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([0]), 1)
        T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([1]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        Tx.copy_async(relay[:], source[:], dispatch="tma_auto", mbar=barriers.ptr_to([0]))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barriers.ptr_to([0]), 16)
        T.ptx.mbarrier.arrive.shared.b64(barriers.ptr_to([1]))

    if (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(barriers.ptr_to([0]), 0)
        branch: T.let = relay[0]
        if branch == 0:
            T.cuda.mbarrier_wait(barriers.ptr_to([1]), 0)
        else:
            T.cuda.mbarrier_wait(barriers.ptr_to([2]), 0)


def _resource_limits() -> numsim.ResourceLimits:
    return numsim.ResourceLimits(
        max_schedules=16,
        max_backtrack_nodes=100,
        max_events_per_run=1_000,
        max_total_events=10_000,
        max_loop_steps=1_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=1_000_000,
    )


def _relay_resource_limits() -> numsim.ResourceLimits:
    return numsim.ResourceLimits(
        max_schedules=128,
        max_backtrack_nodes=1_000,
        max_events_per_run=1_000,
        max_total_events=100_000,
        max_loop_steps=10_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=4_000_000,
    )


def _coverage_bounds() -> numsim.CoverageBounds:
    return numsim.CoverageBounds(max_warp_preemptions=1, max_completion_schedule_deviations=0)


def _tma_coverage_bounds() -> numsim.CoverageBounds:
    return numsim.CoverageBounds(max_warp_preemptions=1, max_completion_schedule_deviations=1)


def _run_synccheck(selected_branch: int, cache_dir):
    return synccheck(
        native_shared_control_relay,
        inputs={"selected_branch": selected_branch},
        cache_dir=cache_dir,
        coverage_bounds=_coverage_bounds(),
        resource_limits=_resource_limits(),
    )


def _run_native(tool, kernel, inputs: dict[str, int], cache_dir):
    return tool(
        kernel,
        inputs=inputs,
        cache_dir=cache_dir,
        coverage_bounds=_coverage_bounds(),
        resource_limits=_relay_resource_limits(),
    )


def _run_tma_native(tool, selected_branch: int, cache_dir):
    return tool(
        native_tma_shared_control_relay,
        inputs={"source": np.array([selected_branch, 11, 12, 13], dtype=np.int32)},
        cache_dir=cache_dir,
        coverage_bounds=_tma_coverage_bounds(),
        resource_limits=_relay_resource_limits(),
    )


def _assert_exact_clean(report) -> dict:
    report.require_clean()
    assert report.verdict == "clean"
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["verdict"] == "clean"
    assert native["findings"] == []
    assert native["incomplete"] == []
    if "search" in native:
        assert native["counterexample"] is None
        assert native["coverage"]["eligible_for_clean"] is True
    else:
        assert "counterexample" not in native
        assert native["execution_error"] is None
        assert native["stats"]["available"] is True
    return native


def test_public_native_synccheck_uses_cross_warp_shared_value_for_exact_branch(tmp_path):
    clean = _run_synccheck(0, tmp_path)
    error = _run_synccheck(1, tmp_path)

    clean_native = _assert_exact_clean(clean)

    assert error.verdict == "error"
    assert [(finding.status, finding.kind) for finding in error.findings] == [
        ("error", "mbarrier_use_before_init")
    ]
    error_native = error.to_dict()["native"]
    assert error_native["incomplete"] == []
    assert error_native["counterexample"] is None
    assert len(error_native["findings"]) == 1
    finding = error_native["findings"][0]
    assert finding["kind"] == "mbarrier_use_before_init"
    assert finding["effect"] == "mbarrier.wait"
    assert finding["operation"]["global_warp_id"] == 1
    assert "PhysicalBarrierId" in finding["message"]
    assert "target_global_cta_id: 0" in finding["message"]

    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]
    assert clean_native["input"]["scalars"]["selected_branch"]["value"]["decimal"] == "0"
    assert error_native["input"]["scalars"]["selected_branch"]["value"]["decimal"] == "1"


def _assert_exact_sync_error(report, *, consumer_warp: int) -> dict:
    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("error", "mbarrier_use_before_init")
    ]
    assert native["counterexample"] is None
    replay = native
    assert replay["verdict"] == "error"
    assert replay["incomplete"] == []
    assert len(replay["findings"]) == 1
    finding = replay["findings"][0]
    assert finding["kind"] == "mbarrier_use_before_init"
    assert finding["effect"] == "mbarrier.wait"
    assert finding["operation"]["global_warp_id"] == consumer_warp
    assert "PhysicalBarrierId" in finding["message"]
    assert "target_global_cta_id: 0" in finding["message"]
    return replay


@pytest.mark.parametrize("producer_warp", [0, 1])
def test_public_native_synccheck_resolves_alternate_producer_exactly(producer_warp, tmp_path):
    common = {"producer_warp": producer_warp}
    clean = _run_native(
        synccheck, native_alternate_producer_relay, {**common, "selected_branch": 0}, tmp_path
    )
    error = _run_native(
        synccheck, native_alternate_producer_relay, {**common, "selected_branch": 1}, tmp_path
    )

    clean_native = _assert_exact_clean(clean)
    _assert_exact_sync_error(error, consumer_warp=2)
    error_native = error.to_dict()["native"]
    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]
    assert clean_native["input"]["scalars"]["producer_warp"]["value"]["decimal"] == str(
        producer_warp
    )
    assert error_native["input"]["scalars"]["producer_warp"]["value"]["decimal"] == str(
        producer_warp
    )


def test_public_native_synccheck_resolves_transitive_relay_exactly(tmp_path):
    clean = _run_native(synccheck, native_transitive_shared_relay, {"selected_branch": 0}, tmp_path)
    error = _run_native(synccheck, native_transitive_shared_relay, {"selected_branch": 1}, tmp_path)

    clean_native = _assert_exact_clean(clean)
    _assert_exact_sync_error(error, consumer_warp=2)
    error_native = error.to_dict()["native"]
    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]


def test_public_native_synccheck_uses_concrete_atomic_selected_sync_trace(tmp_path):
    lock_array = np.zeros(1, dtype=np.int32)
    report = _run_native(
        synccheck,
        native_atomic_winner_relay,
        {"lock": lock_array},
        tmp_path,
    )

    _assert_exact_clean(report)
    native = report.to_dict()["native"]
    assert native["counterexample"] is None
    assert native["findings"] == []
    assert native["incomplete"] == []
    assert native["search"]["algorithm"] == "fixed_sync_state"
    # The legal concrete CAS order selects one numeric path; the fixed verifier
    # then checks every protocol interleaving of the recorded programs.
    assert native["search"]["program_count"] == 2
    np.testing.assert_array_equal(lock_array, np.zeros(1, dtype=np.int32))


def test_public_native_synccheck_uses_completed_tma_payload_for_exact_branch(tmp_path):
    clean = _run_tma_native(synccheck, 0, tmp_path)
    error = _run_tma_native(synccheck, 1, tmp_path)

    clean_native = _assert_exact_clean(clean)
    _assert_exact_sync_error(error, consumer_warp=1)
    error_native = error.to_dict()["native"]
    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]
