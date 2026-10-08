from __future__ import annotations

import numpy as np

from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tvm.script import tirx as T


@T.prim_func
def native_unordered_atomic_plain(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    if (warp == 0) and (lane == 0):
        shared[0] = 0
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        old: T.let = T.cuda.atomic_add(shared.ptr_to([0]), T.int32(1))
        output[0] = old
    if (warp == 1) and (lane == 0):
        shared[0] = 7


@T.prim_func
def native_unordered_atomic_atomic(output: T.Buffer((2,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    if (warp == 0) and (lane == 0):
        shared[0] = 0
    T.cuda.cta_sync()
    if lane == 0:
        old: T.let = T.cuda.atomic_add(shared.ptr_to([0]), T.int32(1))
        output[warp] = old


@T.prim_func
def native_ptx_atom_cas_shared(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    slot = T.alloc_buffer((1,), "uint32", scope="shared")
    if lane == 0:
        slot[0] = T.uint32(0)
    T.cuda.cta_sync()
    if lane == 0:
        old = T.alloc_local((1,), "uint32")
        T.ptx.atom.relaxed.cta.shared.cas.b32(
            old[0], slot.ptr_to([0]), T.uint32(0), T.uint32(1)
        )
        output[0] = old[0]


@T.prim_func
def native_barrier_ordered_atomic_plain(output: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    if (warp == 0) and (lane == 0):
        shared[0] = 0
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        old: T.let = T.cuda.atomic_add(shared.ptr_to([0]), T.int32(1))
        output[0] = old
    T.ptx.bar.sync(T.uint32(5), T.uint32(64))
    if (warp == 1) and (lane == 0):
        output[0] = shared[0]


@T.prim_func
def native_reusable_named_barrier_orders_last_arriver(
    output: T.Buffer((64,), "int32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    if (warp == 0) and (lane == 0):
        shared[0] = -1
    T.cuda.cta_sync()

    for phase in T.serial(64):
        if (warp == 1) and (lane == 0):
            shared[0] = phase
        T.ptx.bar.sync(T.uint32(5), T.uint32(64))
        if (warp == 0) and (lane == 0):
            output[phase] = shared[0]
        T.ptx.bar.sync(T.uint32(6), T.uint32(64))


@T.prim_func
def native_long_loop_shadow_retirement():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "int32", scope="shared")
    if lane == 0:
        for step in T.serial(1_152):
            shared[step % 64] = step


def _run(kernel, output: np.ndarray, cache_dir):
    return racecheck(
        kernel,
        inputs={"output": output},
        cache_dir=cache_dir,
    )


def test_public_native_racecheck_reports_unordered_atomic_plain_conflict(tmp_path):
    report = _run(native_unordered_atomic_plain, np.zeros(1, dtype=np.int32), tmp_path)

    assert report.verdict == "error"
    assert len(report.findings) == 1
    assert report.findings[0].status == "error"
    assert report.findings[0].details["access_pair"] in {"write_read", "write_write"}
    native = report.to_dict()["native"]
    assert native["schema_version"] == 3
    assert native["execution_model"] == "direct_online_vc"
    assert native["incomplete"] == []
    assert "search" not in native
    assert "counterexample" not in native
    assert native["access_count"] > 0
    assert native["accesses_complete"] is False
    assert native["accesses"] == []
    assert len(native["findings"]) == 1
    finding = native["findings"][0]
    assert finding["access_pair"] in {"write_read", "write_write"}
    assert finding["ordering_domain"] == "memory"
    assert finding["ordering_failure"] == "missing_release_acquire"
    assert {finding["prior"]["access_kind"], finding["current"]["access_kind"]} == {
        "atomic_read_modify_write",
        "write",
    }
    assert finding["prior"]["space"] == "shared"
    assert finding["current"]["space"] == "shared"
    assert finding["prior"]["span"] == finding["current"]["span"]
    assert finding["overlap"] == finding["prior"]["span"]
    assert finding["overlap"]["byte_len"] == 4


def test_public_native_racecheck_atomic_modification_order_is_not_a_race(tmp_path):
    report = _run(native_unordered_atomic_atomic, np.zeros(2, dtype=np.int32), tmp_path)

    report.require_clean()
    assert report.verdict == "clean"
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert "search" not in native
    assert "counterexample" not in native
    assert native["access_count"] > 0
    assert native["stats"]["available"] is True


def test_public_native_racecheck_accepts_ptx_atom_cas(tmp_path):
    report = _run(native_ptx_atom_cas_shared, np.zeros(1, dtype=np.uint32), tmp_path)

    report.require_clean()
    assert report.verdict == "clean"
    assert report.findings == []
    assert report.to_dict()["native"]["incomplete"] == []


def test_public_native_racecheck_barrier_orders_atomic_before_plain_read(tmp_path):
    report = _run(native_barrier_ordered_atomic_plain, np.zeros(1, dtype=np.int32), tmp_path)

    report.require_clean()
    assert report.verdict == "clean"
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert "search" not in native
    assert "counterexample" not in native
    assert native["access_count"] > 0
    assert native["stats"]["available"] is True


def test_public_native_racecheck_named_barrier_publishes_last_arriver_before_resume(
    tmp_path,
):
    output = np.full(64, -1, dtype=np.int32)
    report = racecheck(
        native_reusable_named_barrier_orders_last_arriver,
        inputs={"output": output},
        cache_dir=tmp_path,
        max_workers=32,
    )

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["access_count"] > 0


def test_public_native_racecheck_long_loop_is_complete_and_deterministic(tmp_path):
    def run_once():
        return racecheck(
            native_long_loop_shadow_retirement,
            inputs={},
            cache_dir=tmp_path,
            max_workers=1,
        )

    first = run_once()
    second = run_once()

    for report in (first, second):
        report.require_clean()
        assert report.verdict == "clean"
        assert report.findings == []
        native = report.to_dict()["native"]
        assert native["incomplete"] == []
        assert "search" not in native
        assert "counterexample" not in native
        assert native["access_count"] == 1_152
        assert native["accesses_complete"] is False
        assert native["accesses"] == []
        assert native["stats"]["available"] is True

    first_native = first.to_dict()["native"]
    second_native = second.to_dict()["native"]
    assert first_native["access_count"] == second_native["access_count"]
    assert first_native["stats"]["task_count"] == second_native["stats"]["task_count"]
