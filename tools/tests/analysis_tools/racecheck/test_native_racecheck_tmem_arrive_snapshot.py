"""Native Racecheck integration coverage."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tests.analysis_tools.racecheck._native_race_trace import full_direct_race_run
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

_TMEM_LAYOUT = TileLayout(S[(128, 64) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def native_tmem_arrive_snapshot(
    producer_warp: T.int32, load_after_arrive: T.int32, output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 2)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    tmem_base: T.let = address[0]
    tmem = T.decl_buffer(
        (128, 64), "uint32", scope="tmem", layout=_TMEM_LAYOUT, allocated_addr=tmem_base
    )

    if warp == producer_warp:
        tmem[lane, 0] = T.cast(7000 + lane, "uint32")
        if load_after_arrive == 0:
            output[lane] = tmem[lane, 0]
        # Publish every producer lane's pre-arrive accesses through lane 0.
        T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
        if lane == 0:
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
        if load_after_arrive != 0:
            output[lane] = tmem[lane, 0]
    else:
        if lane == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
        tmem[lane, 0] = T.cast(9000 + lane, "uint32")

    T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[1]))
    if warp == 0:
        if lane == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
        T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF))
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-tmem-arrive-snapshot")


def _resource_limits() -> numsim.ResourceLimits:
    return numsim.ResourceLimits(
        max_schedules=100,
        max_backtrack_nodes=10_000,
        max_events_per_run=10_000,
        max_total_events=100_000,
        max_loop_steps=10_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=2_000_000,
    )


def _coverage() -> numsim.CoverageBounds:
    return numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0)


def _inputs(*, producer_warp: int, load_after_arrive: int) -> dict:
    return {
        "producer_warp": np.int32(producer_warp),
        "load_after_arrive": np.int32(load_after_arrive),
        "output": np.zeros(32, dtype=np.uint32),
    }


def _run_race(*, producer_warp: int, load_after_arrive: int, cache_dir):
    return racecheck(
        native_tmem_arrive_snapshot,
        inputs=_inputs(producer_warp=producer_warp, load_after_arrive=load_after_arrive),
        cache_dir=cache_dir,
        max_workers=1,
    )


def _run_sync(*, producer_warp: int, cache_dir):
    return synccheck(
        native_tmem_arrive_snapshot,
        inputs=_inputs(producer_warp=producer_warp, load_after_arrive=1),
        cache_dir=cache_dir,
        max_workers=1,
        coverage_bounds=_coverage(),
        resource_limits=_resource_limits(),
    )


def _assert_clean(report) -> dict:
    report.require_clean()
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["verdict"] == "clean"
    assert native["findings"] == []
    assert native["incomplete"] == []
    return native


def _operation_key(operation: dict) -> tuple:
    return (
        operation["kernel_index"],
        operation["global_warp_id"],
        operation["per_warp_sequence"],
        operation["source_op_id"],
        tuple(
            (frame["loop_site_id"], frame["iteration_ordinal"])
            for frame in operation["loop_frames"]
        ),
    )


def _assert_same_warp_before(earlier: dict, later: dict) -> None:
    assert earlier["global_warp_id"] == later["global_warp_id"]
    assert earlier["per_warp_sequence"] < later["per_warp_sequence"]


def _mbarrier_effect(replay: dict, effect_name: str) -> dict:
    matches = [effect for effect in replay["sync"]["effects"] if effect["effect"] == effect_name]
    if effect_name == "mbarrier.arrive":
        matches = [effect for effect in matches if effect["outcome"]["completed_generation"] == 0]
    assert len(matches) == 1
    return matches[0]


def _assert_race_abort_sync_state(replay: dict) -> None:
    assert replay["sync"]["verdict"] == "clean"
    assert replay["sync"]["findings"] == []
    assert replay["sync"]["incomplete"] == []


def _assert_post_arrive_tmem_war(
    report, *, producer_warp: int, cache_dir
) -> tuple[dict, dict, dict]:
    assert report.verdict == "error"
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("error", report.to_dict()["native"]["findings"][0]["kind"])
    ]
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    _assert_race_abort_sync_state(native)
    assert len(native["findings"]) == 1
    replay = full_direct_race_run(
        native_tmem_arrive_snapshot,
        _inputs(producer_warp=producer_warp, load_after_arrive=1),
        cache_dir,
        native,
    )
    assert replay["incomplete"] == []
    _assert_race_abort_sync_state(replay)
    assert len(replay["findings"]) == 1

    finding = replay["findings"][0]
    compact_finding = native["findings"][0]
    assert {
        _operation_key(compact_finding["prior"]["operation"]),
        _operation_key(compact_finding["current"]["operation"]),
    } == {
        _operation_key(finding["prior"]["operation"]),
        _operation_key(finding["current"]["operation"]),
    }
    assert finding["access_pair"] in {"read_write", "write_read"}
    witnesses = (finding["prior"], finding["current"])
    read = next(item for item in witnesses if item["access_kind"] == "read")
    write = next(item for item in witnesses if item["access_kind"] == "write")
    assert read["operation"]["global_warp_id"] == producer_warp
    assert write["operation"]["global_warp_id"] == 1 - producer_warp
    assert read["space"] == write["space"] == "tmem"
    assert read["lane"] == write["lane"] == 0
    assert read["span"] == write["span"] == finding["overlap"]
    assert finding["overlap"]["byte_len"] == 4

    tmem_accesses = [access for access in replay["accesses"] if access["space"] == "tmem"]
    assert [access["access_kind"] for access in tmem_accesses] == ["write", "read"]
    assert all(access["width"] == 4 for access in tmem_accesses)
    assert all(access["active_lane_count"] == 32 for access in tmem_accesses)
    assert all(len(access["lanes"]) == 32 for access in tmem_accesses)
    initial_write, recorded_read = tmem_accesses
    assert initial_write["operation"]["global_warp_id"] == producer_warp
    assert _operation_key(recorded_read["operation"]) == _operation_key(read["operation"])
    assert initial_write["lanes"][0] == {"lane": 0, "spans": [finding["overlap"]]}
    assert recorded_read["lanes"][0] == {"lane": 0, "spans": [finding["overlap"]]}
    assert finding["current"] == write
    return replay, read, write


def test_public_native_tmem_post_arrive_load_races_with_late_consumer_store(native_cache_dir):
    clean = _assert_clean(
        _run_race(producer_warp=1, load_after_arrive=0, cache_dir=native_cache_dir)
    )
    race = _run_race(producer_warp=1, load_after_arrive=1, cache_dir=native_cache_dir)
    sync = _assert_clean(_run_sync(producer_warp=1, cache_dir=native_cache_dir))
    replay, read, write = _assert_post_arrive_tmem_war(
        race, producer_warp=1, cache_dir=native_cache_dir
    )

    assert clean["engine"]["artifact_key"] == race.to_dict()["native"]["engine"]["artifact_key"]
    assert sync["coverage"]["eligible_for_clean"] is True
    arrive = _mbarrier_effect(replay, "mbarrier.arrive")
    wait = _mbarrier_effect(replay, "mbarrier.wait")
    assert arrive["outcome"]["completed_generation"] == 0
    assert wait["outcome"]["staged"] == {"kind": "ready", "generation": 0, "consumed_now": True}
    assert wait["outcome"]["committed"]["generation"] == 0

    # The ready wait outcome proves arrive completed before wait. Program order
    # supplies the remaining per-warp edges without relying on schedule replay.
    _assert_same_warp_before(arrive["operation"], read["operation"])
    _assert_same_warp_before(wait["operation"], write["operation"])


def test_public_native_tmem_post_arrive_load_races_with_preblocked_consumer_store(native_cache_dir):
    _assert_clean(_run_race(producer_warp=0, load_after_arrive=0, cache_dir=native_cache_dir))
    race = _run_race(producer_warp=0, load_after_arrive=1, cache_dir=native_cache_dir)
    _assert_clean(_run_sync(producer_warp=0, cache_dir=native_cache_dir))
    replay, read, write = _assert_post_arrive_tmem_war(
        race, producer_warp=0, cache_dir=native_cache_dir
    )

    arrive = _mbarrier_effect(replay, "mbarrier.arrive")
    wait = _mbarrier_effect(replay, "mbarrier.wait")
    assert arrive["outcome"]["completed_generation"] == 0
    assert wait["outcome"]["staged"] == {"kind": "registered", "generation": 0}
    assert wait["outcome"]["committed"] == {
        "kind": "ready",
        "generation": 0,
        "consumed_now": False,
    }

    # The registered/committed outcomes prove wait blocked before arrive and
    # was released by it. Program order proves each warp's post-sync access.
    _assert_same_warp_before(arrive["operation"], read["operation"])
    _assert_same_warp_before(wait["operation"], write["operation"])
