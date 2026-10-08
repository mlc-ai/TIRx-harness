"""Native Racecheck integration coverage."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T


@T.prim_func
def native_racecheck_arrive_snapshot(
    producer_warp: T.int32, overwrite: T.int32, output: T.Buffer((1,), "int32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == producer_warp:
        if lane == 0:
            shared[0] = 1
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))
            if overwrite != 0:
                shared[0] = 2
    else:
        if lane == 0:
            T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
            output[0] = shared[0]


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-arrive-snapshot")


def _resource_limits() -> numsim.ResourceLimits:
    return numsim.ResourceLimits(
        max_schedules=100,
        max_backtrack_nodes=10_000,
        max_events_per_run=10_000,
        max_total_events=100_000,
        max_loop_steps=10_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=1_000_000,
    )


def _inputs(*, producer_warp: int, overwrite: int) -> dict:
    return {
        "producer_warp": np.int32(producer_warp),
        "overwrite": np.int32(overwrite),
        "output": np.zeros(1, dtype=np.int32),
    }


def _coverage() -> numsim.CoverageBounds:
    return numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0)


def _run_race(*, producer_warp: int, overwrite: int, cache_dir):
    return racecheck(
        native_racecheck_arrive_snapshot,
        inputs=_inputs(producer_warp=producer_warp, overwrite=overwrite),
        cache_dir=cache_dir,
        max_workers=1,
    )


def _run_sync(*, producer_warp: int, cache_dir):
    return synccheck(
        native_racecheck_arrive_snapshot,
        inputs=_inputs(producer_warp=producer_warp, overwrite=1),
        cache_dir=cache_dir,
        max_workers=1,
        coverage_bounds=_coverage(),
        resource_limits=_resource_limits(),
    )


def _assert_clean_control(report) -> dict:
    report.require_clean()
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    return native


def _assert_overwrite_race(report, *, producer_warp: int) -> tuple[dict, dict, dict]:
    assert report.verdict == "error"
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("error", report.to_dict()["native"]["findings"][0]["kind"])
    ]
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    replay = native
    assert replay["execution_model"] == "direct_online_vc"
    assert replay["sync"]["findings"] == []
    assert replay["sync"]["incomplete"] == []
    assert len(replay["findings"]) == 1

    finding = replay["findings"][0]
    assert finding["access_pair"] in {"write_read", "read_write"}
    witnesses = (finding["prior"], finding["current"])
    write = next(item for item in witnesses if item["access_kind"] == "write")
    read = next(item for item in witnesses if item["access_kind"] == "read")
    assert write["operation"]["global_warp_id"] == producer_warp
    assert read["operation"]["global_warp_id"] == 1 - producer_warp
    assert write["lane"] == read["lane"] == 0
    assert write["space"] == read["space"] == "shared"
    assert write["span"] == read["span"] == finding["overlap"]
    assert finding["overlap"]["byte_len"] == 4
    return replay, write, read


def _mbarrier_effect(replay: dict, effect_name: str) -> dict:
    matches = [effect for effect in replay["sync"]["effects"] if effect["effect"] == effect_name]
    assert len(matches) == 1
    return matches[0]


def test_public_native_racecheck_late_wait_acquires_only_arrive_snapshot(native_cache_dir):
    clean = _run_race(producer_warp=1, overwrite=0, cache_dir=native_cache_dir)
    race = _run_race(producer_warp=1, overwrite=1, cache_dir=native_cache_dir)
    sync = _run_sync(producer_warp=1, cache_dir=native_cache_dir)

    clean_native = _assert_clean_control(clean)
    sync_native = _assert_clean_control(sync)
    replay, overwrite, read = _assert_overwrite_race(race, producer_warp=1)
    assert clean_native["input"]["digest"] != race.to_dict()["native"]["input"]["digest"]
    assert sync_native["coverage"]["eligible_for_clean"] is True

    arrive = _mbarrier_effect(replay, "mbarrier.arrive")
    wait = _mbarrier_effect(replay, "mbarrier.wait")
    assert wait["outcome"]["staged"] == {"kind": "ready", "generation": 0, "consumed_now": True}
    assert arrive["operation"]["global_warp_id"] == overwrite["operation"]["global_warp_id"]
    assert arrive["operation"]["per_warp_sequence"] < overwrite["operation"]["per_warp_sequence"]
    assert wait["operation"]["global_warp_id"] == read["operation"]["global_warp_id"]
    assert wait["operation"]["per_warp_sequence"] < read["operation"]["per_warp_sequence"]


def test_public_native_racecheck_blocked_wait_acquires_only_arrive_snapshot(native_cache_dir):
    clean = _run_race(producer_warp=0, overwrite=0, cache_dir=native_cache_dir)
    race = _run_race(producer_warp=0, overwrite=1, cache_dir=native_cache_dir)
    sync = _run_sync(producer_warp=0, cache_dir=native_cache_dir)

    _assert_clean_control(clean)
    _assert_clean_control(sync)
    replay, overwrite, read = _assert_overwrite_race(race, producer_warp=0)

    arrive = _mbarrier_effect(replay, "mbarrier.arrive")
    wait = _mbarrier_effect(replay, "mbarrier.wait")
    assert wait["outcome"]["staged"] == {"kind": "registered", "generation": 0}
    # The completing arrive consumes a generation with a pre-registered waiter;
    # committing the resumed wait must not consume it a second time.
    assert wait["outcome"]["committed"] == {
        "kind": "ready",
        "generation": 0,
        "consumed_now": False,
    }
    assert arrive["operation"]["global_warp_id"] == overwrite["operation"]["global_warp_id"]
    assert arrive["operation"]["per_warp_sequence"] < overwrite["operation"]["per_warp_sequence"]
    assert wait["operation"]["global_warp_id"] == read["operation"]["global_warp_id"]
    assert wait["operation"]["per_warp_sequence"] < read["operation"]["per_warp_sequence"]
