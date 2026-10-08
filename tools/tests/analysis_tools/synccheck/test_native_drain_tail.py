"""Shared native-analysis integration coverage for the public analysis tools."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T


@T.prim_func
def native_drain_tail_mbarrier(
    work_total: T.int32, probe_oob: T.int32, output: T.Buffer((1,), "int32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    ready = T.alloc_buffer((1,), "uint64", scope="shared")
    free = T.alloc_buffer((1,), "uint64", scope="shared")
    iteration = T.local_scalar("int32")
    phase = T.local_scalar("int32")

    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(ready[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(free[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        iteration = 0
        phase = 0
        while True:
            if iteration > 0:
                T.cuda.mbarrier_wait(T.address_of(free[0]), phase)
                phase = phase ^ 1
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(ready[0]))
            iteration = iteration + 1
            if iteration >= work_total:
                break
    elif (warp == 1) and (lane == 0):
        iteration = 0
        phase = 0
        while True:
            T.cuda.mbarrier_wait(T.address_of(ready[0]), phase)
            phase = phase ^ 1
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(free[0]))
            iteration = iteration + 1
            if iteration >= work_total:
                break

    T.cuda.cta_sync()
    if (probe_oob != 0) and (warp == 0) and (lane == 0):
        output[1] = 1


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-drain-tail-facade")


def _resource_limits() -> numsim.ResourceLimits:
    return numsim.ResourceLimits(
        max_schedules=100,
        max_backtrack_nodes=10_000,
        max_events_per_run=10_000,
        max_total_events=100_000,
        max_loop_steps=100_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=2_000_000,
    )


def _coverage() -> numsim.CoverageBounds:
    return numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0)


def _run(*, work_total: int, probe_oob: int, cache_dir):
    return synccheck(
        native_drain_tail_mbarrier,
        inputs={
            "work_total": np.int32(work_total),
            "probe_oob": np.int32(probe_oob),
            "output": np.zeros(1, dtype=np.int32),
        },
        cache_dir=cache_dir,
        max_workers=1,
        coverage_bounds=_coverage(),
        resource_limits=_resource_limits(),
    )


def _assert_clean(report, *, work_total: int) -> dict:
    report.require_clean()
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["verdict"] == "clean"
    assert native["findings"] == []
    assert native["incomplete"] == []
    assert native["counterexample"] is None
    assert native["coverage"]["eligible_for_clean"] is True
    assert native["input"]["scalars"]["work_total"]["value"]["decimal"] == str(work_total)
    return native


def _assert_probe_error(report) -> tuple[dict, dict]:
    assert report.verdict == "error"
    assert [(finding.status, finding.kind) for finding in report.findings] == [("error", "oob")]
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["counterexample"] is None
    assert native["findings"] == []
    assert native["execution_error"]["kind"] == "oob"
    return native, native


def _effect_frames_and_generations(
    replay: dict, *, effect_name: str, warp: int
) -> list[tuple[int, int]]:
    effects = [
        effect
        for effect in replay["effects"]
        if effect["effect"] == effect_name and effect["operation"]["global_warp_id"] == warp
    ]
    if not effects:
        return []
    assert len({effect["operation"]["source_op_id"] for effect in effects}) == 1
    pairs = []
    for effect in effects:
        frames = effect["operation"]["loop_frames"]
        assert len(frames) == 1
        if effect_name == "mbarrier.arrive":
            assert effect["outcome"]["kind"] == "mbarrier_arrive"
            generation = effect["outcome"]["completed_generation"]
        else:
            assert effect["outcome"]["kind"] == "mbarrier_wait"
            assert effect["outcome"]["committed"]["kind"] == "ready"
            generation = effect["outcome"]["committed"]["generation"]
        pairs.append((frames[0]["iteration_ordinal"], generation))
    return pairs


@pytest.mark.parametrize("work_total", [1, 3])
def test_public_native_drain_tail_has_exact_generations_and_no_phantom_arrive(
    native_cache_dir, work_total
):
    clean = _assert_clean(
        _run(work_total=work_total, probe_oob=0, cache_dir=native_cache_dir), work_total=work_total
    )
    error_native, replay = _assert_probe_error(
        _run(work_total=work_total, probe_oob=1, cache_dir=native_cache_dir)
    )
    assert clean["engine"]["artifact_key"] == error_native["engine"]["artifact_key"]
    assert clean["input"]["digest"] != error_native["input"]["digest"]
    assert error_native["input"]["scalars"]["work_total"]["value"]["decimal"] == str(work_total)

    generations = list(range(work_total))
    expected_all = list(enumerate(generations))
    expected_producer_wait = [(iteration, iteration - 1) for iteration in range(1, work_total)]

    assert (
        _effect_frames_and_generations(replay, effect_name="mbarrier.arrive", warp=0)
        == expected_all
    )
    assert (
        _effect_frames_and_generations(replay, effect_name="mbarrier.wait", warp=0)
        == expected_producer_wait
    )
    assert (
        _effect_frames_and_generations(replay, effect_name="mbarrier.wait", warp=1) == expected_all
    )
    assert (
        _effect_frames_and_generations(replay, effect_name="mbarrier.arrive", warp=1)
        == expected_all
    )

    mbarrier_effects = [
        effect for effect in replay["effects"] if effect["effect"].startswith("mbarrier.")
    ]
    assert len(mbarrier_effects) == 2 + work_total * 3 + max(work_total - 1, 0)
