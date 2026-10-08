"""Native Synccheck coverage for structured break and continue lowering."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T


@T.prim_func
def native_break_continue_facade_control(
    mode: T.int32,
    limit: T.int32,
    skip_odd: T.int32,
    sync_enabled: T.int32,
    probe_oob: T.int32,
    inner_limits: T.Buffer((3,), "int32"),
    output: T.Buffer((6,), "int32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    step = T.local_scalar("int32")
    outer = T.local_scalar("int32")
    inner = T.local_scalar("int32")
    emitted = T.local_scalar("int32")

    if (sync_enabled != 0) or (warp == 0):
        if mode == 0:
            step = 0
            while True:
                if step >= limit:
                    break
                if sync_enabled != 0:
                    T.ptx.bar.sync(T.uint32(11), T.uint32(64))
                if (warp == 0) and (lane == 0):
                    output[step] = step
                step = step + 1
        elif mode == 1:
            outer = 0
            while outer < limit:
                inner = 0
                while True:
                    if inner >= inner_limits[outer]:
                        break
                    if sync_enabled != 0:
                        T.ptx.bar.sync(T.uint32(11), T.uint32(64))
                    if (warp == 0) and (lane == 0):
                        output[outer * 2 + inner] = outer * 2 + inner
                    inner = inner + 1
                outer = outer + 1
        else:
            step = 0
            emitted = 0
            while step < limit:
                step = step + 1
                if (skip_odd != 0) and (step % 2 == 1):
                    continue
                if sync_enabled != 0:
                    T.ptx.bar.sync(T.uint32(11), T.uint32(64))
                if (warp == 0) and (lane == 0):
                    output[emitted] = step
                emitted = emitted + 1
    if (probe_oob != 0) and (warp == 0) and (lane == 0):
        output[6] = 1


@T.prim_func
def native_data_dependent_for_break(
    extent: T.Buffer((1,), "int32"),
    break_after: T.int32,
    probe_oob: T.int32,
    output: T.Buffer((4,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    for iteration in T.serial(extent[0]):
        T.ptx.bar.sync(T.uint32(12), T.uint32(32))
        if lane == 0:
            output[iteration] = iteration
        if iteration >= break_after:
            break
    if (probe_oob != 0) and (lane == 0):
        output[4] = 1


@T.prim_func
def native_lane_divergent_break():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    iteration = T.local_scalar("int32")
    iteration = 0

    while iteration < 6:
        T.ptx.bar.sync(T.uint32(13), T.uint32(32))
        iteration = iteration + 1
        if (lane == 0) and (iteration >= 3):
            break


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-break-continue-facades")


def _resource_limits() -> numsim.ResourceLimits:
    return numsim.ResourceLimits(
        max_schedules=100,
        max_backtrack_nodes=10_000,
        max_events_per_run=10_000,
        max_total_events=100_000,
        max_loop_steps=100_000,
        max_wall_time_ms=30_000,
        max_diagnostic_bytes=1_000_000,
    )


def _coverage() -> numsim.CoverageBounds:
    return numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0)


def _inputs(
    *,
    mode: int,
    limit: int,
    sync_enabled: int,
    probe_oob: int = 0,
    skip_odd: int = 0,
    inner_limits: tuple[int, int, int] = (0, 0, 0),
) -> dict:
    return {
        "mode": np.int32(mode),
        "limit": np.int32(limit),
        "skip_odd": np.int32(skip_odd),
        "sync_enabled": np.int32(sync_enabled),
        "probe_oob": np.int32(probe_oob),
        "inner_limits": np.asarray(inner_limits, dtype=np.int32),
        "output": np.zeros(6, dtype=np.int32),
    }


def _run_synccheck(*, cache_dir, **input_kwargs):
    return synccheck(
        native_break_continue_facade_control,
        inputs=_inputs(sync_enabled=1, **input_kwargs),
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


def _assert_oob_error(report) -> dict:
    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert [(finding.status, finding.kind) for finding in report.findings] == [("error", "oob")]
    assert native["counterexample"] is None
    replay = native
    assert replay["verdict"] == "error"
    assert replay["incomplete"] == []
    assert replay["execution_error"]["kind"] == "oob"
    return replay


def _assert_deadlock_error(report) -> dict:
    assert report.verdict == "error"
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("error", "warp_collective_divergence")
    ]
    assert native["counterexample"] is None
    replay = native
    assert replay["verdict"] == "error"
    assert replay["incomplete"] == []
    assert replay["execution_error"]["kind"] == "warp_collective_divergence"
    assert "requires all 32 lanes" in replay["execution_error"]["message"]
    assert "mask 0xfffffffe" in replay["execution_error"]["message"]
    return replay


def _barrier_frames(payload: dict) -> list[tuple[int, ...]]:
    sync = payload.get("sync", payload)
    effects = [
        effect
        for effect in sync["effects"]
        if effect["effect"] == "bar.sync.register" and effect["operation"]["global_warp_id"] == 0
    ]
    assert len({effect["operation"]["source_op_id"] for effect in effects}) == 1
    return [
        tuple(frame["iteration_ordinal"] for frame in effect["operation"]["loop_frames"])
        for effect in effects
    ]


def test_public_native_while_break_has_no_phantom_post_break_effects(native_cache_dir):
    clean = _assert_clean(_run_synccheck(cache_dir=native_cache_dir, mode=0, limit=3))
    assert clean["input"]["scalars"]["limit"]["value"]["decimal"] == "3"

    probe = _assert_oob_error(
        _run_synccheck(cache_dir=native_cache_dir, mode=0, limit=3, probe_oob=1)
    )
    assert _barrier_frames(probe) == [(0,), (1,), (2,)]


def test_public_native_nested_break_exits_only_inner_loop(native_cache_dir):
    kwargs = {"mode": 1, "limit": 3, "inner_limits": (2, 2, 2)}
    clean = _assert_clean(_run_synccheck(cache_dir=native_cache_dir, **kwargs))
    assert clean["input"]["scalars"]["limit"]["value"]["decimal"] == "3"

    probe = _assert_oob_error(_run_synccheck(cache_dir=native_cache_dir, **kwargs, probe_oob=1))
    assert _barrier_frames(probe) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]


def test_public_native_continue_skips_barrier_for_that_iteration(native_cache_dir):
    kwargs = {"mode": 2, "limit": 12, "skip_odd": 1}
    clean = _assert_clean(_run_synccheck(cache_dir=native_cache_dir, **kwargs))
    assert clean["input"]["scalars"]["skip_odd"]["value"]["decimal"] == "1"

    probe = _assert_oob_error(_run_synccheck(cache_dir=native_cache_dir, **kwargs, probe_oob=1))
    assert _barrier_frames(probe) == [(1,), (3,), (5,), (7,), (9,), (11,)]


def test_public_native_data_dependent_for_break_has_no_phantom_iterations(native_cache_dir):
    def inputs(probe_oob: int) -> dict:
        return {
            "extent": np.array([7], dtype=np.int32),
            "break_after": np.int32(2),
            "probe_oob": np.int32(probe_oob),
            "output": np.zeros(4, dtype=np.int32),
        }

    kwargs = {
        "cache_dir": native_cache_dir,
        "max_workers": 1,
        "coverage_bounds": _coverage(),
        "resource_limits": _resource_limits(),
    }
    _assert_clean(synccheck(native_data_dependent_for_break, inputs=inputs(0), **kwargs))
    probe = _assert_oob_error(
        synccheck(native_data_dependent_for_break, inputs=inputs(1), **kwargs)
    )
    assert _barrier_frames(probe) == [(0,), (1,), (2,)]


def test_public_native_lane_divergent_break_is_exact_collective_error(native_cache_dir):
    replay = _assert_deadlock_error(
        synccheck(
            native_lane_divergent_break,
            inputs={},
            cache_dir=native_cache_dir,
            max_workers=1,
            coverage_bounds=_coverage(),
            resource_limits=_resource_limits(),
        )
    )
