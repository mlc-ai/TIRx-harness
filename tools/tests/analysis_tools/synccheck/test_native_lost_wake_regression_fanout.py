"""Kernel-level guard: synccheck must not fake a deadlock on a correct kernel.

Companion to test_native_lost_wake_regression_pingpong.py, attacking the same
lost wake from the other side: a crowd of watchers re-registering at once against
`record_change`'s unlocked `waiter_count` fast path.

A six-CTA token ring, one warp and one worker each.  Every store is the sole
enabler of all further progress, so a single lost wake quiesces the launch
instead of being rescued by an unrelated write -- that is what makes detection
certain rather than probabilistic.

Unfixed base: 20/20 detected.  Fixed: 10/10 green, ~13s.
"""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T

# One warp per CTA: every participant becomes its own cluster, and therefore
# its own executor worker thread.
_RELAY_WARPS = 6
# Relay steps per participant.  Each step is one park/wake registration window
# per waiting warp; the count is what turns a rare interleaving into a
# reliable verdict.
_RELAY_ROUNDS = 60_000
# Every native `while` quantum must reach a checkpoint, so that a stuttering
# spin parks (and re-registers) instead of burning iterations locally.
_RELAY_QUANTUM = 1


@T.prim_func
def native_lost_wake_relay_fanout(
    rounds: T.int32,
    participants: T.int32,
    token: T.Buffer((1,), "int32"),
):
    T.device_entry()
    cta = T.cta_id([_RELAY_WARPS])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        for step in T.serial(rounds):
            # Participant `cta` owns relay step `step * participants + cta`.
            target: T.let = step * participants + cta
            observed = T.int32(-1)
            while observed < target:
                T.ptx.ld.volatile.global_.s32(observed, token.ptr_to([0]))
            T.ptx.st.volatile.global_.s32(token.ptr_to([0]), target + 1)


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-lost-wake-fanout")


def _resource_limits() -> numsim.ResourceLimits:
    steps = _RELAY_ROUNDS * _RELAY_WARPS
    return numsim.ResourceLimits(
        max_schedules=1,
        max_backtrack_nodes=16,
        max_events_per_run=1_000 * steps,
        max_total_events=2_000 * steps,
        max_loop_steps=1_000 * steps,
        max_wall_time_ms=300_000,
        max_diagnostic_bytes=2_000_000,
    )


def _coverage_bounds() -> numsim.CoverageBounds:
    return numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0)


def _run_relay(cache_dir):
    return synccheck(
        native_lost_wake_relay_fanout,
        inputs={
            "rounds": np.int32(_RELAY_ROUNDS),
            "participants": np.int32(_RELAY_WARPS),
            "token": np.zeros(1, dtype=np.int32),
        },
        cache_dir=cache_dir,
        # One worker per cluster: the relay must cross real threads.
        max_workers=_RELAY_WARPS,
        coverage_bounds=_coverage_bounds(),
        resource_limits=_resource_limits(),
        native_loop_reschedule_quantum=_RELAY_QUANTUM,
        native_loop_iteration_budget=1_000 * _RELAY_ROUNDS * _RELAY_WARPS,
    )


def test_public_native_synccheck_never_fakes_a_deadlock_on_a_parked_relay(
    native_cache_dir,
):
    """A correct cross-CTA relay must never be reported as deadlocked.

    A lost semantic-progress wake surfaces here as `verdict="error"` carrying a
    `deadlock` finding; the assertions below name that shape explicitly so a
    regression is self-describing rather than a bare `require_clean` failure.
    """

    report = _run_relay(native_cache_dir)

    findings = [(finding.status, finding.kind) for finding in report.findings]
    assert findings == [], (
        "Synccheck reported findings on a correct cross-CTA token relay "
        f"({findings}); a `deadlock` finding here means a semantic-progress "
        "wake was lost and a live warp was stranded on a stale generation."
    )
    assert report.verdict == "clean"
    report.require_clean()

    native = report.to_dict()["native"]
    assert native["verdict"] == "clean"
    assert native["findings"] == []
    assert native["incomplete"] == []
    assert native["counterexample"] is None
    assert native["execution_error"] is None
    assert native["coverage"]["status"] == "complete_within_bounds"
    assert native["coverage"]["eligible_for_clean"] is True

    # Pin the shape that gives this test its detection power.  Shrinking the
    # relay, collapsing it onto one worker, or raising the reschedule quantum
    # all silently turn it back into a test that passes with the bug present.
    scalars = native["input"]["scalars"]
    assert scalars["rounds"]["value"]["decimal"] == str(_RELAY_ROUNDS)
    assert scalars["participants"]["value"]["decimal"] == str(_RELAY_WARPS)

    stats = native["stats"]
    assert stats["available"] is True
    # Every participant really was its own cluster on its own worker thread.
    assert stats["worker_count"] == _RELAY_WARPS
    assert stats["scheduling_domain_count"] == _RELAY_WARPS
    # Every relay warp ran to completion: none was left parked on a stale
    # generation.
    assert stats["task_count"] == _RELAY_WARPS
    assert stats["completed_task_count"] == _RELAY_WARPS
    # One normal poll per relay step, plus the park/recheck cycles that carry
    # the registration windows this test exists to cover.  A measured run
    # yields ~1.7M of the latter; the floor keeps a future edit from quietly
    # collapsing the window count.
    assert stats["normal_poll_count"] >= _RELAY_ROUNDS * _RELAY_WARPS
    assert stats["poll_recheck_poll_count"] >= 500_000
