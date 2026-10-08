"""Kernel-level guard: synccheck must not fake a deadlock on a correct kernel.

Before the fix, `SemanticProgress::record_change` read `waiter_count` outside the
`waiters` mutex while `SemanticProgressWatch::poll` registered after its staleness
check, so a wake could be lost and a spinning warp stranded -- reported as a
confident `deadlock`.

Shape is dictated by what makes the bug reachable, all measured: the race only
exists ACROSS CTAs (warps of one CTA share an OS thread); a lost wake is only
permanent when no later `record_change` can rescue the waiter, hence a strict
two-party chain; the noise bursts keep the peer mid-registration when the
decisive flag write lands; and the jitter stops the two threads settling into a
rhythm.  Do not "simplify" these away -- without jitter this catches only 12/20
at 256 rounds, and at the default reschedule quantum of 64 it catches 0/20.

Unfixed base: 45/45 detected.  Fixed: 12/12 green, ~4s.
"""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T

# Ping-pong rounds and the maximum jittered noise burst per round.  Sized from
# measurement, not taste: 2048 x 64 caught the unfixed base 40/40 while costing
# well under a second of clean-tree wall time.
_ROUNDS = 2048
_BURST = 64


@T.prim_func
def native_lost_wake_pingpong(
    rounds: T.int32,
    burst: T.int32,
    flags: T.Buffer((2,), "int32"),
    noise: T.Buffer((2,), "int32"),
):
    """Two CTAs hand a ticket back and forth; each publishes then immediately spins.

    Correct by construction and terminating: CTA 0 publishes `flags[0] = t + 1`
    and then waits for `flags[1] >= t + 1`; CTA 1 waits for `flags[0] >= t + 1`
    and then publishes `flags[1] = t + 1`.  Any `deadlock` verdict on it is a
    false positive.
    """
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    ticket = T.local_scalar("int32")
    step = T.local_scalar("int32")
    limit = T.local_scalar("int32")
    seen = T.local_scalar("int32")

    if lane == 0:
        ticket = 0
        while ticket < rounds:
            # Jitter the burst so the two workers never lock into a rhythm in
            # which the peer is always fully parked before the flag write.
            limit = (ticket * 7 + cta * 3) % burst + 1
            if cta == 0:
                step = 0
                while step < limit:
                    noise[0] = ticket * 128 + step + 1
                    step = step + 1
                flags[0] = ticket + 1
                seen = flags[1]
                while seen < ticket + 1:
                    seen = flags[1]
            else:
                seen = flags[0]
                while seen < ticket + 1:
                    seen = flags[0]
                step = 0
                while step < limit:
                    noise[1] = ticket * 128 + step + 1
                    step = step + 1
                flags[1] = ticket + 1
            ticket = ticket + 1


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-lost-wake-pingpong")


def _resource_limits() -> numsim.ResourceLimits:
    return numsim.ResourceLimits(
        max_schedules=1,
        max_backtrack_nodes=1,
        max_events_per_run=1_000_000_000,
        max_total_events=1_000_000_000,
        max_loop_steps=1_000_000_000,
        max_wall_time_ms=300_000,
        max_diagnostic_bytes=8_000_000,
    )


def _coverage() -> numsim.CoverageBounds:
    return numsim.CoverageBounds(max_warp_preemptions=0, max_completion_schedule_deviations=0)


def _run(cache_dir):
    return synccheck(
        native_lost_wake_pingpong,
        inputs={
            "rounds": np.int32(_ROUNDS),
            "burst": np.int32(_BURST),
            "flags": np.zeros(2, dtype=np.int32),
            "noise": np.zeros(2, dtype=np.int32),
        },
        cache_dir=cache_dir,
        # Domains cap this at the two CTAs; the assertion below pins it.
        max_workers=8,
        # Park on every spin iteration.  At the default quantum of 64 the spin
        # rarely parks and the bug stops being reachable from this kernel.
        native_loop_reschedule_quantum=1,
        native_loop_iteration_budget=100_000_000,
        coverage_bounds=_coverage(),
        resource_limits=_resource_limits(),
    )


def test_native_synccheck_never_fakes_a_deadlock_on_a_tight_cross_cta_pingpong(native_cache_dir):
    report = _run(native_cache_dir)

    native = report.to_dict()["native"]
    stats = native["stats"]

    # The kernel is a correct, terminating ping-pong.  Anything other than a
    # clean verdict here is the lost semantic-progress wake resurfacing.
    execution_error = native["execution_error"]
    assert execution_error is None, (
        f"synccheck reported {execution_error['kind']} on a correct ping-pong: "
        f"{execution_error['message']} "
        f"(blocked_operations={execution_error['blocked_operations']})"
    )
    report.require_clean()
    assert report.verdict == "clean"
    assert report.findings == []
    assert native["verdict"] == "clean"
    assert native["findings"] == []
    assert native["incomplete"] == []
    assert native["coverage"]["status"] == "complete_within_bounds"
    assert native["coverage"]["eligible_for_clean"] is True

    # Preconditions for the race to be reachable at all.  If a future change
    # collapses this launch onto one worker thread, or stops the spin from
    # parking, the kernel would go on passing while covering nothing -- so fail
    # loudly instead.
    assert stats["available"] is True
    assert stats["task_count"] == 2
    assert stats["completed_task_count"] == 2
    assert stats["scheduling_domain_count"] == 2, (
        "the two CTAs must land in two scheduling domains, otherwise notifier and "
        "waiter never run on different threads"
    )
    assert stats["worker_count"] == 2, (
        "the launch must run on two executor worker threads for the lost-wake window to exist"
    )
    assert stats["poll_count"] > 10_000, (
        "the native while spins must actually park and re-poll; a tiny poll "
        "count means the ping-pong was not exercised"
    )
