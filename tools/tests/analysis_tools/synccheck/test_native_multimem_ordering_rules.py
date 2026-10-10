"""Native Synccheck over the NVLS message-passing litmus: the cross-rank flag
barrier completes exactly when every rank's arrival counts toward the value
each rank waits for, whatever the memory ordering of arrivals and polls.

Racecheck's twin, `test_native_multimem_ordering_rules`, holds every ordering
counterexample here to its races; Synccheck must still prove them live.
"""

from __future__ import annotations

import pytest

from tests.numsim.support import multimem_litmus as ml
from tirx_harness.numsim.checkers import _run_synccheck as synccheck

WORLDS = (2, 4)


@pytest.fixture(scope="module")
def litmus_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-multimem-ordering-sync")


def _mp(world: int, cache_dir, **modes):
    return synccheck(ml.message_passing, ml.message_passing_inputs(world, **modes),
                     cache_dir=cache_dir, native_loop_iteration_budget=4096)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [
    {},
    {"arrive": "release_gpu"},
    {"arrive": "fence_cta_relaxed"},
    {"arrive": "relaxed"},
    {"gpu_rank": 1},
    {"wait": "acquire_gpu"},
    {"wait": "relaxed"},
    {"wait": "weak_fence_sys"},
    {"wait": "weak_red_fence_sys"},
    {"wait": "wait_until_gpu"},
    {"alias": "none"},
    {"sync_before": 0},
    {"sync_after": 0},
    {"arrive": "fence_sys_relaxed", "late_store": 1},
    {"wait": "relaxed_fence_sys", "early_read": 1},
], ids=lambda modes: "-".join(f"{k}={v}" for k, v in modes.items()) or "release_acquire_sys")
def test_ordering_counterexamples_still_complete(modes, world, litmus_cache):
    """Scope, strength, proxy and cumulativity decide what a completed wait
    orders, not whether it completes: every `multimem.red` still adds to every
    replica, so each rank's flag reaches `world`."""
    report = _mp(world, litmus_cache, **modes)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [
    pytest.param({"silent_rank": 0}, id="first_rank_never_arrives"),
    pytest.param({"silent_rank": 1}, id="peer_never_arrives"),
    pytest.param({"arrive_slot": 1}, id="arrivals_on_another_word"),
    # multimem.st writes the same value to every replica rather than
    # combining with it, so `world` arrivals leave the flag at 1.
    pytest.param({"arrive": "store_release_sys"}, id="store_instead_of_reduction"),
])
def test_flag_that_never_reaches_world_deadlocks_every_rank(modes, world, litmus_cache):
    report = _mp(world, litmus_cache, **modes)

    assert report.verdict == "error"
    [finding] = report.findings
    assert finding.kind == "deadlock"
    assert f"blocked warps 0-{world - 1}" in report.format()


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("op,message", [
    # 8.2.3: only multimem operations are valid on a multimem address...
    ("plain_load", "only multimem operations may access it"),
    ("plain_store", "only multimem operations may access it"),
    ("unicast_red", "only multimem operations may access it"),
    # ...and a multimem operation needs one.
    ("multimem_st_unicast", "is not in a multicast window"),
    ("multimem_red_unicast", "is not in a multicast window"),
])
def test_address_kind_misuse_is_a_memory_error(op, message, world, litmus_cache):
    report = synccheck(ml.concurrent_writes, ml.concurrent_write_inputs(world, op),
                       cache_dir=litmus_cache)

    assert report.verdict == "error"
    assert message in report.format()


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("read_first", [0, 1], ids=["raw", "war"])
@pytest.mark.parametrize("alias", [0, 1], ids=["unbridged", "bridged"])
def test_multicast_broadcast_barrier_completes(alias, read_first, world, litmus_cache):
    report = synccheck(ml.broadcast,
                       ml.broadcast_inputs(world, read_first=read_first, alias=alias),
                       cache_dir=litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []
