"""Native Synccheck over the unicast peer-memory litmus: every rank's wait on a
flag a peer sets through its symmetric-memory address completes exactly when
some peer's arrival lands on that flag, whatever the memory ordering.

Racecheck's twin, `test_native_peer_ordering_rules`, holds every ordering
counterexample here to its races; Synccheck must still prove them live.
"""

from __future__ import annotations

import pytest

from tests.numsim.support import peer_litmus as pl
from tirx_harness.numsim import NumSimExecutionError
from tirx_harness.numsim.checkers import _run_synccheck as synccheck

WORLDS = (2, 4)
PRIVATE = "which is not symmetric memory"


@pytest.fixture(scope="module")
def litmus_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-peer-ordering-sync")


def _check(kernel, inputs, cache_dir):
    return synccheck(kernel, inputs, cache_dir=cache_dir, native_loop_iteration_budget=4096)


def _ids(modes):
    return "-".join(f"{k}={v}" for k, v in modes.items()) or "default"


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [
    {},
    {"arrive": "release_gpu"},
    {"arrive": "fence_gpu_relaxed"},
    {"arrive": "relaxed"},
    {"arrive": "store_release_sys"},
    {"arrive": "weak_store"},
    {"wait": "acquire_gpu"},
    {"wait": "relaxed"},
    {"wait": "relaxed_fence_gpu"},
    {"wait": "wait_until_sys"},
    {"wait": "wait_until_gpu"},
    {"wait": "none"},
    {"flag_shift": 0},
    {"sync_before": 0},
    {"sync_after": 0},
    {"arrive": "fence_sys_relaxed", "late_store": 1},
    {"wait": "relaxed_fence_sys", "early_read": 1},
], ids=_ids)
def test_push_counterexamples_still_complete(modes, world, litmus_cache):
    """Scope, strength and cumulativity decide what a completed wait orders,
    not whether it completes: each rank's flag still receives one arrival."""
    report = _check(pl.push, pl.push_inputs(world, **modes), litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [
    {}, {"publish": "release_gpu"}, {"poll": "acquire_gpu"}, {"poll": "relaxed"},
    {"reuse": 1}, {"reuse": 1, "ack": "relaxed"}, {"reuse": 1, "ack": "none"},
], ids=_ids)
def test_pull_counterexamples_still_complete(modes, world, litmus_cache):
    report = _check(pl.pull, pl.pull_inputs(world, **modes), litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [
    {}, {"copy": 1}, {"proxy_fence": "none"},
    {"direction": 1}, {"direction": 1, "copy": 1}, {"direction": 1, "store_wait": "read"},
    {"direction": 1, "store_wait": "none"},
], ids=_ids)
def test_bulk_counterexamples_still_complete(modes, world, litmus_cache):
    report = _check(pl.bulk, pl.bulk_inputs(world, **modes), litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("silent", [0, 1])
def test_a_rank_that_never_arrives_deadlocks_its_receiver(silent, world, litmus_cache):
    """Rank ``silent`` never signals rank ``silent + 1``, which alone blocks."""
    report = _check(pl.push, pl.push_inputs(world, silent_rank=silent), litmus_cache)

    assert report.verdict == "error"
    [finding] = report.findings
    assert finding.kind == "deadlock"
    assert f"blocked warps {(silent + 1) % world}" in report.format()


@pytest.mark.parametrize("kernel,inputs", [
    (pl.push, lambda: pl.push_inputs(2, symmetric=False)),
    (pl.pull, lambda: pl.pull_inputs(2, symmetric=False)),
    (pl.atomics, lambda: pl.atomics_inputs(2, "atom_add_sys", symmetric=False)),
    (pl.bulk, lambda: pl.bulk_inputs(2, symmetric=False)),
    (pl.bulk, lambda: pl.bulk_inputs(2, direction=1, symmetric=False)),
], ids=["store", "load", "atom", "bulk_load", "bulk_store"])
def test_peer_access_to_private_memory_is_a_memory_error(kernel, inputs, litmus_cache):
    report = synccheck(kernel, inputs(), cache_dir=litmus_cache)

    assert report.verdict == "error"
    assert PRIVATE in report.format()


def test_tensor_map_over_a_peers_private_memory_is_refused_at_launch(litmus_cache):
    with pytest.raises(NumSimExecutionError, match="TensorMap 'tmap': .*" + PRIVATE):
        synccheck(pl.bulk, pl.bulk_inputs(2, direction=1, copy=1, tmap_symmetric=False),
                  cache_dir=litmus_cache)
