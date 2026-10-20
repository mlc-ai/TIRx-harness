"""Native Racecheck against the PTX memory consistency model across NVLS ranks.

Each rule of the PTX ISA's "Memory Consistency Model" chapter that a
multi-rank multimem protocol depends on is exercised by its passing form and
by every way of breaking it, on `multimem_litmus` kernels with one warp per
rank. A counterexample asserts the exact set of (writer rank, reader rank,
16-byte vector) pairs the rule leaves unordered, not merely an error verdict.
"""

from __future__ import annotations

import itertools

import pytest

from tests.numsim.support import multimem_litmus as ml
from tirx_harness.numsim.checkers import _run_racecheck as racecheck

WORLDS = (2, 4)
VECTORS = range(ml.LANES)


@pytest.fixture(scope="module")
def litmus_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-multimem-ordering")


def _mp(world: int, cache_dir, **modes):
    report = racecheck(ml.message_passing, inputs=ml.message_passing_inputs(world, **modes),
                       cache_dir=cache_dir)
    assert report.native_payload["incomplete"] == []
    return report


def _rank(access) -> int:
    return access["operation"]["global_warp_id"]


def _data_races(report) -> list[dict]:
    """Races on the 16-byte data vectors, leaving out races on the flag word."""
    return [
        finding for finding in report.native_payload["findings"]
        if finding["kind"] == "data_race" and finding["overlap"]["byte_len"] == 16
    ]


def _pairs(races) -> set[tuple[int, int, int]]:
    """(writer rank, reader rank, vector) of each race."""
    pairs = set()
    for finding in races:
        sides = (finding["prior"], finding["current"])
        writer = next(side for side in sides if side["access_kind"] == "write")
        reader = next(side for side in sides if side is not writer)
        pairs.add((_rank(writer), _rank(reader), finding["overlap"]["byte_offset"] // 16))
    return pairs


def _cross(world: int, vectors=VECTORS) -> set[tuple[int, int, int]]:
    return {(w, r, v) for w, r in itertools.permutations(range(world), 2) for v in vectors}


def _every(world: int, vectors=VECTORS) -> set[tuple[int, int, int]]:
    return {(w, r, v) for w in range(world) for r in range(world) for v in vectors}


def _mismatches(report) -> list[dict]:
    return [f for f in report.native_payload["findings"] if f["kind"] == "scope_mismatch"]


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [
    {},
    # 8.8: a release fence followed by a strong write is a release pattern...
    {"arrive": "fence_sys_relaxed"},
    # ...and a strong read followed by an acquire fence is an acquire pattern.
    {"wait": "relaxed_fence_sys"},
    {"wait": "wait_until_sys"},
    # 8.9.5: the alias fence may sit anywhere on the base causality path.
    {"alias": "after_wait"},
], ids=lambda modes: "-".join(f"{k}={v}" for k, v in modes.items()) or "release_acquire_sys")
def test_sys_release_acquire_with_alias_fence_on_the_path_is_clean(modes, world, litmus_cache):
    report = _mp(world, litmus_cache, **modes)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes,side,scope", [
    ({"arrive": "release_gpu"}, "release", "gpu"),
    # 8.9.4: the *first* operation of the release pattern -- the fence -- must
    # be morally strong with the acquire, so a `.sys` red does not rescue it.
    ({"arrive": "fence_gpu_relaxed"}, "release", "gpu"),
    ({"arrive": "fence_cta_relaxed"}, "release", "cta"),
    ({"wait": "acquire_gpu"}, "acquire", "gpu"),
    # ...and the *last* operation of the acquire pattern.
    ({"wait": "relaxed_fence_gpu"}, "acquire", "gpu"),
    ({"wait": "wait_until_gpu"}, "acquire", "gpu"),
], ids=lambda value: value if isinstance(value, str) else "-".join(map(str, value.values())))
def test_scope_short_of_the_peer_rank_orders_no_cross_rank_pair(modes, side, scope, world,
                                                                 litmus_cache):
    """8.5, 8.7: a scope names the threads an operation can synchronize with;
    only `.sys` names threads on another device."""
    report = _mp(world, litmus_cache, **modes)

    assert report.verdict == "error"
    races = _data_races(report)
    assert _pairs(races) == _cross(world)
    assert {f["ordering_failure"] for f in races} == {"missing_inter_actor_sync"}
    mismatches = _mismatches(report)
    assert mismatches and all(f["actor_relation"] == "cross_rank" for f in mismatches)
    assert any(f[f"{side}_scope"] == scope for f in mismatches)


@pytest.mark.parametrize("world", WORLDS)
def test_one_gpu_scoped_arrival_breaks_the_flag_rmw_chain(world, litmus_cache):
    """8.9.2: observation order through the flag's reductions needs every link
    morally strong. Rank 1's `.gpu` red is not, with any other rank: its own
    tile reaches no peer, and a peer's release it follows in the chain is cut."""
    report = _mp(world, litmus_cache, gpu_rank=1)

    assert report.verdict == "error"
    pairs = _pairs(_data_races(report))
    assert {(1, r, v) for r in range(world) if r != 1 for v in VECTORS} <= pairs
    assert pairs <= _cross(world)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [
    {"arrive": "relaxed"},
    # A relaxed CAS reads the arrivals but forms no acquire pattern.
    {"wait": "relaxed"},
    # 8.8: an acquire pattern starts with a *strong* read; a weak poll is not one.
    {"wait": "weak_fence_sys"},
    # 8.11.1: the read inside a reduction does not form an acquire pattern.
    {"wait": "weak_red_fence_sys"},
    {"wait": "none"},
    # 8.9.4: synchronizes-with needs the acquire to read the release's write;
    # a flag already at `world` is read without observing any arrival.
    {"arrive_slot": 1, "flag_init": None},
], ids=lambda modes: "-".join(f"{k}={v}" for k, v in modes.items()))
def test_no_release_acquire_pattern_orders_no_cross_rank_pair(modes, world, litmus_cache):
    modes = {**modes, "flag_init": world} if "flag_init" in modes else modes
    report = _mp(world, litmus_cache, **modes)

    assert report.verdict == "error"
    races = _data_races(report)
    assert _pairs(races) == _cross(world)
    assert {f["ordering_failure"] for f in races} == {"missing_inter_actor_sync"}
    assert _mismatches(report) == []


@pytest.mark.parametrize("world", WORLDS)
def test_weak_flag_poll_races_the_arrivals(world, litmus_cache):
    """8.7.1: a weak read is not morally strong with the reds it polls."""
    report = _mp(world, litmus_cache, wait="weak_fence_sys")

    flag = [f for f in report.native_payload["findings"]
            if f["kind"] == "data_race" and f["overlap"]["byte_len"] == 4]
    assert flag and all(f["prior"]["space"] == "global" for f in flag)


@pytest.mark.parametrize("world", WORLDS)
def test_store_after_the_release_fence_is_not_published(world, litmus_cache):
    """8.8: a release pattern orders only what precedes its first instruction.
    Lane 0 stores its vector between the fence and the red."""
    report = _mp(world, litmus_cache, arrive="fence_sys_relaxed", late_store=1)

    races = _data_races(report)
    assert _pairs(races) == _every(world, [0])
    for finding in races:
        same = _rank(finding["prior"]) == _rank(finding["current"])
        # Lane 0's own read follows the store in program order but not the
        # alias fence, which preceded the store.
        assert finding["ordering_failure"] == (
            "missing_proxy_bridge" if same else "missing_inter_actor_sync"
        )


@pytest.mark.parametrize("world", WORLDS)
def test_read_before_the_acquire_fence_is_not_acquired(world, litmus_cache):
    """8.8: an acquire pattern orders only what follows its last instruction.
    Lane 0 reduces its vector between the relaxed poll and the fence."""
    report = _mp(world, litmus_cache, wait="relaxed_fence_sys", early_read=1)

    races = _data_races(report)
    assert _pairs(races) == _cross(world, [0])
    assert {f["ordering_failure"] for f in races} == {"missing_inter_actor_sync"}


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("alias", ["none", "before_stores", "after_reduce"])
def test_alias_fence_off_the_causality_path_bridges_nothing(alias, world, litmus_cache):
    """8.9.5: aliases are ordered only by an alias proxy fence *along the base
    causality path*; one before the writes or after the reads is not on it,
    so even a rank's own vector is unbridged."""
    report = _mp(world, litmus_cache, alias=alias)

    races = _data_races(report)
    assert _pairs(races) == _every(world)
    for finding in races:
        assert finding["ordering_failure"] == "missing_proxy_bridge"
        bridge = finding["proxy_bridge"]
        assert {bridge["prior_proxy"], bridge["current_proxy"]} == {"generic", "multicast_alias"}


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("barrier", ["sync_before", "sync_after"])
def test_lanes_off_the_releasing_lane_need_a_barrier(barrier, world, litmus_cache):
    """8.9.5: cumulativity is transitivity of base causality order. Without
    `bar.warp.sync` before the arrival the other lanes' stores precede no
    release; without it after the wait their reads follow no acquire."""
    report = _mp(world, litmus_cache, **{barrier: 0})

    races = _data_races(report)
    assert _pairs(races) == _cross(world, range(1, ml.LANES))
    assert {f["ordering_failure"] for f in races} == {"missing_inter_actor_sync"}


def _writes(world: int, op: str, cache_dir):
    return racecheck(ml.concurrent_writes, inputs=ml.concurrent_write_inputs(world, op),
                     cache_dir=cache_dir)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("op", [
    "st_relaxed_sys", "red_relaxed_sys",
    # The model treats strong `.sys` accesses through a multicast address and
    # through the replica's unicast address as one location's atomics, which
    # the NVLS flag idiom (multimem.red published, unicast acquire polled)
    # depends on.
    "st_relaxed_sys_and_unicast",
])
def test_morally_strong_concurrent_writes_do_not_race(op, world, litmus_cache):
    report = _writes(world, op, litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
def test_weak_multimem_stores_from_every_rank_race(world, litmus_cache):
    """8.7.1: conflicting writes that are neither morally strong nor ordered."""
    report = _writes(world, "st_weak", litmus_cache)

    findings = report.native_payload["findings"]
    assert findings and all(
        f["kind"] == "data_race" and f["access_pair"] == "write_write"
        and _rank(f["prior"]) != _rank(f["current"])
        for f in findings
    )
    assert {f["overlap"]["byte_offset"] // 4 for f in findings} == set(VECTORS)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("op", ["st_relaxed_gpu", "red_relaxed_gpu"])
def test_gpu_scoped_multimem_writes_are_not_morally_strong_across_ranks(op, world, litmus_cache):
    """8.7: strong, but each `.gpu` scope misses the other rank's thread."""
    report = _writes(world, op, litmus_cache)

    assert report.verdict == "error"
    findings = report.native_payload["findings"]
    assert findings and all(
        (f["kind"], f["release_scope"], f["acquire_scope"], f["actor_relation"])
        == ("scope_mismatch", "gpu", "gpu", "cross_rank")
        for f in findings
    )


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("op", ["plain_load", "plain_store", "unicast_red"])
def test_non_multimem_access_to_a_multimem_address_is_a_memory_error(op, world, litmus_cache):
    """8.2.3: only multimem.* operations are valid on a multimem address."""
    report = _writes(world, op, litmus_cache)

    assert report.verdict == "error"
    assert "only multimem operations may access it" in report.format()


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("op", ["multimem_st_unicast", "multimem_red_unicast"])
def test_multimem_access_to_a_unicast_address_is_a_memory_error(op, world, litmus_cache):
    """The converse: a multimem operation's address must be a multimem address."""
    report = _writes(world, op, litmus_cache)

    assert report.verdict == "error"
    assert "is not in a multicast window" in report.format()


def _broadcast(world: int, cache_dir, *, read_first: int, alias: int):
    return racecheck(ml.broadcast,
                     inputs=ml.broadcast_inputs(world, read_first=read_first, alias=alias),
                     cache_dir=cache_dir)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("read_first", [0, 1], ids=["raw", "war"])
def test_multicast_publication_read_through_unicast_needs_the_alias_fence(read_first, world,
                                                                          litmus_cache):
    """8.9.5 in the other direction: `multimem.st` writes every replica through
    the multicast alias, so a unicast read after it (RAW) or before it (WAR)
    is ordered by the `.sys` barrier only with `fence.proxy.alias` on the path."""
    clean = _broadcast(world, litmus_cache, read_first=read_first, alias=1)
    assert clean.verdict == "clean", clean.format()

    report = _broadcast(world, litmus_cache, read_first=read_first, alias=0)
    races = _data_races(report)
    expected = {(v % world, r, v) for r in range(world) for v in VECTORS}
    assert _pairs(races) == expected
    for finding in races:
        assert finding["ordering_failure"] == "missing_proxy_bridge"
        assert finding["access_pair"] in ("write_read", "read_write")
