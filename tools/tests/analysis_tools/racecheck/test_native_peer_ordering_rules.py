"""Native Racecheck against the PTX memory consistency model over unicast peer
memory.

Every rule a symmetric-memory protocol depends on -- a peer store or load
published or acquired across ranks, a peer's buffer reused after an
acknowledgement, concurrent peer atomics, and bulk/TMA copies to and from a
peer -- is exercised by its passing form and by the ways of breaking it, on
`peer_litmus` kernels with one warp per rank. A counterexample asserts the
exact set of (writer rank, reader rank, 16-byte vector) pairs the rule leaves
unordered, not merely an error verdict. Section numbers refer to the PTX ISA's
"Memory Consistency Model" chapter.
"""

from __future__ import annotations

import pytest

from tests.numsim.support import peer_litmus as pl
from tirx_harness.numsim import NumSimExecutionError
from tirx_harness.numsim.checkers import _run_racecheck as racecheck

WORLDS = (2, 4)
VECTORS = range(pl.LANES)
PRIVATE = "which is not symmetric memory"


@pytest.fixture(scope="module")
def litmus_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-peer-ordering")


def _run(kernel, inputs, cache_dir):
    report = racecheck(kernel, inputs=inputs, cache_dir=cache_dir)
    assert report.native_payload["incomplete"] == []
    return report


def _rank(access) -> int:
    return access["operation"]["global_warp_id"]


def _data_races(report) -> list[dict]:
    """Races on the 16-byte data vectors, leaving out races on flag words."""
    return [
        finding for finding in report.native_payload["findings"]
        if finding["kind"] == "data_race" and finding["overlap"]["byte_len"] == 16
    ]


def _writer_reader(finding) -> tuple[int, int]:
    """Ranks of a race's writing side (a store or an RMW) and the other side."""
    sides = (finding["prior"], finding["current"])
    writer = next(side for side in sides if side["access_kind"] != "read")
    reader = next(side for side in sides if side is not writer)
    return _rank(writer), _rank(reader)


def _pairs(races) -> set[tuple[int, int, int]]:
    """(writer rank, reader rank, vector) of each race. Each rank's buffers
    are 512-byte slots of one heap, so a vector is its offset's 16-byte slot."""
    return {(*_writer_reader(finding), finding["overlap"]["byte_offset"] // 16 % pl.LANES)
            for finding in races}


def _from_predecessor(world: int, vectors=VECTORS) -> set[tuple[int, int, int]]:
    """Rank ``r - 1`` writes what rank ``r`` reads."""
    return {((r - 1) % world, r, v) for r in range(world) for v in vectors}


def _from_successor(world: int, vectors=VECTORS) -> set[tuple[int, int, int]]:
    """Rank ``r + 1`` writes what rank ``r`` reads."""
    return {((r + 1) % world, r, v) for r in range(world) for v in vectors}


def _mismatches(report) -> list[dict]:
    return [f for f in report.native_payload["findings"] if f["kind"] == "scope_mismatch"]


def _failures(races) -> set[str]:
    return {f["ordering_failure"] for f in races}


def _ids(modes):
    return "-".join(f"{k}={v}" for k, v in modes.items()) or "default"


# --- push: peer stores, then a release on the receiver's flag -------------------

@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [
    {},
    # 8.8: a release fence followed by a strong write is a release pattern...
    {"arrive": "fence_sys_relaxed"},
    {"arrive": "store_release_sys"},
    # ...and a strong read followed by an acquire fence is an acquire pattern.
    {"wait": "relaxed_fence_sys"},
    {"wait": "wait_until_sys"},
], ids=_ids)
def test_sys_release_acquire_orders_peer_stores(modes, world, litmus_cache):
    report = _run(pl.push, pl.push_inputs(world, **modes), litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes,side,scope", [
    ({"arrive": "release_gpu"}, "release", "gpu"),
    # 8.9.4: the *first* operation of the release pattern -- the fence -- must
    # be morally strong with the acquire, so a `.sys` red does not rescue it.
    ({"arrive": "fence_gpu_relaxed"}, "release", "gpu"),
    ({"wait": "acquire_gpu"}, "acquire", "gpu"),
    # ...and the *last* operation of the acquire pattern.
    ({"wait": "relaxed_fence_gpu"}, "acquire", "gpu"),
    ({"wait": "wait_until_gpu"}, "acquire", "gpu"),
], ids=lambda value: value if isinstance(value, str) else _ids(value))
def test_scope_short_of_the_peer_rank_orders_no_peer_store(modes, side, scope, world,
                                                           litmus_cache):
    """8.5, 8.7: only `.sys` names threads on another device."""
    report = _run(pl.push, pl.push_inputs(world, **modes), litmus_cache)

    assert report.verdict == "error"
    races = _data_races(report)
    assert _pairs(races) == _from_predecessor(world)
    assert _failures(races) == {"missing_inter_actor_sync"}
    mismatches = _mismatches(report)
    assert mismatches and all(f["actor_relation"] == "cross_rank" for f in mismatches)
    assert any(f[f"{side}_scope"] == scope for f in mismatches)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [
    {"arrive": "relaxed"},
    {"wait": "relaxed"},
    {"wait": "none"},
    # The arrival lands on the sender's own flag: the receiver's wait reads no
    # write of the sender whose vectors it reads.
    {"flag_shift": 0},
], ids=_ids)
def test_no_release_acquire_pattern_orders_no_peer_store(modes, world, litmus_cache):
    report = _run(pl.push, pl.push_inputs(world, **modes), litmus_cache)

    assert report.verdict == "error"
    races = _data_races(report)
    assert _pairs(races) == _from_predecessor(world)
    assert _failures(races) == {"missing_inter_actor_sync"}
    assert _mismatches(report) == []


@pytest.mark.parametrize("world", WORLDS)
def test_weak_flag_store_races_the_acquire_and_publishes_nothing(world, litmus_cache):
    """8.7.1: a weak store is not morally strong with the `ld.acquire` that
    reads it, so it is a data race and no release."""
    report = _run(pl.push, pl.push_inputs(world, arrive="weak_store"), litmus_cache)

    flag = [f for f in report.native_payload["findings"]
            if f["kind"] == "data_race" and f["overlap"]["byte_len"] == 4]
    pairs = set()
    for finding in flag:
        sides = (finding["prior"], finding["current"])
        store = next(side for side in sides if side["access_kind"] == "write")
        poll = next(side for side in sides if side is not store)
        assert poll["operation"]["source"]["op_name"].startswith("tirx.ptx.atom")
        pairs.add((_rank(store), _rank(poll)))
    assert pairs == {((r - 1) % world, r) for r in range(world)}
    assert _pairs(_data_races(report)) == _from_predecessor(world)


@pytest.mark.parametrize("world", WORLDS)
def test_peer_store_after_the_release_fence_is_not_published(world, litmus_cache):
    """8.8: a release pattern orders only what precedes its first instruction."""
    report = _run(pl.push, pl.push_inputs(world, arrive="fence_sys_relaxed", late_store=1),
                  litmus_cache)

    races = _data_races(report)
    assert _pairs(races) == _from_predecessor(world, [0])
    assert _failures(races) == {"missing_inter_actor_sync"}


@pytest.mark.parametrize("world", WORLDS)
def test_read_before_the_acquire_fence_is_not_acquired(world, litmus_cache):
    """8.8: an acquire pattern orders only what follows its last instruction."""
    report = _run(pl.push, pl.push_inputs(world, wait="relaxed_fence_sys", early_read=1),
                  litmus_cache)

    races = _data_races(report)
    assert _pairs(races) == _from_predecessor(world, [0])
    assert _failures(races) == {"missing_inter_actor_sync"}


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("barrier", ["sync_before", "sync_after"])
def test_lanes_off_the_releasing_lane_need_a_barrier(barrier, world, litmus_cache):
    """8.9.5: cumulativity is transitivity of base causality order. Without
    `bar.warp.sync` before the arrival the other lanes' peer stores precede no
    release; without it after the wait their reads follow no acquire."""
    report = _run(pl.push, pl.push_inputs(world, **{barrier: 0}), litmus_cache)

    races = _data_races(report)
    assert _pairs(races) == _from_predecessor(world, range(1, pl.LANES))
    assert _failures(races) == {"missing_inter_actor_sync"}


# --- pull: peer loads after acquiring the publisher's flag ----------------------

@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [{}, {"reuse": 1}], ids=["pull", "pull_ack_reuse"])
def test_peer_loads_after_a_sys_acquire_are_ordered(modes, world, litmus_cache):
    report = _run(pl.pull, pl.pull_inputs(world, **modes), litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes,side", [
    ({"publish": "release_gpu"}, "release"),
    ({"poll": "acquire_gpu"}, "acquire"),
], ids=lambda value: value if isinstance(value, str) else _ids(value))
def test_gpu_scoped_publication_orders_no_peer_load(modes, side, world, litmus_cache):
    report = _run(pl.pull, pl.pull_inputs(world, **modes), litmus_cache)

    races = _data_races(report)
    assert _pairs(races) == _from_successor(world)
    assert _failures(races) == {"missing_inter_actor_sync"}
    mismatches = _mismatches(report)
    assert mismatches and all(f["actor_relation"] == "cross_rank" for f in mismatches)
    assert any(f[f"{side}_scope"] == "gpu" for f in mismatches)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("modes", [{"publish": "relaxed"}, {"poll": "relaxed"}], ids=_ids)
def test_relaxed_publication_orders_no_peer_load(modes, world, litmus_cache):
    report = _run(pl.pull, pl.pull_inputs(world, **modes), litmus_cache)

    races = _data_races(report)
    assert _pairs(races) == _from_successor(world)
    assert _failures(races) == {"missing_inter_actor_sync"}
    assert _mismatches(report) == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("ack", ["relaxed", "none"])
def test_overwriting_a_buffer_a_peer_still_reads_is_a_war_race(ack, world, litmus_cache):
    """The publisher may reuse its outbox only once the puller's loads are
    ordered before the overwrite: a release by the puller after its loads and
    an acquire by the publisher before its stores. Without either, every
    vector the puller loads races the overwrite."""
    report = _run(pl.pull, pl.pull_inputs(world, reuse=1, ack=ack), litmus_cache)

    races = _data_races(report)
    assert _pairs(races) == _from_successor(world)
    assert {f["access_pair"] for f in races} <= {"read_write", "write_read"}
    for finding in races:
        reader = next(s for s in (finding["prior"], finding["current"])
                      if s["access_kind"] == "read")
        assert reader["operation"]["source"]["op_name"] == "tirx.ptx.ld_vec"
    assert _failures(races) == {"missing_inter_actor_sync"}


# --- concurrent peer atomics ----------------------------------------------------

def _atomics(world, op, cache_dir):
    return _run(pl.atomics, pl.atomics_inputs(world, op), cache_dir)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("op", ["atom_add_sys", "red_add_sys", "atom_cas_sys", "st_relaxed_sys"])
def test_morally_strong_peer_updates_do_not_race(op, world, litmus_cache):
    """8.7: strong `.sys` operations on one location from every rank are
    morally strong with each other, so they never form a data race."""
    report = _atomics(world, op, litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("op", ["atom_add_gpu", "red_add_gpu"])
def test_gpu_scoped_peer_atomics_are_not_morally_strong_across_ranks(op, world, litmus_cache):
    """8.7: strong, but each `.gpu` scope misses the other rank's thread."""
    report = _atomics(world, op, litmus_cache)

    assert report.verdict == "error"
    findings = report.native_payload["findings"]
    assert findings and all(
        (f["kind"], f["release_scope"], f["acquire_scope"], f["actor_relation"])
        == ("scope_mismatch", "gpu", "gpu", "cross_rank")
        for f in findings
    )


@pytest.mark.parametrize("world", WORLDS)
def test_weak_peer_stores_from_every_rank_race(world, litmus_cache):
    """8.7.1: conflicting writes that are neither morally strong nor ordered."""
    report = _atomics(world, "st_weak", litmus_cache)

    findings = report.native_payload["findings"]
    assert findings and all(
        f["kind"] == "data_race" and f["access_pair"] == "write_write"
        and _rank(f["prior"]) != _rank(f["current"])
        for f in findings
    )
    assert {f["overlap"]["byte_offset"] // 4 for f in findings} == set(VECTORS)


@pytest.mark.parametrize("world", WORLDS)
def test_weak_read_of_a_word_peers_update_atomically_races(world, litmus_cache):
    """8.7.1: an atomic is morally strong only with strong accesses; rank 0's
    weak `ld` of its own counter races every peer's `atom.add`."""
    report = _atomics(world, "atom_add_sys_weak_read", litmus_cache)

    findings = report.native_payload["findings"]
    assert findings and all(f["kind"] == "data_race" for f in findings)
    assert {_writer_reader(f) for f in findings} == {(w, 0) for w in range(1, world)}
    assert {f["overlap"]["byte_offset"] // 4 for f in findings} == set(VECTORS)


# --- bulk and TMA copies to and from a peer -------------------------------------

@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("copy", [0, 1], ids=["bulk", "tensor"])
@pytest.mark.parametrize("proxy_fence", ["after_wait", "after_wait_all", "producer"])
def test_async_proxy_load_of_a_peer_buffer_after_the_acquire_and_a_proxy_fence(
        proxy_fence, copy, world, litmus_cache):
    """8.9.5: a generic-proxy write and an async-proxy read of it are ordered
    by base causality order with `fence.proxy.async` on the path -- after the
    consumer's acquire, or before the producer's release."""
    report = _run(pl.bulk, pl.bulk_inputs(world, copy=copy, proxy_fence=proxy_fence),
                  litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("copy", [0, 1], ids=["bulk", "tensor"])
@pytest.mark.parametrize("proxy_fence", [
    "none",
    # Off the path: it precedes the acquire that orders the writes.
    "before_wait",
    # `.shared::cta` bridges shared memory, not the global writes.
    "after_wait_shared",
])
def test_async_proxy_load_of_a_peer_buffer_without_a_global_proxy_fence_races(
        proxy_fence, copy, world, litmus_cache):
    report = _run(pl.bulk, pl.bulk_inputs(world, copy=copy, proxy_fence=proxy_fence),
                  litmus_cache)

    races = _data_races(report)
    assert _pairs(races) == _from_successor(world)
    assert _failures(races) == {"missing_proxy_bridge"}
    for finding in races:
        bridge = finding["proxy_bridge"]
        assert {bridge["prior_proxy"], bridge["current_proxy"]} == {"generic", "async"}


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("copy", [0, 1], ids=["bulk", "tensor"])
def test_completed_async_proxy_store_to_a_peer_is_published_by_the_release(
        copy, world, litmus_cache):
    """`cp.async.bulk.wait_group 0` completes the store's writes, so the
    `.sys` release that follows orders them before the peer's reads."""
    report = _run(pl.bulk, pl.bulk_inputs(world, direction=1, copy=copy), litmus_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("copy", [0, 1], ids=["bulk", "tensor"])
@pytest.mark.parametrize("store_wait", ["read", "none"])
def test_incomplete_async_proxy_store_to_a_peer_is_not_published(
        store_wait, copy, world, litmus_cache):
    """`wait_group.read` waits only until the store has read shared memory; its
    global writes, like those of an unwaited store, are neither complete nor
    bridged to the generic proxy when the release executes."""
    report = _run(pl.bulk, pl.bulk_inputs(world, direction=1, copy=copy, store_wait=store_wait),
                  litmus_cache)

    races = _data_races(report)
    assert _pairs(races) == _from_predecessor(world)
    assert _failures(races) <= {"missing_proxy_bridge", "missing_inter_actor_sync"}
    assert "missing_proxy_bridge" in _failures(races)


# --- addressing a peer's private memory -----------------------------------------

@pytest.mark.parametrize("kernel,inputs", [
    (pl.push, lambda: pl.push_inputs(2, symmetric=False)),
    (pl.pull, lambda: pl.pull_inputs(2, symmetric=False)),
    (pl.atomics, lambda: pl.atomics_inputs(2, "red_add_sys", symmetric=False)),
    (pl.bulk, lambda: pl.bulk_inputs(2, symmetric=False)),
    (pl.bulk, lambda: pl.bulk_inputs(2, direction=1, symmetric=False)),
], ids=["store", "load", "red", "bulk_load", "bulk_store"])
def test_peer_access_to_private_memory_is_a_memory_error(kernel, inputs, litmus_cache):
    report = racecheck(kernel, inputs=inputs(), cache_dir=litmus_cache)

    assert report.verdict == "error"
    assert PRIVATE in report.format()


def test_tensor_map_over_a_peers_private_memory_is_refused_at_launch(litmus_cache):
    with pytest.raises(NumSimExecutionError, match="TensorMap 'tmap': .*" + PRIVATE):
        racecheck(pl.bulk, inputs=pl.bulk_inputs(2, copy=1, tmap_symmetric=False),
                  cache_dir=litmus_cache)
