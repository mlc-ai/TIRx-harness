"""Native Racecheck over the TIRx ports of CUTLASS's and FlashInfer's NVLS
multimem kernels.

The GEMM + all-reduce publishes each C tile with a TMA tensor store and reads
it back through ``multimem.ld_reduce`` on every rank. Under the PTX memory model that
read needs a ``.sys`` release/acquire edge from every rank's store and a
``fence.proxy.alias`` on the path; FlashInfer's and CUTLASS's own signalling
miss both, while ``sys_fenced`` is clean.
"""

from __future__ import annotations

import pytest

from tests.numsim.support import multimem_kernels as mk
from tirx_harness.numsim.checkers import _run_racecheck as racecheck

WORLDS = (2, 4)
WARPS_PER_RANK = mk.GEMM_SHAPE[1] * mk.GEMM_WARPS


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-multimem-kernels")


def _gemm(protocol: str, world: int, cache_dir, dtype: str = "float32"):
    report = racecheck(
        mk.gemm_all_reduce_two_shot(*mk.GEMM_SHAPE, world, dtype),
        inputs=mk.gemm_inputs(*mk.GEMM_SHAPE, world, protocol, dtype, integer=True),
        cache_dir=cache_dir,
    )
    assert report.native_payload["incomplete"] == []
    return report


def _rank(access) -> int:
    return access["operation"]["global_warp_id"] // WARPS_PER_RANK


def _races(report, failure):
    return [
        finding for finding in report.native_payload["findings"]
        if finding["kind"] == "data_race" and finding["ordering_failure"] == failure
    ]


@pytest.mark.parametrize("world", WORLDS)
def test_cutlass_two_shot_all_reduce_is_clean(world, kernel_cache):
    report = racecheck(
        mk.two_shot_all_reduce(world, world), inputs=mk.two_shot_inputs(world, world),
        cache_dir=kernel_cache,
    )

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("protocol", [
    "sys_fenced",
    # A fence-then-relaxed arrival is a release pattern.
    "sys_fenced_relaxed_arrive",
    # One alias fence anywhere on the store-to-reduce path is enough...
    "sys_fenced_producer_alias_only",
    "sys_fenced_consumer_alias_only",
    # ...and after `wait_group 0` the TMA store needs no async->generic fence.
    "sys_fenced_no_async_fence",
])
def test_sys_scoped_alias_fenced_protocol_is_clean(protocol, world, kernel_cache):
    report = _gemm(protocol, world, kernel_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_half_precision_sys_fenced_protocol_is_clean(dtype, world, kernel_cache):
    report = _gemm("sys_fenced", world, kernel_cache, dtype)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
def test_unfenced_staging_races_the_tma_store(world, kernel_cache):
    """The epilogue writes the staging tile through the generic proxy; without
    `fence.proxy.async.shared::cta` the TMA store's async read is unordered."""
    report = _gemm("sys_fenced_no_staging_fence", world, kernel_cache)

    assert report.verdict == "error"
    findings = report.native_payload["findings"]
    assert findings and findings == _races(report, "missing_proxy_bridge")
    assert all(f["prior"]["space"] == f["current"]["space"] == "shared" for f in findings)


@pytest.mark.parametrize("world", WORLDS)
def test_arrival_after_a_read_only_store_wait_publishes_no_tile(world, kernel_cache):
    """`cp.async.bulk.wait_group.read` waits only until the TMA store has read
    its shared source. Its global writes are neither complete nor bridged to
    the generic proxy, so the `.sys` release orders none of them before any
    rank's multimem access to the tile -- even the rank's own -- and a peer's
    access may even precede the write. A write that completes after the
    release has no edge to the peers at all; one that completes before it
    still lacks the async->alias bridge."""
    report = _gemm("sys_fenced_read_only_store_wait", world, kernel_cache)

    assert report.verdict == "error"
    findings = report.native_payload["findings"]
    unbridged = _races(report, "missing_proxy_bridge")
    assert unbridged and len(unbridged) + len(_races(report, "missing_inter_actor_sync")) == len(
        findings)
    store = "tirx.ptx.cp_async_bulk_tensor_s2g"
    for finding in findings:
        ops = {finding[side]["operation"]["source"]["op_name"] for side in ("prior", "current")}
        assert store in ops and len(ops) == 2
        assert ops - {store} <= {"tirx.ptx.multimem_ld_reduce_f_vec",
                                 "tirx.ptx.multimem_st_f_vec"}
    for finding in unbridged:
        assert set(finding["proxy_bridge"].values()) == {"global", "async", "multicast_alias"}
    assert any(f["current"]["operation"]["source"]["op_name"] == store for f in findings)
    assert {_rank(f["prior"]) == _rank(f["current"]) for f in findings} == {True, False}


@pytest.mark.parametrize("world", WORLDS)
def test_mma_skipping_the_load_wait_reads_unloaded_operands(world, kernel_cache):
    """The MMA reads shared operands the TMA has not written yet, and the
    pipeline barriers fall out of step; which mbarrier violations surface
    depends on how far the MMA warp runs ahead of the producer."""
    report = racecheck(
        mk.gemm_all_reduce_two_shot(*mk.GEMM_SHAPE, world),
        inputs=mk.gemm_inputs(*mk.GEMM_SHAPE, world, "sys_fenced_unwaited_load", integer=True),
        cache_dir=kernel_cache,
    )

    assert report.verdict == "error"
    kinds = {finding.kind for finding in report.findings}
    assert "uninitialized_read" in kinds
    barrier_kinds = kinds - {"uninitialized_read"}
    assert barrier_kinds and all(kind.startswith("mbarrier_") for kind in barrier_kinds)


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("protocol", ["sys_fenced_no_alias", "sys_fenced_alias_after_arrive"])
def test_alias_fence_off_the_path_is_a_missing_proxy_bridge(protocol, world, kernel_cache):
    """FlashInfer's alias fence follows the arrival, so it orders nothing the
    store published; with no other alias fence every reduce is unbridged."""
    report = _gemm(protocol, world, kernel_cache)

    assert report.verdict == "error"
    findings = report.native_payload["findings"]
    assert findings and findings == _races(report, "missing_proxy_bridge")
    for finding in findings:
        assert set(finding["proxy_bridge"].values()) == {"global", "async", "multicast_alias"}


@pytest.mark.parametrize("world", WORLDS)
def test_relaxed_wait_does_not_acquire_the_tiles(world, kernel_cache):
    report = _gemm("sys_fenced_relaxed_wait", world, kernel_cache)

    assert report.verdict == "error"
    findings = report.native_payload["findings"]
    assert findings and findings == _races(report, "missing_inter_actor_sync")


@pytest.mark.parametrize("world", WORLDS)
def test_flashinfer_signalling_orders_no_tile(world, kernel_cache):
    """A relaxed `.gpu` CAS acquires nothing, so even the rank's own tile is
    unordered; the `.gpu` final barrier is also short of the peers."""
    report = _gemm("flashinfer", world, kernel_cache)

    assert report.verdict == "error"
    races = _races(report, "missing_inter_actor_sync")
    assert {_rank(f["prior"]) == _rank(f["current"]) for f in races} == {True, False}
    others = [
        f for f in report.native_payload["findings"]
        if f["kind"] != "data_race" or f["ordering_failure"] != "missing_inter_actor_sync"
    ]
    assert others and all(
        f["kind"] == "scope_mismatch" and f["actor_relation"] == "cross_rank" for f in others
    )


@pytest.mark.parametrize("world", WORLDS)
def test_cutlass_gpu_scope_reaches_at_most_the_local_rank(world, kernel_cache):
    """CUTLASS's `.gpu` release/acquire never orders a peer's tile. It orders
    the rank's own tile, which then lacks only the alias bridge, when the
    acquire reads the rank's own arrival; when a peer's `.gpu` red arrived
    last, that link of the RMW chain is not morally strong with the acquire,
    so the local release does not synchronize either."""
    report = _gemm("cutlass", world, kernel_cache)

    assert report.verdict == "error"
    unbridged = _races(report, "missing_proxy_bridge")
    unsynced = _races(report, "missing_inter_actor_sync")
    assert unbridged and all(_rank(f["prior"]) == _rank(f["current"]) for f in unbridged)
    assert unsynced and any(_rank(f["prior"]) != _rank(f["current"]) for f in unsynced)
    mismatches = [f for f in report.native_payload["findings"] if f["kind"] == "scope_mismatch"]
    assert mismatches and all(
        (f["release_scope"], f["acquire_scope"], f["actor_relation"]) == ("gpu", "gpu", "cross_rank")
        for f in mismatches
    )
    assert len(unbridged) + len(unsynced) + len(mismatches) == len(report.native_payload["findings"])
