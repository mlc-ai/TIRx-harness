"""Static-origin checks for aligned blocking named-barrier syncs.

Within one named-barrier generation, the presence of any blocking aligned sync
requires every blocking sync contribution to be aligned. Contributions from
one warp must share a static TIR call site; distinct warps may use separate
inlined sites. This holds for full- and sub-CTA counts. ``bar.arrive``
contributions stay independent, and a generation whose blocking syncs are all
unaligned (``T.ptx.barrier.sync``) stays clean.

The ``T.cuda.cta_sync()`` wrapper and handwritten ``T.ptx.bar.sync`` lower to
the same aligned instruction contract and are therefore compared as one
population.
"""

from __future__ import annotations

import pytest

from tirx_harness.numsim.checkers import _run_synccheck as internal_synccheck
from tvm.script import tirx as T


@T.prim_func
def native_split_site_full_cta_bar_sync():
    T.device_entry()
    _bx = T.cta_id([1])
    tx = T.thread_id([1024])

    if tx < 512:
        T.ptx.bar.sync(3, 1024)
    else:
        T.ptx.bar.sync(3, 1024)


@T.prim_func
def native_split_site_sub_cta_bar_sync():
    T.device_entry()
    warp = T.warp_id([3])
    _lane = T.lane_id([32])

    if warp == 0:
        T.ptx.bar.sync(3, 64)
    elif warp == 1:
        T.ptx.bar.sync(3, 64)


@T.prim_func
def native_same_warp_split_static_sites():
    T.device_entry()
    lane = T.lane_id([32])

    if lane < 16:
        T.ptx.bar.sync(3, 32)
    else:
        T.ptx.bar.sync(3, 32)


@T.prim_func
def native_split_site_full_cta_arrive_plus_sync():
    T.device_entry()
    warp = T.warp_id([2])
    _lane = T.lane_id([32])

    if warp == 0:
        T.ptx.bar.arrive(3, 64)
    else:
        T.ptx.bar.sync(3, 64)


@T.prim_func
def native_split_site_full_cta_unaligned_sync():
    T.device_entry()
    warp = T.warp_id([2])
    _lane = T.lane_id([32])

    if warp == 0:
        T.ptx.barrier.sync(3, 64)
    else:
        T.ptx.barrier.sync(3, 64)


@T.prim_func
def native_unaligned_plus_one_aligned_sync_site():
    T.device_entry()
    warp = T.warp_id([2])
    _lane = T.lane_id([32])

    if warp == 0:
        T.ptx.barrier.sync(3, 64)
    else:
        T.ptx.bar.sync(3, 64)


@T.prim_func
def native_aligned_plus_one_unaligned_sync_site():
    T.device_entry()
    warp = T.warp_id([2])
    _lane = T.lane_id([32])

    if warp == 0:
        T.ptx.bar.sync(3, 64)
    else:
        T.ptx.barrier.sync(3, 64)


@T.prim_func
def native_uniform_loop_full_cta_bar_sync():
    T.device_entry()
    _warp = T.warp_id([2])
    _lane = T.lane_id([32])

    # Keep one backend instruction so this tests dynamic reuse, not unrolling.
    for _iteration in T.serial(2, unroll=False):
        T.ptx.bar.sync(3, 64)


@T.prim_func
def native_sequential_full_cta_bar_sync_sites():
    T.device_entry()
    _warp = T.warp_id([2])
    _lane = T.lane_id([32])

    T.ptx.bar.sync(3, 64)
    T.ptx.bar.sync(3, 64)


@T.prim_func
def native_same_site_bar_sync_with_skewed_loop_iterations():
    T.device_entry()
    warp = T.warp_id([2])
    _lane = T.lane_id([32])

    # Complete generation zero with arrive+sync, then use an unaligned barrier
    # to offset the warps by one iteration at the same static bar.sync below.
    if warp == 0:
        T.ptx.bar.arrive(3, 64)
        T.ptx.barrier.sync(4, 64)

    # Keep one backend instruction while the two warps reach different iterations.
    for iteration in T.serial(3, unroll=False):
        if (warp != 0) or (iteration < 2):
            T.ptx.bar.sync(3, 64)
        if (warp != 0) and (iteration == 0):
            T.ptx.barrier.sync(4, 64)


@T.prim_func
def native_arrive_plus_two_split_sync_sites():
    T.device_entry()
    warp = T.warp_id([3])
    _lane = T.lane_id([32])

    if warp == 0:
        T.ptx.bar.arrive(3, 96)
    elif warp == 1:
        T.ptx.bar.sync(3, 96)
    else:
        T.ptx.bar.sync(3, 96)


@T.prim_func
def native_split_site_cta_sync_wrapper_and_handwritten_bar_zero():
    T.device_entry()
    warp = T.warp_id([2])
    _lane = T.lane_id([32])

    # ``T.cuda.cta_sync()`` lowers to the same aligned ``bar.sync(0, 64)``
    # instruction as the handwritten call, so one full-CTA generation on
    # barrier 0 completes from two static TIR sites.
    if warp == 0:
        T.cuda.cta_sync()
    else:
        T.ptx.bar.sync(0, 64)


@T.prim_func
def native_repeated_warpgroup_sync_generations():
    # The engine-internal warpgroup rendezvous (`plan_internal_warpgroup_sync`
    # behind the kernel engine's "warpgroup" rendezvous scope) is not reachable
    # from TIR today: generated tile helpers with a real warpgroup boundary
    # emit an ordinary aligned ``bar.sync`` on barrier 8, and participation-only
    # warpgroup scopes never construct a barrier. This is the closest reachable
    # shape: each warpgroup repeatedly executes its own aligned warpgroup sync,
    # so per-barrier generations must advance independently without crosstalk
    # between the two groups.
    T.device_entry()
    wg = T.warpgroup_id([2])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])

    # Keep one backend instruction per site so this tests dynamic generation
    # reuse, not unrolling.
    for _iteration in T.serial(3, unroll=False):
        if wg == 0:
            T.cuda.warpgroup_sync(6)
        else:
            T.cuda.warpgroup_sync(7)


@T.prim_func
def native_setmaxnreg_between_warpgroup_syncs():
    # Two ``setmaxnreg`` in one warpgroup are legal only when a warpgroup sync
    # separates them. The engine now credits that sync from any aligned
    # ``bar.sync`` whose contract covers exactly the four warps of the group,
    # so this pins the alignment+contract keyed accounting end-to-end.
    T.device_entry()
    _wg = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])

    T.ptx.setmaxnreg.dec.sync.aligned.u32(88)
    T.cuda.warpgroup_sync(7)
    T.ptx.setmaxnreg.inc.sync.aligned.u32(232)


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-synccheck-split-site-named-barrier")


def _run(kernel, *, inputs: dict[str, object], cache_dir):
    return internal_synccheck(
        kernel,
        inputs=inputs,
        cache_dir=cache_dir,
        max_workers=1,
    )


def _assert_alignment_mismatch_error(report):
    assert report.verdict == "error"
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("error", "fixed_sync_protocol_error")
    ]
    assert "mixes aligned and unaligned blocking syncs" in report.findings[0].message


def _assert_same_warp_divergence_error(report):
    assert report.verdict == "error"
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("error", "warp_collective_divergence")
    ]


def test_split_sites_across_full_cta_warps_stay_clean(native_cache_dir):
    report = _run(
        native_split_site_full_cta_bar_sync,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_split_sites_across_sub_cta_warps_stay_clean(native_cache_dir):
    report = _run(
        native_split_site_sub_cta_bar_sync,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_split_sites_within_one_warp_are_flagged(native_cache_dir):
    report = _run(
        native_same_warp_split_static_sites,
        inputs={},
        cache_dir=native_cache_dir,
    )

    _assert_same_warp_divergence_error(report)


def test_split_site_full_cta_arrive_plus_sync_stays_clean(native_cache_dir):
    report = _run(
        native_split_site_full_cta_arrive_plus_sync,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_split_site_full_cta_unaligned_sync_stays_clean(native_cache_dir):
    report = _run(
        native_split_site_full_cta_unaligned_sync,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_unaligned_plus_one_aligned_sync_site_is_flagged(native_cache_dir):
    report = _run(
        native_unaligned_plus_one_aligned_sync_site,
        inputs={},
        cache_dir=native_cache_dir,
    )

    _assert_alignment_mismatch_error(report)


def test_aligned_plus_one_unaligned_sync_site_is_flagged(native_cache_dir):
    report = _run(
        native_aligned_plus_one_unaligned_sync_site,
        inputs={},
        cache_dir=native_cache_dir,
    )

    _assert_alignment_mismatch_error(report)


def test_uniform_loop_full_cta_bar_sync_stays_clean(native_cache_dir):
    report = _run(
        native_uniform_loop_full_cta_bar_sync,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_sequential_full_cta_bar_sync_sites_stay_clean(native_cache_dir):
    report = _run(
        native_sequential_full_cta_bar_sync_sites,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_same_site_bar_sync_with_skewed_loop_iterations_stays_clean(native_cache_dir):
    report = _run(
        native_same_site_bar_sync_with_skewed_loop_iterations,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_arrive_mixed_split_sync_sites_stay_clean(native_cache_dir):
    report = _run(
        native_arrive_plus_two_split_sync_sites,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_cta_sync_wrapper_and_handwritten_bar_zero_split_site_stays_clean(
    native_cache_dir,
):
    """The wrapper and a handwritten ``bar.sync(0, CTA)`` unify to one check."""

    report = _run(
        native_split_site_cta_sync_wrapper_and_handwritten_bar_zero,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_repeated_warpgroup_sync_generations_stay_clean(native_cache_dir):
    report = _run(
        native_repeated_warpgroup_sync_generations,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()


def test_setmaxnreg_between_warpgroup_syncs_stays_clean(native_cache_dir):
    report = _run(
        native_setmaxnreg_between_warpgroup_syncs,
        inputs={},
        cache_dir=native_cache_dir,
    )

    report.require_clean()
    assert report.findings == []
