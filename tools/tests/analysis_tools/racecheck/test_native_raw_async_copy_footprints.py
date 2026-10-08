"""Native Racecheck coverage for the raw async-copy PTX forms.

Every Racecheck case here returned ``incomplete`` before these forms had a
footprint (or, for ``cp.async.mbarrier.arrive`` without ``.noinc``, a barrier
lifecycle at all). Each form contributes:

* a correct kernel that must now analyze to completion with the right verdict,
* a mutated kernel with a genuine race that Racecheck must still flag,
* where the footprint is narrower than the whole window (``.cp_mask``,
  ``.multicast``), a negative control proving it NARROWS: a racing access to a
  byte or CTA the copy does not touch must stay clean, and
* a Synccheck assertion, because Synccheck filters these gap kinds.

Scope note on the Synccheck assertions: they pin that Synccheck does not move
**on these kernels**. Synccheck verdicts for ``cp.async.mbarrier.arrive`` do
move in general --- the ``.noinc`` spelling goes from ``incomplete``
("mbarrier completion issue has no committed generation") to ``clean``, which
is the ``canonical_protocol_generation`` arm added alongside this work doing
its job.
"""

from __future__ import annotations

import numpy as np
import pytest
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck


_BULK_G2S_CTA = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
_BULK_G2S_CTA_IGNORE_OOB = f"{_BULK_G2S_CTA}.ignore_oob"
_BULK_G2S_CLUSTER_MULTICAST = (
    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
)
_BULK_S2C = "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
_BULK_S2G_MASKED = "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.cp_mask"
_TMA_GATHER4 = (
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
    "mbarrier::complete_tx::bytes.cta_group::1"
)


@pytest.fixture(scope="module")
def raw_async_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-raw-async-copy")


def _assert_clean(report) -> None:
    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.native_payload["findings"] == [], report.native_payload["findings"]
    assert report.verdict == "clean"


def _assert_no_race(report) -> None:
    """No race and no coverage gap, without pinning the advisory-driven verdict.

    A negative control deliberately reads memory the copy never wrote, which
    draws an ``uninitialized_read`` advisory (``review``). That advisory is the
    point; a race finding would not be.
    """

    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.native_payload["findings"] == [], report.native_payload["findings"]
    assert report.verdict in {"clean", "review"}, report.native_payload


def _assert_flagged(report, *, kind: str) -> dict:
    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.verdict == "error", report.native_payload
    findings = report.native_payload["findings"]
    assert findings, report.native_payload
    assert all(finding["kind"] == "data_race" for finding in findings)
    assert any(finding["access_pair"] == kind for finding in findings), findings
    return findings[0]


# --------------------------------------------------------------------------
# cp.async.mbarrier.arrive without .noinc
#
# Both kernels initialize the barrier for two ordinary arrivals. The plain
# spelling raises generation 0's pending count by one, so the deferred arrive-on
# bound to warp 0's `cp.async` is the third and last required arrival. If that
# raise were not modeled, the two ordinary arrivals alone would flip the phase
# and the consumer would read the destination while the copy is still in flight
# -- which is exactly what the mutated kernel does on purpose.
# --------------------------------------------------------------------------
@T.prim_func
def cp_async_mbarrier_arrive_pending_count(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 2)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        T.ptx["cp.async.ca.shared.global"](
            T.address_of(shared[0]),
            T.address_of(source[0]),
            16,
        )
        T.ptx.cp.async_.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))
    elif (warp == 1) and (lane == 0):
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(16):
            output[element] = shared[element]


@T.prim_func
def cp_async_mbarrier_arrive_unwaited_read(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 2)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        T.ptx["cp.async.ca.shared.global"](
            T.address_of(shared[0]),
            T.address_of(source[0]),
            16,
        )
        T.ptx.cp.async_.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))
    elif (warp == 1) and (lane == 0):
        for element in T.serial(16):
            output[element] = shared[element]
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)


def _cp_async_arrive_inputs() -> dict:
    return {
        "source": np.arange(16, dtype=np.uint8),
        "output": np.zeros(16, dtype=np.uint8),
    }


def test_cp_async_mbarrier_arrive_pending_count_completes_clean(raw_async_cache):
    report = racecheck(
        cp_async_mbarrier_arrive_pending_count,
        inputs=_cp_async_arrive_inputs(),
        cache_dir=raw_async_cache,
    )

    _assert_clean(report)


def test_cp_async_mbarrier_arrive_flags_an_unwaited_destination_read(raw_async_cache):
    report = racecheck(
        cp_async_mbarrier_arrive_unwaited_read,
        inputs=_cp_async_arrive_inputs(),
        cache_dir=raw_async_cache,
    )

    finding = _assert_flagged(report, kind="read_write")
    assert finding["prior"]["operation"]["global_warp_id"] == 1
    assert finding["prior"]["access_kind"] == "read"
    assert finding["current"]["operation"]["global_warp_id"] == 0
    assert finding["current"]["access_kind"] == "write"


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (cp_async_mbarrier_arrive_pending_count, "clean"),
        (cp_async_mbarrier_arrive_unwaited_read, "review"),
    ],
    ids=["ordered", "unwaited"],
)
def test_cp_async_mbarrier_arrive_synccheck_is_unchanged(raw_async_cache, kernel, expected):
    """Synccheck filters these gaps, so it must report no finding on either.

    Both kernels are synchronization-correct: every arrival is budgeted and the
    wait passes. The mutated kernel only draws the ``uninitialized_read``
    advisory that its deliberate race implies, which is a ``review``, not a
    synchronization finding.
    """

    report = synccheck(
        kernel,
        inputs=_cp_async_arrive_inputs(),
        cache_dir=raw_async_cache,
    )

    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.native_payload["findings"] == [], report.native_payload["findings"]
    assert report.verdict == expected, report.native_payload
    advisory_kinds = {advisory["kind"] for advisory in report.native_payload.get("advisories", ())}
    assert advisory_kinds <= {"uninitialized_read"}, advisory_kinds


# --------------------------------------------------------------------------
# cp.async.bulk.g2s.cta, with and without .ignore_oob
# --------------------------------------------------------------------------
@T.prim_func
def bulk_g2s_cta_waited(source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.ptx[_BULK_G2S_CTA](shared.ptr_to([0]), source.ptr_to([0]), 16, barrier.ptr_to([0]))
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(16):
            output[element] = shared[element]


@T.prim_func
def bulk_g2s_cta_unwaited(source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.ptx[_BULK_G2S_CTA](shared.ptr_to([0]), source.ptr_to([0]), 16, barrier.ptr_to([0]))
        for element in T.serial(16):
            output[element] = shared[element]
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)


@T.prim_func
def bulk_g2s_cta_ignore_oob_waited(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.ptx[_BULK_G2S_CTA_IGNORE_OOB](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            16,
            T.uint32(3),
            T.uint32(4),
            barrier.ptr_to([0]),
        )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(9):
            output[element + 3] = shared[element + 3]


@T.prim_func
def bulk_g2s_cta_ignore_oob_short_source(
    source: T.Buffer((13,), "uint8"), output: T.Buffer((13,), "uint8")
):
    """`.ignore_oob` whose window reaches past the end of its source binding.

    The copy reads only ``[0, 13)`` and writes the whole 16-byte destination, so
    the published source read must be the in-bounds slice. A planner that
    bounds-checked the full window against the source would report a false
    out-of-bounds access here.
    """

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.ptx[_BULK_G2S_CTA_IGNORE_OOB](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            16,
            T.uint32(0),
            T.uint32(3),
            barrier.ptr_to([0]),
        )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(13):
            output[element] = shared[element]


def _bulk16_inputs() -> dict:
    return {
        "source": np.arange(16, dtype=np.uint8),
        "output": np.zeros(16, dtype=np.uint8),
    }


def _bulk13_inputs() -> dict:
    return {
        "source": np.arange(13, dtype=np.uint8),
        "output": np.zeros(13, dtype=np.uint8),
    }


@pytest.mark.parametrize(
    "kernel",
    [bulk_g2s_cta_waited, bulk_g2s_cta_ignore_oob_waited],
    ids=["plain", "ignore_oob"],
)
def test_bulk_g2s_cta_completes_clean(raw_async_cache, kernel):
    report = racecheck(kernel, inputs=_bulk16_inputs(), cache_dir=raw_async_cache)

    _assert_clean(report)


@T.prim_func
def bulk_g2s_cta_two_lane(
    ignore_oob: T.int32,
    source: T.Buffer((48,), "uint8"),
    output: T.Buffer((48,), "uint8"),
):
    """Two disjoint issues contribute to the same barrier byte expectation."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((48,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 32)
    T.cuda.warp_sync()
    if lane < 2:
        base: T.int32 = lane * 16
        if ignore_oob != 0:
            T.ptx[_BULK_G2S_CTA_IGNORE_OOB](
                shared.ptr_to([base]),
                source.ptr_to([base]),
                16,
                T.uint32(0),
                T.uint32(0),
                barrier.ptr_to([0]),
            )
        else:
            T.ptx[_BULK_G2S_CTA](
                shared.ptr_to([base]),
                source.ptr_to([base]),
                16,
                barrier.ptr_to([0]),
            )
    if lane == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(32):
            output[element] = shared[element]


def _two_lane_inputs(ignore_oob: int) -> dict:
    return {
        "ignore_oob": np.int32(ignore_oob),
        "source": np.arange(48, dtype=np.uint8),
        "output": np.zeros(48, dtype=np.uint8),
    }


def test_bulk_g2s_cta_ignore_oob_does_not_bounds_check_the_ignored_bytes(raw_async_cache):
    report = racecheck(
        bulk_g2s_cta_ignore_oob_short_source,
        inputs=_bulk13_inputs(),
        cache_dir=raw_async_cache,
    )

    assert report.native_payload["execution_error"] is None, report.native_payload[
        "execution_error"
    ]
    _assert_clean(report)


def test_bulk_g2s_cta_flags_an_unwaited_destination_read(raw_async_cache):
    report = racecheck(bulk_g2s_cta_unwaited, inputs=_bulk16_inputs(), cache_dir=raw_async_cache)

    finding = _assert_flagged(report, kind="read_write")
    assert finding["current"]["access_kind"] == "write"
    assert finding["current"]["span"]["byte_len"] == 16


@pytest.mark.parametrize(
    "kernel",
    [bulk_g2s_cta_waited, bulk_g2s_cta_ignore_oob_waited, bulk_g2s_cta_unwaited],
    ids=["plain", "ignore_oob", "unwaited"],
)
def test_bulk_g2s_cta_synccheck_reports_no_finding(raw_async_cache, kernel):
    report = synccheck(kernel, inputs=_bulk16_inputs(), cache_dir=raw_async_cache)

    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.native_payload["findings"] == [], report.native_payload["findings"]


def test_bulk_g2s_cta_ignore_oob_short_source_synccheck_reports_no_finding(raw_async_cache):
    report = synccheck(
        bulk_g2s_cta_ignore_oob_short_source,
        inputs=_bulk13_inputs(),
        cache_dir=raw_async_cache,
    )

    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.native_payload["findings"] == [], report.native_payload["findings"]


# --------------------------------------------------------------------------
# cp.async.bulk.g2s.cluster.multicast
#
# CTA 0 issues one multicast copy into both CTAs of the cluster. The published
# destination footprint must name each target CTA's own allocation, so the
# consumer on CTA 1 races iff it reads before its own barrier completes.
# --------------------------------------------------------------------------
@T.prim_func
def bulk_g2s_multicast_waited(source: T.Buffer((16,), "uint8"), output: T.Buffer((2, 16), "uint8")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
    if (cta == 0) and (lane == 0):
        T.ptx[_BULK_G2S_CLUSTER_MULTICAST](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            16,
            barrier.ptr_to([0]),
            T.uint16(3),
        )
    if lane == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(16):
            output[cta, element] = shared[element]


@T.prim_func
def bulk_g2s_multicast_unwaited_remote_read(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((2, 16), "uint8")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
    if (cta == 0) and (lane == 0):
        T.ptx[_BULK_G2S_CLUSTER_MULTICAST](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            16,
            barrier.ptr_to([0]),
            T.uint16(3),
        )
    if (cta == 1) and (lane == 0):
        for element in T.serial(16):
            output[cta, element] = shared[element]
    if lane == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    if (cta == 0) and (lane == 0):
        for element in T.serial(16):
            output[cta, element] = shared[element]


@T.prim_func
def bulk_g2s_multicast_unselected_cta_read(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((2, 16), "uint8")
):
    """Negative control: the `.multicast` footprint must NARROW to its mask.

    Identical to ``bulk_g2s_multicast_unwaited_remote_read`` except the mask is
    ``1`` instead of ``3``, so CTA 1 is NOT a multicast target and never
    receives the copy. Its unwaited read is therefore not a race. A planner that
    published the write to every CTA in the cluster instead of only the selected
    ones would report the same ``read_write`` the ``cta_mask=3`` kernel does.
    """

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.ptx[_BULK_G2S_CLUSTER_MULTICAST](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            16,
            barrier.ptr_to([0]),
            T.uint16(1),
        )
    if (cta == 1) and (lane == 0):
        for element in T.serial(16):
            output[cta, element] = shared[element]
    if (cta == 0) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(16):
            output[cta, element] = shared[element]
    T.cuda.cluster_sync()


def _multicast_inputs() -> dict:
    return {
        "source": np.arange(16, dtype=np.uint8),
        "output": np.zeros((2, 16), dtype=np.uint8),
    }


def test_bulk_g2s_multicast_completes_clean(raw_async_cache):
    report = racecheck(
        bulk_g2s_multicast_waited, inputs=_multicast_inputs(), cache_dir=raw_async_cache
    )

    _assert_clean(report)


def test_bulk_g2s_multicast_flags_an_unwaited_remote_target_read(raw_async_cache):
    report = racecheck(
        bulk_g2s_multicast_unwaited_remote_read,
        inputs=_multicast_inputs(),
        cache_dir=raw_async_cache,
    )

    finding = _assert_flagged(report, kind="read_write")
    # The racing write is the multicast leg into CTA 1's own allocation, which
    # only exists once the footprint is published per target CTA.
    assert finding["current"]["access_kind"] == "write"
    assert finding["prior"]["operation"]["global_warp_id"] == 1


def test_bulk_g2s_multicast_ignores_a_read_on_an_unselected_cta(raw_async_cache):
    report = racecheck(
        bulk_g2s_multicast_unselected_cta_read,
        inputs=_multicast_inputs(),
        cache_dir=raw_async_cache,
    )

    _assert_no_race(report)


@pytest.mark.parametrize(
    "kernel",
    [
        bulk_g2s_multicast_waited,
        bulk_g2s_multicast_unwaited_remote_read,
        bulk_g2s_multicast_unselected_cta_read,
    ],
    ids=["ordered", "unwaited", "unselected"],
)
def test_bulk_g2s_multicast_synccheck_reports_no_finding(raw_async_cache, kernel):
    report = synccheck(kernel, inputs=_multicast_inputs(), cache_dir=raw_async_cache)

    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.native_payload["findings"] == [], report.native_payload["findings"]


# --------------------------------------------------------------------------
# cp.async.bulk shared -> remote cluster shared (raw DSMEM)
#
# CTA 0 pushes into CTA 1's shared memory through a `mapa` pointer and signals
# CTA 1's barrier. The producer/consumer edge must be real: a consumer that
# reads its landing buffer without waiting is the sender-runs-ahead hazard
# class and must still be flagged.
# --------------------------------------------------------------------------
@T.prim_func
def bulk_s2s_cluster_waited(source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    staging = T.alloc_buffer((16,), "uint8", scope="shared")
    landing = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    remote_landing_bits = T.alloc_local((1,), "uint64")
    remote_barrier_bits = T.alloc_local((1,), "uint64")
    T.ptx.mapa.u64(remote_landing_bits[0], landing.ptr_to([0]), T.uint32(1))
    T.ptx.mapa.u64(remote_barrier_bits[0], barrier.ptr_to([0]), T.uint32(1))
    remote_landing: T.let[
        T.Var(name="raw_async_remote_landing", ty=PointerType(PrimType("uint8"), "shared"))
    ] = T.reinterpret(PointerType(PrimType("uint8"), "shared"), remote_landing_bits[0])
    remote_barrier: T.let[
        T.Var(name="raw_async_remote_barrier", ty=PointerType(PrimType("uint64"), "shared"))
    ] = T.reinterpret(PointerType(PrimType("uint64"), "shared"), remote_barrier_bits[0])
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane < 16):
        staging[lane] = source[lane]
    T.cuda.warp_sync()
    if (cta == 0) and (lane == 0):
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(remote_barrier, 16)
        T.ptx[_BULK_S2C](remote_landing, staging.ptr_to([0]), 16, remote_barrier)
    if (cta == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(16):
            output[element] = landing[element]
    T.cuda.cluster_sync()


@T.prim_func
def bulk_s2s_cluster_unwaited(source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    staging = T.alloc_buffer((16,), "uint8", scope="shared")
    landing = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    remote_landing_bits = T.alloc_local((1,), "uint64")
    remote_barrier_bits = T.alloc_local((1,), "uint64")
    T.ptx.mapa.u64(remote_landing_bits[0], landing.ptr_to([0]), T.uint32(1))
    T.ptx.mapa.u64(remote_barrier_bits[0], barrier.ptr_to([0]), T.uint32(1))
    remote_landing: T.let[
        T.Var(
            name="raw_async_unwaited_remote_landing",
            ty=PointerType(PrimType("uint8"), "shared"),
        )
    ] = T.reinterpret(PointerType(PrimType("uint8"), "shared"), remote_landing_bits[0])
    remote_barrier: T.let[
        T.Var(
            name="raw_async_unwaited_remote_barrier",
            ty=PointerType(PrimType("uint64"), "shared"),
        )
    ] = T.reinterpret(PointerType(PrimType("uint64"), "shared"), remote_barrier_bits[0])
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane < 16):
        staging[lane] = source[lane]
    T.cuda.warp_sync()
    if (cta == 0) and (lane == 0):
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(remote_barrier, 16)
        T.ptx[_BULK_S2C](remote_landing, staging.ptr_to([0]), 16, remote_barrier)
    if (cta == 1) and (lane == 0):
        for element in T.serial(16):
            output[element] = landing[element]
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cluster_sync()


def _dsmem_inputs() -> dict:
    return {
        "source": np.arange(16, dtype=np.uint8),
        "output": np.zeros(16, dtype=np.uint8),
    }


def test_bulk_s2s_cluster_completes_clean(raw_async_cache):
    report = racecheck(bulk_s2s_cluster_waited, inputs=_dsmem_inputs(), cache_dir=raw_async_cache)

    _assert_clean(report)


def test_bulk_s2s_cluster_flags_the_unwaited_consumer(raw_async_cache):
    report = racecheck(bulk_s2s_cluster_unwaited, inputs=_dsmem_inputs(), cache_dir=raw_async_cache)

    finding = _assert_flagged(report, kind="read_write")
    # The DSMEM write lands in the receiver's own allocation but is issued by
    # the sender's warp, which is what makes the sender-runs-ahead class
    # visible at all.
    assert finding["current"]["access_kind"] == "write"
    assert finding["current"]["operation"]["global_warp_id"] == 0
    assert finding["prior"]["operation"]["global_warp_id"] == 1


@pytest.mark.parametrize(
    "kernel",
    [bulk_s2s_cluster_waited, bulk_s2s_cluster_unwaited],
    ids=["ordered", "unwaited"],
)
def test_bulk_s2s_cluster_synccheck_reports_no_finding(raw_async_cache, kernel):
    report = synccheck(kernel, inputs=_dsmem_inputs(), cache_dir=raw_async_cache)

    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.native_payload["findings"] == [], report.native_payload["findings"]


# --------------------------------------------------------------------------
# cp.async.bulk.s2g with .cp_mask
# --------------------------------------------------------------------------
@T.prim_func
def bulk_s2g_masked_waited(source: T.Buffer((16,), "uint8"), destination: T.Buffer((32,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    if lane < 16:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx[_BULK_S2G_MASKED](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            16,
            T.uint64(0x1000000000000000),
            T.uint16(0x00FF),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)
        destination[0] = T.uint8(0xA5)


@T.prim_func
def bulk_s2g_masked_unwaited(
    raced_byte: T.int32,
    source: T.Buffer((16,), "uint8"),
    destination: T.Buffer((32,), "uint8"),
):
    """Race an unwaited generic write against a `.cp_mask` copy.

    ``byte_mask=0x00FF`` selects destination bytes 0..7 and leaves 8..15
    untouched, so the verdict must depend on which byte ``raced_byte`` names.
    """

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    if lane < 16:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx[_BULK_S2G_MASKED](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            16,
            T.uint64(0x1000000000000000),
            T.uint16(0x00FF),
        )
        destination[raced_byte] = T.uint8(0xA5)
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


def _masked_s2g_inputs(raced_byte: int | None = None) -> dict:
    inputs = {
        "source": np.arange(16, dtype=np.uint8),
        "destination": np.zeros(32, dtype=np.uint8),
    }
    if raced_byte is not None:
        inputs = {"raced_byte": np.int32(raced_byte), **inputs}
    return inputs


def test_bulk_s2g_masked_completes_clean(raw_async_cache):
    report = racecheck(
        bulk_s2g_masked_waited, inputs=_masked_s2g_inputs(), cache_dir=raw_async_cache
    )

    _assert_clean(report)


def test_bulk_s2g_masked_flags_a_write_to_a_selected_byte(raw_async_cache):
    report = racecheck(
        bulk_s2g_masked_unwaited, inputs=_masked_s2g_inputs(3), cache_dir=raw_async_cache
    )

    finding = _assert_flagged(report, kind="write_write")
    assert finding["overlap"]["byte_offset"] == 3
    assert finding["overlap"]["byte_len"] == 1


def test_bulk_s2g_masked_ignores_a_write_to_an_unselected_byte(raw_async_cache):
    """Negative control: the `.cp_mask` footprint must NARROW, not cover.

    Byte 11 is masked out of ``byte_mask=0x00FF``, so the copy never writes it
    and the unwaited generic write to it is not a race. A planner that
    published the whole 16-byte window instead of the selected runs would
    report the same ``write_write`` as byte 3 does.
    """

    report = racecheck(
        bulk_s2g_masked_unwaited, inputs=_masked_s2g_inputs(11), cache_dir=raw_async_cache
    )

    _assert_clean(report)


@pytest.mark.parametrize(
    ("kernel", "raced_byte"),
    [(bulk_s2g_masked_waited, None), (bulk_s2g_masked_unwaited, 3)],
    ids=["ordered", "unwaited"],
)
def test_bulk_s2g_masked_synccheck_reports_no_finding(raw_async_cache, kernel, raced_byte):
    report = synccheck(kernel, inputs=_masked_s2g_inputs(raced_byte), cache_dir=raw_async_cache)

    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.native_payload["findings"] == [], report.native_payload["findings"]


# --------------------------------------------------------------------------
# cp.async.bulk.tensor gather4
# --------------------------------------------------------------------------
@T.prim_func
def tma_gather4_waited(input_map: T.TensorMap(), output: T.Buffer((4, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 4), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 64)
        T.ptx[_TMA_GATHER4](
            T.address_of(shared[0, 0]),
            T.address_of(input_map),
            0,
            0,
            1,
            2,
            3,
            T.address_of(barrier[0]),
        )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for row in T.serial(4):
            for column in T.serial(4):
                output[row, column] = shared[row, column]


@T.prim_func
def tma_gather4_unwaited(input_map: T.TensorMap(), output: T.Buffer((4, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 4), "float32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 64)
        T.ptx[_TMA_GATHER4](
            T.address_of(shared[0, 0]),
            T.address_of(input_map),
            0,
            0,
            1,
            2,
            3,
            T.address_of(barrier[0]),
        )
        for row in T.serial(4):
            for column in T.serial(4):
                output[row, column] = shared[row, column]
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)


def _gather4_inputs() -> dict:
    source = np.arange(16, dtype=np.float32).reshape(4, 4) + np.float32(0.5)
    return {
        "input_map": numsim.TensorMap(
            base=source,
            global_shape=(4, 4),
            global_strides=(16,),
            box_shape=(4, 1),
            element_strides=(1, 1),
        ).numpy(),
        "output": np.zeros((4, 4), dtype=np.float32),
    }


def test_tma_gather4_completes_clean(raw_async_cache):
    report = racecheck(tma_gather4_waited, inputs=_gather4_inputs(), cache_dir=raw_async_cache)

    _assert_clean(report)


def test_tma_gather4_flags_an_unwaited_destination_read(raw_async_cache):
    report = racecheck(tma_gather4_unwaited, inputs=_gather4_inputs(), cache_dir=raw_async_cache)

    finding = _assert_flagged(report, kind="read_write")
    assert finding["current"]["access_kind"] == "write"
    assert finding["current"]["space"] == "shared"


@pytest.mark.parametrize(
    "kernel", [tma_gather4_waited, tma_gather4_unwaited], ids=["ordered", "unwaited"]
)
def test_tma_gather4_synccheck_reports_no_finding(raw_async_cache, kernel):
    report = synccheck(kernel, inputs=_gather4_inputs(), cache_dir=raw_async_cache)

    assert report.native_payload["incomplete"] == [], report.native_payload["incomplete"]
    assert report.native_payload["findings"] == [], report.native_payload["findings"]
