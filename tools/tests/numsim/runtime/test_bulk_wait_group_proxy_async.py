"""`cp.async.bulk.wait_group.sync_restrict::proxy::async N`.

PTX ISA 9.7.10.28.6.2: the restricted wait waits for the same bulk
async-groups as the plain wait, but "only orders async-proxy global and shared
memory accesses before subsequent async-proxy accesses; not before subsequent
generic-proxy accesses".
"""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim, synccheck
from tvm_tirx.script import tirx as T

RESTRICTED_WAIT = "cp.async.bulk.wait_group.sync_restrict::proxy::async"

# Wait modes after the bulk shared-to-global copy.
PLAIN = 0
RESTRICTED = 1
RESTRICTED_THEN_PROXY_FENCE = 2
RESTRICTED_THEN_PLAIN = 3
PLAIN_READ = 4
NO_WAIT = 5


@T.function
def s2g_then_generic_global_read(
    mode: T.int32,
    source: T.Tensor((16,), "uint8"),
    destination: T.Tensor((16,), "uint8"),
    output: T.Tensor((16,), "uint8"),
):
    """The waiting thread reads the copy's global destination generically."""
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((16,), "uint8", align=16)
    if lane < 16:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
            destination.ptr_to([0]), shared.ptr_to([0]), T.cast(16, "uint32")
        )
        T.ptx.cp.async_.bulk.commit_group()
        if mode == PLAIN:
            T.ptx.cp.async_.bulk.wait_group(0)
        else:
            T.ptx[RESTRICTED_WAIT](0)
            if mode == RESTRICTED_THEN_PROXY_FENCE:
                T.ptx.fence.proxy.async_.global_()
            elif mode == RESTRICTED_THEN_PLAIN:
                T.ptx.cp.async_.bulk.wait_group(0)
        for index in T.serial(16):
            output[index] = destination[index]


@T.function
def s2g_then_generic_shared_write(
    mode: T.int32,
    source: T.Tensor((16,), "uint8"),
    destination: T.Tensor((16,), "uint8"),
):
    """The waiting thread overwrites the copy's shared source generically."""
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((16,), "uint8", align=16)
    if lane < 16:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
            destination.ptr_to([0]), shared.ptr_to([0]), T.cast(16, "uint32")
        )
        T.ptx.cp.async_.bulk.commit_group()
        if mode == PLAIN:
            T.ptx.cp.async_.bulk.wait_group(0)
        elif mode == PLAIN_READ:
            T.ptx.cp.async_.bulk.wait_group.read(0)
        else:
            T.ptx[RESTRICTED_WAIT](0)
            if mode == RESTRICTED_THEN_PROXY_FENCE:
                T.ptx.fence.proxy.async_.shared__cta()
        shared[0] = T.uint8(0xEE)
        if mode == PLAIN_READ:
            T.ptx.cp.async_.bulk.wait_group(0)


@T.function
def s2g_then_async_g2s_round_trip(
    mode: T.int32,
    source: T.Tensor((16,), "uint8"),
    destination: T.Tensor((16,), "uint8"),
    output: T.Tensor((16,), "uint8"),
):
    """A later async-proxy copy reads the global destination back into the
    shared source; only async-proxy accesses follow the restricted wait."""
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((16,), "uint8", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=8)
    if lane < 16:
        shared[lane] = source[lane]
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
            destination.ptr_to([0]), shared.ptr_to([0]), T.cast(16, "uint32")
        )
        T.ptx.cp.async_.bulk.commit_group()
        if mode == PLAIN:
            T.ptx.cp.async_.bulk.wait_group(0)
        elif mode == RESTRICTED:
            T.ptx[RESTRICTED_WAIT](0)
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]),
            destination.ptr_to([0]),
            T.cast(16, "uint32"),
            barrier.ptr_to([0]),
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        if mode == NO_WAIT:
            T.ptx.cp.async_.bulk.wait_group(0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = shared[lane]


@T.function
def restricted_wait_keeps_newest_groups_pending(
    group_read: T.int32,
    source: T.Tensor((32,), "uint8"),
    destination: T.Tensor((32,), "uint8"),
    output: T.Tensor((16,), "uint8"),
):
    """`N` keeps the newest group pending: a later async copy may read the
    older group's destination but not the newer one's."""
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((32,), "uint8", align=16)
    staged = T.alloc_shared((16,), "uint8", align=16)
    barrier = T.alloc_shared((1,), "uint64", align=8)
    shared[lane] = source[lane]
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        for group in T.serial(2):
            T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
                destination.ptr_to([group * 16]),
                shared.ptr_to([group * 16]),
                T.cast(16, "uint32"),
            )
            T.ptx.cp.async_.bulk.commit_group()
        T.ptx[RESTRICTED_WAIT](1)
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](
            staged.ptr_to([0]),
            destination.ptr_to([group_read * 16]),
            T.cast(16, "uint32"),
            barrier.ptr_to([0]),
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.cp.async_.bulk.wait_group(0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = staged[lane]


def _source(size=16):
    return (np.arange(size, dtype=np.uint8) * 7 + 3).astype(np.uint8)


def _global_read_inputs(mode):
    return {
        "mode": np.int32(mode),
        "source": _source(),
        "destination": np.zeros(16, np.uint8),
        "output": np.zeros(16, np.uint8),
    }


def _run(kernel, inputs, cache_dir):
    synccheck(kernel, inputs).require_clean()
    return numsim.Engine().run(numsim.transpile(kernel, cache_dir=cache_dir), inputs)


@pytest.mark.parametrize(
    "mode", [PLAIN, RESTRICTED, RESTRICTED_THEN_PROXY_FENCE, RESTRICTED_THEN_PLAIN]
)
def test_restricted_wait_completes_the_bulk_group(tmp_path, mode):
    result = _run(s2g_then_generic_global_read, _global_read_inputs(mode), tmp_path)
    np.testing.assert_array_equal(result.outputs["destination"], _source())
    np.testing.assert_array_equal(result.outputs["output"], _source())


@pytest.mark.parametrize("mode", [PLAIN, PLAIN_READ, RESTRICTED, RESTRICTED_THEN_PROXY_FENCE])
def test_restricted_wait_copies_the_source_before_it_is_overwritten(tmp_path, mode):
    inputs = {
        "mode": np.int32(mode),
        "source": _source(),
        "destination": np.zeros(16, np.uint8),
    }
    result = _run(s2g_then_generic_shared_write, inputs, tmp_path)
    np.testing.assert_array_equal(result.outputs["destination"], _source())


@pytest.mark.parametrize("mode", [PLAIN, RESTRICTED])
def test_restricted_wait_feeds_a_later_async_copy(tmp_path, mode):
    result = _run(s2g_then_async_g2s_round_trip, _global_read_inputs(mode), tmp_path)
    np.testing.assert_array_equal(result.outputs["destination"], _source())
    np.testing.assert_array_equal(result.outputs["output"], _source())


def test_restricted_wait_pending_count_completes_older_groups(tmp_path):
    inputs = {
        "group_read": np.int32(0),
        "source": _source(32),
        "destination": np.zeros(32, np.uint8),
        "output": np.zeros(16, np.uint8),
    }
    result = _run(restricted_wait_keeps_newest_groups_pending, inputs, tmp_path)
    np.testing.assert_array_equal(result.outputs["destination"], _source(32))
    np.testing.assert_array_equal(result.outputs["output"], _source(32)[:16])
