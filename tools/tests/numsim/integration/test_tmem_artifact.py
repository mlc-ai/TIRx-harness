from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm_tirx.script import tirx as T


@T.function
def dynamic_tmem_live_lease(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.alloc_shared((1,), "uint32")
    value = T.alloc_local((1,), "uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 64)
    T.cuda.warp_sync()
    value[0] = T.cast(7000 + lane, "uint32")
    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0], value[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    value[0] = T.uint32(0)
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[lane] = value[0]
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 64)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.function
def dynamic_tmem_use_before_alloc(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.alloc_shared((1,), "uint32")
    value = T.alloc_local((1,), "uint32")
    if lane == 0:
        address[0] = T.uint32(0)
    T.cuda.warp_sync()
    value[0] = T.cast(lane, "uint32")
    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0], value[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    output[lane] = value[0]


@T.function
def dynamic_tmem_outside_live_lease(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.alloc_shared((1,), "uint32")
    value = T.alloc_local((1,), "uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.warp_sync()
    value[0] = T.cast(lane, "uint32")
    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0] + T.uint32(32), value[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    output[lane] = T.uint32(1)


@T.function
def dynamic_tmem_use_after_dealloc(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.alloc_shared((1,), "uint32")
    value = T.alloc_local((1,), "uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.warp_sync()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
    value[0] = T.cast(lane, "uint32")
    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0], value[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    output[lane] = T.uint32(1)


@T.function
def tmem_uninitialized_read(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.alloc_shared((1,), "uint32")
    value = T.alloc_local((1,), "uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.warp_sync()
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[lane] = value[0]
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.function
def dynamic_tmem_multiple_allocations(output: T.Tensor((3,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    addresses = T.alloc_shared((3,), "uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(addresses[0]), 128)
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(addresses[1]), 64)
    if lane == 0:
        output[0] = addresses[0]
        output[1] = addresses[1]
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(addresses[0], 128)
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(addresses[2]), 32)
    if lane == 0:
        output[2] = addresses[2]
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(addresses[1], 64)
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(addresses[2], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.function
def dynamic_tmem_leak(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.alloc_shared((1,), "uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    if lane == 0:
        output[0] = address[0]


def test_tmem_runtime_address_uses_a_live_dynamic_lease(tmp_path):
    output = np.zeros(32, dtype=np.uint32)
    module = numsim.transpile(dynamic_tmem_live_lease, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], 7000 + np.arange(32, dtype=np.uint32))


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (dynamic_tmem_use_before_alloc, "not covered by any live allocation"),
        (dynamic_tmem_outside_live_lease, "not covered by any live allocation"),
        (dynamic_tmem_use_after_dealloc, "not covered by any live allocation"),
    ],
)
def test_tmem_dynamic_lease_rejects_invalid_lifetimes_and_ranges(tmp_path, kernel, message):
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match=message):
        numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})


def test_tmem_allocator_returns_nonzero_bases_and_reuses_released_ranges(tmp_path):
    output = np.full(3, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    module = numsim.transpile(dynamic_tmem_multiple_allocations, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.array([0, 128, 0], dtype=np.uint32))


def test_tmem_live_allocation_at_kernel_exit_is_rejected(tmp_path):
    module = numsim.transpile(dynamic_tmem_leak, cache_dir=tmp_path)
    with pytest.raises(
        numsim.NumSimExecutionError, match="kernel exited with live TMEM allocations"
    ):
        numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.uint32)})


def test_tmem_uninitialized_reads_are_zero_filled_and_require_review(tmp_path):
    module = numsim.transpile(tmem_uninitialized_read, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.full(32, np.uint32(0xFFFFFFFF))})

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(32, dtype=np.uint32))
    assert result.verdict == "review"
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}
