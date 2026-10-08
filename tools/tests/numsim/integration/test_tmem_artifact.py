from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tests.numsim.support.kernels import (
    tmem_b_lane_column_mapping,
    tmem_d_alias_per_cta,
    tmem_dynamic_allocated_addr,
    tmem_f_lane_mapping,
    tmem_packed_alias,
    tmem_uninitialized_read,
)
from tvm.script import tirx as T
from tvm.tirx.layout import S, TCol, TileLayout, TLane

_DYNAMIC_TMEM_LAYOUT = TileLayout(S[(128, 64) : (1 @ TLane, 1 @ TCol)])


@T.prim_func
def dynamic_tmem_live_lease(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    tmem = T.decl_buffer(
        (128, 64), "uint32", scope="tmem", layout=_DYNAMIC_TMEM_LAYOUT, allocated_addr=address[0]
    )
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 64)
    T.cuda.warp_sync()
    tmem[lane, 0] = T.cast(7000 + lane, "uint32")
    output[lane] = tmem[lane, 0]
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 64)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def dynamic_tmem_use_before_alloc(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    tmem = T.decl_buffer(
        (128, 64), "uint32", scope="tmem", layout=_DYNAMIC_TMEM_LAYOUT, allocated_addr=address[0]
    )
    if lane == 0:
        address[0] = T.uint32(0)
    T.cuda.warp_sync()
    tmem[lane, 0] = T.cast(lane, "uint32")
    output[lane] = tmem[lane, 0]


@T.prim_func
def dynamic_tmem_outside_live_lease(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    tmem = T.decl_buffer(
        (128, 64), "uint32", scope="tmem", layout=_DYNAMIC_TMEM_LAYOUT, allocated_addr=address[0]
    )
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.warp_sync()
    tmem[lane, 32] = T.cast(lane, "uint32")
    output[lane] = T.uint32(1)


@T.prim_func
def dynamic_tmem_use_after_dealloc(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    tmem = T.decl_buffer(
        (128, 64), "uint32", scope="tmem", layout=_DYNAMIC_TMEM_LAYOUT, allocated_addr=address[0]
    )
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.warp_sync()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
    tmem[lane, 0] = T.cast(lane, "uint32")
    output[lane] = T.uint32(1)


@T.prim_func
def dynamic_tmem_multiple_allocations(output: T.Buffer((3,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    addresses = T.alloc_buffer((3,), "uint32", scope="shared")
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


@T.prim_func
def dynamic_tmem_leak(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    if lane == 0:
        output[0] = address[0]


def test_tmem_layout_d_aliases_and_isolates_ctas(tmp_path):
    output = np.zeros((2, 128), dtype=np.uint32)
    expected = np.stack([np.arange(128, dtype=np.uint32), 1000 + np.arange(128, dtype=np.uint32)])

    module = numsim.transpile(tmem_d_alias_per_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tmem_layout_f_maps_rows_to_half_slabs(tmp_path):
    output = np.zeros(64, dtype=np.uint32)
    module = numsim.transpile(tmem_f_lane_mapping, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], 2000 + np.arange(64, dtype=np.uint32))


def test_tmem_layout_b_splits_columns_across_lane_halves(tmp_path):
    output = np.zeros((64, 2), dtype=np.uint32)
    expected = np.stack(
        [3000 + np.arange(64, dtype=np.uint32), 4000 + np.arange(64, dtype=np.uint32)], axis=1
    )
    module = numsim.transpile(tmem_b_lane_column_mapping, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tmem_subword_views_alias_one_physical_cell(tmp_path):
    output = np.zeros(128, dtype=np.uint32)
    module = numsim.transpile(tmem_packed_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.full(128, 0x33441122, dtype=np.uint32)
    )


def test_tmem_runtime_address_without_a_dynamic_lease_is_rejected(tmp_path):
    output = np.zeros(32, dtype=np.uint32)
    module = numsim.transpile(tmem_dynamic_allocated_addr, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match="not covered by any live allocation"):
        numsim.Engine().run(module, {"output": output})


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
