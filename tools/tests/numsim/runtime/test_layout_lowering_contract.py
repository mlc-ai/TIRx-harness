from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import Axis, ComposeLayout, S, TileLayout

_PADDED_TILE = TileLayout(S[(2, 4) : (8, 1)])
_SMALL_SWIZZLE = ComposeLayout(0, 1, 2, TileLayout(S[(8,)]))
_TMA_SWIZZLE_32B = ComposeLayout(2, 1, 3, TileLayout(S[(64,)]))
_PADDED_COMPOSE = ComposeLayout(
    0,
    1,
    2,
    TileLayout(S[(2, 4) : (1, 4)]),
)
_TMEM_LANE_COLUMN = TileLayout(S[(2, 2) : (64 @ Axis.get("TLane"), 3 @ Axis.get("TCol"))])
_TMEM_PHYSICAL = TileLayout(S[(128, 4) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))])
_LOCAL_WITH_GAPS = TileLayout(S[(32, 2) : (1 @ Axis.get("laneid"), 2)] + 3)


def _offset_tmem_layout(layout, col_offset):
    offset = dict(layout.offset)
    offset[Axis.get("TCol")] = offset.get(Axis.get("TCol"), 0) + col_offset
    return TileLayout.from_iters(layout.shard, layout.replica, offset)


@T.function
def shared_dyn_padded_tile_physical_bytes(output: T.Tensor((12,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    logical = T.alloc_tensor(
        ty_args=[T.Tensor((2, 4), "uint32", scope="shared.dyn", layout=_PADDED_TILE)]
    )
    physical = T.decl_tensor(logical.data, ty_args=[T.Tensor((12,), "uint32", scope="shared.dyn")])
    if lane < 12:
        physical[lane] = T.cast(T.uint32(0xDEAD0000) + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 8:
        row = T.meta_var(lane // 4)
        col = T.meta_var(lane % 4)
        logical[row, col] = T.cast(100 + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 12:
        output[lane] = physical[lane]


@T.function
def compose_layout_physical_bytes(output: T.Tensor((14,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    logical = T.alloc_shared((2, 4), "uint32", layout=_PADDED_COMPOSE)
    physical = T.decl_tensor(logical.data, ty_args=[T.Tensor((14,), "uint32", scope="shared")])
    if lane < 14:
        physical[lane] = T.cast(T.uint32(0xC0000000) + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 8:
        row = T.meta_var(lane // 4)
        col = T.meta_var(lane % 4)
        logical[row, col] = T.cast(500 + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 14:
        output[lane] = physical[lane]


@T.function
def overlapping_shared_views(output: T.Tensor((12,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    storage = T.alloc_shared((12,), "uint32")
    left = T.decl_tensor(storage.data, ty_args=[T.Tensor((8,), "uint32", scope="shared")])
    right = T.decl_tensor(
        T.ptr_byte_offset(storage.data, 16, ty=T.Ptr("uint32", "shared")),
        ty_args=[T.Tensor((8,), "uint32", scope="shared")],
    )
    if lane < 8:
        left[lane] = T.cast(1000 + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 8:
        right[lane] = T.cast(2000 + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 12:
        output[lane] = storage[lane]


@T.function
def runtime_offset_shared_views(output: T.Tensor((8,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    storage = T.alloc_tensor(ty_args=[T.Tensor((8,), "uint32", scope="shared.dyn")])
    for phase in T.serial(2):
        view = T.decl_tensor(
            T.ptr_byte_offset(storage.data, (phase * 4) * 4, ty=T.Ptr("uint32", "shared.dyn")),
            ty_args=[T.Tensor((4,), "uint32", scope="shared.dyn")],
        )
        if lane < 4:
            view[lane] = T.cast(100 * phase + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 8:
        output[lane] = storage[lane]


@T.function
def tmem_lane_column_physical_alias(output: T.Tensor((4,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    logical = T.tmem((2, 2), "uint32", allocated_addr=7, layout=_TMEM_LANE_COLUMN)
    physical = T.tmem((128, 4), "uint32", allocated_addr=7, layout=_TMEM_PHYSICAL)
    if lane == 0:
        logical[0, 0] = T.uint32(11)
        logical[0, 1] = T.uint32(12)
        logical[1, 0] = T.uint32(21)
        logical[1, 1] = T.uint32(22)
        output[0] = physical[0, 0]
        output[1] = physical[0, 3]
        output[2] = physical[64, 0]
        output[3] = physical[64, 3]


@T.function
def runtime_tmem_layout_base(output: T.Tensor((32, 2), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    physical = T.tmem(
        (32, 8),
        "uint32",
        allocated_addr=10,
        layout=TileLayout(S[(32, 8) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))]),
    )
    for phase in T.serial(2):
        view = T.tmem(
            (32, 4),
            "uint32",
            allocated_addr=10,
            layout=_offset_tmem_layout(
                TileLayout(S[(32, 4) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))]), phase * 4
            ),
        )
        view[lane, 0] = T.cast(1000 * phase + lane, "uint32")
    output[lane, 0] = physical[lane, 0]
    output[lane, 1] = physical[lane, 4]


def test_shared_dyn_tile_layout_maps_logical_values_into_the_exact_physical_span(
    tmp_path,
):
    output = np.zeros(12, dtype=np.uint32)
    module = numsim.transpile(shared_dyn_padded_tile_physical_bytes, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.arange(12, dtype=np.uint32) + np.uint32(0xDEAD0000)
    expected[:4] = np.arange(100, 104, dtype=np.uint32)
    expected[8:] = np.arange(104, 108, dtype=np.uint32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_compose_layout_combines_tile_and_swizzle_into_physical_alias_bytes(
    tmp_path,
):
    output = np.zeros(14, dtype=np.uint32)
    module = numsim.transpile(compose_layout_physical_bytes, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.arange(14, dtype=np.uint32) + np.uint32(0xC0000000)
    logical_to_physical = np.array([0, 5, 8, 13, 1, 4, 9, 12])
    expected[logical_to_physical] = np.arange(500, 508, dtype=np.uint32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_overlapping_declared_views_observe_one_shared_physical_allocation(tmp_path):
    output = np.zeros(12, dtype=np.uint32)
    module = numsim.transpile(overlapping_shared_views, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.concatenate(
        [
            np.arange(1000, 1004, dtype=np.uint32),
            np.arange(2000, 2008, dtype=np.uint32),
        ]
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_runtime_shared_view_offset_selects_the_observable_backing_region(tmp_path):
    module = numsim.transpile(runtime_offset_shared_views, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(8, dtype=np.uint32)})

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.concatenate([np.arange(4, dtype=np.uint32), 100 + np.arange(4, dtype=np.uint32)]),
    )


def test_tmem_tlane_tcol_coordinates_are_observable_through_a_physical_alias(
    tmp_path,
):
    output = np.zeros(4, dtype=np.uint32)
    module = numsim.transpile(tmem_lane_column_physical_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.array([11, 12, 21, 22], dtype=np.uint32)
    )


def test_runtime_tmem_layout_base_selects_physical_columns(tmp_path):
    module = numsim.transpile(runtime_tmem_layout_base, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((32, 2), dtype=np.uint32)})

    expected = np.stack(
        [np.arange(32, dtype=np.uint32), 1000 + np.arange(32, dtype=np.uint32)], axis=1
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
