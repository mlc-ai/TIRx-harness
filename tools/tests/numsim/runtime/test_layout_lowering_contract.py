from __future__ import annotations

import numpy as np

from tirx_harness import numsim, racecheck, synccheck
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, S, TCol, TileLayout, TLane, laneid


_PADDED_TILE = TileLayout(S[(2, 4) : (8, 1)])
_SMALL_SWIZZLE = ComposeLayout(0, 1, 2, TileLayout(S[(8,)]))
_TMA_SWIZZLE_32B = ComposeLayout(2, 1, 3, TileLayout(S[(64,)]))
_PADDED_COMPOSE = ComposeLayout(
    0,
    1,
    2,
    TileLayout(S[(2, 4) : (1, 4)]),
)
_TMEM_LANE_COLUMN = TileLayout(S[(2, 2) : (64 @ TLane, 3 @ TCol)])
_TMEM_PHYSICAL = TileLayout(S[(128, 4) : (1 @ TLane, 1 @ TCol)])
_LOCAL_WITH_GAPS = TileLayout(S[(32, 2) : (1 @ laneid, 2)] + 3)


@T.prim_func
def physical_buffers_with_strided_alias(output: T.Buffer((32, 3), "int32", layout=None)):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    compact = T.alloc_buffer((2, 4), "int32", scope="shared", layout=None)
    storage = T.alloc_buffer((16,), "int32", scope="shared", layout=None)
    alias = T.decl_buffer(
        (2, 4), "int32", data=storage.data, strides=(6, 1), elem_offset=1,
        scope="shared", layout=None,
    )
    private = T.alloc_buffer((2,), "int32", scope="local", layout=None)
    private[0] = lane * 10
    private[1] = private[0] + 3
    if lane < 16:
        storage[lane] = -1
    if lane < 8:
        compact[lane // 4, lane % 4] = lane + 100
    T.cuda.warp_sync()
    if lane < 8:
        alias[lane // 4, lane % 4] = lane + 200
    T.cuda.warp_sync()
    output[lane, 0] = compact[((31 - lane) % 8) // 4, (31 - lane) % 4]
    output[lane, 1] = storage[lane % 16]
    output[lane, 2] = private[1]


def test_physical_buffers_preserve_coordinates_aliases_and_lane_private_storage(tmp_path):
    bindings = {"output": np.zeros((32, 3), dtype=np.int32)}
    synccheck(physical_buffers_with_strided_alias, bindings).require_clean()
    report = racecheck(physical_buffers_with_strided_alias, bindings)
    # Reading storage after writing its explicit alias is intentional here.
    assert [(finding.status, finding.kind) for finding in report.findings] == [
        ("review", "alias_stale_read")
    ], report.format()
    module = numsim.transpile(physical_buffers_with_strided_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, bindings)

    storage = np.full(16, -1, dtype=np.int32)
    storage[1:5] = np.arange(200, 204)
    storage[7:11] = np.arange(204, 208)
    lanes = np.arange(32, dtype=np.int32)
    expected = np.column_stack((100 + (31 - lanes) % 8, storage[lanes % 16], lanes * 10 + 3))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def _offset_tmem_layout(layout, col_offset):
    offset = dict(layout.offset)
    offset[TCol] = offset.get(TCol, 0) + col_offset
    return TileLayout.from_iters(layout.shard, layout.replica, offset)


@T.prim_func
def shared_dyn_padded_tile_physical_bytes(output: T.Buffer((12,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    logical = T.alloc_buffer((2, 4), "uint32", scope="shared.dyn", layout=_PADDED_TILE)
    physical = T.decl_buffer((12,), "uint32", data=logical.data, scope="shared.dyn")
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


@T.prim_func
def compose_layout_physical_bytes(output: T.Buffer((14,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    logical = T.alloc_buffer((2, 4), "uint32", scope="shared", layout=_PADDED_COMPOSE)
    physical = T.decl_buffer((14,), "uint32", data=logical.data, scope="shared")
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


@T.prim_func
def overlapping_shared_views(output: T.Buffer((12,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    storage = T.alloc_buffer((12,), "uint32", scope="shared")
    left = T.decl_buffer((8,), "uint32", data=storage.data, scope="shared")
    right = T.decl_buffer((8,), "uint32", data=storage.data, elem_offset=4, scope="shared")
    if lane < 8:
        left[lane] = T.cast(1000 + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 8:
        right[lane] = T.cast(2000 + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 12:
        output[lane] = storage[lane]


@T.prim_func
def runtime_offset_shared_views(output: T.Buffer((8,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    storage = T.alloc_buffer((8,), "uint32", scope="shared.dyn")
    for phase in T.serial(2):
        view = T.decl_buffer(
            (4,),
            "uint32",
            data=storage.data,
            elem_offset=phase * 4,
            scope="shared.dyn",
        )
        if lane < 4:
            view[lane] = T.cast(100 * phase + lane, "uint32")
    T.cuda.warp_sync()
    if lane < 8:
        output[lane] = storage[lane]


@T.prim_func
def typed_tma_swizzle_physical_alias(
    source: T.Buffer((8, 8), "uint32"),
    physical_output: T.Buffer((256,), "uint8"),
    logical_output: T.Buffer((8, 8), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 8), "uint32", scope="shared", layout=_TMA_SWIZZLE_32B)
    physical = T.decl_buffer((256,), "uint8", data=shared.data, scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        Tx.copy_async(
            shared[:, :],
            source[:, :],
            dispatch="tma_auto",
            mbar=T.address_of(barrier[0]),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 256)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for index in T.serial(256):
            physical_output[index] = physical[index]
        physical[36 * 4] = T.uint8(0xE7)
        T.ptx.fence.proxy.async_.shared__cta()
        Tx.copy_async(logical_output[:, :], shared[:, :], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def tmem_lane_column_physical_alias(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    logical = T.decl_buffer(
        (2, 2),
        "uint32",
        scope="tmem",
        layout=_TMEM_LANE_COLUMN,
        allocated_addr=7,
    )
    physical = T.decl_buffer(
        (128, 4),
        "uint32",
        scope="tmem",
        layout=_TMEM_PHYSICAL,
        allocated_addr=7,
    )
    if lane == 0:
        logical[0, 0] = T.uint32(11)
        logical[0, 1] = T.uint32(12)
        logical[1, 0] = T.uint32(21)
        logical[1, 1] = T.uint32(22)
        output[0] = physical[0, 0]
        output[1] = physical[0, 3]
        output[2] = physical[64, 0]
        output[3] = physical[64, 3]


@T.prim_func
def runtime_tmem_layout_base(output: T.Buffer((32, 2), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    physical = T.decl_buffer(
        (32, 8),
        "uint32",
        scope="tmem",
        layout=TileLayout(S[(32, 8) : (1 @ TLane, 1 @ TCol)]),
        allocated_addr=10,
    )
    for phase in T.serial(2):
        view = T.decl_buffer(
            (32, 4),
            "uint32",
            scope="tmem",
            layout=_offset_tmem_layout(TileLayout(S[(32, 4) : (1 @ TLane, 1 @ TCol)]), phase * 4),
            allocated_addr=10,
        )
        view[lane, 0] = T.cast(1000 * phase + lane, "uint32")
    output[lane, 0] = physical[lane, 0]
    output[lane, 1] = physical[lane, 4]


@T.prim_func
def local_physical_span_with_gaps(output: T.Buffer((32, 8), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    storage = T.alloc_buffer((6,), "int32", scope="local")
    logical = storage.view(32, 2, layout=_LOCAL_WITH_GAPS)
    physical = logical.local()
    mediated = logical.local(layout=logical.layout.storage())
    for slot in T.serial(6):
        physical[slot] = lane * 10 + slot
        output[lane, slot] = physical[slot]
    output[lane, 6] = mediated[0]
    output[lane, 7] = mediated[1]


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


def test_typed_tma_uses_the_transpiler_lowered_swizzle_and_exposes_alias_writes(
    tmp_path,
):
    source = np.arange(1000, 1064, dtype=np.uint32).reshape(8, 8)
    physical_output = np.zeros(256, dtype=np.uint8)
    logical_output = np.zeros((8, 8), dtype=np.uint32)
    module = numsim.transpile(typed_tma_swizzle_physical_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "physical_output": physical_output,
            "logical_output": logical_output,
        },
    )

    rows, cols = np.indices((8, 8))
    logical_to_physical = rows * 8 + (cols ^ ((rows // 4) * 4))
    expected_physical_words = np.empty(64, dtype=np.uint32)
    expected_physical_words[logical_to_physical.ravel()] = source.ravel()
    expected_physical = expected_physical_words.astype("<u4", copy=False).view(np.uint8)
    np.testing.assert_array_equal(result.outputs["physical_output"], expected_physical)
    expected_logical = source.copy()
    expected_logical[4, 0] = (expected_logical[4, 0] & np.uint32(0xFFFFFF00)) | np.uint32(0xE7)
    np.testing.assert_array_equal(result.outputs["logical_output"], expected_logical)


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


def test_local_view_exposes_raw_span_including_layout_gaps_and_offset(tmp_path):
    module = numsim.transpile(local_physical_span_with_gaps, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((32, 8), dtype=np.int32)},
    )

    expected = np.empty((32, 8), dtype=np.int32)
    for lane in range(32):
        expected[lane, :6] = lane * 10 + np.arange(6, dtype=np.int32)
        expected[lane, 6:] = lane * 10 + np.array([3, 5], dtype=np.int32)
    np.testing.assert_array_equal(result.outputs["output"], expected)
