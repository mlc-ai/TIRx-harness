from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout, tid_in_wg, wg_local_layout


def _transposed_warpgroup_layout() -> TileLayout:
    return TileLayout(S[(32, 4, 2) : (1 @ tid_in_wg, 32 @ tid_in_wg, 1)])


@T.prim_func
def _copy_cross_warp_owner_remap(
    source: T.Buffer((128, 2), "float32"),
    output: T.Buffer((128, 2), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    source_storage = T.alloc_buffer((2,), "float32", scope="local")
    destination_storage = T.alloc_buffer((2,), "float32", scope="local")
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=_transposed_warpgroup_layout())
    Tx.wg.copy(source_view[:, :], source[:, :])
    Tx.wg.copy(destination_view[:, :], source_view[:, :])
    Tx.wg.copy(output[:, :], destination_view[:, :])


@T.prim_func
def _mul_cross_warp_owner_remap(
    source: T.Buffer((128, 2), "float32"),
    output: T.Buffer((128, 2), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    source_storage = T.alloc_buffer((2,), "float32", scope="local")
    destination_storage = T.alloc_buffer((2,), "float32", scope="local")
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=_transposed_warpgroup_layout())
    Tx.wg.copy(source_view[:, :], source[:, :])
    Tx.wg.mul(destination_view[:, :], source_view[:, :], T.float32(3))
    Tx.wg.copy(output[:, :], destination_view[:, :])


@T.prim_func
def _cast_unary_binary_cross_warp_owner_remap(
    source: T.Buffer((128, 2), "float32"),
    output: T.Buffer((128, 2), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    source_storage = T.alloc_buffer((2,), "float32", scope="local")
    cast_storage = T.alloc_buffer((2,), "float16", scope="local")
    rounded_storage = T.alloc_buffer((2,), "float32", scope="local")
    sqrt_storage = T.alloc_buffer((2,), "float32", scope="local")
    result_storage = T.alloc_buffer((2,), "float32", scope="local")
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    cast_view = cast_storage.view(128, 2, layout=_transposed_warpgroup_layout())
    rounded_view = rounded_storage.view(128, 2, layout=wg_local_layout(2))
    sqrt_view = sqrt_storage.view(128, 2, layout=_transposed_warpgroup_layout())
    result_view = result_storage.view(128, 2, layout=wg_local_layout(2))
    Tx.wg.copy(source_view[:, :], source[:, :])
    Tx.wg.cast(cast_view[:, :], source_view[:, :])
    Tx.wg.cast(rounded_view[:, :], cast_view[:, :])
    Tx.wg.sqrt(sqrt_view[:, :], rounded_view[:, :])
    Tx.wg.add(result_view[:, :], sqrt_view[:, :], rounded_view[:, :])
    Tx.wg.copy(output[:, :], result_view[:, :])


def test_copy_transports_unique_owners_across_warps(tmp_path):
    source = np.arange(256, dtype=np.float32).reshape(128, 2) + np.float32(0.25)
    output = np.zeros_like(source)

    module = numsim.transpile(_copy_cross_warp_owner_remap, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_pointwise_transports_unique_owners_across_warps(tmp_path):
    source = np.arange(256, dtype=np.float32).reshape(128, 2) / np.float32(16)
    output = np.zeros_like(source)

    module = numsim.transpile(_mul_cross_warp_owner_remap, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source * np.float32(3))


def test_cast_unary_and_binary_share_owner_transport(tmp_path):
    roots = (np.arange(256, dtype=np.float32) % np.float32(16)).reshape(128, 2)
    source = roots * roots
    output = np.zeros_like(source)

    module = numsim.transpile(
        _cast_unary_binary_cross_warp_owner_remap,
        cache_dir=tmp_path,
    )
    result = numsim.Engine().run(module, {"source": source, "output": output})

    rounded = source.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(result.outputs["output"], np.sqrt(rounded) + rounded)
