from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze, verify
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout, laneid, wg_local_layout


@T.prim_func
def _valid_forced_ldstmatrix():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 128), "bfloat16", scope="shared")
    fp32_fragment = T.alloc_tcgen05_ldst_frag("16x256b", (128, 8), "float32")
    bfloat16_fragment = T.alloc_cast_frag(fp32_fragment, "bfloat16")
    Tx.wg.copy(shared[:, :], bfloat16_fragment.permute(1, 0), dispatch="ldstmatrix")


@T.prim_func
def _valid_bound_forced_ldstmatrix():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", align=16)
    fp32_fragment = T.alloc_tcgen05_ldst_frag("16x256b", (64, 8), "float32")
    bfloat16_fragment = T.alloc_cast_frag(fp32_fragment, "bfloat16")
    for vb in T.unroll(8):
        column: T.let = vb * 8
        Tx.wg.copy(shared[:, column : column + 8], bfloat16_fragment[:, :], dispatch="ldstmatrix")


@T.prim_func
def _invalid_bound_forced_ldstmatrix_alignment():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", align=16)
    fp32_fragment = T.alloc_tcgen05_ldst_frag("16x256b", (64, 8), "float32")
    bfloat16_fragment = T.alloc_cast_frag(fp32_fragment, "bfloat16")
    for vb in T.unroll(7):
        column: T.let = vb * 8 + 1
        Tx.wg.copy(shared[:, column : column + 8], bfloat16_fragment[:, :], dispatch="ldstmatrix")


@T.prim_func
def _invalid_forced_ldstmatrix_backing_alignment():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 128), "bfloat16", scope="shared", align=8)
    fp32_fragment = T.alloc_tcgen05_ldst_frag("16x256b", (128, 8), "float32")
    bfloat16_fragment = T.alloc_cast_frag(fp32_fragment, "bfloat16")
    Tx.wg.copy(shared[:, :], bfloat16_fragment.permute(1, 0), dispatch="ldstmatrix")


@T.prim_func
def _invalid_forced_ldstmatrix_layout():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 128), "bfloat16", scope="shared")
    unrelated_local_layout = T.alloc_buffer((8, 128), "bfloat16", scope="local")
    Tx.wg.copy(shared[:, :], unrelated_local_layout[:, :], dispatch="ldstmatrix")


@T.prim_func
def _invalid_forced_reg_pair(source: T.Buffer((32,), "float16")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "float16", scope="shared")
    Tx.copy(shared[:], source[:], dispatch="reg")


@T.prim_func
def _invalid_forced_gmem_smem_pair(source: T.Buffer((32,), "float16")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    local: T.f16[32]
    Tx.copy(local[:], source[:], dispatch="gmem_smem")


@T.prim_func
def _valid_forced_reg(output: T.Buffer((32,), "float16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    local: T.f16[1]
    Tx.copy(output[lane : lane + 1], local[:], dispatch="reg")


@T.prim_func
def _valid_forced_gmem_smem(source: T.Buffer((32,), "float16")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "float16", scope="shared")
    Tx.copy(shared[:], source[:], dispatch="gmem_smem")


@T.prim_func
def _valid_forced_fallback(source: T.Buffer((32,), "float16"), output: T.Buffer((32,), "float16")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    Tx.copy(output[:], source[:], dispatch="fallback")


@T.prim_func
def _warpgroup_shared_overlap_forced_fallback(output: T.Buffer((17,), "float32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    shared = T.alloc_buffer((17,), "float32", scope="shared")
    if thread < 17:
        shared[thread] = T.cast(thread, "float32")
    T.cuda.cta_sync()
    Tx.wg.copy(shared[0:16], shared[1:17], dispatch="fallback")
    T.cuda.cta_sync()
    if thread < 17:
        output[thread] = shared[thread]


@T.prim_func
def _warpgroup_shared_overlap_auto_fallback(output: T.Buffer((17,), "float32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    shared = T.alloc_buffer((17,), "float32", scope="shared")
    if thread < 17:
        shared[thread] = T.cast(thread, "float32")
    T.cuda.cta_sync()
    Tx.wg.copy(shared[0:16], shared[1:17])
    T.cuda.cta_sync()
    if thread < 17:
        output[thread] = shared[thread]


@T.prim_func
def _warpgroup_shared_overlap_reg_hint(output: T.Buffer((17,), "float32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    shared = T.alloc_buffer((17,), "float32", scope="shared")
    if thread < 17:
        shared[thread] = T.cast(thread, "float32")
    T.cuda.cta_sync()
    Tx.wg.copy(shared[0:16], shared[1:17], dispatch="reg")
    T.cuda.cta_sync()
    if thread < 17:
        output[thread] = shared[thread]


@T.prim_func
def _warpgroup_shared_overlap_gmem_smem_hint(output: T.Buffer((17,), "float32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    shared = T.alloc_buffer((17,), "float32", scope="shared")
    if thread < 17:
        shared[thread] = T.cast(thread, "float32")
    T.cuda.cta_sync()
    Tx.wg.copy(shared[0:16], shared[1:17], dispatch="gmem_smem")
    T.cuda.cta_sync()
    if thread < 17:
        output[thread] = shared[thread]


@T.prim_func
def _auto_local_to_local_copy(output: T.Buffer((32, 2), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source: T.f32[2]
    destination: T.f32[2]
    source[0] = T.cast(lane, "float32") + T.float32(0.5)
    source[1] = T.cast(lane, "float32") + T.float32(1.5)
    destination[0] = T.float32(-1)
    destination[1] = T.float32(-1)
    Tx.warp.copy(destination, source)
    output[lane, 0] = destination[0]
    output[lane, 1] = destination[1]


@T.prim_func
def _invalid_degenerate_local_copy(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source: T.f32[1]
    destination: T.f32[1]
    source[0] = T.cast(lane, "float32")
    Tx.warp.copy(destination, source)
    output[lane] = destination[0]


@T.prim_func
def _thread_owned_local_copy(
    source: T.Buffer((128, 2), "float32"), output: T.Buffer((128, 2), "float32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    thread = T.thread_id_in_wg([128])
    source_storage: T.f32[2]
    destination_storage: T.f32[2]
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=wg_local_layout(2))
    for column in T.serial(2):
        source_view[thread, column] = source[thread, column]
    Tx.wg.copy(destination_view, source_view)
    for column in T.serial(2):
        output[thread, column] = destination_view[thread, column]


@T.prim_func
def _same_warp_owner_remap(
    source: T.Buffer((32, 32), "float32"), output: T.Buffer((32, 32), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_storage: T.f32[32]
    destination_storage: T.f32[32]
    source_view = source_storage.view(32, 32, layout=TileLayout(S[(32, 32) : (1 @ laneid, 1)]))
    destination_view = destination_storage.view(
        32, 32, layout=TileLayout(S[(32, 32) : (1, 1 @ laneid)])
    )
    for column in T.serial(32):
        source_view[lane, column] = source[lane, column]
    Tx.warp.copy(destination_view, source_view)
    for row in T.serial(32):
        output[row, lane] = destination_view[row, lane]


@T.prim_func
def _cross_warp_owner_remap():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    source_storage: T.f32[2]
    destination_storage: T.f32[2]
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=wg_local_layout(2))
    Tx.wg.copy(destination_view[64:128, :], source_view[0:64, :])


@T.prim_func
def _invalid_forced_thread_owned_local_copy():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    source_storage: T.f32[2]
    destination_storage: T.f32[2]
    source_view = source_storage.view(128, 2, layout=wg_local_layout(2))
    destination_view = destination_storage.view(128, 2, layout=wg_local_layout(2))
    Tx.wg.copy(destination_view, source_view, dispatch="fallback")


@pytest.mark.parametrize(
    "kernel",
    [
        _valid_forced_ldstmatrix,
        _valid_bound_forced_ldstmatrix,
        _invalid_bound_forced_ldstmatrix_alignment,
        _invalid_forced_ldstmatrix_backing_alignment,
        _invalid_forced_ldstmatrix_layout,
    ],
)
def test_ldstmatrix_dispatch_hint_does_not_select_numsim_semantics(kernel, tmp_path):
    module = numsim.transpile(kernel, cache_dir=tmp_path)


@pytest.mark.parametrize(
    "kernel",
    [
        _valid_forced_reg,
        _valid_forced_gmem_smem,
        _valid_forced_fallback,
        _invalid_forced_reg_pair,
        _invalid_forced_gmem_smem_pair,
    ],
)
def test_ordinary_copy_dispatch_does_not_select_semantics(kernel):
    verify(analyze(kernel))


@pytest.mark.parametrize(
    "kernel",
    [
        _warpgroup_shared_overlap_forced_fallback,
        _warpgroup_shared_overlap_auto_fallback,
        _warpgroup_shared_overlap_reg_hint,
        _warpgroup_shared_overlap_gmem_smem_hint,
    ],
)
def test_warpgroup_shared_copy_snapshots_overlap_independent_of_dispatch(kernel, tmp_path):
    output = np.zeros((17,), dtype=np.float32)

    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    def check_output() -> None:
        np.testing.assert_array_equal(
            result.outputs["output"], np.asarray([*range(1, 17), 16], dtype=np.float32)
        )

    if kernel is _warpgroup_shared_overlap_auto_fallback:
        check_output()
    else:
        check_output()


def test_auto_local_to_local_copy_runs_for_every_active_lane(tmp_path):
    output = np.zeros((32, 2), dtype=np.float32)

    module = numsim.transpile(_auto_local_to_local_copy, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    lanes = np.arange(32, dtype=np.float32)
    expected = np.stack([lanes + np.float32(0.5), lanes + np.float32(1.5)], axis=1)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_single_element_local_copy_runs_for_every_active_lane(tmp_path):
    output = np.zeros((32,), dtype=np.float32)
    result = numsim.Engine().run(
        numsim.transpile(_invalid_degenerate_local_copy, cache_dir=tmp_path), {"output": output}
    )
    np.testing.assert_array_equal(result.outputs["output"], np.arange(32, dtype=np.float32))


def test_fallback_hint_is_accepted_for_thread_owned_local_copy(tmp_path):
    numsim.transpile(_invalid_forced_thread_owned_local_copy, cache_dir=tmp_path)


def test_default_thread_owned_local_copy_uses_physical_owners(tmp_path):
    source = np.arange(128 * 2, dtype=np.float32).reshape(128, 2) + np.float32(0.25)
    output = np.zeros_like(source)

    module = numsim.transpile(_thread_owned_local_copy, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_same_warp_copy_remaps_lane_owned_values(tmp_path):
    source = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) + np.float32(0.25)
    output = np.zeros_like(source)

    module = numsim.transpile(_same_warp_owner_remap, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_cross_warp_copy_owner_remap_is_accepted_statically():
    verify(analyze(_cross_warp_owner_remap))
