from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tests.numsim.support.kernels import (
    inactive_tcgen_transfer_is_noop,
    tcgen_float16_cta_group2,
    tcgen_local_to_tmem_roundtrip,
    tcgen_scale_bitcast_cta_group2,
    tcgen_scale_bitcast_shared_to_tmem,
    tcgen_shared_to_tmem_rank3,
    tcgen_shared_to_tmem_replica,
    tcgen_tmem_to_local_roundtrip,
)
from tests.numsim.microtests.cases.tcgen05_transfer_semantics import (
    _ldst_oracle,
    _make_raw_ldst_kernel,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import (
    R,
    S,
    TCol,
    TileLayout,
    TLane,
    tcgen05_atom_layout,
    tmem_datapath_layout,
    wg_local_layout,
)
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import (
    sf_smem_layout,
    sf_tmem_layout,
)

_WRONG_M64_TMEM_IDENTITY = TileLayout(S[(64, 8) : (1 @ TLane, 1 @ TCol)])
_WRONG_CP_LANE_PERMUTATION = TileLayout(
    S[(2, 16, 4) : (1 @ TLane, 2 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]
)


@pytest.mark.parametrize("packed", (False, True))
def test_split_tmem_transfer_preserves_the_gap(packed, tmp_path):
    kernel = _make_raw_ldst_kernel("16x32bx2", packed, split_padding=6)
    expected = _ldst_oracle("16x32bx2", packed, split_padding=6)
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "load_output": np.zeros((2, 4, 32, 8), dtype=np.uint32),
            "store_output": np.zeros((2, 128, 32), dtype=np.uint32),
        },
    )
    np.testing.assert_array_equal(result.outputs["load_output"], expected["loads"])
    np.testing.assert_array_equal(result.outputs["store_output"], expected["stores"])


@T.prim_func
def _fast_m64_tmem_to_local_roundtrip(
    source: T.Buffer((128, 32), "float32"), output: T.Buffer((128, 32), "float32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    thread = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (64, 64),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 64),
        allocated_addr=0,
    )
    local_in = T.alloc_buffer((32,), "float32", scope="local")
    local_out = T.alloc_buffer((32,), "float32", scope="local")
    tile_in = local_in.view(64, 64, layout=tcgen05_atom_layout("16x256b", (64, 64), "float32"))
    tile_out = local_out.view(64, 64, layout=tcgen05_atom_layout("16x256b", (64, 64), "float32"))
    for slot in T.serial(32):
        local_in[slot] = source[thread, slot]
    Tx.wg.copy_async(tmem[:, :], tile_in[:, :], dispatch="tmem<->local")
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    Tx.wg.copy_async(tile_out[:, :], tmem[:, :], dispatch="tmem<->local")
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for slot in T.serial(32):
        output[thread, slot] = local_out[slot]


@T.prim_func
def _fast_m128_tmem_to_local_roundtrip(
    source: T.Buffer((128, 64), "float32"), output: T.Buffer((128, 64), "float32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    thread = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (128, 128),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 128),
        allocated_addr=0,
    )
    local_in = T.alloc_buffer((64,), "float32", scope="local")
    local_out = T.alloc_buffer((64,), "float32", scope="local")
    tile_in = local_in.view(128, 64, layout=wg_local_layout(64))
    tile_out = local_out.view(128, 64, layout=wg_local_layout(64))
    for slot in T.serial(64):
        local_in[slot] = source[thread, slot]
    Tx.wg.copy_async(tmem[0:128, 0:64], tile_in[:, :], dispatch="tmem<->local")
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    Tx.wg.copy_async(tile_out[:, :], tmem[0:128, 0:64], dispatch="tmem<->local")
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for slot in T.serial(64):
        output[thread, slot] = local_out[slot]


@T.prim_func
def _fast_m128_tmem_slice_to_local_roundtrip(
    source: T.Buffer((128, 64), "float32"), output: T.Buffer((128, 64), "float32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    thread = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (128, 64),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 64),
        allocated_addr=0,
    )
    local_in = T.alloc_buffer((64,), "float32", scope="local")
    local_out = T.alloc_buffer((16,), "float32", scope="local")
    tile_in = local_in.view(128, 64, layout=wg_local_layout(64))
    tile_out = local_out.view(128, 16, layout=wg_local_layout(16))
    for slot in T.serial(64):
        local_in[slot] = source[thread, slot]
    Tx.wg.copy_async(tmem[:, :], tile_in[:, :], dispatch="tmem<->local")
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    for block in T.serial(4):
        Tx.wg.copy_async(
            tile_out[:, :],
            tmem[0:128, block * 16 : block * 16 + 16],
            dispatch="tmem<->local",
        )
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for slot in T.serial(16):
            output[thread, block * 16 + slot] = local_out[slot]


@T.prim_func
def _tmem_dtype_view_roundtrip(
    source: T.Buffer((128, 64), "float32"), output: T.Buffer((128, 64), "uint32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (64, 128),
        "float32",
        scope="tmem",
        layout=TileLayout(S[(64, 2, 64) : (1 @ TLane, 64 @ TLane, 1 @ TCol)]),
        allocated_addr=0,
    )
    for column in T.serial(64):
        tmem[row % 64, (row // 64) * 64 + column] = source[row, column]
    T.cuda.cta_sync()
    fragment = T.alloc_tcgen05_ldst_frag("32x32b", (128, 64), "uint32")
    rows = tmem.rearrange("h (b t) -> (b h) t", b=2)
    bits = T.decl_buffer(
        rows.shape, "uint32", layout=rows.layout, scope="tmem",
        allocated_addr=0,
    )
    Tx.wg.copy_async(fragment[:, :], bits[:, :])
    local = fragment.local()
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for column in T.serial(64):
        output[row, column] = local[column]


@T.prim_func
def _dynamic_tmem_origin_roundtrip(
    source: T.Buffer((256, 4), "float32"), output: T.Buffer((256, 4), "float32")
):
    T.device_entry()
    warpgroup = T.warpgroup_id([2])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    thread = T.meta_var(warpgroup * 128 + warp * 32 + lane)
    tmem = T.decl_buffer(
        (64, 64),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 64),
        allocated_addr=0,
    )
    local_in = T.alloc_buffer((4,), "float32", scope="local")
    local_out = T.alloc_buffer((4,), "float32", scope="local")
    tile_in = local_in.view(64, 8, layout=tcgen05_atom_layout("16x256b", (64, 8), "float32"))
    tile_out = local_out.view(64, 8, layout=tcgen05_atom_layout("16x256b", (64, 8), "float32"))
    column_base = T.meta_var(warpgroup * 8)
    for slot in T.serial(4):
        local_in[slot] = source[thread, slot]
    Tx.wg.copy_async(tmem[:, column_base : column_base + 8], tile_in[:, :], dispatch="tmem<->local")
    T.ptx.tcgen05.wait__st.sync.aligned()
    Tx.wg.copy_async(
        tile_out[:, :], tmem[:, column_base : column_base + 8], dispatch="tmem<->local"
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for slot in T.serial(4):
        output[thread, slot] = local_out[slot]


def _make_tcgen_16b_roundtrip(shape: str, rows: int, dtype: str, datapath: str):
    col_factor = {"16x64b": 2, "16x128b": 4, "16x256b": 8}[shape]
    regs_factor = {"16x64b": 1, "16x128b": 2, "16x256b": 4}[shape]
    cols = col_factor * 2
    per_thread_elements = regs_factor * 2 * (2 if rows == 128 else 1)
    tmem_rows = 64 if datapath == "F" else 128
    tmem_cols = max(64, cols)
    atom_layout = tcgen05_atom_layout(shape, (rows, cols), dtype)
    tmem_layout = tmem_datapath_layout(datapath, tmem_rows, tmem_cols)

    @T.prim_func
    def kernel(
        source: T.Buffer((128, per_thread_elements), dtype),
        output: T.Buffer((128, per_thread_elements), dtype),
    ):
        T.device_entry()
        _warpgroup = T.warpgroup_id([1])
        warp = T.warp_id_in_wg([4])
        lane = T.lane_id([32])
        thread = T.meta_var(warp * 32 + lane)
        tmem = T.decl_buffer(
            (tmem_rows, tmem_cols),
            dtype,
            scope="tmem",
            layout=tmem_layout,
            allocated_addr=0,
        )
        register_in = T.alloc_buffer((per_thread_elements,), dtype, scope="local")
        register_out = T.alloc_buffer((per_thread_elements,), dtype, scope="local")
        for index in T.serial(per_thread_elements):
            register_in[index] = source[thread, index]
        fragment_in = register_in.view(rows, cols, layout=atom_layout)
        fragment_out = register_out.view(rows, cols, layout=atom_layout)
        T.cuda.cta_sync()
        Tx.wg.copy_async(tmem[0:rows, 0:cols], fragment_in[:, :], dispatch="tmem<->local")
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cta_sync()
        Tx.wg.copy_async(fragment_out[:, :], tmem[0:rows, 0:cols], dispatch="tmem<->local")
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for index in T.serial(per_thread_elements):
            output[thread, index] = register_out[index]

    return kernel, per_thread_elements


@T.prim_func
def _tcgen_ldst_wrong_m64_tmem_layout():
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    tmem = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=_WRONG_M64_TMEM_IDENTITY,
        allocated_addr=0,
    )
    fragment = T.alloc_tcgen05_ldst_frag("16x256b", (64, 8), "float32")
    Tx.wg.copy_async(fragment[:, :], tmem[:, :])


@T.prim_func
def _tcgen_cp_wrong_destination_lane_permutation():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    tmem = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=_WRONG_CP_LANE_PERMUTATION,
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy_async(tmem[:, :], shared[:, :])


@T.prim_func
def _tcgen_cp_wrong_declared_shape():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    tmem = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=TileLayout(S[(32, 4) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy_async(
            tmem[:, :],
            shared[:, :],
            shape="32x256b",
            multicast="warpx4",
        )


@T.prim_func
def _tcgen_cp_two_pairs_in_one_cluster(
    source: T.Buffer((4, 128, 4), "uint8"),
    output: T.Buffer((4, 128, 4), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([4])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer(
        (128, 4),
        "uint8",
        scope="shared",
        layout=sf_smem_layout(128, SF_K=4, sf_per_mma=4),
    )
    scale_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=0,
    )
    for col in T.serial(4):
        shared[row, col] = source[cta, row, col]
    T.cuda.cluster_sync()
    if ((cta % 2) == 0) and (warp == 0) and (lane == 0):
        Tx.copy_async(scale_tmem[:, :], shared[:, :], cta_group=2)
    T.cuda.cluster_sync()
    for col in T.serial(4):
        output[cta, row, col] = scale_tmem[row, col]


def _decode_e4m3fn(source: np.ndarray) -> np.ndarray:
    exponent = ((source >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (source & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    result = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    return np.where((source & np.uint8(0x80)) != 0, -result, result)


def test_tcgen_tmem_to_wg_local_uses_layout_owners(tmp_path):
    source = np.arange(128 * 4, dtype=np.float32).reshape(128, 4)
    output = np.zeros_like(source)

    module = numsim.transpile(tcgen_tmem_to_local_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


@pytest.mark.parametrize(
    ("kernel", "shape"),
    [
        (_fast_m64_tmem_to_local_roundtrip, (128, 32)),
        (_fast_m128_tmem_to_local_roundtrip, (128, 64)),
    ],
)
def test_large_f32_tmem_to_local_preserves_values(tmp_path, kernel, shape):
    source = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 2048) / 17
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros_like(source)})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_m128_f32_tmem_slice_to_local_uses_bulk_layout_transfer(tmp_path):
    shape = (128, 64)
    source = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 2048) / 17
    module = numsim.transpile(_fast_m128_tmem_slice_to_local_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros_like(source)})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_same_width_tmem_dtype_view_preserves_physical_bits(tmp_path):
    source = (np.arange(128 * 64, dtype=np.float32).reshape(128, 64) - 4096) / 17
    output = np.zeros(source.shape, dtype=np.uint32)

    module = numsim.transpile(_tmem_dtype_view_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source.view(np.uint32))


def test_dynamic_tmem_origin_can_be_reused_across_transfers(tmp_path):
    source = (np.arange(256 * 4, dtype=np.float32).reshape(256, 4) - 512) / 17
    module = numsim.transpile(_dynamic_tmem_origin_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros_like(source)})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_tcgen_wg_local_to_tmem_uses_one_physical_writer(tmp_path):
    source = (1000 + np.arange(128 * 4, dtype=np.float32)).reshape(128, 4)
    output = np.zeros_like(source)

    module = numsim.transpile(tcgen_local_to_tmem_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


@pytest.mark.parametrize(
    ("shape", "rows", "dtype", "datapath"),
    [
        ("16x64b", 64, "float16", "D"),
        ("16x128b", 128, "bfloat16", "D"),
        ("16x256b", 64, "float16", "F"),
    ],
)
def test_tcgen_16xb_roundtrip_preserves_16bit_register_halves(
    tmp_path, shape, rows, dtype, datapath
):
    kernel, per_thread_elements = _make_tcgen_16b_roundtrip(shape, rows, dtype, datapath)
    values = np.linspace(-4, 6, 128 * per_thread_elements, dtype=np.float32).reshape(
        128, per_thread_elements
    )
    numpy_dtype = np.float16 if dtype == "float16" else pytest.importorskip("ml_dtypes").bfloat16
    source = values.astype(numpy_dtype)
    output = np.zeros_like(source)

    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"].view(np.uint16), source.view(np.uint16))


def test_tcgen_cp_expands_tlane_replicas(tmp_path):
    source = np.arange(32 * 4, dtype=np.float32).reshape(32, 4)
    output = np.zeros((128, 4), dtype=np.float32)
    expected = np.tile(source, (4, 1))

    module = numsim.transpile(tcgen_shared_to_tmem_replica, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tcgen_cp_supports_rank3_multi_instruction_layout(tmp_path):
    source = np.arange(4 * 32 * 16, dtype=np.uint8).reshape(4, 32, 16)
    output = np.zeros((128, 4, 16), dtype=np.uint8)
    expected = np.tile(source.transpose(1, 0, 2), (4, 1, 1))

    module = numsim.transpile(tcgen_shared_to_tmem_rank3, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tcgen_cp_bitcasts_uint8_scale_payload_into_float8_tmem(tmp_path):
    source = np.resize(
        np.array([0x00, 0x30, 0x38, 0x3C, 0x40, 0xB8, 0xFE], dtype=np.uint8), (128, 4)
    )
    output = np.zeros((128, 4), dtype=np.float32)
    exponent = ((source >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (source & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    expected = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    expected = np.where((source & np.uint8(0x80)) != 0, -expected, expected)

    module = numsim.transpile(tcgen_scale_bitcast_shared_to_tmem, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tcgen_cp_cta_group2_reads_and_writes_each_cta_scale_backing(tmp_path):
    source = np.resize(
        np.array([0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8, 0xBC, 0xC0], dtype=np.uint8), (2, 128, 4)
    )
    source[1] = np.roll(source[1], 3, axis=0)
    output = np.zeros((2, 128, 4), dtype=np.float32)
    exponent = ((source >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (source & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    expected = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    expected = np.where((source & np.uint8(0x80)) != 0, -expected, expected)

    module = numsim.transpile(tcgen_scale_bitcast_cta_group2, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tcgen_cp_cta_group2_supports_float16_payloads(tmp_path):
    source = np.linspace(-5, 7, 2 * 32 * 8, dtype=np.float16).reshape(2, 32, 8)
    output = np.zeros((2, 128, 8), dtype=np.float16)
    expected = np.tile(source, (1, 4, 1))

    module = numsim.transpile(tcgen_float16_cta_group2, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tcgen_cp_cta_group2_routes_each_pair_in_four_cta_cluster(tmp_path):
    source = np.resize(
        np.array([0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8, 0xBC, 0xC0], dtype=np.uint8),
        (4, 128, 4),
    )
    for cta in range(4):
        source[cta] = np.roll(source[cta], 3 * cta, axis=0)
    expected = _decode_e4m3fn(source)

    result = numsim.Engine().run(
        numsim.transpile(_tcgen_cp_two_pairs_in_one_cluster, cache_dir=tmp_path),
        {"source": source, "output": np.zeros((4, 128, 4), dtype=np.float32)},
    )

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_all_inactive_tcgen_transfer_is_a_noop(tmp_path):
    output = np.zeros(4, dtype=np.int32)

    module = numsim.transpile(inactive_tcgen_transfer_is_noop, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.full(4, 7, dtype=np.int32))


def test_tcgen_ldst_rejects_tmem_layout_outside_fixed_instruction_abi(tmp_path):
    with pytest.raises(
        UnsupportedTIRxError,
        match="TMEM/local layouts do not match any fixed tcgen05.ld/st D, F, or B instruction ABI",
    ):
        numsim.transpile(_tcgen_ldst_wrong_m64_tmem_layout, cache_dir=tmp_path)


def test_tcgen_cp_rejects_destination_lane_permutation(tmp_path):
    with pytest.raises(UnsupportedTIRxError):
        numsim.transpile(_tcgen_cp_wrong_destination_lane_permutation, cache_dir=tmp_path)


def test_tcgen_cp_rejects_declared_shape_that_disagrees_with_layout(tmp_path):
    with pytest.raises(UnsupportedTIRxError):
        numsim.transpile(_tcgen_cp_wrong_declared_shape, cache_dir=tmp_path)
