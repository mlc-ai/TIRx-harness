"""Small source-backed TIRx kernels for transpiler tests."""

from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.lang.pipeline import TCGen05Bar
from tvm.tirx.layout import (
    ComposeLayout,
    R,
    S,
    TCol,
    TileLayout,
    TLane,
    laneid,
    tcgen05_atom_layout,
    tmem_datapath_layout,
    wg_local_layout,
)
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import (
    sf_smem_layout,
    sf_tmem_layout,
)
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout


_TEST_SWIZZLE_LAYOUT = ComposeLayout(
    3,
    3,
    3,
    TileLayout(S[(2, 128, 2, 64) : (16384, 64, 8192, 1)]),
)

_TEST_PERMUTED_SHARED_LAYOUT = TileLayout(S[(4, 32) : (1, 4)])
_TEST_FP8_SF_VIEW_LAYOUT = _TEST_PERMUTED_SHARED_LAYOUT.unpack(4).broadcast(4)
_TEST_TMA_SWIZZLE_128B_LAYOUT = ComposeLayout(2, 3, 3, TileLayout(S[(256,)]))
_TEST_MMA_F16_32B_LAYOUT = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_TEST_MMA_FP8_128X128_LAYOUT = mma_shared_layout(
    "float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (128, 128)
)
_TEST_MMA_FP8_128X32_LAYOUT = mma_shared_layout(
    "float8_e4m3fn", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32)
)
_TEST_MMA_FP8_8X128_LAYOUT = mma_shared_layout(
    "float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (8, 128)
)
_TEST_MMA_PACKED_FP4_128X32_LAYOUT = mma_shared_layout(
    "uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32)
)
_TEST_MMA_PACKED_FP4_8X32_LAYOUT = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 32))
_TEST_WARP_GEMM_D_FRAG = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])
_TEST_WARP_GEMM_A_FRAG_K8 = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])
_TEST_WARP_GEMM_B_FRAG_K8 = TileLayout(S[(4, 2, 8) : (1 @ laneid, 1, 4 @ laneid)])
_TEST_WARP_GEMM_A_FRAG = _TEST_WARP_GEMM_A_FRAG_K8.tile_to([16, 16], [16, 8])
_TEST_WARP_GEMM_B_FRAG = _TEST_WARP_GEMM_B_FRAG_K8.tile_to([16, 8], [8, 8])


@T.prim_func
def no_op_kernel():
    T.device_entry()
    _cta = T.cta_id([2])
    _warp = T.warp_id([3])
    _lane = T.lane_id([32])


@T.prim_func
def warpgroup_scope_coordinates(output: T.Buffer((2, 4, 32, 3), "int32")):
    T.device_entry()
    _cta = T.cta_id([1])
    warpgroup = T.warpgroup_id([2])
    warp_in_group = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    thread_in_group = T.thread_id_in_wg([128])
    output[warpgroup, warp_in_group, lane, 0] = warpgroup
    output[warpgroup, warp_in_group, lane, 1] = warp_in_group
    output[warpgroup, warp_in_group, lane, 2] = thread_in_group


@T.prim_func
def extent_free_warp_id(output: T.Buffer((2, 32), "int32")):
    T.device_entry()
    launch_warp = T.warp_id([2])
    implicit_warp = T.warp_id()
    lane = T.lane_id([32])
    output[launch_warp, lane] = implicit_warp


@T.prim_func
def bound_dynamic_cta_extent(output: T.Buffer((2,), "int32")):
    cta_count: T.let = T.min(T.int32(4), T.int32(2))
    T.device_entry()
    cta = T.cta_id([cta_count])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[cta] = cta + 1


@T.prim_func
def elect_sync_integer_branch(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if T.cuda.elect_sync():
        output[lane] = 1
    else:
        output[lane] = 2


@T.prim_func
def fetch_register_coordinates(output: T.Buffer((2, 32, 2), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    cluster_rank = T.cuda.mov_sreg(32, "cluster_ctarank")
    fetched_lane = T.cuda.mov_sreg(32, "laneid")
    output[cta, lane, 0] = cluster_rank
    output[cta, lane, 1] = fetched_lane


@T.prim_func
def unsupported_exp(source: T.Buffer((32,), "float32"), destination: T.Buffer((32,), "float32")):
    T.device_entry()
    lane = T.lane_id([32])
    destination[lane] = T.sin(source[lane])


@T.prim_func
def lane_add(
    left: T.Buffer((100,), "float32"),
    right: T.Buffer((100,), "float32"),
    output: T.Buffer((100,), "float32"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    index = T.meta_var(warp * 32 + lane)
    if index < 100:
        output[index] = left[index] + right[index]


@T.prim_func
def overlapping_alias_write(
    source: T.Buffer((32,), "float32"), destination: T.Buffer((32,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination[lane] = source[lane] + T.float32(1)


@T.prim_func
def global_decl_buffer_alias_reinterpret(
    storage: T.Buffer((128,), "uint8"),
    source: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    alias = T.decl_buffer((32,), "float32", data=storage.data, scope="global")
    alias[lane] = source[lane]
    output[lane] = alias[lane]


@T.prim_func
def global_dynamic_subview_by_warp(
    source: T.Buffer((64,), "int32"), output: T.Buffer((4, 16), "int32")
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    warp_rows = source.view(1, 1, 4, 4, 4).sub[0, 0, :, warp, :]
    if lane < 16:
        output[warp, lane] = warp_rows[lane // 4, lane % 4]


@T.prim_func
def matrix_add_2d(
    left: T.Buffer((3, 5), "float32"),
    right: T.Buffer((3, 5), "float32"),
    output: T.Buffer((3, 5), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 15:
        row = T.meta_var(lane // 5)
        col = T.meta_var(lane % 5)
        output[row, col] = left[row, col] + right[row, col]


@T.prim_func
def shared_alias_per_cta(output: T.Buffer((2, 32), "float32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "float32", scope="shared")
    alias = T.decl_buffer((32,), "float32", data=shared.data, elem_offset=16, scope="shared")
    alias[lane] = T.cast(cta * 100 + lane, "float32")
    output[cta, lane] = shared[16 + lane]


@T.prim_func
def shared_uninitialized_read(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "float32", scope="shared")
    output[lane] = shared[lane]


@T.prim_func
def local_array_per_lane(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    local = T.alloc_buffer((2,), "float32", scope="local")
    local[0] = T.cast(lane, "float32")
    local[1] = T.cast(lane + 1, "float32")
    output[lane] = local[0] + local[1]


@T.prim_func
def tmem_d_alias_per_cta(output: T.Buffer((2, 128), "uint32")):
    T.device_entry()
    cta = T.cta_id([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    primary = T.decl_buffer(
        (128, 4), "uint32", scope="tmem", layout=tmem_datapath_layout("D", 128, 4), allocated_addr=3
    )
    alias = T.decl_buffer(
        (128, 4), "uint32", scope="tmem", layout=tmem_datapath_layout("D", 128, 4), allocated_addr=3
    )
    primary[row, 0] = T.cast(cta * 1000 + row, "uint32")
    output[cta, row] = alias[row, 0]


@T.prim_func
def tmem_f_lane_mapping(output: T.Buffer((64,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    f_view = T.decl_buffer(
        (64, 4), "uint32", scope="tmem", layout=tmem_datapath_layout("F", 64, 4), allocated_addr=0
    )
    d_alias = T.decl_buffer(
        (128, 4), "uint32", scope="tmem", layout=tmem_datapath_layout("D", 128, 4), allocated_addr=0
    )
    physical_lane = T.meta_var((row // 16) * 32 + row % 16)
    f_view[row, 1] = T.cast(2000 + row, "uint32")
    output[row] = d_alias[physical_lane, 1]


@T.prim_func
def tmem_b_lane_column_mapping(output: T.Buffer((64, 2), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    b_view = T.decl_buffer(
        (64, 8), "uint32", scope="tmem", layout=tmem_datapath_layout("B", 64, 8), allocated_addr=2
    )
    d_alias = T.decl_buffer(
        (128, 4), "uint32", scope="tmem", layout=tmem_datapath_layout("D", 128, 4), allocated_addr=2
    )
    b_view[row, 0] = T.cast(3000 + row, "uint32")
    b_view[row, 4] = T.cast(4000 + row, "uint32")
    output[row, 0] = d_alias[row, 0]
    output[row, 1] = d_alias[row + 64, 0]


@T.prim_func
def tmem_packed_alias(output: T.Buffer((128,), "uint32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    packed_layout = TileLayout(S[(128, 2) : (1 @ TLane, 1 @ TCol)])
    word_layout = TileLayout(S[(128, 2) : (1 @ TLane, 1 @ TCol)])
    low = T.decl_buffer((128, 2), "uint16", scope="tmem", layout=packed_layout, allocated_addr=5)
    word = T.decl_buffer((128, 2), "uint32", scope="tmem", layout=word_layout, allocated_addr=5)
    low[row, 0] = T.uint16(0x1122)
    low[row, 1] = T.uint16(0x3344)
    output[row] = word[row, 0]


@T.prim_func
def tmem_dynamic_allocated_addr(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    if lane == 0:
        address[0] = T.uint32(6)
    T.cuda.warp_sync()
    tmem = T.decl_buffer(
        (32, 2),
        "uint32",
        scope="tmem",
        layout=TileLayout(S[(32, 2) : (1 @ TLane, 1 @ TCol)]),
        allocated_addr=address[0],
    )
    tmem[lane, 1] = T.cast(5000 + lane, "uint32")
    output[lane] = tmem[lane, 1]


@T.prim_func
def tmem_uninitialized_read(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tmem = T.decl_buffer(
        (32, 2),
        "uint32",
        scope="tmem",
        layout=TileLayout(S[(32, 2) : (1 @ TLane, 1 @ TCol)]),
        allocated_addr=0,
    )
    output[lane] = tmem[lane, 0]


@T.prim_func
def tile_copy_cast_mul(
    source: T.Buffer((32, 4), "float16"),
    weight: T.Buffer((32, 4), "float16"),
    output: T.Buffer((32, 4), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_f16: T.f16[4]
    weight_f16: T.f16[4]
    source_f32: T.f32[4]
    weight_f32: T.f32[4]
    result_f32: T.f32[4]
    result_f16: T.f16[4]
    Tx.copy(source_f16[:], source[lane, 0:4])
    Tx.copy(weight_f16[:], weight[lane, 0:4])
    Tx.cast(source_f32[:], source_f16[:])
    Tx.cast(weight_f32[:], weight_f16[:])
    Tx.mul(result_f32[:], source_f32[:], weight_f32[:])
    Tx.mul(result_f32[:], result_f32[:], T.float32(0.5))
    Tx.cast(result_f16[:], result_f32[:])
    Tx.copy(output[lane, 0:4], result_f16[:])


@T.prim_func
def tile_directed_rounding(output: T.Buffer((32, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs: T.f32[1]
    rhs: T.f32[1]
    nearest: T.f32[1]
    down: T.f32[1]
    up: T.f32[1]
    zero: T.f32[1]
    lhs[0] = T.float32(1)
    rhs[0] = T.float32(5.960464477539063e-08)
    Tx.add(nearest, lhs, rhs, rounding_mode="rn")
    Tx.add(down, lhs, rhs, rounding_mode="rm")
    Tx.add(up, lhs, rhs, rounding_mode="rp")
    Tx.add(zero, lhs, rhs, rounding_mode="rz")
    output[lane, 0] = nearest[0]
    output[lane, 1] = down[0]
    output[lane, 2] = up[0]
    output[lane, 3] = zero[0]


@T.prim_func
def tile_reductions(source: T.Buffer((32, 8), "float32"), output: T.Buffer((32, 4), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    row: T.f32[8]
    sum_value: T.f32[1]
    max_value: T.f32[1]
    min_value: T.f32[1]
    Tx.copy(row[:], source[lane, 0:8])
    Tx.sum(sum_value, row)
    Tx.max(max_value, row, dispatch="local")
    Tx.min(min_value, row, dispatch="local")
    output[lane, 0] = sum_value[0]
    output[lane, 1] = max_value[0]
    output[lane, 2] = min_value[0]
    sum_value[0] = T.float32(10)
    Tx.sum(sum_value, row, accum=True)
    output[lane, 3] = sum_value[0]


@T.prim_func
def wg_local_layout_roundtrip(
    source: T.Buffer((128, 4), "float32"), output: T.Buffer((128, 4), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    local = T.alloc_buffer((4,), "float32", scope="local")
    tile = local.view(128, 4, layout=wg_local_layout(4))
    Tx.wg.copy(tile[:, :], source[:, :])
    Tx.wg.mul(tile[:, :], tile[:, :], T.float32(2))
    Tx.wg.copy(output[:, :], tile[:, :])


@T.prim_func
def owner_driven_fp8_register_to_shared(
    source: T.Buffer((128, 8), "float8_e4m3fn"),
    output: T.Buffer((128, 8), "float8_e4m3fn"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    tid = T.thread_id_in_wg([128])
    tile = T.wg_reg_tile(8, dtype="float8_e4m3fn")
    local = tile.local()
    shared = T.alloc_buffer(
        (2, 128, 32),
        "float8_e4m3fn",
        scope="shared",
        layout=_TEST_MMA_FP8_128X32_LAYOUT,
    )
    stage: T.int32
    stage = 1
    for col in T.serial(8):
        local[col] = source[tid, col]
    Tx.wg.copy(shared[stage, :, 8:16], tile, dispatch="reg")
    T.cuda.cta_sync()
    for col in T.serial(8):
        output[tid, col] = shared[stage, tid, 8 + col]


@T.prim_func
def owner_driven_tile_pointwise(
    source: T.Buffer((128, 16), "float16"),
    weight: T.Buffer((128, 16), "float16"),
    scale_input: T.Buffer((128,), "float32"),
    output: T.Buffer((128, 16), "float16"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    tid = T.thread_id_in_wg([128])
    source_f32 = T.alloc_buffer((128, 16), "float32", scope="local", layout=wg_local_layout(16))
    weight_f32 = T.alloc_buffer((128, 16), "float32", scope="local", layout=wg_local_layout(16))
    result_f16 = T.alloc_buffer((128, 16), "float16", scope="local", layout=wg_local_layout(16))
    copied_f16 = T.alloc_buffer((128, 16), "float16", scope="local", layout=wg_local_layout(16))
    for col in T.serial(16):
        source_f32[tid, col] = T.cast(source[tid, col], "float32")
        weight_f32[tid, col] = T.cast(weight[tid, col], "float32")
    scale: T.float32
    scale = scale_input[tid]
    Tx.wg.fma(source_f32[:, :], source_f32[:, :], weight_f32[:, :], T.float32(0.25))
    Tx.wg.mul(source_f32[:, :], source_f32[:, :], scale)
    Tx.wg.cast(result_f16[:, :], source_f32[:, :])
    for col in T.serial(16):
        copied_f16[tid, col] = result_f16[tid, col]
        output[tid, col] = copied_f16[tid, col]


@T.prim_func
def owner_driven_scalar_load_fallback(
    source: T.Buffer((128, 16), "float32"),
    scale: T.Buffer((128,), "float32"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    tid = T.thread_id_in_wg([128])
    tile = T.alloc_buffer((128, 16), "float32", scope="local", layout=wg_local_layout(16))
    for col in T.serial(16):
        tile[tid, col] = source[tid, col]
    Tx.wg.mul(tile[:, :], tile[:, :], scale[tid])
    for col in T.serial(16):
        output[tid, col] = tile[tid, col]


@T.prim_func
def owner_driven_shifted_alias_fallback(output: T.Buffer((128, 17), "float32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    tid = T.thread_id_in_wg([128])
    tile = T.alloc_buffer((128, 17), "float32", scope="local", layout=wg_local_layout(17))
    for col in T.serial(17):
        tile[tid, col] = T.cast(tid * 100 + col, "float32")
    Tx.wg.add(tile[:, 1:17], tile[:, 0:16], T.float32(0))
    for col in T.serial(17):
        output[tid, col] = tile[tid, col]


@T.prim_func
def owner_driven_nonexact_zero_row_min_fallback(output: T.Buffer((128, 16), "float32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    tid = T.thread_id_in_wg([128])
    source = T.alloc_buffer((128, 16), "float32", scope="local", layout=wg_local_layout(16))
    destination = T.alloc_buffer((128, 16), "float32", scope="local", layout=wg_local_layout(16))
    scale = T.alloc_buffer((1,), "float32", scope="local")
    scale[0] = T.float32(1)
    for col in T.serial(16):
        source[tid, col] = T.cast(tid * 16 + col, "float32")
    Tx.wg.mul(destination[:, :], source[tid - tid : tid - tid + 128, :], scale[0])
    for col in T.serial(16):
        output[tid, col] = destination[tid, col]


@T.prim_func
def owner_driven_dynamic_col_min_fallback(output: T.Buffer((128, 16), "float32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    tid = T.thread_id_in_wg([128])
    source = T.alloc_buffer((128, 16), "float32", scope="local", layout=wg_local_layout(16))
    destination = T.alloc_buffer((128, 16), "float32", scope="local", layout=wg_local_layout(16))
    col_min = T.alloc_buffer((1,), "int32", scope="local")
    col_min[0] = 7
    for col in T.serial(16):
        source[tid, col] = T.cast(tid * 16 + col, "float32")
    Tx.wg.mul(
        destination[:, :],
        source[
            :,
            T.if_then_else(col_min[0] == col_min[0], 0, col_min[0]) : T.if_then_else(
                col_min[0] == col_min[0], 0, col_min[0]
            )
            + 16,
        ],
        T.float32(1),
    )
    for col in T.serial(16):
        output[tid, col] = destination[tid, col]


@T.prim_func
def owner_driven_nonexact_zero_scalar_index_fallback(output: T.Buffer((128, 16), "float32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    tid = T.thread_id_in_wg([128])
    tile = T.alloc_buffer((128, 16), "float32", scope="local", layout=wg_local_layout(16))
    scale = T.alloc_buffer((1,), "float32", scope="local")
    scale[0] = T.float32(2)
    for col in T.serial(16):
        tile[tid, col] = T.cast(tid * 16 + col, "float32")
    Tx.wg.mul(tile[:, :], tile[:, :], scale[tid - tid])
    for col in T.serial(16):
        output[tid, col] = tile[tid, col]


@T.prim_func
def tcgen_atom_layout_roundtrip(
    source: T.Buffer((128, 8), "float32"), output: T.Buffer((128, 8), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    _lane = T.lane_id([32])
    local = T.alloc_buffer((8,), "float32", scope="local")
    tile = local.view(128, 8, layout=tcgen05_atom_layout("16x256b", (128, 8), "float32"))
    Tx.wg.copy(tile[:, :], source[:, :])
    Tx.wg.copy(output[:, :], tile[:, :])


@T.prim_func
def warp_gemm_bf16_m16n8k16(
    left: T.Buffer((16, 16), "bfloat16"),
    right: T.Buffer((16, 8), "bfloat16"),
    accumulator: T.Buffer((16, 8), "float32"),
    product: T.Buffer((16, 8), "float32"),
    accumulated: T.Buffer((16, 8), "float32"),
    legacy_order_product: T.Buffer((16, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_fragment = T.alloc_buffer(
        (16, 16), "bfloat16", scope="local", layout=_TEST_WARP_GEMM_A_FRAG
    )
    legacy_order_left_fragment = T.alloc_buffer(
        (16, 16), "bfloat16", scope="local", layout=_TEST_WARP_GEMM_A_FRAG
    )
    right_fragment = T.alloc_buffer(
        (16, 8), "bfloat16", scope="local", layout=_TEST_WARP_GEMM_B_FRAG
    )
    accumulator_fragment = T.alloc_buffer(
        (16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG
    )
    product_fragment = T.alloc_buffer(
        (16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG
    )
    accumulated_fragment = T.alloc_buffer(
        (16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG
    )
    legacy_order_product_fragment = T.alloc_buffer(
        (16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG
    )

    left_registers = left_fragment.local(8)
    for slot in T.unroll(8):
        packed = slot % 2
        row_high = (slot // 2) % 2
        k_high = slot // 4
        row = lane // 4 + 8 * row_high
        col = 2 * (lane % 4) + packed + 8 * k_high
        left_registers[slot] = left[row, col]

    # This legacy flat-slot order swaps the high M and K bits relative to
    # the declared A fragment TileLayout.
    legacy_order_left_registers = legacy_order_left_fragment.local(8)
    for slot in T.unroll(8):
        packed = slot % 2
        k_high = (slot // 2) % 2
        row_high = slot // 4
        row = lane // 4 + 8 * row_high
        col = 2 * (lane % 4) + packed + 8 * k_high
        legacy_order_left_registers[slot] = left[row, col]

    right_registers = right_fragment.local(4)
    for slot in T.unroll(4):
        packed = slot % 2
        k_high = slot // 2
        row = 2 * (lane % 4) + packed + 8 * k_high
        col = lane // 4
        right_registers[slot] = right[row, col]

    accumulator_registers = accumulator_fragment.local(4)
    product_registers = product_fragment.local(4)
    accumulated_registers = accumulated_fragment.local(4)
    legacy_order_product_registers = legacy_order_product_fragment.local(4)
    for slot in T.unroll(4):
        col_low = slot % 2
        row_high = slot // 2
        row = lane // 4 + 8 * row_high
        col = 2 * (lane % 4) + col_low
        accumulator_registers[slot] = accumulator[row, col]
        product_registers[slot] = T.float32(-777)
        accumulated_registers[slot] = T.float32(-888)
        legacy_order_product_registers[slot] = T.float32(-999)

    Tx.warp.gemm(
        product_fragment,
        left_fragment,
        right_fragment,
        accumulator_fragment,
        transpose_A=False,
        transpose_B=False,
        alpha=1.0,
        beta=0.0,
    )
    Tx.warp.gemm(
        accumulated_fragment,
        left_fragment,
        right_fragment,
        accumulator_fragment,
        transpose_A=False,
        transpose_B=False,
        alpha=1.0,
        beta=1.0,
    )
    Tx.warp.gemm(
        legacy_order_product_fragment,
        legacy_order_left_fragment,
        right_fragment,
        accumulator_fragment,
        transpose_A=False,
        transpose_B=False,
        alpha=1.0,
        beta=0.0,
    )

    for slot in T.unroll(4):
        col_low = slot % 2
        row_high = slot // 2
        row = lane // 4 + 8 * row_high
        col = 2 * (lane % 4) + col_low
        product[row, col] = product_registers[slot]
        accumulated[row, col] = accumulated_registers[slot]
        legacy_order_product[row, col] = legacy_order_product_registers[slot]


@T.prim_func
def dense_gemm_async_cta1(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    right_shared = T.alloc_buffer(
        (8, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
            mma_m=128,
            mma_n=8,
        )
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=True,
            dispatch="tcgen05",
            cta_group=1,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_fp8_gemm_async_cta1(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((8, 128), "float8_e4m3fn"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 128), "float8_e4m3fn", scope="shared", layout=_TEST_MMA_FP8_128X128_LAYOUT
    )
    right_shared = T.alloc_buffer(
        (8, 128), "float8_e4m3fn", scope="shared", layout=_TEST_MMA_FP8_8X128_LAYOUT
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_dynamic_right_index(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_start = T.alloc_buffer((1,), "int32", scope="local")
    left_shared = T.alloc_buffer(
        (128, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    right_shared = T.alloc_buffer(
        (16, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    right_start[0] = 1
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[right_start[0] : right_start[0] + 8, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def inactive_gemm_async_is_noop(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    right_shared = T.alloc_buffer(
        (8, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    for _step in T.serial(1):
        if lane >= 0:
            continue
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
    if lane == 0:
        output[0] = 7


@T.prim_func
def dense_gemm_async_two_clusters(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    output: T.Buffer((256, 8), "float32"),
):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    right_shared = T.alloc_buffer(
        (8, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 128 : (cta + 1) * 128, :])
        Tx.copy(right_shared[:, :], right[cta * 8 : (cta + 1) * 8, :])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def shared_permute_layout_roundtrip(
    source: T.Buffer((128,), "uint32"), output: T.Buffer((128,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_shared = T.alloc_buffer((128,), "uint32", scope="shared")
    permuted_shared = T.alloc_buffer(
        (128,), "uint32", scope="shared", layout=_TEST_PERMUTED_SHARED_LAYOUT
    )
    if lane == 0:
        Tx.copy(source_shared[:], source[:])
    T.cuda.warp_sync()
    Tx.warp.permute_layout(permuted_shared[:], source_shared[:])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.copy(output[:], permuted_shared[:])


@T.prim_func
def shared_permute_layout_zero_fills_uninitialized_padding(
    source: T.Buffer((16,), "uint32"), output: T.Buffer((128,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_shared = T.alloc_buffer((128,), "uint32", scope="shared")
    permuted_shared = T.alloc_buffer(
        (128,), "uint32", scope="shared", layout=_TEST_PERMUTED_SHARED_LAYOUT
    )
    if lane == 0:
        Tx.copy(source_shared[0:16], source[:])
    T.cuda.warp_sync()
    Tx.warp.permute_layout(permuted_shared[:], source_shared[:])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.copy(output[:], permuted_shared[:])


@T.prim_func
def shared_permute_layout_zero_fills_fp16_padding(
    source: T.Buffer((16,), "float16"), output: T.Buffer((128,), "float16")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_shared = T.alloc_buffer((128,), "float16", scope="shared")
    permuted_shared = T.alloc_buffer(
        (128,), "float16", scope="shared", layout=_TEST_PERMUTED_SHARED_LAYOUT
    )
    if lane == 0:
        Tx.copy(source_shared[0:16], source[:])
    T.cuda.warp_sync()
    Tx.warp.permute_layout(permuted_shared[:], source_shared[:])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.copy(output[:], permuted_shared[:])


@T.prim_func
def shared_permute_layout_zero_fills_bf16_padding(
    source: T.Buffer((16,), "bfloat16"), output: T.Buffer((128,), "bfloat16")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_shared = T.alloc_buffer((128,), "bfloat16", scope="shared")
    permuted_shared = T.alloc_buffer(
        (128,), "bfloat16", scope="shared", layout=_TEST_PERMUTED_SHARED_LAYOUT
    )
    if lane == 0:
        Tx.copy(source_shared[0:16], source[:])
    T.cuda.warp_sync()
    Tx.warp.permute_layout(permuted_shared[:], source_shared[:])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.copy(output[:], permuted_shared[:])


@T.prim_func
def shared_permute_layout_zero_fills_fp8_padding(
    source: T.Buffer((16,), "float8_e4m3fn"), output: T.Buffer((128,), "float8_e4m3fn")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source_shared = T.alloc_buffer((128,), "float8_e4m3fn", scope="shared")
    permuted_shared = T.alloc_buffer(
        (128,), "float8_e4m3fn", scope="shared", layout=_TEST_PERMUTED_SHARED_LAYOUT
    )
    if lane == 0:
        Tx.copy(source_shared[0:16], source[:])
    T.cuda.warp_sync()
    Tx.warp.permute_layout(permuted_shared[:], source_shared[:])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.copy(output[:], permuted_shared[:])


@T.prim_func
def fp8_scale_permute_tmem_roundtrip(
    source: T.Buffer((128,), "uint32"),
    shared_output: T.Buffer((128, 4), "float32"),
    tmem_output: T.Buffer((128, 4), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    dense_shared = T.alloc_buffer((128,), "uint32", scope="shared")
    permuted_shared = dense_shared.view(128, layout=_TEST_PERMUTED_SHARED_LAYOUT)
    scale_shared = dense_shared.view("float8_e8m0fnu").view(
        128, 16, layout=_TEST_FP8_SF_VIEW_LAYOUT
    )
    scale_tmem = T.decl_buffer(
        (128, 16),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1, sf_reuse=4),
        allocated_addr=0,
    )
    if (warp == 0) and (lane == 0):
        Tx.copy(dense_shared[:], source[:])
    T.cuda.cta_sync()
    if warp == 0:
        Tx.warp.permute_layout(permuted_shared[:], dense_shared[:])
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        Tx.copy_async(scale_tmem[:, :], scale_shared[:, :])
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        for read_row in T.serial(128):
            for group in T.serial(4):
                shared_output[read_row, group] = scale_shared[read_row, group * 4]
                tmem_output[read_row, group] = scale_tmem[read_row, group * 4]


@T.prim_func
def permuted_global_layout_read(
    source: T.Buffer((128,), "uint32", layout=_TEST_PERMUTED_SHARED_LAYOUT),
    output: T.Buffer((128,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for chunk in T.serial(4):
        index = T.meta_var(lane * 4 + chunk)
        output[index] = source[index]


@T.prim_func
def global_permute_layout_roundtrip(
    source: T.Buffer((128,), "uint32"),
    output: T.Buffer((128,), "uint32", layout=_TEST_PERMUTED_SHARED_LAYOUT),
):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    Tx.warp.permute_layout(output[:], source[:], dispatch="warp_xor_swizzle")


@T.prim_func
def dense_gemm_async_tmem_a_transposed_b(
    left: T.Buffer((128, 16), "float16"),
    right_transposed: T.Buffer((16, 16), "float16"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer(
        (16, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    left_tmem = T.decl_buffer(
        (128, 16),
        "float16",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (128, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=8,
    )
    if lane == 0:
        Tx.copy(right_shared[:, :], right_transposed[:, :])
        for row in T.serial(128):
            for col in T.serial(16):
                left_tmem[row, col] = left[row, col]
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_tmem[:, :],
            right_shared[:, :],
            transB=True,
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(16):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_cta_group2(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    output: T.Buffer((256, 16), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    right_shared = T.alloc_buffer(
        (8, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    accumulator = T.decl_buffer(
        (128, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 128 : (cta + 1) * 128, :])
        Tx.copy(right_shared[:, :], right[cta * 8 : (cta + 1) * 8, :])
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(16):
                output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_tmem_a_cta_group2(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    output: T.Buffer((256, 16), "float32"),
):
    """CTA-pair MMA whose A operand is each CTA's own TMEM shard."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer(
        (8, 16), "float16", scope="shared", layout=_TEST_MMA_F16_32B_LAYOUT
    )
    left_tmem = T.decl_buffer(
        (128, 16),
        "float16",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (128, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=8,
    )
    if lane == 0:
        Tx.copy(right_shared[:, :], right[cta * 8 : (cta + 1) * 8, :])
        for row in T.serial(128):
            for col in T.serial(16):
                left_tmem[row, col] = left[cta * 128 + row, col]
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_tmem[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(16):
                output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def block_scaled_fp8_gemm_packed_scales(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((8, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 2), "float8_e8m0fnu"),
    scale_b: T.Buffer((8, 2), "float8_e8m0fnu"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 128), "float8_e4m3fn", scope="shared", layout=_TEST_MMA_FP8_128X128_LAYOUT
    )
    right_shared = T.alloc_buffer(
        (8, 128), "float8_e4m3fn", scope="shared", layout=_TEST_MMA_FP8_8X128_LAYOUT
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 8),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=4),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 8),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=4),
        allocated_addr=24,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(128):
            for packed_call in T.serial(2):
                scale_a_tmem[row, packed_call * 4] = scale_a[row, packed_call]
        for row in T.serial(8):
            for packed_call in T.serial(2):
                scale_b_tmem[row, packed_call * 4] = scale_b[row, packed_call]
    T.cuda.warp_sync()
    if lane == 0:
        for packed_call in T.serial(2):
            Tx.gemm_async(
                accumulator[:, :],
                left_shared[:, :],
                right_shared[:, :],
                SFA=scale_a_tmem[:, :],
                SFB=scale_b_tmem[:, :],
                accum=packed_call != 0,
                dispatch="tcgen05",
                cta_group=1,
            )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def block_scaled_nvfp4_gemm(
    left_packed: T.Buffer((128, 32), "uint8"),
    right_packed: T.Buffer((8, 32), "uint8"),
    scale_a: T.Buffer((128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((8, 4), "float8_e4m3fn"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_TEST_MMA_PACKED_FP4_128X32_LAYOUT
    )
    right_shared_packed = T.alloc_buffer(
        (8, 32), "uint8", scope="shared", layout=_TEST_MMA_PACKED_FP4_8X32_LAYOUT
    )
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=24,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[:, :])
        Tx.copy(right_shared_packed[:, :], right_packed[:, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[row, scale_index]
        for row in T.serial(8):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[row, scale_index]
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def block_scaled_nvfp4_gemm_cta_group2_scale_rows(
    left_packed: T.Buffer((2, 128, 32), "uint8"),
    right_packed: T.Buffer((2, 128, 32), "uint8"),
    scale_a: T.Buffer((2, 128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((2, 256, 4), "float8_e4m3fn"),
    output: T.Buffer((2, 128, 256), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_TEST_MMA_PACKED_FP4_128X32_LAYOUT
    )
    right_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_TEST_MMA_PACKED_FP4_128X32_LAYOUT
    )
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    accumulator = T.decl_buffer(
        (128, 256),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 256),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=256,
    )
    scale_b_tmem = T.decl_buffer(
        (256, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(256, SF_K=4, sf_per_mma=4),
        allocated_addr=264,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[cta, :, :])
        Tx.copy(right_shared_packed[:, :], right_packed[cta, :, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[cta, row, scale_index]
        for row in T.serial(256):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[cta, row, scale_index]
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(256):
                output[cta, row, col] = accumulator[row, col]


@T.prim_func
def tcgen_tmem_to_local_roundtrip(
    source: T.Buffer((128, 4), "float32"), output: T.Buffer((128, 4), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (128, 4),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 4),
        allocated_addr=0,
    )
    local = T.alloc_buffer((4,), "float32", scope="local")
    tile = local.view(128, 4, layout=wg_local_layout(4))
    for col in T.serial(4):
        tmem[row, col] = source[row, col]
    T.cuda.cta_sync()
    Tx.wg.copy_async(tile[:, :], tmem[:, :], dispatch="tmem<->local")
    T.ptx.tcgen05.wait__ld.sync.aligned()
    Tx.wg.copy(output[:, :], tile[:, :])


@T.prim_func
def tcgen_local_to_tmem_roundtrip(
    source: T.Buffer((128, 4), "float32"), output: T.Buffer((128, 4), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer(
        (128, 4),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 4),
        allocated_addr=0,
    )
    local = T.alloc_buffer((4,), "float32", scope="local")
    tile = local.view(128, 4, layout=wg_local_layout(4))
    Tx.wg.copy(tile[:, :], source[:, :])
    Tx.wg.copy_async(tmem[:, :], tile[:, :], dispatch="tmem<->local")
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    for col in T.serial(4):
        output[row, col] = tmem[row, col]


@T.prim_func
def inactive_tcgen_transfer_is_noop(output: T.Buffer((4,), "int32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    tmem = T.decl_buffer(
        (128, 4),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 4),
        allocated_addr=0,
    )
    local = T.alloc_buffer((4,), "float32", scope="local")
    tile = local.view(128, 4, layout=wg_local_layout(4))
    for _step in T.serial(1):
        if lane >= 0:
            continue
        Tx.wg.copy_async(tmem[:, :], tile[:, :])
    if lane == 0:
        output[warp] = 7


@T.prim_func
def tcgen_shared_to_tmem_replica(
    source: T.Buffer((32, 4), "float32"), output: T.Buffer((128, 4), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 4), "float32", scope="shared")
    replicated = T.decl_buffer(
        (32, 4),
        "float32",
        scope="tmem",
        layout=TileLayout(S[(32, 4) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]),
        allocated_addr=0,
    )
    physical = T.decl_buffer(
        (128, 4),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 4),
        allocated_addr=0,
    )
    if warp == 0:
        for col in T.serial(4):
            shared[lane, col] = source[lane, col]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.copy_async(
            replicated[:, :],
            shared[:, :],
            dispatch="smem->tmem",
            shape="32x128b",
            multicast="warpx4",
        )
    T.cuda.cta_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(4):
        output[row, col] = physical[row, col]


@T.prim_func
def tcgen_shared_to_tmem_rank3(
    source: T.Buffer((4, 32, 16), "uint8"),
    output: T.Buffer((128, 4, 16), "uint8"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer(
        (4, 32, 16),
        "uint8",
        scope="shared",
        layout=TileLayout(S[(4, 32, 16) : (512, 16, 1)]),
    )
    replicated = T.decl_buffer(
        (4, 32, 16),
        "uint8",
        scope="tmem",
        layout=TileLayout(S[(4, 32, 16) : (16 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]),
        allocated_addr=0,
    )
    physical = T.decl_buffer(
        (128, 64),
        "uint8",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 64),
        allocated_addr=0,
    )
    for col in T.serial(16):
        shared[warp, lane, col] = source[warp, lane, col]
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        Tx.copy_async(replicated[:, :, :], shared[:, :, :], dispatch="smem->tmem")
    T.cuda.cta_sync()
    physical_row = T.meta_var(warp * 32 + lane)
    for outer in T.serial(4):
        for col in T.serial(16):
            output[physical_row, outer, col] = physical[physical_row, outer * 16 + col]


@T.prim_func
def tcgen_scale_bitcast_shared_to_tmem(
    source: T.Buffer((128, 4), "uint8"), output: T.Buffer((128, 4), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
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
        shared[row, col] = source[row, col]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        Tx.copy_async(scale_tmem[:, :], shared[:, :])
    T.cuda.cta_sync()
    for col in T.serial(4):
        output[row, col] = scale_tmem[row, col]


@T.prim_func
def tcgen_scale_bitcast_cta_group2(
    source: T.Buffer((2, 128, 4), "uint8"), output: T.Buffer((2, 128, 4), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
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
    if (cta == 0) and (warp == 0) and (lane == 0):
        Tx.copy_async(scale_tmem[:, :], shared[:, :], cta_group=2)
    T.cuda.cluster_sync()
    for col in T.serial(4):
        output[cta, row, col] = scale_tmem[row, col]


@T.prim_func
def tcgen_float16_cta_group2(
    source: T.Buffer((2, 32, 8), "float16"),
    output: T.Buffer((2, 128, 8), "float16"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 8), "float16", scope="shared")
    replicated = T.decl_buffer(
        (32, 8),
        "float16",
        scope="tmem",
        layout=TileLayout(S[(32, 8) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]),
        allocated_addr=0,
    )
    physical = T.decl_buffer(
        (128, 8),
        "float16",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    if warp == 0:
        for col in T.serial(8):
            shared[lane, col] = source[cta, lane, col]
    T.cuda.cluster_sync()
    if (cta == 0) and (warp == 0) and (lane == 0):
        Tx.copy_async(
            replicated[:, :],
            shared[:, :],
            dispatch="smem->tmem",
            cta_group=2,
        )
    T.cuda.cluster_sync()
    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(8):
        output[cta, row, col] = physical[row, col]


@T.prim_func
def tcgen_lifecycle_single_cta(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 64)
    if lane == 0:
        output[0] = address[0]
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(T.uint32(0), 64)


@T.prim_func
def tcgen_lifecycle_two_cta(output: T.Buffer((2,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(T.address_of(address[0]), 128)
    if lane == 0:
        output[cta] = address[0]
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
    T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(T.uint32(0), 128)


@T.prim_func
def tcgen_commit_mbarrier(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    if lane == 0:
        output[0] = T.uint32(1)


@T.prim_func
def tcgen_commit_runtime_multicast(output: T.Buffer((2,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pool = T.SMEMPool()
    barrier = TCGen05Bar(pool, 1)
    pool.commit()
    barrier.init(1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    pair_leader = T.meta_var((cta // 2) * 2)
    pair_mask: T.int32
    pair_mask = 0
    pair_mask = pair_mask | 1 << pair_leader
    pair_mask = pair_mask | 1 << pair_leader + 1
    mapped_barrier = T.alloc_local((1,), "uint64")
    T.ptx.mapa.u64(mapped_barrier[0], barrier.ptr_to([0]), T.uint32(pair_leader))
    remote_ptr: T.let[
        T.Var(name="tcgen_runtime_remote_barrier", ty=PointerType(PrimType("uint64"), "shared"))
    ] = T.reinterpret(
        PointerType(PrimType("uint64"), "shared"),
        mapped_barrier[0],
    )
    remote = T.decl_buffer((1,), "uint64", data=remote_ptr, scope="shared")
    if cta == pair_leader and lane == 0:
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
            T.address_of(remote[0]), T.uint16(pair_mask)
        )
    barrier.wait(0, 0)
    if lane == 0:
        output[cta] = T.uint32(1)


@T.prim_func
def divergent_loop_control(output: T.Buffer((128,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for step in T.serial(4):
        if lane == step:
            continue
        if lane < 4:
            if step == 2:
                break
        output[lane * 4 + step] = T.float32(1)


@T.prim_func
def divergent_while_control(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    while lane < 16:
        output[lane] = T.float32(2)
        break


@T.prim_func
def mbarrier_wait_after_full_warp_continue(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    for _step in T.serial(1):
        if lane >= 0:
            continue
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        output[lane] = 1


@T.prim_func
def bar_sync_after_full_warp_continue(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for _step in T.serial(1):
        if lane >= 0:
            continue
        T.ptx.bar.sync(T.uint32(lane), T.uint32(32))
        output[lane] = 1


@T.prim_func
def cta_sync_after_full_warp_continue(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for _step in T.serial(1):
        if lane >= 0:
            continue
        T.cuda.cta_sync()
        output[lane] = 1


@T.prim_func
def dynamic_for_after_full_warp_continue(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for _step in T.serial(1):
        if lane >= 0:
            continue
        for _inner in T.serial(0, lane):
            output[lane] = 1


@T.prim_func
def bound_launch_topology():
    T.device_entry()
    max_ctas: T.let = T.int32(4)
    should_cap: T.let = max_ctas > T.int32(3)
    cta_count: T.let = T.Select(should_cap, T.min(max_ctas - T.int32(1), T.int32(3)), T.int32(1))
    _cta = T.cta_id([cta_count])
    _warp = T.warp_id([2])
    _lane = T.lane_id([32])


@T.prim_func
def scalar_expression_mix(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    wide_lane: T.let = T.cast(lane, "int64")
    shifted: T.let = wide_lane - T.int64(17)
    quotient: T.let = shifted // T.int64(5)
    remainder: T.let = shifted % T.int64(5)
    clamped: T.let = T.min(T.max(quotient, T.int64(-2)), T.int64(2))
    use_clamped: T.let = ((lane < 5) and not (lane == 2)) or lane >= 29
    selected: T.let = T.Select(use_clamped, clamped, remainder)
    output[lane] = T.cast(selected, "float32")


@T.prim_func
def nested_mask_parent_scope(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    enabled: T.let = T.bool(True)
    disabled: T.let = T.bool(False)
    if lane < 16:
        output[lane] = 1
        if (lane < 4) or enabled:
            output[lane] = output[lane] + 1
        if not (lane < 8):
            output[lane] = output[lane] + 2
        if disabled and lane < 8:
            output[lane] = output[lane] + 4


@T.prim_func
def fixed_width_integer_expression_mix(output: T.Buffer((32, 7), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value_i32: T.let = lane - T.int32(1)
    value_u32: T.let = T.cast(value_i32, "uint32")
    value_u64: T.let = T.cast(value_u32, "uint64")
    value_i64: T.let = T.cast(value_u64, "int64")
    narrowed_i32: T.let = T.cast(value_i64, "int32")
    narrowed_i8: T.let = T.cast(narrowed_i32, "int8")
    widened_u16: T.let = T.cast(value_u32, "uint16")
    nonzero: T.let = T.cast(value_u32, "bool")
    flag: T.let = T.cast(nonzero, "uint32")
    incremented: T.let = value_u64 + T.uint64(1)
    decremented: T.let = value_u64 - T.uint64(1)
    _selected: T.let = T.Select(value_u64 != T.uint64(0), incremented, decremented)
    output[T.cast(lane, "uint64"), 0] = _selected
    output[lane, 1] = T.cast(narrowed_i8, "uint64")
    output[lane, 2] = T.cast(widened_u16, "uint64")
    output[lane, 3] = T.cast(flag, "uint64")
    output[lane, 4] = T.cast(value_u32 // T.uint32(5), "uint64")
    output[lane, 5] = T.cast(value_u32 % T.uint32(5), "uint64")
    output[lane, 6] = T.cast(value_u64 + T.uint64(3) < T.uint64(10), "uint64")


@T.prim_func
def raw_scalar_call_mix(source: T.Buffer((32,), "float32"), output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    rank: T.let = T.cuda.thread_rank()
    elected: T.let = T.cuda.elect_sync()
    lane_u32: T.let = T.cast(rank, "uint32")
    low_nibble: T.let = T.bitwise_and(lane_u32, T.uint32(15))
    shifted: T.let = T.bitwise_or(
        T.shift_left(low_nibble, T.uint32(1)), T.shift_right(low_nibble, T.uint32(1))
    )
    mixed: T.let = T.bitwise_xor(shifted, T.uint32(3))
    inverted: T.let = T.bitwise_and(T.bitwise_not(low_nibble), T.uint32(15))
    source_bits: T.let = T.cuda.float_as_uint(source[lane])
    source_value: T.let = T.cuda.uint_as_float(source_bits)
    pair: T.let = T.cuda.make_float2(source_value, T.fma(source_value, T.float32(2), T.float32(1)))
    choose_fma: T.let = T.bitwise_or(T.bitwise_and(rank == 0, elected != T.uint32(0)), rank == 31)
    selected: T.let = T.if_then_else(choose_fma, T.cuda.float2_y(pair), T.cuda.float2_x(pair))
    bitcast_identity: T.let = T.reinterpret("float32", T.cuda.float_as_uint(selected))
    exponential = T.local_scalar("float32")
    T.ptx.ex2.approx.ftz.f32(exponential, T.log(bitcast_identity))
    math_value: T.let = exponential + T.rsqrt(bitcast_identity + T.float32(4))
    output[lane] = (
        math_value
        + T.cast(mixed + inverted, "float32") * T.float32(0.001)
        + T.cast(elected, "float32") * T.float32(0.01)
    )


@T.prim_func
def guarded_if_then_else_load(
    source: T.Buffer((1,), "float32"), output: T.Buffer((32,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.if_then_else(lane == 0, source[lane], T.float32(7))


@T.prim_func
def guarded_select_load(source: T.Buffer((2,), "uint64"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.Select(lane < 2, T.cast(source[lane], "uint32"), T.cast(100 + lane, "uint32"))


@T.prim_func
def unsupported_bfloat_cast(source: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _value: T.let = T.cast(source[lane], "bfloat16")


@T.prim_func
def scalar_buffer_types(
    input_i32: T.Buffer((32,), "int32"),
    output_i32: T.Buffer((32,), "int32"),
    input_u64: T.Buffer((32,), "uint64"),
    output_u64: T.Buffer((32,), "uint64"),
    input_f16: T.Buffer((32,), "float16"),
    output_f16: T.Buffer((32,), "float16"),
    input_bf16: T.Buffer((32,), "bfloat16"),
    output_bf16: T.Buffer((32,), "bfloat16"),
    input_bool: T.Buffer((32,), "bool"),
    output_bool: T.Buffer((32,), "bool"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output_i32[lane] = input_i32[lane] + T.int32(7)
    output_u64[lane] = input_u64[lane] + T.uint64(11)
    output_f16[lane] = T.cast(T.cast(input_f16[lane], "float32") + T.float32(0.5), "float16")
    output_bf16[lane] = T.cast(T.cast(input_bf16[lane], "float32") + T.float32(0.5), "bfloat16")
    output_bool[lane] = input_bool[lane]


@T.prim_func
def float8_buffer_codecs(
    input_e4m3: T.Buffer((32,), "float8_e4m3fn"),
    input_e8m0: T.Buffer((32,), "float8_e8m0fnu"),
    output_e4m3: T.Buffer((32,), "float8_e4m3fn"),
    decoded: T.Buffer((32, 2), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    decoded[lane, 0] = input_e4m3[lane]
    decoded[lane, 1] = input_e8m0[lane]
    output_e4m3[lane] = input_e4m3[lane]


@T.prim_func
def dynamic_rows(input_ptr: T.handle, output_ptr: T.handle):
    rows = T.int32()
    input_buffer = T.match_buffer(input_ptr, (rows, 32), "float32")
    output_buffer = T.match_buffer(output_ptr, (rows, 32), "float32")
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    row: T.int32 = 0
    while row < rows:
        output_buffer[row, lane] = input_buffer[row, lane] + T.float32(2)
        row = row + 1


@T.prim_func
def compose_swizzle_alias(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2, 128, 128), "float16", scope="shared", layout=_TEST_SWIZZLE_LAYOUT)
    dense = T.decl_buffer((32768,), "float16", data=shared.data, scope="shared")
    shared[0, lane, lane] = T.cast(lane + 1, "float16")
    linear: T.let = lane * 8 + lane // 8
    physical: T.let = ((linear ^ ((linear & 56) >> 3)) << 3) + lane % 8
    output[lane] = T.cast(dense[physical], "float32")


@T.prim_func
def physical_address_value():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint64", scope="shared")
    T.evaluate(T.address_of(shared[lane]))


@T.prim_func
def mbarrier_phase_reuse(source: T.Buffer((4,), "float32"), output: T.Buffer((2,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    shared = T.alloc_buffer((4,), "float32", scope="shared", align=128)
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    for phase in T.serial(2):
        if warp == 0:
            if lane == 0:
                if phase == 0:
                    T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
                else:
                    Tx.copy_async(
                        shared[:], source[:], dispatch="tma_auto", mbar=T.address_of(barriers[0])
                    )
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        else:
            if lane == 0:
                T.cuda.mbarrier_wait(T.address_of(barriers[0]), phase)
                output[phase] = phase + 1
        T.cuda.cta_sync()


@T.prim_func
def mbarrier_varying_uniform_phase(output: T.Buffer((2,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    phase: T.int32
    phase = lane - lane
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0:
        if lane == 0:
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
    else:
        if T.cuda.elect_sync():
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), phase)
            output[warp] = 1


@T.prim_func
def ldgsts_copy_roundtrip(source: T.Buffer((8,), "float32"), output: T.Buffer((8,), "float32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float32", scope="shared")
    Tx.copy_async(shared[:], source[:], dispatch="ldgsts")
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    T.cuda.cta_sync()
    if lane < 8:
        output[lane] = shared[lane]


@T.prim_func
def cta_copy_roundtrip(source: T.Buffer((64,), "float32"), output: T.Buffer((64,), "float32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([2])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "float32", scope="shared")
    Tx.cta.copy(shared[:], source[:], dispatch="gmem_smem")
    T.cuda.cta_sync()
    Tx.cta.copy(output[:], shared[:], dispatch="gmem_smem")


@T.prim_func
def cta_ldgsts_copy_roundtrip(
    source: T.Buffer((64,), "float32"), output: T.Buffer((64,), "float32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warp = T.warp_id([2])
    _lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "float32", scope="shared")
    Tx.cta.copy_async(shared[:], source[:], dispatch="ldgsts")
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    T.cuda.cta_sync()
    Tx.cta.copy(output[:], shared[:])


@T.prim_func
def tma_copy_roundtrip(source: T.Buffer((4, 8), "float16"), output: T.Buffer((4, 8), "float16")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 8), "float16", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        Tx.copy_async(
            shared[:, :], source[:, :], dispatch="tma_auto", mbar=T.address_of(barriers[0])
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 64)
    if (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        Tx.copy_async(output[:, :], shared[:, :], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def raw_tma_gather4_bar_address(input_map: T.TensorMap(), output: T.Buffer((4, 4), "float32")):
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
        T.ptx[
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1"
        ](
            T.address_of(shared[0, 0]),
            T.address_of(input_map),
            0,
            0,
            1,
            2,
            3,
            T.cuda.cvta_generic_to_shared(T.address_of(barrier[0])),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 64)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane // 4, lane % 4] = shared[lane // 4, lane % 4]


@T.prim_func
def raw_tma_reduce_add(source: T.Buffer((4,), "float32"), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group"](
            T.address_of(output_map), 0, T.address_of(shared[0])
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def tma_copy_zero_fill_boundary(
    source: T.Buffer((3, 4), "float32"), output: T.Buffer((4, 4), "float32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        start: T.int32 = -1
        Tx.copy_async(
            shared[:, :],
            source[start : start + 4, :],
            dispatch="tma_auto",
            mbar=T.address_of(barriers[0]),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 64)
    if (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        Tx.copy_async(output[:, :], shared[:, :], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def tma_copy_nan_fill_boundary(
    source: T.Buffer((3, 4), "float32"), output: T.Buffer((4, 4), "float32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        start: T.int32 = -1
        Tx.copy_async(
            shared[:, :],
            source[start : start + 4, :],
            dispatch="tma_explicit",
            mbar=T.address_of(barriers[0]),
            oob="nan",
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 64)
    if (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        Tx.copy_async(output[:, :], shared[:, :], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def tma_scalar_hoist_fallbacks(
    source: T.Buffer((8,), "float32"),
    global_start: T.Buffer((1,), "int32"),
    output: T.Buffer((4, 8), "float32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    local_start = T.alloc_buffer((1,), "int32", scope="local")
    alias_storage = T.alloc_buffer((2,), "int32", scope="local")
    conditional_start = T.alloc_buffer((1,), "int32", scope="local")
    alias_start = T.decl_buffer(
        (1,), "int32", data=alias_storage.data, elem_offset=1, scope="local"
    )
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    if (warp == 1) and (lane == 0):
        local_start[0] = 0
        alias_start[0] = 0
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        Tx.copy_async(shared[:], source[:], dispatch="tma_auto", mbar=T.address_of(barriers[0]))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 32)
    if (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        Tx.copy_async(
            output[0, global_start[0] : global_start[0] + 8], shared[:], dispatch="tma_auto"
        )
        Tx.copy_async(
            output[1, local_start[lane] : local_start[lane] + 8], shared[:], dispatch="tma_auto"
        )
        Tx.copy_async(
            output[2, alias_start[0] : alias_start[0] + 8], shared[:], dispatch="tma_auto"
        )
        Tx.copy_async(
            output[
                3,
                T.if_then_else(lane == 0, 0, conditional_start[0]) : T.if_then_else(
                    lane == 0, 0, conditional_start[0]
                )
                + 8,
            ],
            shared[:],
            dispatch="tma_auto",
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def tma_scalar_hoist_respects_boundary(
    source: T.Buffer((8,), "float32"), output: T.Buffer((16,), "float32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    start = T.alloc_buffer((1,), "int32", scope="local")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    if (warp == 1) and (lane == 0):
        start[0] = 0
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        Tx.copy_async(shared[:], source[:], dispatch="tma_auto", mbar=T.address_of(barriers[0]))
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 32)
    if (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        Tx.copy_async(output[start[0] : start[0] + 8], shared[:], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
    T.cuda.cta_sync()
    if (warp == 1) and (lane == 0):
        start[0] = 8
        Tx.copy_async(output[start[0] : start[0] + 8], shared[:], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def tma_copy_cluster_multicast(
    source: T.Buffer((8,), "float32"), output: T.Buffer((2, 8), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.copy_async(
            shared[:],
            source[:],
            dispatch="tma_auto",
            mbar=T.address_of(barriers[0]),
            cta_group=2,
            cta_mask=3,
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 64)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane == 0:
        Tx.copy_async(output[cta, :], shared[:], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def dsmem_copy_remote_cta(output: T.Buffer((8,), "float32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source = T.alloc_buffer((8,), "float32", scope="shared", align=128)
    destination = T.alloc_buffer((8,), "float32", scope="shared", align=128)
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 1) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    if (cta == 0) and (lane < 8):
        source[lane] = T.cast(lane + 5, "float32")
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.copy_async(
            destination[:],
            source[:],
            dispatch="dsmem",
            mbar=T.address_of(barriers[0]),
            remote_cta_id=1,
        )
    if (cta == 1) and (lane == 0):
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 32)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if (cta == 1) and (lane == 0):
        Tx.copy_async(output[:], destination[:], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def tma_copy_transaction_mismatch(
    source: T.Buffer((4, 8), "float16"), output: T.Buffer((4, 8), "float16")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 8), "float16", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if (warp == 0) and (lane == 0):
        Tx.copy_async(
            shared[:, :], source[:, :], dispatch="tma_auto", mbar=T.address_of(barriers[0])
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 68)
    if (warp == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        Tx.copy_async(output[:, :], shared[:, :], dispatch="tma_auto")
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def raw_tma_roundtrip(input_map: T.TensorMap(), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx.prefetch.tensormap(T.address_of(input_map))
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0, 0]), T.address_of(input_map), 0, 0, T.address_of(barriers[0]))
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 48)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, 0, T.address_of(shared[0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def raw_tma_fp4_align8_roundtrip(input_map: T.TensorMap(), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2, 64), "uint8", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0, 0]), T.address_of(input_map), 0, 0, T.address_of(barriers[0]))
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 128)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, 0, T.address_of(shared[0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def raw_tma_dynamic_wait_group_loop(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(1)
        output[0] = 10
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        output[1] = 11


@T.prim_func
def raw_tma_rank3_roundtrip(input_map: T.TensorMap(), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2, 3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0, 0, 0]),
                T.address_of(input_map),
                0,
                0,
                0,
                T.address_of(barriers[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 96)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, 0, 0, T.address_of(shared[0, 0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def raw_tma_split_swizzle_atoms(input_map: T.TensorMap(), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2, 2, 32), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0, 0, 0]),
                T.address_of(input_map),
                0,
                0,
                T.address_of(barriers[0]),
            )
        )
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[1, 0, 0]),
                T.address_of(input_map),
                32,
                0,
                T.address_of(barriers[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 512)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, 0, T.address_of(shared[0, 0, 0])
            )
        )
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 32, 0, T.address_of(shared[1, 0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def raw_tma_zero_fill(input_map: T.TensorMap(), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0, 0]),
                T.address_of(input_map),
                0,
                T.cast(-1, "int32"),
                T.address_of(barriers[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 64)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, 0, T.address_of(shared[0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def raw_tma_swizzle_to_dense(input_map: T.TensorMap(), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8, 8), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0, 0]), T.address_of(input_map), 0, 0, T.address_of(barriers[0]))
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 256)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, 0, T.address_of(shared[0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


def raw_tma_swizzle_to_matching_layout(swizzle_bytes):
    columns = swizzle_bytes // 4
    layout = ComposeLayout(2, swizzle_bytes.bit_length() - 5, 3, TileLayout(S[(8 * columns,)]))

    @T.prim_func
    def kernel(
        input_map: T.TensorMap(), output: T.Buffer((8, columns), "float32")
    ):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        shared = T.alloc_buffer((8, columns), "float32", scope="shared", layout=layout)
        barriers = T.alloc_buffer((1,), "uint64", scope="shared")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()
        if lane == 0:
            T.evaluate(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
                ](T.address_of(shared[0, 0]), T.address_of(input_map), 0, 0, T.address_of(barriers[0]))
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 8 * swizzle_bytes)
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
            for row in T.serial(8):
                for col in T.serial(columns):
                    output[row, col] = shared[row, col]

    return kernel


@T.prim_func
def raw_tma_sm100_barrier_address(
    input_map_even: T.TensorMap(), input_map_odd: T.TensorMap(), output: T.Buffer((2, 4), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
            ](
                T.address_of(shared[0]),
                T.Select(cta == 0, T.address_of(input_map_even), T.address_of(input_map_odd)),
                0,
                0,
                T.cuda.sm100_2sm_leader_smem_addr(T.address_of(barriers[0])),
            )
        )
        if cta == 0:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 32)
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


@T.prim_func
def raw_tma_transaction_mismatch(input_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0, 0]), T.address_of(input_map), 0, 0, T.address_of(barriers[0]))
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 52)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


@T.prim_func
def ordering_only_control_calls(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.griddepcontrol.wait()
    T.cuda.cta_sync()
    output[lane] = lane + 1


@T.prim_func
def mbarrier_missing_arrivals():
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 64)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


@T.prim_func
def mbarrier_remote_cta(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        else:
            remote_barrier = T.alloc_local((1,), "uint64")
            T.ptx.mapa.u64(remote_barrier[0], barriers.ptr_to([0]), T.uint32(0))
            T.ptx.mbarrier.arrive.b64(remote_barrier[0], T.uint32(1), pred=T.bool(True))
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1


@T.prim_func
def scoped_syncs(output: T.Buffer((2, 4, 32), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    T.cuda.warp_sync()
    T.cuda.warpgroup_sync(7)
    T.ptx.bar.sync(T.uint32(6), T.uint32(128))
    T.cuda.cta_sync()
    T.cuda.cluster_sync()
    output[cta, warp, lane] = cta * 10000 + warp * 100 + lane


@T.prim_func
def remote_shared_read_and_warp_reduce(output: T.Buffer((2, 3, 32), "float32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "float32", scope="shared")
    value: T.float32 = T.float32(0)
    if lane == 0:
        shared[0] = T.cast(cta + 1, "float32")
    T.cuda.cluster_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane < 2:
        mapped = T.alloc_local((1,), "uint64")
        T.ptx.mapa.u64(mapped[0], shared.ptr_to([0]), T.uint32(lane))
        remote_ptr: T.let[
            T.Var(name="remote_read_ptr", ty=PointerType(PrimType("float32"), "shared"))
        ] = T.reinterpret(
            PointerType(PrimType("float32"), "shared"),
            mapped[0],
        )
        remote = T.decl_buffer((1,), "float32", scope="shared", data=remote_ptr)
        value = remote[0]
    output[cta, 0, lane] = T.cuda.warp_sum(value, width=2)
    output[cta, 1, lane] = T.cuda.warp_max(value, width=2)
    output[cta, 2, lane] = T.cuda.warp_min(value, width=2)


@T.prim_func
def remote_shared_write_ownership(output: T.Buffer((2,), "float32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "float32", scope="shared")
    if lane == 0:
        shared[0] = T.float32(-1)
    T.cuda.cluster_sync()
    if cta == 0:
        if lane < 2:
            mapped = T.alloc_local((1,), "uint64")
            T.ptx.mapa.u64(mapped[0], shared.ptr_to([0]), T.uint32(lane))
            remote_ptr: T.let[
                T.Var(name="remote_write_ptr", ty=PointerType(PrimType("float32"), "shared"))
            ] = T.reinterpret(
                PointerType(PrimType("float32"), "shared"),
                mapped[0],
            )
            remote = T.decl_buffer((1,), "float32", scope="shared", data=remote_ptr)
            remote[0] = T.cast(10 + lane, "float32")
    T.cuda.cluster_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        output[cta] = shared[0]


@T.prim_func
def bulk_shared_to_cluster_u64_addresses(output: T.Buffer((2, 4), "float32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane < 4):
        shared[lane] = T.cast(cta * 10 + lane + 1, "float32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        remote_destination = T.alloc_local((1,), "uint64")
        remote_barrier = T.alloc_local((1,), "uint64")
        T.ptx.mapa.u64(remote_destination[0], T.address_of(shared[0]), T.uint32(1))
        T.ptx.mapa.u64(remote_barrier[0], T.address_of(barriers[0]), T.uint32(1))
        remote_destination_ptr: T.let[
            T.Var(
                name="bulk_remote_destination_ptr",
                ty=PointerType(PrimType("float32"), "shared"),
            )
        ] = T.reinterpret(
            PointerType(PrimType("float32"), "shared"),
            remote_destination[0],
        )
        remote_barrier_ptr: T.let[
            T.Var(
                name="bulk_remote_barrier_ptr",
                ty=PointerType(PrimType("uint64"), "shared"),
            )
        ] = T.reinterpret(
            PointerType(PrimType("uint64"), "shared"),
            remote_barrier[0],
        )
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
            remote_barrier_ptr, T.uint32(16), pred=True
        )
        T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
            remote_destination_ptr,
            T.address_of(shared[0]),
            T.uint32(16),
            remote_barrier_ptr,
        )
    if (cta == 1) and (lane == 0):
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


@T.prim_func
def parallel_cluster_remote_shared_exchange(output: T.Buffer((2, 2, 2), "float32")):
    T.device_entry()
    cluster = T.cluster_id([2])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "float32", scope="shared")
    if lane == 0:
        shared[0] = T.cast(cluster * 100 + cta * 10 + 1, "float32")
    T.cuda.cluster_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        peer = T.meta_var(1 - cta)
        mapped = T.alloc_local((1,), "uint64")
        T.ptx.mapa.u64(mapped[0], shared.ptr_to([0]), T.uint32(peer))
        remote_ptr: T.let[
            T.Var(name="parallel_remote_read_ptr", ty=PointerType(PrimType("float32"), "shared"))
        ] = T.reinterpret(
            PointerType(PrimType("float32"), "shared"),
            mapped[0],
        )
        remote = T.decl_buffer((1,), "float32", scope="shared", data=remote_ptr)
        output[cluster, cta, 0] = shared[0]
        output[cluster, cta, 1] = remote[0]


@T.prim_func
def mapped_remote_mbarrier_pointer(output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        if cta == 0:
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        else:
            mapped = T.alloc_local((1,), "uint64")
            T.ptx.mapa.u64(mapped[0], barriers.ptr_to([0]), T.uint32(0))
            remote_ptr: T.let[
                T.Var(name="remote_barrier_ptr", ty=PointerType(PrimType("uint64"), "shared"))
            ] = T.reinterpret(
                PointerType(PrimType("uint64"), "shared"),
                mapped[0],
            )
            remote_barrier = T.decl_buffer((1,), "uint64", scope="shared", data=remote_ptr)
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(remote_barrier[0]))
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1


@T.prim_func
def pointer_derived_shared_raw_roundtrip(
    source: T.Buffer((32,), "uint32"),
    loaded: T.Buffer((32,), "uint32"),
    aliased: T.Buffer((32,), "uint32"),
    addresses: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((192,), "uint8", scope="shared")
    base_data: T.let[T.Var(name="raw_shared_base", ty=PointerType(PrimType("void"), "shared"))] = (
        T.reinterpret(PointerType(PrimType("void"), "shared"), shared.ptr_to([32]))
    )
    base = T.decl_buffer((36,), "uint32", data=base_data, scope="shared")
    stage_data: T.let[
        T.Var(name="raw_shared_stage", ty=PointerType(PrimType("uint32"), "shared"))
    ] = T.ptr_byte_offset(base.data, T.uint32(16), "uint32")
    stage = T.decl_buffer((32,), "uint32", data=stage_data, scope="shared")
    T.ptx.st.shared.u32(stage.ptr_to([lane]), source[lane])
    T.cuda.warp_sync()
    T.ptx.ld.shared.u32(loaded[lane], stage.ptr_to([lane]))
    aliased[lane] = base[4 + lane]
    addresses[lane] = T.cuda.cvta_generic_to_shared(stage.ptr_to([lane]))


@T.prim_func
def get_tmem_addr_lane_values(output: T.Buffer((64,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lane_u32 = T.cast(lane, "uint32")
    output[lane] = T.cuda.get_tmem_addr(T.uint32(0xFFF0FFF0), T.int32(32), lane_u32 * T.uint32(3))
    output[32 + lane] = T.cuda.get_tmem_addr(T.int32(0x00100010), T.int32(-32), T.int32(0) - lane)


@T.prim_func
def get_tmem_addr_unsigned_row_values(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lane_u32 = T.cast(lane, "uint32")
    output[lane] = T.cuda.get_tmem_addr(T.uint32(0xFFF00010), T.uint32(32) + lane_u32, T.uint32(0))


@T.prim_func
def raw_shared_v4_u32_roundtrip(
    source: T.Buffer((128,), "uint32"), output: T.Buffer((128,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    words_data: T.let[T.Var(name="raw_v4_words", ty=PointerType(PrimType("void"), "shared"))] = (
        T.reinterpret(PointerType(PrimType("void"), "shared"), shared.ptr_to([0]))
    )
    words = T.decl_buffer((128,), "uint32", data=words_data, scope="shared")
    base = lane * 4
    T.ptx.st.shared.v4.u32(
        shared.ptr_to([lane * 16]),
        source[base],
        source[base + 1],
        source[base + 2],
        source[base + 3],
    )
    T.cuda.warp_sync()
    for slot in T.unroll(4):
        output[base + slot] = words[base + slot]


@T.prim_func
def raw_shared_b128_roundtrip(
    source: T.Buffer((128,), "uint32"), output: T.Buffer((128,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    registers = T.alloc_local((4,), "uint32")
    shared = T.alloc_buffer((128,), "uint32", scope="shared")
    for index in T.unroll(4):
        registers[index] = source[lane * 4 + index]
    T.ptx.st.weak.shared__cta.b128(
        shared.ptr_to([lane * 4]),
        registers.view("uint128")[0],
    )
    T.cuda.cta_sync()
    for index in T.unroll(4):
        output[lane * 4 + index] = shared[lane * 4 + index]


@T.prim_func
def raw_shared_padding_load(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    sfq_stage = T.alloc_buffer((2,), "uint32", scope="shared")
    if lane == 0:
        sfq_stage[0] = T.uint32(17)
    T.cuda.warp_sync()
    index: T.let = T.if_then_else(lane == 0, 0, 1)
    T.ptx.ld.shared.u32(output[lane], sfq_stage.ptr_to([index]))


@T.prim_func
def raw_shared_uninitialized_load(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2,), "uint32", scope="shared")
    if lane == 0:
        shared[0] = T.uint32(17)
    T.cuda.warp_sync()
    index: T.let = T.if_then_else(lane == 0, 0, 1)
    T.ptx.ld.shared.u32(output[lane], shared.ptr_to([index]))


@T.prim_func
def raw_shared_byte_storage_u32_load(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")
    value = T.cast(lane + 1, "uint32") * T.uint32(0x01010101)
    for byte in T.unroll(4):
        shared[lane * 4 + byte] = T.cast(T.shift_right(value, T.uint32(byte * 8)), "uint8")
    T.cuda.warp_sync()
    T.ptx.ld.shared.u32(output[lane], shared.ptr_to([lane * 4]))


@T.prim_func
def raw_shared_u16_storage_u32_load(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint16", scope="shared")
    shared[lane * 2] = T.cast(lane, "uint16")
    shared[lane * 2 + 1] = T.cast(lane + 1, "uint16")
    T.cuda.warp_sync()
    T.ptx.ld.shared.u32(output[lane], shared.ptr_to([lane * 2]))


@T.prim_func
def raw_shared_byte_storage_u32_store(output: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "int8", scope="shared")
    value = T.cast(lane + 1, "uint32") * T.uint32(0x01020304)
    address_bits: T.uint64 = T.reinterpret("uint64", shared.ptr_to([0]))
    shifted = T.reinterpret("handle", address_bits + T.cast(lane * 4, "uint64"))
    T.ptx.st.shared.u32(shifted, value)
    T.cuda.warp_sync()
    for byte in T.unroll(4):
        output[lane * 4 + byte] = T.cast(shared[lane * 4 + byte], "uint8")


@T.prim_func
def raw_shared_u16_storage_u32_store(output: T.Buffer((2,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint16", scope="shared")
    if lane == 0:
        address_bits: T.uint64 = T.reinterpret("uint64", shared.ptr_to([0]))
        shifted = T.reinterpret("handle", address_bits + T.uint64(0))
        T.ptx.st.shared.u32(shifted, T.uint32(0x11223344))
        output[0] = shared[0]
        output[1] = shared[1]


@T.prim_func
def raw_ldmatrix_x4_b16_fragments(output: T.Buffer((128,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    values = T.alloc_local((4,), "uint32")
    for byte in T.unroll(16):
        shared[lane * 16 + byte] = T.cast(lane * 16 + byte, "uint8")
    T.cuda.warp_sync()
    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
        values[0],
        values[1],
        values[2],
        values[3],
        shared.ptr_to([lane * 16]),
    )
    for matrix in T.unroll(4):
        output[matrix * 32 + lane] = values[matrix]


@T.prim_func
def shared_virtual_backing_addresses(addresses: T.Buffer((2,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    first = T.alloc_buffer((20,), "uint8", scope="shared")
    second = T.alloc_buffer((32,), "uint8", scope="shared")
    if lane == 0:
        addresses[0] = T.cuda.cvta_generic_to_shared(first.ptr_to([4]))
        addresses[1] = T.cuda.cvta_generic_to_shared(second.ptr_to([8]))


@T.prim_func
def shared_virtual_swizzled_backing_alignment(addresses: T.Buffer((3,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    first = T.alloc_buffer((20,), "uint8", scope="shared")
    second = T.alloc_buffer(
        (8, 32), "float32", scope="shared", layout=_TEST_TMA_SWIZZLE_128B_LAYOUT, align=128
    )
    second_alias = T.decl_buffer((1,), "float32", data=second.data, elem_offset=3, scope="shared")
    if lane == 0:
        addresses[0] = T.cuda.cvta_generic_to_shared(first.ptr_to([4]))
        addresses[1] = T.cuda.cvta_generic_to_shared(second.ptr_to([0, 0]))
        addresses[2] = T.cuda.cvta_generic_to_shared(second_alias.ptr_to([0]))


@T.prim_func
def raw_global_memory_variants(
    source: T.Buffer((32,), "uint64"),
    values: T.Buffer((32,), "float32"),
    loaded: T.Buffer((128,), "uint64"),
    stored: T.Buffer((5, 32), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.ld.acquire.gpu.global_.u64(loaded[lane], source.ptr_to([lane]))
    T.ptx.ld.volatile.global_.u64(loaded[32 + lane], source.ptr_to([lane]))
    T.ptx.ld.relaxed.gpu.global_.u64(loaded[64 + lane], source.ptr_to([lane]))
    T.ptx.ld.mmio.relaxed.sys.global_.u64(loaded[96 + lane], source.ptr_to([lane]))
    T.ptx.st.global_.f32(
        stored.ptr_to([0, lane]),
        values[lane] + T.float32(1),
    )
    T.ptx.st.relaxed.gpu.global_.f32(
        stored.ptr_to([1, lane]),
        values[lane] + T.float32(2),
    )
    T.ptx.st.release.gpu.global_.f32(
        stored.ptr_to([2, lane]),
        values[lane] + T.float32(3),
    )
    T.ptx.st.volatile.global_.f32(
        stored.ptr_to([3, lane]),
        values[lane] + T.float32(4),
    )
    T.ptx.st.mmio.relaxed.sys.global_.f32(
        stored.ptr_to([4, lane]),
        values[lane] + T.float32(5),
    )


@T.prim_func
def raw_load_rejects_integer_address(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    forged: T.let[T.Var(name="forged_pointer", ty=PointerType(PrimType("uint32")))] = T.reinterpret(
        PointerType(PrimType("uint32")), T.uint64(64)
    )
    T.ptx.ld.global_.u32(output[lane], forged)


@T.prim_func
def direct_cuda_ldg(
    source_f32: T.Buffer((32,), "float32"),
    source_i32: T.Buffer((32,), "int32"),
    output_f32: T.Buffer((32,), "float32"),
    output_i32: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output_f32[lane] = T.cuda.ldg(source_f32.ptr_to([31 - lane]), "float32")
    output_i32[lane] = T.cuda.ldg(source_i32.ptr_to([31 - lane]), "int32")


@T.prim_func
def direct_tvm_access_ptr_shared(output: T.Buffer((128,), "uint32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    thread: T.let = warp * 32 + lane
    shared = T.alloc_buffer((128,), "uint32", scope="shared")
    shared[thread] = T.cast(thread * 3 + 1, "uint32")
    T.cuda.cta_sync()
    T.ptx.ld.shared.u32(output[thread], shared.access_ptr("r", offset=thread))


@T.prim_func
def warp_pure_calls(
    source: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "float32"),
    reduced: T.Buffer((32,), "uint32"),
    packed: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.let = source[lane]
    reverse: T.let = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), value, 31 - lane, 32, 32)
    magnitude = T.local_scalar("float32")
    reciprocal = T.local_scalar("float32")
    T.ptx.max.f32(magnitude, value, T.float32(0) - value)
    T.ptx.rcp.approx.ftz.f32(reciprocal, magnitude + T.float32(1))
    pair: T.let = T.cuda.make_float2(value, T.float32(2))
    multiplier: T.let = T.cuda.make_float2(T.float32(3), T.float32(4))
    product: T.let = T.cuda.fmul2_rn(pair, multiplier)
    output[lane] = reverse + reciprocal + T.cuda.float2_x(product)
    any_last: T.let = T.cuda.any_sync(T.uint32(0xFFFFFFFF), lane == 31)
    reduced[lane] = T.cuda.reduce_add_sync_u32(
        T.uint32(0xFFFFFFFF), T.cast(lane, "uint32")
    ) + T.cast(any_last, "uint32")
    packed[lane] = T.cuda.float22bfloat162_rn(value, reverse)


@T.prim_func
def native_varying_assert(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    with T.Assert(lane < 31, "lane must be below 31"):
        output[lane] = T.float32(1)


@T.prim_func
def dps_float_arithmetic(output: T.Buffer((32, 5), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    scalar = T.alloc_buffer((1,), "float32", scope="local")
    pair_fma = T.alloc_buffer((1,), "uint64", scope="local")
    pair_add = T.alloc_buffer((1,), "uint64", scope="local")
    x: T.let = T.cast(lane, "float32")
    lhs: T.let = T.cuda.make_float2(x, x + T.float32(1))
    rhs: T.let = T.cuda.make_float2(T.float32(2), T.float32(3))
    addend: T.let = T.cuda.make_float2(T.float32(4), T.float32(5))
    T.ptx.fma.rn.f32(scalar[0], x, T.float32(2), T.float32(1))
    T.ptx.fma.rn.f32x2(pair_fma[0], lhs, rhs, addend)
    T.ptx.add.rn.f32x2(pair_add[0], lhs, rhs)
    output[lane, 0] = scalar[0]
    output[lane, 1] = T.cuda.float2_x(pair_fma[0])
    output[lane, 2] = T.cuda.float2_y(pair_fma[0])
    output[lane, 3] = T.cuda.float2_x(pair_add[0])
    output[lane, 4] = T.cuda.float2_y(pair_add[0])
