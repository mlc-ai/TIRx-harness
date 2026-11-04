"""Small source-backed TIRx kernels for transpiler tests."""

from tvm_tirx.backend.cuda.layout import SwizzleMode, mma_shared_layout, tmem_datapath_layout
from tvm_tirx.ir import PtrType
from tvm_tirx.ir.type import PrimType
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import Axis, ComposeLayout, S, TileLayout

_TEST_SWIZZLE_LAYOUT = ComposeLayout(
    3,
    3,
    3,
    TileLayout(S[(2, 128, 2, 64) : (16384, 64, 8192, 1)]),
)

_TEST_PERMUTED_SHARED_LAYOUT = TileLayout(S[(4, 32) : (1, 4)])
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
_TEST_WARP_GEMM_D_FRAG = TileLayout(
    S[(2, 8, 4, 2) : (2, 4 @ Axis.get("laneid"), 1 @ Axis.get("laneid"), 1)]
)
_TEST_WARP_GEMM_A_FRAG_K8 = TileLayout(
    S[(2, 8, 4, 2) : (2, 4 @ Axis.get("laneid"), 1 @ Axis.get("laneid"), 1)]
)
_TEST_WARP_GEMM_B_FRAG_K8 = TileLayout(
    S[(4, 2, 8) : (1 @ Axis.get("laneid"), 1, 4 @ Axis.get("laneid"))]
)
_TEST_WARP_GEMM_A_FRAG = _TEST_WARP_GEMM_A_FRAG_K8.tile_to([16, 16], [16, 8])
_TEST_WARP_GEMM_B_FRAG = _TEST_WARP_GEMM_B_FRAG_K8.tile_to([16, 8], [8, 8])


@T.function
def no_op_kernel():
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=96))


@T.function
def warpgroup_scope_coordinates(output: T.Tensor((2, 4, 32, 3), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=256))
    warpgroup = T.cuda.warpgroup_id()
    warp_in_group = T.cuda.warp_in_warpgroup()
    lane = T.cuda.lane_id()
    thread_in_group = T.cuda.thread_in_warpgroup()
    output[warpgroup, warp_in_group, lane, 0] = warpgroup
    output[warpgroup, warp_in_group, lane, 1] = warp_in_group
    output[warpgroup, warp_in_group, lane, 2] = thread_in_group


@T.function
def extent_free_warp_id(output: T.Tensor((2, 32), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    launch_warp = T.cuda.warp_id()
    implicit_warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    output[launch_warp, lane] = implicit_warp


@T.function
def bound_dynamic_cta_extent(output: T.Tensor((2,), "int32")):
    cta_count: T.let = T.min(T.int32(4), T.int32(2))
    T.device_entry(launch=T.cuda.LaunchConfig(grid=cta_count, block=32))
    cta = T.cuda.block_idx("x")
    lane = T.cuda.lane_id()
    if lane == 0:
        output[cta] = cta + 1


@T.function
def elect_sync_integer_branch(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    if T.cuda.elect_sync():
        output[lane] = 1
    else:
        output[lane] = 2


@T.function
def fetch_register_coordinates(output: T.Tensor((2, 32, 2), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    cluster_rank = T.cuda.mov_sreg(32, "cluster_ctarank")
    fetched_lane = T.cuda.mov_sreg(32, "laneid")
    output[cta, lane, 0] = cluster_rank
    output[cta, lane, 1] = fetched_lane


@T.function
def unsupported_exp(source: T.Tensor((32,), "float32"), destination: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    destination[lane] = T.sin(source[lane])


@T.function
def lane_add(
    left: T.Tensor((100,), "float32"),
    right: T.Tensor((100,), "float32"),
    output: T.Tensor((100,), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    index = T.meta_var(warp * 32 + lane)
    if index < 100:
        output[index] = left[index] + right[index]


@T.function
def overlapping_alias_write(
    source: T.Tensor((32,), "float32"), destination: T.Tensor((32,), "float32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    destination[lane] = source[lane] + T.float32(1)


@T.function
def global_decl_buffer_alias_reinterpret(
    storage: T.Tensor((128,), "uint8"),
    source: T.Tensor((32,), "float32"),
    output: T.Tensor((32,), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    alias = T.decl_tensor(storage.data, ty_args=[T.Tensor((32,), "float32", scope="global")])
    alias[lane] = source[lane]
    output[lane] = alias[lane]


@T.function
def global_dynamic_subview_by_warp(
    source: T.Tensor((64,), "int32"), output: T.Tensor((4, 16), "int32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    warp_rows = source.view(1, 1, 4, 4, 4).sub[0, 0, :, warp, :]
    if lane < 16:
        output[warp, lane] = warp_rows[lane // 4, lane % 4]


@T.function
def matrix_add_2d(
    left: T.Tensor((3, 5), "float32"),
    right: T.Tensor((3, 5), "float32"),
    output: T.Tensor((3, 5), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    if lane < 15:
        row = T.meta_var(lane // 5)
        col = T.meta_var(lane % 5)
        output[row, col] = left[row, col] + right[row, col]


@T.function
def shared_alias_per_cta(output: T.Tensor((2, 32), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32))
    cta = T.cuda.block_idx("x")
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((64,), "float32")
    alias = T.decl_tensor(
        T.ptr_byte_offset(shared.data, 64, ty=T.Ptr("float32", "shared")),
        ty_args=[T.Tensor((32,), "float32", scope="shared")],
    )
    alias[lane] = T.cast(cta * 100 + lane, "float32")
    output[cta, lane] = shared[16 + lane]


@T.function
def shared_uninitialized_read(output: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((32,), "float32")
    output[lane] = shared[lane]


@T.function
def local_array_per_lane(output: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    local = T.alloc_local((2,), "float32")
    local[0] = T.cast(lane, "float32")
    local[1] = T.cast(lane + 1, "float32")
    output[lane] = local[0] + local[1]


@T.function
def tmem_d_alias_per_cta(output: T.Tensor((2, 128), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128))
    cta = T.cuda.block_idx("x")
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    row = T.meta_var(warp * 32 + lane)
    primary = T.tmem((128, 4), "uint32", allocated_addr=3, layout=tmem_datapath_layout("D", 128, 4))
    alias = T.tmem((128, 4), "uint32", allocated_addr=3, layout=tmem_datapath_layout("D", 128, 4))
    primary[row, 0] = T.cast(cta * 1000 + row, "uint32")
    output[cta, row] = alias[row, 0]


@T.function
def tmem_f_lane_mapping(output: T.Tensor((64,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    row = T.meta_var(warp * 32 + lane)
    f_view = T.tmem((64, 4), "uint32", allocated_addr=0, layout=tmem_datapath_layout("F", 64, 4))
    d_alias = T.tmem((128, 4), "uint32", allocated_addr=0, layout=tmem_datapath_layout("D", 128, 4))
    physical_lane = T.meta_var((row // 16) * 32 + row % 16)
    f_view[row, 1] = T.cast(2000 + row, "uint32")
    output[row] = d_alias[physical_lane, 1]


@T.function
def tmem_b_lane_column_mapping(output: T.Tensor((64, 2), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    row = T.meta_var(warp * 32 + lane)
    b_view = T.tmem((64, 8), "uint32", allocated_addr=2, layout=tmem_datapath_layout("B", 64, 8))
    d_alias = T.tmem((128, 4), "uint32", allocated_addr=2, layout=tmem_datapath_layout("D", 128, 4))
    b_view[row, 0] = T.cast(3000 + row, "uint32")
    b_view[row, 4] = T.cast(4000 + row, "uint32")
    output[row, 0] = d_alias[row, 0]
    output[row, 1] = d_alias[row + 64, 0]


@T.function
def tmem_packed_alias(output: T.Tensor((128,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    row = T.meta_var(warp * 32 + lane)
    packed_layout = TileLayout(S[(128, 2) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))])
    word_layout = TileLayout(S[(128, 2) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))])
    low = T.tmem((128, 2), "uint16", allocated_addr=5, layout=packed_layout)
    word = T.tmem((128, 2), "uint32", allocated_addr=5, layout=word_layout)
    low[row, 0] = T.uint16(0x1122)
    low[row, 1] = T.uint16(0x3344)
    output[row] = word[row, 0]


@T.function
def tmem_dynamic_allocated_addr(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.alloc_shared((1,), "uint32")
    if lane == 0:
        address[0] = T.uint32(6)
    T.cuda.warp_sync()
    tmem = T.tmem(
        (32, 2),
        "uint32",
        allocated_addr=address[0],
        layout=TileLayout(S[(32, 2) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))]),
    )
    tmem[lane, 1] = T.cast(5000 + lane, "uint32")
    output[lane] = tmem[lane, 1]


@T.function
def tmem_uninitialized_read(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    tmem = T.tmem(
        (32, 2),
        "uint32",
        allocated_addr=0,
        layout=TileLayout(S[(32, 2) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))]),
    )
    output[lane] = tmem[lane, 0]


@T.function
def permuted_global_layout_read(
    source: T.Tensor((128,), "uint32", layout=_TEST_PERMUTED_SHARED_LAYOUT),
    output: T.Tensor((128,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    for chunk in T.serial(4):
        index = T.meta_var(lane * 4 + chunk)
        output[index] = source[index]


@T.function
def tcgen_lifecycle_single_cta(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address = T.alloc_shared((1,), "uint32")
    T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 64)
    if lane == 0:
        output[0] = address[0]
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
    T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(T.uint32(0), 64)


@T.function
def tcgen_lifecycle_two_cta(output: T.Tensor((2,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    address = T.alloc_shared((1,), "uint32")
    T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(T.address_of(address[0]), 128)
    if lane == 0:
        output[cta] = address[0]
    T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
    T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(T.uint32(0), 128)


@T.function
def tcgen_commit_mbarrier(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    barrier = T.alloc_shared((1,), "uint64")
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


@T.function
def divergent_loop_control(output: T.Tensor((128,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    for step in T.serial(4):
        if lane == step:
            continue
        if lane < 4:
            if step == 2:
                break
        output[lane * 4 + step] = T.float32(1)


@T.function
def divergent_while_control(output: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    while lane < 16:
        output[lane] = T.float32(2)
        break


@T.function
def mbarrier_wait_after_full_warp_continue(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((1,), "uint64")
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


@T.function
def bar_sync_after_full_warp_continue(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    for _step in T.serial(1):
        if lane >= 0:
            continue
        T.ptx.bar.sync(T.uint32(lane), T.uint32(32))
        output[lane] = 1


@T.function
def cta_sync_after_full_warp_continue(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    for _step in T.serial(1):
        if lane >= 0:
            continue
        T.cuda.cta_sync()
        output[lane] = 1


@T.function
def dynamic_for_after_full_warp_continue(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    for _step in T.serial(1):
        if lane >= 0:
            continue
        for _inner in T.serial(0, lane):
            output[lane] = 1


@T.function
def mbarrier_phase_reuse(source: T.Tensor((4,), "float32"), output: T.Tensor((2,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((1,), "uint64")
    shared = T.alloc_shared((4,), "float32", align=128)
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
                    T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
                        shared.ptr_to([0]), source.ptr_to([0]), T.uint32(16), barriers.ptr_to([0])
                    )
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        else:
            if lane == 0:
                T.cuda.mbarrier_wait(T.address_of(barriers[0]), phase)
                output[phase] = phase + 1
        T.cuda.cta_sync()


@T.function
def bound_launch_topology():
    max_ctas: T.let = T.int32(4)
    should_cap: T.let = max_ctas > T.int32(3)
    cta_count: T.let = T.Select(should_cap, T.min(max_ctas - T.int32(1), T.int32(3)), T.int32(1))
    T.device_entry(launch=T.cuda.LaunchConfig(grid=cta_count, block=64))


@T.function
def scalar_expression_mix(output: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    wide_lane: T.let = T.cast(lane, "int64")
    shifted: T.let = wide_lane - T.int64(17)
    quotient: T.let = shifted // T.int64(5)
    remainder: T.let = shifted % T.int64(5)
    clamped: T.let = T.min(T.max(quotient, T.int64(-2)), T.int64(2))
    use_clamped: T.let = ((lane < 5) and not (lane == 2)) or lane >= 29
    selected: T.let = T.Select(use_clamped, clamped, remainder)
    output[lane] = T.cast(selected, "float32")


@T.function
def nested_mask_parent_scope(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
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


@T.function
def fixed_width_integer_expression_mix(output: T.Tensor((32, 7), "uint64")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
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


@T.function
def raw_scalar_call_mix(source: T.Tensor((32,), "float32"), output: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
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
    bitcast_identity: T.let = T.reinterpret(T.cuda.float_as_uint(selected), ty="float32")
    exponential = T.local_scalar("float32")
    T.ptx.ex2.approx.ftz.f32(exponential, T.log(bitcast_identity))
    math_value: T.let = exponential + T.rsqrt(bitcast_identity + T.float32(4))
    output[lane] = (
        math_value
        + T.cast(mixed + inverted, "float32") * T.float32(0.001)
        + T.cast(elected, "float32") * T.float32(0.01)
    )


@T.function
def guarded_if_then_else_load(
    source: T.Tensor((1,), "float32"), output: T.Tensor((32,), "float32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    output[lane] = T.if_then_else(lane == 0, source[lane], T.float32(7))


@T.function
def guarded_select_load(source: T.Tensor((2,), "uint64"), output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    output[lane] = T.Select(lane < 2, T.cast(source[lane], "uint32"), T.cast(100 + lane, "uint32"))


@T.function
def unsupported_bfloat_cast(source: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    _value: T.let = T.cast(source[lane], "bfloat16")


@T.function
def scalar_buffer_types(
    input_i32: T.Tensor((32,), "int32"),
    output_i32: T.Tensor((32,), "int32"),
    input_u64: T.Tensor((32,), "uint64"),
    output_u64: T.Tensor((32,), "uint64"),
    input_f16: T.Tensor((32,), "float16"),
    output_f16: T.Tensor((32,), "float16"),
    input_bf16: T.Tensor((32,), "bfloat16"),
    output_bf16: T.Tensor((32,), "bfloat16"),
    input_bool: T.Tensor((32,), "bool"),
    output_bool: T.Tensor((32,), "bool"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    output_i32[lane] = input_i32[lane] + T.int32(7)
    output_u64[lane] = input_u64[lane] + T.uint64(11)
    output_f16[lane] = T.cast(T.cast(input_f16[lane], "float32") + T.float32(0.5), "float16")
    output_bf16[lane] = T.cast(T.cast(input_bf16[lane], "float32") + T.float32(0.5), "bfloat16")
    output_bool[lane] = input_bool[lane]


@T.function
def float8_buffer_codecs(
    input_e4m3: T.Tensor((32,), "float8_e4m3fn"),
    input_e8m0: T.Tensor((32,), "float8_e8m0fnu"),
    output_e4m3: T.Tensor((32,), "float8_e4m3fn"),
    decoded: T.Tensor((32, 2), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    decoded[lane, 0] = input_e4m3[lane]
    decoded[lane, 1] = input_e8m0[lane]
    output_e4m3[lane] = input_e4m3[lane]


_DYNAMIC_ROWS = T.dynamic("rows", "int32")


@T.function
def dynamic_rows(
    input_buffer: T.Tensor((_DYNAMIC_ROWS, 32), "float32"),
    output_buffer: T.Tensor((_DYNAMIC_ROWS, 32), "float32"),
):
    rows = _DYNAMIC_ROWS
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    row: T.int32 = 0
    while row < rows:
        output_buffer[row, lane] = input_buffer[row, lane] + T.float32(2)
        row = row + 1


@T.function
def compose_swizzle_alias(output: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((2, 128, 128), "float16", layout=_TEST_SWIZZLE_LAYOUT)
    dense = T.decl_tensor(shared.data, ty_args=[T.Tensor((32768,), "float16", scope="shared")])
    shared[0, lane, lane] = T.cast(lane + 1, "float16")
    linear: T.let = lane * 8 + lane // 8
    physical: T.let = ((linear ^ ((linear & 56) >> 3)) << 3) + lane % 8
    output[lane] = T.cast(dense[physical], "float32")


@T.function
def physical_address_value():
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((32,), "uint64")
    T.evaluate(T.address_of(shared[lane]))


@T.function
def mbarrier_varying_uniform_phase(output: T.Tensor((2,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((1,), "uint64")
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


@T.function
def raw_tma_gather4_bar_address(input_map: T.Ptr(T.TensorMap), output: T.Tensor((4, 4), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((4, 4), "float32")
    barrier = T.alloc_shared((1,), "uint64")
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


@T.function
def raw_tma_reduce_add(source: T.Tensor((4,), "float32"), output_map: T.Ptr(T.TensorMap)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((4,), "float32")
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


@T.function
def raw_tma_roundtrip(input_map: T.Ptr(T.TensorMap), output_map: T.Ptr(T.TensorMap)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((3, 4), "float32")
    barriers = T.alloc_shared((1,), "uint64")
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


@T.function
def raw_tma_fp4_align8_roundtrip(input_map: T.Ptr(T.TensorMap), output_map: T.Ptr(T.TensorMap)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((2, 64), "uint8")
    barriers = T.alloc_shared((1,), "uint64")
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


@T.function
def raw_tma_dynamic_wait_group_loop(output: T.Tensor((2,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    if lane == 0:
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(1)
        output[0] = 10
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        output[1] = 11


@T.function
def raw_tma_rank3_roundtrip(input_map: T.Ptr(T.TensorMap), output_map: T.Ptr(T.TensorMap)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((2, 3, 4), "float32")
    barriers = T.alloc_shared((1,), "uint64")
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


@T.function
def raw_tma_split_swizzle_atoms(input_map: T.Ptr(T.TensorMap), output_map: T.Ptr(T.TensorMap)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((2, 2, 32), "float32")
    barriers = T.alloc_shared((1,), "uint64")
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


@T.function
def raw_tma_zero_fill(input_map: T.Ptr(T.TensorMap), output_map: T.Ptr(T.TensorMap)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((4, 4), "float32")
    barriers = T.alloc_shared((1,), "uint64")
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


@T.function
def raw_tma_swizzle_to_dense(input_map: T.Ptr(T.TensorMap), output_map: T.Ptr(T.TensorMap)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((8, 8), "float32")
    barriers = T.alloc_shared((1,), "uint64")
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

    @T.function
    def kernel(input_map: T.Ptr(T.TensorMap), output: T.Tensor((8, columns), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((8, columns), "float32", layout=layout)
        barriers = T.alloc_shared((1,), "uint64")
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
                    0,
                    T.address_of(barriers[0]),
                )
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 8 * swizzle_bytes)
            T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
            for row in T.serial(8):
                for col in T.serial(columns):
                    output[row, col] = shared[row, col]

    return kernel


@T.function
def raw_tma_transaction_mismatch(input_map: T.Ptr(T.TensorMap)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((3, 4), "float32")
    barriers = T.alloc_shared((1,), "uint64")
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


@T.function
def ordering_only_control_calls(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((1,), "uint64")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.ptx.griddepcontrol.wait()
    T.cuda.cta_sync()
    output[lane] = lane + 1


@T.function
def mbarrier_missing_arrivals():
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((1,), "uint64")
    if (warp == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 64)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if warp == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barriers[0]))
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)


@T.function
def mbarrier_remote_cta(output: T.Tensor((2,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((1,), "uint64")
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


@T.function
def scoped_syncs(output: T.Tensor((2, 4, 32), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    T.cuda.warp_sync()
    T.cuda.warpgroup_sync(7)
    T.ptx.bar.sync(T.uint32(6), T.uint32(128))
    T.cuda.cta_sync()
    T.cuda.cluster_sync()
    output[cta, warp, lane] = cta * 10000 + warp * 100 + lane


@T.function
def remote_shared_read_and_warp_reduce(output: T.Tensor((2, 3, 32), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((1,), "float32")
    value: T.float32 = T.float32(0)
    if lane == 0:
        shared[0] = T.cast(cta + 1, "float32")
    T.cuda.cluster_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane < 2:
        mapped = T.alloc_local((1,), "uint64")
        T.ptx.mapa.u64(mapped[0], shared.ptr_to([0]), T.uint32(lane))
        remote_ptr: T.let[
            T.Var(name="remote_read_ptr", ty=PtrType(PrimType("float32"), "shared"))
        ] = T.reinterpret(mapped[0], ty=PtrType(PrimType("float32"), "shared"))
        remote = T.decl_tensor(remote_ptr, ty_args=[T.Tensor((1,), "float32", scope="shared")])
        value = remote[0]
    output[cta, 0, lane] = T.cuda.warp_sum(value, width=2)
    output[cta, 1, lane] = T.cuda.warp_max(value, width=2)
    output[cta, 2, lane] = T.cuda.warp_min(value, width=2)


@T.function
def remote_shared_write_ownership(output: T.Tensor((2,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((1,), "float32")
    if lane == 0:
        shared[0] = T.float32(-1)
    T.cuda.cluster_sync()
    if cta == 0:
        if lane < 2:
            mapped = T.alloc_local((1,), "uint64")
            T.ptx.mapa.u64(mapped[0], shared.ptr_to([0]), T.uint32(lane))
            remote_ptr: T.let[
                T.Var(name="remote_write_ptr", ty=PtrType(PrimType("float32"), "shared"))
            ] = T.reinterpret(mapped[0], ty=PtrType(PrimType("float32"), "shared"))
            remote = T.decl_tensor(remote_ptr, ty_args=[T.Tensor((1,), "float32", scope="shared")])
            remote[0] = T.cast(10 + lane, "float32")
    T.cuda.cluster_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        output[cta] = shared[0]


@T.function
def bulk_shared_to_cluster_u64_addresses(output: T.Tensor((2, 4), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((4,), "float32")
    barriers = T.alloc_shared((1,), "uint64")
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
                ty=PtrType(PrimType("float32"), "shared"),
            )
        ] = T.reinterpret(remote_destination[0], ty=PtrType(PrimType("float32"), "shared"))
        remote_barrier_ptr: T.let[
            T.Var(
                name="bulk_remote_barrier_ptr",
                ty=PtrType(PrimType("uint64"), "shared"),
            )
        ] = T.reinterpret(remote_barrier[0], ty=PtrType(PrimType("uint64"), "shared"))
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


@T.function
def parallel_cluster_remote_shared_exchange(output: T.Tensor((2, 2, 2), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=4, block=32, cluster=2))
    cluster = T.cuda.cluster_id("x")
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((1,), "float32")
    if lane == 0:
        shared[0] = T.cast(cluster * 100 + cta * 10 + 1, "float32")
    T.cuda.cluster_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        peer = T.meta_var(1 - cta)
        mapped = T.alloc_local((1,), "uint64")
        T.ptx.mapa.u64(mapped[0], shared.ptr_to([0]), T.uint32(peer))
        remote_ptr: T.let[
            T.Var(name="parallel_remote_read_ptr", ty=PtrType(PrimType("float32"), "shared"))
        ] = T.reinterpret(mapped[0], ty=PtrType(PrimType("float32"), "shared"))
        remote = T.decl_tensor(remote_ptr, ty_args=[T.Tensor((1,), "float32", scope="shared")])
        output[cluster, cta, 0] = shared[0]
        output[cluster, cta, 1] = remote[0]


@T.function
def mapped_remote_mbarrier_pointer(output: T.Tensor((2,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    barriers = T.alloc_shared((1,), "uint64")
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
                T.Var(name="remote_barrier_ptr", ty=PtrType(PrimType("uint64"), "shared"))
            ] = T.reinterpret(mapped[0], ty=PtrType(PrimType("uint64"), "shared"))
            remote_barrier = T.decl_tensor(
                remote_ptr, ty_args=[T.Tensor((1,), "uint64", scope="shared")]
            )
            T.ptx.mbarrier.arrive.shared.b64(T.address_of(remote_barrier[0]))
    T.cuda.cluster_sync()
    if lane == 0:
        output[cta] = 1


@T.function
def pointer_derived_shared_raw_roundtrip(
    source: T.Tensor((32,), "uint32"),
    loaded: T.Tensor((32,), "uint32"),
    aliased: T.Tensor((32,), "uint32"),
    addresses: T.Tensor((32,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((192,), "uint8")
    base_data: T.let[T.Var(name="raw_shared_base", ty=PtrType(PrimType("void"), "shared"))] = (
        T.reinterpret(shared.ptr_to([32]), ty=PtrType(PrimType("void"), "shared"))
    )
    base = T.decl_tensor(base_data, ty_args=[T.Tensor((36,), "uint32", scope="shared")])
    stage_data: T.let[T.Var(name="raw_shared_stage", ty=PtrType(PrimType("uint32"), "shared"))] = (
        T.ptr_byte_offset(base.data, T.uint32(16), ty=T.Ptr("uint32", "shared"))
    )
    stage = T.decl_tensor(stage_data, ty_args=[T.Tensor((32,), "uint32", scope="shared")])
    T.ptx.st.shared.u32(stage.ptr_to([lane]), source[lane])
    T.cuda.warp_sync()
    T.ptx.ld.shared.u32(loaded[lane], stage.ptr_to([lane]))
    aliased[lane] = base[4 + lane]
    addresses[lane] = T.cuda.cvta_generic_to_shared(stage.ptr_to([lane]))


@T.function
def get_tmem_addr_lane_values(output: T.Tensor((64,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    lane_u32 = T.cast(lane, "uint32")
    output[lane] = T.cuda.get_tmem_addr(T.uint32(0xFFF0FFF0), T.int32(32), lane_u32 * T.uint32(3))
    output[32 + lane] = T.cuda.get_tmem_addr(T.int32(0x00100010), T.int32(-32), T.int32(0) - lane)


@T.function
def get_tmem_addr_unsigned_row_values(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    lane_u32 = T.cast(lane, "uint32")
    output[lane] = T.cuda.get_tmem_addr(T.uint32(0xFFF00010), T.uint32(32) + lane_u32, T.uint32(0))


@T.function
def raw_shared_v4_u32_roundtrip(
    source: T.Tensor((128,), "uint32"), output: T.Tensor((128,), "uint32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((512,), "uint8")
    words_data: T.let[T.Var(name="raw_v4_words", ty=PtrType(PrimType("void"), "shared"))] = (
        T.reinterpret(shared.ptr_to([0]), ty=PtrType(PrimType("void"), "shared"))
    )
    words = T.decl_tensor(words_data, ty_args=[T.Tensor((128,), "uint32", scope="shared")])
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


@T.function
def raw_shared_b128_roundtrip(
    source: T.Tensor((128,), "uint32"), output: T.Tensor((128,), "uint32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    registers = T.alloc_local((4,), "uint32")
    shared = T.alloc_shared((128,), "uint32")
    for index in T.unroll(4):
        registers[index] = source[lane * 4 + index]
    T.ptx.st.weak.shared__cta.b128(
        shared.ptr_to([lane * 4]),
        registers.view("uint128")[0],
    )
    T.cuda.cta_sync()
    for index in T.unroll(4):
        output[lane * 4 + index] = shared[lane * 4 + index]


@T.function
def raw_shared_padding_load(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    sfq_stage = T.alloc_shared((2,), "uint32")
    if lane == 0:
        sfq_stage[0] = T.uint32(17)
    T.cuda.warp_sync()
    index: T.let = T.if_then_else(lane == 0, 0, 1)
    T.ptx.ld.shared.u32(output[lane], sfq_stage.ptr_to([index]))


@T.function
def raw_shared_uninitialized_load(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((2,), "uint32")
    if lane == 0:
        shared[0] = T.uint32(17)
    T.cuda.warp_sync()
    index: T.let = T.if_then_else(lane == 0, 0, 1)
    T.ptx.ld.shared.u32(output[lane], shared.ptr_to([index]))


@T.function
def raw_shared_byte_storage_u32_load(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "uint8")
    value = T.cast(lane + 1, "uint32") * T.uint32(0x01010101)
    for byte in T.unroll(4):
        shared[lane * 4 + byte] = T.cast(T.shift_right(value, T.uint32(byte * 8)), "uint8")
    T.cuda.warp_sync()
    T.ptx.ld.shared.u32(output[lane], shared.ptr_to([lane * 4]))


@T.function
def raw_shared_u16_storage_u32_load(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((64,), "uint16")
    shared[lane * 2] = T.cast(lane, "uint16")
    shared[lane * 2 + 1] = T.cast(lane + 1, "uint16")
    T.cuda.warp_sync()
    T.ptx.ld.shared.u32(output[lane], shared.ptr_to([lane * 2]))


@T.function
def raw_shared_byte_storage_u32_store(output: T.Tensor((128,), "uint8")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((128,), "int8")
    value = T.cast(lane + 1, "uint32") * T.uint32(0x01020304)
    address_bits: T.uint64 = T.reinterpret(shared.ptr_to([0]), ty="uint64")
    shifted = T.reinterpret(address_bits + T.cast(lane * 4, "uint64"), ty="handle")
    T.ptx.st.shared.u32(shifted, value)
    T.cuda.warp_sync()
    for byte in T.unroll(4):
        output[lane * 4 + byte] = T.cast(shared[lane * 4 + byte], "uint8")


@T.function
def raw_shared_u16_storage_u32_store(output: T.Tensor((2,), "uint16")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((64,), "uint16")
    if lane == 0:
        address_bits: T.uint64 = T.reinterpret(shared.ptr_to([0]), ty="uint64")
        shifted = T.reinterpret(address_bits + T.uint64(0), ty="handle")
        T.ptx.st.shared.u32(shifted, T.uint32(0x11223344))
        output[0] = shared[0]
        output[1] = shared[1]


@T.function
def raw_ldmatrix_x4_b16_fragments(output: T.Tensor((128,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((512,), "uint8")
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


@T.function
def shared_virtual_backing_addresses(addresses: T.Tensor((2,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    first = T.alloc_shared((20,), "uint8")
    second = T.alloc_shared((32,), "uint8")
    if lane == 0:
        addresses[0] = T.cuda.cvta_generic_to_shared(first.ptr_to([4]))
        addresses[1] = T.cuda.cvta_generic_to_shared(second.ptr_to([8]))


@T.function
def shared_virtual_swizzled_backing_alignment(addresses: T.Tensor((3,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    first = T.alloc_shared((20,), "uint8")
    second = T.alloc_shared((8, 32), "float32", layout=_TEST_TMA_SWIZZLE_128B_LAYOUT, align=128)
    second_alias = T.decl_tensor(
        T.ptr_byte_offset(second.data, 12, ty=T.Ptr("float32", "shared")),
        ty_args=[T.Tensor((1,), "float32", scope="shared")],
    )
    if lane == 0:
        addresses[0] = T.cuda.cvta_generic_to_shared(first.ptr_to([4]))
        addresses[1] = T.cuda.cvta_generic_to_shared(second.ptr_to([0, 0]))
        addresses[2] = T.cuda.cvta_generic_to_shared(second_alias.ptr_to([0]))


@T.function
def raw_global_memory_variants(
    source: T.Tensor((32,), "uint64"),
    values: T.Tensor((32,), "float32"),
    loaded: T.Tensor((128,), "uint64"),
    stored: T.Tensor((5, 32), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
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


@T.function
def raw_load_rejects_integer_address(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    forged: T.let[T.Var(name="forged_pointer", ty=PtrType(PrimType("uint32")))] = T.reinterpret(
        T.uint64(64), ty=PtrType(PrimType("uint32"))
    )
    T.ptx.ld.global_.u32(output[lane], forged)


@T.function
def warp_pure_calls(
    source: T.Tensor((32,), "float32"),
    output: T.Tensor((32,), "float32"),
    reduced: T.Tensor((32,), "uint32"),
    packed: T.Tensor((32,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    value: T.let = source[lane]
    reverse: T.let = T.gpu_warp_shuffle(T.uint32(0xFFFFFFFF), value, 31 - lane, 32, 32)
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


@T.function
def native_varying_assert(output: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    T.assert_(lane < 31, "lane must be below 31")
    output[lane] = T.float32(1)


@T.function
def dps_float_arithmetic(output: T.Tensor((32, 5), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    scalar = T.alloc_local((1,), "float32")
    pair_fma = T.alloc_local((1,), "uint64")
    pair_add = T.alloc_local((1,), "uint64")
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
