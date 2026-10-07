from __future__ import annotations

import math

import numpy as np
import pytest

from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.errors import NumSimExecutionError
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T


@T.prim_func
def scalar_warp_intrinsics(
    output_u32: T.Buffer((32, 7), "uint32"),
    output_f32: T.Buffer((32, 4), "float32"),
    output_u64: T.Buffer((32,), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    full: T.let = T.uint32(0xFFFFFFFF)
    value: T.let = T.cast(lane + 1, "uint32")
    output_u32[lane, 0] = T.cuda.__shfl_up_sync(full, value, 1, 32)
    output_u32[lane, 1] = T.cuda.__shfl_down_sync(full, value, 1, 32)
    output_u32[lane, 2] = T.cuda.__shfl_xor_sync(full, value, 1, 32)
    output_u32[lane, 3] = T.cuda.__activemask()
    packed_float2: T.let = T.cuda.make_float2(T.cast(lane, "float32"), T.cast(lane + 1, "float32"))
    output_u32[lane, 4] = T.cuda.float22bfloat162_rn_from_float2(packed_float2)
    output_u32[lane, 5] = T.call_intrin(
        "uint32", "tirx.cuda.sm100_2sm_leader_smem_addr", T.uint64(0xFFFFFFFFF)
    )
    output_u32[lane, 6] = T.cuda.smem_addr_from_uint64(T.uint64(0x123456789))
    output_f32[lane, 0] = T.cuda.half2float(T.cast(T.cast(lane, "float32") + 0.25, "float16"))
    output_f32[lane, 1] = T.cuda.bfloat162float(T.cast(T.cast(lane, "float32") + 0.5, "bfloat16"))
    maximum = T.local_scalar("float32")
    minimum = T.local_scalar("float32")
    T.ptx.max.f32(maximum, T.cast(lane, "float32"), T.float32(17), T.float32(-3))
    T.ptx.min.f32(minimum, T.cast(lane, "float32"), T.float32(17), T.float32(-3))
    output_f32[lane, 2] = maximum
    output_f32[lane, 3] = minimum
    output_u64[lane] = T.cuda.clock64()


@T.prim_func
def ptx_three_source_nan_minmax(
    lhs: T.Buffer((32,), "float32"),
    middle: T.Buffer((32,), "float32"),
    rhs: T.Buffer((32,), "float32"),
    maximum: T.Buffer((32,), "float32"),
    minimum: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.max.NaN.f32(maximum[lane], lhs[lane], middle[lane], rhs[lane])
    T.ptx.min.NaN.f32(minimum[lane], lhs[lane], middle[lane], rhs[lane])


@T.prim_func
def timer_finalize_is_numerical_noop(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    profiler_buffer = T.alloc_buffer((4,), "uint64", scope="local")
    profiler_tag = T.alloc_buffer((1,), "uint64", scope="local")
    profiler_write_offset = T.alloc_buffer((1,), "int32", scope="local")
    T.cuda.timer_finalize(
        profiler_buffer.data,
        profiler_tag.data,
        profiler_write_offset.data,
        1,
        lane == 0,
    )
    output[lane] = T.cast(lane + 7, "uint32")


@T.prim_func
def current_scalar_device_intrinsics(
    output_f32: T.Buffer((3, 32), "float32"),
    output_u32: T.Buffer((3, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    T.evaluate(T.cuda.iket.mark("numsim"))
    sentinel: T.let = T.cuda.iket.sentinel_token("sentinel")
    T.evaluate(T.cuda.iket.range_end(sentinel))
    token: T.let = T.cuda.iket.range_start("range")
    T.evaluate(T.cuda.iket.range_push("stack"))
    T.evaluate(T.cuda.iket.range_pop())
    T.evaluate(T.cuda.iket.range_end(token))
    official: T.let = T.call_intrin("uint32", "tirx.cuda.iket_official_event", T.int32(7), "numsim")

    output_f32[0, lane] = T.cuda.fdividef(T.cast(lane + 1, "float32"), T.float32(2))
    T.ptx.cvt.rn.f32.s32(output_f32[1, lane], lane + 1)
    output_f32[2, lane] = T.log2(T.cast(lane + 1, "float32"))
    output_u32[0, lane] = sentinel
    output_u32[1, lane] = token
    output_u32[2, lane] = official


@T.prim_func
def packed_f16x2_conversion(
    high: T.Buffer((32,), "float32"),
    low: T.Buffer((32,), "float32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.cvt.rn.f16x2.f32(output[lane], high[lane], low[lane])


@T.prim_func
def shuffle_xor_crosses_to_earlier_width_group(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), T.cast(lane, "uint32"), 16, 16)


@T.prim_func
def cuda_packed_nan_semantics(
    output_u64: T.Buffer((32, 2), "uint64"), output_u32: T.Buffer((32, 4), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    lhs2: T.let = T.uint64(0x7FE12345FFC54321)
    rhs2: T.let = T.uint64(0x7FD2468A7FA13579)
    output_u64[lane, 0] = T.cuda.fadd2_rn(lhs2, rhs2)
    output_u64[lane, 1] = T.cuda.fmul2_rn(lhs2, rhs2)
    lhs_bf16: T.let = T.uint32(0xFFC27FC1)
    rhs_bf16: T.let = T.uint32(0x7FC47FE3)
    output_u32[lane, 0] = T.cuda.hmin2(lhs_bf16, rhs_bf16)
    output_u32[lane, 1] = T.cuda.hmax2(lhs_bf16, rhs_bf16)
    nan_a: T.let = T.cuda.uint_as_float(T.uint32(0x7FE12345))
    nan_b: T.let = T.cuda.uint_as_float(T.uint32(0xFFC54321))
    output_u32[lane, 2] = T.cuda.float22bfloat162_rn(nan_a, nan_b)
    output_u32[lane, 3] = T.cuda.float22bfloat162_rn_from_float2(T.cuda.make_float2(nan_a, nan_b))


@T.prim_func
def cta_votes_and_grid_sync(
    flags: T.Buffer((2,), "int32"),
    output_and: T.Buffer((128,), "int64"),
    output_or: T.Buffer((128,), "int64"),
    output_grid: T.Buffer((4,), "int32"),
):
    T.device_entry()
    cluster = T.cluster_id([2])
    _cta = T.cta_id_in_cluster([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    index: T.let = cluster * 64 + warp * 32 + lane
    output_and[index] = T.cuda.syncthreads_and(
        T.cast(T.bitwise_or(cluster == 0, T.bitwise_or(warp != 1, lane != 31)), "int32")
    )
    output_or[index] = T.cuda.syncthreads_or(
        T.cast(T.bitwise_and(cluster == 0, T.bitwise_and(warp == 1, lane == 5)), "int32")
    )
    T.cuda.thread_fence()
    T.ptx.fence.acq_rel.gpu()
    T.ptx.griddepcontrol.launch_dependents()
    T.cuda.nano_sleep(T.uint64(1))
    T.cuda.printf("cluster=%d", cluster)
    if warp == 0 and lane == 0:
        flags[cluster] = cluster + 1
    T.cuda.grid_sync()
    if lane == 0:
        output_grid[cluster * 2 + warp] = flags[0] + flags[1]


@T.prim_func
def floating_predicate_collectives(
    predicates: T.Buffer((2, 64), "float32"),
    ballots: T.Buffer((2, 64), "uint32"),
    any_results: T.Buffer((2, 64), "int32"),
    and_results: T.Buffer((2, 64), "int64"),
    or_results: T.Buffer((2, 64), "int64"),
):
    T.device_entry()
    cluster = T.cluster_id([2])
    _cta = T.cta_id_in_cluster([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    index: T.let = warp * 32 + lane
    predicate: T.let = predicates[cluster, index]
    full: T.let = T.uint32(0xFFFFFFFF)
    ballots[cluster, index] = T.cuda.ballot_sync(full, predicate)
    any_results[cluster, index] = T.cuda.any_sync(full, predicate)
    and_results[cluster, index] = T.cuda.syncthreads_and(predicate)
    or_results[cluster, index] = T.cuda.syncthreads_or(predicate)


@T.prim_func
def mbarrier_nonblocking_queries(output: T.Buffer((32, 4), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    T.ptx.mbarrier.test_wait.parity.shared.b64(
        output[lane, 0], T.address_of(barrier[0]), T.uint32(1)
    )
    T.ptx.mbarrier.try_wait.parity.shared.b64(
        output[lane, 1], T.address_of(barrier[0]), T.uint32(0), T.uint32(1)
    )
    T.ptx.mbarrier.try_wait.parity.shared.b64(
        output[lane, 2], T.address_of(barrier[0]), T.uint32(0)
    )
    if lane == 0:
        T.ptx.mbarrier.arrive.shared.b64(T.address_of(barrier[0]))
    T.ptx.mbarrier.test_wait.parity.shared.b64(
        output[lane, 3], T.address_of(barrier[0]), T.uint32(0)
    )


@T.prim_func
def mbarrier_state_token_query(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    token = T.alloc_buffer((1,), "uint64", scope="local")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), T.uint32(1))
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        barrier_address: T.let = T.cuda.cvta_generic_to_shared(T.address_of(barrier[0]))
        T.ptx.mbarrier.arrive.shared__cta.b64(token[0], barrier_address, T.uint32(1))
        T.ptx.mbarrier.try_wait.shared__cta.b64(output[lane], barrier_address, token[0])


@T.prim_func
def mbarrier_stale_state_token(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    tokens = T.alloc_buffer((3,), "uint64", scope="local")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), T.uint32(1))
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        barrier_address: T.let = T.cuda.cvta_generic_to_shared(T.address_of(barrier[0]))
        for generation in T.serial(3):
            T.ptx.mbarrier.arrive.shared__cta.b64(tokens[generation], barrier_address, T.uint32(1))
            T.ptx.mbarrier.try_wait.shared__cta.b64(
                output[generation], barrier_address, tokens[generation]
            )
        T.ptx.mbarrier.try_wait.shared__cta.b64(output[3], barrier_address, tokens[0])


@T.prim_func
def pointer_conversions_and_descriptor(
    source: T.Buffer((32, 8), "float32"),
    half: T.Buffer((32, 8), "float16"),
    roundtrip: T.Buffer((32, 8), "float32"),
    descriptor: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.cuda.float8tohalf8(T.address_of(source[lane, 0]), T.address_of(half[lane, 0]))
    T.cuda.half8tofloat8(T.address_of(half[lane, 0]), T.address_of(roundtrip[lane, 0]))
    T.cuda.float22half2(T.address_of(half[lane, 0]), T.address_of(source[lane, 0]))
    T.cuda.runtime_instr_desc(T.address_of(descriptor[lane]), lane % 4)


@T.prim_func
def ptx_dps_arithmetic(
    output_f32: T.Buffer((32,), "float32"),
    output_f32x2: T.Buffer((32,), "uint64"),
    output_f64: T.Buffer((32, 4), "float64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value_f32: T.let = T.cast(lane, "float32")
    T.ptx.sub.rn.f32(output_f32[lane], value_f32, T.float32(3))
    packed_lhs: T.let = T.cuda.make_float2(value_f32, value_f32 + T.float32(10))
    packed_rhs: T.let = T.cuda.make_float2(T.float32(1), T.float32(2))
    T.ptx.sub.rn.f32x2(output_f32x2[lane], packed_lhs, packed_rhs)
    value_f64: T.let = T.cast(lane, "float64")
    T.ptx.add.rn.f64(output_f64[lane, 0], value_f64, T.float64(2))
    T.ptx.sub.rn.f64(output_f64[lane, 1], value_f64, T.float64(2))
    T.ptx.mul.rn.f64(output_f64[lane, 2], value_f64, T.float64(2))
    T.ptx.fma.rn.f64(output_f64[lane, 3], value_f64, T.float64(2), T.float64(1))


@T.prim_func
def ptx_dps_modifier_forms(
    output_f32: T.Buffer((32, 4), "float32"), output_f32x2: T.Buffer((32, 4), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tiny: T.let = T.cuda.uint_as_float(T.uint32(1))
    increment: T.let = T.float32(2**-25)
    nan: T.let = T.cuda.uint_as_float(T.uint32(0x7FC00001))
    T.ptx.add.rp.f32(output_f32[lane, 0], T.float32(1), increment)
    T.ptx.sub.rm.f32(output_f32[lane, 1], T.float32(-1), increment)
    T.ptx.mul.rz.ftz.f32(output_f32[lane, 2], tiny, T.float32(1))
    T.ptx.fma.rn.sat.f32(output_f32[lane, 3], nan, T.float32(1), T.float32(0))

    packed_lhs: T.let = T.cuda.make_float2(T.float32(1), tiny)
    packed_rhs: T.let = T.cuda.make_float2(increment, T.float32(1))
    packed_addend: T.let = T.cuda.make_float2(tiny, tiny)
    T.ptx.add.rp.ftz.f32x2(output_f32x2[lane, 0], packed_lhs, packed_rhs)
    T.ptx.sub.rm.f32x2(output_f32x2[lane, 1], packed_lhs, packed_rhs)
    T.ptx.mul.rz.ftz.f32x2(
        output_f32x2[lane, 2],
        packed_lhs,
        T.cuda.make_float2(tiny, T.float32(1)),
    )
    T.ptx.fma.rp.f32x2(
        output_f32x2[lane, 3],
        packed_lhs,
        T.cuda.make_float2(T.float32(1), T.float32(1)),
        packed_addend,
    )


@T.prim_func
def canonical_mapa(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    shared[lane] = lane + 37
    T.cuda.cta_sync()
    mapped = T.local_scalar("uint64")
    T.ptx.mapa.shared__cluster.u64(mapped, T.address_of(shared[0]), T.uint32(0))
    mapped_ptr: T.let[
        T.Var(name="canonical_mapped_ptr", ty=PointerType(PrimType("int32"), "shared"))
    ] = T.reinterpret(PointerType(PrimType("int32"), "shared"), mapped)
    mapped_buffer = T.decl_buffer((1,), "int32", scope="shared", data=mapped_ptr)
    output[lane] = mapped_buffer[0]


@T.prim_func
def explicit_mapa_forms(output: T.Buffer((32,), "int32"), addresses: T.Buffer((32,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    shared[lane] = lane + 37
    T.cuda.cta_sync()
    generic_u64 = T.local_scalar("uint64")
    cluster_u32 = T.local_scalar("uint32")
    cluster_u64 = T.local_scalar("uint64")
    shared_address: T.let = T.cuda.cvta_generic_to_shared(T.address_of(shared[1]))
    T.ptx.mapa.u64(generic_u64, T.address_of(shared[1]), T.uint32(0))
    T.ptx.mapa.shared__cluster.u32(cluster_u32, shared_address, T.uint32(0))
    T.ptx.mapa.shared__cluster.u64(cluster_u64, T.address_of(shared[1]), T.uint32(0))
    mapped_ptr: T.let[
        T.Var(name="explicit_mapa_shared_cluster_u64", ty=PointerType(PrimType("int32"), "shared"))
    ] = T.reinterpret(
        PointerType(PrimType("int32"), "shared"),
        cluster_u64,
    )
    mapped = T.decl_buffer((1,), "int32", scope="shared", data=mapped_ptr)
    output[lane] = mapped[0]
    addresses[lane] = T.cuda.cvta_generic_to_shared(T.address_of(shared[1]))


@T.prim_func
def explicit_cvta_shared_cluster_u64(addresses: T.Buffer((32,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    T.ptx.cvta.to.shared__cluster.u64(addresses[lane], T.address_of(shared[lane]))


@T.prim_func
def mapa_u32_reads_the_peer_cta(output: T.Buffer((2, 32), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    shared[lane] = cta * 100 + lane
    T.cuda.cluster_sync()
    remote_address = T.local_scalar("uint32")
    remote_value = T.local_scalar("int32")
    T.ptx.mapa.shared__cluster.u32(
        remote_address,
        T.cuda.cvta_generic_to_shared(shared.ptr_to([lane])),
        T.uint32(1 - cta),
    )
    T.ptx.ld.shared__cluster.s32(remote_value, remote_address)
    output[cta, lane] = remote_value


@T.prim_func
def mapa_exposes_integer_address_bits(
    addresses32: T.Buffer((2, 32, 3), "uint32"),
    addresses64: T.Buffer((2, 32, 3), "uint64"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint8", scope="shared")
    local32 = T.alloc_local((1,), "uint32")
    mapped32 = T.alloc_local((1,), "uint32")
    remapped32 = T.alloc_local((1,), "uint32")
    converted64 = T.alloc_local((1,), "uint64")
    mapped_shared64 = T.alloc_local((1,), "uint64")
    mapped_generic64 = T.alloc_local((1,), "uint64")
    peer = T.cast(1 - cta, "uint32")
    local32[0] = T.cuda.cvta_generic_to_shared(shared.ptr_to([lane]))
    T.ptx.mapa.shared__cluster.u32(mapped32[0], local32[0], peer)
    T.ptx.mapa.shared__cluster.u32(remapped32[0], mapped32[0], T.cast(cta, "uint32"))
    T.ptx.cvta.to.shared__cluster.u64(converted64[0], shared.ptr_to([lane]))
    T.ptx.mapa.shared__cluster.u64(
        mapped_shared64[0], T.cast(local32[0], "uint64"), peer
    )
    T.ptx.mapa.u64(mapped_generic64[0], T.address_of(shared[lane]), peer)
    addresses32[cta, lane, 0] = local32[0]
    addresses32[cta, lane, 1] = mapped32[0]
    addresses32[cta, lane, 2] = remapped32[0]
    addresses64[cta, lane, 0] = converted64[0]
    addresses64[cta, lane, 1] = mapped_shared64[0]
    addresses64[cta, lane, 2] = mapped_generic64[0]


@T.prim_func
def mapped_addresses_use_ordinary_integer_dataflow(output: T.Buffer((2, 32), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    address_slots = T.alloc_buffer((32,), "uint32", scope="shared")
    mapped = T.alloc_local((1,), "uint32")
    loaded = T.alloc_local((1,), "uint32")
    value = T.alloc_local((1,), "int32")
    shared[lane] = cta * 100 + lane
    T.cuda.cluster_sync()
    T.ptx.mapa.shared__cluster.u32(
        mapped[0],
        T.cuda.cvta_generic_to_shared(shared.ptr_to([lane])),
        T.cast(lane % 2, "uint32"),
    )
    address_slots[31 - lane] = mapped[0]
    T.cuda.warp_sync()
    loaded[0] = address_slots[31 - lane]
    selected: T.uint32 = T.if_then_else(lane % 2 == 0, mapped[0], loaded[0])
    shuffled: T.uint32 = T.cuda.__shfl_xor_sync(
        T.uint32(0xFFFFFFFF), selected, 1, 32
    )
    T.ptx.ld.shared__cluster.s32(value[0], shuffled)
    output[cta, lane] = value[0]


@T.prim_func
def fetch_logical_and_representative_registers(
    output32: T.Buffer((3, 32, 13), "int32"), output64: T.Buffer((3, 32, 3), "int64")
):
    T.device_entry()
    cluster = T.cluster_id([3])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output32[cluster, lane, 0] = T.cuda.mov_sreg(32, "clusterid.x")
    output32[cluster, lane, 1] = T.cuda.mov_sreg(32, "clusterid.y")
    output32[cluster, lane, 2] = T.cuda.mov_sreg(32, "clusterid.z")
    output32[cluster, lane, 3] = T.cuda.mov_sreg(32, "nclusterid.x")
    output32[cluster, lane, 4] = T.cuda.mov_sreg(32, "nclusterid.y")
    output32[cluster, lane, 5] = T.cuda.mov_sreg(32, "cluster_ctarank")
    output32[cluster, lane, 6] = T.cuda.mov_sreg(32, "cluster_nctarank")
    output32[cluster, lane, 7] = T.cuda.mov_sreg(32, "laneid")
    output32[cluster, lane, 8] = T.cuda.mov_sreg(32, "lanemask_eq")
    output32[cluster, lane, 9] = T.cuda.mov_sreg(32, "lanemask_lt")
    output32[cluster, lane, 10] = T.cuda.mov_sreg(32, "lanemask_gt")
    output32[cluster, lane, 11] = T.cuda.mov_sreg(32, "clock")
    output32[cluster, lane, 12] = T.cuda.mov_sreg(32, "globaltimer_hi")
    output64[cluster, lane, 0] = T.cuda.mov_sreg(64, "gridid")
    output64[cluster, lane, 1] = T.cuda.mov_sreg(64, "clock64")
    output64[cluster, lane, 2] = T.cuda.mov_sreg(64, "globaltimer")


@T.prim_func
def fetch_warp_identifier_registers(output: T.Buffer((2, 32, 2), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    output[warp, lane, 0] = T.cuda.mov_sreg(32, "warpid")
    output[warp, lane, 1] = T.cuda.mov_sreg(32, "nwarpid")


@T.prim_func
def integer_trap_predicate(flag: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.cuda.trap_when_assert_failed(flag[0])
    output[lane] = lane + 1


@T.prim_func
def floating_trap_predicate(flag: T.Buffer((1,), "float32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.cuda.trap_when_assert_failed(flag[0])
    output[lane] = lane + 1


@T.prim_func
def misaligned_cuda_pointer_helpers(
    source: T.Buffer((32,), "uint8"), destination: T.Buffer((32,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.cuda.float22half2(T.address_of(destination[1]), T.address_of(source[1]))


@T.prim_func
def named_barrier_arrive_then_sync(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    if warp == 0:
        T.ptx.bar.arrive(T.uint32(3), T.uint32(64))
    else:
        T.ptx.bar.sync(T.uint32(3), T.uint32(64))
    if warp == 0:
        T.ptx.bar.arrive(T.uint32(4), T.uint32(64))
    else:
        T.ptx.barrier.sync(T.uint32(4), T.uint32(64))
    output[warp * 32 + lane] = warp + 1


@T.prim_func
def divergent_unaligned_named_barrier(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2,), "int32", scope="shared")
    if lane == 0:
        shared[warp] = warp + 11
    if lane == 0:
        T.ptx.barrier.sync(T.uint32(5), T.uint32(64))
    else:
        T.ptx.barrier.sync(T.uint32(5), T.uint32(64))
    output[warp * 32 + lane] = shared[1 - warp]


@T.prim_func
def directed_f64_rounding(output: T.Buffer((1,), "float64")):
    T.device_entry()
    T.ptx.add.rm.f64(output[0], T.float64(1), T.float64(-(2**-54)))


def test_directed_f64_rounding_preserves_the_predecessor_of_one(tmp_path):
    module = numsim.transpile(directed_f64_rounding, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, np.float64)})
    np.testing.assert_array_equal(result.outputs["output"], [np.nextafter(1.0, 0.0)])


def test_scalar_and_warp_intrinsics_execute_with_native_semantics(tmp_path):
    module = numsim.transpile(scalar_warp_intrinsics, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "output_u32": np.zeros((32, 7), dtype=np.uint32),
            "output_f32": np.zeros((32, 4), dtype=np.float32),
            "output_u64": np.ones(32, dtype=np.uint64),
        },
    )
    lanes = np.arange(32, dtype=np.uint32)
    expected_u32 = np.zeros((32, 7), dtype=np.uint32)
    expected_u32[:, 0] = np.maximum(lanes, 1)
    expected_u32[:, 1] = np.minimum(lanes + 2, 32)
    expected_u32[:, 2] = (lanes ^ 1) + 1
    expected_u32[:, 3] = np.uint32(0xFFFFFFFF)
    lane_bits = lanes.astype(np.float32).view(np.uint32)
    next_bits = (lanes + 1).astype(np.float32).view(np.uint32)
    lane_bf16 = (lane_bits + np.uint32(0x7FFF) + ((lane_bits >> 16) & 1)) >> 16
    next_bf16 = (next_bits + np.uint32(0x7FFF) + ((next_bits >> 16) & 1)) >> 16
    expected_u32[:, 4] = lane_bf16 | (next_bf16 << 16)
    expected_u32[:, 5] = np.uint32(0xFEFFFFFF)
    expected_u32[:, 6] = np.uint32(0x23456789)
    np.testing.assert_array_equal(result.outputs["output_u32"], expected_u32)
    np.testing.assert_array_equal(result.outputs["output_u64"], np.zeros(32, dtype=np.uint64))
    np.testing.assert_array_equal(result.outputs["output_f32"][:, 2], np.maximum(lanes, 17))
    np.testing.assert_array_equal(result.outputs["output_f32"][:, 3], np.full(32, -3, np.float32))
    np.testing.assert_array_equal(
        result.outputs["output_f32"][:, 0],
        (lanes.astype(np.float32) + np.float32(0.25)).astype(np.float16).astype(np.float32),
    )
    bf16_source = lanes.astype(np.float32) + np.float32(0.5)
    bf16_source_bits = bf16_source.view(np.uint32)
    bf16_bits = (
        bf16_source_bits + np.uint32(0x7FFF) + ((bf16_source_bits >> np.uint32(16)) & np.uint32(1))
    ) >> np.uint32(16)
    np.testing.assert_array_equal(
        result.outputs["output_f32"][:, 1], (bf16_bits << np.uint32(16)).view(np.float32)
    )


def test_three_source_nan_minmax_propagates_any_nan(tmp_path):
    lhs = np.linspace(-3.0, 4.0, 32, dtype=np.float32)
    middle = np.linspace(7.0, -2.0, 32, dtype=np.float32)
    rhs = np.linspace(1.0, 5.0, 32, dtype=np.float32)
    middle[::5] = np.nan

    module = numsim.transpile(ptx_three_source_nan_minmax, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "lhs": lhs,
            "middle": middle,
            "rhs": rhs,
            "maximum": np.zeros(32, dtype=np.float32),
            "minimum": np.zeros(32, dtype=np.float32),
        },
    )

    nan_mask = np.isnan(middle)
    assert np.all(np.isnan(result.outputs["maximum"][nan_mask]))
    assert np.all(np.isnan(result.outputs["minimum"][nan_mask]))
    np.testing.assert_array_equal(
        result.outputs["maximum"][~nan_mask],
        np.maximum(np.maximum(lhs[~nan_mask], middle[~nan_mask]), rhs[~nan_mask]),
    )
    np.testing.assert_array_equal(
        result.outputs["minimum"][~nan_mask],
        np.minimum(np.minimum(lhs[~nan_mask], middle[~nan_mask]), rhs[~nan_mask]),
    )


def test_timer_finalize_is_a_numerical_noop(tmp_path):
    module = numsim.transpile(timer_finalize_is_numerical_noop, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(7, 39, dtype=np.uint32))


def test_removed_flat_arithmetic_helper_is_rejected_by_the_target_parser():
    with pytest.raises(AttributeError, match="tirx.ptx.add_f32"):
        T.call_intrin(
            "",
            "tirx.ptx_add_f32",
            T.float32(0),
            T.float32(1),
            T.float32(2),
        )


def test_cuda_shuffle_rejects_invalid_width(tmp_path):
    @T.prim_func
    def kernel(output: T.Buffer((32,), "uint32"), width: T.int32):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        output[lane] = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), T.cast(lane, "uint32"), 0, width)

    module = numsim.transpile(kernel, cache_dir=tmp_path)
    inputs = {"output": np.zeros(32, np.uint32), "width": 3}
    with pytest.raises(NumSimExecutionError, match="invalid warp shuffle selector/width"):
        numsim.Engine().run(module, inputs)
    inputs["width"] = 32
    np.testing.assert_array_equal(numsim.Engine().run(module, inputs).outputs["output"], 0)


def test_shuffle_xor_width_can_read_an_earlier_group(tmp_path):
    module = numsim.transpile(shuffle_xor_crosses_to_earlier_width_group, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})
    expected = np.concatenate((np.arange(16, dtype=np.uint32), np.arange(16, dtype=np.uint32)))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cuda_packed_ops_use_gpu_canonical_nan_encodings(tmp_path):
    module = numsim.transpile(cuda_packed_nan_semantics, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "output_u64": np.zeros((32, 2), dtype=np.uint64),
            "output_u32": np.zeros((32, 4), dtype=np.uint32),
        },
    )
    np.testing.assert_array_equal(
        result.outputs["output_u64"],
        np.full((32, 2), np.uint64(0x7FFFFFFF7FFFFFFF), dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        result.outputs["output_u32"], np.full((32, 4), np.uint32(2147450879), dtype=np.uint32)
    )


def test_cta_votes_grid_sync_and_ordering_markers(tmp_path):
    module = numsim.transpile(cta_votes_and_grid_sync, cache_dir=tmp_path)
    for max_workers in (1, 2, 8):
        result = numsim.Engine(max_workers=max_workers).run(
            module,
            {
                "flags": np.zeros(2, dtype=np.int32),
                "output_and": np.zeros(128, dtype=np.int64),
                "output_or": np.zeros(128, dtype=np.int64),
                "output_grid": np.zeros(4, dtype=np.int32),
            },
        )
        np.testing.assert_array_equal(
            result.outputs["output_and"][:64], np.ones(64, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            result.outputs["output_and"][64:], np.zeros(64, dtype=np.int64)
        )
        np.testing.assert_array_equal(result.outputs["output_or"][:64], np.ones(64, dtype=np.int64))
        np.testing.assert_array_equal(
            result.outputs["output_or"][64:], np.zeros(64, dtype=np.int64)
        )
        np.testing.assert_array_equal(result.outputs["output_grid"], np.full(4, 3, dtype=np.int32))


def test_mbarrier_one_shot_queries_observe_parity(tmp_path):
    module = numsim.transpile(mbarrier_nonblocking_queries, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((32, 4), dtype=np.uint32)})
    expected = np.ones((32, 4), dtype=np.uint32)
    expected[:, 1:3] = 0
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_mbarrier_arrival_state_token_selects_the_exact_completed_generation(tmp_path):
    module = numsim.transpile(mbarrier_state_token_query, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})
    expected = np.zeros(32, dtype=np.uint32)
    expected[0] = 1
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_mbarrier_state_token_rejects_a_generation_older_than_the_previous_one(tmp_path):
    module = numsim.transpile(mbarrier_stale_state_token, cache_dir=tmp_path)
    with pytest.raises(
        NumSimExecutionError,
        match="state token names generation 0, but the current generation is 2",
    ):
        numsim.Engine().run(module, {"output": np.zeros(4, dtype=np.uint32)})


def test_pointer_conversions_and_runtime_descriptor_patch(tmp_path):
    source = np.arange(32 * 8, dtype=np.float32).reshape(32, 8) / np.float32(7)
    descriptor = np.full(32, np.uint32(0xFFFFFFFF), dtype=np.uint32)
    module = numsim.transpile(pointer_conversions_and_descriptor, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "half": np.zeros((32, 8), dtype=np.float16),
            "roundtrip": np.zeros((32, 8), dtype=np.float32),
            "descriptor": descriptor,
        },
    )
    expected_half = source.astype(np.float16)
    np.testing.assert_array_equal(result.outputs["half"], expected_half)
    np.testing.assert_array_equal(result.outputs["roundtrip"], expected_half.astype(np.float32))
    sf_id = np.arange(32, dtype=np.uint32) % np.uint32(4)
    expected_descriptor = (
        (descriptor & np.uint32(~0x60000030 & 0xFFFFFFFF)) | (sf_id << 29) | (sf_id << 4)
    )
    np.testing.assert_array_equal(result.outputs["descriptor"], expected_descriptor)


def test_ptx_dps_sub_and_f64_arithmetic(tmp_path):
    module = numsim.transpile(ptx_dps_arithmetic, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "output_f32": np.zeros(32, dtype=np.float32),
            "output_f32x2": np.zeros(32, dtype=np.uint64),
            "output_f64": np.zeros((32, 4), dtype=np.float64),
        },
    )
    lanes_f32 = np.arange(32, dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output_f32"], lanes_f32 - np.float32(3))
    packed = result.outputs["output_f32x2"].view(np.float32).reshape(32, 2)
    np.testing.assert_array_equal(packed[:, 0], lanes_f32 - np.float32(1))
    np.testing.assert_array_equal(packed[:, 1], lanes_f32 + np.float32(8))
    lanes_f64 = np.arange(32, dtype=np.float64)
    expected = np.stack(
        (lanes_f64 + 2.0, lanes_f64 - 2.0, lanes_f64 * 2.0, lanes_f64 * 2.0 + 1.0), axis=1
    )
    np.testing.assert_array_equal(result.outputs["output_f64"], expected)


def test_ptx_dps_rounding_ftz_and_saturation_forms(tmp_path):
    module = numsim.transpile(ptx_dps_modifier_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "output_f32": np.zeros((32, 4), dtype=np.float32),
            "output_f32x2": np.zeros((32, 4), dtype=np.uint64),
        },
    )
    next_up = np.nextafter(np.float32(1), np.float32(np.inf))
    next_down_negative = np.nextafter(np.float32(-1), np.float32(-np.inf))
    expected_f32 = np.array([next_up, next_down_negative, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output_f32"], np.tile(expected_f32, (32, 1)))

    packed = result.outputs["output_f32x2"].view(np.float32).reshape(32, 4, 2)
    tiny = np.float32(np.ldexp(1.0, -149))
    expected_packed = np.array(
        [
            [next_up, np.float32(1)],
            [np.nextafter(np.float32(1), np.float32(-np.inf)), np.float32(-1)],
            [np.float32(0), np.float32(0)],
            [next_up, tiny + tiny],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(packed, np.tile(expected_packed, (32, 1, 1)))


def test_canonical_mapa_integer_address_resolves_shared_memory(tmp_path):
    module = numsim.transpile(canonical_mapa, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})
    np.testing.assert_array_equal(result.outputs["output"], np.full(32, 37, dtype=np.int32))


def test_public_mapa_u64_and_shared_cluster_widths_transpile(tmp_path):
    module = numsim.transpile(explicit_mapa_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "output": np.zeros(32, dtype=np.int32),
            "addresses": np.zeros(32, dtype=np.uint32),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(result.outputs["output"], np.full(32, 38, dtype=np.int32))
        np.testing.assert_array_equal(result.outputs["addresses"], np.full(32, 4, dtype=np.uint32))

    check()


def test_public_cvta_shared_cluster_u64_zero_extends_shared_addresses(tmp_path):
    module = numsim.transpile(explicit_cvta_shared_cluster_u64, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"addresses": np.zeros(32, dtype=np.uint64)},
    )
    np.testing.assert_array_equal(
        result.outputs["addresses"], np.arange(0, 32 * 4, 4, dtype=np.uint64)
    )


def test_mapa_and_cvta_expose_the_device_validated_integer_bits(tmp_path):
    module = numsim.transpile(mapa_exposes_integer_address_bits, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "addresses32": np.zeros((2, 32, 3), dtype=np.uint32),
            "addresses64": np.zeros((2, 32, 3), dtype=np.uint64),
        },
    )

    byte_offsets = np.arange(32, dtype=np.uint32)
    local = np.stack((byte_offsets, np.uint32(0x01000000) | byte_offsets))
    peer = local[::-1]
    expected32 = np.stack((local, peer, local), axis=-1)
    expected64 = np.stack(
        (
            local.astype(np.uint64),
            peer.astype(np.uint64),
            np.uint64(0x0000FFFE00000000) | peer.astype(np.uint64),
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(result.outputs["addresses32"], expected32)
    np.testing.assert_array_equal(result.outputs["addresses64"], expected64)


def test_mapped_addresses_survive_integer_storage_select_and_shuffle(tmp_path):
    for checker in (synccheck, racecheck):
        checker(
            mapped_addresses_use_ordinary_integer_dataflow,
            {"output": np.zeros((2, 32), dtype=np.int32)},
        ).require_clean()
    module = numsim.transpile(mapped_addresses_use_ordinary_integer_dataflow, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((2, 32), dtype=np.int32)},
    )
    source_lane = np.arange(32, dtype=np.int32) ^ np.int32(1)
    expected = (source_lane % 2) * 100 + source_lane
    np.testing.assert_array_equal(result.outputs["output"], np.stack((expected, expected)))


def test_mapa_u32_preserves_the_remote_cta_rank(tmp_path):
    module = numsim.transpile(mapa_u32_reads_the_peer_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((2, 32), dtype=np.int32)},
    )
    lanes = np.arange(32, dtype=np.int32)
    expected = np.stack((100 + lanes, lanes), axis=0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_fetch_register_models_logical_coordinates_and_uses_stable_time_tokens(tmp_path):
    output32 = np.zeros((3, 32, 13), dtype=np.int32)
    output64 = np.full((3, 32, 3), -1, dtype=np.int64)
    result = numsim.Engine().run(
        numsim.transpile(fetch_logical_and_representative_registers, cache_dir=tmp_path),
        {"output32": output32, "output64": output64},
    )

    actual32 = result.outputs["output32"]
    lanes = np.arange(32, dtype=np.uint32)
    for cluster in range(3):
        np.testing.assert_array_equal(actual32[cluster, :, 0], cluster)
        np.testing.assert_array_equal(actual32[cluster, :, 1:3], 0)
        np.testing.assert_array_equal(actual32[cluster, :, 3], 3)
        np.testing.assert_array_equal(actual32[cluster, :, 4], 1)
        np.testing.assert_array_equal(actual32[cluster, :, 5], 0)
        np.testing.assert_array_equal(actual32[cluster, :, 6], 1)
        np.testing.assert_array_equal(actual32[cluster, :, 7], lanes.astype(np.int32))
        np.testing.assert_array_equal(
            actual32[cluster, :, 8].view(np.uint32), np.left_shift(np.uint32(1), lanes)
        )
        np.testing.assert_array_equal(
            actual32[cluster, :, 9].view(np.uint32),
            np.left_shift(np.uint32(1), lanes) - np.uint32(1),
        )
        np.testing.assert_array_equal(
            actual32[cluster, :, 10].view(np.uint32),
            np.left_shift(np.uint32(0xFFFFFFFF), lanes) & ~np.left_shift(np.uint32(1), lanes),
        )
        np.testing.assert_array_equal(actual32[cluster, :, 11:13], 0)
    np.testing.assert_array_equal(result.outputs["output64"], 0)


def test_fetch_register_separates_launched_warps_from_sm100_warp_identifier_capacity(tmp_path):
    result = numsim.Engine().run(
        numsim.transpile(fetch_warp_identifier_registers, cache_dir=tmp_path),
        {"output": np.zeros((2, 32, 2), dtype=np.int32)},
    )

    expected = np.empty((2, 32, 2), dtype=np.int32)
    expected[:, :, 0] = np.arange(2, dtype=np.int32)[:, None]
    expected[:, :, 1] = 64
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_integer_trap_predicate_uses_cpp_truth_conversion(tmp_path):
    module = numsim.transpile(integer_trap_predicate, cache_dir=tmp_path)
    passed = numsim.Engine().run(
        module, {"flag": np.array([7], dtype=np.int32), "output": np.zeros(32, dtype=np.int32)}
    )
    np.testing.assert_array_equal(passed.outputs["output"], np.arange(1, 33, dtype=np.int32))

    with pytest.raises(NumSimExecutionError, match="assertion condition failed"):
        numsim.Engine().run(
            module, {"flag": np.array([0], dtype=np.int32), "output": np.zeros(32, dtype=np.int32)}
        )


def test_floating_collective_predicates_use_cpp_truth_conversion(tmp_path):
    predicates = np.ones((2, 64), dtype=np.float32)
    predicates[0] = np.float32(0.0)
    predicates[0, 1] = np.float32(-0.0)
    predicates[0, 2] = np.float32(1.0)
    predicates[0, 3] = np.float32(-2.0)
    predicates[0, 4] = np.float32(np.nan)
    bindings = {
        "predicates": predicates,
        "ballots": np.zeros((2, 64), dtype=np.uint32),
        "any_results": np.zeros((2, 64), dtype=np.int32),
        "and_results": np.zeros((2, 64), dtype=np.int64),
        "or_results": np.zeros((2, 64), dtype=np.int64),
    }

    result = numsim.Engine().run(
        numsim.transpile(floating_predicate_collectives, cache_dir=tmp_path), bindings
    )

    expected_ballots = np.zeros((2, 64), dtype=np.uint32)
    expected_ballots[0, :32] = np.uint32(0x1C)
    expected_ballots[1] = np.uint32(0xFFFFFFFF)
    np.testing.assert_array_equal(result.outputs["ballots"], expected_ballots)
    expected_any = np.zeros((2, 64), dtype=np.int32)
    expected_any[0, :32] = 1
    expected_any[1] = 1
    np.testing.assert_array_equal(result.outputs["any_results"], expected_any)
    np.testing.assert_array_equal(
        result.outputs["and_results"],
        np.concatenate(
            (np.zeros((1, 64), dtype=np.int64), np.ones((1, 64), dtype=np.int64)), axis=0
        ),
    )
    np.testing.assert_array_equal(result.outputs["or_results"], np.ones((2, 64), dtype=np.int64))


@pytest.mark.parametrize("predicate", [np.float32(7.0), np.float32(np.nan)])
def test_floating_trap_accepts_nonzero_and_nan(tmp_path, predicate):
    result = numsim.Engine().run(
        numsim.transpile(floating_trap_predicate, cache_dir=tmp_path),
        {"flag": np.array([predicate], dtype=np.float32), "output": np.zeros(32, dtype=np.int32)},
    )
    np.testing.assert_array_equal(result.outputs["output"], np.arange(1, 33, dtype=np.int32))


@pytest.mark.parametrize("predicate", [np.float32(0.0), np.float32(-0.0)])
def test_floating_trap_rejects_signed_zero(tmp_path, predicate):
    with pytest.raises(NumSimExecutionError, match="assertion condition failed"):
        numsim.Engine().run(
            numsim.transpile(floating_trap_predicate, cache_dir=tmp_path),
            {
                "flag": np.array([predicate], dtype=np.float32),
                "output": np.zeros(32, dtype=np.int32),
            },
        )


def test_cuda_pointer_helpers_check_typed_dereference_alignment(tmp_path):
    module = numsim.transpile(misaligned_cuda_pointer_helpers, cache_dir=tmp_path)
    with pytest.raises(NumSimExecutionError, match="8-byte alignment"):
        numsim.Engine().run(
            module,
            {"source": np.arange(32, dtype=np.uint8), "destination": np.zeros(32, dtype=np.uint8)},
        )


def test_named_barrier_arrive_contributes_without_blocking(tmp_path):
    module = numsim.transpile(named_barrier_arrive_then_sync, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.int32)})
    np.testing.assert_array_equal(
        result.outputs["output"], np.repeat(np.array([1, 2], dtype=np.int32), 32)
    )


def test_unaligned_named_barrier_recombines_disjoint_lane_paths(tmp_path):
    module = numsim.transpile(divergent_unaligned_named_barrier, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(64, dtype=np.int32)})

    np.testing.assert_array_equal(
        result.outputs["output"], np.repeat(np.array([12, 11], dtype=np.int32), 32)
    )


def test_current_scalar_device_intrinsics_have_stable_numeric_representatives(tmp_path):
    module = numsim.transpile(current_scalar_device_intrinsics, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "output_f32": np.zeros((3, 32), dtype=np.float32),
            "output_u32": np.full((3, 32), np.uint32(0xFFFFFFFF), dtype=np.uint32),
        },
    )

    values = np.arange(1, 33, dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output_f32"][0], values / np.float32(2))
    np.testing.assert_array_equal(result.outputs["output_f32"][1], values)
    log2_reference = np.asarray([math.log2(float(value)) for value in values], dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["output_f32"][2], log2_reference)
    np.testing.assert_array_equal(result.outputs["output_u32"], np.zeros((3, 32), dtype=np.uint32))


def test_packed_f16x2_conversion_rounds_and_preserves_register_order(tmp_path):
    high = np.linspace(-3.25, 4.5, 32, dtype=np.float32)
    low = np.linspace(7.0, -1.5, 32, dtype=np.float32)
    high[0] = np.float32(2.0)
    low[0] = np.float32(1.0)
    high[1] = np.float32(1.00048828125)
    low[1] = np.float32(-0.0)

    result = numsim.Engine().run(
        numsim.transpile(packed_f16x2_conversion, cache_dir=tmp_path),
        {"high": high, "low": low, "output": np.zeros(32, dtype=np.uint32)},
    )

    high_bits = high.astype(np.float16).view(np.uint16).astype(np.uint32)
    low_bits = low.astype(np.float16).view(np.uint16).astype(np.uint32)
    expected = np.left_shift(high_bits, np.uint32(16)) | low_bits
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.outputs["output"][0] == np.uint32(0x40003C00)
