from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim, racecheck, synccheck
from tests.numsim.support.kernels import (
    get_tmem_addr_lane_values,
    get_tmem_addr_unsigned_row_values,
    pointer_derived_shared_raw_roundtrip,
    raw_global_memory_variants,
    raw_ldmatrix_x4_b16_fragments,
    raw_load_rejects_integer_address,
    raw_shared_b128_roundtrip,
    raw_shared_byte_storage_u32_load,
    raw_shared_byte_storage_u32_store,
    raw_shared_padding_load,
    raw_shared_u16_storage_u32_load,
    raw_shared_u16_storage_u32_store,
    raw_shared_uninitialized_load,
    raw_shared_v4_u32_roundtrip,
    shared_virtual_backing_addresses,
    shared_virtual_swizzled_backing_alignment,
)
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm.script import tirx as T


@T.prim_func
def raw_bit_type_relaxed_roundtrip(
    source_f32: T.Buffer((32,), "float32"),
    source_i32: T.Buffer((32,), "int32"),
    output_f32: T.Buffer((32,), "float32"),
    output_i32: T.Buffer((32,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_f32 = T.alloc_buffer((32,), "uint32", scope="shared")
    shared_i32 = T.alloc_buffer((32,), "uint32", scope="shared")
    T.ptx.st.shared.b32(shared_f32.ptr_to([lane]), source_f32[lane])
    T.ptx.st.shared.b32(shared_i32.ptr_to([lane]), source_i32[lane])
    T.ptx.ld.shared.b32(output_f32[lane], shared_f32.ptr_to([lane]))
    T.ptx.ld.shared.b32(output_i32[lane], shared_i32.ptr_to([lane]))


@T.prim_func
def raw_integer_type_relaxed_roundtrip(
    source_i8: T.Buffer((32,), "int8"),
    source_u8: T.Buffer((32,), "uint8"),
    source_i16: T.Buffer((32,), "int16"),
    source_u16: T.Buffer((32,), "uint16"),
    source_i32: T.Buffer((32,), "int32"),
    source_u32: T.Buffer((32,), "uint32"),
    source_i64: T.Buffer((32,), "int64"),
    source_u64: T.Buffer((32,), "uint64"),
    output_i8: T.Buffer((32,), "int8"),
    output_u8: T.Buffer((32,), "uint8"),
    output_i16: T.Buffer((32,), "int16"),
    output_u16: T.Buffer((32,), "uint16"),
    output_i32: T.Buffer((32,), "int32"),
    output_u32: T.Buffer((32,), "uint32"),
    output_i64: T.Buffer((32,), "int64"),
    output_u64: T.Buffer((32,), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_u8 = T.alloc_buffer((32,), "uint8", scope="shared")
    shared_s8 = T.alloc_buffer((32,), "uint8", scope="shared")
    shared_u16 = T.alloc_buffer((32,), "uint16", scope="shared")
    shared_s16 = T.alloc_buffer((32,), "uint16", scope="shared")
    shared_u32 = T.alloc_buffer((32,), "uint32", scope="shared")
    shared_s32 = T.alloc_buffer((32,), "uint32", scope="shared")
    shared_u64 = T.alloc_buffer((32,), "uint64", scope="shared")
    shared_s64 = T.alloc_buffer((32,), "uint64", scope="shared")
    loaded_u8_as_i8 = T.local_scalar("int8")
    loaded_s8_as_u8 = T.local_scalar("uint8")
    loaded_u16_as_i16 = T.local_scalar("int16")
    loaded_s16_as_u16 = T.local_scalar("uint16")
    loaded_u32_as_i32 = T.local_scalar("int32")
    loaded_s32_as_u32 = T.local_scalar("uint32")
    loaded_u64_as_i64 = T.local_scalar("int64")
    loaded_s64_as_u64 = T.local_scalar("uint64")
    T.ptx.st.shared.u8(shared_u8.ptr_to([lane]), source_i8[lane])
    T.ptx.st.shared.s8(shared_s8.ptr_to([lane]), source_u8[lane])
    T.ptx.st.shared.u16(shared_u16.ptr_to([lane]), source_i16[lane])
    T.ptx.st.shared.s16(shared_s16.ptr_to([lane]), source_u16[lane])
    T.ptx.st.shared.u32(shared_u32.ptr_to([lane]), source_i32[lane])
    T.ptx.st.shared.s32(shared_s32.ptr_to([lane]), source_u32[lane])
    T.ptx.st.shared.u64(shared_u64.ptr_to([lane]), source_i64[lane])
    T.ptx.st.shared.s64(shared_s64.ptr_to([lane]), source_u64[lane])
    T.ptx.ld.shared.u8(loaded_u8_as_i8, shared_u8.ptr_to([lane]))
    T.ptx.ld.shared.s8(loaded_s8_as_u8, shared_s8.ptr_to([lane]))
    T.ptx.ld.shared.u16(loaded_u16_as_i16, shared_u16.ptr_to([lane]))
    T.ptx.ld.shared.s16(loaded_s16_as_u16, shared_s16.ptr_to([lane]))
    T.ptx.ld.shared.u32(loaded_u32_as_i32, shared_u32.ptr_to([lane]))
    T.ptx.ld.shared.s32(loaded_s32_as_u32, shared_s32.ptr_to([lane]))
    T.ptx.ld.shared.u64(loaded_u64_as_i64, shared_u64.ptr_to([lane]))
    T.ptx.ld.shared.s64(loaded_s64_as_u64, shared_s64.ptr_to([lane]))
    output_i8[lane] = loaded_u8_as_i8
    output_u8[lane] = loaded_s8_as_u8
    output_i16[lane] = loaded_u16_as_i16
    output_u16[lane] = loaded_s16_as_u16
    output_i32[lane] = loaded_u32_as_i32
    output_u32[lane] = loaded_s32_as_u32
    output_i64[lane] = loaded_u64_as_i64
    output_u64[lane] = loaded_s64_as_u64


@T.prim_func
def raw_global_nc_load(source: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.ld.global_.ca.nc.s32(output[lane], source.ptr_to([lane]))


@T.prim_func
def raw_predicated_global_store(
    source: T.Buffer((32,), "uint32"),
    predicate: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.st.global_.u32(output.ptr_to([lane]), source[lane], pred=predicate[lane])


@T.prim_func
def raw_subword_vector_stores(
    source: T.Buffer((128,), "uint16"),
    output_v2: T.Buffer((64,), "uint16"),
    output_v4: T.Buffer((128,), "uint16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint16", scope="shared")
    source_base = lane * 4
    T.ptx.st.global_.v2.b16(
        output_v2.ptr_to([lane * 2]), source[source_base], source[source_base + 1]
    )
    T.ptx.st.shared.v4.b16(
        shared.ptr_to([source_base]),
        source[source_base],
        source[source_base + 1],
        source[source_base + 2],
        source[source_base + 3],
    )
    T.cuda.cta_sync()
    for element in T.unroll(4):
        output_v4[source_base + element] = shared[source_base + element]


@T.prim_func
def raw_global_v4_u32_load(source: T.Buffer((128,), "uint32"), output: T.Buffer((128,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    values = T.alloc_local((4,), "uint32")
    T.ptx.ld.global_.v4.b32(values[0], values[1], values[2], values[3], source.ptr_to([lane * 4]))
    for index in T.unroll(4):
        output[lane * 4 + index] = values[index]


@T.prim_func
def raw_shared_v4_load_to_local(source: T.Buffer((4,), "uint32"), output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "uint32", scope="shared")
    local = T.alloc_local((4,), "uint32")
    if lane == 0:
        for i in T.serial(4):
            shared[i] = source[i]
        T.ptx.ld.shared.v4.u32(local[0], local[1], local[2], local[3], shared.ptr_to([0]))
        for i in T.serial(4):
            output[i] = local[i]


@T.prim_func
def raw_global_nc_v4_load_to_each_lane(
    source: T.Buffer((4,), "uint64"), output: T.Buffer((32, 4), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    local = T.alloc_local((4,), "uint64")
    T.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v4.u64"](
        local[0], local[1], local[2], local[3], source.ptr_to([0])
    )
    for i in T.serial(4):
        output[lane, i] = local[i]


@T.prim_func
def raw_global_v2_and_v8_destination_loads(
    source32: T.Buffer((256,), "uint32"),
    source64: T.Buffer((64,), "uint64"),
    out_v2_32: T.Buffer((64,), "uint32"),
    out_v8_32: T.Buffer((256,), "uint32"),
    out_v2_64: T.Buffer((64,), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pair32 = T.alloc_local((2,), "uint32")
    eight32 = T.alloc_local((8,), "uint32")
    pair64 = T.alloc_local((2,), "uint64")
    T.ptx.ld.global_.v2.b32(pair32[0], pair32[1], source32.ptr_to([lane * 2]))
    T.ptx["ld.global.L1::evict_last.L2::evict_first.v8.u32"](
        eight32[0],
        eight32[1],
        eight32[2],
        eight32[3],
        eight32[4],
        eight32[5],
        eight32[6],
        eight32[7],
        source32.ptr_to([lane * 8]),
    )
    T.ptx.ld.global_.v2.b64(pair64[0], pair64[1], source64.ptr_to([lane * 2]))
    for index in T.unroll(2):
        out_v2_32[lane * 2 + index] = pair32[index]
        out_v2_64[lane * 2 + index] = pair64[index]
    for index in T.unroll(8):
        out_v8_32[lane * 8 + index] = eight32[index]


@T.prim_func
def raw_shared_v2_ordered_destination_loads(
    source: T.Buffer((64,), "uint32"),
    relaxed: T.Buffer((64,), "uint32"),
    acquired: T.Buffer((64,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    relaxed_pair = T.alloc_local((2,), "uint32")
    acquired_pair = T.alloc_local((2,), "uint32")
    for index in T.unroll(2):
        shared[lane * 2 + index] = source[lane * 2 + index]
    T.cuda.warp_sync()
    T.ptx.ld.relaxed.cta.shared.v2.u32(relaxed_pair[0], relaxed_pair[1], shared.ptr_to([lane * 2]))
    T.ptx.ld.acquire.cta.shared.v2.u32(
        acquired_pair[0], acquired_pair[1], shared.ptr_to([lane * 2])
    )
    for index in T.unroll(2):
        relaxed[lane * 2 + index] = relaxed_pair[index]
        acquired[lane * 2 + index] = acquired_pair[index]


@T.prim_func
def raw_shared_v2_volatile_destination_load(
    source: T.Buffer((64,), "uint32"), output: T.Buffer((64,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    pair = T.alloc_local((2,), "uint32")
    for index in T.unroll(2):
        shared[lane * 2 + index] = source[lane * 2 + index]
    T.cuda.warp_sync()
    T.ptx.ld.volatile.shared.v2.u32(pair[0], pair[1], shared.ptr_to([lane * 2]))
    for index in T.unroll(2):
        output[lane * 2 + index] = pair[index]


@T.prim_func
def raw_shared_ordered_b128_load(
    source: T.Buffer((32, 4), "uint32"),
    output: T.Buffer((32, 4), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 4), "uint32", scope="shared", align=16)
    loaded = T.alloc_local((1,), "uint128")
    for index in T.unroll(4):
        shared[lane, index] = source[lane, index]
    T.cuda.warp_sync()
    T.ptx.ld.acquire.cta.shared.b128(
        loaded[0],
        T.cuda.cvta_generic_to_shared(shared.ptr_to([lane, 0])),
    )
    for index in T.unroll(4):
        output[lane, index] = loaded.view("uint32")[index]


@T.prim_func
def raw_shared_misaligned_ordered_b128_load(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "uint32", scope="shared", align=16)
    loaded = T.alloc_local((1,), "uint128")
    if lane == 0:
        for index in T.serial(8):
            shared[index] = T.uint32(index + 1)
        T.ptx.ld.acquire.cta.shared.b128(
            loaded[0],
            T.cuda.cvta_generic_to_shared(shared.ptr_to([1])),
        )
        for index in T.serial(4):
            output[index] = loaded.view("uint32")[index]


@T.prim_func
def raw_volatile_b128_uninitialized(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    loaded = T.alloc_local((1,), "uint128")
    if lane == 0:
        T.ptx.ld.volatile.shared.b128(loaded[0], shared.ptr_to([0]))
        for index in T.serial(4):
            output[index] = loaded.view("uint32")[index]


@T.prim_func
def raw_global_nc_v2_destination_load(
    source: T.Buffer((64,), "uint32"), output: T.Buffer((64,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    pair = T.alloc_local((2,), "uint32")
    T.ptx.ld.global_.nc.v2.u32(pair[0], pair[1], source.ptr_to([lane * 2]))
    for index in T.unroll(2):
        output[lane * 2 + index] = pair[index]


# Only the scalar-legal hints appear here: ptxas rejects any
# `.level2::eviction_priority` on a scalar access ("Instruction 'ld' requires
# '.v8.b32/.v4.b64' type with '.L2::evict_normal' modifier"), so the L2
# spellings are covered by the form-key sweep and by
# `raw_memory_family_eviction_and_prefetch_hints`, which stays NumSim-only.
@T.prim_func
def raw_global_nc_eviction_and_prefetch_hint_matrix(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32, 6), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["ld.global.nc.L1::evict_normal.u32"](output[lane, 0], source.ptr_to([lane]))
    T.ptx["ld.global.nc.L1::evict_unchanged.u32"](output[lane, 1], source.ptr_to([lane]))
    T.ptx["ld.global.nc.L1::evict_first.u32"](output[lane, 2], source.ptr_to([lane]))
    T.ptx["ld.global.nc.L1::evict_last.L2::256B.u32"](output[lane, 3], source.ptr_to([lane]))
    T.ptx["ld.global.nc.L2::64B.u32"](output[lane, 4], source.ptr_to([lane]))
    T.ptx["ld.global.nc.L2::128B.u32"](output[lane, 5], source.ptr_to([lane]))


@T.prim_func
def raw_sub_word_global_roundtrip(
    source: T.Buffer((32,), "int32"),
    out_b8: T.Buffer((32,), "uint32"),
    out_s8: T.Buffer((32,), "int32"),
    out_b16: T.Buffer((32,), "uint16"),
    out_s16: T.Buffer((32,), "int16"),
    out_b8_wide: T.Buffer((32,), "uint16"),
    out_b8_pair: T.Buffer((32, 2), "uint16"),
    out_b16_wide: T.Buffer((32,), "uint32"),
    out_s16_wide: T.Buffer((32,), "int32"),
    out_u16_wide_store: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    bytes8 = T.alloc_buffer((32,), "uint8", scope="shared")
    halves = T.alloc_buffer((32,), "uint16", scope="shared")
    halves_from_wide = T.alloc_buffer((32,), "uint16", scope="shared")
    value: T.let = source[lane]
    loaded_b8 = T.local_scalar("uint8")
    loaded_b8_wide = T.local_scalar("uint16")
    loaded_s8 = T.local_scalar("int8")
    loaded_b16 = T.local_scalar("uint16")
    loaded_s16 = T.local_scalar("int16")
    # PTX also lets a 16-bit access name a 32-bit register: loads zero- or
    # sign-extend into it and stores truncate out of it (PTX ISA 9.7.9.8).
    loaded_b16_wide = T.local_scalar("uint32")
    loaded_s16_wide = T.local_scalar("int32")
    loaded_u16_wide_store = T.local_scalar("uint32")
    T.ptx.st.shared.b8(bytes8.ptr_to([lane]), T.cast(value, "uint16"))
    T.ptx.st.shared.b16(halves.ptr_to([lane]), T.cast(value, "uint16"))
    T.ptx.st.shared.u16(halves_from_wide.ptr_to([lane]), T.cast(value, "uint32"))
    T.cuda.warp_sync()
    T.ptx.ld.shared.b8(loaded_b8, bytes8.ptr_to([lane]))
    T.ptx.ld.shared.b8(loaded_b8_wide, bytes8.ptr_to([lane]))
    T.ptx.ld.shared.v2.b8(
        out_b8_pair[lane, 0], out_b8_pair[lane, 1], bytes8.ptr_to([lane // 2 * 2])
    )
    T.ptx.ld.shared.s8(loaded_s8, bytes8.ptr_to([lane]))
    T.ptx.ld.shared.b16(loaded_b16, halves.ptr_to([lane]))
    T.ptx.ld.shared.s16(loaded_s16, halves.ptr_to([lane]))
    T.ptx.ld.shared.b16(loaded_b16_wide, halves.ptr_to([lane]))
    T.ptx.ld.shared.s16(loaded_s16_wide, halves.ptr_to([lane]))
    T.ptx.ld.shared.u16(loaded_u16_wide_store, halves_from_wide.ptr_to([lane]))
    out_b8[lane] = T.cast(loaded_b8, "uint32")
    out_b8_wide[lane] = loaded_b8_wide
    out_s8[lane] = T.cast(loaded_s8, "int32")
    out_b16[lane] = loaded_b16
    out_s16[lane] = loaded_s16
    out_b16_wide[lane] = loaded_b16_wide
    out_s16_wide[lane] = loaded_s16_wide
    out_u16_wide_store[lane] = loaded_u16_wide_store


@T.prim_func
def raw_sub_word_ordered_forms(
    source: T.Buffer((32,), "uint32"),
    acquired_u8: T.Buffer((32,), "uint32"),
    acquired_u16: T.Buffer((32,), "uint16"),
    volatile_s16: T.Buffer((32,), "int16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    bytes8 = T.alloc_buffer((32,), "uint8", scope="shared")
    halves = T.alloc_buffer((32,), "uint16", scope="shared")
    loaded_u8 = T.local_scalar("uint8")
    loaded_u16 = T.local_scalar("uint16")
    loaded_s16 = T.local_scalar("int16")
    T.ptx.st.relaxed.cta.shared.u8(bytes8.ptr_to([lane]), T.cast(source[lane], "uint8"))
    T.ptx.st.release.cta.shared.u16(halves.ptr_to([lane]), T.cast(source[lane], "uint16"))
    T.cuda.warp_sync()
    T.ptx.ld.acquire.cta.shared.u8(loaded_u8, bytes8.ptr_to([lane]))
    T.ptx.ld.acquire.cta.shared.u16(loaded_u16, halves.ptr_to([lane]))
    T.ptx.ld.volatile.shared.s16(loaded_s16, halves.ptr_to([lane]))
    acquired_u8[lane] = T.cast(loaded_u8, "uint32")
    acquired_u16[lane] = loaded_u16
    volatile_s16[lane] = loaded_s16


@T.prim_func
def raw_memory_family_eviction_and_prefetch_hints(
    source: T.Buffer((32,), "uint32"),
    loaded: T.Buffer((32, 5), "uint32"),
    stored: T.Buffer((32, 3), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["ld.global.L1::evict_first.L2::128B.u32"](loaded[lane, 0], source.ptr_to([lane]))
    T.ptx["ld.relaxed.gpu.global.L1::evict_unchanged.L2::64B.u32"](
        loaded[lane, 1], source.ptr_to([lane])
    )
    T.ptx["ld.acquire.gpu.global.L1::evict_last.u32"](loaded[lane, 2], source.ptr_to([lane]))
    T.ptx["ld.volatile.global.L2::256B.u32"](loaded[lane, 3], source.ptr_to([lane]))
    T.ptx["ld.global.nc.L1::evict_normal.u32"](loaded[lane, 4], source.ptr_to([lane]))
    T.ptx["st.global.L1::evict_first.u32"](stored.ptr_to([lane, 0]), source[lane])
    T.ptx["st.relaxed.gpu.global.L1::evict_unchanged.u32"](stored.ptr_to([lane, 1]), source[lane])
    T.ptx.st.release.gpu.global_.u32(stored.ptr_to([lane, 2]), source[lane])


@T.prim_func
def raw_shared_u32_address_b128_source_store(
    source: T.Buffer((32, 4), "uint32"), output: T.Buffer((32, 4), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 4), "uint32", scope="shared")
    local = T.alloc_local((4,), "uint32")
    for i in T.unroll(4):
        local[i] = source[lane, i]
    shared_address: T.let = T.cuda.cvta_generic_to_shared(shared.ptr_to([lane, 0]))
    T.ptx.st.weak.shared__cta.v4.b32(
        shared_address,
        local[0],
        local[1],
        local[2],
        local[3],
    )
    T.cuda.warp_sync()
    for i in T.unroll(4):
        output[lane, i] = shared[lane, i]


def test_ptx_v4_destination_load_copies_all_elements_to_local_storage(tmp_path):
    source = np.asarray([0x12345678, 7, 0xABCDEF01, 99], dtype=np.uint32)
    module = numsim.transpile(raw_shared_v4_load_to_local, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros_like(source)})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_global_nc_v4_destination_load_populates_every_calling_lane(tmp_path):
    source = np.asarray([0x0123456789ABCDEF, 7, 0xFEDCBA9876543210, 99], dtype=np.uint64)
    module = numsim.transpile(raw_global_nc_v4_load_to_each_lane, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.zeros((32, 4), dtype=np.uint64)}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.tile(source, (32, 1)))


def test_shared_u32_address_b128_source_store_preserves_all_16_bytes(tmp_path):
    source = (
        np.arange(32 * 4, dtype=np.uint32).reshape(32, 4) * np.uint32(0x01010101)
    ) ^ np.uint32(0xA55AA55A)
    module = numsim.transpile(raw_shared_u32_address_b128_source_store, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros_like(source)})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_v2_and_v8_destination_loads_move_every_element(tmp_path):
    source32 = (np.arange(256, dtype=np.uint32) * np.uint32(0x01020408)) ^ np.uint32(0x5AA55AA5)
    source64 = (np.arange(64, dtype=np.uint64) * np.uint64(0x0102040810204080)) ^ np.uint64(
        0xA55A5AA5A55A5AA5
    )
    module = numsim.transpile(raw_global_v2_and_v8_destination_loads, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source32": source32,
            "source64": source64,
            "out_v2_32": np.zeros(64, dtype=np.uint32),
            "out_v8_32": np.zeros(256, dtype=np.uint32),
            "out_v2_64": np.zeros(64, dtype=np.uint64),
        },
    )

    np.testing.assert_array_equal(result.outputs["out_v2_32"], source32[:64])
    np.testing.assert_array_equal(result.outputs["out_v8_32"], source32)
    np.testing.assert_array_equal(result.outputs["out_v2_64"], source64)


def test_ordered_v2_destination_loads_keep_their_memory_semantics(tmp_path):
    """`.relaxed`, `.acquire` and `.volatile` all take `.vec` in PTX.

    The vector spelling must select the same ordering specialization the
    scalar spelling does, not silently fall back to a plain load.
    """

    source = (np.arange(64, dtype=np.uint32) * np.uint32(0x9E3779B1)) ^ np.uint32(0x1234_5678)
    module = numsim.transpile(raw_shared_v2_ordered_destination_loads, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "relaxed": np.zeros(64, dtype=np.uint32),
            "acquired": np.zeros(64, dtype=np.uint32),
        },
    )

    for name in ("relaxed", "acquired"):
        np.testing.assert_array_equal(result.outputs[name], source)

    assert "v2::mem::variant::Relaxed<v2::mem::variant::Cta>" in module.rust_source
    assert "v2::mem::variant::Acquire<v2::mem::variant::Cta>" in module.rust_source

    volatile_module = numsim.transpile(raw_shared_v2_volatile_destination_load, cache_dir=tmp_path)
    volatile_result = numsim.Engine().run(
        volatile_module, {"source": source, "output": np.zeros(64, dtype=np.uint32)}
    )
    np.testing.assert_array_equal(volatile_result.outputs["output"], source)
    assert "v2::mem::variant::Volatile" in volatile_module.rust_source


def test_ordered_b128_load_preserves_both_64_bit_halves(tmp_path):
    source = (
        np.arange(32 * 4, dtype=np.uint32).reshape(32, 4) * np.uint32(0x1020_4081)
    ) ^ np.asarray([0xA55A_A55A, 0x0123_4567, 0x89AB_CDEF, 0x55AA_55AA], dtype=np.uint32)
    module = numsim.transpile(raw_shared_ordered_b128_load, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros_like(source)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_ordered_b128_load_retains_16_byte_alignment_contract(tmp_path):
    module = numsim.transpile(raw_shared_misaligned_ordered_b128_load, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="b128 requires 16-byte alignment"):
        numsim.Engine().run(module, {"output": np.zeros(4, dtype=np.uint32)})


def test_volatile_b128_load_preserves_uninitialized_review(tmp_path):
    # Keep the original formerly unsupported fixture, including uninitialized
    # shared bytes: supporting the load must not turn REVIEW into clean.
    for checker in (synccheck, racecheck):
        report = checker(raw_volatile_b128_uninitialized, {"output": np.zeros(4, np.uint32)})
        assert report.verdict == "review", report.format()
        assert {finding.kind for finding in report.findings} == {"uninitialized_read"}
    module = numsim.transpile(raw_volatile_b128_uninitialized, cache_dir=tmp_path)
    assert "v2::mem::variant::Volatile" in module.rust_source
    result = numsim.Engine().run(module, {"output": np.full(4, 0xDEADBEEF, np.uint32)})
    np.testing.assert_array_equal(result.outputs["output"], np.zeros(4, np.uint32))
    assert result.verdict == "review"
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}


def test_global_nc_v2_destination_load_moves_both_elements(tmp_path):
    source = (np.arange(64, dtype=np.uint32) * np.uint32(0x0BADF00D)) ^ np.uint32(0x55AA55AA)
    module = numsim.transpile(raw_global_nc_v2_destination_load, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.zeros(64, dtype=np.uint32)}
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_sub_word_ptx_types_extend_into_their_register_width(tmp_path):
    """PTX zero-extends `.b8`/`.u8`/`.b16`/`.u16` and sign-extends `.s8`/`.s16`.

    The byte written by a `.b8` store must come back as an unsigned 32-bit
    value through `.b8`, and as a sign-extended 32-bit value through `.s8`.
    The same holds one width up, where a 16-bit access may name a 32-bit
    register on either side of the transfer.
    """

    source = np.asarray([(-128 + (index * 9) % 256) for index in range(32)], dtype=np.int32)
    module = numsim.transpile(raw_sub_word_global_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "out_b8": np.zeros(32, dtype=np.uint32),
            "out_s8": np.zeros(32, dtype=np.int32),
            "out_b16": np.zeros(32, dtype=np.uint16),
            "out_s16": np.zeros(32, dtype=np.int16),
            "out_b8_wide": np.zeros(32, dtype=np.uint16),
            "out_b8_pair": np.zeros((32, 2), dtype=np.uint16),
            "out_b16_wide": np.zeros(32, dtype=np.uint32),
            "out_s16_wide": np.zeros(32, dtype=np.int32),
            "out_u16_wide_store": np.zeros(32, dtype=np.uint32),
        },
    )

    low_byte = (source.astype(np.uint32) & np.uint32(0xFF)).astype(np.uint32)
    np.testing.assert_array_equal(result.outputs["out_b8"], low_byte)
    np.testing.assert_array_equal(result.outputs["out_b8_wide"], low_byte.astype(np.uint16))
    np.testing.assert_array_equal(
        result.outputs["out_b8_pair"], low_byte.reshape(16, 2).repeat(2, axis=0).astype(np.uint16)
    )
    np.testing.assert_array_equal(
        result.outputs["out_s8"], low_byte.astype(np.uint8).astype(np.int8).astype(np.int32)
    )
    low_half = (source.astype(np.uint32) & np.uint32(0xFFFF)).astype(np.uint16)
    np.testing.assert_array_equal(result.outputs["out_b16"], low_half)
    np.testing.assert_array_equal(result.outputs["out_s16"], low_half.astype(np.int16))
    np.testing.assert_array_equal(result.outputs["out_b16_wide"], low_half.astype(np.uint32))
    np.testing.assert_array_equal(
        result.outputs["out_s16_wide"], low_half.astype(np.int16).astype(np.int32)
    )
    # `.u16` stored out of a 32-bit register keeps only the low half.
    np.testing.assert_array_equal(result.outputs["out_u16_wide_store"], low_half.astype(np.uint32))


def test_sub_word_ptx_types_keep_their_ordering_specializations(tmp_path):
    source = (np.arange(32, dtype=np.uint32) * np.uint32(0x9E3779B1)) ^ np.uint32(0x0BADF00D)
    module = numsim.transpile(raw_sub_word_ordered_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "acquired_u8": np.zeros(32, dtype=np.uint32),
            "acquired_u16": np.zeros(32, dtype=np.uint16),
            "volatile_s16": np.zeros(32, dtype=np.int16),
        },
    )

    np.testing.assert_array_equal(result.outputs["acquired_u8"], source & np.uint32(0xFF))
    low_half = (source & np.uint32(0xFFFF)).astype(np.uint16)
    np.testing.assert_array_equal(result.outputs["acquired_u16"], low_half)
    np.testing.assert_array_equal(result.outputs["volatile_s16"], low_half.astype(np.int16))


def test_ptx_eviction_and_prefetch_hints_preserve_loaded_values(tmp_path):
    source = (np.arange(32, dtype=np.uint32) * np.uint32(0x01020408)) ^ np.uint32(0x5AA55AA5)
    module = numsim.transpile(raw_global_nc_eviction_and_prefetch_hint_matrix, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.zeros((32, 6), dtype=np.uint32)}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.repeat(source[:, None], 6, axis=1))


def test_raw_memory_hint_forms_preserve_loaded_and_stored_values(tmp_path):
    source = (np.arange(32, dtype=np.uint32) * np.uint32(0x9E3779B1)) ^ np.uint32(0x5AA55AA5)
    module = numsim.transpile(raw_memory_family_eviction_and_prefetch_hints, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "loaded": np.zeros((32, 5), dtype=np.uint32),
            "stored": np.zeros((32, 3), dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["loaded"], np.repeat(source[:, None], 5, axis=1))
    np.testing.assert_array_equal(result.outputs["stored"], np.repeat(source[:, None], 3, axis=1))


def test_pointer_derived_shared_views_raw_memory_and_cvta_preserve_aliasing(
    tmp_path, expect_harness_surface
):
    source = (np.arange(32, dtype=np.uint32) * np.uint32(17)) ^ np.uint32(0xA5A55A5A)
    loaded = np.zeros(32, dtype=np.uint32)
    aliased = np.zeros(32, dtype=np.uint32)
    addresses = np.zeros(32, dtype=np.uint32)

    spec = analyze(pointer_derived_shared_raw_roundtrip)
    assert spec.unsupported == ()
    module = numsim.transpile(pointer_derived_shared_raw_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "loaded": loaded, "aliased": aliased, "addresses": addresses}
    )

    np.testing.assert_array_equal(result.outputs["loaded"], source)
    np.testing.assert_array_equal(result.outputs["aliased"], source)
    np.testing.assert_array_equal(
        result.outputs["addresses"], np.arange(48, 48 + 32 * 4, 4, dtype=np.uint32)
    )

    def check_pointer_identity(value):
        np.testing.assert_array_equal(value, np.arange(48, 48 + 32 * 4, 4, dtype=np.uint32))

    expect_harness_surface(
        lambda: result.outputs["addresses"],
        check_pointer_identity,
    )


def test_cvta_assigns_distinct_aligned_virtual_bases_to_shared_backings(tmp_path):
    addresses = np.zeros(2, dtype=np.uint32)

    module = numsim.transpile(shared_virtual_backing_addresses, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"addresses": addresses})

    np.testing.assert_array_equal(result.outputs["addresses"], np.array([4, 72], dtype=np.uint32))


def test_cvta_aligns_each_shared_backing_to_its_declared_alignment(tmp_path):
    addresses = np.zeros(3, dtype=np.uint32)

    module = numsim.transpile(shared_virtual_swizzled_backing_alignment, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"addresses": addresses})

    np.testing.assert_array_equal(
        result.outputs["addresses"], np.array([4, 128, 140], dtype=np.uint32)
    )


def test_get_tmem_addr_packs_wrapped_row_and_column_offsets_per_lane(tmp_path):
    output = np.zeros(64, dtype=np.uint32)

    module = numsim.transpile(get_tmem_addr_lane_values, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    lanes = np.arange(32, dtype=np.uint32)
    expected = np.empty(64, dtype=np.uint32)
    expected[:32] = (np.uint32(0x0010) << np.uint32(16)) | (
        (np.uint32(0xFFF0) + lanes * np.uint32(3)) & np.uint32(0xFFFF)
    )
    expected[32:] = (np.uint32(0xFFF0) << np.uint32(16)) | (
        (np.uint32(0x0010) - lanes) & np.uint32(0xFFFF)
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "get_tmem_addr(" in module.rust_source
    assert "fn get_tmem_addr(" not in module.rust_source


def test_get_tmem_addr_accepts_unsigned_row_offsets(tmp_path):
    module = numsim.transpile(get_tmem_addr_unsigned_row_values, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})

    lanes = np.arange(32, dtype=np.uint32)
    expected = (
        ((np.uint32(0xFFF0) + np.uint32(32) + lanes) & np.uint32(0xFFFF)) << 16
    ) | np.uint32(0x10)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_shared_v4_u32_store_writes_four_contiguous_words_per_lane(tmp_path):
    source = (np.arange(128, dtype=np.uint32) * np.uint32(0x01010101)) ^ np.uint32(0xA55AA55A)
    output = np.zeros_like(source)

    module = numsim.transpile(raw_shared_v4_u32_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_global_v4_u32_load_reads_four_contiguous_words_per_lane(tmp_path):
    source = (np.arange(128, dtype=np.uint32) * np.uint32(0x01020408)) ^ np.uint32(0x5AA55AA5)
    output = np.zeros_like(source)

    module = numsim.transpile(raw_global_v4_u32_load, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_shared_b128_store_copies_four_contiguous_words_per_lane(tmp_path):
    source = (np.arange(128, dtype=np.uint32) * np.uint32(0x01020408)) ^ np.uint32(0xA55AA55A)

    module = numsim.transpile(raw_shared_b128_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros_like(source)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_bit_types_follow_ptx_relaxed_type_rules(tmp_path):
    source_f32 = np.linspace(-7.0, 9.0, 32, dtype=np.float32)
    source_i32 = np.arange(-16, 16, dtype=np.int32)
    module = numsim.transpile(raw_bit_type_relaxed_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source_f32": source_f32,
            "source_i32": source_i32,
            "output_f32": np.zeros(32, dtype=np.float32),
            "output_i32": np.zeros(32, dtype=np.int32),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(result.outputs["output_f32"], source_f32)
        np.testing.assert_array_equal(result.outputs["output_i32"], source_i32)

    check()


def test_raw_signed_and_unsigned_types_follow_ptx_relaxed_type_rules(tmp_path):
    source_i8 = np.arange(-16, 16, dtype=np.int8)
    source_u8 = np.arange(32, dtype=np.uint8) ^ np.uint8(0x80)
    source_i16 = np.arange(-16, 16, dtype=np.int16) * np.int16(257)
    source_u16 = np.arange(32, dtype=np.uint16) ^ np.uint16(0x8000)
    source_i32 = np.arange(-16, 16, dtype=np.int32)
    source_u32 = np.arange(32, dtype=np.uint32) ^ np.uint32(0x80000000)
    source_i64 = np.arange(-16, 16, dtype=np.int64) * np.int64(0x0101010101010101)
    source_u64 = np.arange(32, dtype=np.uint64) ^ np.uint64(0x8000000000000000)
    module = numsim.transpile(raw_integer_type_relaxed_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source_i8": source_i8,
            "source_u8": source_u8,
            "source_i16": source_i16,
            "source_u16": source_u16,
            "source_i32": source_i32,
            "source_u32": source_u32,
            "source_i64": source_i64,
            "source_u64": source_u64,
            "output_i8": np.zeros(32, dtype=np.int8),
            "output_u8": np.zeros(32, dtype=np.uint8),
            "output_i16": np.zeros(32, dtype=np.int16),
            "output_u16": np.zeros(32, dtype=np.uint16),
            "output_i32": np.zeros(32, dtype=np.int32),
            "output_u32": np.zeros(32, dtype=np.uint32),
            "output_i64": np.zeros(32, dtype=np.int64),
            "output_u64": np.zeros(32, dtype=np.uint64),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(result.outputs["output_i8"], source_i8)
        np.testing.assert_array_equal(result.outputs["output_u8"], source_u8)
        np.testing.assert_array_equal(result.outputs["output_i16"], source_i16)
        np.testing.assert_array_equal(result.outputs["output_u16"], source_u16)
        np.testing.assert_array_equal(result.outputs["output_i32"], source_i32)
        np.testing.assert_array_equal(result.outputs["output_u32"], source_u32)
        np.testing.assert_array_equal(result.outputs["output_i64"], source_i64)
        np.testing.assert_array_equal(result.outputs["output_u64"], source_u64)

    check()


def test_raw_global_nc_load_preserves_global_payload(tmp_path):
    source = np.arange(-16, 16, dtype=np.int32)
    module = numsim.transpile(raw_global_nc_load, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros(32, dtype=np.int32)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_shared_load_behavior_does_not_depend_on_buffer_name(tmp_path):
    output = np.full(32, np.uint32(0xDEADBEEF), dtype=np.uint32)

    module = numsim.transpile(raw_shared_padding_load, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.array([17, *([0] * 31)], dtype=np.uint32)
    )
    assert result.verdict == "review"
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}


def test_ordinary_raw_shared_load_zero_fills_uninitialized_bytes_for_review(tmp_path):
    output = np.full(32, np.uint32(0xDEADBEEF), dtype=np.uint32)

    module = numsim.transpile(raw_shared_uninitialized_load, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(
        result.outputs["output"], np.array([17, *([0] * 31)], dtype=np.uint32)
    )
    assert result.verdict == "review"
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}


def test_raw_u32_load_retags_explicit_byte_storage_for_one_operation(tmp_path):
    output = np.zeros(32, dtype=np.uint32)

    module = numsim.transpile(raw_shared_byte_storage_u32_load, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.arange(1, 33, dtype=np.uint32) * np.uint32(0x01010101)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_u32_load_spans_two_u16_elements(tmp_path):
    """A `.b32` access to a `uint16` buffer reads two adjacent elements.

    The instruction width is independent of the backing element width; this
    case reads a pair of halves through one aligned 32-bit load.
    """

    output = np.zeros(32, dtype=np.uint32)

    module = numsim.transpile(raw_shared_u16_storage_u32_load, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    lane = np.arange(32, dtype=np.uint32)
    np.testing.assert_array_equal(result.outputs["output"], lane | ((lane + 1) << 16))


def test_raw_u32_store_retags_explicit_byte_storage_for_one_operation(tmp_path):
    output = np.zeros(128, dtype=np.uint8)

    module = numsim.transpile(raw_shared_byte_storage_u32_store, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    values = np.arange(1, 33, dtype=np.uint32) * np.uint32(0x01020304)
    np.testing.assert_array_equal(result.outputs["output"], values.view(np.uint8))


def test_raw_u32_store_spans_two_u16_elements(tmp_path):
    """The store counterpart of `test_raw_u32_load_spans_two_u16_elements`."""

    module = numsim.transpile(raw_shared_u16_storage_u32_store, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(2, dtype=np.uint16)})

    np.testing.assert_array_equal(
        result.outputs["output"], np.array([0x3344, 0x1122], dtype=np.uint16)
    )


def test_raw_ldmatrix_x4_b16_uses_ptx_address_and_fragment_lane_mapping(tmp_path):
    output = np.zeros(128, dtype=np.uint32)

    module = numsim.transpile(raw_ldmatrix_x4_b16_fragments, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.empty((4, 32), dtype=np.uint32)
    for matrix in range(4):
        for lane in range(32):
            provider_lane = matrix * 8 + lane // 4
            start = provider_lane * 16 + (lane % 4) * 4
            fragment = bytes((start + byte) & 0xFF for byte in range(4))
            expected[matrix, lane] = np.uint32(int.from_bytes(fragment, "little"))
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(-1))


def test_raw_global_load_store_modifiers_share_immediate_physical_memory(tmp_path):
    source = np.arange(32, dtype=np.uint64) * np.uint64(0x1_0000_0001)
    values = np.linspace(-2.0, 3.0, 32, dtype=np.float32)
    loaded = np.zeros(128, dtype=np.uint64)
    stored = np.zeros((5, 32), dtype=np.float32)

    module = numsim.transpile(raw_global_memory_variants, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "values": values, "loaded": loaded, "stored": stored}
    )

    np.testing.assert_array_equal(result.outputs["loaded"], np.tile(source, 4))
    expected_stored = np.stack(
        tuple(values + np.float32(increment) for increment in range(1, 6)), axis=0
    )
    np.testing.assert_array_equal(result.outputs["stored"], expected_stored)


def test_raw_store_predicate_selects_writing_lanes(tmp_path):
    lane = np.arange(32, dtype=np.uint32)
    source = lane * np.uint32(17) + np.uint32(3)
    predicate = (lane % np.uint32(3) == 1).astype(np.uint32)
    sentinel = np.full(32, np.uint32(0xDEADBEEF), dtype=np.uint32)

    result = numsim.Engine().run(
        numsim.transpile(raw_predicated_global_store, cache_dir=tmp_path),
        {"source": source, "predicate": predicate, "output": sentinel},
    )

    np.testing.assert_array_equal(
        result.outputs["output"], np.where(predicate != 0, source, sentinel)
    )


def test_raw_subword_vector_stores_preserve_each_element(tmp_path):
    source = np.arange(128, dtype=np.uint16) * np.uint16(257) + np.uint16(3)

    result = numsim.Engine().run(
        numsim.transpile(raw_subword_vector_stores, cache_dir=tmp_path),
        {
            "source": source,
            "output_v2": np.zeros(64, dtype=np.uint16),
            "output_v4": np.zeros(128, dtype=np.uint16),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["output_v2"], source.reshape(32, 4)[:, :2].reshape(-1)
    )
    np.testing.assert_array_equal(result.outputs["output_v4"], source)


def test_raw_load_rejects_an_unmapped_integer_when_the_address_is_consumed(tmp_path):
    from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module
    spec = analyze(raw_load_rejects_integer_address)
    assert spec.unsupported == ()
    source = emit_rust_module(spec, raw_load_rejects_integer_address)
    assert "physical_ptr_from_generic_addresses_u64" in source
    module = numsim.transpile(raw_load_rejects_integer_address, cache_dir=tmp_path)
    from tirx_harness import racecheck, synccheck

    inputs = {"output": np.zeros(32, dtype=np.uint32)}
    for checker in (synccheck, racecheck):
        report = checker(raw_load_rejects_integer_address, inputs)
        # Missing binding is not proof of OOB. Like an unbound store, this
        # load must fail closed at the consumer, not during integer transport.
        assert report.verdict == "incomplete", report.format()
        assert [finding.kind for finding in report.findings] == ["analysis_incomplete"]
        assert [finding.details["reason"] for finding in report.findings] == [
            "integer_address_without_binding"
        ]
        assert "T.ptx.ld" in report.format()
    with pytest.raises(numsim.NumSimExecutionError, match="integer_address_without_binding"):
        numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.uint32)})
