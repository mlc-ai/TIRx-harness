from __future__ import annotations

import tvm
from tirx_harness.numsim.transpiler.frontend import analyze

import ml_dtypes
import numpy as np
import pytest

from tirx_harness import numsim, racecheck
from tirx_harness.numsim.bindings import prepare_bindings
from tests.numsim.support.manifest import emitted_calls, parse_kernel
from tvm import tirx
from tvm.script import tirx as T


_OBSOLETE_CP_ASYNC_BULK_SOURCE = """
@T.prim_func
def invalid(source: T.handle):
    T.device_entry()
    T.ptx.cp_async.bulk(source, 0, source, 8, 16, 7, dtype="uint16")
"""


@T.prim_func
def cp_async_plain_4(source: T.Buffer((128,), "uint8"), output: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")
    T.ptx["cp.async.ca.shared.global.L2::64B"](
        T.address_of(shared[lane * 4]), T.address_of(source[lane * 4]), 4
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


@T.prim_func
def cp_async_cache_hint_16(source: T.Buffer((512,), "uint8"), output: T.Buffer((512,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    T.ptx["cp.async.cg.shared.global.L2::cache_hint.L2::128B"](
        T.address_of(shared[lane * 16]),
        T.address_of(source[lane * 16]),
        16,
        T.uint64(0x12F0000000000000),
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(16):
        output[lane * 16 + element] = shared[lane * 16 + element]


@T.prim_func
def cp_async_bound_raw_shared_offset(
    source: T.Buffer((128,), "uint8"), output: T.Buffer((128,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint8", scope="shared")
    shared_base: T.uint32 = T.cuda.cvta_generic_to_shared(shared.ptr_to([0]))
    T.ptx["cp.async.ca.shared.global"](
        shared_base + T.cast(128 + lane * 4, "uint32"),
        T.address_of(source[lane * 4]),
        4,
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[128 + lane * 4 + element]


@T.prim_func
def cp_async_predicate_skip_8(source: T.Buffer((256,), "uint8"), output: T.Buffer((256,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint8", scope="shared")
    for element in T.serial(8):
        shared[lane * 8 + element] = T.uint8(0xA5)
    T.cuda.warp_sync()
    T.ptx["cp.async.ca.shared.global"](
        T.address_of(shared[lane * 8]),
        T.address_of(source[lane * 8]),
        8,
        pred=lane < 16,
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(8):
        output[lane * 8 + element] = shared[lane * 8 + element]


@T.prim_func
def cp_async_ca_ignore_src_zero_fill_8(
    source: T.Buffer((128,), "uint8"), output: T.Buffer((256,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint8", scope="shared")
    for element in T.serial(8):
        shared[lane * 8 + element] = T.uint8(0xA5)
    T.cuda.warp_sync()
    T.ptx["cp.async.ca.shared.global"](
        T.address_of(shared[lane * 8]),
        T.address_of(source[lane * 8]),
        8,
        T.ptx.pred(lane >= 16),
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(8):
        output[lane * 8 + element] = shared[lane * 8 + element]


@T.prim_func
def cp_async_cg_ignore_src_zero_fill_16(
    source: T.Buffer((256,), "uint8"), output: T.Buffer((512,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    for element in T.serial(16):
        shared[lane * 16 + element] = T.uint8(0x5A)
    T.cuda.warp_sync()
    T.ptx["cp.async.cg.shared.global"](
        T.address_of(shared[lane * 16]),
        T.address_of(source[lane * 16]),
        16,
        T.ptx.pred(lane >= 16),
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(16):
        output[lane * 16 + element] = shared[lane * 16 + element]


@T.prim_func
def cuda_atomic_add_float32(
    counter: T.Buffer((1,), "float32"),
    old_values: T.Buffer((4,), "float32"),
    final_value: T.Buffer((1,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        old_values[lane] = T.cuda.atomic_add(counter.data, T.float32(0.5))
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_float16(
    counter: T.Buffer((1,), "float16"),
    old_values: T.Buffer((4,), "float16"),
    final_value: T.Buffer((1,), "float16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        old_values[lane] = T.cuda.atomic_add(counter.data, T.cast(T.float32(0.5), "float16"))
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_bfloat16(
    counter: T.Buffer((1,), "bfloat16"),
    old_values: T.Buffer((4,), "bfloat16"),
    final_value: T.Buffer((1,), "bfloat16"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        old_values[lane] = T.cuda.atomic_add(counter.data, T.cast(T.float32(0.5), "bfloat16"))
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_float32x2(
    counter: T.Buffer((1,), "float32x2"),
    increments: T.Buffer((4,), "float32x2"),
    old_values: T.Buffer((4,), "float32x2"),
    final_value: T.Buffer((1,), "float32x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_float16x2(
    counter: T.Buffer((1,), "float16x2"),
    increments: T.Buffer((2,), "float16x2"),
    old_values: T.Buffer((2,), "float16x2"),
    final_value: T.Buffer((1,), "float16x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 2:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_float16x2_shared(
    initial: T.Buffer((1,), "float16x2"),
    increments: T.Buffer((2,), "float16x2"),
    old_values: T.Buffer((2,), "float16x2"),
    final_value: T.Buffer((1,), "float16x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    counter = T.alloc_buffer((1,), "float16x2", scope="shared")
    if lane == 0:
        counter[0] = initial[0]
    T.cuda.warp_sync()
    if lane < 2:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_add_bfloat16x2(
    counter: T.Buffer((1,), "bfloat16x2"),
    increments: T.Buffer((2,), "bfloat16x2"),
    old_values: T.Buffer((2,), "bfloat16x2"),
    final_value: T.Buffer((1,), "bfloat16x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 2:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def ptx_red_half_bfloat16x2(
    counter: T.Buffer((1,), "uint32"), increments: T.Buffer((4,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        T.ptx["red.global.add.noftz.bf16x2"](
            counter.ptr_to([0]),
            increments[lane],
        )


@T.prim_func
def ptx_predicated_red_f32(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["red.global.add.f32"](
        output.ptr_to([lane]),
        T.float32(1.0),
        pred=lane < 16,
    )


@T.prim_func
def ptx_atomic_add_f32_vectors(
    counter: T.Buffer((6,), "float32"),
    increments: T.Buffer((6,), "float32"),
    old_values: T.Buffer((6,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx["atom.global.add.v4.f32"](
        old_values[0],
        old_values[1],
        old_values[2],
        old_values[3],
        counter.ptr_to([0]),
        increments[0],
        increments[1],
        increments[2],
        increments[3],
        pred=lane == 0,
    )
    T.ptx["atom.global.add.v2.f32"](
        old_values[4],
        old_values[5],
        counter.ptr_to([4]),
        increments[4],
        increments[5],
        pred=lane == 0,
    )


@T.prim_func
def cuda_atomic_add_float32x4(
    counter: T.Buffer((1,), "float32x4"),
    increments: T.Buffer((2,), "float32x4"),
    old_values: T.Buffer((2,), "float32x4"),
    final_value: T.Buffer((1,), "float32x4"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 2:
        old_values[lane] = T.cuda.atomic_add(counter.data, increments[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = counter[0]


@T.prim_func
def cuda_atomic_cas_uint64x2(
    cell: T.Buffer((1,), "uint64x2"),
    compares: T.Buffer((3,), "uint64x2"),
    replacements: T.Buffer((3,), "uint64x2"),
    old_values: T.Buffer((3,), "uint64x2"),
    final_value: T.Buffer((1,), "uint64x2"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 3:
        old_values[lane] = T.cuda.atomic_cas(cell.data, compares[lane], replacements[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = cell[0]


@T.prim_func
def cuda_atomic_cas_float32x4(
    cell: T.Buffer((1,), "float32x4"),
    compares: T.Buffer((2,), "float32x4"),
    replacements: T.Buffer((2,), "float32x4"),
    old_values: T.Buffer((2,), "float32x4"),
    final_value: T.Buffer((1,), "float32x4"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 2:
        old_values[lane] = T.cuda.atomic_cas(cell.data, compares[lane], replacements[lane])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = cell[0]


@T.prim_func
def cuda_atomic_cas_uint32x4(
    cell: T.Buffer((1,), "uint32x4"),
    compare: T.Buffer((1,), "uint32x4"),
    replacement: T.Buffer((1,), "uint32x4"),
    old_value: T.Buffer((1,), "uint32x4"),
    final_value: T.Buffer((1,), "uint32x4"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old_value[0] = T.cuda.atomic_cas(cell.data, compare[0], replacement[0])
    T.cuda.warp_sync()
    if lane == 0:
        final_value[0] = cell[0]


@T.prim_func
def ptx_relaxed_type_atomics(
    bit_counter: T.Buffer((1,), "uint32"),
    signed_counter: T.Buffer((1,), "uint32"),
    old_values: T.Buffer((2,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.atom.relaxed.gpu.global_.or_.b32(
            old_values[0],
            T.address_of(bit_counter[0]),
            T.cuda.uint_as_float(T.uint32(0x0F0F0F0F)),
        )
        old_signed: T.int32
        T.ptx.atom.relaxed.gpu.global_.min.s32(
            old_signed, T.address_of(signed_counter[0]), T.int32(1)
        )
        old_values[1] = T.reinterpret("uint32", old_signed)


@T.prim_func
def st_bulk_default_weak(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "uint32", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(lane + 1, "uint32")
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.st_bulk.shared__cta(shared.ptr_to([0]), T.uint64(8))
    T.cuda.warp_sync()
    if lane < 4:
        output[lane] = shared[lane]


@T.prim_func
def legacy_global_acquire(source: T.Buffer((32,), "uint64"), output: T.Buffer((32,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    local = T.alloc_buffer((1,), "uint64", scope="local")
    T.ptx.ld.acquire.gpu.global_.u64(local[0], T.address_of(source[lane]))
    output[lane] = local[0]


@T.prim_func
def guarded_ldg32(source: T.Buffer((32,), "float32"), output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    local = T.alloc_buffer((32,), "float32", scope="local")
    local[lane] = T.float32(-123.5)
    T.evaluate(T.s_tir.ldg32(local.data, lane < 16, source[lane], lane))
    output[lane] = local[lane]


@T.prim_func
def legacy_ldmatrix_x2_trans(output: T.Buffer((128,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((256,), "uint16", scope="shared")
    local = T.alloc_buffer((4,), "uint16", scope="local")
    for element in T.serial(8):
        shared[lane * 8 + element] = T.Cast("uint16", lane * 8 + element)
    T.cuda.warp_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(True, 2, ".b16", local.data, 0, shared.data, lane * 8, dtype="uint16")
    )
    for element in T.serial(4):
        output[lane * 4 + element] = local[element]


@T.prim_func
def legacy_ldmatrix_i8_transpose_fallback(output: T.Buffer((512,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1024,), "uint8", scope="shared")
    local = T.alloc_buffer((16,), "uint8", scope="local")
    for element in T.serial(32):
        shared[lane * 32 + element] = T.Cast("uint8", lane * 32 + element)
    T.cuda.warp_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(True, 4, ".b16", local.data, 0, shared.data, 32, dtype="uint8")
    )
    for element in T.serial(16):
        output[lane * 16 + element] = local[element]


@T.prim_func
def legacy_ldmatrix_i8_transpose_two_warps(output: T.Buffer((1024,), "uint8")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    thread = warp * 32 + lane
    shared = T.alloc_buffer((1024,), "uint8", scope="shared")
    local = T.alloc_buffer((16,), "uint8", scope="local")
    for element in T.serial(16):
        shared[thread * 16 + element] = T.Cast("uint8", thread * 16 + element)
    T.cuda.cta_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(True, 4, ".b16", local.data, 0, shared.data, 32, dtype="uint8")
    )
    for element in T.serial(16):
        output[thread * 16 + element] = local[element]


@T.prim_func
def legacy_ldmatrix_i8_x4(output: T.Buffer((512,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    local = T.alloc_buffer((16,), "uint8", scope="local")
    for element in T.serial(16):
        shared[lane * 16 + element] = T.Cast("uint8", lane * 16 + element)
    T.cuda.warp_sync()
    T.evaluate(
        T.ptx_legacy.ldmatrix(
            False, 4, ".b16", local.data, 0, shared.data, lane * 16, dtype="uint8"
        )
    )
    for element in T.serial(16):
        output[lane * 16 + element] = local[element]


@T.prim_func
def bulk_g2s_cta(source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](
            T.address_of(shared[0]),
            T.address_of(source[0]),
            T.cast(16, "uint32"),
            T.address_of(barrier[0]),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = shared[lane]


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
def raw_tma_reduce_add_bfloat16(source: T.Buffer((8,), "bfloat16"), output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "bfloat16", scope="shared")
    if lane < 8:
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
def raw_tma_prefetch(input_map: T.TensorMap(), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile"](T.address_of(input_map), 0, 0)
        output[0] = 1


@T.prim_func
def raw_bulk_prefetch(source: T.Buffer((32,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["cp.async.bulk.prefetch.L2.global"](T.address_of(source[0]), T.uint32(128))
        output[0] = source[0]


@T.prim_func
def raw_prefetchu(source: T.Buffer((32,), "int32"), output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["prefetchu.L1"](T.address_of(source[0]))
        output[0] = source[0]


def _tensor_map(
    array: np.ndarray,
    *,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
) -> np.ndarray:
    return numsim.TensorMap(
        base=array,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
    ).numpy()


def test_cp_async_public_sizes_predicate_and_fill_modes(tmp_path):
    source4 = np.arange(128, dtype=np.uint8) ^ np.uint8(0x5A)
    module4 = numsim.transpile(cp_async_plain_4, cache_dir=tmp_path / "plain")
    result4 = numsim.Engine().run(module4, {"source": source4, "output": np.zeros_like(source4)})
    np.testing.assert_array_equal(result4.outputs["output"], source4)

    source16 = np.arange(512, dtype=np.uint8) ^ np.uint8(0xC7)
    module16 = numsim.transpile(cp_async_cache_hint_16, cache_dir=tmp_path / "cache_hint")
    result16 = numsim.Engine().run(
        module16, {"source": source16, "output": np.zeros_like(source16)}
    )
    np.testing.assert_array_equal(result16.outputs["output"], source16)

    source8 = np.arange(256, dtype=np.uint8) ^ np.uint8(0x3C)
    module8 = numsim.transpile(cp_async_predicate_skip_8, cache_dir=tmp_path / "predicate")
    result8 = numsim.Engine().run(module8, {"source": source8, "output": np.zeros_like(source8)})
    expected8 = source8.copy()
    expected8[128:] = np.uint8(0xA5)
    np.testing.assert_array_equal(result8.outputs["output"], expected8)


def test_cp_async_ignore_src_skips_oob_reads_and_zero_fills(tmp_path):
    source8 = np.arange(128, dtype=np.uint8) ^ np.uint8(0x6C)
    module8 = numsim.transpile(cp_async_ca_ignore_src_zero_fill_8, cache_dir=tmp_path / "ca")
    result8 = numsim.Engine().run(
        module8,
        {"source": source8, "output": np.full(256, 0xFF, dtype=np.uint8)},
    )
    np.testing.assert_array_equal(
        result8.outputs["output"], np.concatenate((source8, np.zeros(128, dtype=np.uint8)))
    )

    source16 = np.arange(256, dtype=np.uint8) ^ np.uint8(0x93)
    module16 = numsim.transpile(cp_async_cg_ignore_src_zero_fill_16, cache_dir=tmp_path / "cg")
    result16 = numsim.Engine().run(
        module16,
        {"source": source16, "output": np.full(512, 0xFF, dtype=np.uint8)},
    )
    np.testing.assert_array_equal(
        result16.outputs["output"], np.concatenate((source16, np.zeros(256, dtype=np.uint8)))
    )


def test_cp_async_preserves_bound_raw_shared_base_and_lane_offset(tmp_path):
    source = np.arange(128, dtype=np.uint8) ^ np.uint8(0xA6)
    module = numsim.transpile(cp_async_bound_raw_shared_offset, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros_like(source)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_obsolete_cp_async_bulk_wrapper_is_rejected_during_parsing():
    with pytest.raises(
        tvm.error.DiagnosticError,
        match=r"'cp_async' is not a ptx instruction",
    ):
        tvm.script.from_source(_OBSOLETE_CP_ASYNC_BULK_SOURCE, {"T": T})


def test_cuda_float_atomic_add_and_default_st_bulk_forms(tmp_path):
    atomic = numsim.transpile(cuda_atomic_add_float32, cache_dir=tmp_path / "atomic")
    atomic_result = numsim.Engine().run(
        atomic,
        {
            "counter": np.zeros(1, dtype=np.float32),
            "old_values": np.zeros(4, dtype=np.float32),
            "final_value": np.zeros(1, dtype=np.float32),
        },
    )
    np.testing.assert_array_equal(
        np.sort(atomic_result.outputs["old_values"]),
        np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
    )
    np.testing.assert_array_equal(atomic_result.outputs["final_value"], [2.0])

    st_bulk = numsim.transpile(st_bulk_default_weak, cache_dir=tmp_path / "st_bulk")
    st_bulk_result = numsim.Engine().run(st_bulk, {"output": np.zeros(4, dtype=np.uint32)})
    np.testing.assert_array_equal(st_bulk_result.outputs["output"], [0, 0, 3, 4])


@pytest.mark.parametrize(
    ("kernel", "dtype"),
    [(cuda_atomic_add_float16, np.float16), (cuda_atomic_add_bfloat16, ml_dtypes.bfloat16)],
)
def test_cuda_16bit_float_atomic_add_uses_16bit_storage(kernel, dtype, tmp_path):
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "counter": np.asarray([1.0], dtype=dtype),
            "old_values": np.zeros(4, dtype=dtype),
            "final_value": np.zeros(1, dtype=dtype),
        },
    )
    np.testing.assert_array_equal(
        result.outputs["old_values"], np.asarray([1.0, 1.5, 2.0, 2.5], dtype=dtype)
    )
    np.testing.assert_array_equal(result.outputs["final_value"], np.asarray([3.0], dtype=dtype))


def _float32x2_binding(
    values,
):
    packed = np.asarray(values, dtype=np.float32).reshape(-1, 2).copy().view(np.uint64).reshape(-1)
    return packed


def _float16x2_binding(
    values,
):
    packed = np.asarray(values, dtype=np.float16).reshape(-1, 2).copy().view(np.uint32).reshape(-1)
    return packed


def _bfloat16x2_binding(
    values,
):
    packed = (
        np.asarray(values, dtype=ml_dtypes.bfloat16)
        .reshape(-1, 2)
        .copy()
        .view(np.uint32)
        .reshape(-1)
    )
    return packed


def _float32x4_binding(
    values,
):
    packed = (
        np.asarray(values, dtype=np.float32).reshape(-1, 4).copy().view(np.dtype("V16")).reshape(-1)
    )
    return packed


def _float32x4_bits_binding(
    values,
):
    packed = (
        np.asarray(values, dtype=np.uint32).reshape(-1, 4).copy().view(np.dtype("V16")).reshape(-1)
    )
    return packed


def _uint64x2_binding(
    values,
):
    packed = (
        np.asarray(values, dtype=np.uint64).reshape(-1, 2).copy().view(np.dtype("V16")).reshape(-1)
    )
    return packed


def _uint32x4_binding(
    values,
):
    packed = (
        np.asarray(values, dtype=np.uint32).reshape(-1, 4).copy().view(np.dtype("V16")).reshape(-1)
    )
    return packed


def _unpack_float32x2(array):
    return np.asarray(array).view(np.float32).reshape(-1, 2)


def _unpack_float16x2(array):
    return np.asarray(array).view(np.float16).reshape(-1, 2)


def _unpack_bfloat16x2(array):
    return np.asarray(array).view(ml_dtypes.bfloat16).reshape(-1, 2)


def _unpack_float32x4(array):
    return np.asarray(array).view(np.float32).reshape(-1, 4)


def _unpack_float32x4_bits(array):
    return np.asarray(array).view(np.uint32).reshape(-1, 4)


def _unpack_uint64x2(array):
    return np.asarray(array).view(np.uint64).reshape(-1, 2)


def _unpack_uint32x4(array):
    return np.asarray(array).view(np.uint32).reshape(-1, 4)


def test_cuda_float32x2_atomic_add_updates_components_and_returns_old_pairs(tmp_path):
    module = numsim.transpile(cuda_atomic_add_float32x2, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "counter": _float32x2_binding(
                [[0.5, -1.0]],
            ),
            "increments": _float32x2_binding(
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            ),
            "old_values": _float32x2_binding(
                np.zeros((4, 2)),
            ),
            "final_value": _float32x2_binding(
                np.zeros((1, 2)),
            ),
        },
    )

    np.testing.assert_array_equal(
        _unpack_float32x2(result.outputs["old_values"]),
        np.asarray([[0.5, -1.0], [1.5, 9.0], [3.5, 29.0], [6.5, 59.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _unpack_float32x2(result.outputs["final_value"]),
        np.asarray([[10.5, 99.0]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("kernel", "binding", "unpack", "dtype"),
    [
        (
            cuda_atomic_add_float16x2,
            _float16x2_binding,
            _unpack_float16x2,
            np.float16,
        ),
        (
            cuda_atomic_add_bfloat16x2,
            _bfloat16x2_binding,
            _unpack_bfloat16x2,
            ml_dtypes.bfloat16,
        ),
    ],
)
def test_cuda_packed_16bit_vector_atomic_add_is_componentwise(
    kernel, binding, unpack, dtype, tmp_path
):
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "counter": binding(
                [[0.5, 1.0]],
            ),
            "increments": binding(
                [[1.0, 2.0], [3.0, 4.0]],
            ),
            "old_values": binding(
                np.zeros((2, 2)),
            ),
            "final_value": binding(
                np.zeros((1, 2)),
            ),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(
            unpack(result.outputs["old_values"]),
            np.asarray([[0.5, 1.0], [1.5, 3.0]], dtype=dtype),
        )
        np.testing.assert_array_equal(
            unpack(result.outputs["final_value"]), np.asarray([[4.5, 7.0]], dtype=dtype)
        )

    check()


def test_cuda_float16x2_atomic_add_supports_shared_memory(tmp_path):
    module = numsim.transpile(cuda_atomic_add_float16x2_shared, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "initial": _float16x2_binding(
                [[0.5, 1.0]],
            ),
            "increments": _float16x2_binding(
                [[1.0, 2.0], [3.0, 4.0]],
            ),
            "old_values": _float16x2_binding(
                np.zeros((2, 2)),
            ),
            "final_value": _float16x2_binding(
                np.zeros((1, 2)),
            ),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(
            _unpack_float16x2(result.outputs["old_values"]),
            np.asarray([[0.5, 1.0], [1.5, 3.0]], dtype=np.float16),
        )
        np.testing.assert_array_equal(
            _unpack_float16x2(result.outputs["final_value"]),
            np.asarray([[4.5, 7.0]], dtype=np.float16),
        )

    check()


def test_ptx_red_half_bfloat16x2_reduces_both_components(tmp_path):
    initial = np.asarray([[0.5, -1.0]], dtype=ml_dtypes.bfloat16)
    increments = np.asarray(
        [[1.0, 2.0], [-0.5, 1.0], [2.0, -1.0], [0.5, 3.0]],
        dtype=ml_dtypes.bfloat16,
    )
    module = numsim.transpile(ptx_red_half_bfloat16x2, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "counter": initial.copy().view(np.uint32).reshape(-1),
            "increments": increments.copy().view(np.uint32).reshape(-1),
        },
    )

    np.testing.assert_array_equal(
        _unpack_bfloat16x2(result.outputs["counter"]),
        np.asarray([[3.5, 4.0]], dtype=ml_dtypes.bfloat16),
    )


def test_ptx_predicated_red_f32_skips_inactive_lanes(tmp_path):
    module = numsim.transpile(ptx_predicated_red_f32, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"output": np.zeros((32,), dtype=np.float32)},
    )

    expected = np.concatenate(
        (np.ones((16,), dtype=np.float32), np.zeros((16,), dtype=np.float32))
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cuda_float32x4_atomic_add_is_componentwise(tmp_path):
    module = numsim.transpile(cuda_atomic_add_float32x4, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "counter": _float32x4_binding(
                [[0.5, 1.0, 1.5, 2.0]],
            ),
            "increments": _float32x4_binding(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            ),
            "old_values": _float32x4_binding(
                np.zeros((2, 4)),
            ),
            "final_value": _float32x4_binding(
                np.zeros((1, 4)),
            ),
        },
    )
    np.testing.assert_array_equal(
        _unpack_float32x4(result.outputs["old_values"]),
        np.asarray([[0.5, 1.0, 1.5, 2.0], [1.5, 3.0, 4.5, 6.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _unpack_float32x4(result.outputs["final_value"]),
        np.asarray([[6.5, 9.0, 11.5, 14.0]], dtype=np.float32),
    )


def test_ptx_vector_f32_atomic_add_uses_scalar_storage_and_predicate(tmp_path):
    initial = np.asarray([0.5, 1.0, 1.5, 2.0, 10.0, 20.0], dtype=np.float32)
    increments = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    module = numsim.transpile(ptx_atomic_add_f32_vectors, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "counter": initial.copy(),
            "increments": increments,
            "old_values": np.full(6, np.nan, dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(result.outputs["old_values"], initial)
    np.testing.assert_array_equal(result.outputs["counter"], initial + increments)


def test_cuda_uint64x2_atomic_cas_compares_and_replaces_one_128bit_value(tmp_path):
    module = numsim.transpile(cuda_atomic_cas_uint64x2, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "cell": _uint64x2_binding(
                [[7, 9]],
            ),
            "compares": _uint64x2_binding(
                [[7, 9], [11, 999], [11, 13]],
            ),
            "replacements": _uint64x2_binding(
                [[11, 13], [17, 19], [23, 29]],
            ),
            "old_values": _uint64x2_binding(
                np.zeros((3, 2), dtype=np.uint64),
            ),
            "final_value": _uint64x2_binding(
                np.zeros((1, 2), dtype=np.uint64),
            ),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(
            _unpack_uint64x2(result.outputs["old_values"]),
            np.asarray([[7, 9], [11, 13], [11, 13]], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            _unpack_uint64x2(result.outputs["final_value"]),
            np.asarray([[23, 29]], dtype=np.uint64),
        )

    check()


def test_cuda_128bit_atomic_cas_compares_float_vectors_by_bits(tmp_path):
    module = numsim.transpile(cuda_atomic_cas_float32x4, cache_dir=tmp_path)
    initial = np.asarray([[0x80000000, 0x7FC12345, 0x3F800000, 0x40000000]], dtype=np.uint32)
    exact_compare = initial.copy()
    signed_zero_mismatch = initial.copy()
    signed_zero_mismatch[0, 0] = 0
    replacement = np.asarray([[1, 2, 3, 4], [11, 13, 17, 19]], dtype=np.uint32)
    result = numsim.Engine().run(
        module,
        {
            "cell": _float32x4_bits_binding(
                initial,
            ),
            "compares": _float32x4_bits_binding(
                np.concatenate([signed_zero_mismatch, exact_compare]),
            ),
            "replacements": _float32x4_bits_binding(
                replacement,
            ),
            "old_values": _float32x4_bits_binding(
                np.zeros((2, 4), dtype=np.uint32),
            ),
            "final_value": _float32x4_bits_binding(
                np.zeros((1, 4), dtype=np.uint32),
            ),
        },
    )

    np.testing.assert_array_equal(
        _unpack_float32x4_bits(result.outputs["old_values"]), np.concatenate([initial, initial])
    )
    np.testing.assert_array_equal(
        _unpack_float32x4_bits(result.outputs["final_value"]), replacement[1:2]
    )


def test_cuda_128bit_atomic_cas_uses_generic_total_width_vector_abi(tmp_path):
    module = numsim.transpile(cuda_atomic_cas_uint32x4, cache_dir=tmp_path)
    initial = np.asarray([[2, 3, 5, 7]], dtype=np.uint32)
    replacement = np.asarray([[11, 13, 17, 19]], dtype=np.uint32)
    result = numsim.Engine().run(
        module,
        {
            "cell": _uint32x4_binding(
                initial,
            ),
            "compare": _uint32x4_binding(
                initial,
            ),
            "replacement": _uint32x4_binding(
                replacement,
            ),
            "old_value": _uint32x4_binding(
                np.zeros((1, 4), dtype=np.uint32),
            ),
            "final_value": _uint32x4_binding(
                np.zeros((1, 4), dtype=np.uint32),
            ),
        },
    )
    np.testing.assert_array_equal(_unpack_uint32x4(result.outputs["old_value"]), initial)
    np.testing.assert_array_equal(_unpack_uint32x4(result.outputs["final_value"]), replacement)


def test_ptx_atomics_apply_instruction_type_after_relaxed_operand_conformance(tmp_path):
    bit_initial = np.array([0xF0000000], dtype=np.uint32)
    signed_initial = np.array([0xFFFFFFFF], dtype=np.uint32)
    module = numsim.transpile(ptx_relaxed_type_atomics, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "bit_counter": bit_initial,
            "signed_counter": signed_initial,
            "old_values": np.zeros(2, dtype=np.uint32),
        },
    )
    np.testing.assert_array_equal(
        result.outputs["old_values"], np.array([4026531840, 4294967295], dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        result.outputs["bit_counter"], np.array([0xFF0F0F0F], dtype=np.uint32)
    )
    np.testing.assert_array_equal(result.outputs["signed_counter"], signed_initial)


def test_legacy_acquire_and_guarded_ldg32_write_local_lvalues(tmp_path):
    source_u64 = np.arange(32, dtype=np.uint64) * np.uint64(0x1_0000_0001)
    acquire = numsim.transpile(legacy_global_acquire, cache_dir=tmp_path / "acquire")
    acquire_result = numsim.Engine().run(
        acquire, {"source": source_u64, "output": np.zeros_like(source_u64)}
    )
    np.testing.assert_array_equal(acquire_result.outputs["output"], source_u64)

    source_f32 = np.arange(32, dtype=np.float32) + np.float32(0.25)
    ldg = numsim.transpile(guarded_ldg32, cache_dir=tmp_path / "ldg32")
    ldg_result = numsim.Engine().run(
        ldg, {"source": source_f32, "output": np.zeros_like(source_f32)}
    )
    expected = source_f32.copy()
    expected[16:] = np.float32(0)
    np.testing.assert_array_equal(ldg_result.outputs["output"], expected)


def test_legacy_ldmatrix_transpose_preserves_b16_fragment_abi(tmp_path):
    module = numsim.transpile(legacy_ldmatrix_x2_trans, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(128, dtype=np.uint16)})
    expected = np.empty((32, 4), dtype=np.uint16)
    for matrix in range(2):
        for lane in range(32):
            row = lane // 4
            pair = lane % 4
            low = (matrix * 64) + (pair * 2) * 8 + row
            high = low + 8
            expected[lane, matrix * 2] = np.uint16(low)
            expected[lane, matrix * 2 + 1] = np.uint16(high)
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(-1))


def test_legacy_ldmatrix_8bit_transpose_matches_tirx_manual_gather(tmp_path):
    module = numsim.transpile(legacy_ldmatrix_i8_transpose_fallback, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(512, dtype=np.uint8)})
    expected = np.empty((32, 16), dtype=np.uint8)
    stride = 32
    for lane in range(32):
        for element in range(16):
            source = (
                ((element % 8) // 4) * stride * 16
                + (lane % 4) * 4 * stride
                + (element % 4) * stride
                + lane // 4
                + (element // 8) * 8
            )
            expected[lane, element] = np.uint8(source & 0xFF)
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(-1))


def test_legacy_ldmatrix_8bit_transpose_uses_full_thread_index_across_warps(tmp_path):
    module = numsim.transpile(legacy_ldmatrix_i8_transpose_two_warps, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1024, dtype=np.uint8)})
    expected = np.empty((64, 16), dtype=np.uint8)
    stride = 32
    for thread in range(64):
        for element in range(16):
            source = (
                ((element % 8) // 4) * stride * 16
                + (thread % 4) * 4 * stride
                + (element % 4) * stride
                + thread // 4
                + (element // 8) * 8
            )
            expected[thread, element] = np.uint8(source & 0xFF)
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(-1))


def test_legacy_ldmatrix_8bit_nontranspose_keeps_b16_fragment_abi(tmp_path):
    module = numsim.transpile(legacy_ldmatrix_i8_x4, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(512, dtype=np.uint8)})
    expected = np.empty((32, 16), dtype=np.uint8)
    for lane in range(32):
        row = lane // 4
        fragment = lane % 4
        for matrix in range(4):
            source = (matrix * 8 + row) * 16 + fragment * 4
            expected[lane, matrix * 4 : matrix * 4 + 4] = np.arange(
                source, source + 4, dtype=np.uint16
            ).astype(np.uint8)
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(-1))


def test_bulk_g2s_cta_copies_exact_physical_bytes(tmp_path):
    source = np.arange(16, dtype=np.uint8) ^ np.uint8(0xC3)
    module = numsim.transpile(bulk_g2s_cta, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros_like(source)})
    np.testing.assert_array_equal(result.outputs["output"], source)


def test_bulk_g2s_cta_has_exact_racecheck_payload_accesses():
    source = np.arange(16, dtype=np.uint8) ^ np.uint8(0xC3)
    report = racecheck(
        bulk_g2s_cta,
        inputs={"source": source, "output": np.zeros_like(source)},
    )

    assert report.verdict == "clean", report.format()


def test_raw_tma_gather4_bar_address_and_prefetch(tmp_path):
    source = np.arange(16, dtype=np.float32).reshape(4, 4) + np.float32(0.5)
    input_map = _tensor_map(
        source,
        global_shape=(4, 4),
        global_strides=(16,),
        box_shape=(4, 1),
    )
    gather = numsim.transpile(raw_tma_gather4_bar_address, cache_dir=tmp_path / "gather")
    result = numsim.Engine().run(gather, {"input_map": input_map, "output": np.zeros_like(source)})
    np.testing.assert_array_equal(result.outputs["output"], source)

    prefetch = numsim.transpile(raw_tma_prefetch, cache_dir=tmp_path / "prefetch")
    prefetch_result = numsim.Engine().run(
        prefetch, {"input_map": input_map, "output": np.zeros(1, dtype=np.int32)}
    )
    np.testing.assert_array_equal(prefetch_result.outputs["output"], [1])


def test_raw_tma_reduce_uses_retained_logical_dtype(tmp_path):
    source = np.array([1.5, -2.0, 4.25, 3.0], dtype=np.float32)
    destination = np.array([10.0, 20.0, -1.0, 8.0], dtype=np.float32)
    initial_destination = destination.copy()
    output_map = _tensor_map(
        destination,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    prepared = prepare_bindings(
        {"output_map": output_map}, expected_tensor_map_names={"output_map"}
    )
    assert prepared.tensor_map_outputs["output_map"].dtype == "float32"

    module = numsim.transpile(raw_tma_reduce_add, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output_map": output_map})
    np.testing.assert_array_equal(result.outputs["output_map"], initial_destination + source)


def test_raw_tma_reduce_bfloat16_uses_plain_logical_dtype_array(tmp_path):
    source = np.asarray([1.5, -2.0, 4.25, 3.0, -0.5, 0.25, 16.0, -8.0], dtype=ml_dtypes.bfloat16)
    destination = np.asarray(
        [10.0, 20.0, -1.0, 8.0, 2.0, -4.0, 0.5, 32.0], dtype=ml_dtypes.bfloat16
    )
    initial_destination = destination.copy()
    output_map = _tensor_map(
        destination,
        global_shape=(8,),
        global_strides=(),
        box_shape=(8,),
    )

    prepared = prepare_bindings(
        {"output_map": output_map}, expected_tensor_map_names={"output_map"}
    )
    assert prepared.tensor_map_outputs["output_map"].dtype == "bfloat16"

    module = numsim.transpile(raw_tma_reduce_add_bfloat16, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output_map": output_map})
    expected = np.asarray(
        initial_destination.astype(np.float32) + source.astype(np.float32), dtype=ml_dtypes.bfloat16
    )
    np.testing.assert_array_equal(result.outputs["output_map"], expected.view(np.uint16))


def test_memory_family_public_ops_have_closed_world_registration():
    assert analyze(cp_async_plain_4).unsupported == ()
    assert analyze(cp_async_cache_hint_16).unsupported == ()
    assert analyze(cp_async_ca_ignore_src_zero_fill_8).unsupported == ()
    assert analyze(cp_async_cg_ignore_src_zero_fill_16).unsupported == ()
    assert analyze(cuda_atomic_add_float32).unsupported == ()
    assert analyze(cuda_atomic_add_float32x2).unsupported == ()
    assert analyze(cuda_atomic_add_float16x2).unsupported == ()
    assert analyze(cuda_atomic_add_float16x2_shared).unsupported == ()
    assert analyze(cuda_atomic_add_bfloat16x2).unsupported == ()
    assert analyze(cuda_atomic_add_float32x4).unsupported == ()
    assert analyze(ptx_atomic_add_f32_vectors).unsupported == ()
    assert analyze(cuda_atomic_cas_uint64x2).unsupported == ()
    assert analyze(cuda_atomic_cas_float32x4).unsupported == ()
    assert analyze(cuda_atomic_cas_uint32x4).unsupported == ()
    assert analyze(st_bulk_default_weak).unsupported == ()
    assert analyze(legacy_global_acquire).unsupported == ()
    assert analyze(guarded_ldg32).unsupported == ()
    assert analyze(legacy_ldmatrix_x2_trans).unsupported == ()
    assert analyze(bulk_g2s_cta).unsupported == ()
    assert analyze(raw_tma_gather4_bar_address).unsupported == ()
    assert analyze(raw_tma_reduce_add).unsupported == ()
    assert analyze(raw_tma_reduce_add_bfloat16).unsupported == ()
    assert analyze(raw_tma_prefetch).unsupported == ()
    assert analyze(raw_bulk_prefetch).unsupported == ()
    assert analyze(raw_prefetchu).unsupported == ()


@pytest.mark.parametrize("op_name", ["atom", "red"])
def test_scalar_atomic_add_has_no_s64_form_while_min_and_max_do(op_name):
    """PTX ISA 9.7.14.5 Table 35 lists `.s64` under `.min, .max` only.

    `atom.add.s64` / `red.add.s64` are therefore not PTX forms, and the missing
    `Add` x `I64` engine row is the ISA rather than a NumSim coverage gap. The
    `min` case is the positive control that the s64 carrier itself resolves.
    """

    pointer = tirx.Var("pointer", "handle")
    wrapper = getattr(T.ptx, op_name)
    destination = tirx.decl_buffer((1,), "int64", name="destination")

    def call(operation):
        operands = (
            (destination[0], pointer, T.int64(1))
            if op_name == "atom"
            else (
                pointer,
                T.int64(1),
            )
        )
        return operation(*operands)

    with pytest.raises(ValueError, match=r"\.add requires"):
        call(wrapper.global_.add.s64)

    destination_operand = "destination, " if op_name == "atom" else ""
    kernel = parse_kernel(
        f"""
@T.prim_func
def kernel(counter: T.Buffer((1,), "int64")):
    T.device_entry()
    destination = T.local_scalar("int64")
    T.ptx.{op_name}.global_.min.s64({destination_operand}T.address_of(counter[0]), T.int64(1))
"""
    )
    (emitted,) = emitted_calls(kernel, f"tirx.ptx.{op_name}")
    assert "v2::mem::variant::Minimum" in emitted.generics
    assert "v2::reg::variant::I64" in emitted.generics
