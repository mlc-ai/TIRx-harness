from __future__ import annotations

import numpy as np
import pytest
import tvm

from tirx_harness import numsim
from tirx_harness.numsim.api import ExecutionSubset
from tirx_harness.numsim.transpiler import suspend_scaffold
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T


@T.prim_func
def integer_address_same_backing(
    source: T.Buffer((33,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address_bits: T.uint64 = T.reinterpret("uint64", source.ptr_to([0]))
    if lane < 16:
        address_bits = T.reinterpret("uint64", source.ptr_to([lane]))
    else:
        address_bits = T.reinterpret("uint64", source.ptr_to([31 - lane]))
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", address_bits))


@T.prim_func
def integer_address_different_backing(
    left: T.Buffer((32,), "uint32"),
    right: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address_bits: T.uint64 = T.reinterpret("uint64", left.ptr_to([lane]))
    if lane >= 16:
        address_bits = T.reinterpret("uint64", right.ptr_to([lane]))
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", address_bits))


@T.prim_func
def integer_address_byte_offset(
    source: T.Buffer((33,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    address_bits: T.uint64 = T.reinterpret("uint64", source.ptr_to([lane]))
    shifted = T.reinterpret("handle", address_bits + T.uint64(4))
    typed_shifted = T.ptr_byte_offset(shifted, T.uint32(0), "uint32")
    T.ptx.ld.global_.u32(output[lane], typed_shifted)


@T.prim_func
def unused_out_of_bounds_shared_view(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    storage = T.alloc_buffer((4,), "uint8", scope="shared")
    alias_data: T.let[
        T.Var(
            name="unused_out_of_bounds_shared_data",
            ty=PointerType(PrimType("uint32"), "shared"),
        )
    ] = T.reinterpret(PointerType(PrimType("uint32"), "shared"), storage.ptr_to([8]))
    _alias = T.decl_buffer((1,), "uint32", data=alias_data, scope="shared")
    if lane == 0:
        output[0] = T.uint32(17)


@T.prim_func
def accessed_out_of_bounds_shared_view(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    storage = T.alloc_buffer((4,), "uint8", scope="shared")
    alias_data: T.let[
        T.Var(
            name="accessed_out_of_bounds_shared_data",
            ty=PointerType(PrimType("uint32"), "shared"),
        )
    ] = T.reinterpret(PointerType(PrimType("uint32"), "shared"), storage.ptr_to([8]))
    alias = T.decl_buffer((1,), "uint32", data=alias_data, scope="shared")
    if lane == 0:
        output[0] = alias[0]


@T.prim_func
def byte_storage_integer_address_atomic_and_load(
    storage: T.Buffer((4,), "uint8"), output: T.Buffer((2,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        address_bits: T.uint64 = T.reinterpret("uint64", storage.ptr_to([0]))
        shifted = T.reinterpret("handle", address_bits + T.uint64(0))
        T.ptx.atom.release.gpu.global_.add.u32(
            output[0],
            shifted,
            T.uint32(5),
        )
        T.ptx.ld.global_.u32(output[1], shifted)


@T.prim_func
def non_byte_integer_address_atomic_uses_instruction_width(
    storage: T.Buffer((2,), "uint16"), output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        address_bits: T.uint64 = T.reinterpret("uint64", storage.ptr_to([0]))
        shifted = T.reinterpret("handle", address_bits + T.uint64(0))
        T.ptx.atom.release.gpu.global_.add.u32(
            output[0],
            shifted,
            T.uint32(1),
        )


@T.prim_func
def mega_atomic_and_reduction_ops(
    atom_u32: T.Buffer((1,), "uint32"),
    atom_u64: T.Buffer((1,), "uint64"),
    red_i32: T.Buffer((1,), "int32"),
    red_u32: T.Buffer((1,), "uint32"),
    red_or_u64: T.Buffer((1,), "uint64"),
    old_shared: T.Buffer((32,), "int32"),
    old_u32: T.Buffer((32,), "uint32"),
    old_u64: T.Buffer((32,), "uint64"),
    final_shared: T.Buffer((1,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((1,), "int32", scope="shared")
    if lane == 0:
        shared[0] = 0
    T.cuda.warp_sync()
    old_shared[lane] = T.cuda.atomic_add(shared.ptr_to([0]), T.int32(1))
    T.ptx.atom.release.gpu.global_.add.u32(
        old_u32[lane],
        atom_u32.ptr_to([0]),
        T.uint32(1),
    )
    T.ptx.atom.global_.add.u64(old_u64[lane], atom_u64.ptr_to([0]), T.uint64(1))
    T.ptx.red.release.sys.global_.add.s32(
        red_i32.ptr_to([0]),
        T.int32(1),
    )
    T.ptx.red.gpu.global_.add.u32(red_u32.ptr_to([0]), T.uint32(1))
    T.ptx.red.release.gpu.global_.or_.b64(
        red_or_u64.ptr_to([0]),
        T.shift_left(T.uint64(1), T.cast(lane, "uint64")),
    )
    T.cuda.warp_sync()
    if lane == 0:
        final_shared[0] = shared[0]


@T.prim_func
def atomic_cache_hints_are_numerical_noops(
    atom_counter: T.Buffer((1,), "uint32"),
    red_counter: T.Buffer((1,), "uint32"),
    old_values: T.Buffer((3,), "uint32"),
    final_values: T.Buffer((2,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.atom.release.gpu.global_.add.u32(
            old_values[0],
            atom_counter.ptr_to([0]),
            T.uint32(1),
        )
        T.ptx["atom.release.gpu.global.add.L2::cache_hint.u32"](
            old_values[1],
            atom_counter.ptr_to([0]),
            T.uint32(1),
            T.uint64(0),
        )
        T.ptx["atom.release.gpu.global.add.L2::cache_hint.u32"](
            old_values[2],
            atom_counter.ptr_to([0]),
            T.uint32(1),
            T.uint64(1),
        )
        T.ptx.red.gpu.global_.add.u32(red_counter.ptr_to([0]), T.uint32(1))
        T.ptx["red.gpu.global.add.L2::cache_hint.u32"](
            red_counter.ptr_to([0]),
            T.uint32(1),
            T.uint64(0),
        )
        T.ptx["red.gpu.global.add.L2::cache_hint.u32"](
            red_counter.ptr_to([0]),
            T.uint32(1),
            T.uint64(1),
        )
        final_values[0] = atom_counter[0]
        final_values[1] = red_counter[0]


@T.prim_func
def global_i32_atomic_add_forms(
    cuda_counter: T.Buffer((1,), "int32"),
    ptx_counter: T.Buffer((1,), "int32"),
    old_cuda: T.Buffer((4,), "int32"),
    old_ptx: T.Buffer((4,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 4:
        old_cuda[lane] = T.cuda.atomic_add(cuda_counter.ptr_to([0]), T.int32(1))
        T.ptx.atom.release.gpu.global_.add.s32(
            old_ptx[lane],
            ptx_counter.ptr_to([0]),
            T.int32(1),
        )


@T.prim_func
def global_i32_atomic_cas(
    cell: T.Buffer((1,), "int32"),
    old_values: T.Buffer((2,), "int32"),
    final_value: T.Buffer((1,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old_values[0] = T.cuda.atomic_cas(cell.ptr_to([0]), T.int32(7), T.int32(11))
        old_values[1] = T.cuda.atomic_cas(cell.ptr_to([0]), T.int32(7), T.int32(13))
        final_value[0] = cell[0]


@T.prim_func
def mega_st_bulk_zero(output: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "uint32", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(lane + 1, "uint32")
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.st_bulk.weak.shared__cta(shared.ptr_to([0]), T.cast(T.int32(8), "uint64"))
    T.cuda.warp_sync()
    if lane < 4:
        output[lane] = shared[lane]


@T.prim_func
def lane_wise_st_bulk_zero(output: T.Buffer((64,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    shared[lane * 2] = T.uint32(lane + 1)
    shared[lane * 2 + 1] = T.uint32(lane + 101)
    T.cuda.warp_sync()
    T.ptx.st_bulk.shared__cta(shared.ptr_to([lane * 2]), T.uint64(8))
    T.cuda.warp_sync()
    output[lane * 2] = shared[lane * 2]
    output[lane * 2 + 1] = shared[lane * 2 + 1]


@T.prim_func
def lane_wise_bulk_s2g(source: T.Buffer((128,), "uint32"), destination: T.Buffer((128,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint32", scope="shared")
    base = lane * 4
    for index in T.serial(4):
        shared[base + index] = source[base + index]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
        destination.ptr_to([base]), shared.ptr_to([base]), T.cast(16, "uint32")
    )
    T.ptx.cp.async_.bulk.commit_group()
    T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def mega_bulk_roundtrip(source: T.Buffer((32,), "uint8"), destination: T.Buffer((32,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 32)
        T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            T.cast(32, "uint32"),
            barrier.ptr_to([0]),
            T.uint64(0x12F0000000000000),
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            T.cast(32, "uint32"),
            T.uint64(0x1000000000000000),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def mega_bulk_roundtrip_without_cache_hint(
    source: T.Buffer((32,), "uint8"), destination: T.Buffer((32,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 32)
        T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), source.ptr_to([0]), T.cast(32, "uint32"), barrier.ptr_to([0])
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
            destination.ptr_to([0]), shared.ptr_to([0]), T.cast(32, "uint32")
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def mega_bulk_roundtrip_alternate_cache_hints(
    source: T.Buffer((32,), "uint8"), destination: T.Buffer((32,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 32)
        T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            T.cast(32, "uint32"),
            barrier.ptr_to([0]),
            T.uint64(0x1000000000000000),
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            T.cast(32, "uint32"),
            T.uint64(0x12F0000000000000),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def mega_bulk_g2s_multicast(source: T.Buffer((16,), "uint8"), output: T.Buffer((2, 16), "uint8")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        T.ptx[
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
        ](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            T.cast(16, "uint32"),
            barrier.ptr_to([0]),
            T.cast(3, "uint16"),
        )
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if lane < 16:
        output[cta, lane] = shared[lane]


@T.prim_func
def mega_bulk_s2g_cp_mask(source: T.Buffer((16,), "uint8"), destination: T.Buffer((16,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    if lane < 16:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.cp_mask"](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            T.cast(16, "uint32"),
            T.uint64(0x1000000000000000),
            T.cast(0xAAAA, "uint16"),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def bulk_s2g_read_then_full_wait(
    source: T.Buffer((16,), "uint8"),
    destination: T.Buffer((16,), "uint8"),
    observed: T.Buffer((2,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    if lane == 0:
        for index in T.serial(16):
            shared[index] = source[index]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            T.cast(16, "uint32"),
            T.uint64(0x1000000000000000),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        observed[0] = destination[0]
        T.ptx.cp.async_.bulk.wait_group(0)
        observed[1] = destination[0]


@T.prim_func
def bulk_s2g_multiple_groups(
    source: T.Buffer((32,), "uint8"),
    destination: T.Buffer((32,), "uint8"),
    observed: T.Buffer((2,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    if lane == 0:
        for index in T.serial(16):
            shared[index] = source[index]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            T.cast(16, "uint32"),
            T.uint64(0x1000000000000000),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        for index in T.serial(16):
            shared[index] = source[index + 16]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"](
            destination.ptr_to([16]),
            shared.ptr_to([0]),
            T.cast(16, "uint32"),
            T.uint64(0x1000000000000000),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(1)
        observed[0] = destination[0]
        T.ptx.cp.async_.bulk.wait_group(0)
        observed[1] = destination[16]


@T.prim_func
def bulk_s2g_read_only_exit(
    source: T.Buffer((16,), "uint8"), destination: T.Buffer((16,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    if lane == 0:
        for index in T.serial(16):
            shared[index] = source[index]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            T.cast(16, "uint32"),
            T.uint64(0x1000000000000000),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def mega_cluster_arrive_wait(exchange: T.Buffer((2,), "int32"), output: T.Buffer((2,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        exchange[cta] = cta + 11
    T.ptx.barrier.cluster.arrive.release.aligned()
    T.ptx.barrier.cluster.wait.acquire.aligned()
    if lane == 0:
        output[cta] = exchange[1 - cta]


@T.prim_func
def mega_cluster_arrive_only_participant(
    exchange: T.Buffer((2,), "int32"), output: T.Buffer((1,), "int32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        exchange[cta] = cta + 21
    T.ptx.barrier.cluster.arrive.release.aligned()
    if cta == 0:
        T.ptx.barrier.cluster.wait.acquire.aligned()
        if lane == 0:
            output[0] = exchange[1]
    else:
        T.evaluate(0)


@T.prim_func
def mega_lane_private_mbarrier_init(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((32,), "uint64", scope="shared")
    T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane]), 1)
    if lane == 0:
        output[0] = 17


@T.prim_func
def invalid_partially_aliased_mbarrier_init(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barriers = T.alloc_buffer((16,), "uint64", scope="shared")
    T.ptx.mbarrier.init.shared.b64(barriers.ptr_to([lane // 2]), 1)
    if lane == 0:
        output[0] = 17


@T.prim_func
def global_acquire_poll_woken_by_atomic(
    signal: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    current = T.local_scalar("uint32")
    if warp == 0:
        if lane == 0:
            T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            while current != T.uint32(1):
                T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            output[0] = current
    elif lane == 0:
        T.ptx.atom.release.gpu.global_.add.u32(current, signal.ptr_to([0]), T.uint32(1))


@T.prim_func
def global_acquire_poll_woken_by_cross_cluster_atomic(
    signal: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    current = T.local_scalar("uint32")
    if cta == 0:
        if lane == 0:
            T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            while current != T.uint32(1):
                T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            output[0] = current
    elif lane == 0:
        T.ptx.atom.release.gpu.global_.add.u32(current, signal.ptr_to([0]), T.uint32(1))


@T.prim_func
def direct_global_acquire_poll_woken_by_same_cluster_atomic(
    done_counter: T.Buffer((1,), "int32"),
    work_total: T.Buffer((1,), "int32"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    old = T.local_scalar("int32")
    current = T.local_scalar("int32")
    if warp == 0:
        if lane == 0:
            T.ptx.ld.acquire.gpu.global_.s32(current, done_counter.ptr_to([0]))
            while current < work_total[0]:
                T.ptx.ld.acquire.gpu.global_.s32(current, done_counter.ptr_to([0]))
            output[0] = done_counter[0]
    elif lane == 0:
        T.ptx.atom.release.gpu.global_.add.s32(old, done_counter.ptr_to([0]), T.int32(1))


@T.prim_func
def global_volatile_poll_zero_init_first_reload(
    signal: T.Buffer((1,), "uint64"), output: T.Buffer((1,), "uint64")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    current = T.local_scalar("uint64")
    if lane == 0:
        current = T.uint64(0)
        while current != T.uint64(1):
            T.ptx.ld.volatile.global_.u64(current, signal.ptr_to([0]))
        output[0] = current


@T.prim_func
def global_acquire_poll_two_lane_ranges(
    signal: T.Buffer((2,), "uint32"), output: T.Buffer((2,), "uint32")
):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    current = T.local_scalar("uint32")
    if warp == 0:
        if lane < 2:
            T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([lane]))
            while current != T.uint32(1):
                T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([lane]))
            output[lane] = current
    elif warp == 1:
        if lane == 0:
            T.ptx.atom.release.gpu.global_.add.u32(current, signal.ptr_to([0]), T.uint32(1))
        T.ptx.bar.sync(T.uint32(0), T.uint32(64))
        if lane == 0:
            T.ptx.atom.release.gpu.global_.add.u32(current, signal.ptr_to([1]), T.uint32(1))
    else:
        T.ptx.bar.sync(T.uint32(0), T.uint32(64))


@T.prim_func
def non_polling_loop_with_extra_body_effect(
    signal: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    current = T.local_scalar("uint32")
    mirror = T.local_scalar("uint32")
    if lane == 0:
        T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
        while current != T.uint32(1):
            T.ptx.ld.acquire.gpu.global_.b32(current, signal.ptr_to([0]))
            mirror = current
        output[0] = mirror


@T.prim_func
def direct_global_poll_with_body_effect(
    signal: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    current = T.local_scalar("int32")
    if lane == 0:
        T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([0]))
        while current < T.int32(1):
            output[0] = output[0] + 1
            T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([0]))


@T.prim_func
def direct_global_poll_with_two_watched_loads(
    left: T.Buffer((1,), "int32"), right: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_current = T.local_scalar("int32")
    right_current = T.local_scalar("int32")
    if lane == 0:
        T.ptx.ld.acquire.gpu.global_.s32(left_current, left.ptr_to([0]))
        T.ptx.ld.acquire.gpu.global_.s32(right_current, right.ptr_to([0]))
        while left_current < right_current:
            T.ptx.ld.acquire.gpu.global_.s32(left_current, left.ptr_to([0]))
            T.ptx.ld.acquire.gpu.global_.s32(right_current, right.ptr_to([0]))


@T.prim_func
def plain_global_buffer_poll_two_lane_ranges(
    signal: T.Buffer((2,), "int32"), output: T.Buffer((2,), "int32")
):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    current = T.local_scalar("int32")
    old = T.local_scalar("int32")
    if warp == 0:
        if lane < 2:
            T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([lane]))
            while current < T.int32(1):
                T.ptx.ld.acquire.gpu.global_.s32(current, signal.ptr_to([lane]))
            T.ptx.ld.acquire.gpu.global_.s32(output[lane], signal.ptr_to([lane]))
    elif warp == 1:
        if lane == 0:
            T.ptx.atom.release.gpu.global_.add.s32(old, signal.ptr_to([0]), T.int32(1))
        T.ptx.bar.sync(T.uint32(0), T.uint32(64))
        if lane == 0:
            T.ptx.atom.release.gpu.global_.add.s32(old, signal.ptr_to([1]), T.int32(1))
    else:
        T.ptx.bar.sync(T.uint32(0), T.uint32(64))


@T.prim_func
def plain_global_buffer_poll_with_body_effect(
    signal: T.Buffer((1,), "int32"), output: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        while signal[0] < T.int32(1):
            output[0] = output[0] + T.int32(1)


@T.prim_func
def plain_global_buffer_poll_with_two_loads(
    left: T.Buffer((1,), "int32"), right: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        while left[0] < right[0]:
            T.evaluate(0)


_INVALID_ST_BULK_SIZE_SOURCE = """
@T.prim_func
def invalid():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((2,), "uint32", scope="shared")
    if lane == 0:
        T.ptx.st_bulk.shared__cta(shared.ptr_to([0]), T.float32(8))
"""

_INVALID_G2S_MASK_WITHOUT_MULTICAST_SOURCE = """
@T.prim_func
def invalid(source: T.Buffer((16,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]), source.ptr_to([0]), T.uint32(16),
            barrier.ptr_to([0]), T.uint16(1)
        )
"""

_INVALID_S2G_MASK_WITHOUT_MODIFIER_SOURCE = """
@T.prim_func
def invalid(destination: T.Buffer((16,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    if lane == 0:
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
            destination.ptr_to([0]), shared.ptr_to([0]),
            T.uint32(16), T.uint16(65535)
        )
"""


@T.prim_func
def cluster_barrier_default_and_release_forms(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.barrier.cluster.arrive()
    T.ptx.barrier.cluster.wait()
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait()
    if lane == 0:
        output[0] = 1


@T.prim_func
def cluster_barrier_unaligned_divergent_arrive(output: T.Buffer((1,), "int32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane < 16:
        T.ptx.barrier.cluster.arrive()
    else:
        T.ptx.barrier.cluster.arrive()
    T.ptx.barrier.cluster.wait()
    if lane == 0:
        output[0] = 2


def test_integer_addresses_select_lanes_from_the_same_backing(tmp_path):
    source = np.arange(33, dtype=np.uint32) * np.uint32(7) + np.uint32(3)
    output = np.zeros(32, dtype=np.uint32)

    module = numsim.transpile(integer_address_same_backing, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    indices = np.concatenate((np.arange(16), np.arange(15, -1, -1)))
    np.testing.assert_array_equal(result.outputs["output"], source[indices])


def test_partitioned_integer_address_flow_preserves_lane_values(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_SYNC_SPLIT_MIN_LINES", 0)

    source = np.arange(33, dtype=np.uint32) * np.uint32(7) + np.uint32(3)
    module = numsim.transpile(integer_address_same_backing, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros(32, dtype=np.uint32)},
    )

    indices = np.concatenate((np.arange(16), np.arange(15, -1, -1)))
    np.testing.assert_array_equal(result.outputs["output"], source[indices])


def test_integer_addresses_select_different_global_backings_per_lane(tmp_path):
    left = np.arange(32, dtype=np.uint32)
    right = np.arange(32, dtype=np.uint32) + np.uint32(1000)

    module = numsim.transpile(integer_address_different_backing, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"left": left, "right": right, "output": np.zeros(32, dtype=np.uint32)}
    )
    expected = np.concatenate((left[:16], right[16:]))

    def check() -> None:
        assert not np.shares_memory(left, right)
        np.testing.assert_array_equal(result.outputs["output"], expected)

    check()


def test_integer_address_arithmetic_uses_native_byte_offsets(tmp_path):
    source = np.arange(33, dtype=np.uint32) + np.uint32(17)
    output = np.zeros(32, dtype=np.uint32)

    module = numsim.transpile(integer_address_byte_offset, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source[1:])


def test_unused_out_of_bounds_shared_view_has_no_memory_effect(tmp_path):
    module = numsim.transpile(unused_out_of_bounds_shared_view, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.uint32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([17], dtype=np.uint32))


def test_accessed_out_of_bounds_shared_view_is_rejected(tmp_path):
    module = numsim.transpile(accessed_out_of_bounds_shared_view, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match="exceeds allocation"):
        numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.uint32)})


def test_integer_address_atomic_uses_instruction_width_on_byte_storage(tmp_path):
    module = numsim.transpile(byte_storage_integer_address_atomic_and_load, cache_dir=tmp_path)
    storage = np.array([7, 0, 0, 0], dtype=np.uint8)
    output = np.zeros(2, dtype=np.uint32)
    result = numsim.Engine().run(
        module, {"storage": storage, "output": output}, outputs=("storage", "output")
    )

    np.testing.assert_array_equal(
        result.outputs["storage"], np.array([12, 0, 0, 0], dtype=np.uint8)
    )
    np.testing.assert_array_equal(result.outputs["output"], np.array([7, 12], dtype=np.uint32))


def test_integer_address_does_not_retain_its_source_pointee_width(tmp_path):
    module = numsim.transpile(
        non_byte_integer_address_atomic_uses_instruction_width, cache_dir=tmp_path
    )
    result = numsim.Engine().run(
        module,
        {"storage": np.zeros(2, dtype=np.uint16), "output": np.zeros(1, dtype=np.uint32)},
        outputs=("storage", "output"),
    )

    np.testing.assert_array_equal(result.outputs["storage"], np.array([1, 0], dtype=np.uint16))
    np.testing.assert_array_equal(result.outputs["output"], np.zeros(1, dtype=np.uint32))


def test_mega_atomic_and_reduction_signatures_execute_exactly(tmp_path):
    bindings = {
        "atom_u32": np.zeros(1, dtype=np.uint32),
        "atom_u64": np.zeros(1, dtype=np.uint64),
        "red_i32": np.zeros(1, dtype=np.int32),
        "red_u32": np.zeros(1, dtype=np.uint32),
        "red_or_u64": np.zeros(1, dtype=np.uint64),
        "old_shared": np.zeros(32, dtype=np.int32),
        "old_u32": np.zeros(32, dtype=np.uint32),
        "old_u64": np.zeros(32, dtype=np.uint64),
        "final_shared": np.zeros(1, dtype=np.int32),
    }

    module = numsim.transpile(mega_atomic_and_reduction_ops, cache_dir=tmp_path)
    result = numsim.Engine().run(module, bindings)

    np.testing.assert_array_equal(result.outputs["old_shared"], np.arange(32, dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["old_u32"], np.arange(32, dtype=np.uint32))
    np.testing.assert_array_equal(result.outputs["old_u64"], np.arange(32, dtype=np.uint64))
    assert result.outputs["final_shared"][0] == 32
    assert result.outputs["atom_u32"][0] == 32
    assert result.outputs["atom_u64"][0] == 32
    assert result.outputs["red_i32"][0] == 32
    assert result.outputs["red_u32"][0] == 32
    assert result.outputs["red_or_u64"][0] == np.uint64(2**32 - 1)


def test_atomic_and_reduction_cache_hints_are_numerical_noops(tmp_path):
    module = numsim.transpile(atomic_cache_hints_are_numerical_noops, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "atom_counter": np.zeros(1, dtype=np.uint32),
            "red_counter": np.zeros(1, dtype=np.uint32),
            "old_values": np.zeros(3, dtype=np.uint32),
            "final_values": np.zeros(2, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["old_values"], np.arange(3, dtype=np.uint32))
    np.testing.assert_array_equal(result.outputs["final_values"], np.array([3, 3], dtype=np.uint32))


def test_cuda_and_release_ptx_global_i32_atomic_add_resolve_integer_addresses(tmp_path):
    module = numsim.transpile(global_i32_atomic_add_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "cuda_counter": np.zeros(1, dtype=np.int32),
            "ptx_counter": np.zeros(1, dtype=np.int32),
            "old_cuda": np.zeros(4, dtype=np.int32),
            "old_ptx": np.zeros(4, dtype=np.int32),
        },
    )

    np.testing.assert_array_equal(result.outputs["old_cuda"], np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["old_ptx"], np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["cuda_counter"], np.array([4], dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["ptx_counter"], np.array([4], dtype=np.int32))


def test_global_i32_atomic_cas_returns_old_and_only_writes_on_compare_success(tmp_path):
    module = numsim.transpile(global_i32_atomic_cas, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "cell": np.array([7], dtype=np.int32),
            "old_values": np.zeros(2, dtype=np.int32),
            "final_value": np.zeros(1, dtype=np.int32),
        },
    )

    np.testing.assert_array_equal(result.outputs["old_values"], np.array([7, 11], dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["cell"], np.array([11], dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["final_value"], np.array([11], dtype=np.int32))


def test_mega_st_bulk_zeroes_the_exact_shared_byte_range(tmp_path):
    module = numsim.transpile(mega_st_bulk_zero, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(4, dtype=np.uint32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([0, 0, 3, 4], dtype=np.uint32))


def test_st_bulk_executes_independently_for_every_active_lane(tmp_path):
    module = numsim.transpile(lane_wise_st_bulk_zero, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.ones(64, dtype=np.uint32)})
    np.testing.assert_array_equal(result.outputs["output"], np.zeros(64, dtype=np.uint32))


def test_non_tensor_bulk_copy_executes_independently_for_every_active_lane(tmp_path):
    source = np.arange(128, dtype=np.uint32) * np.uint32(17) + np.uint32(3)
    module = numsim.transpile(lane_wise_bulk_s2g, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "destination": np.zeros_like(source)})
    np.testing.assert_array_equal(result.outputs["destination"], source)


def test_mega_one_dimensional_bulk_copy_roundtrips_bytes(tmp_path):
    source = np.arange(32, dtype=np.uint8) ^ np.uint8(0xA5)
    module = numsim.transpile(mega_bulk_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "destination": np.zeros(32, dtype=np.uint8)}
    )

    np.testing.assert_array_equal(result.outputs["destination"], source)


@pytest.mark.parametrize(
    "kernel", [mega_bulk_roundtrip_without_cache_hint, mega_bulk_roundtrip_alternate_cache_hints]
)
def test_one_dimensional_bulk_cache_hints_are_numerical_noops(tmp_path, kernel):
    source = np.arange(32, dtype=np.uint8) ^ np.uint8(0xA5)
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "destination": np.zeros(32, dtype=np.uint8)}
    )

    np.testing.assert_array_equal(result.outputs["destination"], source)


def test_bulk_g2s_multicast_copies_to_every_selected_cta(tmp_path):
    source = np.arange(16, dtype=np.uint8) ^ np.uint8(0xA5)
    module = numsim.transpile(mega_bulk_g2s_multicast, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.zeros((2, 16), dtype=np.uint8)}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.broadcast_to(source, (2, 16)))


def test_bulk_s2g_cp_mask_updates_only_selected_bytes(tmp_path):
    source = np.arange(16, dtype=np.uint8) + np.uint8(0x20)
    destination = np.full(16, np.uint8(0xE7), dtype=np.uint8)
    module = numsim.transpile(mega_bulk_s2g_cp_mask, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "destination": destination})

    expected = destination.copy()
    expected[1::2] = source[1::2]
    np.testing.assert_array_equal(result.outputs["destination"], expected)


def test_bulk_read_wait_does_not_publish_destination_before_full_wait(tmp_path):
    source = np.arange(16, dtype=np.uint8) + np.uint8(0x40)
    destination = np.full(16, np.uint8(0x17), dtype=np.uint8)
    module = numsim.transpile(bulk_s2g_read_then_full_wait, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {"source": source, "destination": destination, "observed": np.zeros(2, dtype=np.uint8)},
    )

    np.testing.assert_array_equal(result.outputs["destination"], source)
    np.testing.assert_array_equal(
        result.outputs["observed"], np.array([0x17, source[0]], dtype=np.uint8)
    )


def test_bulk_groups_publish_oldest_completed_prefix_in_fifo_order(tmp_path):
    source = np.concatenate(
        (
            np.arange(16, dtype=np.uint8) + np.uint8(0x20),
            np.arange(16, dtype=np.uint8) + np.uint8(0x80),
        )
    )
    module = numsim.transpile(bulk_s2g_multiple_groups, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "destination": np.zeros(32, dtype=np.uint8),
            "observed": np.zeros(2, dtype=np.uint8),
        },
    )

    np.testing.assert_array_equal(result.outputs["destination"], source)
    np.testing.assert_array_equal(
        result.outputs["observed"], np.array([source[0], source[16]], dtype=np.uint8)
    )


def test_bulk_read_only_group_publishes_during_kernel_exit_drain(tmp_path):
    source = np.arange(16, dtype=np.uint8) ^ np.uint8(0xA5)
    module = numsim.transpile(bulk_s2g_read_only_exit, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module, {"source": source, "destination": np.zeros(16, dtype=np.uint8)}
    )

    np.testing.assert_array_equal(result.outputs["destination"], source)


def test_mega_cluster_arrive_wait_rendezvous_exposes_peer_global_writes(tmp_path):
    module = numsim.transpile(mega_cluster_arrive_wait, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"exchange": np.zeros(2, dtype=np.int32), "output": np.zeros(2, dtype=np.int32)}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.array([12, 11], dtype=np.int32))


def test_cluster_arrive_wait_rejects_partial_cluster_subset(tmp_path):
    module = numsim.transpile(mega_cluster_arrive_wait, cache_dir=tmp_path)
    with pytest.raises(
        numsim.NumSimExecutionError, match="CTA subset must be a union of complete clusters"
    ):
        numsim.Engine().run(
            module,
            {"exchange": np.zeros(2, dtype=np.int32), "output": np.full(2, -1, dtype=np.int32)},
            subset=ExecutionSubset(cta_ids=[0]),
        )


def test_mega_cluster_wait_does_not_require_every_arriving_warp_to_wait(tmp_path):
    module = numsim.transpile(mega_cluster_arrive_only_participant, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"exchange": np.zeros(2, dtype=np.int32), "output": np.zeros(1, dtype=np.int32)}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.array([22], dtype=np.int32))


def test_cluster_barrier_hub_is_captured_by_async_split_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(suspend_scaffold, "_ROOT_UNIFORM_IF_SPLIT_MIN_LINES", 0)
    module = numsim.transpile(mega_cluster_arrive_only_participant, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"exchange": np.zeros(2, dtype=np.int32), "output": np.zeros(1, dtype=np.int32)}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.array([22], dtype=np.int32))


def test_mega_lane_private_mbarrier_init_initializes_one_barrier_per_lane(tmp_path):
    module = numsim.transpile(mega_lane_private_mbarrier_init, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.array([17], dtype=np.int32))


def test_mega_mbarrier_init_rejects_partial_lane_aliasing(tmp_path):
    module = numsim.transpile(invalid_partially_aliased_mbarrier_init, cache_dir=tmp_path)
    with pytest.raises(
        numsim.NumSimExecutionError,
        match="mbarrier.init pointer must be warp-uniform or one-to-one across active lanes",
    ):
        numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})


def test_global_acquire_poll_reschedules_and_atomic_makes_progress(
    tmp_path, expect_harness_surface
):
    module = numsim.transpile(global_acquire_poll_woken_by_atomic, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"signal": np.zeros(1, dtype=np.uint32), "output": np.zeros(1, dtype=np.uint32)}
    )

    np.testing.assert_array_equal(result.outputs["signal"], np.array([1], dtype=np.uint32))
    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.uint32))
    assert result.stats["poll_order"] == [0, 1, 0]

    def check_scheduler_progress(value):
        assert value["poll_order"] == [0, 1, 0]
        assert value["completed_task_count"] == value["task_count"]

    expect_harness_surface(
        lambda: result.stats,
        check_scheduler_progress,
    )


def test_cross_cluster_atomic_progresses_with_shared_or_parallel_workers(tmp_path):
    module = numsim.transpile(global_acquire_poll_woken_by_cross_cluster_atomic, cache_dir=tmp_path)
    for max_workers in (1, 2):
        result = numsim.Engine(max_workers=max_workers).run(
            module, {"signal": np.zeros(1, dtype=np.uint32), "output": np.zeros(1, dtype=np.uint32)}
        )

        def check() -> None:
            np.testing.assert_array_equal(result.outputs["signal"], np.array([1], dtype=np.uint32))
            np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.uint32))
            assert result.stats["worker_count"] == max_workers
            assert result.stats["scheduling_domain_count"] == 2

        check()


def test_direct_global_acquire_poll_reschedules_same_cluster_peer(tmp_path):
    module = numsim.transpile(
        direct_global_acquire_poll_woken_by_same_cluster_atomic, cache_dir=tmp_path
    )
    result = numsim.Engine().run(
        module,
        {
            "done_counter": np.zeros(1, dtype=np.int32),
            "work_total": np.ones(1, dtype=np.int32),
            "output": np.zeros(1, dtype=np.int32),
        },
    )

    np.testing.assert_array_equal(result.outputs["done_counter"], np.ones(1, dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["output"], np.ones(1, dtype=np.int32))
    assert result.stats["poll_order"] == [0, 1, 0]


def test_global_volatile_poll_performs_first_reload_without_short_loop_suspend(tmp_path):
    module = numsim.transpile(global_volatile_poll_zero_init_first_reload, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"signal": np.ones(1, dtype=np.uint64), "output": np.zeros(1, dtype=np.uint64)}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.uint64))
    assert result.stats["poll_order"] == [0]


def test_global_poll_reschedules_with_lane_varying_active_mask(tmp_path):
    module = numsim.transpile(global_acquire_poll_two_lane_ranges, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"signal": np.zeros(2, dtype=np.uint32), "output": np.zeros(2, dtype=np.uint32)}
    )

    np.testing.assert_array_equal(result.outputs["signal"], np.ones(2, dtype=np.uint32))
    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.uint32))
    assert result.stats["poll_order"] == [0, 1, 2, 1, 0]


def test_plain_global_buffer_poll_uses_general_time_slice(tmp_path):
    module = numsim.transpile(plain_global_buffer_poll_two_lane_ranges, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"signal": np.zeros(2, dtype=np.int32), "output": np.zeros(2, dtype=np.int32)}
    )

    np.testing.assert_array_equal(result.outputs["signal"], np.ones(2, dtype=np.int32))
    np.testing.assert_array_equal(result.outputs["output"], np.ones(2, dtype=np.int32))
    assert result.stats["poll_order"] == [0, 1, 2, 1, 0]


def test_native_loop_with_extra_body_effect_uses_engine_budget(tmp_path):
    module = numsim.transpile(non_polling_loop_with_extra_body_effect, cache_dir=tmp_path)

    with pytest.raises(
        numsim.NumSimExecutionError, match="configured native loop iteration budget 1"
    ):
        numsim.Engine(native_loop_iteration_budget=1).run(
            module,
            {"signal": np.zeros(1, dtype=np.uint32), "output": np.zeros(1, dtype=np.uint32)},
        )


@pytest.mark.parametrize(
    ("kernel", "inputs"),
    [
        (
            direct_global_poll_with_body_effect,
            {"signal": np.zeros(1, dtype=np.int32), "output": np.zeros(1, dtype=np.int32)},
        ),
        (
            direct_global_poll_with_two_watched_loads,
            {"left": np.zeros(1, dtype=np.int32), "right": np.ones(1, dtype=np.int32)},
        ),
        (
            plain_global_buffer_poll_with_body_effect,
            {"signal": np.zeros(1, dtype=np.int32), "output": np.zeros(1, dtype=np.int32)},
        ),
        (
            plain_global_buffer_poll_with_two_loads,
            {"left": np.zeros(1, dtype=np.int32), "right": np.ones(1, dtype=np.int32)},
        ),
    ],
)
def test_all_native_while_shapes_use_engine_loop_budget(kernel, inputs, tmp_path):
    module = numsim.transpile(kernel, cache_dir=tmp_path)

    with pytest.raises(
        numsim.NumSimExecutionError, match="configured native loop iteration budget 1"
    ):
        numsim.Engine(native_loop_iteration_budget=1).run(module, inputs)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (_INVALID_ST_BULK_SIZE_SOURCE, r"operand 'size'.*dtype uint64"),
        (_INVALID_G2S_MASK_WITHOUT_MULTICAST_SOURCE, r"expects 4 operand\(s\).*got 5"),
        (_INVALID_S2G_MASK_WITHOUT_MODIFIER_SOURCE, r"expects 3 operand\(s\).*got 4"),
    ],
)
def test_table_driven_bulk_memory_syntax_rejects_unbound_operands(source: str, message: str):
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match=message):
        tvm.script.from_source(source, {"T": T})


def test_cluster_barrier_accepts_default_release_and_unaligned_forms(tmp_path):
    module = numsim.transpile(cluster_barrier_default_and_release_forms, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})
    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.int32))


def test_cluster_barrier_unaligned_arrive_accumulates_divergent_lanes(tmp_path):
    module = numsim.transpile(cluster_barrier_unaligned_divergent_arrive, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.int32)})
    np.testing.assert_array_equal(result.outputs["output"], np.array([2], dtype=np.int32))
