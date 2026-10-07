from __future__ import annotations

import numpy as np
import pytest
import tvm

from tirx_harness import numsim, racecheck, synccheck
from tests.numsim.support.kernels import (
    raw_tma_dynamic_wait_group_loop,
    raw_tma_fp4_align8_roundtrip,
    raw_tma_rank3_roundtrip,
    raw_tma_roundtrip,
    raw_tma_sm100_barrier_address,
    raw_tma_split_swizzle_atoms,
    raw_tma_swizzle_to_matching_layout,
    raw_tma_swizzle_to_dense,
    raw_tma_transaction_mismatch,
    raw_tma_zero_fill,
)
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm.script import tirx as T


@T.prim_func
def raw_tma_fp4_align16_physical_alias(
    input_map: T.TensorMap(), output: T.Buffer((2, 64), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    physical = T.alloc_buffer((256,), "uint8", scope="shared")
    destination = T.decl_buffer((2, 128), "uint8", data=physical.data, scope="shared")
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
                T.address_of(destination[0, 0]),
                T.address_of(input_map),
                0,
                0,
                T.address_of(barriers[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 128)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.warp_sync()
    for iteration in T.serial(4):
        linear = iteration * 32 + lane
        row = linear // 64
        packed_byte = linear % 64
        atom = packed_byte // 8 ^ row
        byte_in_atom = packed_byte % 8
        output[row, packed_byte] = physical[row * 128 + atom * 16 + byte_in_atom]


@T.prim_func
def raw_tma_fp4_align16_uninitialized_padding(
    input_map: T.TensorMap(), output: T.Buffer((1,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    physical = T.alloc_buffer((128,), "uint8", scope="shared")
    destination = T.decl_buffer((1, 128), "uint8", data=physical.data, scope="shared")
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
                T.address_of(destination[0, 0]),
                T.address_of(input_map),
                0,
                0,
                T.address_of(barriers[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 64)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
        output[0] = physical[8]


@T.prim_func
def raw_tma_fp4_align16_store_is_unsupported(output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, T.address_of(shared[0])
            )
        )


@T.prim_func
def raw_tma_multicast_cta_group1_per_target_barriers(
    input_map: T.TensorMap(), output: T.Buffer((2, 4), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1"
            ](
                T.address_of(shared[0]),
                T.address_of(input_map),
                0,
                T.address_of(barriers[0]),
                T.uint16(3),
            )
        )
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


@T.prim_func
def raw_tma_static_single_bit_mask_is_unicast(
    input_map: T.TensorMap(), output: T.Buffer((2, 4), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane < 4:
        shared[lane] = T.float32(-7)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](T.address_of(shared[0]), T.address_of(input_map), 0, T.address_of(barriers[0]))
        )
    if (cta == 0) and (lane == 0):
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


@T.prim_func
def raw_tma_dynamic_single_bit_mask_is_multicast(
    input_map: T.TensorMap(), cta_mask: T.int32, output: T.Buffer((4,), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1"
            ](
                T.address_of(shared[0]),
                T.address_of(input_map),
                0,
                T.address_of(barriers[0]),
                T.cast(cta_mask, "uint16"),
            )
        )
    if (cta == 1) and (lane == 0):
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if (cta == 1) and (lane < 4):
        output[lane] = shared[lane]


@T.prim_func
def raw_tma_select_descriptor(
    first_map: T.TensorMap(),
    second_map: T.TensorMap(),
    choose_first: T.int32,
    output: T.Buffer((4,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0]),
                T.Select(choose_first != 0, T.address_of(first_map), T.address_of(second_map)),
                0,
                T.address_of(barriers[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.warp_sync()
    if lane < 4:
        output[lane] = shared[lane]


@T.prim_func
def raw_tma_if_then_else_descriptor(
    first_map: T.TensorMap(),
    second_map: T.TensorMap(),
    choose_first: T.int32,
    output: T.Buffer((4,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0]),
                T.if_then_else(
                    choose_first != 0, T.address_of(first_map), T.address_of(second_map)
                ),
                0,
                T.address_of(barriers[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.warp_sync()
    if lane < 4:
        output[lane] = shared[lane]


@T.prim_func
def raw_tma_materialized_tensor_map_address(
    first_map: T.TensorMap(),
    second_map: T.TensorMap(),
    choose_first: T.int32,
    output: T.Buffer((4,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        selected_map: T.uint64 = T.reinterpret("uint64", T.address_of(first_map))
        selected_map = T.if_then_else(
            choose_first != 0,
            T.reinterpret("uint64", T.address_of(first_map)),
            T.reinterpret("uint64", T.address_of(second_map)),
        )
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0]),
                T.reinterpret(T.handle().ty, selected_map),
                0,
                T.address_of(barriers[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.warp_sync()
    if lane < 4:
        output[lane] = shared[lane]


@T.prim_func
def raw_tma_multicast_cta_group2_across_pairs(
    input_map: T.TensorMap(), output: T.Buffer((4, 4), "float32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([4])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta % 2 == 0) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
            ](
                T.address_of(shared[0]),
                T.address_of(input_map),
                0,
                T.address_of(barriers[0]),
                T.uint16(15),
            )
        )
    if (cta % 2 == 0) and (lane == 0):
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 32)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()
    if lane < 4:
        output[cta, lane] = shared[lane]


@T.prim_func
def raw_tma_outer_element_stride2(
    input_map: T.TensorMap(), roundtrip_map: T.TensorMap(), output: T.Buffer((3, 4), "float32")
):
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
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 48)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.warp_sync()
    if lane < 12:
        output[lane // 4, lane % 4] = shared[lane // 4, lane % 4]
    T.cuda.warp_sync()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                T.address_of(roundtrip_map), 0, 0, T.address_of(shared[0, 0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def raw_tma_read_then_full_wait(
    source: T.Buffer((4,), "float32"), first_map: T.TensorMap(), second_map: T.TensorMap()
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"](
                T.address_of(first_map), 0, T.address_of(shared[0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        for index in T.serial(4):
            shared[index] = shared[index] + T.float32(100)
        T.ptx.fence.proxy.async_.shared__cta()
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"](
                T.address_of(second_map), 0, T.address_of(shared[0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def raw_tma_uncommitted_exit_store(output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(lane, "float32")
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, T.address_of(shared[0])
            )
        )


@T.prim_func
def raw_tma_missing_wait(output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(lane, "float32")
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, T.address_of(shared[0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()


@T.prim_func
def raw_tma_read_only_exit(output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared")
    if lane < 4:
        shared[lane] = T.cast(lane, "float32")
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map), 0, T.address_of(shared[0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def raw_tma_predicated_s2g(output_map: T.TensorMap(), issue: T.int32):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    if lane < 12:
        shared[lane // 4, lane % 4] = T.cast(lane + 1, "float32")
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                T.address_of(output_map),
                0,
                0,
                T.address_of(shared[0, 0]),
                pred=T.And(lane == 0, issue != 0),
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def raw_tma_predicated_g2s(
    input_map: T.TensorMap(), output: T.Buffer((3, 4), "float32"), issue: T.int32
):
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
    T.evaluate(
        T.ptx[
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
        ](
            T.address_of(shared[0, 0]),
            T.address_of(input_map),
            0,
            0,
            T.address_of(barriers[0]),
            pred=T.And(lane == 0, issue != 0),
        )
    )
    if (lane == 0) and (issue != 0):
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 48)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.warp_sync()
    if (lane < 12) and (issue != 0):
        output[lane // 4, lane % 4] = shared[lane // 4, lane % 4]


@T.prim_func
def raw_tma_per_lane_coordinates(
    input_map: T.TensorMap(), output: T.Buffer((32, 4), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    # Each lane issues an independent 16-byte box.  Keep every destination
    # base 128-byte aligned, as required by the raw TMA shared-address form.
    shared = T.alloc_buffer((32, 32), "float32", scope="shared", align=128)
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    T.evaluate(
        T.ptx[
            "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
        ](
            T.address_of(shared[lane, 0]),
            T.address_of(input_map),
            lane * 4,
            T.address_of(barriers[0]),
        )
    )
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 512)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.warp_sync()
    for column in T.serial(4):
        output[lane, column] = shared[lane, column]


@T.prim_func
def raw_tma_gather4_per_lane_coordinates(
    input_map: T.TensorMap(), output: T.Buffer((32, 4, 4), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    # A gather writes 64 bytes here; the padded middle dimension gives each
    # lane an independently aligned 128-byte destination component.
    shared = T.alloc_buffer((32, 8, 4), "float32", scope="shared", align=128)
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    T.evaluate(
        T.ptx[
            "cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
        ](
            T.address_of(shared[lane, 0, 0]),
            T.address_of(input_map),
            0,
            lane * 4,
            lane * 4 + 1,
            lane * 4 + 2,
            lane * 4 + 3,
            T.address_of(barriers[0]),
        )
    )
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barriers[0]), 2048)
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.warp_sync()
    for row, column in T.grid(4, 4):
        output[lane, row, column] = shared[lane, row, column]


@T.prim_func
def raw_tma_warp_issued_s2g(output_map: T.TensorMap()):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((3, 4), "float32", scope="shared")
    if lane < 12:
        shared[lane // 4, lane % 4] = T.cast(lane + 1, "float32")
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    T.evaluate(
        T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
            T.address_of(output_map), 0, 0, T.address_of(shared[0, 0])
        )
    )
    T.ptx.cp.async_.bulk.commit_group()
    T.ptx.cp.async_.bulk.wait_group.read(0)


def _tensor_map(
    array: np.ndarray,
    *,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
    fp4_shared_layout: str | None = None,
    swizzle: str | None = None,
    interleave: str | None = None,
    fill_mode: str | None = None,
    element_strides: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base = array
    return (
        numsim.TensorMap(
            base=base,
            global_shape=global_shape,
            global_strides=global_strides,
            box_shape=box_shape,
            element_strides=(1,) * len(global_shape)
            if element_strides is None
            else element_strides,
            fp4_shared_layout=fp4_shared_layout,
            swizzle=swizzle,
            interleave=interleave,
            fill_mode=fill_mode,
        ).numpy(),
        base,
    )


def test_raw_tensor_map_g2c_and_s2g_roundtrip(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.25)
    output = np.zeros_like(source)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
    )

    module = numsim.transpile(raw_tma_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"input_map": input_map, "output_map": output_map}, outputs={"output": "output_map"}
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_tma_s2g_instruction_predicate_controls_the_store(tmp_path):
    initial = np.full((3, 4), np.float32(-7))
    module = numsim.transpile(raw_tma_predicated_s2g, cache_dir=tmp_path)

    def run(issue: int) -> np.ndarray:
        output_map, _ = _tensor_map(
            initial.copy(),
            global_shape=(4, 3),
            global_strides=(16,),
            box_shape=(4, 3),
        )
        result = numsim.Engine().run(
            module,
            {"output_map": output_map, "issue": issue},
            outputs={"output": "output_map"},
        )
        return result.outputs["output"]

    np.testing.assert_array_equal(run(1), np.arange(1, 13, dtype=np.float32).reshape(3, 4))
    np.testing.assert_array_equal(run(0), initial)


def test_raw_tma_g2s_instruction_predicate_controls_the_load(tmp_path):
    source = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
    )
    module = numsim.transpile(raw_tma_predicated_g2s, cache_dir=tmp_path)

    def run(issue: int) -> np.ndarray:
        initial = np.full((3, 4), np.float32(-7))
        result = numsim.Engine().run(
            module,
            {"input_map": input_map, "output": initial.copy(), "issue": issue},
            outputs=("output",),
        )
        return result.outputs["output"]

    np.testing.assert_array_equal(run(1), source)
    np.testing.assert_array_equal(run(0), np.full((3, 4), np.float32(-7)))


def test_raw_tma_g2s_issues_once_per_active_lane(tmp_path):
    source = np.arange(128, dtype=np.float32).reshape(32, 4) + np.float32(0.25)
    input_map, _ = _tensor_map(
        source,
        global_shape=(128,),
        global_strides=(),
        box_shape=(4,),
    )
    result = numsim.Engine().run(
        numsim.transpile(raw_tma_per_lane_coordinates, cache_dir=tmp_path),
        {"input_map": input_map, "output": np.zeros_like(source)},
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_tma_gather4_issues_once_per_active_lane(tmp_path):
    source = np.arange(512, dtype=np.float32).reshape(128, 4) + np.float32(0.25)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 128),
        global_strides=(16,),
        box_shape=(4, 1),
    )
    result = numsim.Engine().run(
        numsim.transpile(raw_tma_gather4_per_lane_coordinates, cache_dir=tmp_path),
        {"input_map": input_map, "output": np.zeros((32, 4, 4), dtype=np.float32)},
        outputs=("output",),
    )

    np.testing.assert_array_equal(result.outputs["output"], source.reshape(32, 4, 4))


@pytest.mark.parametrize("checker", [synccheck, racecheck])
def test_raw_tma_gather4_per_lane_coordinates_are_checker_clean(checker):
    source = np.arange(512, dtype=np.float32).reshape(128, 4) + np.float32(0.25)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 128),
        global_strides=(16,),
        box_shape=(4, 1),
    )
    report = checker(
        raw_tma_gather4_per_lane_coordinates,
        {"input_map": input_map, "output": np.zeros((32, 4, 4), dtype=np.float32)},
    )

    assert report.verdict == "clean"
    assert report.findings == []


def test_raw_tma_s2g_accepts_one_full_issuing_warp(tmp_path):
    output = np.zeros((3, 4), dtype=np.float32)
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
    )
    result = numsim.Engine().run(
        numsim.transpile(raw_tma_warp_issued_s2g, cache_dir=tmp_path),
        {"output_map": output_map},
        outputs={"output": "output_map"},
    )

    np.testing.assert_array_equal(
        result.outputs["output"], np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    )


@pytest.mark.parametrize(
    ("dimension", "value"),
    (
        ("interleave", None),
        ("interleave", "none"),
        ("fill_mode", None),
        ("fill_mode", "zero"),
        ("fill_mode", "nan"),
        ("swizzle", None),
        ("swizzle", "32B"),
        ("swizzle", "64B"),
        ("swizzle", "128B"),
    ),
    ids=(
        "interleave-python-none",
        "interleave-string-none",
        "fill-none",
        "fill-zero",
        "fill-nan",
        "swizzle-none",
        "swizzle-32B",
        "swizzle-64B",
        "swizzle-128B",
    ),
)
def test_tensor_map_descriptor_binding_form_executes(dimension, value, tmp_path):
    cache_dir = tmp_path.parent / "tensor-map-descriptor-evidence"
    if dimension == "fill_mode":
        source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(10)
        output = np.zeros((4, 4), dtype=np.float32)
        input_map, _ = _tensor_map(
            source,
            global_shape=(4, 3),
            global_strides=(16,),
            box_shape=(4, 4),
            fill_mode=value,
        )
        output_map, _ = _tensor_map(
            output,
            global_shape=(4, 4),
            global_strides=(16,),
            box_shape=(4, 4),
        )
        result = numsim.Engine().run(
            numsim.transpile(raw_tma_zero_fill, cache_dir=cache_dir),
            {"input_map": input_map, "output_map": output_map},
            outputs={"output": "output_map"},
        )
        if value == "nan":
            actual_bits = result.outputs["output"].view(np.uint32)
            np.testing.assert_array_equal(actual_bits[0], np.full(4, 0x7FF7_7FF7, dtype=np.uint32))
            np.testing.assert_array_equal(result.outputs["output"][1:], source)
        else:
            expected = np.concatenate([np.zeros((1, 4), dtype=np.float32), source], axis=0)
            np.testing.assert_array_equal(result.outputs["output"], expected)
        return

    if dimension == "swizzle" and value is not None:
        columns = {"32B": 8, "64B": 16, "128B": 32}[value]
        kernel = {
            "32B": raw_tma_swizzle_to_matching_layout(32),
            "64B": raw_tma_swizzle_to_matching_layout(64),
            "128B": raw_tma_swizzle_to_matching_layout(128),
        }[value]
        source = np.arange(8 * columns, dtype=np.float32).reshape(8, columns) + np.float32(0.25)
        input_map, _ = _tensor_map(
            source,
            global_shape=(columns, 8),
            global_strides=(columns * 4,),
            box_shape=(columns, 8),
            swizzle=value,
        )
        result = numsim.Engine().run(
            numsim.transpile(kernel, cache_dir=cache_dir),
            {"input_map": input_map, "output": np.zeros_like(source)},
        )
        np.testing.assert_array_equal(result.outputs["output"], source)
        return

    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.25)
    output = np.zeros_like(source)
    descriptor_options = {dimension: value}
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
        **descriptor_options,
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
    )
    result = numsim.Engine().run(
        numsim.transpile(raw_tma_roundtrip, cache_dir=cache_dir),
        {"input_map": input_map, "output_map": output_map},
        outputs={"output": "output_map"},
    )
    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_tensor_map_roundtrip_runs_as_a_numsim_case(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMSIM_CACHE_DIR", str(tmp_path))
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.25)
    output = np.zeros_like(source)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
    )
    case = numsim.NumSimCase(
        kernel=raw_tma_roundtrip,
        args={"input_map": input_map, "output_map": output_map},
        outputs={"output": "output_map"},
        reference=lambda: {"output": source.copy()},
    )

    numsim.run_case(case).require_ok()


def test_bulk_read_wait_allows_source_reuse_before_full_wait(tmp_path):
    source = np.arange(4, dtype=np.float32) + np.float32(0.25)
    first = np.zeros_like(source)
    second = np.zeros_like(source)
    first_map, _ = _tensor_map(
        first,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    second_map, _ = _tensor_map(
        second,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )

    module = numsim.transpile(raw_tma_read_then_full_wait, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "first_map": first_map, "second_map": second_map},
        outputs={"first": "first_map", "second": "second_map"},
    )

    np.testing.assert_array_equal(result.outputs["first"], source)
    np.testing.assert_array_equal(result.outputs["second"], source + np.float32(100))


def test_uncommitted_bulk_store_completes_at_kernel_exit(tmp_path):
    output = np.zeros(4, dtype=np.float32)
    output_map, _ = _tensor_map(
        output,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    module = numsim.transpile(raw_tma_uncommitted_exit_store, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module, {"output_map": output_map}, outputs={"output": "output_map"}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.arange(4, dtype=np.float32))


def test_committed_bulk_group_completes_at_kernel_exit_without_explicit_wait(tmp_path):
    output = np.full(4, np.float32(-1), dtype=np.float32)
    output_map, _ = _tensor_map(
        output,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    module = numsim.transpile(raw_tma_missing_wait, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module, {"output_map": output_map}, outputs={"output": "output_map"}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.arange(4, dtype=np.float32))


def test_bulk_read_only_wait_completes_at_kernel_exit(tmp_path):
    output = np.full(4, np.float32(-1), dtype=np.float32)
    output_map, _ = _tensor_map(
        output,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    module = numsim.transpile(raw_tma_read_only_exit, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module, {"output_map": output_map}, outputs={"output": "output_map"}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.arange(4, dtype=np.float32))


def test_raw_tensor_map_element_stride_controls_transfer_count(tmp_path):
    source = np.arange(20, dtype=np.float32).reshape(5, 4) + np.float32(0.25)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 5),
        global_strides=(16,),
        box_shape=(4, 5),
        element_strides=(0, 2),
    )
    output = np.zeros((3, 4), dtype=np.float32)
    roundtrip = np.full_like(source, np.float32(-1))
    roundtrip_map, _ = _tensor_map(
        roundtrip,
        global_shape=(4, 5),
        global_strides=(16,),
        box_shape=(4, 5),
        element_strides=(0, 2),
    )

    module = numsim.transpile(raw_tma_outer_element_stride2, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"input_map": input_map, "roundtrip_map": roundtrip_map, "output": output},
        outputs={"observed": "output", "roundtrip": "roundtrip_map"},
    )

    np.testing.assert_array_equal(result.outputs["observed"], source[[0, 2, 4]])
    expected_roundtrip = np.full_like(source, np.float32(-1))
    expected_roundtrip[[0, 2, 4]] = source[[0, 2, 4]]
    np.testing.assert_array_equal(result.outputs["roundtrip"], expected_roundtrip)


def test_tensor_map_numpy_rejects_invalid_descriptor_metadata():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)

    with pytest.raises(ValueError, match="box dimensions must be in 1..256"):
        _tensor_map(
            source,
            global_shape=(4, 3),
            global_strides=(16,),
            box_shape=(4, 257),
        )


def test_raw_tma_s2g_returns_the_tensor_map_output(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(0.25)
    output = np.zeros_like(source)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
        interleave="none",
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
    )

    module = numsim.transpile(raw_tma_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"input_map": input_map, "output_map": output_map})

    np.testing.assert_array_equal(result.outputs["output_map"], source)


def test_raw_tma_cta_group1_multicast_completes_each_target_barrier(tmp_path):
    source = np.arange(4, dtype=np.float32) + np.float32(0.25)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    output = np.zeros((2, 4), dtype=np.float32)

    module = numsim.transpile(raw_tma_multicast_cta_group1_per_target_barriers, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"input_map": input_map, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.broadcast_to(source, (2, 4)))


def test_raw_tma_static_single_bit_mask_is_unicast_from_the_issuing_cta(tmp_path):
    source = np.arange(4, dtype=np.float32) + np.float32(2.5)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    output = np.zeros((2, 4), dtype=np.float32)

    module = numsim.transpile(raw_tma_static_single_bit_mask_is_unicast, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"input_map": input_map, "output": output})

    expected = np.stack((source, np.full(4, -7, dtype=np.float32)))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tma_dynamic_single_bit_mask_retains_multicast_semantics(tmp_path):
    source = np.arange(4, dtype=np.float32) + np.float32(2.5)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    output = np.zeros(4, dtype=np.float32)

    module = numsim.transpile(raw_tma_dynamic_single_bit_mask_is_multicast, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"input_map": input_map, "cta_mask": 2, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


@pytest.mark.parametrize("kernel", [
    raw_tma_select_descriptor,
    raw_tma_if_then_else_descriptor,
    raw_tma_materialized_tensor_map_address,
])
def test_raw_tma_tensor_map_selector_branches_at_runtime(kernel, tmp_path):
    first = np.arange(4, dtype=np.float32) + np.float32(11)
    second = np.arange(4, dtype=np.float32) - np.float32(7)
    first_map, _ = _tensor_map(
        first,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    second_map, _ = _tensor_map(
        second,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    module = numsim.transpile(kernel, cache_dir=tmp_path)

    for choose_first, expected in ((0, second), (1, first)):
        output = np.zeros(4, dtype=np.float32)
        result = numsim.Engine().run(
            module,
            {
                "first_map": first_map,
                "second_map": second_map,
                "choose_first": choose_first,
                "output": output,
            },
        )
        np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tma_cta_group2_multicast_completes_every_target_pair(tmp_path):
    source = np.arange(4, dtype=np.float32) - np.float32(1.5)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4,),
        global_strides=(),
        box_shape=(4,),
    )
    output = np.zeros((4, 4), dtype=np.float32)

    module = numsim.transpile(raw_tma_multicast_cta_group2_across_pairs, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"input_map": input_map, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.broadcast_to(source, (4, 4)))


def _aligned_u8(values):
    storage = np.empty(values.size + 31, np.uint8)
    offset = -storage.ctypes.data % 32
    aligned = storage[offset : offset + values.size].reshape(values.shape)
    aligned[...] = values
    return aligned


def test_raw_fp4_align16_tma_writes_packed_data_into_swizzled_16b_atoms(tmp_path):
    source = _aligned_u8(np.arange(128, dtype=np.uint8).reshape(2, 64))
    input_map, _ = _tensor_map(
        source,
        global_shape=(128, 2),
        global_strides=(64,),
        box_shape=(128, 2),
        fp4_shared_layout="align16_padded",
        swizzle="128B",
    )
    output = np.zeros_like(source)

    module = numsim.transpile(raw_tma_fp4_align16_physical_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"input_map": input_map, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_fp4_align16_tma_padding_read_is_zero_filled_and_requires_review(tmp_path):
    source = _aligned_u8(np.arange(64, dtype=np.uint8).reshape(1, 64))
    input_map, _ = _tensor_map(
        source,
        global_shape=(128, 1),
        global_strides=(64,),
        box_shape=(128, 1),
        fp4_shared_layout="align16_padded",
        swizzle="128B",
    )
    module = numsim.transpile(raw_tma_fp4_align16_uninitialized_padding, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module, {"input_map": input_map, "output": np.full(1, 0xFF, dtype=np.uint8)}
    )

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(1, dtype=np.uint8))
    assert result.verdict == "review"
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}


def test_raw_fp4_align16_tensor_map_store_is_rejected(tmp_path):
    output = _aligned_u8(np.zeros(64, dtype=np.uint8))
    output_map, _ = _tensor_map(
        output,
        global_shape=(128,),
        global_strides=(),
        box_shape=(128,),
        fp4_shared_layout="align16_padded",
        swizzle="128B",
    )
    module = numsim.transpile(raw_tma_fp4_align16_store_is_unsupported, cache_dir=tmp_path)

    with pytest.raises(
        numsim.NumSimExecutionError,
        match="align16 padded FP4 TensorMap does not support shared-to-global Tensor Copy",
    ):
        numsim.Engine().run(module, {"output_map": output_map})


def test_raw_fp4_tensor_map_keeps_align8_shared_bytes_packed(tmp_path):
    source = np.arange(128, dtype=np.uint8).reshape(2, 64)
    output = np.zeros_like(source)
    input_map, _ = _tensor_map(
        source,
        global_shape=(128, 2),
        global_strides=(64,),
        box_shape=(128, 2),
        fp4_shared_layout="align8_packed",
        swizzle="64B",
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(128, 2),
        global_strides=(64,),
        box_shape=(128, 2),
        fp4_shared_layout="align8_packed",
        swizzle="64B",
    )

    module = numsim.transpile(raw_tma_fp4_align8_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"input_map": input_map, "output_map": output_map}, outputs={"output": "output_map"}
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_static_bulk_wait_group_immediates_execute_in_source_order(tmp_path):
    output = np.zeros(2, dtype=np.int32)

    spec = analyze(raw_tma_dynamic_wait_group_loop)
    assert spec.unsupported == ()
    module = numsim.transpile(raw_tma_dynamic_wait_group_loop, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.array([10, 11], dtype=np.int32))


def test_bulk_wait_group_rejects_runtime_count_during_numsim_transpilation(tmp_path):
    source = """
@T.prim_func
def invalid(pending: T.Buffer((1,), "int32")):
    T.device_entry()
    T.ptx.cp.async_.bulk.wait_group.read(pending[0])
"""

    kernel = tvm.script.from_source(source, {"T": T})
    with pytest.raises(numsim.UnsupportedTIRxError, match="pending_group_count must be static"):
        numsim.transpile(kernel, cache_dir=tmp_path)


def test_raw_rank3_tensor_map_uses_descriptor_strides(tmp_path):
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - np.float32(7)
    output = np.zeros_like(source)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3, 2),
        global_strides=(16, 48),
        box_shape=(4, 3, 2),
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 3, 2),
        global_strides=(16, 48),
        box_shape=(4, 3, 2),
    )

    module = numsim.transpile(raw_tma_rank3_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"input_map": input_map, "output_map": output_map}, outputs={"output": "output_map"}
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_tma_two_swizzled_atoms_cover_a_wider_k_tile(tmp_path):
    source = np.arange(128, dtype=np.float32).reshape(2, 64) - np.float32(19)
    output = np.zeros_like(source)
    input_map, _ = _tensor_map(
        source,
        global_shape=(64, 2),
        global_strides=(256,),
        box_shape=(32, 2),
        swizzle="128B",
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(64, 2),
        global_strides=(256,),
        box_shape=(32, 2),
        swizzle="128B",
    )

    module = numsim.transpile(raw_tma_split_swizzle_atoms, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"input_map": input_map, "output_map": output_map}, outputs={"output": "output_map"}
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_tensor_map_oob_zero_fill_counts_the_full_box(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(10)
    output = np.full((4, 4), np.float32(-1))
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 4),
        fill_mode="zero",
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 4),
        global_strides=(16,),
        box_shape=(4, 4),
    )

    module = numsim.transpile(raw_tma_zero_fill, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"input_map": input_map, "output_map": output_map}, outputs={"output": "output_map"}
    )

    expected = np.concatenate([np.zeros((1, 4), dtype=np.float32), source], axis=0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tensor_map_oob_nan_uses_hardware_fill_pattern(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(10)
    output = np.zeros((4, 4), dtype=np.float32)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 4),
        fill_mode="nan",
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 4),
        global_strides=(16,),
        box_shape=(4, 4),
    )

    module = numsim.transpile(raw_tma_zero_fill, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"input_map": input_map, "output_map": output_map}, outputs={"output": "output_map"}
    )

    actual_bits = result.outputs["output"].view(np.uint32)
    np.testing.assert_array_equal(actual_bits[0], np.full(4, 0x7FF7_7FF7, dtype=np.uint32))
    np.testing.assert_array_equal(result.outputs["output"][1:], source)


def test_raw_tensor_map_reused_unit_scratch_zero_fills_after_valid_rows(tmp_path):
    source = np.arange(8, dtype=np.float32).reshape(2, 4) + np.float32(10)
    output = np.full((4, 4), np.float32(-1))
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 2),
        global_strides=(16,),
        box_shape=(4, 4),
        fill_mode="zero",
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 4),
        global_strides=(16,),
        box_shape=(4, 4),
    )

    module = numsim.transpile(raw_tma_zero_fill, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"input_map": input_map, "output_map": output_map}, outputs={"output": "output_map"}
    )

    zeros = np.zeros((1, 4), dtype=np.float32)
    expected = np.concatenate([zeros, source, zeros], axis=0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tensor_map_oob_zero_fills_with_default_float_oob_mode(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(10)
    output = np.full((4, 4), np.float32(-1))
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 4),
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(4, 4),
        global_strides=(16,),
        box_shape=(4, 4),
    )

    module = numsim.transpile(raw_tma_zero_fill, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"input_map": input_map, "output_map": output_map}, outputs={"output": "output_map"}
    )

    expected = np.concatenate([np.zeros((1, 4), dtype=np.float32), source], axis=0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tensor_map_swizzle_is_a_physical_shared_layout(tmp_path):
    source = np.arange(64, dtype=np.float32).reshape(8, 8)
    output = np.zeros_like(source)
    input_map, _ = _tensor_map(
        source,
        global_shape=(8, 8),
        global_strides=(32,),
        box_shape=(8, 8),
        swizzle="32B",
    )
    output_map, _ = _tensor_map(
        output,
        global_shape=(8, 8),
        global_strides=(32,),
        box_shape=(8, 8),
    )

    module = numsim.transpile(raw_tma_swizzle_to_dense, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"input_map": input_map, "output_map": output_map}, outputs={"output": "output_map"}
    )

    expected = source.copy()
    expected[4:] = np.concatenate([source[4:, 4:], source[4:, :4]], axis=1)
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize("swizzle_bytes", (32, 64, 128))
def test_raw_tensor_map_swizzle_matches_canonical_shared_atom(tmp_path, swizzle_bytes):
    columns = swizzle_bytes // 4
    swizzle = f"{swizzle_bytes}B"
    kernel = raw_tma_swizzle_to_matching_layout(swizzle_bytes)
    source = np.arange(8 * columns, dtype=np.float32).reshape(8, columns) + np.float32(0.25)
    output = np.zeros_like(source)
    input_map, _ = _tensor_map(
        source,
        global_shape=(columns, 8),
        global_strides=(columns * 4,),
        box_shape=(columns, 8),
        swizzle=swizzle,
    )

    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"input_map": input_map, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_sm100_two_cta_barrier_address_targets_pair_base(tmp_path):
    spec = analyze(raw_tma_sm100_barrier_address)
    assert spec.unsupported == ()

    even = np.arange(4, dtype=np.float32).reshape(1, 4) + np.float32(10)
    odd = np.arange(4, dtype=np.float32).reshape(1, 4) + np.float32(20)
    even_map, _ = _tensor_map(
        even,
        global_shape=(4, 1),
        global_strides=(16,),
        box_shape=(4, 1),
    )
    odd_map, _ = _tensor_map(
        odd,
        global_shape=(4, 1),
        global_strides=(16,),
        box_shape=(4, 1),
    )

    module = numsim.transpile(raw_tma_sm100_barrier_address, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "input_map_even": even_map,
            "input_map_odd": odd_map,
            "output": np.zeros((2, 4), dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], np.concatenate([even, odd], axis=0))


def test_raw_tensor_map_under_delivery_reports_exact_bytes(tmp_path):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    input_map, _ = _tensor_map(
        source,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
    )
    module = numsim.transpile(raw_tma_transaction_mismatch, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="transactions=48/52"):
        numsim.Engine().run(module, {"input_map": input_map})
