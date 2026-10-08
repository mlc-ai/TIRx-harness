from __future__ import annotations

import numpy as np
import pytest
from tvm import tirx
from tvm.ir import Call
from tvm.script import tirx as T
from tvm_ffi import structural_map, structural_walk

from tirx_harness import numsim, racecheck, synccheck
from tests.numsim.support.execution import run_checked
from tests.numsim.support.manifest import call_op_names


@T.prim_func
def raw_bulk_s2c_uses_mapped_u32_addresses(output: T.Buffer((16,), "uint8")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    source = T.alloc_buffer((16,), "uint8", scope="shared", align=128)
    destination = T.alloc_buffer((16,), "uint8", scope="shared", align=128)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 1) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    if (cta == 0) and (lane < 16):
        source[lane] = T.cast(lane + 19, "uint8")
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        remote_barrier = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_barrier[0],
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
            T.uint32(1),
        )
        remote_destination = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_destination[0],
            T.cuda.cvta_generic_to_shared(destination.ptr_to([0])),
            T.uint32(1),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
            remote_barrier[0], T.uint32(16), pred=True
        )
        T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
            remote_destination[0],
            source.ptr_to([0]),
            T.uint32(16),
            remote_barrier[0],
            pred=False,
        )
        T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
            remote_destination[0],
            source.ptr_to([0]),
            T.uint32(16),
            remote_barrier[0],
            pred=lane == 0,
        )
    if (cta == 1) and (lane == 0):
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if (cta == 1) and (lane < 16):
        output[lane] = destination[lane]


@T.prim_func
def raw_st_async_uses_mapped_u32_addresses(output: T.Buffer((8,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = T.alloc_buffer((8,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 1) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        remote_barrier = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_barrier[0],
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
            T.uint32(1),
        )
        remote_destination = T.alloc_local((1,), "uint32")
        remote_destination_hi = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_destination[0],
            T.cuda.cvta_generic_to_shared(destination.ptr_to([0])),
            T.uint32(1),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
            remote_barrier[0], T.uint32(32), pred=True
        )
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
            remote_destination[0],
            T.uint32(0x10203040),
            T.uint32(0x50607080),
            T.uint32(0x90A0B0C0),
            T.uint32(0xD0E0F001),
            remote_barrier[0],
        )
        T.ptx.add.u32(remote_destination_hi[0], remote_destination[0], T.uint32(16))
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
            remote_destination_hi[0],
            T.uint32(0x12345678),
            T.uint32(0x9ABCDEF0),
            T.uint32(0x0BADF00D),
            T.uint32(0xCAFEBABE),
            remote_barrier[0],
        )
    if (cta == 1) and (lane == 0):
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if (cta == 1) and (lane < 8):
        output[lane] = destination[lane]


@T.prim_func
def raw_st_async_uses_mapped_u32_expression_offset(output: T.Buffer((8,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    destination = T.alloc_buffer((8,), "uint32", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if (cta == 1) and (lane == 0):
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        remote_barrier = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_barrier[0],
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
            T.uint32(1),
        )
        remote_destination = T.alloc_local((1,), "uint32")
        T.ptx.mapa.shared__cluster.u32(
            remote_destination[0],
            T.cuda.cvta_generic_to_shared(destination.ptr_to([0])),
            T.uint32(1),
        )
        T.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
            remote_barrier[0], T.uint32(32), pred=True
        )
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
            remote_destination[0],
            T.uint32(0x10203040),
            T.uint32(0x50607080),
            T.uint32(0x90A0B0C0),
            T.uint32(0xD0E0F001),
            remote_barrier[0],
        )
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.v4.u32(
            remote_destination[0] + T.uint32(16),
            T.uint32(0x12345678),
            T.uint32(0x9ABCDEF0),
            T.uint32(0x0BADF00D),
            T.uint32(0xCAFEBABE),
            remote_barrier[0],
        )
    if (cta == 1) and (lane == 0):
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if (cta == 1) and (lane < 8):
        output[lane] = destination[lane]


@T.prim_func
def bulk_s2g_cp_mask_without_cache(
    source: T.Buffer((16,), "uint8"),
    byte_mask: T.Buffer((1,), "int32"),
    destination: T.Buffer((16,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared", align=16)
    if lane < 16:
        shared[lane] = source[lane]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.cp_mask"](
            destination.ptr_to([0]),
            shared.ptr_to([0]),
            T.cast(16, "uint32"),
            T.cast(byte_mask[0], "uint16"),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def bulk_g2s_ignore_oob_dead_bytes(
    source: T.Buffer((16,), "uint8"),
    ignored: T.Buffer((2,), "int32"),
    output: T.Buffer((16,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.ignore_oob"](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            T.cast(16, "uint32"),
            T.cast(ignored[0], "uint32"),
            T.cast(ignored[1], "uint32"),
            barrier.ptr_to([0]),
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.warp_sync()
    if (lane >= ignored[0]) and (lane < 16 - ignored[1]):
        output[lane] = shared[lane]


@T.prim_func
def bulk_g2s_ignore_oob_live_read(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((1,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane < 16:
        shared[lane] = T.uint8(0xE7)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.ignore_oob"](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            T.cast(16, "uint32"),
            T.cast(3, "uint32"),
            T.cast(4, "uint32"),
            barrier.ptr_to([0]),
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        output[0] = shared[0]


@T.prim_func
def bulk_g2s_ignore_oob_right_outside_binding(
    source: T.Buffer((13,), "uint8"), output: T.Buffer((13,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.ignore_oob"](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            T.cast(16, "uint32"),
            T.cast(0, "uint32"),
            T.cast(3, "uint32"),
            barrier.ptr_to([0]),
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.warp_sync()
    if lane < 13:
        output[lane] = shared[lane]


def lane_varying_st_bulk_size(size_dtype="uint64"):
    @T.prim_func
    def kernel(
        output: T.Buffer((32,), "uint8"), size: T.int64, enabled: T.int32, zero_first: T.int32
    ):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        shared = T.alloc_buffer((32,), "uint8", scope="shared", align=8)
        shared[lane] = T.cast(lane + 1, "uint8")
        T.cuda.warp_sync()
        if lane < 2:
            T.ptx.st_bulk.shared__cta(
                shared.ptr_to([lane * 8]),
                T.cast(
                    T.if_then_else((lane == 0) & (zero_first != 0), 0, (lane + 1) * size),
                    size_dtype,
                ),
                pred=enabled != 0,
            )
        T.cuda.warp_sync()
        output[lane] = shared[lane]

    return kernel


@T.prim_func
def lane_varying_bulk_s2g_size_and_mask(
    source: T.Buffer((64,), "uint8"), destination: T.Buffer((64,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint8", scope="shared")
    if lane == 0:
        for index in T.serial(64):
            shared[index] = source[index]
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane < 2:
        base: T.int32 = lane * 32
        num_bytes: T.int32 = (lane + 1) * 16
        byte_mask: T.int32 = T.Select(lane == 0, T.int32(0x00FF), T.int32(0xFF00))
        T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.cp_mask"](
            destination.ptr_to([base]),
            shared.ptr_to([base]),
            T.cast(num_bytes, "uint32"),
            T.uint64(0x1000000000000000),
            T.cast(byte_mask, "uint16"),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def lane_varying_bulk_multicast_cta_mask(
    source: T.Buffer((32,), "uint8"), output: T.Buffer((2, 16), "uint8")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.proxy.async_.shared__cluster()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
    if (cta == 0) and (lane < 2):
        base: T.int32 = lane * 16
        cta_mask: T.int32 = T.shift_left(T.int32(1), lane)
        T.ptx[
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
        ](
            shared.ptr_to([base]),
            source.ptr_to([base]),
            T.cast(16, "uint32"),
            barrier.ptr_to([0]),
            T.cast(cta_mask, "uint16"),
        )
    if lane == 0:
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.cluster_sync()
    if lane < 16:
        output[cta, lane] = shared[cta * 16 + lane]


@T.prim_func
def bulk_g2s_cluster_static_true_predicate(
    source: T.Buffer((16,), "uint8"), output: T.Buffer((16,), "uint8")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), 16)
        T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            T.uint32(16),
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
            pred=T.bool(True),
        )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = shared[lane]


@T.prim_func
def bulk_g2s_cluster_static_false_predicate(source: T.Buffer((16,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    if lane == 0:
        T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([0]),
            source.ptr_to([0]),
            T.uint32(16),
            T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
            pred=T.bool(False),
        )


@T.prim_func
def bulk_g2s_cluster_dynamic_predicate(source: T.Buffer((16,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16,), "uint8", scope="shared", align=16)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
        shared.ptr_to([0]),
        source.ptr_to([0]),
        T.uint32(16),
        T.cuda.cvta_generic_to_shared(barrier.ptr_to([0])),
        pred=lane == 0,
    )


@T.prim_func
def raw_bulk_prefetch(
    source: T.Buffer((64,), "uint8"), num_bytes: T.uint32, output: T.Buffer((32,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["cp.async.bulk.prefetch.L2.global"](source.ptr_to([16]), num_bytes)
    output[lane] = source[lane + 16]


def _cp_mask_readonly_kernel(before):
    @T.prim_func
    def kernel(
        destination: T.Buffer((16,), "uint8"),
        byte_mask: T.uint32,
        output: T.Buffer((1,), "uint32"),
    ):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        shared = T.alloc_shared((16,), "uint8", align=16)
        value = T.alloc_local((1,), "uint32")
        if lane < 16:
            shared[lane] = T.uint8(0x35)
        T.cuda.warp_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        if lane == 0:
            if before:
                T.ptx["ld.global.u32.proxy::readonly"](value[0], destination.ptr_to([0]))
            T.ptx["cp.async.bulk.global.shared::cta.bulk_group.cp_mask"](
                destination.ptr_to([0]),
                shared.ptr_to([0]),
                T.uint32(16),
                T.cast(byte_mask, "uint16"),
            )
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)
            if not before:
                T.ptx["ld.global.u32.proxy::readonly"](value[0], destination.ptr_to([0]))
            output[0] = value[0]

    return kernel


def test_cp_mask_readonly_overlap_uses_selected_bytes(tmp_path):
    from tirx_harness import racecheck, synccheck

    for before in (False, True):
        kernel = _cp_mask_readonly_kernel(before)
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        for mask in (0, 0xF0, 1):
            args = {
                "destination": np.full(16, 0xD3, np.uint8),
                "byte_mask": mask,
                "output": np.zeros(1, np.uint32),
            }
            for checker in (synccheck, racecheck):
                report = checker(kernel, args)
                if mask == 1:
                    assert report.verdict == "error", report.format()
                    assert "write overlaps readonly bytes" in str(report.to_dict())
                else:
                    report.require_clean()
            if mask == 1:
                with pytest.raises(
                    numsim.NumSimExecutionError, match="write overlaps readonly bytes"
                ):
                    numsim.Engine().run(module, args)
            else:
                result = numsim.Engine().run(module, args)
                assert result.verdict == "clean"
                np.testing.assert_array_equal(result.outputs["output"], [0xD3D3D3D3])
                np.testing.assert_array_equal(
                    result.outputs["destination"],
                    np.where((mask >> np.arange(16)) & 1, 0x35, 0xD3),
                )


def test_cp_mask_without_cache_policy_copies_only_selected_bytes(tmp_path):
    inputs = {
        "source": np.arange(16, dtype=np.uint8),
        "byte_mask": np.array([0x55AA], np.int32),
        "destination": np.full(16, 99, np.uint8),
    }
    result = run_checked(bulk_s2g_cp_mask_without_cache, inputs, cache_dir=tmp_path)
    expected = np.array([99, 1, 99, 3, 99, 5, 99, 7, 8, 99, 10, 99, 12, 99, 14, 99], np.uint8)
    np.testing.assert_array_equal(result.outputs["destination"], expected)


def test_bulk_g2s_cluster_false_predicate_does_not_access_uninitialized_barrier(tmp_path):
    result = run_checked(
        bulk_g2s_cluster_static_false_predicate,
        {"source": np.arange(16, dtype=np.uint8)},
        cache_dir=tmp_path,
    )
    assert result.verdict == "clean"


def test_bulk_g2s_cta_two_issuing_lanes_credit_both_transfers(tmp_path):
    for kernel in (two_lane_bulk_g2s_cta_plain, two_lane_bulk_g2s_cta_ignore_oob):
        result = run_checked(
            kernel,
            {"source": np.arange(48, dtype=np.uint8), "output": np.zeros(48, dtype=np.uint8)},
            cache_dir=tmp_path,
        )
        np.testing.assert_array_equal(result.outputs["output"], np.r_[np.arange(32), np.zeros(16)])


def test_bulk_g2s_cluster_dynamic_predicate_transpiles(tmp_path):
    # Preserve the old compile-only probe; its barrier is not initialized for execution.
    module = numsim.transpile(bulk_g2s_cluster_dynamic_predicate, cache_dir=tmp_path)
    assert "tirx.ptx.cp_async_bulk_g2s_cluster" in call_op_names(module.spec.kernels[0])


def test_raw_bulk_s2c_preserves_mapped_remote_cta_ownership(tmp_path):
    from tirx_harness import racecheck, synccheck

    for checker in (synccheck, racecheck):
        checker(
            raw_bulk_s2c_uses_mapped_u32_addresses, {"output": np.zeros(16, np.uint8)}
        ).require_clean()
    module = numsim.transpile(raw_bulk_s2c_uses_mapped_u32_addresses, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(16, dtype=np.uint8)})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(19, 35, dtype=np.uint8))


def mapped_st_async_case(offset_form):
    """Use the same remote-store protocol for instruction and expression offsets."""
    kernel = raw_st_async_uses_mapped_u32_addresses
    if offset_form == "instruction":
        return kernel
    loads = {}
    variables = {}

    def record(node):
        if type(node).__name__ == "TensorLoad":
            loads[node.source.name] = node
        elif type(node).__name__ == "Var":
            variables[node.name] = node

    from tvm.ir import Expr

    structural_walk(kernel.body, (Expr, record))
    base = loads["remote_destination"]
    dynamic_offset = tirx.Cast("uint32", variables["lane"] + 16)
    expressions = {
        "constant": base + tirx.const(16, "uint32"),
        "dynamic": base + dynamic_offset,
        "wrapped": (base + tirx.const(0xFFFFFFFF, "uint32")) + tirx.const(17, "uint32"),
        "reversed": dynamic_offset + base,
        "subtracted": (base + tirx.const(32, "uint32")) - dynamic_offset,
        "multiply": base * tirx.const(2, "uint32"),
        "two_addresses": base + base,
        "reverse_subtract": dynamic_offset - base,
    }

    def replace(call):
        if call.op.name == "tirx.ptx.st_async_vec" and call.args[0].same_as(
            loads["remote_destination_hi"]
        ):
            return Call(
                call.op,
                [expressions[offset_form], *call.args[1:]],
                attrs=call.attrs,
                ty_args=call.ty_args,
                span=call.span,
                ret_ty=call.ty,
            )
        return call

    return kernel.with_body(structural_map(kernel.body, (Call, replace)))


@pytest.mark.parametrize(
    "offset_form", ["instruction", "constant", "dynamic", "reversed", "subtracted", "wrapped"]
)
def test_raw_st_async_preserves_mapped_remote_cta_ownership(tmp_path, offset_form):
    kernel = mapped_st_async_case(offset_form)
    inputs = {"output": np.zeros(8, dtype=np.uint32)}
    for checker in (synccheck, racecheck):
        checker(kernel, inputs).require_clean()
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(8, dtype=np.uint32)})

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.array(
            [
                0x10203040,
                0x50607080,
                0x90A0B0C0,
                0xD0E0F001,
                0x12345678,
                0x9ABCDEF0,
                0x0BADF00D,
                0xCAFEBABE,
            ],
            dtype=np.uint32,
        ),
    )


def test_raw_st_async_preserves_mapped_expression_byte_offset(tmp_path):
    module = numsim.transpile(raw_st_async_uses_mapped_u32_expression_offset, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(8, dtype=np.uint32)})

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.array(
            [
                0x10203040,
                0x50607080,
                0x90A0B0C0,
                0xD0E0F001,
                0x12345678,
                0x9ABCDEF0,
                0x0BADF00D,
                0xCAFEBABE,
            ],
            dtype=np.uint32,
        ),
    )


@pytest.mark.parametrize("checker", [synccheck, racecheck])
def test_raw_st_async_mapped_expression_offsets_are_disjoint(checker):
    report = checker(
        raw_st_async_uses_mapped_u32_expression_offset,
        {"output": np.zeros(8, dtype=np.uint32)},
    )

    assert report.verdict == "clean"
    assert report.findings == []


@pytest.mark.parametrize("offset_form", ["multiply", "two_addresses", "reverse_subtract"])
def test_mapped_shared_address_rejects_invalid_consumed_bits(tmp_path, offset_form):
    module = numsim.transpile(mapped_st_async_case(offset_form), cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match="CTA rank|shared address"):
        numsim.Engine().run(module, {"output": np.zeros(8, dtype=np.uint32)})


@pytest.mark.parametrize(("left", "right"), [(0, 4), (3, 0), (3, 5)])
def test_ignore_oob_copies_only_the_valid_middle(tmp_path, left: int, right: int):
    source = np.arange(16, dtype=np.uint8) + np.uint8(11)
    initial = np.full(16, np.uint8(0xD3), dtype=np.uint8)
    module = numsim.transpile(bulk_g2s_ignore_oob_dead_bytes, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "ignored": np.array([left, right], dtype=np.int32), "output": initial},
    )

    expected = initial.copy()
    expected[left : 16 - right] = source[left : 16 - right]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_ignore_oob_dead_bytes_are_zero_filled_and_require_review_if_read(tmp_path):
    module = numsim.transpile(bulk_g2s_ignore_oob_live_read, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": np.arange(16, dtype=np.uint8), "output": np.full(1, 0xFF, dtype=np.uint8)},
    )

    np.testing.assert_array_equal(result.outputs["output"], np.zeros(1, dtype=np.uint8))
    assert result.verdict == "review"
    assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}


@pytest.mark.parametrize("ignored", [(16, 0), (0, 16)])
def test_ignore_oob_rejects_counts_outside_ptx_range(tmp_path, ignored):
    module = numsim.transpile(bulk_g2s_ignore_oob_dead_bytes, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match="must be in 0..=15"):
        numsim.Engine().run(
            module,
            {
                "source": np.arange(16, dtype=np.uint8),
                "ignored": np.array(ignored, dtype=np.int32),
                "output": np.zeros(16, dtype=np.uint8),
            },
        )


def test_ignore_oob_allows_the_ignored_right_edge_outside_the_binding(tmp_path):
    source = np.arange(13, dtype=np.uint8) ^ np.uint8(0xA5)
    module = numsim.transpile(bulk_g2s_ignore_oob_right_outside_binding, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros(13, dtype=np.uint8)})
    np.testing.assert_array_equal(result.outputs["output"], source)


ST_BULK_SIZE_CASES = ((8, 1, 0), (0, 1, 0), (8, 1, 1), (-8, 0, 0))


def st_bulk_size_inputs(size, enabled, zero_first):
    inputs = {
        "output": np.zeros(32, np.uint8),
        "size": size,
        "enabled": enabled,
        "zero_first": zero_first,
    }
    expected = np.arange(1, 33, dtype=np.uint8)
    if size == 8 and enabled:
        expected[8 if zero_first else 0 : 24] = 0
    return inputs, expected


def test_st_bulk_size_is_evaluated_per_issuing_lane(tmp_path):
    from tirx_harness import racecheck, synccheck
    from tirx_harness.numsim.errors import NumSimExecutionError

    for dtype in ("uint64", "int64", "uint32", "int32"):
        kernel = lane_varying_st_bulk_size(dtype)
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        for size, enabled, zero_first in ST_BULK_SIZE_CASES:
            inputs, expected = st_bulk_size_inputs(size, enabled, zero_first)
            for checker in (synccheck, racecheck):
                checker(kernel, inputs).require_clean()
            result = numsim.Engine().run(module, inputs)
            np.testing.assert_array_equal(result.outputs["output"], expected)
        # High bits must survive transport: large 64-bit sizes are not small
        # valid 32-bit stores. Invalid and inactive lanes share the same path.
        for size in (-8, 1, 16777224, *((4294967304,) if dtype.endswith("64") else ())):
            inputs, _ = st_bulk_size_inputs(size, 1, 0)
            for checker in (synccheck, racecheck):
                report = checker(kernel, inputs)
                assert report.verdict == "error", report.format()
                assert any("st.bulk byte count" in f.message for f in report.findings)
            with pytest.raises(NumSimExecutionError, match="st.bulk byte count"):
                numsim.Engine().run(module, inputs)


def test_bulk_s2g_size_and_byte_mask_are_evaluated_per_issuing_lane(tmp_path):
    source = np.arange(64, dtype=np.uint8) + np.uint8(0x20)
    initial = np.full(64, np.uint8(0xE7), dtype=np.uint8)
    module = numsim.transpile(lane_varying_bulk_s2g_size_and_mask, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "destination": initial})

    expected = initial.copy()
    expected[0:8] = source[0:8]
    expected[40:48] = source[40:48]
    expected[56:64] = source[56:64]
    np.testing.assert_array_equal(result.outputs["destination"], expected)


def test_bulk_multicast_cta_mask_is_evaluated_per_issuing_lane(tmp_path):
    source = np.arange(32, dtype=np.uint8) ^ np.uint8(0x5A)
    module = numsim.transpile(lane_varying_bulk_multicast_cta_mask, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.zeros((2, 16), dtype=np.uint8)}
    )
    np.testing.assert_array_equal(result.outputs["output"], source.reshape(2, 16))


def test_bulk_g2s_cluster_accepts_static_true_predicate_as_unconditional(tmp_path):
    source = np.arange(16, dtype=np.uint8) ^ np.uint8(0xA5)
    module = numsim.transpile(bulk_g2s_cluster_static_true_predicate, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": np.zeros(16, dtype=np.uint8)})

    np.testing.assert_array_equal(result.outputs["output"], source)
    assert "tirx.ptx.cp_async_bulk_g2s_cluster" in call_op_names(module.spec.kernels[0])


def test_raw_bulk_prefetch_preserves_global_memory(tmp_path):
    source = np.arange(64, dtype=np.uint8) ^ np.uint8(0xA5)
    module = numsim.transpile(raw_bulk_prefetch, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "num_bytes": np.uint32(32), "output": np.zeros(32, dtype=np.uint8)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source[16:48])
    assert result.verdict == "clean"
    assert result.diagnostics == []
    assert "tirx.ptx.cp_async_bulk_prefetch" in call_op_names(module.spec.kernels[0])


@T.prim_func
def two_lane_bulk_g2s_cta_plain(source: T.Buffer((48,), "uint8"), output: T.Buffer((48,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((48,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 32)
    T.cuda.warp_sync()
    if lane < 2:
        base: T.int32 = lane * 16
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](
            shared.ptr_to([base]), source.ptr_to([base]), 16, barrier.ptr_to([0])
        )
    if lane == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(32):
            output[element] = shared[element]

@T.prim_func
def two_lane_bulk_g2s_cta_ignore_oob(
    source: T.Buffer((48,), "uint8"), output: T.Buffer((48,), "uint8")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((48,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 32)
    T.cuda.warp_sync()
    if lane < 2:
        base: T.int32 = lane * 16
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.ignore_oob"](
            shared.ptr_to([base]),
            source.ptr_to([base]),
            16,
            T.uint32(0),
            T.uint32(0),
            barrier.ptr_to([0]),
        )
    if lane == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for element in T.serial(32):
            output[element] = shared[element]
