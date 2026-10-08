from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.runtime.test_tile_codegen import _typed_tma_reduce_kernel
from tests.numsim.support.manifest import call_op_names
from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tvm.script import tirx as T


_BULK_REDUCE_ADD_F32 = "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"


@T.prim_func
def bulk_reduce_add_f32_read_then_full_wait(
    source: T.Buffer((8,), "float32"),
    destination: T.Buffer((4,), "float32"),
    observed: T.Buffer((2,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float32", scope="shared", align=16)
    if lane == 0:
        for index in T.serial(8):
            shared[index] = source[index]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx[_BULK_REDUCE_ADD_F32](destination.ptr_to([0]), shared.ptr_to([0]), T.uint32(16))
        T.ptx[_BULK_REDUCE_ADD_F32](destination.ptr_to([0]), shared.ptr_to([4]), T.uint32(16))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        observed[0] = destination[0]
        for index in T.serial(8):
            shared[index] = T.float32(1000)
        T.ptx.cp.async_.bulk.wait_group(0)
        observed[1] = destination[0]


@T.prim_func
def bulk_reduce_add_f32_read_wait_then_exit(
    source: T.Buffer((4,), "float32"), destination: T.Buffer((4,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared", align=16)
    if lane == 0:
        for index in T.serial(4):
            shared[index] = source[index]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx[_BULK_REDUCE_ADD_F32](destination.ptr_to([0]), shared.ptr_to([0]), T.uint32(16))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        for index in T.serial(4):
            shared[index] = T.float32(1000)


@T.prim_func
def bulk_reduce_add_f32_two_ctas(destination: T.Buffer((4,), "float32")):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared", align=16)
    if lane == 0:
        for index in T.serial(4):
            shared[index] = T.cast((cta + 1) * (index + 1), "float32")
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx[_BULK_REDUCE_ADD_F32](destination.ptr_to([0]), shared.ptr_to([0]), T.uint32(16))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def bulk_reduce_add_f32_partially_overlapping_ctas(
    destination: T.Buffer((8,), "float32"),
):
    T.device_entry()
    cta = T.cta_id([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float32", scope="shared", align=16)
    if lane == 0:
        for index in T.serial(8):
            shared[index] = T.float32(1)
        T.ptx.fence.proxy.async_.shared__cta()
        destination_offset: T.int32 = cta * 4
        num_bytes: T.uint32 = T.Select(cta == 0, T.uint32(32), T.uint32(16))
        T.ptx[_BULK_REDUCE_ADD_F32](
            destination.ptr_to([destination_offset]),
            shared.ptr_to([0]),
            num_bytes,
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def bulk_reduce_add_f32_destination_update_before_full_wait(
    source: T.Buffer((4,), "float32"), destination: T.Buffer((4,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared", align=16)
    if lane == 0:
        for index in T.serial(4):
            shared[index] = source[index]
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx[_BULK_REDUCE_ADD_F32](destination.ptr_to([0]), shared.ptr_to([0]), T.uint32(16))
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)
        for index in T.serial(4):
            T.ptx.red.relaxed.sys.global_.add.f32(destination.ptr_to([index]), T.float32(100))
            shared[index] = T.float32(1000)
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def bulk_reduce_add_f32_runtime_layout(
    source_offset: T.int32,
    destination_offset: T.int32,
    num_bytes: T.int32,
    destination: T.Buffer((8,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((8,), "float32", scope="shared", align=16)
    if lane == 0:
        for index in T.serial(8):
            shared[index] = T.cast(index + 1, "float32")
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx[_BULK_REDUCE_ADD_F32](
            destination.ptr_to([destination_offset]),
            shared.ptr_to([source_offset]),
            T.cast(num_bytes, "uint32"),
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group(0)


@T.prim_func
def bulk_reduce_add_u32_analysis_probe(destination: T.Buffer((4,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "uint32", scope="shared", align=16)
    if lane == 0:
        T.ptx["cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32"](
            destination.ptr_to([0]), shared.ptr_to([0]), T.uint32(16)
        )


@T.prim_func
def bulk_reduce_relaxed_gpu_f32_analysis_probe(destination: T.Buffer((4,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4,), "float32", scope="shared", align=16)
    if lane == 0:
        T.ptx["cp.reduce.async.bulk.relaxed.gpu.global.shared::cta.bulk_group.add.f32"](
            destination.ptr_to([0]), shared.ptr_to([0]), T.uint32(16)
        )


def test_bulk_reduce_former_rejects_preserve_type_and_scope(tmp_path):
    # Original compile-only probes: shared data is intentionally uninitialized.
    for kernel, dtype, scope in (
        (bulk_reduce_add_u32_analysis_probe, "U32", "Sys"),
        (bulk_reduce_relaxed_gpu_f32_analysis_probe, "F32", "Gpu"),
    ):
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        assert (
            f"BulkS2gReduce<v2::reg::variant::{dtype}, v2::async_copy::variant::ReduceAdd, "
            f"v2::mem::variant::{scope}>"
        ) in module.rust_source


def test_bulk_reduce_is_atomic_captures_source_and_publishes_only_at_full_wait(tmp_path):
    source = np.arange(1, 9, dtype=np.float32)
    destination = np.full(4, np.float32(10), dtype=np.float32)
    initial_destination = destination.copy()
    module = numsim.transpile(bulk_reduce_add_f32_read_then_full_wait, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {"source": source, "destination": destination, "observed": np.zeros(2, np.float32)},
    )

    np.testing.assert_array_equal(
        result.outputs["destination"], initial_destination + source[:4] + source[4:]
    )
    np.testing.assert_array_equal(result.outputs["observed"], np.array([10, 16], np.float32))
    assert result.verdict == "clean"
    assert result.diagnostics == []
    assert "tirx.ptx.cp_reduce_async_bulk_s2g" in call_op_names(module.spec.kernels[0])


def test_bulk_reduce_preserves_f32_subnormals(tmp_path):
    smallest_normal = np.float32(np.finfo(np.float32).tiny)
    smallest_subnormal = np.nextafter(np.float32(0), np.float32(1), dtype=np.float32)
    source = np.array(
        [-smallest_subnormal, smallest_subnormal, smallest_subnormal, -smallest_subnormal],
        dtype=np.float32,
    )
    destination = np.array(
        [smallest_normal, -smallest_normal, smallest_subnormal, -smallest_subnormal],
        dtype=np.float32,
    )
    initial_destination = destination.copy()
    module = numsim.transpile(bulk_reduce_add_f32_read_wait_then_exit, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"source": source, "destination": destination})

    expected = np.array([0x007FFFFF, 0x807FFFFF, 2, 0x80000002], dtype=np.uint32)
    np.testing.assert_array_equal(
        result.outputs["destination"].view(np.uint32), expected
    )

    # Tensor and non-tensor F32 reductions share the same non-flushing addition.
    tensor_module = numsim.transpile(_typed_tma_reduce_kernel("float32", "add"), cache_dir=tmp_path)
    tensor_result = numsim.Engine().run(
        tensor_module, {"source": source, "output": initial_destination}
    )
    np.testing.assert_array_equal(
        tensor_result.outputs["output"].view(np.uint32),
        expected,
    )


def test_bulk_reduce_is_atomic_across_ctas_without_lost_updates(tmp_path):
    destination = np.arange(10, 14, dtype=np.float32)
    initial_destination = destination.copy()
    module = numsim.transpile(bulk_reduce_add_f32_two_ctas, cache_dir=tmp_path)

    result = numsim.Engine().run(module, {"destination": destination})

    expected = initial_destination + np.arange(1, 5, dtype=np.float32) * np.float32(3)
    np.testing.assert_array_equal(result.outputs["destination"], expected)
    assert result.verdict == "clean"
    assert result.diagnostics == []


def test_bulk_reduce_partial_overlap_is_elementwise_atomic(tmp_path):
    report = racecheck(
        bulk_reduce_add_f32_partially_overlapping_ctas,
        inputs={"destination": np.zeros(8, dtype=np.float32)},
        cache_dir=tmp_path,
        max_workers=2,
    )

    assert report.native_payload["execution_error"] is None, report.native_payload
    assert report.native_payload.get("incomplete", []) == [], report.native_payload
    assert report.verdict == "clean", report.format()
    assert report.findings == []


def test_bulk_reduce_reads_destination_at_full_completion(tmp_path):
    source = np.arange(1, 5, dtype=np.float32)
    destination = np.arange(10, 14, dtype=np.float32)
    initial_destination = destination.copy()
    module = numsim.transpile(
        bulk_reduce_add_f32_destination_update_before_full_wait,
        cache_dir=tmp_path,
    )

    result = numsim.Engine().run(module, {"source": source, "destination": destination})

    np.testing.assert_array_equal(
        result.outputs["destination"], initial_destination + np.float32(100) + source
    )
    assert result.verdict == "clean"
    assert result.diagnostics == []


@pytest.mark.parametrize(
    ("source_offset", "destination_offset", "num_bytes", "message"),
    [
        (1, 0, 16, "16-byte aligned source and destination"),
        (0, 1, 16, "16-byte aligned source and destination"),
        (0, 0, 12, "positive multiple of 16"),
    ],
)
def test_bulk_reduce_rejects_invalid_runtime_layout(
    tmp_path, source_offset: int, destination_offset: int, num_bytes: int, message: str
):
    module = numsim.transpile(bulk_reduce_add_f32_runtime_layout, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match=message):
        numsim.Engine().run(
            module,
            {
                "source_offset": np.int32(source_offset),
                "destination_offset": np.int32(destination_offset),
                "num_bytes": np.int32(num_bytes),
                "destination": np.zeros(8, np.float32),
            },
        )
