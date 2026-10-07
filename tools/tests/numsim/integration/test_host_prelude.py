"""Observable NumSim behavior for canonical host-created TensorMaps."""

from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.errors import UnsupportedTIRxError


@T.prim_func
def host_encoded_tensor_map(output: T.Buffer((32,), "int32")):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "int32",
        1,
        output.data,
        32,
        32,
        1,
        0,
        0,
        0,
        0,
    )
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = lane


@T.prim_func
def host_encoded_tensor_map_store(output: T.Buffer((32,), "int32")):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "int32",
        1,
        output.data,
        32,
        32,
        1,
        0,
        0,
        0,
        0,
    )
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    shared[lane] = lane
    T.cuda.warp_sync()
    T.ptx.fence.proxy.async_.shared__cta()
    if lane == 0:
        T.evaluate(
            T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"](
                T.address_of(tensor_map), 0, T.address_of(shared[0])
            )
        )
        T.ptx.cp.async_.bulk.commit_group()
        T.ptx.cp.async_.bulk.wait_group.read(0)


@T.prim_func
def host_encoded_reinterpreted_tensor_map(
    source: T.Buffer((64,), "uint8"), output: T.Buffer((32,), "uint8")
):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "uint16",
        1,
        source.data,
        32,
        32,
        1,
        0,
        0,
        0,
        0,
    )
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = source[lane]


@T.prim_func
def host_encoded_offset_tensor_map(
    source: T.Buffer((32,), "uint8"), output: T.Buffer((16,), "uint8")
):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "uint8",
        1,
        T.handle_add_byte_offset(source.data, 16),
        16,
        16,
        1,
        0,
        0,
        0,
        0,
    )
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
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0]),
                T.address_of(tensor_map),
                0,
                T.address_of(barrier[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = shared[lane]


@T.prim_func
def host_encoded_fp4_tensor_map(
    source: T.Buffer((256,), "float4_e2m1fn"),
    output: T.Buffer((64,), "uint8"),
):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "float4_e2m1fn",
        1,
        source.data,
        256,
        128,
        1,
        0,
        0,
        0,
        0,
        13,
    )
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint8", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0]),
                T.address_of(tensor_map),
                0,
                T.address_of(barrier[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 64)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    output[lane * 2] = shared[lane * 2]
    output[lane * 2 + 1] = shared[lane * 2 + 1]


@T.prim_func
def host_encoded_dynamic_integer_tensor_map(
    source: T.Buffer((32,), "uint8"),
    output: T.Buffer((16,), "uint8"),
    delta: T.int32,
):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "uint8",
        1,
        source.data,
        T.truncdiv(delta, T.int32(2)) + T.int32(35),
        T.floordiv(delta, T.int32(2)) + T.int32(20),
        1,
        0,
        0,
        0,
        0,
    )
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
        T.evaluate(
            T.ptx[
                "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
            ](
                T.address_of(shared[0]),
                T.address_of(tensor_map),
                16,
                T.address_of(barrier[0]),
            )
        )
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(barrier[0]), 16)
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.warp_sync()
    if lane < 16:
        output[lane] = shared[lane]


@T.prim_func
def host_encoded_unregistered_integer_tensor_map(source: T.Buffer((32,), "uint8"), delta: T.int32):
    tensor_map: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
    T.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        tensor_map,
        "uint8",
        1,
        source.data,
        T.bitwise_and(delta, T.int32(31)) + T.int32(1),
        16,
        1,
        0,
        0,
        0,
        0,
    )
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])


@T.prim_func
def direct_cuda_launch(output: T.Buffer((4, 64), "int32")):
    cta_in_cluster = T.launch_thread("clusterCtaIdx.x", 2)
    global_cta = T.launch_thread("blockIdx.x", 4)
    thread = T.launch_thread("threadIdx.x", 64)
    output[global_cta, thread] = global_cta * 1000 + cta_in_cluster * 100 + thread


def test_host_encoded_tensor_map_uses_its_typed_buffer_argument(tmp_path):
    module = numsim.transpile(host_encoded_tensor_map, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(32, dtype=np.int32)})

    np.testing.assert_array_equal(result.outputs["output"], np.arange(32, dtype=np.int32))


def test_explicit_tensor_map_override_updates_its_distinct_backing(tmp_path):
    module = numsim.transpile(host_encoded_tensor_map_store, cache_dir=tmp_path)
    declared_base = np.zeros(32, dtype=np.int32)
    override_base = np.zeros(32, dtype=np.int32)
    override = numsim.TensorMap(
        base=override_base,
        global_shape=(32,),
        global_strides=(),
        box_shape=(32,),
        element_strides=(1,),
    ).numpy()

    result = numsim.Engine().run(
        module,
        {"output": declared_base, "tensor_map": override},
        outputs=("tensor_map",),
    )

    np.testing.assert_array_equal(result.outputs["tensor_map"], np.arange(32, dtype=np.int32))
    np.testing.assert_array_equal(declared_base, np.zeros(32, dtype=np.int32))


def test_host_encoded_tensor_map_is_accepted_by_native_checkers():
    inputs = {"output": np.zeros(32, dtype=np.int32)}

    sync_report = synccheck(host_encoded_tensor_map, inputs=inputs)
    race_report = racecheck(host_encoded_tensor_map, inputs=inputs)

    assert sync_report.verdict == "clean", sync_report.format()
    assert race_report.verdict == "clean", race_report.format()


def test_host_encoded_tensor_map_can_reinterpret_byte_storage(tmp_path):
    module = numsim.transpile(host_encoded_reinterpreted_tensor_map, cache_dir=tmp_path)
    source = np.arange(64, dtype=np.uint8)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": np.zeros(32, dtype=np.uint8),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], source[:32])


def test_host_encoded_tensor_map_base_offset_selects_the_backing_suffix(tmp_path):
    module = numsim.transpile(host_encoded_offset_tensor_map, cache_dir=tmp_path)
    source = np.arange(32, dtype=np.uint8)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": np.zeros(16, dtype=np.uint8),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], source[16:])


def test_host_encoded_fp4_tensor_map_preserves_packed_bytes(tmp_path):
    module = numsim.transpile(host_encoded_fp4_tensor_map, cache_dir=tmp_path)
    source = (np.arange(128, dtype=np.uint8) * np.uint8(29) + np.uint8(7)).astype(np.uint8)
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "output": np.zeros(64, dtype=np.uint8),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], source[:64])


def test_dynamic_tensor_map_expressions_run_in_the_loaded_artifact_prologue(tmp_path):
    module = numsim.transpile(host_encoded_dynamic_integer_tensor_map, cache_dir=tmp_path)

    # Loading imports and validates the Rust extension without any runtime
    # bindings.  The implicit descriptor is constructed only by ``run`` below.
    module.load()
    source = np.arange(1, 33, dtype=np.uint8)
    inputs = {
        "source": source,
        "output": np.zeros(16, dtype=np.uint8),
        "delta": np.int32(-7),
    }
    result = numsim.Engine().run(module, inputs)

    # The independent descriptor oracles are global_shape=32 and box_shape=16:
    # -7/2 truncates to -3 while floor division is -4. Reading at coordinate
    # 16 makes the distinction observable: an incorrect global_shape=31 would
    # zero-fill the final element as out of bounds.
    trunc_oracle = -(abs(-7) // abs(2))
    assert trunc_oracle == -3
    assert int(np.floor_divide(np.int64(-7), np.int64(2))) == -4
    np.testing.assert_array_equal(result.outputs["output"], source[16:])


def test_dynamic_tensor_map_prologue_is_shared_by_native_checkers():
    inputs = {
        "source": np.arange(1, 33, dtype=np.uint8),
        "output": np.zeros(16, dtype=np.uint8),
        "delta": np.int32(-7),
    }

    sync_report = synccheck(host_encoded_dynamic_integer_tensor_map, inputs=inputs)
    race_report = racecheck(host_encoded_dynamic_integer_tensor_map, inputs=inputs)

    assert sync_report.verdict == "clean", sync_report.format()
    assert race_report.verdict == "clean", race_report.format()


def test_host_tensor_map_integer_expressions_fail_closed_on_unregistered_nodes(tmp_path):
    with pytest.raises(UnsupportedTIRxError, match="unsupported integer operation BitwiseAnd"):
        numsim.transpile(host_encoded_unregistered_integer_tensor_map, cache_dir=tmp_path)


def test_direct_cuda_launch_coordinates_match_numsim_topology(tmp_path):
    module = numsim.transpile(direct_cuda_launch, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros((4, 64), dtype=np.int32)})

    global_cta = np.arange(4, dtype=np.int32)[:, None]
    thread = np.arange(64, dtype=np.int32)[None, :]
    expected = global_cta * 1000 + (global_cta % 2) * 100 + thread
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_direct_cuda_launch_is_accepted_by_native_checkers():
    inputs = {"output": np.zeros((4, 64), dtype=np.int32)}

    sync_report = synccheck(direct_cuda_launch, inputs=inputs)
    race_report = racecheck(direct_cuda_launch, inputs=inputs)

    assert sync_report.verdict == "clean", sync_report.format()
    assert race_report.verdict == "clean", race_report.format()
