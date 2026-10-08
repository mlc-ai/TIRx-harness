from __future__ import annotations

import numpy as np
import pytest

from tirx_harness.numsim.bindings import prepare_bindings
from tirx_harness.numsim.cases import TensorMap
from tirx_harness.numsim.errors import NumSimExecutionError


def _tensor_map(base: np.ndarray) -> np.ndarray:
    return TensorMap(
        base,
        global_shape=(4, 3),
        global_strides=(16,),
        box_shape=(4, 3),
        element_strides=(1, 1),
    ).numpy()


def test_ndarray_binding_preserves_numpy_aliases() -> None:
    storage = np.arange(12, dtype=np.int32)
    prepared = prepare_bindings({"whole": storage, "tail": storage[4:]})

    assert prepared.buffers["whole"].allocation == prepared.buffers["tail"].allocation
    assert prepared.buffers["tail"].data_offset == 16


def test_ndarray_binding_preserves_strided_views() -> None:
    storage = np.arange(12, dtype=np.int32).reshape(3, 4)
    view = storage[:, ::2]
    prepared = prepare_bindings({"view": view})

    assert prepared.buffers["view"].shape == (3, 2)
    assert prepared.buffers["view"].byte_strides == view.strides
    np.testing.assert_array_equal(prepared.logical_snapshot("view"), view)


def test_kernel_abi_supplies_logical_bfloat16_dtype() -> None:
    bits = np.array([0x3F80, 0xC000], dtype=np.uint16)
    prepared = prepare_bindings({"values": bits}, expected_buffer_dtypes={"values": "bfloat16"})

    assert prepared.buffers["values"].dtype == "bfloat16"
    np.testing.assert_array_equal(prepared.logical_snapshot("values"), [1.0, -2.0])


def test_packed_float4_uses_uint8_carrier_and_kernel_dtype() -> None:
    packed = np.array([0x42, 0x65], dtype=np.uint8)
    prepared = prepare_bindings(
        {"values": packed}, expected_buffer_dtypes={"values": "float4_e2m1fn"}
    )

    assert prepared.buffers["values"].dtype == "float4_e2m1fn"
    np.testing.assert_array_equal(prepared.logical_snapshot("values"), [1.0, 2.0, 3.0, 4.0])


def test_tensor_map_numpy_is_copyable_descriptor_state(tmp_path) -> None:
    base_a = np.arange(12, dtype=np.float32).reshape(3, 4)
    base_b = np.arange(12, dtype=np.float32).reshape(3, 4) + 100
    storage = np.zeros(256, dtype=np.uint8)
    storage[:128] = _tensor_map(base_a)
    storage[128:] = _tensor_map(base_b)

    prepared = prepare_bindings({"storage": storage})

    assert len(prepared.descriptor_allocations) == 3
    payload = prepared.to_payload()
    assert payload["output_allocations"] == sorted(prepared.descriptor_allocations)

    import tvm
    from tvm.script import tirx as T
    from tirx_harness import numsim, racecheck, synccheck

    disable = tvm.script.from_source('''@T.prim_func
def disable(descriptor: T.Buffer((128,), "uint8")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx["tensormap_replace.tile.swizzle_mode.global.b1024.b32"](descriptor.ptr_to([0]), 0)
        T.ptx.fence.proxy.tensormap__generic.release.gpu()
''', {"T": T})
    module = numsim.transpile(disable, cache_dir=tmp_path)
    for swizzle in ("128B", "128B_ATOM_32B", "128B_ATOM_32B_FLIP_8B", "128B_ATOM_64B"):
        descriptor = TensorMap(
            base_a, global_shape=(4, 3), global_strides=(16,), box_shape=(4, 3),
            element_strides=(1, 1), swizzle=swizzle,
        ).numpy()
        tag = descriptor[63]
        inputs = {"descriptor": descriptor}
        numsim.Engine().run(module, inputs, outputs=("descriptor",))
        assert descriptor[63] == tag  # The independent atomicity field survives.
        rebound = prepare_bindings(inputs, expected_tensor_map_names={"descriptor"})
        assert len(rebound.descriptor_allocations) == 2
        for checker in (synccheck, racecheck):
            checker(disable, inputs).require_clean()


def test_tensor_map_numpy_reuses_immutable_numpy_owner() -> None:
    owner = bytes(range(64))
    base = np.frombuffer(owner, dtype=np.uint8)
    descriptor = TensorMap(
        base,
        global_shape=(64,),
        global_strides=(),
        box_shape=(16,),
        element_strides=(1,),
    ).numpy()

    prepared = prepare_bindings({"input_map": descriptor})

    assert any(allocation.data is owner for allocation in prepared.allocations)


def test_direct_tensor_map_is_an_ndarray_input() -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    descriptor = _tensor_map(base)
    prepared = prepare_bindings({"input_map": descriptor}, expected_tensor_map_names={"input_map"})

    assert descriptor.dtype == np.uint8
    assert descriptor.shape == (128,)
    assert prepared.tensor_map_outputs["input_map"].global_shape == (4, 3)


def test_direct_tensor_map_rejects_non_descriptor_array() -> None:
    with pytest.raises(NumSimExecutionError, match="valid descriptor"):
        prepare_bindings(
            {"input_map": np.zeros(128, dtype=np.uint8)},
            expected_tensor_map_names={"input_map"},
        )


def test_plain_ndarray_payload_uses_kernel_buffer_metadata() -> None:
    payload = prepare_bindings({"values": np.arange(4, dtype=np.int32)}).to_payload()

    assert payload["buffers"]["values"] == {
        "allocation": 0,
        "data_offset": 0,
        "dtype": "int32",
        "itemsize": 4,
        "shape": [4],
        "byte_strides": [4],
    }
    assert payload["output_allocations"] == [0]


def test_tensor_map_identity_normalizes_host_addresses() -> None:
    def prepare(*, swizzle: str | None = None):
        base = np.arange(32, dtype=np.float16).reshape(4, 8)
        descriptor = TensorMap(
            base,
            global_shape=(8, 4),
            global_strides=(16,),
            box_shape=(8, 2),
            element_strides=(1, 1),
            swizzle=swizzle,
        ).numpy()
        return prepare_bindings(
            {"descriptor": descriptor}, expected_tensor_map_names={"descriptor"}
        )

    baseline = prepare().identity_payload()

    assert prepare().identity_payload() == baseline
    assert prepare(swizzle="32B").identity_payload() != baseline


def test_magic_byte_in_plain_array_is_not_a_tensor_map() -> None:
    values = np.zeros(128, dtype=np.uint8)
    values[63] = 0xA7

    prepared = prepare_bindings({"values": values})

    assert prepared.descriptor_allocations == frozenset()
    assert prepared.buffers["values"].dtype == "uint8"


def test_noncanonical_reserved_bytes_are_not_a_tensor_map() -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    images = [_tensor_map(base).copy() for _ in range(2)]
    images[0][64] = 1
    # 64B swizzle with 32B atomicity has no canonical host encoding. It
    # must not be discovered and silently reconstructed as no swizzle.
    images[1][60] |= 2 << 2
    images[1][63] |= 1 << 3

    for values in images:
        prepared = prepare_bindings({"values": values})

        assert prepared.descriptor_allocations == frozenset()
        assert prepared.buffers["values"].dtype == "uint8"
