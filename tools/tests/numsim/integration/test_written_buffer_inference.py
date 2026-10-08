from __future__ import annotations

import numpy as np
import pytest
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.errors import NumSimExecutionError
from tirx_harness.numsim.transpiler.frontend import analyze, verify


@T.prim_func
def raw_store_through_dynamic_alias(
    source: T.Buffer((32,), "uint32"), destination: T.Buffer((33,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    alias_data: T.let[T.Var(name="offset_destination", ty=PointerType(PrimType("uint32")))] = (
        T.ptr_byte_offset(T.reinterpret("handle", destination.ptr_to([0])), T.uint32(4), "uint32")
    )
    alias = T.decl_buffer((32,), "uint32", data=alias_data, scope="global")
    T.ptx.st.global_.u32(
        alias.access_ptr("w", offset=lane),
        source[lane] + T.uint32(1),
    )


@T.prim_func
def raw_store_through_access_ptr(
    source: T.Buffer((32,), "uint32"), destination: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.st.global_.u32(destination.access_ptr("w", offset=lane), source[lane])


@T.prim_func
def static_alias_store(storage: T.Buffer((128,), "uint8"), source: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    alias = T.decl_buffer((32,), "float32", data=storage.data, scope="global")
    alias[lane] = source[lane]


@T.prim_func
def atomic_through_pointer_offset(
    counter: T.Buffer((2,), "uint32"), old_value: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        target = T.ptr_byte_offset(counter.ptr_to([0]), T.uint32(4), "uint32")
        T.ptx.atom.release.gpu.global_.add.u32(
            old_value[0],
            target,
            T.uint32(3),
        )


@T.prim_func
def unresolved_raw_store(pointer_bits: T.Buffer((1,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.ptx.st.global_.u32(T.reinterpret("handle", pointer_bits[0]), T.uint32(1))


@T.prim_func
def raw_tcgen_descriptor_store_to_global_output(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(output[0]),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=128,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )


def test_raw_store_pointer_forms_reach_the_host_output(tmp_path):
    source = np.arange(32, dtype=np.uint32)
    destination = np.zeros(33, dtype=np.uint32)

    module = numsim.transpile(raw_store_through_dynamic_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "destination": destination})

    assert set(result.outputs) == {"source", "destination"}
    np.testing.assert_array_equal(result.outputs["destination"][1:], source + np.uint32(1))
    assert result.outputs["destination"][0] == 0


def test_structured_alias_store_maps_back_to_parameter_owner(tmp_path):
    source = np.linspace(-3, 5, 32, dtype=np.float32)
    storage = np.zeros(128, dtype=np.uint8)

    module = numsim.transpile(static_alias_store, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"storage": storage, "source": source})

    assert set(result.outputs) == {"storage", "source"}
    np.testing.assert_array_equal(result.outputs["storage"], source.view(np.uint8))


def test_tvm_access_ptr_destination_maps_to_parameter_owner(tmp_path):
    source = np.arange(32, dtype=np.uint32)
    module = numsim.transpile(raw_store_through_access_ptr, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "destination": np.zeros(32, dtype=np.uint32)}
    )

    assert set(result.outputs) == {"source", "destination"}
    np.testing.assert_array_equal(result.outputs["destination"], source)


def test_atomic_pointer_offset_mutates_the_parameter(tmp_path):
    module = numsim.transpile(atomic_through_pointer_offset, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"counter": np.array([5, 7], dtype=np.uint32), "old_value": np.zeros(1, dtype=np.uint32)},
    )

    def check() -> None:
        assert set(result.outputs) == {"counter", "old_value"}
        np.testing.assert_array_equal(result.outputs["counter"], np.array([5, 10], dtype=np.uint32))
        np.testing.assert_array_equal(result.outputs["old_value"], np.array([7], dtype=np.uint32))

    check()


@pytest.mark.parametrize("frozen", [False, True])
def test_selected_outputs_preserve_all_host_writes(frozen, tmp_path):
    counter = np.array([5, 7], dtype=np.uint32)
    old_value = np.zeros(1, dtype=np.uint32)
    inputs = {"counter": counter, "old_value": old_value}
    if frozen:
        case = numsim.NumSimCase(
            kernel=atomic_through_pointer_offset,
            args=inputs,
            outputs=("old_value",),
            reference=lambda: {"old_value": np.array([7], dtype=np.uint32)},
        )
        numsim.run_case(case).require_ok()
    else:
        module = numsim.transpile(atomic_through_pointer_offset, cache_dir=tmp_path)
        result = numsim.Engine().run(module, inputs, outputs=("old_value",))
        assert set(result.outputs) == {"old_value"}
    np.testing.assert_array_equal(old_value, [7])
    np.testing.assert_array_equal(counter, [5, 10])


@pytest.mark.parametrize("address_kind", ["bound", "null", "unbound"])
def test_dynamic_raw_write_requires_bound_address(address_kind, tmp_path):
    spec = analyze(unresolved_raw_store)
    verify(spec)
    bits = np.zeros(1, np.uint64)
    bits[0] = {"bound": bits.ctypes.data, "null": 0, "unbound": 0x1000}[address_kind]
    inputs = {"pointer_bits": bits}
    for checker in (synccheck, racecheck):
        report = checker(unresolved_raw_store, inputs)
        if address_kind == "bound":
            report.require_clean()
        else:
            assert report.verdict == ("error" if address_kind == "null" else "incomplete")
            message = "null pointer" if address_kind == "null" else "integer_address_without_binding"
            assert message in report.format()
    module = numsim.transpile(unresolved_raw_store, cache_dir=tmp_path)
    if address_kind == "bound":
        result = numsim.Engine().run(module, inputs)
        assert set(result.outputs) == {"pointer_bits"}
        expected = (int(bits[0]) & 0xFFFFFFFF00000000) | 1
        np.testing.assert_array_equal(result.outputs["pointer_bits"], [expected])
    else:
        with pytest.raises(
            NumSimExecutionError, match="null pointer|integer_address_without_binding"
        ):
            numsim.Engine().run(module, inputs)
