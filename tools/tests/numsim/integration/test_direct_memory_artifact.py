from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tests.numsim.support.kernels import direct_cuda_ldg, direct_tvm_access_ptr_shared
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T


@T.prim_func
def store_through_read_only_access_ptr(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.st.global_.u32(output.access_ptr("r", offset=lane, extent=1), T.uint32(7))


@T.prim_func
def load_through_write_only_access_ptr(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.ld.global_.u32(output[lane], source.access_ptr("w", offset=lane, extent=1))


@T.prim_func
def load_after_access_ptr_extent(
    source: T.Buffer((64,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    one_element = source.access_ptr("r", offset=lane, extent=1)
    one_past = T.ptr_byte_offset(one_element, T.uint32(4), "uint32")
    T.ptx.ld.global_.u32(output[lane], one_past)


@T.prim_func
def nested_access_ptr_cannot_widen_permissions(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    read_only = output.access_ptr("r", offset=lane, extent=1)
    widened = T.tvm_access_ptr(T.type_annotation("uint32"), read_only, 0, 1, 2)
    T.ptx.st.global_.u32(widened, T.uint32(9))


@T.prim_func
def write_through_read_only_decl_buffer(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    alias_data: T.let[
        T.Var(name="read_only_alias", ty=PointerType(PrimType("uint32"), "global"))
    ] = source.access_ptr("r", ptr_type="uint32", offset=0, extent=32)
    alias = T.decl_buffer((32,), "uint32", data=alias_data, scope="global")
    alias[lane] = output[lane]


@T.prim_func
def permitted_decl_buffer_accesses(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    read_data: T.let[
        T.Var(name="permitted_read_alias", ty=PointerType(PrimType("uint32"), "global"))
    ] = source.access_ptr("r", ptr_type="uint32", offset=0, extent=32)
    write_data: T.let[
        T.Var(name="permitted_write_alias", ty=PointerType(PrimType("uint32"), "global"))
    ] = output.access_ptr("w", ptr_type="uint32", offset=0, extent=32)
    read_alias = T.decl_buffer((32,), "uint32", data=read_data, scope="global")
    write_alias = T.decl_buffer((32,), "uint32", data=write_data, scope="global")
    write_alias[lane] = read_alias[lane] + T.uint32(5)


@T.prim_func
def oversized_decl_buffer_from_access_ptr(source: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    alias_data: T.let[T.Var(name="short_alias", ty=PointerType(PrimType("uint32"), "global"))] = (
        source.access_ptr("r", ptr_type="uint32", offset=0, extent=1)
    )
    _alias = T.decl_buffer((2,), "uint32", data=alias_data, scope="global")


@T.prim_func
def raw_cp_async_zero_fill(
    source: T.Buffer((128,), "float32"), output: T.Buffer((128,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared")
    T.evaluate(
        T.ptx["cp.async.cg.shared.global.L2::128B"](
            T.address_of(shared[lane * 4]),
            T.address_of(source[lane * 4]),
            16,
            T.cast(T.if_then_else(lane < 16, 16, 0), "uint32"),
        )
    )
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


@T.prim_func
def pass_emitted_cp_async_raw(
    source: T.Buffer((128,), "float32"), output: T.Buffer((128,), "float32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "float32", scope="shared")
    T.ptx["cp.async.cg.shared.global"](shared.ptr_to([lane * 4]), source.ptr_to([lane * 4]), 16)
    T.ptx.cp.async_.commit_group()
    T.ptx.cp.async_.wait_group(0)
    for element in T.serial(4):
        output[lane * 4 + element] = shared[lane * 4 + element]


def test_cuda_ldg_reads_typed_global_memory(tmp_path):
    source_f32 = np.linspace(-2.0, 3.0, 32, dtype=np.float32)
    source_i32 = np.arange(32, dtype=np.int32) * 7 - 11
    output_f32 = np.zeros(32, dtype=np.float32)
    output_i32 = np.zeros(32, dtype=np.int32)

    spec = analyze(direct_cuda_ldg)
    assert spec.unsupported == ()
    module = numsim.transpile(direct_cuda_ldg, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source_f32": source_f32,
            "source_i32": source_i32,
            "output_f32": output_f32,
            "output_i32": output_i32,
        },
    )

    np.testing.assert_array_equal(result.outputs["output_f32"], source_f32[::-1])
    np.testing.assert_array_equal(result.outputs["output_i32"], source_i32[::-1])


def test_raw_cp_async_issues_nonbulk_group_for_zero_fill_lanes(tmp_path):
    source = np.arange(128, dtype=np.float32) + np.float32(0.25)
    output = np.full_like(source, np.float32(-1))

    module = numsim.transpile(raw_cp_async_zero_fill, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = source.copy()
    expected[64:] = np.float32(0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_pass_emitted_cp_async_raw_restores_element_offsets(tmp_path):
    source = np.arange(128, dtype=np.float32) + np.float32(0.375)
    output = np.zeros_like(source)

    module = numsim.transpile(pass_emitted_cp_async_raw, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_tvm_access_ptr_applies_element_offset_to_shared_pointer(tmp_path):
    output = np.zeros(128, dtype=np.uint32)

    spec = analyze(direct_tvm_access_ptr_shared)
    assert spec.unsupported == ()
    module = numsim.transpile(direct_tvm_access_ptr_shared, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.arange(128, dtype=np.uint32) * 3 + 1
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "physical_ptr_access_view(" in module.rust_source
    assert "access_ptr_byte_offsets" not in module.rust_source
    assert ".with_element_offset_extent(" not in module.rust_source


def test_decl_buffer_views_preserve_permitted_accesses(tmp_path):
    source = np.arange(32, dtype=np.uint32)
    module = numsim.transpile(permitted_decl_buffer_accesses, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros(32, dtype=np.uint32)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source + np.uint32(5))


@pytest.mark.parametrize(
    ("kernel", "arguments", "message"),
    [
        (
            store_through_read_only_access_ptr,
            {"output": np.zeros(32, dtype=np.uint32)},
            "non-writable physical pointer",
        ),
        (
            load_through_write_only_access_ptr,
            {
                "source": np.arange(32, dtype=np.uint32),
                "output": np.zeros(32, dtype=np.uint32),
            },
            "non-readable physical pointer",
        ),
        (
            nested_access_ptr_cannot_widen_permissions,
            {"output": np.zeros(32, dtype=np.uint32)},
            "cannot add write access",
        ),
        (
            write_through_read_only_decl_buffer,
            {
                "source": np.arange(32, dtype=np.uint32),
                "output": np.arange(32, dtype=np.uint32),
            },
            "non-writable DeclBuffer view",
        ),
        (
            oversized_decl_buffer_from_access_ptr,
            {"source": np.arange(32, dtype=np.uint32)},
            "outside tvm_access_ptr range",
        ),
    ],
)
def test_access_ptr_contract_fails_closed(kernel, arguments, message, tmp_path):
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match=message):
        numsim.Engine().run(module, arguments)


def test_integer_pointer_arithmetic_does_not_carry_access_ptr_extent(tmp_path):
    source = np.arange(64, dtype=np.uint32)
    module = numsim.transpile(load_after_access_ptr_extent, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {"source": source, "output": np.zeros(32, dtype=np.uint32)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source[1:33])
