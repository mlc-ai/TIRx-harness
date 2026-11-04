from __future__ import annotations

import re

import numpy as np
import pytest

from tests.numsim.support.kernels import tmem_d_alias_per_cta
from tests.numsim.support.manifest import emitted_module
from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tvm_tirx.ir import PtrType
from tvm_tirx.ir.type import PrimType
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import ComposeLayout, S, TileLayout

_PADDED_TILE = TileLayout(S[(2, 2) : (4, 1)])
_PADDED_COMPOSE = ComposeLayout(0, 0, 0, _PADDED_TILE)


@T.function
def in_bounds_internal_alias(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    storage = T.alloc_shared((8,), "uint32")
    alias = T.decl_tensor(
        T.ptr_byte_offset(storage.data, 16, ty=T.Ptr("uint32", "shared")),
        ty_args=[T.Tensor((4,), "uint32", scope="shared")],
    )
    alias[0] = 1
    output[0] = alias[0]


@T.function
def alias_extends_internal_owner(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    storage = T.alloc_shared((4,), "uint32")
    alias = T.decl_tensor(storage.data, ty_args=[T.Tensor((8,), "uint32", scope="shared")])
    alias[7] = 1
    output[0] = alias[7]


@T.function
def dead_alias_extends_internal_owner(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    storage = T.alloc_shared((4,), "uint32")
    _dead = T.decl_tensor(storage.data, ty_args=[T.Tensor((8,), "uint32", scope="shared")])
    output[0] = 17


@T.function
def loaded_alias_extends_internal_owner(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    storage = T.alloc_shared((4,), "uint32")
    alias = T.decl_tensor(storage.data, ty_args=[T.Tensor((8,), "uint32", scope="shared")])
    storage[0] = 29
    output[0] = alias[0]


@T.function
def evaluated_buffer_data_projection_is_pure(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    T.evaluate(output.data)
    output[0] = 23


@T.function
def pointer_origin_disagrees_with_declared_scope(
    source: T.Tensor((4,), "uint32"), output: T.Tensor((1,), "uint32")
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    pointer: T.let[
        T.Var(
            name="global_pointer",
            ty=PtrType(PrimType("uint32"), "shared"),
        )
    ] = T.reinterpret(source.ptr_to([0]), ty=PtrType(PrimType("uint32"), "shared"))
    alias = T.decl_tensor(pointer, ty_args=[T.Tensor((4,), "uint32", scope="shared")])
    output[0] = alias[0]


@T.function
def mapa_shared_address_in_generic_handle(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, cluster=1))
    shared = T.alloc_shared((1,), "uint64")
    remote_bits: T.uint64
    T.ptx.mapa.u64(remote_bits, T.address_of(shared[0]), T.uint32(0))
    remote_pointer: T.let[
        T.Var(name="remote_pointer", ty=PtrType(PrimType("uint64"), "global"))
    ] = T.reinterpret(remote_bits, ty=PtrType(PrimType("uint64"), "global"))
    remote = T.decl_tensor(remote_pointer, ty_args=[T.Tensor((1,), "uint64", scope="global")])
    output[0] = T.cast(remote[0], "uint32")


@T.function
def mapa_source_through_nominal_global_declbuffer(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, cluster=1))
    shared = T.alloc_shared((8,), "uint64")
    alias_pointer: T.let[
        T.Var(name="nominal_global_alias_pointer", ty=PtrType(PrimType("uint64"), "global"))
    ] = T.reinterpret(shared.ptr_to([0]), ty=PtrType(PrimType("uint64"), "global"))
    alias = T.decl_tensor(alias_pointer, ty_args=[T.Tensor((8,), "uint64", scope="global")])
    mapped = T.local_scalar("uint64")
    T.ptx.mapa.shared__cluster.u64(mapped, alias.ptr_to([0]), T.uint32(0))
    output[0] = 0


@T.function
def zero_extent_pool_uses_concrete_view_span(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    storage = T.alloc_tensor(ty_args=[T.Tensor((0,), "uint8", scope="shared.dyn")])
    alias = T.decl_tensor(storage.data, ty_args=[T.Tensor((8,), "uint32", scope="shared.dyn")])
    alias[7] = 1
    output[0] = alias[7]


@T.function
def zero_extent_pool_without_span_evidence(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    T.alloc_tensor(ty_args=[T.Tensor((0,), "uint8", scope="shared.dyn")])
    output[0] = 0


@T.function
def repeated_tile_owner(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    storage = T.alloc_shared((2, 3), "uint32", layout=_PADDED_TILE)
    alias = T.decl_tensor(storage.data, ty_args=[T.Tensor((10,), "uint32", scope="shared")])
    alias[9] = 7
    output[0] = alias[9]


@T.function
def repeated_compose_owner(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    storage = T.alloc_shared((2, 4), "uint32", layout=_PADDED_COMPOSE)
    alias = T.decl_tensor(storage.data, ty_args=[T.Tensor((14,), "uint32", scope="shared")])
    alias[13] = 9
    output[0] = alias[13]


@T.function
def odd_float4_owner(output: T.Tensor((1,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    storage = T.alloc_shared((3,), "float4_e2m1fn")
    alias = T.decl_tensor(storage.data, ty_args=[T.Tensor((2,), "uint8", scope="shared")])
    output[0] = alias[1]


@T.function
def pointer_origin_through_pure_if_then_else(
    left: T.Tensor((4,), "uint32"),
    right: T.Tensor((4,), "uint32"),
    choose_left: T.int32,
    output: T.Tensor((1,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    selected_bits: T.let = T.if_then_else(
        choose_left != 0,
        T.reinterpret(left.ptr_to([0]), ty="uint64"),
        T.reinterpret(right.ptr_to([0]), ty="uint64"),
    )
    pointer: T.let[
        T.Var(
            name="selected_pointer",
            ty=PtrType(PrimType("uint32"), "global"),
        )
    ] = T.reinterpret(selected_bits, ty=PtrType(PrimType("uint32"), "global"))
    alias = T.decl_tensor(pointer, ty_args=[T.Tensor((4,), "uint32", scope="global")])
    output[0] = alias[0]


@T.function
def view_of_pointer_derived_view(
    source: T.Tensor((8,), "float32"),
    start: T.Tensor((1,), "int32"),
    output: T.Tensor((2,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    # Integer address bits leave the window an opaque pointer, not a view of `source`.
    address: T.let = T.reinterpret(source.ptr_to([start[0]]), ty="uint64")
    pointer: T.let[T.Var(name="window_pointer", ty=PtrType(PrimType("float32"), "global"))] = (
        T.reinterpret(address, ty=PtrType(PrimType("float32"), "global"))
    )
    window = T.decl_tensor(pointer, ty_args=[T.Tensor((4,), "float32", scope="global")])
    bits = T.decl_tensor(window.data, ty_args=[T.Tensor((4,), "uint32")])
    if lane == 0:
        output[0] = bits[0]
        output[1] = bits[3]


@T.function
def view_of_pointer_derived_view_past_its_extent(
    source: T.Tensor((8,), "float32"),
    start: T.Tensor((1,), "int32"),
    output: T.Tensor((1,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address: T.let = T.reinterpret(source.ptr_to([start[0]]), ty="uint64")
    pointer: T.let[T.Var(name="window_pointer", ty=PtrType(PrimType("float32"), "global"))] = (
        T.reinterpret(address, ty=PtrType(PrimType("float32"), "global"))
    )
    window = T.decl_tensor(pointer, ty_args=[T.Tensor((4,), "float32", scope="global")])
    bits = T.decl_tensor(window.data, ty_args=[T.Tensor((4,), "uint32")])
    if lane == 0:
        output[0] = bits[start[0] + 2]


@T.function
def offset_views_of_pointer_derived_view(
    source: T.Tensor((8,), "float32"),
    start: T.Tensor((1,), "int32"),
    step: T.Tensor((1,), "int32"),
    output: T.Tensor((2,), "uint32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address: T.let = T.reinterpret(source.ptr_to([start[0]]), ty="uint64")
    pointer: T.let[T.Var(name="window_pointer", ty=PtrType(PrimType("float32"), "global"))] = (
        T.reinterpret(address, ty=PtrType(PrimType("float32"), "global"))
    )
    window = T.decl_tensor(pointer, ty_args=[T.Tensor((6,), "float32", scope="global")])
    fixed = T.decl_tensor(
        T.ptr_byte_offset(window.data, 8, ty=T.Ptr("float32", "global")),
        ty_args=[T.Tensor((2,), "uint32", scope="global")],
    )
    moved = T.decl_tensor(
        T.address_of(window[step[0]]), ty_args=[T.Tensor((2,), "uint32", scope="global")]
    )
    if lane == 0:
        output[0] = fixed[0]
        output[1] = moved[1]


@T.function
def wide_view_of_pointer_derived_view(
    source: T.Tensor((16,), "uint16"),
    shift: T.Tensor((1,), "int32"),
    output: T.Tensor((1,), "uint64"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    address: T.let = T.reinterpret(source.ptr_to([0]), ty="uint64")
    pointer: T.let[T.Var(name="halves", ty=PtrType(PrimType("uint16"), "global"))] = T.reinterpret(
        address, ty=PtrType(PrimType("uint16"), "global")
    )
    halves = T.decl_tensor(pointer, ty_args=[T.Tensor((16,), "uint16", scope="global")])
    words = T.decl_tensor(
        T.ptr_byte_offset(halves.data, shift[0] * 2, ty=T.Ptr("uint64", "global")),
        ty_args=[T.Tensor((2,), "uint64", scope="global")],
    )
    if lane == 0:
        output[0] = words[0]


@T.function
def runtime_offset_window(
    source: T.Tensor((8,), "float32"),
    start: T.Tensor((1,), "int32"),
    index: T.Tensor((1,), "int32"),
    output: T.Tensor((1,), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    # The declared extent is an upper bound past the end of `source`.
    window = T.decl_tensor(T.address_of(source[start[0]]), ty_args=[T.Tensor((64,), "float32")])
    if lane == 0:
        output[0] = window[index[0]]


@T.function
def runtime_offset_register_window(
    source: T.Tensor((8,), "float32"),
    start: T.Tensor((1,), "int32"),
    output: T.Tensor((1,), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    registers = T.alloc_local((8,), "float32")
    for i in T.serial(8):
        registers[i] = source[i]
    window = T.decl_tensor(
        T.address_of(registers[start[0]]), ty_args=[T.Tensor((2,), "float32", scope="local")]
    )
    if lane == 0:
        output[0] = window[1]


@T.function
def runtime_offset_wide_view(
    source: T.Tensor((16,), "uint16"),
    start: T.Tensor((1,), "int32"),
    output: T.Tensor((1,), "uint64"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    window = T.decl_tensor(T.address_of(source[start[0]]), ty_args=[T.Tensor((8,), "uint16")])
    words = T.decl_tensor(window.data, ty_args=[T.Tensor((2,), "uint64")])
    if lane == 0:
        output[0] = words[0]


def _shared_backing_sizes(kernel):
    source = emitted_module(kernel)
    return [
        int(size)
        for size in re.findall(r"allocate_cta_shared\(&physical, topology, (\d+)\)", source)
    ]


def _runtime_pointer_view(kernel, byte_len, itemsize):
    source = emitted_module(kernel)
    views = [line for line in source.splitlines() if "let pointer_view_" in line]
    assert len(views) == 1
    assert ".pointer_space_for_mask(ctx.active_mask())?" in views[0]
    assert f"0_usize, {byte_len}_usize, {itemsize}_usize, &ctx, ctx.active_mask())?" in views[0]
    return source


def test_internal_alias_must_fit_the_owner_allocation():
    assert _shared_backing_sizes(in_bounds_internal_alias) == [8 * 4]

    with pytest.raises(UnsupportedTIRxError, match="exceeds owner storage range"):
        emitted_module(alias_extends_internal_owner)


def test_unused_internal_alias_does_not_expand_or_reject_its_owner(tmp_path):
    assert _shared_backing_sizes(dead_alias_extends_internal_owner) == [4 * 4]

    module = numsim.transpile(dead_alias_extends_internal_owner, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.uint32)})
    np.testing.assert_array_equal(result.outputs["output"], np.array([17], dtype=np.uint32))


def test_evaluated_buffer_data_projection_is_a_structural_noop(tmp_path):
    module = numsim.transpile(evaluated_buffer_data_projection_is_pure, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.uint32)})
    np.testing.assert_array_equal(result.outputs["output"], np.array([23], dtype=np.uint32))


def test_dynamic_view_scope_annotation_does_not_reclassify_address_bits(tmp_path):
    module = numsim.transpile(pointer_origin_disagrees_with_declared_scope, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": np.array([31, 37, 41, 43], dtype=np.uint32),
            "output": np.zeros(1, dtype=np.uint32),
        },
    )
    np.testing.assert_array_equal(result.outputs["output"], np.array([31], dtype=np.uint32))


def test_tensor_load_does_not_make_oversized_static_alias_fail_planning(tmp_path):
    assert _shared_backing_sizes(loaded_alias_extends_internal_owner) == [4 * 4]
    numsim.transpile(loaded_alias_extends_internal_owner, cache_dir=tmp_path)


def test_mapa_bits_in_a_nominal_global_declbuffer_keep_runtime_address_resolution():
    source = _runtime_pointer_view(mapa_shared_address_in_generic_handle, 8, 8)
    assert "Ld<v2::reg::variant::U64, v2::Generic>" in source


def test_mapa_source_in_a_nominal_global_declbuffer_keeps_runtime_address_resolution():
    _runtime_pointer_view(mapa_source_through_nominal_global_declbuffer, 64, 8)


def test_zero_extent_pool_owner_uses_only_concrete_view_span_evidence():
    assert _shared_backing_sizes(zero_extent_pool_uses_concrete_view_span) == [32]
    with pytest.raises(UnsupportedTIRxError, match="no positive static view span"):
        emitted_module(zero_extent_pool_without_span_evidence)


@pytest.mark.parametrize(
    ("kernel", "expected_bytes"),
    [(repeated_tile_owner, 40), (repeated_compose_owner, 56)],
)
def test_repeated_layout_atoms_size_the_complete_logical_domain(kernel, expected_bytes):
    assert _shared_backing_sizes(kernel) == [expected_bytes]


def test_odd_float4_spans_round_up_to_complete_physical_bytes():
    assert _shared_backing_sizes(odd_float4_owner) == [2]


def test_integer_addresses_selected_at_runtime_keep_dynamic_views():
    source = _runtime_pointer_view(pointer_origin_through_pure_if_then_else, 16, 4)
    assert "Ld<v2::reg::variant::U32, v2::Generic>" in source


def _pointer_view_inputs(output_len):
    return {
        "source": np.arange(8, dtype=np.float32) + np.float32(0.5),
        "start": np.array([2], dtype=np.int32),
        "output": np.zeros(output_len, dtype=np.uint32),
    }


def test_view_of_a_pointer_derived_view_reads_through_its_parent_pointer(tmp_path):
    """`decl_tensor(parent.data)` of a pointer-derived parent is pointer-derived too."""

    module = numsim.transpile(view_of_pointer_derived_view, cache_dir=tmp_path)
    inputs = _pointer_view_inputs(2)
    result = numsim.Engine().run(module, inputs)

    expected = inputs["source"][[2, 5]].view(np.uint32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_view_of_a_pointer_derived_view_keeps_its_own_extent(tmp_path):
    module = numsim.transpile(view_of_pointer_derived_view_past_its_extent, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="out of bounds|outside|exceeds"):
        numsim.Engine().run(module, _pointer_view_inputs(1))


def test_offset_views_of_a_pointer_derived_view_apply_each_offset_once(tmp_path):
    """The data expression of a pointer-rooted view already carries its offset."""

    module = numsim.transpile(offset_views_of_pointer_derived_view, cache_dir=tmp_path)
    inputs = {
        "source": np.arange(8, dtype=np.float32) + np.float32(0.5),
        "start": np.array([1], dtype=np.int32),
        "step": np.array([3], dtype=np.int32),
        "output": np.zeros(2, dtype=np.uint32),
    }
    result = numsim.Engine().run(module, inputs)

    # window starts at source[1]: fixed[0] is window[2], moved[1] is window[3 + 1].
    expected = inputs["source"][[3, 5]].view(np.uint32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def _halves_inputs(shift):
    return {
        "source": np.arange(16, dtype=np.uint16) + np.uint16(0x100),
        "shift": np.array([shift], dtype=np.int32),
        "output": np.zeros(1, dtype=np.uint64),
    }


def test_wide_view_of_a_pointer_derived_view_at_an_aligned_offset(tmp_path):
    module = numsim.transpile(wide_view_of_pointer_derived_view, cache_dir=tmp_path)
    inputs = _halves_inputs(shift=4)
    result = numsim.Engine().run(module, inputs)

    np.testing.assert_array_equal(result.outputs["output"], inputs["source"][4:8].view(np.uint64))


def test_wide_view_of_a_pointer_derived_view_requires_its_element_alignment(tmp_path):
    """A pointer-rooted view is checked at its address, as a tensor-rooted one is."""

    module = numsim.transpile(wide_view_of_pointer_derived_view, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="DeclBuffer requires 8-byte alignment"):
        numsim.Engine().run(module, _halves_inputs(shift=1))


def _window_inputs(start, index):
    return {
        "source": np.arange(8, dtype=np.float32) + np.float32(0.5),
        "start": np.array([start], dtype=np.int32),
        "index": np.array([index], dtype=np.int32),
        "output": np.zeros(1, dtype=np.float32),
    }


def test_runtime_offset_view_is_bounded_by_its_parent_not_its_declared_extent(tmp_path):
    """`decl_tensor(data=source.ptr_to([start]))` views `source` from `start`.

    It is the parent's storage that bounds it, as a TIR buffer with a runtime
    `elem_offset` was, so an upper-bound extent past the parent is legal.
    """

    module = numsim.transpile(runtime_offset_window, cache_dir=tmp_path)
    inputs = _window_inputs(start=2, index=3)
    result = numsim.Engine().run(module, inputs)

    np.testing.assert_array_equal(result.outputs["output"], inputs["source"][[5]])


def test_runtime_offset_view_still_rejects_an_access_past_its_parent(tmp_path):
    module = numsim.transpile(runtime_offset_window, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="out of bounds|outside|exceeds"):
        numsim.Engine().run(module, _window_inputs(start=2, index=6))


def test_register_view_at_a_runtime_offset_reads_its_element(tmp_path):
    """Register storage keeps `address_of(registers[start])` a plain pointer view."""

    module = numsim.transpile(runtime_offset_register_window, cache_dir=tmp_path)
    inputs = {
        "source": np.arange(8, dtype=np.float32) + np.float32(0.5),
        "start": np.array([2], dtype=np.int32),
        "output": np.zeros(1, dtype=np.float32),
    }
    result = numsim.Engine().run(module, inputs)

    np.testing.assert_array_equal(result.outputs["output"], inputs["source"][[3]])


def _wide_view_inputs(start):
    return {
        "source": np.arange(16, dtype=np.uint16) + np.uint16(0x100),
        "start": np.array([start], dtype=np.int32),
        "output": np.zeros(1, dtype=np.uint64),
    }


def test_wider_view_at_an_aligned_runtime_offset_reads_its_words(tmp_path):
    module = numsim.transpile(runtime_offset_wide_view, cache_dir=tmp_path)
    inputs = _wide_view_inputs(start=4)
    result = numsim.Engine().run(module, inputs)

    np.testing.assert_array_equal(result.outputs["output"], inputs["source"][4:8].view(np.uint64))


def test_wider_view_at_a_misaligned_runtime_offset_fails_at_its_declaration(tmp_path):
    module = numsim.transpile(runtime_offset_wide_view, cache_dir=tmp_path)

    with pytest.raises(numsim.NumSimExecutionError, match="not a multiple of the 8-byte element"):
        numsim.Engine().run(module, _wide_view_inputs(start=1))


def test_all_declared_tmem_views_share_one_physical_backing():
    source = emitted_module(tmem_d_alias_per_cta)
    assert source.count("allocate_cta_tmem(") == 1
    backings = re.findall(r"runtime_buffer_tmem\(\s*(\w+)\.clone\(\),", source)
    assert len(backings) == 2
    assert len(set(backings)) == 1
