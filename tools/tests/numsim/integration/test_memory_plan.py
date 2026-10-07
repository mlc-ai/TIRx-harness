from __future__ import annotations

import re

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tests.numsim.support.kernels import tmem_d_alias_per_cta
from tests.numsim.support.manifest import emitted_module
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.tirx.layout import ComposeLayout, S, TileLayout

_PADDED_TILE = TileLayout(S[(2, 2) : (4, 1)])
_PADDED_COMPOSE = ComposeLayout(0, 0, 0, _PADDED_TILE)


@T.prim_func
def in_bounds_internal_alias(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((8,), "uint32", scope="shared")
    alias = T.decl_buffer((4,), "uint32", data=storage.data, scope="shared", elem_offset=4)
    alias[0] = 1
    output[0] = alias[0]


@T.prim_func
def alias_extends_internal_owner(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((4,), "uint32", scope="shared")
    alias = T.decl_buffer((8,), "uint32", data=storage.data, scope="shared")
    alias[7] = 1
    output[0] = alias[7]


@T.prim_func
def dead_alias_extends_internal_owner(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((4,), "uint32", scope="shared")
    _dead = T.decl_buffer((8,), "uint32", data=storage.data, scope="shared")
    output[0] = 17


@T.prim_func
def loaded_alias_extends_internal_owner(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((4,), "uint32", scope="shared")
    alias = T.decl_buffer((8,), "uint32", data=storage.data, scope="shared")
    storage[0] = 29
    output[0] = alias[0]


@T.prim_func
def evaluated_buffer_data_projection_is_pure(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    T.evaluate(output.data)
    output[0] = 23


@T.prim_func
def pointer_origin_disagrees_with_declared_scope(
    source: T.Buffer((4,), "uint32"), output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    pointer: T.let[
        T.Var(
            name="global_pointer",
            ty=PointerType(PrimType("uint32"), "shared"),
        )
    ] = T.reinterpret(PointerType(PrimType("uint32"), "shared"), source.ptr_to([0]))
    alias = T.decl_buffer((4,), "uint32", data=pointer, scope="shared")
    output[0] = alias[0]


@T.prim_func
def mapa_shared_address_in_generic_handle(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    shared = T.alloc_buffer((1,), "uint64", scope="shared")
    remote_bits: T.uint64
    T.ptx.mapa.u64(remote_bits, T.address_of(shared[0]), T.uint32(0))
    remote_pointer: T.let[
        T.Var(name="remote_pointer", ty=PointerType(PrimType("uint64"), "global"))
    ] = T.reinterpret(PointerType(PrimType("uint64"), "global"), remote_bits)
    remote = T.decl_buffer((1,), "uint64", data=remote_pointer)
    output[0] = T.cast(remote[0], "uint32")


@T.prim_func
def mapa_source_through_nominal_global_declbuffer(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    _cta = T.cta_id_in_cluster([1])
    _warp = T.warp_id([1])
    shared = T.alloc_buffer((8,), "uint64", scope="shared")
    alias_pointer: T.let[
        T.Var(name="nominal_global_alias_pointer", ty=PointerType(PrimType("uint64"), "global"))
    ] = T.reinterpret(PointerType(PrimType("uint64"), "global"), shared.ptr_to([0]))
    alias = T.decl_buffer(
        (8,),
        "uint64",
        data=alias_pointer,
        scope="global",
    )
    mapped = T.local_scalar("uint64")
    T.ptx.mapa.shared__cluster.u64(mapped, alias.ptr_to([0]), T.uint32(0))
    output[0] = 0


@T.prim_func
def pool_capacity_bounds_alias(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((0,), "uint8", scope="shared")
    T.attr(storage.data, "tirx.pool_max_bytes", 32)
    alias = T.decl_buffer((8,), "uint32", data=storage.data, scope="shared")
    alias[7] = 1
    output[0] = alias[7]


@T.prim_func
def alias_exceeds_pool_capacity(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((0,), "uint8", scope="shared")
    T.attr(storage.data, "tirx.pool_max_bytes", 16)
    alias = T.decl_buffer((8,), "uint32", data=storage.data, scope="shared")
    alias[7] = 1
    output[0] = alias[7]


@T.prim_func
def zero_extent_pool_uses_concrete_view_span(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    alias = T.decl_buffer((8,), "uint32", data=storage.data, scope="shared.dyn")
    alias[7] = 1
    output[0] = alias[7]


@T.prim_func
def zero_extent_pool_without_span_evidence(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    T.alloc_buffer((0,), "uint8", scope="shared.dyn")
    output[0] = 0


@T.prim_func
def conflicting_pool_capacities(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((0,), "uint8", scope="shared")
    T.attr(storage.data, "tirx.pool_max_bytes", 16)
    T.attr(storage.data, "tirx.pool_max_bytes", 32)
    alias = T.decl_buffer((4,), "uint32", data=storage.data, scope="shared")
    output[0] = alias[0]


@T.prim_func
def negative_pool_capacity(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((0,), "uint8", scope="shared")
    T.attr(storage.data, "tirx.pool_max_bytes", -1)
    output[0] = 0


@T.prim_func
def repeated_tile_owner(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((2, 3), "uint32", scope="shared", layout=_PADDED_TILE)
    alias = T.decl_buffer((10,), "uint32", data=storage.data, scope="shared")
    alias[9] = 7
    output[0] = alias[9]


@T.prim_func
def repeated_compose_owner(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((2, 4), "uint32", scope="shared", layout=_PADDED_COMPOSE)
    alias = T.decl_buffer((14,), "uint32", data=storage.data, scope="shared")
    alias[13] = 9
    output[0] = alias[13]


@T.prim_func
def odd_float4_owner(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((3,), "float4_e2m1fn", scope="shared")
    alias = T.decl_buffer((2,), "uint8", data=storage.data, scope="shared")
    output[0] = alias[1]


@T.prim_func
def high_nibble_float4_alias(output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    storage = T.alloc_buffer((2,), "uint8", scope="shared")
    alias = T.decl_buffer((3,), "float4_e2m1fn", data=storage.data, elem_offset=1, scope="shared")
    output[0] = T.cast(alias[0], "uint32")


@T.prim_func
def pointer_origin_through_pure_if_then_else(
    left: T.Buffer((4,), "uint32"),
    right: T.Buffer((4,), "uint32"),
    choose_left: T.int32,
    output: T.Buffer((1,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    selected_bits: T.let = T.if_then_else(
        choose_left != 0,
        T.reinterpret("uint64", left.ptr_to([0])),
        T.reinterpret("uint64", right.ptr_to([0])),
    )
    pointer: T.let[
        T.Var(
            name="selected_pointer",
            ty=PointerType(PrimType("uint32"), "global"),
        )
    ] = T.reinterpret(PointerType(PrimType("uint32"), "global"), selected_bits)
    alias = T.decl_buffer((4,), "uint32", data=pointer, scope="global")
    output[0] = alias[0]


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


def test_pool_capacity_attr_sizes_owner_without_opportunistic_growth(tmp_path):
    assert _shared_backing_sizes(pool_capacity_bounds_alias) == [32]
    module = numsim.transpile(pool_capacity_bounds_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(1, dtype=np.uint32)})
    assert result.verdict == "clean", result.diagnostics
    np.testing.assert_array_equal(result.outputs["output"], np.array([1], dtype=np.uint32))
    with pytest.raises(UnsupportedTIRxError, match="exceeds owner storage range"):
        emitted_module(alias_exceeds_pool_capacity)


def test_zero_extent_pool_owner_uses_only_concrete_view_span_evidence():
    assert _shared_backing_sizes(zero_extent_pool_uses_concrete_view_span) == [32]
    with pytest.raises(UnsupportedTIRxError, match="no positive static view span"):
        emitted_module(zero_extent_pool_without_span_evidence)


def test_conflicting_pool_capacity_attrs_fail_closed():
    with pytest.raises(UnsupportedTIRxError, match="conflicting capacities 16 and 32"):
        emitted_module(conflicting_pool_capacities)


def test_malformed_pool_capacity_attr_fails_closed():
    with pytest.raises(UnsupportedTIRxError, match="cannot be negative"):
        emitted_module(negative_pool_capacity)


@pytest.mark.parametrize(
    ("kernel", "expected_bytes"),
    [(repeated_tile_owner, 40), (repeated_compose_owner, 56)],
)
def test_repeated_layout_atoms_size_the_complete_logical_domain(kernel, expected_bytes):
    assert _shared_backing_sizes(kernel) == [expected_bytes]


def test_odd_float4_spans_round_up_to_complete_physical_bytes():
    assert _shared_backing_sizes(odd_float4_owner) == [2]


def test_float4_alias_can_begin_at_the_high_nibble():
    source = emitted_module(high_nibble_float4_alias)
    assert re.search(
        r"let buffer_2 = runtime_buffer_shared\(\s*"
        r"numsim_cta_shared_backing.clone\(\),\s*0,\s*2,\s*2,\s*0,\s*\);",
        source,
    )
    index = re.search(r"let (broadcast_\d+) = WarpValue::splat\(\(1_i32\) as i64\);", source)
    assert index is not None
    assert f"{index[1]}[lane].div_euclid(2_i64)" in source
    assert f"{index[1]}[lane].rem_euclid(2_i64) == 0" in source


def test_integer_addresses_selected_at_runtime_keep_dynamic_views():
    source = _runtime_pointer_view(pointer_origin_through_pure_if_then_else, 16, 4)
    assert "Ld<v2::reg::variant::U32, v2::Generic>" in source


def test_all_declared_tmem_views_share_one_physical_backing():
    source = emitted_module(tmem_d_alias_per_cta)
    assert source.count("allocate_cta_tmem(") == 1
    backings = re.findall(r"runtime_buffer_tmem\(\s*(\w+)\.clone\(\),", source)
    assert len(backings) == 2
    assert len(set(backings)) == 1
