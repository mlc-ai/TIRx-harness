from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T


@T.prim_func
def integer_address_alias(source: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    slot = T.alloc_buffer((1,), "uint64", scope="local")
    alias = T.decl_buffer((1,), "uint64", data=slot.data, scope="local")
    slot[0] = T.reinterpret("uint64", source.ptr_to([0]))
    T.ptx.ld.global_.u32(output[0], T.reinterpret("handle", alias[0]))


@T.prim_func
def integer_address_across_alias_views(
    storage: T.Buffer((64,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    alias_data: T.let[
        T.Var(name="integer_address_alias", ty=PointerType(PrimType("uint32"), "global"))
    ] = T.ptr_byte_offset(storage.ptr_to([0]), T.uint32(32 * 4), "uint32")
    alias = T.decl_buffer((32,), "uint32", data=alias_data, scope="global")
    slot = T.alloc_buffer((1,), "uint64", scope="local")
    if lane % 2 == 0:
        slot[0] = T.reinterpret("uint64", storage.ptr_to([lane]))
    else:
        slot[0] = T.reinterpret("uint64", alias.ptr_to([lane]))
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", slot[0]))


@T.prim_func
def integer_address_mixed_spaces_subset_global_load(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    slot = T.alloc_buffer((1,), "uint64", scope="local")
    if lane % 2 == 0:
        slot[0] = T.reinterpret("uint64", source.ptr_to([lane]))
    else:
        slot[0] = T.reinterpret("uint64", shared.ptr_to([lane]))
    if lane % 2 == 0:
        T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", slot[0]))


@T.prim_func
def integer_address_mixed_spaces_active_typed_load(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    slot = T.alloc_buffer((1,), "uint64", scope="local")
    if lane % 2 == 0:
        slot[0] = T.reinterpret("uint64", source.ptr_to([lane]))
    else:
        slot[0] = T.reinterpret("uint64", shared.ptr_to([lane]))
    T.ptx.ld.global_.u32(output[lane], T.reinterpret("handle", slot[0]))


@T.prim_func
def integer_address_mixed_spaces_subset_global_store(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    slot = T.alloc_buffer((1,), "uint64", scope="local")
    if lane % 2 == 0:
        slot[0] = T.reinterpret("uint64", output.ptr_to([lane]))
    else:
        slot[0] = T.reinterpret("uint64", shared.ptr_to([lane]))
    T.ptx.st.global_.u32(
        T.reinterpret("handle", slot[0]),
        T.cast(lane + 1, "uint32"),
        pred=lane % 2 == 0,
    )


@T.prim_func
def integer_address_arithmetic(source: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    base = T.alloc_buffer((1,), "uint64", scope="local")
    added = T.alloc_buffer((1,), "uint64", scope="local")
    reverse_added = T.alloc_buffer((1,), "uint64", scope="local")
    subtracted = T.alloc_buffer((1,), "uint64", scope="local")
    invalid_subtract = T.alloc_buffer((1,), "uint64", scope="local")
    base[0] = T.reinterpret("uint64", source.ptr_to([0]))
    added[0] = base[0] + T.uint64(4)
    reverse_added[0] = T.uint64(4) + base[0]
    subtracted[0] = added[0] - T.uint64(4)
    invalid_subtract[0] = T.uint64(4) - base[0]
    output[0] = 0


@T.prim_func
def integer_address_presence(source: T.Buffer((1,), "uint32"), output: T.Buffer((2,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    token: T.uint64 = T.reinterpret("uint64", T.address_of(source[0]))
    output[0] = T.cast(token != T.uint64(0), "uint32")
    output[1] = T.cast(T.uint64(0) == token, "uint32")


@T.prim_func
def inline_reinterpreted_local_view(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed = T.alloc_local((1,), "uint64")
    words = T.decl_buffer(
        (2,),
        "uint32",
        data=T.reinterpret(
            PointerType(PrimType("uint32")),
            T.address_of(packed[0]),
        ),
        scope="local",
    )
    packed[0] = T.uint64(0x1122334455660000) + T.cast(lane, "uint64")
    output[lane] = words[0]


@T.prim_func
def forged_integer_pointer(output: T.Buffer((1,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    integer = T.alloc_buffer((1,), "uint64", scope="local")
    round_trip = T.alloc_buffer((1,), "uint64", scope="local")
    integer[0] = T.uint64(0x1000)
    forged_handle: T.let[T.handle] = T.reinterpret("handle", integer[0])
    round_trip[0] = T.reinterpret("uint64", forged_handle)
    if lane == 0:
        output[0] = round_trip[0]


@T.prim_func
def loop_and_multiple_stores(source: T.Buffer((1,), "uint32"), output: T.Buffer((1,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    stable = T.alloc_buffer((1,), "uint64", scope="local")
    mixed = T.alloc_buffer((1,), "uint64", scope="local")
    cyclic = T.alloc_buffer((1,), "uint64", scope="local")
    stable[0] = T.reinterpret("uint64", source.ptr_to([0]))
    mixed[0] = T.reinterpret("uint64", source.ptr_to([0]))
    for _ in T.serial(2):
        stable[0] = stable[0] + T.uint64(4)
        mixed[0] = T.uint64(0)
        cyclic[0] = cyclic[0] + T.uint64(4)
    output[0] = 0


@T.prim_func
def leading_null_pointer_initializer(
    enabled: T.int32,
    source: T.Buffer((1,), "uint32"),
    output: T.Buffer((1,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    slot = T.alloc_buffer((1,), "uint64", scope="local")
    slot[0] = T.uint64(0)
    if enabled != 0:
        slot[0] = T.reinterpret("uint64", source.ptr_to([0]))
    if enabled != 0:
        T.ptx.ld.global_.u32(output[0], T.reinterpret("handle", slot[0]))
    else:
        output[0] = 0


@T.prim_func
def raw_loaded_integer_offsets_pointer(
    source: T.Buffer((32,), "uint32"),
    offsets: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    loaded_offset = T.alloc_local((1,), "uint32")
    pointer_base = T.alloc_local((1,), "uint64")
    byte_offset = T.alloc_local((1,), "uint64")
    destination = T.alloc_local((1,), "uint64")
    T.ptx.ld.global_.u32(loaded_offset[0], offsets.ptr_to([lane]))
    pointer_base[0] = T.reinterpret("uint64", output.ptr_to([0]))
    byte_offset[0] = T.cast(loaded_offset[0], "uint64") * T.uint64(4)
    destination[0] = pointer_base[0] + byte_offset[0]
    T.ptx.st.global_.u32(T.reinterpret("handle", destination[0]), source[lane])


@T.prim_func
def raw_shared_integer_address_round_trip(
    source: T.Buffer((32,), "uint32"),
    output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    tokens = T.alloc_buffer((32,), "uint64", scope="shared")
    loaded = T.alloc_buffer((1,), "uint64", scope="local")
    T.ptx.st.shared.u64(
        tokens.ptr_to([lane]),
        T.reinterpret("uint64", source.ptr_to([lane])),
    )
    T.cuda.warp_sync()
    T.ptx.ld.shared.u64(loaded[0], tokens.ptr_to([31 - lane]))
    T.ptx.ld.global_.u32(
        output[lane],
        T.reinterpret("handle", loaded[0]),
    )


@T.prim_func
def runtime_sized_global_address_round_trip(
    rows: T.int32,
    source_ptr: T.handle,
    output: T.Buffer((1,), "uint32"),
):
    source = T.match_buffer(source_ptr, (rows,), "uint32")
    T.device_entry()
    _warp = T.warp_id([1])
    address = T.alloc_local((1,), "uint64")
    address[0] = T.reinterpret("uint64", source.ptr_to([rows - 1]))
    T.ptx.ld.global_.u32(output[0], T.reinterpret("handle", address[0]))


@T.prim_func
def wider_dynamic_view_used_only_for_address(
    output: T.Buffer((1,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    slot = T.alloc_buffer((1,), "uint32", scope="shared")
    wide = T.decl_buffer(
        (1,),
        "uint64",
        data=T.reinterpret(
            PointerType(PrimType("uint64"), "global"),
            T.address_of(slot[0]),
        ),
        scope="global",
    )
    output[0] = T.cuda.cvta_generic_to_shared(wide.ptr_to([0]))


def test_address_storage_has_no_program_visible_pointer_sidecars():
    source = emit_rust_module(analyze(integer_address_alias), integer_address_alias)
    assert "PhysicalPtrSlot" not in source
    assert "MappedSharedAddressSlot" not in source
    assert "RuntimePointerTokenRegistry" not in source


def test_integer_address_aliases_resolve_to_the_same_runtime_bytes(tmp_path):
    source = np.array([0x12345678], dtype=np.uint32)

    result = numsim.Engine().run(
        numsim.transpile(integer_address_alias, cache_dir=tmp_path),
        {"source": source, "output": np.zeros(1, dtype=np.uint32)},
    )

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_integer_addresses_cross_distinct_views_of_one_physical_backing(tmp_path):
    storage = np.arange(64, dtype=np.uint32) * np.uint32(7) + np.uint32(3)
    module = numsim.transpile(integer_address_across_alias_views, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"storage": storage, "output": np.zeros(32, dtype=np.uint32)},
    )

    expected = np.where(
        np.arange(32) % 2 == 0,
        storage[:32],
        storage[32:],
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_integer_address_space_checks_ignore_inactive_lane_values(tmp_path):
    source = np.arange(32, dtype=np.uint32) + np.uint32(17)
    result = numsim.Engine().run(
        numsim.transpile(integer_address_mixed_spaces_subset_global_load, cache_dir=tmp_path),
        {"source": source, "output": np.zeros(32, dtype=np.uint32)},
    )

    expected = np.zeros(32, dtype=np.uint32)
    expected[0::2] = source[0::2]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_typed_pointer_load_rejects_active_lane_from_another_space(tmp_path):
    module = numsim.transpile(integer_address_mixed_spaces_active_typed_load, cache_dir=tmp_path)
    with pytest.raises(
        numsim.NumSimExecutionError,
        match="resolved address.*does not match PTX state space global",
    ):
        numsim.Engine().run(
            module,
            {
                "source": np.arange(32, dtype=np.uint32),
                "output": np.zeros(32, dtype=np.uint32),
            },
        )


def test_predicated_store_resolves_only_lanes_that_access_memory(tmp_path):
    result = numsim.Engine().run(
        numsim.transpile(
            integer_address_mixed_spaces_subset_global_store,
            cache_dir=tmp_path,
        ),
        {"output": np.zeros(32, dtype=np.uint32)},
    )

    expected = np.zeros(32, dtype=np.uint32)
    expected[0::2] = np.arange(1, 33, 2, dtype=np.uint32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_pointer_arithmetic_uses_ordinary_integer_storage():
    source = emit_rust_module(analyze(integer_address_arithmetic), integer_address_arithmetic)
    assert "PhysicalPtrSlot" not in source
    assert "physical_pointer_slot" not in source


def test_address_derived_integer_is_non_null(tmp_path):
    result = numsim.Engine().run(
        numsim.transpile(integer_address_presence, cache_dir=tmp_path),
        {
            "source": np.array([7], dtype=np.uint32),
            "output": np.zeros(2, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], np.array([1, 0], dtype=np.uint32))


def test_decl_buffer_accepts_inline_reinterpreted_local_pointer(tmp_path):
    result = numsim.Engine().run(
        numsim.transpile(inline_reinterpreted_local_view, cache_dir=tmp_path),
        {"output": np.zeros(32, dtype=np.uint32)},
    )

    np.testing.assert_array_equal(
        result.outputs["output"], np.uint32(0x55660000) + np.arange(32, dtype=np.uint32)
    )


def test_integer_handle_reinterpret_round_trips_bits_without_resolution(tmp_path):
    result = numsim.Engine().run(
        numsim.transpile(forged_integer_pointer, cache_dir=tmp_path),
        {"output": np.zeros(1, dtype=np.uint64)},
    )
    np.testing.assert_array_equal(
        result.outputs["output"], np.array([0x1000], dtype=np.uint64)
    )


def test_loops_and_multiple_stores_do_not_create_pointer_sidecars():
    source = emit_rust_module(analyze(loop_and_multiple_stores), loop_and_multiple_stores)
    assert "PhysicalPtrSlot" not in source
    assert "RuntimePointerTokenRegistry" not in source


def test_leading_null_initializer_does_not_poison_integer_address_resolution(tmp_path):
    module = numsim.transpile(leading_null_pointer_initializer, cache_dir=tmp_path)
    source = np.array([0x12345678], dtype=np.uint32)

    enabled = numsim.Engine().run(
        module,
        {
            "enabled": np.int32(1),
            "source": source,
            "output": np.zeros(1, dtype=np.uint32),
        },
    )
    disabled = numsim.Engine().run(
        module,
        {
            "enabled": np.int32(0),
            "source": source,
            "output": np.ones(1, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(enabled.outputs["output"], source)
    np.testing.assert_array_equal(disabled.outputs["output"], np.zeros(1, dtype=np.uint32))


def test_raw_loaded_integer_offsets_an_address_with_ordinary_arithmetic(tmp_path):
    source = np.arange(32, dtype=np.uint32) * np.uint32(17) + np.uint32(5)
    offsets = np.arange(31, -1, -1, dtype=np.uint32)
    result = numsim.Engine().run(
        numsim.transpile(raw_loaded_integer_offsets_pointer, cache_dir=tmp_path),
        {
            "source": source,
            "offsets": offsets,
            "output": np.zeros(32, dtype=np.uint32),
        },
    )

    expected = np.empty(32, dtype=np.uint32)
    expected[offsets] = source
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_shared_integer_address_round_trip_preserves_cross_lane_bits(tmp_path):
    source = np.arange(32, dtype=np.uint32) * np.uint32(19) + np.uint32(7)
    result = numsim.Engine().run(
        numsim.transpile(raw_shared_integer_address_round_trip, cache_dir=tmp_path),
        {
            "source": source,
            "output": np.zeros(32, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], source[::-1])


def test_runtime_sized_global_root_participates_in_integer_address_resolution(tmp_path):
    source = np.arange(17, dtype=np.uint32) * np.uint32(23) + np.uint32(11)
    result = numsim.Engine().run(
        numsim.transpile(runtime_sized_global_address_round_trip, cache_dir=tmp_path),
        {
            "rows": len(source),
            "source": source,
            "output": np.zeros(1, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(result.outputs["output"], source[-1:])


def test_dynamic_view_formation_does_not_consume_its_full_extent(tmp_path):
    result = numsim.Engine().run(
        numsim.transpile(wider_dynamic_view_used_only_for_address, cache_dir=tmp_path),
        {"output": np.zeros(1, dtype=np.uint32)},
    )

    assert int(result.outputs["output"][0]) < 1 << 24
