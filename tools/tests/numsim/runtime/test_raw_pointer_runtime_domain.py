from __future__ import annotations

from pathlib import Path

import numpy as np
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T

from tirx_harness import numsim


@T.prim_func
def raw_ptx_64bit_runtime_domain(
    source_bits: T.Buffer((32,), "uint64"),
    source_i64: T.Buffer((32,), "int64"),
    source_u64: T.Buffer((32,), "uint64"),
    output_f64: T.Buffer((32,), "float64"),
    output_i64: T.Buffer((32,), "int64"),
    stored_i64: T.Buffer((32,), "int64"),
    stored_u64: T.Buffer((32,), "uint64"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.ptx.ld.weak.global_.b64(output_f64[lane], source_bits.ptr_to([lane]))
    T.ptx.ld.global_.s64(output_i64[lane], source_i64.ptr_to([lane]))
    T.ptx.st.global_.s64(stored_i64.ptr_to([lane]), source_i64[lane])
    T.ptx.st.global_.b64(stored_u64.ptr_to([lane]), source_u64[lane])


@T.prim_func
def pointer_offset_runtime_domain(
    source_f64: T.Buffer((33,), "float64"),
    source_u8: T.Buffer((33,), "uint8"),
    output_f64: T.Buffer((32,), "float64"),
    output_u8: T.Buffer((32,), "uint8"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.ptr_byte_offset(source_f64.ptr_to([lane]), T.uint32(4), "float64"))
    aligned_f64 = source_f64.ptr_to([lane])
    output_f64[lane] = T.cuda.ldg(aligned_f64, "float64")
    shifted_u8: T.let[T.Var(name="shifted_u8", ty=PointerType(PrimType("uint8")))] = (
        T.ptr_byte_offset(source_u8.ptr_to([0]), T.uint32(1), "uint8")
    )
    alias_u8 = T.decl_buffer((32,), "uint8", data=shifted_u8, scope="global")
    output_u8[lane] = alias_u8[lane]


@T.prim_func
def access_ptr_float32_runtime_domain(
    source_f32: T.Buffer((32,), "float32"),
    source_u32: T.Buffer((32,), "uint32"),
    direct_output: T.Buffer((32,), "float32"),
    nested_output: T.Buffer((32,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(source_f32.access_ptr("r", offset=lane, extent=1))
    direct_output[lane] = source_f32[lane]
    writable = nested_output.access_ptr("w", offset=lane, extent=1)
    nested = T.tvm_access_ptr(T.type_annotation("uint32"), writable, 0, 1, 2)
    T.ptx.st.global_.u32(nested, source_u32[lane])


@T.prim_func
def shared_handle_address_runtime_domain(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    T.evaluate(T.cuda.smem_addr_from_uint64(T.address_of(shared[lane])))
    output[lane] = T.cuda.cvta_generic_to_shared(T.address_of(shared[lane]))


@T.prim_func
def packed_float4_shared_address_runtime_domain(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed = T.alloc_buffer((16,), "uint8", scope="shared")
    values = packed.view("float4_e2m1fn")
    output[lane] = T.cuda.cvta_generic_to_shared(T.address_of(values[lane]))


def test_raw_ptx_64bit_runtime_domain_preserves_exact_bytes(tmp_path: Path):
    source_bits = (np.arange(32, dtype=np.uint64) * np.uint64(0x0102040810204081)) ^ np.uint64(
        0x7FF80000000000A5
    )
    source_i64 = np.arange(-16, 16, dtype=np.int64) * np.int64(0x01020304050607)
    source_u64 = np.arange(32, dtype=np.uint64) ^ np.uint64(0xFEDCBA9876543210)
    module = numsim.transpile(raw_ptx_64bit_runtime_domain, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module,
        {
            "source_bits": source_bits,
            "source_i64": source_i64,
            "source_u64": source_u64,
            "output_f64": np.zeros(32, dtype=np.float64),
            "output_i64": np.zeros(32, dtype=np.int64),
            "stored_i64": np.zeros(32, dtype=np.int64),
            "stored_u64": np.zeros(32, dtype=np.uint64),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(result.outputs["output_f64"].view(np.uint64), source_bits)
        np.testing.assert_array_equal(result.outputs["output_i64"], source_i64)
        np.testing.assert_array_equal(result.outputs["stored_i64"], source_i64)
        np.testing.assert_array_equal(result.outputs["stored_u64"], source_u64)

    check()


def test_pointer_offset_runtime_domain_uses_exact_byte_offsets_and_aliases(tmp_path: Path):
    source_bytes = np.arange(33 * 8, dtype=np.uint8) ^ np.uint8(0xA5)
    source_f64 = source_bytes.view(np.float64)
    source_u8 = (np.arange(33, dtype=np.uint8) * np.uint8(7)) ^ np.uint8(0xD3)
    module = numsim.transpile(pointer_offset_runtime_domain, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module,
        {
            "source_f64": source_f64,
            "source_u8": source_u8,
            "output_f64": np.zeros(32, dtype=np.float64),
            "output_u8": np.zeros(32, dtype=np.uint8),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(
            result.outputs["output_f64"].view(np.uint8), source_bytes[: 32 * 8]
        )
        np.testing.assert_array_equal(result.outputs["output_u8"], source_u8[1:])

    check()


def test_access_ptr_float32_runtime_domain_preserves_nested_alias_offsets(tmp_path: Path):
    source_f32 = np.arange(32, dtype=np.float32) * np.float32(1.25) - np.float32(9)
    source_u32 = np.arange(32, dtype=np.uint32) ^ np.uint32(0xA55AA55A)
    module = numsim.transpile(access_ptr_float32_runtime_domain, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(
        module,
        {
            "source_f32": source_f32,
            "source_u32": source_u32,
            "direct_output": np.zeros(32, dtype=np.float32),
            "nested_output": np.zeros(32, dtype=np.uint32),
        },
    )

    def check() -> None:
        np.testing.assert_array_equal(result.outputs["direct_output"], source_f32)
        np.testing.assert_array_equal(result.outputs["nested_output"], source_u32)

    check()


def test_shared_handle_address_runtime_domain_preserves_relative_byte_addresses(tmp_path: Path):
    module = numsim.transpile(shared_handle_address_runtime_domain, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(module, {"output": np.zeros(32, dtype=np.uint32)})

    addresses = result.outputs["output"]
    expected_offsets = np.arange(32, dtype=np.uint32) * np.uint32(4)
    np.testing.assert_array_equal(addresses - addresses[0], expected_offsets)


def test_packed_float4_address_maps_each_nibble_pair_to_one_byte(tmp_path: Path):
    module = numsim.transpile(packed_float4_shared_address_runtime_domain, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(module, {"output": np.zeros(32, dtype=np.uint32)})

    addresses = result.outputs["output"]
    expected_offsets = np.arange(32, dtype=np.uint32) // np.uint32(2)
    np.testing.assert_array_equal(addresses - addresses[0], expected_offsets)
