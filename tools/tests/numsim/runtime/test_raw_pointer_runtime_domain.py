from __future__ import annotations

from pathlib import Path

import numpy as np

from tirx_harness import numsim
from tvm_tirx.script import tirx as T


@T.function
def raw_ptx_64bit_runtime_domain(
    source_bits: T.Tensor((32,), "uint64"),
    source_i64: T.Tensor((32,), "int64"),
    source_u64: T.Tensor((32,), "uint64"),
    output_f64: T.Tensor((32,), "float64"),
    output_i64: T.Tensor((32,), "int64"),
    stored_i64: T.Tensor((32,), "int64"),
    stored_u64: T.Tensor((32,), "uint64"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    T.ptx.ld.weak.global_.b64(output_f64[lane], source_bits.ptr_to([lane]))
    T.ptx.ld.global_.s64(output_i64[lane], source_i64.ptr_to([lane]))
    T.ptx.st.global_.s64(stored_i64.ptr_to([lane]), source_i64[lane])
    T.ptx.st.global_.b64(stored_u64.ptr_to([lane]), source_u64[lane])


@T.function
def shared_handle_address_runtime_domain(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((32,), "uint32")
    T.evaluate(T.cuda.smem_addr_from_uint64(T.address_of(shared[lane])))
    output[lane] = T.cuda.cvta_generic_to_shared(T.address_of(shared[lane]))


@T.function
def packed_float4_shared_address_runtime_domain(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    packed = T.alloc_shared((16,), "uint8")
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
