"""Null predicates on integer addresses and internal local pointers."""

import numpy as np
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck


@T.prim_func
def backed_pointer_null_predicates(output: T.Buffer((32, 3), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "int32", scope="shared")
    local = T.alloc_local((1,), "int32")
    output[lane, 0] = T.Cast("int32", T.isnullptr(output.data))
    output[lane, 1] = T.Cast("int32", T.isnullptr(shared.ptr_to([lane])))
    output[lane, 2] = T.Cast("int32", T.isnullptr(local.data))


@T.prim_func
def numeric_null_handle(addresses: T.Buffer((32,), "uint64"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.Cast("int32", T.isnullptr(T.reinterpret("handle", addresses[lane])))


def test_isnullptr_on_backed_pointers(tmp_path):
    inputs = {"output": np.full((32, 3), -1, dtype=np.int32)}
    for checker in (synccheck, racecheck):
        checker(backed_pointer_null_predicates, inputs).require_clean()
    module = numsim.transpile(backed_pointer_null_predicates, cache_dir=tmp_path)
    result = numsim.Engine().run(module, inputs)
    np.testing.assert_array_equal(result.outputs["output"], np.zeros((32, 3), dtype=np.int32))


def test_numeric_null_predicate_compares_bits_without_resolving_memory(tmp_path):
    addresses = np.resize(np.array([0, 1, 0x1000, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64), 32)
    inputs = {"addresses": addresses, "output": np.full(32, -1, dtype=np.int32)}
    for checker in (synccheck, racecheck):
        checker(numeric_null_handle, inputs).require_clean()
    result = numsim.Engine().run(numsim.transpile(numeric_null_handle, cache_dir=tmp_path), inputs)
    np.testing.assert_array_equal(result.outputs["output"], (addresses == 0).astype(np.int32))
