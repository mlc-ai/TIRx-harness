"""Null predicates on integer addresses and internal local pointers."""

import numpy as np

from tirx_harness import numsim, racecheck, synccheck
from tvm_tirx.script import tirx as T


@T.function
def backed_pointer_null_predicates(output: T.Tensor((32, 3), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((32,), "int32")
    local = T.alloc_local((1,), "int32")
    output[lane, 0] = T.Cast("int32", T.isnullptr(output.data))
    output[lane, 1] = T.Cast("int32", T.isnullptr(shared.ptr_to([lane])))
    output[lane, 2] = T.Cast("int32", T.isnullptr(local.data))


@T.function
def numeric_null_handle(addresses: T.Tensor((32,), "uint64"), output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    output[lane] = T.Cast("int32", T.isnullptr(T.reinterpret(addresses[lane], ty="handle")))


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
