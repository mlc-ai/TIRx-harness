from __future__ import annotations

import numpy as np
import pytest

import tvm_tirx
from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm_tirx.script import tirx as T


@T.function
def ptx_addr_global_load(source: T.Tensor((33,), "int32"), output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    T.evaluate(T.ptx.ld.global_.s32(output[lane], T.ptx.addr(source.ptr_to([lane]), 4)))


def test_ptx_addr_applies_a_signed_byte_offset(tmp_path):
    source = np.arange(33, dtype=np.int32) * 7 - 11
    output = np.zeros(32, dtype=np.int32)
    spec = analyze(ptx_addr_global_load)
    assert spec.unsupported == ()
    module = numsim.transpile(ptx_addr_global_load, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source[1:])


def test_ptx_addr_rejects_signed_int32_overflow():
    source = """
@T.function
def overflow(source: T.Tensor((32,), "int32"), output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    T.evaluate(T.ptx.ld.global_.s32(
        output[lane], T.ptx.addr(source.ptr_to([lane]), 2147483648)
    ))
"""

    with pytest.raises(ValueError, match="outside signed int32 range"):
        tvm_tirx.script.from_source(source, {"T": T})


def test_ptx_addr_rejects_recursive_wrapper():
    source = """
@T.function
def recursive(source: T.Tensor((32,), "int32"), output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    T.evaluate(T.ptx.ld.global_.s32(
        output[lane], T.ptx.addr(T.ptx.addr(source.ptr_to([lane]), 4), 4)
    ))
"""

    with pytest.raises(ValueError, match="cannot be nested"):
        tvm_tirx.script.from_source(source, {"T": T})


def test_ptx_addr_rejects_non_pointer_base():
    source = """
@T.function
def non_pointer(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    T.evaluate(T.ptx.ld.global_.s32(
        output[lane], T.ptx.addr(T.int32(7), 4)
    ))
"""

    with pytest.raises(ValueError, match="must be a pointer"):
        tvm_tirx.script.from_source(source, {"T": T})


def test_ptx_addr_rejects_disallowed_tmem_operand():
    source = """
@T.function
def disallowed(output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    lane = T.cuda.lane_id()
    shared = T.alloc_shared((32,), "uint32")
    value = T.alloc_local((1,), "uint32")
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
        value[0], T.ptx.addr(T.address_of(shared[0]), 4)
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[lane] = value[0]
"""

    with pytest.raises(ValueError, match="does not support T.ptx.addr"):
        tvm_tirx.script.from_source(source, {"T": T})
