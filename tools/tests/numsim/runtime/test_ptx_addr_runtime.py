from __future__ import annotations

import numpy as np
import pytest
import tvm
from tirx_harness import numsim
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm.script import tirx as T


@T.prim_func
def ptx_addr_global_load(source: T.Buffer((33,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
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
@T.prim_func
def overflow(source: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.ptx.ld.global_.s32(
        output[lane], T.ptx.addr(source.ptr_to([lane]), 2147483648)
    ))
"""

    with pytest.raises(tvm.error.DiagnosticError, match="outside signed int32 range"):
        tvm.script.from_source(source, {"T": T})


def test_ptx_addr_rejects_recursive_wrapper():
    source = """
@T.prim_func
def recursive(source: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.ptx.ld.global_.s32(
        output[lane], T.ptx.addr(T.ptx.addr(source.ptr_to([lane]), 4), 4)
    ))
"""

    with pytest.raises(tvm.error.DiagnosticError, match="cannot be nested"):
        tvm.script.from_source(source, {"T": T})


def test_ptx_addr_rejects_non_pointer_base():
    source = """
@T.prim_func
def non_pointer(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.ptx.ld.global_.s32(
        output[lane], T.ptx.addr(T.int32(7), 4)
    ))
"""

    with pytest.raises(tvm.error.DiagnosticError, match="must be a pointer"):
        tvm.script.from_source(source, {"T": T})


def test_ptx_addr_rejects_disallowed_tmem_operand():
    source = """
@T.prim_func
def disallowed(output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    _warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    value = T.alloc_local((1,), "uint32")
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
        value[0], T.ptx.addr(T.address_of(shared[0]), 4)
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[lane] = value[0]
"""

    with pytest.raises(tvm.error.DiagnosticError, match="does not support T.ptx.addr"):
        tvm.script.from_source(source, {"T": T})
