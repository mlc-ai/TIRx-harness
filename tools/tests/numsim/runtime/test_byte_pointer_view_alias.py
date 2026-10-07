from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T


@T.prim_func
def byte_pointer_uint16_alias(output: T.Buffer((32,), "uint16")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    raw = T.alloc_buffer((64,), "uint8", scope="shared")
    raw_data: T.let[
        T.Var(
            name="raw_byte_pointer",
            ty=PointerType(PrimType("void"), "shared"),
        )
    ] = T.reinterpret(PointerType(PrimType("void"), "shared"), raw.ptr_to([0]))
    wide = T.decl_buffer((32,), "uint16", data=raw_data, scope="shared")
    wide[lane] = T.cast(lane * 257, "uint16")
    T.cuda.warp_sync()
    output[lane] = T.cast(raw[lane * 2], "uint16") | T.shift_left(
        T.cast(raw[lane * 2 + 1], "uint16"), T.uint16(8)
    )


def test_byte_pointer_can_form_a_wider_physical_alias(tmp_path):
    output = np.zeros(32, dtype=np.uint16)
    module = numsim.transpile(byte_pointer_uint16_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})
    np.testing.assert_array_equal(
        result.outputs["output"], np.arange(32, dtype=np.uint16) * np.uint16(257)
    )
    assert "with_pointee_itemsize(2_usize).runtime_view" in module.rust_source
