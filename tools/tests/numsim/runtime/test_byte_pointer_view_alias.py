from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm_tirx.ir import PtrType
from tvm_tirx.ir.type import PrimType
from tvm_tirx.script import tirx as T


@T.function
def byte_pointer_uint16_alias(output: T.Tensor((32,), "uint16")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    raw = T.alloc_shared((64,), "uint8")
    raw_data: T.let[
        T.Var(
            name="raw_byte_pointer",
            ty=PtrType(PrimType("void"), "shared"),
        )
    ] = T.reinterpret(raw.ptr_to([0]), ty=PtrType(PrimType("void"), "shared"))
    wide = T.decl_tensor(raw_data, ty_args=[T.Tensor((32,), "uint16", scope="shared")])
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
