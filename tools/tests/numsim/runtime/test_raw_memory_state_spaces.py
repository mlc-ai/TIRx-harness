from __future__ import annotations

import pytest
import tvm

from tvm.script import tirx as T


@pytest.mark.parametrize(
    ("instruction", "space"),
    [
        ('T.ptx["ld.const.u32"](output[lane], source.ptr_to([lane]))', "const"),
        ('T.ptx["ld.param::entry.u32"](output[lane], source.ptr_to([lane]))', "param::entry"),
        ('T.ptx["ld.param::func.u32"](output[lane], source.ptr_to([lane]))', "param::func"),
        ('T.ptx["ld.volatile.const.u32"](output[lane], source.ptr_to([lane]))', "const"),
        ('T.ptx["st.param::func.u32"](source.ptr_to([lane]), T.uint32(lane))', "param::func"),
    ],
)
def test_target_ptx_dialect_rejects_symbol_only_state_spaces(instruction: str, space: str):
    source = f"""
@T.prim_func
def invalid(source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    {instruction}
"""
    with pytest.raises(tvm.error.DiagnosticError, match="not a valid modifier") as caught:
        tvm.script.from_source(source, extra_vars={"T": T})
    assert space in str(caught.value)
