from __future__ import annotations

import pytest

import tvm_tirx
from tvm_tirx.script import tirx as T


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
@T.function
def invalid(source: T.Tensor((32,), "uint32"), output: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    {instruction}
"""
    # tvm-tirx rejects an unknown modifier in a T.ptx["..."] spelling with KeyError.
    with pytest.raises(KeyError, match="not a valid modifier") as caught:
        tvm_tirx.script.from_source(source, extra_vars={"T": T})
    assert space in str(caught.value)
