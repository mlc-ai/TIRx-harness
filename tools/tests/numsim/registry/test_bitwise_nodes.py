"""Primitive bitwise nodes are subject to NumSim's scalar dtype contract."""

import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness.numsim.transpiler.frontend import analyze


@pytest.mark.parametrize(
    "operation",
    ["bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "shift_left", "shift_right"],
)
def test_vector_bitwise_nodes_are_rejected(operation):
    arguments = "x" if operation == "bitwise_not" else "x, x"
    kernel = tvm.script.from_source(
        "@T.prim_func\n"
        'def kernel(values: T.Buffer((32,), "int32x2")):\n'
        "    T.device_entry()\n"
        "    lane = T.lane_id([32])\n"
        "    x: T.let = values[lane]\n"
        f"    T.evaluate(T.{operation}({arguments}))\n",
        extra_vars={"T": T},
    )
    spec = analyze(kernel).kernels[0]
    assert any(
        "requires scalar integer or boolean operands, got int32x2" in item
        for item in spec.unsupported
    )
