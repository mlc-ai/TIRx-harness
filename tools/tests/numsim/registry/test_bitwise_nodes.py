"""Primitive bitwise nodes are subject to NumSim's scalar dtype contract."""

import pytest

import tvm_tirx
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize(
    "operation",
    ["bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "shift_left", "shift_right"],
)
def test_vector_bitwise_nodes_are_rejected(operation):
    arguments = "x" if operation == "bitwise_not" else "x, x"
    kernel = tvm_tirx.script.from_source(
        "@T.function\n"
        'def kernel(values: T.Tensor((32,), "int32x2")):\n'
        "    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))\n"
        "    lane = T.cuda.lane_id()\n"
        "    x: T.let = values[lane]\n"
        f"    T.evaluate(T.{operation}({arguments}))\n",
        extra_vars={"T": T},
    )
    spec = analyze(kernel).kernels[0]
    assert any(
        "requires scalar integer or boolean operands, got int32x2" in item
        for item in spec.unsupported
    )
