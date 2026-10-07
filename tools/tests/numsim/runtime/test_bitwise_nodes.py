"""TVM 0.27 primitive nodes preserve integer widths and boolean masks."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness.numsim.transpiler.frontend import analyze

DTYPES = [f"{sign}int{bits}" for sign in ("", "u") for bits in (8, 16, 32, 64)]


def bitwise_case(dtype):
    bits = np.dtype(dtype).itemsize * 8
    kernel = tvm.script.from_source(
        f'''@T.prim_func
def bitwise_nodes(values: T.Buffer((32,), "{dtype}"), output: T.Buffer((32, 6), "{dtype}")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    x: T.let = values[lane]
    mask: T.let = T.cast(5, "{dtype}")
    count: T.let = T.cast(lane * {bits - 1} // 31, "{dtype}")
    output[lane, 0] = x & mask
    output[lane, 1] = x | mask
    output[lane, 2] = x ^ mask
    output[lane, 3] = ~x
    output[lane, 4] = x << count
    output[lane, 5] = x >> count
''',
        extra_vars={"T": T},
    )
    info = np.iinfo(dtype)
    values = np.resize(np.array([info.min, info.max, 0, 1, 5, 13], dtype=dtype), 32)
    expected = np.empty((32, 6), dtype=dtype)
    for lane, x in enumerate(values):
        x = int(x)
        count = lane * (bits - 1) // 31
        for column, value in enumerate((x & 5, x | 5, x ^ 5, ~x, x << count, x >> count)):
            value &= (1 << bits) - 1
            if dtype.startswith("int") and value >= 1 << (bits - 1):
                value -= 1 << bits
            expected[lane, column] = value
    return kernel, {"values": values, "output": np.zeros_like(expected)}, expected


@pytest.mark.parametrize("dtype", DTYPES)
def test_bitwise_nodes_match_integer_oracle(dtype, tmp_path):
    kernel, inputs, expected = bitwise_case(dtype)
    spec = analyze(kernel).kernels[0]
    kinds = {entry.kind for entry in spec.source_map}
    assert {"BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseNot", "LShift", "RShift"} <= kinds
    result = run_checked(kernel, inputs, outputs=("output",), cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], expected)


@T.prim_func
def boolean_bitwise(output: T.Buffer((32, 4), "bool")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    a: T.let = lane % 2 == 0
    b: T.let = lane % 3 == 0
    output[lane, 0] = T.bitwise_and(a, b)
    output[lane, 1] = T.bitwise_or(a, b)
    output[lane, 2] = T.bitwise_xor(a, b)
    output[lane, 3] = T.bitwise_not(a)


def boolean_case():
    lane = np.arange(32)
    a, b = lane % 2 == 0, lane % 3 == 0
    expected = np.stack((a & b, a | b, a ^ b, ~a), axis=1)
    return boolean_bitwise, {"output": np.zeros_like(expected)}, expected


def test_boolean_bitwise_nodes_preserve_masks(tmp_path):
    kernel, inputs, expected = boolean_case()
    result = run_checked(kernel, inputs, outputs=("output",), cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], expected)


@T.prim_func
def bitwise_launch_extent(output: T.Buffer((32,), "int32")):
    T.device_entry()
    width: T.let = 64
    _warp = T.warp_id([1])
    lane = T.lane_id([width >> 1])
    output[lane] = lane


def test_bitwise_node_in_static_launch_extent(tmp_path):
    result = run_checked(
        bitwise_launch_extent, {"output": np.zeros(32, dtype="int32")},
        outputs=("output",), cache_dir=tmp_path,
    )
    np.testing.assert_array_equal(result.outputs["output"], np.arange(32))
