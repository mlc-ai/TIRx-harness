from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm_tirx.script import tirx as T


@T.function
def distinct_same_named_vars(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    first: T.let[T.Var(name="same", ty="int32")] = T.int32(1)
    _second: T.let[T.Var(name="same", ty="int32")] = T.int32(2)
    output[lane] = first


@T.function
def parameter_and_inner_var_share_a_name(same: T.int32, output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    _shadow: T.let[T.Var(name="same", ty="int32")] = T.int32(99)
    output[lane] = same


@T.function
def runtime_shape_and_inner_var_share_a_name(
    rows: T.int32,
    input_buffer: T.Tensor((rows,), "int32"),
    output_buffer: T.Tensor((rows,), "int32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    _shadow: T.let[T.Var(name="rows", ty="int32")] = T.int32(1)
    if lane < rows:
        output_buffer[lane] = input_buffer[lane]


def test_distinct_tir_vars_with_the_same_name_do_not_alias(tmp_path) -> None:
    output = np.zeros(32, dtype=np.int32)

    result = numsim.Engine().run(
        numsim.transpile(distinct_same_named_vars, cache_dir=tmp_path),
        {"output": output},
    )

    np.testing.assert_array_equal(result.outputs["output"], np.ones(32, dtype=np.int32))


def test_parameter_identity_is_not_replaced_by_an_inner_var_with_the_same_name(tmp_path) -> None:
    output = np.zeros(32, dtype=np.int32)

    result = numsim.Engine().run(
        numsim.transpile(parameter_and_inner_var_share_a_name, cache_dir=tmp_path),
        {"same": 7, "output": output},
    )

    np.testing.assert_array_equal(result.outputs["output"], np.full(32, 7, dtype=np.int32))


def test_runtime_shape_uses_the_exact_shape_var_identity(tmp_path) -> None:
    source = np.arange(17, dtype=np.int32)
    output = np.zeros_like(source)

    result = numsim.Engine().run(
        numsim.transpile(runtime_shape_and_inner_var_share_a_name, cache_dir=tmp_path),
        {"rows": 17, "input_buffer": source, "output_buffer": output},
    )

    np.testing.assert_array_equal(result.outputs["output_buffer"], source)
