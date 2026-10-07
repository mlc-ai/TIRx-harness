from __future__ import annotations

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim
from tests.numsim.support.kernels import (
    fetch_register_coordinates,
    raw_scalar_call_mix,
    unsupported_exp,
)
from tirx_harness.numsim.transpiler import native_frontend
from tirx_harness.numsim.transpiler.frontend import analyze

def _synthetic_call_kernel(
    result_dtype: str, op_name: str, arguments: tuple[str, ...], value_dtype: str | None
):
    call = f"T.call_intrin({result_dtype!r}, {op_name!r}, {', '.join(arguments)})"
    parameters = 'output: T.Buffer((32,), "int32")'
    if value_dtype is not None:
        parameters += f', values: T.Buffer((32,), "{value_dtype}")'
    return tvm.script.from_source(
        "@T.prim_func\n"
        f"def kernel({parameters}):\n"
        "    T.device_entry()\n"
        "    lane = T.lane_id()\n"
        f"    value: T.let = {call}\n"
        "    output[lane] = T.int32(1)\n",
        {"T": T},
    )


def _classify(
    result_dtype: str, op_name: str, *arguments: str, value_dtype: str | None = None
) -> tuple[list[object], list[str]]:
    kernel = analyze(
        _synthetic_call_kernel(result_dtype, op_name, tuple(arguments), value_dtype)
    ).kernels[0]
    calls = [
        entry
        for entry in kernel.source_map
        if entry.kind == "Call" and str(entry.node.op.name) == op_name
    ]
    rejections = [item for item in kernel.unsupported if f"Call({op_name}(" in item]
    return calls, rejections


def _accepted(
    result_dtype: str, op_name: str, *arguments: str, value_dtype: str | None = None
) -> bool:
    calls, rejections = _classify(result_dtype, op_name, *arguments, value_dtype=value_dtype)
    assert rejections == []
    assert len(calls) == 1
    return True


def _rejected(result_dtype: str, op_name: str, *arguments: str) -> str:
    _calls, rejections = _classify(result_dtype, op_name, *arguments)
    assert len(rejections) == 1
    return rejections[0]


def test_registry_accepts_the_typed_raw_scalar_fixture():
    spec = analyze(raw_scalar_call_mix)

    assert spec.unsupported == ()


def test_fetch_register_uses_launch_coordinates_without_hardware_state(tmp_path):
    output = np.zeros((2, 32, 2), dtype=np.int32)

    module = numsim.transpile(fetch_register_coordinates, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.zeros_like(output)
    expected[:, :, 0] = np.arange(2, dtype=np.int32)[:, None]
    expected[:, :, 1] = np.arange(32, dtype=np.int32)[None, :]
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert "tirx.cuda.mov_sreg" not in module.rust_source


def test_registry_remains_closed_for_unregistered_calls():
    spec = analyze(unsupported_exp)

    assert any("Call(tirx.sin(float32)->float32)" in item for item in spec.unsupported)


@pytest.mark.parametrize(
    "op_name",
    [
        "tirx.fabs",
        "tirx.log1p",
        "tirx.sigmoid",
    ],
)
def test_registry_accepts_float32_gate_intrinsics(op_name: str):
    assert _accepted("float32", op_name, "T.float32(1)")


@pytest.mark.parametrize("op_name", ["tirx.fabs", "tirx.log1p", "tirx.sigmoid"])
def test_gate_intrinsics_reject_unmodeled_float64_forms(op_name: str):
    assert f"Call({op_name}(float64)->float64)" in _rejected("float64", op_name, "T.float64(1)")


@pytest.mark.parametrize(
    ("result_dtype", "op_name", "arguments"),
    [
        ("uint64", "tirx.cuda.float_as_uint", ("T.float32(1)",)),
        ("float32", "prim.if_then_else", ("T.int32(1)", "T.float32(1)", "T.float32(1)")),
        ("uint64", "tirx.reinterpret", ("T.uint32(1)",)),
        ("handle", "tirx.reinterpret", ("T.uint32(1)",)),
        ("uint32", "tirx.cuda.elect_sync", ("T.uint32(1)",)),
    ],
)
def test_registry_rejects_wrong_arity_or_dtype(result_dtype: str, op_name: str, arguments):
    assert f"Call({op_name}(" in _rejected(result_dtype, op_name, *arguments)


@pytest.mark.parametrize(
    ("vector_dtype", "packed_dtype"),
    [("float16x2", "uint32"), ("float32x2", "uint64")],
)
def test_registry_accepts_packed_vector_reinterpret_forms(vector_dtype: str, packed_dtype: str):
    assert _accepted(vector_dtype, "tirx.reinterpret", f"T.{packed_dtype}(1)")
    assert _accepted(packed_dtype, "tirx.reinterpret", "values[lane]", value_dtype=vector_dtype)


@pytest.mark.parametrize("value_dtype", ["float32", "float16x2", "bfloat16x2"])
@pytest.mark.parametrize(
    "op_name",
    [
        "tirx.cuda.__shfl_sync",
        "tirx.cuda.__shfl_up_sync",
        "tirx.cuda.__shfl_down_sync",
        "tirx.cuda.__shfl_xor_sync",
    ],
)
def test_registry_accepts_cuda_shuffle_value_forms(value_dtype: str, op_name: str):
    assert _accepted(
        value_dtype,
        op_name,
        "T.uint32(1)",
        "values[lane]",
        "T.uint32(1)",
        "T.int32(32)",
        value_dtype=value_dtype,
    )


@pytest.mark.parametrize(
    "op_name",
    [
        "tirx.tvm_warp_shuffle",
        "tirx.tvm_warp_shuffle_up",
        "tirx.tvm_warp_shuffle_down",
        "tirx.tvm_warp_shuffle_xor",
    ],
)
def test_registry_accepts_tvm_shuffle_forms(op_name: str):
    assert _accepted(
        "float32",
        op_name,
        "T.uint32(1)",
        "T.float32(1)",
        "T.int32(1)",
        "T.int32(32)",
        "T.int32(32)",
    )


@pytest.mark.parametrize(
    "predicate_dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    ],
)
def test_registry_accepts_scalar_collective_predicates(predicate_dtype: str):
    predicate = f"T.{predicate_dtype}(1)"
    assert _accepted("uint32", "tirx.cuda.ballot_sync", "T.uint32(1)", predicate)
    assert _accepted("int32", "tirx.cuda.any_sync", "T.uint32(1)", predicate)
    assert _accepted("int64", "tirx.cuda.syncthreads_and", predicate)
    assert _accepted("int64", "tirx.cuda.syncthreads_or", predicate)


def test_registry_exposes_scalar_and_collective_names():
    names = {
        row["ir_name"]
        for row in native_frontend.registry_ops()
        if row["family"] in {"pure_scalar", "warp_collective"}
    }

    assert "prim.bitwise_and" not in names
    assert "tirx.cuda.thread_rank" in names
    assert "tirx.cuda.warp_reduce" in names
    assert "tirx.cuda.func_call" not in names
    assert "tirx.ptx.mapa" not in names
    assert "tirx.cuda.elect_sync" in names
    assert "tirx.cuda.mov_sreg" in names
    assert "tirx.ptx.fma_f32" not in names
    assert "tirx.ptx.ex2" not in names
    assert "tirx.ptx.rcp" not in names
    assert _accepted("float32", "tirx.fma", "T.float32(1)", "T.float32(1)", "T.float32(1)")
    assert _accepted("handle", "tirx.reinterpret", "T.uint64(1)")
    assert _accepted(
        "uint64", "tirx.reinterpret", "values.ptr_to([lane])", value_dtype="uint8"
    )
