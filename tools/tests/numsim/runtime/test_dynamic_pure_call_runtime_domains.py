from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim


INTEGER_DTYPES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)
SCALAR_DTYPES = (
    "bool",
    *INTEGER_DTYPES,
    "float16",
    "bfloat16",
    "float32",
    "float64",
)
NUMPY_DTYPES = {
    "bool": np.bool_,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


def _name(dtype: str) -> str:
    return dtype.replace("_", "")


def _make_integer_runtime_kernel():
    parameters = ['    output_bool: T.Buffer((4,), "bool"),']
    parameters.extend(
        f'    output_{_name(dtype)}: T.Buffer((20,), "{dtype}"),' for dtype in INTEGER_DTYPES
    )
    statements = [
        "        output_bool[0] = T.bitwise_and(T.bool(True), T.bool(False))",
        "        output_bool[1] = T.bitwise_or(T.bool(True), T.bool(False))",
        "        output_bool[2] = T.bitwise_xor(T.bool(True), T.bool(False))",
        "        output_bool[3] = T.bitwise_not(T.bool(True))",
    ]
    for dtype in INTEGER_DTYPES:
        output = f"output_{_name(dtype)}"
        lhs = f'T.cast(lane + 53, "{dtype}")'
        rhs = f'T.cast(lane + 15, "{dtype}")'
        statements.extend(
            (
                f"        {output}[0] = T.bitwise_and({lhs}, {rhs})",
                f"        {output}[1] = T.bitwise_or({lhs}, {rhs})",
                f"        {output}[2] = T.bitwise_xor({lhs}, {rhs})",
                f"        {output}[3] = T.bitwise_not({lhs})",
            )
        )
        for index, amount_dtype in enumerate(INTEGER_DTYPES):
            amount = f'T.cast(1, "{amount_dtype}")'
            statements.extend(
                (
                    f'        {output}[{4 + index * 2}] = T.cast(T.shift_left({lhs}, {amount}), "{dtype}")',
                    f'        {output}[{5 + index * 2}] = T.cast(T.shift_right({lhs}, {amount}), "{dtype}")',
                )
            )
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def dynamic_integer_runtime_domain(",
                *parameters,
                "):",
                "    T.device_entry()",
                "    _warp = T.warp_id([1])",
                "    lane = T.lane_id([32])",
                "    if lane == 0:",
                *statements,
            )
        ),
        extra_vars={"T": T},
    )


DYNAMIC_INTEGER_RUNTIME_DOMAIN = _make_integer_runtime_kernel()


def _make_if_then_else_runtime_kernel():
    parameters = []
    for dtype in SCALAR_DTYPES:
        parameters.append(f'    output_{_name(dtype)}: T.Buffer((2,), "{dtype}"),')
    parameters.extend(
        (
            '    left: T.Buffer((32,), "uint32"),',
            '    right: T.Buffer((32,), "uint32"),',
            '    pointer_output: T.Buffer((34,), "uint32"),',
        )
    )
    statements = [
        '        output_bool[0] = T.call_intrin("bool", "prim.if_then_else", lane == 0, T.bool(True), T.bool(False))',
        '        output_bool[1] = T.call_intrin("bool", "prim.if_then_else", lane != 0, T.bool(True), T.bool(False))',
    ]
    for dtype in SCALAR_DTYPES:
        if dtype == "bool":
            continue
        output = f"output_{_name(dtype)}"
        statements.extend(
            (
                f'        {output}[0] = T.call_intrin("{dtype}", "prim.if_then_else", lane == 0, T.cast(3, "{dtype}"), T.cast(7, "{dtype}"))',
                f'        {output}[1] = T.call_intrin("{dtype}", "prim.if_then_else", lane != 0, T.cast(3, "{dtype}"), T.cast(7, "{dtype}"))',
            )
        )
    pointer_statements = (
        '    selected: T.let[T.handle] = T.call_intrin("handle", "prim.if_then_else", lane % 2 == 0, left.ptr_to([lane]), right.ptr_to([lane]))',
        "    T.ptx.ld.global_.u32(pointer_output[lane], selected)",
        "    if lane == 0:",
        '        selected_left: T.let[T.handle] = T.call_intrin("handle", "prim.if_then_else", T.bool(True), left.ptr_to([0]), right.ptr_to([0]))',
        '        selected_right: T.let[T.handle] = T.call_intrin("handle", "prim.if_then_else", T.bool(False), left.ptr_to([0]), right.ptr_to([0]))',
        "        T.ptx.ld.global_.u32(pointer_output[32], selected_left)",
        "        T.ptx.ld.global_.u32(pointer_output[33], selected_right)",
    )
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def dynamic_if_then_else_runtime_domain(",
                *parameters,
                "):",
                "    T.device_entry()",
                "    _warp = T.warp_id([1])",
                "    lane = T.lane_id([32])",
                "    if lane == 0:",
                *statements,
                *pointer_statements,
            )
        ),
        extra_vars={"T": T},
    )


DYNAMIC_IF_THEN_ELSE_RUNTIME_DOMAIN = _make_if_then_else_runtime_kernel()


@T.prim_func
def IF_THEN_ELSE_MIXED_POINTER_SPACES(
    source: T.Buffer((32,), "uint32"), output: T.Buffer((32,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32,), "uint32", scope="shared")
    shared[lane] = T.uint32(0xA7)
    T.cuda.warp_sync()
    selected: T.let[T.handle] = T.call_intrin(
        "handle",
        "prim.if_then_else",
        lane % 2 == 0,
        source.ptr_to([lane]),
        shared.ptr_to([lane]),
    )
    T.ptx.ld.global_.u32(output[lane], selected)


def _make_tvm_shuffle_runtime_kernel():
    parameters = [
        f'    output_{_name(dtype)}: T.Buffer((32, 4), "{dtype}"),' for dtype in SCALAR_DTYPES
    ]
    statements = []
    for dtype in SCALAR_DTYPES:
        value = "lane % 2 == 0" if dtype == "bool" else f'T.cast(lane, "{dtype}")'
        output = f"output_{_name(dtype)}"
        statements.extend(
            (
                f"    {output}[lane, 0] = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), {value}, 31 - lane, 32, 32)",
                f"    {output}[lane, 1] = T.tvm_warp_shuffle_xor(T.uint32(0xFFFFFFFF), {value}, 1, 32, 32)",
                f"    {output}[lane, 2] = T.tvm_warp_shuffle_up(T.uint32(0xFFFFFFFF), {value}, 1, 32, 32)",
                f"    {output}[lane, 3] = T.tvm_warp_shuffle_down(T.uint32(0xFFFFFFFF), {value}, 1, 32, 32)",
            )
        )
    return tvm.script.from_source(
        "\n".join(
            (
                "@T.prim_func",
                "def dynamic_tvm_shuffle_runtime_domain(",
                *parameters,
                "):",
                "    T.device_entry()",
                "    _warp = T.warp_id([1])",
                "    lane = T.lane_id([32])",
                *statements,
            )
        ),
        extra_vars={"T": T},
    )


DYNAMIC_TVM_SHUFFLE_RUNTIME_DOMAIN = _make_tvm_shuffle_runtime_kernel()


def _bfloat16_bits(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32).view(np.uint32).astype(np.uint32) >> np.uint32(16)


def _empty_output(dtype: str, shape: tuple[int, ...]) -> object:
    if dtype == "bfloat16":
        return np.zeros(shape, dtype=np.uint16)
    return np.zeros(shape, dtype=NUMPY_DTYPES[dtype])


def test_complete_integer_bitwise_and_shift_domain_executes(tmp_path: Path):
    module = numsim.transpile(DYNAMIC_INTEGER_RUNTIME_DOMAIN, cache_dir=tmp_path)
    arguments: dict[str, object] = {"output_bool": np.zeros(4, dtype=np.bool_)}
    for dtype in INTEGER_DTYPES:
        arguments[f"output_{_name(dtype)}"] = np.zeros(20, dtype=NUMPY_DTYPES[dtype])
    result = numsim.Engine(max_workers=1).run(module, arguments)

    def check() -> None:
        np.testing.assert_array_equal(
            result.outputs["output_bool"], np.array([False, True, True, False])
        )
        for dtype in INTEGER_DTYPES:
            np_dtype = NUMPY_DTYPES[dtype]
            lhs = np.array(53, dtype=np_dtype)
            rhs = np.array(15, dtype=np_dtype)
            expected = np.empty(20, dtype=np_dtype)
            expected[0] = np.bitwise_and(lhs, rhs)
            expected[1] = np.bitwise_or(lhs, rhs)
            expected[2] = np.bitwise_xor(lhs, rhs)
            expected[3] = np.bitwise_not(lhs)
            expected[4::2] = np.left_shift(lhs, 1)
            expected[5::2] = np.right_shift(lhs, 1)
            np.testing.assert_array_equal(result.outputs[f"output_{_name(dtype)}"], expected)

    check()


def test_complete_if_then_else_domain_executes_both_branches(tmp_path: Path):
    module = numsim.transpile(DYNAMIC_IF_THEN_ELSE_RUNTIME_DOMAIN, cache_dir=tmp_path)
    arguments: dict[str, object] = {
        f"output_{_name(dtype)}": _empty_output(dtype, (2,)) for dtype in SCALAR_DTYPES
    }
    arguments.update(
        {
            "left": np.full(32, 0x35, dtype=np.uint32),
            "right": np.full(32, 0xA7, dtype=np.uint32),
            "pointer_output": np.zeros(34, dtype=np.uint32),
        }
    )
    result = numsim.Engine(max_workers=1).run(module, arguments)

    def check() -> None:
        np.testing.assert_array_equal(result.outputs["output_bool"], [True, False])
        for dtype in SCALAR_DTYPES:
            if dtype == "bool":
                continue
            actual = result.outputs[f"output_{_name(dtype)}"]
            if dtype == "bfloat16":
                expected = _bfloat16_bits(np.array([3, 7], dtype=np.float32)).astype(np.uint16)
            else:
                expected = np.array([3, 7], dtype=NUMPY_DTYPES[dtype])
            np.testing.assert_array_equal(actual, expected)
        alternating = np.where(np.arange(32) % 2 == 0, 0x35, 0xA7).astype(np.uint32)
        np.testing.assert_array_equal(
            result.outputs["pointer_output"],
            np.concatenate((alternating, np.array([0x35, 0xA7], dtype=np.uint32))),
        )

    check()


def test_if_then_else_mixed_pointer_spaces_fail_closed(tmp_path: Path):
    module = numsim.transpile(IF_THEN_ELSE_MIXED_POINTER_SPACES, cache_dir=tmp_path)
    with pytest.raises(
        numsim.NumSimExecutionError,
        match="does not match PTX state space global",
    ):
        numsim.Engine(max_workers=1).run(
            module,
            {
                "source": np.full(32, 0x35, dtype=np.uint32),
                "output": np.zeros(32, dtype=np.uint32),
            },
        )


def test_complete_tvm_shuffle_payload_domain_executes(tmp_path: Path):
    module = numsim.transpile(DYNAMIC_TVM_SHUFFLE_RUNTIME_DOMAIN, cache_dir=tmp_path)
    arguments = {f"output_{_name(dtype)}": _empty_output(dtype, (32, 4)) for dtype in SCALAR_DTYPES}
    result = numsim.Engine(max_workers=1).run(module, arguments)

    def check() -> None:
        lanes = np.arange(32)
        for dtype in SCALAR_DTYPES:
            if dtype == "bool":
                values = lanes % 2 == 0
            elif dtype == "bfloat16":
                values = _bfloat16_bits(lanes.astype(np.float32)).astype(np.uint16)
            else:
                values = lanes.astype(NUMPY_DTYPES[dtype])
            shuffle_up = np.concatenate((values[:1], values[:-1]))
            shuffle_down = np.concatenate((values[1:], values[-1:]))
            expected = np.stack(
                (values[::-1], values[lanes ^ 1], shuffle_up, shuffle_down),
                axis=1,
            )
            np.testing.assert_array_equal(result.outputs[f"output_{_name(dtype)}"], expected)

    check()
