from __future__ import annotations

import re
import subprocess

import numpy as np
import pytest
from tvm import tirx

from tirx_harness import numsim
from tests.numsim.support.kernels import fixed_width_integer_expression_mix
from tests.numsim.support.manifest import device_kernel, emitted_module, parse_kernel
from tirx_harness.numsim.transpiler.cache import rust_tool
from tvm.script import tirx as T


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_float_immediates_preserve_bits_through_native_rust_literals(tmp_path, dtype):
    unsigned = np.dtype("uint32" if dtype == "float32" else "uint64")
    floating = np.dtype(dtype)
    info = np.finfo(floating)
    # Include both zeros, subnormal/normal boundaries, decimal-rounding ties,
    # extreme exponents, and reproducible bit patterns away from those edges.
    edges = np.array(
        [
            0.0,
            -0.0,
            info.smallest_subnormal,
            -info.smallest_subnormal,
            info.tiny,
            np.nextafter(floating.type(info.tiny), floating.type(0)),
            info.max,
            -info.max,
            2**-24,
            2**-25,
            1e-8,
            -1e-8,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=floating,
    )
    random = np.frombuffer(np.random.default_rng(1947).bytes(32 * unsigned.itemsize), unsigned)
    random = random.view(floating)
    expected = np.concatenate([edges, random[np.isfinite(random)]])

    def kernel(values):
        output = tirx.decl_buffer((len(values),), dtype, name="output")
        body = tirx.SeqStmt(
            [
                tirx.BufferStore(output, tirx.FloatImm(dtype, float(value)), [index])
                for index, value in enumerate(values)
            ]
        )
        return device_kernel(body, (output,))

    rust_type = "f32" if dtype == "float32" else "f64"
    literals = re.findall(
        rf"WarpValue::splat\(([^\n()]+(?:_{rust_type}|::(?:(?:NEG_)?INFINITY|NAN)))\)",
        emitted_module(kernel(expected)),
    )
    assert len(literals) == len(expected)
    bits = ", ".join(f"{int(value)}_u{unsigned.itemsize * 8}" for value in expected.view(unsigned))
    source = tmp_path / "literals.rs"
    source.write_text(
        f"fn main() {{ let values: [{rust_type}; {len(expected)}] = [{', '.join(literals)}];"
        f" assert_eq!(values.map({rust_type}::to_bits), [{bits}]); }}"
    )
    executable = tmp_path / "literals"
    subprocess.run(
        [rust_tool("rustc"), "--edition=2021", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)

    # Cache serialization preserves floating graph values without a JSON parse.
    module = numsim.transpile(kernel(expected), cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros_like(expected)})
    assert result.verdict == "clean", result.diagnostics
    np.testing.assert_array_equal(result.outputs["output"].view(unsigned), expected.view(unsigned))


@T.prim_func
def low_precision_expression_chain(
    fp16_a: T.Buffer((32,), "float16"),
    fp16_b: T.Buffer((32,), "float16"),
    fp16_c: T.Buffer((32,), "float16"),
    bf16_a: T.Buffer((32,), "bfloat16"),
    bf16_b: T.Buffer((32,), "bfloat16"),
    bf16_c: T.Buffer((32,), "bfloat16"),
    fp16_output: T.Buffer((32,), "float32"),
    bf16_output: T.Buffer((32,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    fp16_output[lane] = T.cast((fp16_a[lane] + fp16_b[lane]) + fp16_c[lane], "float32")
    bf16_output[lane] = T.cast((bf16_a[lane] + bf16_b[lane]) + bf16_c[lane], "float32")


@T.prim_func
def truncating_integer_expression_mix(output: T.Buffer((32, 3), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    dividend: T.let = lane - T.int32(7)
    output[lane, 0] = T.truncdiv(dividend, T.int32(2))
    output[lane, 1] = T.truncmod(dividend, T.int32(2))
    output[lane, 2] = T.floordiv(dividend, T.int32(2))


@T.prim_func
def captured_expression_inputs(
    source: T.Buffer((32,), "int32"),
    scratch: T.Buffer((32,), "int32"),
    output: T.Buffer((32, 6), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane, 0] = source[lane] * T.int32(2) + T.int32(2)
    output[lane, 1] = source[lane] * T.int32(2) + T.int32(3)
    scratch[lane] = source[lane] * T.int32(3) + T.int32(2)
    output[lane, 2] = scratch[lane] * T.int32(2) + T.int32(3)
    scratch[lane] = source[lane] * T.int32(-3) + T.int32(4)
    output[lane, 3] = scratch[lane] * T.int32(2) + T.int32(3)
    output[lane, 4] = T.cast((source[lane] & T.int32(31)) < T.int32(17), "int32")
    output[lane, 5] = T.if_then_else(
        lane < 16,
        source[lane] * T.int32(2) + T.int32(7),
        source[lane] * T.int32(-2) + T.int32(5),
    )


def test_expression_captures_preserve_constants_load_order_and_wrapping(tmp_path):
    source = np.resize(
        np.array([0, 1, -1, 2**30, -(2**30), 2**31 - 1, -(2**31), 17], dtype=np.int32),
        32,
    )
    wide = source.astype(np.int64)
    expected = np.stack(
        [
            wide * 2 + 2,
            wide * 2 + 3,
            (wide * 3 + 2) * 2 + 3,
            (wide * -3 + 4) * 2 + 3,
            (wide & 31) < 17,
            np.where(np.arange(32) < 16, wide * 2 + 7, wide * -2 + 5),
        ],
        axis=1,
    ).astype(np.int32)
    module = numsim.transpile(
        captured_expression_inputs,
        cache_dir=tmp_path,
    )
    result = numsim.Engine().run(
        module,
        {
            "source": source,
            "scratch": np.zeros(32, dtype=np.int32),
            "output": np.zeros((32, 6), dtype=np.int32),
        },
    )
    assert result.verdict == "clean", result.diagnostics
    np.testing.assert_array_equal(result.outputs["output"], expected)


def _body(statements: str, *parameters: str) -> str:
    signature = ", ".join(parameters)
    return emitted_module(
        parse_kernel(
            f"""
@T.prim_func
def kernel({signature}):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
{statements}
"""
        )
    )


def _lines_with(body: str, fragment: str) -> list[str]:
    return [line.strip() for line in body.splitlines() if fragment in line]


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype", "target_rust_type"),
    [
        (source_dtype, target_dtype, target_rust_type)
        for source_dtype in ("int32", "int64", "uint32", "uint64")
        for target_dtype, target_rust_type in (
            ("int32", "i32"),
            ("int64", "i64"),
            ("uint32", "u32"),
            ("uint64", "u64"),
        )
    ],
)
def test_uniform_integer_casts_preserve_target_width(
    source_dtype: str, target_dtype: str, target_rust_type: str
):
    body = _body(
        f'    output[lane] = T.cast(value, "{target_dtype}")',
        f"value: T.{source_dtype}",
        f'output: T.Buffer((32,), "{target_dtype}")',
    )

    splats = _lines_with(body, "WarpValue::splat(")
    assert len(splats) == 1
    if source_dtype != target_dtype:
        assert f"as {target_rust_type}" in splats[0]
    else:
        assert " as " not in splats[0]


def test_varying_casts_and_arithmetic_keep_fixed_width_types():
    body = _body(
        """    as_u32: T.uint32 = T.cast(lane, "uint32")
    as_u64: T.uint64 = T.cast(as_u32, "uint64")
    output[lane] = T.cast(as_u64 + T.uint64(3) < T.uint64(10), "uint32")""",
        'output: T.Buffer((32,), "uint32")',
    )

    assert "frontend_expr::i32::cast::u32(" in body
    assert "frontend_expr::u32::cast::u64(" in body
    assert "frontend_expr::u64::add::vs(" in body
    assert "10_u64" in body
    assert "WarpMask::from_predicate" in body


def test_varying_mask_algebra_uses_native_bit_operations():
    body = _body(
        """    left: T.bool = lane < 16
    right: T.bool = lane % 2 == 0
    output[lane] = T.cast(left and right, "uint32")
    output[lane] = T.cast(left or right, "uint32")
    output[lane] = T.cast(not left, "uint32")""",
        'output: T.Buffer((32,), "uint32")',
    )

    assert "let and_rhs_mask_" in body
    assert "ctx.set_active_mask(and_rhs_mask_" in body
    assert "let or_rhs_mask_" in body
    assert "ctx.set_active_mask(or_rhs_mask_" in body
    assert re.search(r"if \(\(!\(\w+\)\)\)\.contains\(lane\)", body)


def test_composite_mask_is_atomic_when_embedded_in_select_branch_masks():
    body = _body(
        """    left: T.bool = lane < 16
    right: T.bool = lane % 2 == 0
    output[lane] = T.Select(left or right, T.int32(1), T.int32(0))""",
        'output: T.Buffer((32,), "int32")',
    )

    assert "ctx.set_active_mask(or_rhs_mask_" in body
    assert "select_then_mask_" in body and "or_result_" in body
    assert "select_else_mask_" in body and "or_result_" in body


@pytest.mark.parametrize(("kind", "operator"), [("And", "and"), ("Or", "or")])
@pytest.mark.parametrize("uniform_first", [False, True])
def test_varying_mask_and_uniform_boolean_avoid_lane_scan(
    kind: str, operator: str, uniform_first: bool
):
    lanes = "(lane < 16)"
    enabled = "(flag != 0)"
    lhs, rhs = (enabled, lanes) if uniform_first else (lanes, enabled)
    body = _body(
        f'    output[lane] = T.cast({lhs} {operator} {rhs}, "uint32")',
        "flag: T.int32",
        'output: T.Buffer((32,), "uint32")',
    )

    assert f"ctx.set_active_mask({kind.lower()}_rhs_mask_" in body
    assert "WarpMask::from_lanes" not in body
    if uniform_first:
        assert re.search(rf"= if .* \{{ {kind.lower()}_parent_\d+ \}}", body)
    else:
        mask_operator = "&" if kind == "And" else "-"
        assert re.search(
            rf"= {kind.lower()}_parent_\d+ {re.escape(mask_operator)} \(\w+\);",
            body,
        )


@pytest.mark.parametrize(
    ("dtype", "rust_type"),
    [("int32", "i32"), ("int64", "i64"), ("uint32", "u32"), ("uint64", "u64")],
)
def test_fixed_width_arithmetic_and_comparison_use_the_tir_dtype(dtype: str, rust_type: str):
    body = _body(
        f"""    output[lane] = value + T.{dtype}(3)
    compared[lane] = T.cast(value >= T.{dtype}(7), "uint32")""",
        f"value: T.{dtype}",
        f'output: T.Buffer((32,), "{dtype}")',
        'compared: T.Buffer((32,), "uint32")',
    )

    added = _lines_with(body, ".wrapping_add(")
    assert len(added) == 1
    assert f"3_{rust_type}" in added[0]
    compared = [line for line in _lines_with(body, "WarpValue::splat(") if ">=" in line]
    assert len(compared) == 1
    assert f"7_{rust_type}" in compared[0]


@pytest.mark.parametrize(
    ("source_dtype", "source_rust_type", "target_dtype", "target_rust_type"),
    [
        ("int32", "i32", "uint64", "u64"),
        ("int64", "i64", "uint64", "u64"),
        ("uint32", "u32", "int32", "i32"),
        ("uint64", "u64", "int64", "i64"),
    ],
)
def test_varying_integer_casts_preserve_target_width(
    source_dtype: str, source_rust_type: str, target_dtype: str, target_rust_type: str
):
    body = _body(
        f'    output[lane] = T.cast(values[lane], "{target_dtype}")',
        f'values: T.Buffer((32,), "{source_dtype}")',
        f'output: T.Buffer((32,), "{target_dtype}")',
    )

    casts = _lines_with(body, f"frontend_expr::{source_rust_type}::cast::{target_rust_type}(")
    assert casts
    assert f"v2::reg::variant::{source_rust_type.upper()}" in body


@pytest.mark.parametrize(
    ("kind", "method"), [("floordiv", "checked_div"), ("floormod", "checked_rem")]
)
def test_unsigned_floor_operations_keep_uint32_semantics(kind: str, method: str):
    body = _body(
        f"    output[lane] = T.{kind}(value, T.uint32(5))",
        "value: T.uint32",
        'output: T.Buffer((32,), "uint32")',
    )

    operations = _lines_with(body, f".{method}(")
    assert len(operations) == 1
    assert "5_u32" in operations[0]
    assert "as i64" not in operations[0]


@pytest.mark.parametrize(("kind", "method"), [("truncdiv", "checked_div"), ("truncmod", "checked_rem")])
def test_truncating_integer_operations_keep_checked_overflow_semantics(kind: str, method: str):
    body = _body(
        f"    output[lane] = T.{kind}(value, T.int32(-1))",
        "value: T.int32",
        'output: T.Buffer((32,), "int32")',
    )

    operations = _lines_with(body, f".{method}(")
    assert len(operations) == 1
    assert "-1_i32" in operations[0]
    assert "EngineError::message" in operations[0]


def test_float_division_uses_native_f32_semantics():
    body = _body(
        "    output[lane] = values[lane] / T.float32(2.0)",
        'values: T.Buffer((32,), "float32")',
        'output: T.Buffer((32,), "float32")',
    )

    divisions = _lines_with(body, "frontend_expr::f32::div::")
    assert len(divisions) == 1
    assert "2.0_f32" in divisions[0]
    assert "checked_div" not in body


def test_signed_index_adapter_is_explicit_for_varying_unsigned_values():
    body = _body(
        "    output[lane] = values[values[lane]]",
        'values: T.Buffer((32,), "uint64")',
        'output: T.Buffer((32,), "uint64")',
    )

    assert re.search(r"frontend_expr::u64::cast::i64\(&load_\d+\)", body)


def test_integer_narrowing_and_boolean_casts_have_typed_rust_code():
    body = _body(
        """    as_i8[lane] = T.cast(signed, "int8")
    as_u16[lane] = T.cast(unsigned, "uint16")
    as_bool[lane] = T.cast(T.cast(unsigned, "bool"), "uint32")""",
        "signed: T.int32",
        "unsigned: T.uint32",
        'as_i8: T.Buffer((32,), "int8")',
        'as_u16: T.Buffer((32,), "uint16")',
        'as_bool: T.Buffer((32,), "uint32")',
    )

    assert re.search(r"\(buffers\.scalar_\d+\) as i8", body)
    assert re.search(r"\(buffers\.scalar_\d+\) as u16", body)
    assert re.search(r"if \(buffers\.scalar_\d+\) != 0_u32 \{ 1_u32 \} else \{ 0_u32 \}", body)


def test_bfloat_cast_rounds_through_the_engine_codec():
    body = _body(
        '    output[lane] = T.cast(values[lane], "bfloat16")',
        'values: T.Buffer((32,), "float32")',
        'output: T.Buffer((32,), "bfloat16")',
    )

    assert "f32_to_bf16_bits" in body
    assert "bf16_bits_to_f32" in body


@pytest.mark.parametrize(
    ("dtype", "decoder", "encoder"),
    [
        ("float8_e4m3fn", "float8_e4m3fn_bits_to_f32", "f32_to_float8_e4m3fn_bits"),
        ("float8_e8m0fnu", "float8_e8m0fnu_bits_to_f32", "f32_to_float8_e8m0fnu_bits"),
    ],
)
def test_float8_immediates_use_numeric_codec(dtype: str, decoder: str, encoder: str):
    body = _body(
        f"    output[lane] = T.{dtype}(1.0)",
        f'output: T.Buffer((32,), "{dtype}")',
    )

    immediates = [line for line in _lines_with(body, "1.0_f32") if "splat" in line]
    assert len(immediates) == 1
    assert decoder in immediates[0]
    assert encoder in immediates[0]


def test_fixed_width_integer_operations_preserve_casts_and_wrap_observably(tmp_path):
    output = np.zeros((32, 7), dtype=np.uint64)

    module = numsim.transpile(fixed_width_integer_expression_mix, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = []
    for lane in range(32):
        unsigned = (lane - 1) & ((1 << 32) - 1)
        expected.append(
            [
                unsigned + 1 if unsigned else (1 << 64) - 1,
                (lane - 1) & ((1 << 64) - 1),
                unsigned & 0xFFFF,
                int(unsigned != 0),
                unsigned // 5,
                unsigned % 5,
                int(unsigned + 3 < 10),
            ]
        )
    expected = np.array(expected, dtype=np.uint64)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_truncating_integer_division_and_remainder_match_tir_semantics(tmp_path):
    output = np.zeros((32, 3), dtype=np.int32)

    module = numsim.transpile(truncating_integer_expression_mix, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    dividend = np.arange(32, dtype=np.int32) - 7
    quotient_magnitude = np.abs(dividend) // 2
    quotient = np.where(dividend < 0, -quotient_magnitude, quotient_magnitude)
    floor = np.floor_divide(dividend, np.int32(2))
    expected = np.stack((quotient, dividend - quotient * 2, floor), axis=1).astype(np.int32)
    assert tuple(expected[0, (0, 2)]) == (-3, -4)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_low_precision_arithmetic_rounds_at_every_expression_node(tmp_path):
    fp16_one = np.ones(32, dtype=np.float16)
    fp16_half_ulp = np.full(32, 2**-11, dtype=np.float16)
    bf16_one = np.full(32, np.uint16(0x3F80), dtype=np.uint16)
    bf16_half_ulp = np.full(32, np.uint16(0x3B80), dtype=np.uint16)
    fp16_output = np.zeros(32, dtype=np.float32)
    bf16_output = np.zeros(32, dtype=np.float32)

    module = numsim.transpile(low_precision_expression_chain, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "fp16_a": fp16_one,
            "fp16_b": fp16_half_ulp,
            "fp16_c": fp16_half_ulp,
            "bf16_a": bf16_one,
            "bf16_b": bf16_half_ulp,
            "bf16_c": bf16_half_ulp,
            "fp16_output": fp16_output,
            "bf16_output": bf16_output,
        },
    )

    expected = np.ones(32, dtype=np.float32)
    np.testing.assert_array_equal(result.outputs["fp16_output"], expected)
    np.testing.assert_array_equal(result.outputs["bf16_output"], expected)
    assert module.rust_source.count("f32_to_fp16_bits") >= 2
    assert module.rust_source.count("f32_to_bf16_bits") >= 2
