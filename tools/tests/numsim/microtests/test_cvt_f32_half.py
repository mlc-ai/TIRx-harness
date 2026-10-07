"""Scalar half CVT uses a value-grid oracle independent of native ULP correction."""

import math

import numpy as np
from tvm.backend.cuda.ptx.table import TABLE, mods, operand_dtypes, variants

from tests.numsim.support.cvt_cases import append_cvt_mode, scalar_cvt_kernel
from tests.numsim.support.execution import run_checked


def _grid(destination):
    codes = np.arange(0x7C00 if destination == "f16" else 0x7F80, dtype=np.uint16)
    return (
        codes.view(np.float16).astype(np.float64)
        if destination == "f16"
        else (codes.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    )


def _encode(value, grid, rounding, relu, satfinite):
    if math.isnan(value):
        return 0x7FFF
    sign = 0x8000 if math.copysign(1, value) < 0 else 0
    if sign:
        rounding = {"rm": "rp", "rp": "rm"}.get(rounding, rounding)
    assert rounding in {"rn", "rz", "rm", "rp"}
    magnitude = abs(value)
    if math.isinf(value):
        code = len(grid)
    elif magnitude > grid[-1]:
        overflow = grid[-1] + (grid[-1] - grid[-2]) / 2
        code = len(grid) - (rounding in {"rz", "rm"} or (rounding == "rn" and magnitude < overflow))
    else:
        upper = int(np.searchsorted(grid, magnitude))
        lower = max(0, upper - 1)
        if rounding == "rp" or grid[upper] == magnitude:
            code = upper
        elif rounding == "rn":
            code = min((lower, upper), key=lambda index: (abs(grid[index] - magnitude), index & 1))
        else:
            code = lower
    if satfinite:
        code = min(code, len(grid) - 1)
    return 0 if relu and sign else sign | code


def _inputs():
    points = []
    for destination, mantissa in (("f16", 10), ("bf16", 7)):
        grid = _grid(destination)
        for code in (
            0,
            1,
            2,
            (1 << mantissa) - 1,
            1 << mantissa,
            len(grid) // 2,
            len(grid) // 2 + 1,
            len(grid) - 2,
        ):
            midpoint = np.float32((grid[code] + grid[code + 1]) / 2)
            points.extend((grid[code], grid[code + 1]))
            points.extend(
                np.nextafter(midpoint, np.float32(toward)) for toward in (-np.inf, np.inf)
            )
            points.append(midpoint)
        midpoint = np.float32(grid[-1] + (grid[-1] - grid[-2]) / 2)
        points.extend(
            (
                np.nextafter(midpoint, np.float32(0)),
                midpoint,
                np.nextafter(midpoint, np.float32(np.inf)),
            )
        )
    points.extend((1.0, 1.0005, 1e-40, 1e-8, 65504.0, np.finfo(np.float32).max))
    signed = np.asarray([value * sign for value in points for sign in (1, -1)], np.float32)
    # Construct special values by bits so signaling NaNs survive input setup.
    special = np.asarray(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00000,
            0x7F800001,
            0xFF800001,
        ],
        np.uint32,
    ).view(np.float32)
    return np.resize(np.concatenate((signed, special)), 256)


def _case():
    forms = [
        ("cvt." + ".".join(t for t in tokens if t), m)
        for tokens in variants(TABLE["cvt"])
        if (m := mods(TABLE["cvt"], tokens))["dtype"] in {"f16", "bf16"} and m["atype"] == "f32"
    ]
    assert len(forms) == 36
    raw = np.repeat(_inputs().view(np.uint32), 2)
    size = len(raw)
    source = raw.astype(np.uint64) | np.uint64(0xABCDEF1200000000)
    expected = np.empty((4 * len(forms), size), np.uint64)
    lines = []
    for index, (spelling, m) in enumerate(forms):
        grid = _grid(m["dtype"])
        converted = []
        for value_bits in raw:
            bits = int(value_bits)
            if m["ftz"] and bits & 0x7FFFFFFF < 0x800000:
                bits &= 0x80000000
            value = float(np.uint32(bits).view(np.float32))
            result = _encode(value, grid, m["rnd"], bool(m["relu"]), bool(m["satfinite"]))
            if m["sat"]:
                result = 0 if result & 0x8000 or np.isnan(value) else min(result, 0x3C00)
            converted.append(result)
        converted = np.asarray(converted, np.uint64)
        for mode in range(4):
            row = 4 * index + mode
            dst_type = (
                "uint64"
                if mode and "uint64" in operand_dtypes(TABLE["cvt"].operands[0], m)
                else "uint16"
            )
            wide_source = mode and "uint64" in operand_dtypes(TABLE["cvt"].operands[1], m)
            sentinel = 0xABCD01235A5A if dst_type == "uint64" else 0x5A5A
            operand = f"source[{size if mode == 3 else 'lane'}]"
            if not wide_source:
                operand = f'T.reinterpret("float32", T.cast({operand}, "uint32"))'
            append_cvt_mode(lines, expected, row, spelling, dst_type, operand, converted, sentinel)
    return scalar_cvt_kernel(source, expected, lines)


def test_scalar_half_modifiers_match_independent_grid():
    kernel, inputs, expected = _case()
    actual = run_checked(kernel, inputs).outputs["output"]
    np.testing.assert_array_equal(actual, expected)
