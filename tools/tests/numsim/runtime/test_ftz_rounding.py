"""FTZ detects tiny results before rounding can promote them to normal."""

import math
from fractions import Fraction

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.runtime.test_ptx_float_register_ops import _round_fraction
from tests.numsim.support.execution import run_checked


FORMS = tuple(
    (op, mode, ftz)
    for op in ("div", "mul", "fma", "rcp", "mul2", "fma2")
    for mode in ("rn", "rz", "rm", "rp")
    for ftz in (False, True)
)


def ftz_boundary_case():
    lines = []
    for row, (op, mode, ftz) in enumerate(FORMS):
        packed = op.endswith("2")
        name = op.removesuffix("2")
        spelling = f"{name}.{mode}.{'ftz.' if ftz else ''}{'f32x2' if packed else 'f32'}"
        if packed:
            args = "packed_a[0], T.uint64(0x0080000000800000)"
            if name == "fma":
                args += ", T.uint64(0)"
            lines += [
                f'    T.ptx["{spelling}"](packed_d[0], {args})',
                f'    output[{row}, lane, 0] = T.reinterpret("float32", T.Cast("uint32", packed_d[0]))',
                f'    output[{row}, lane, 1] = T.reinterpret("float32", T.Cast("uint32", packed_d[0] >> T.uint64(32)))',
            ]
        else:
            args = {
                "div": "A[lane], B[lane]",
                "mul": "A[lane], T.float32(2.0**-126)",
                "fma": "A[lane], T.float32(2.0**-126), T.float32(0)",
                "rcp": "B[lane]",
            }[op]
            lines += [
                f'    T.ptx["{spelling}"](output[{row}, lane, 0], {args})',
                f"    output[{row}, lane, 1] = output[{row}, lane, 0]",
            ]
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def kernel(A: T.Buffer((32,), "float32"), B: T.Buffer((32,), "float32"),
           output: T.Buffer(({len(FORMS)}, 32, 2), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    packed_a = T.alloc_local((1,), "uint64")
    packed_d = T.alloc_local((1,), "uint64")
    bits = T.Cast("uint64", T.reinterpret("uint32", A[lane]))
    packed_a[0] = bits | ((bits ^ T.uint64(0x80000000)) << T.uint64(32))
{chr(10).join(lines)}
""",
        {"T": T},
    )
    inputs = {
        "A": np.resize(
            np.asarray(
                [
                    0x3F7FFFFE,
                    0x3F7FFFFF,
                    0x3F800000,
                    0x3F800001,
                    0xBF7FFFFE,
                    0xBF7FFFFF,
                    0xBF800000,
                    0xBF800001,
                ],
                np.uint32,
            ).view(np.float32),
            32,
        ),
        "B": np.resize(
            np.asarray([0x7E800000] * 8 + [0x7E800001] * 8, np.uint32).view(np.float32), 32
        ),
        "output": np.zeros((len(FORMS), 32, 2), np.float32),
    }
    expected = np.zeros_like(inputs["output"])
    for row, (op, mode, ftz) in enumerate(FORMS):
        for lane, (a, b) in enumerate(zip(inputs["A"], inputs["B"])):
            if op == "rcp":
                a = np.float32(1)
            elif op != "div":
                b = np.float32(2.0**126)
            for half in range(2):
                numerator = -a if half and op.endswith("2") else a
                exact = Fraction(float(numerator)) / Fraction(float(b))
                expected[row, lane, half] = (
                    math.copysign(0.0, numerator)
                    if ftz and abs(exact) < Fraction(1, 2**126)
                    else _round_fraction(exact, "float32", mode)
                )
    return kernel, inputs, expected


def test_ftz_rounding_boundary(tmp_path):
    kernel, inputs, expected = ftz_boundary_case()
    actual = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(
        actual.outputs["output"].view(np.uint32), expected.view(np.uint32)
    )


@pytest.mark.numsim_gpu
def test_ftz_rounding_boundary_gpu(gpu_runner):
    kernel, inputs, expected = ftz_boundary_case()
    actual = gpu_runner(kernel, inputs, outputs=("output",), arch="sm_100a")
    np.testing.assert_array_equal(actual["output"].view(np.uint32), expected.view(np.uint32))
