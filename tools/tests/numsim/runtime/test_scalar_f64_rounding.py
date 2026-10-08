"""FP64 directed rounding and fused residuals, with rational and GPU oracles."""

from fractions import Fraction

import numpy as np
import pytest

from tests.numsim.runtime.test_ptx_float_register_ops import _round_fraction
from tests.numsim.support.execution import run_checked
from tests.numsim.support.register_cases import masked_arithmetic_case

OPS = ("add", "sub", "mul", "fma", "mad", "div")


def scalar_f64_case(op):
    # Small increments, fused residuals, signed underflow and exact cancellation.
    # Keep ordinary special-value coverage in the existing scalar tests.
    columns = (
        [1, -1, 1 + 2**-27, -1 - 2**-27, 2**-1022, -(2**-1022), 1, -1],
        [2**-54, 2**-54, 1 - 2**-27, 1 - 2**-27, 0.5, 0.5, 3, 3],
        [-1, 1, -1, 1, 0, 0, -3, 3],
    )
    inputs = {
        name: np.resize(np.repeat(values, 2), 32).astype(np.float64)
        for name, values in zip(("A", "B", "C"), columns)
    }
    modes = (
        ("rn", "rz", "rm", "rp", "") if op in ("add", "sub", "mul") else ("rn", "rz", "rm", "rp")
    )
    sources = ("A", "B", "C") if op in ("fma", "mad") else ("A", "B")
    forms = [(".".join(filter(None, (op, mode, "f64"))), sources) for mode in modes]
    expected = []
    for mode in modes:
        row = []
        for a, b, c in zip(*(inputs[name] for name in ("A", "B", "C"))):
            a, b, c = map(Fraction, (float(a), float(b), float(c)))
            exact = {
                "add": a + b,
                "sub": a - b,
                "mul": a * b,
                "div": a / b,
                "fma": a * b + c,
                "mad": a * b + c,
            }[op]
            # All exact cancellations here have opposite-sign nonzero terms.
            row.append(
                _round_fraction(exact, "float64", mode) if exact else -0.0 if mode == "rm" else 0.0
            )
        expected.append(row)
    expected = np.array(expected, np.float64)
    masked = expected.copy()
    masked[:, 1::2] = inputs["C"][1::2]
    kernel, inputs = masked_arithmetic_case(forms, inputs, preserved="C", operand_dtype="float64")
    return kernel, inputs, np.concatenate((expected, masked))


@pytest.mark.parametrize("op", OPS)
def test_scalar_f64_rounding(op, tmp_path):
    kernel, inputs, expected = scalar_f64_case(op)
    actual = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(
        actual.outputs["output"].view(np.uint64), expected.view(np.uint64)
    )


@pytest.mark.numsim_gpu
def test_scalar_f64_rounding_gpu(gpu_runner):
    for op in OPS:
        kernel, inputs, expected = scalar_f64_case(op)
        actual = gpu_runner(kernel, inputs, outputs=("output",), arch="sm_100a")
        np.testing.assert_array_equal(actual["output"].view(np.uint64), expected.view(np.uint64))
