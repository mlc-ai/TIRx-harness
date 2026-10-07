"""Representative new register semantics, not another table/carrier census.

Baseline register tests own ordinary arithmetic and conversion coverage. Keep
bit-sensitive sign, packed-lane, saturation, OOB and three-input cases here.
"""

import numpy as np
import pytest

from tests.numsim.support.execution import run_checked
from tests.numsim.support.register_cases import masked_arithmetic_case


# PTX spelling, raw carrier, operand carrier, input vectors, expected bits.
CASES = [
    ("neg.f64", "uint64", "float64", [[0, 0x8000000000000000]], [0x8000000000000000, 0]),
    ("abs.f64", "uint64", "float64", [[0xBFF0000000000000, 0xFFF0000000000000]], [0x3FF0000000000000, 0x7FF0000000000000]),
    ("neg.ftz.f32", "uint32", "float32", [[1, 0x80000001]], [0x80000000, 0]),
    ("lg2.approx.f32", "uint32", "float32", [[0x3F800000, 0x40800000]], [0, 0x40000000]),
    ("rsqrt.approx.ftz.f64", "uint64", "float64", [[0x3FF0000000000000, 0x4010000000000000]], [0x3FF0000000000000, 0x3FE0000000000000]),
    ("min.f32", "uint32", "float32", [[0, 0x3F800000], [0x80000000, 0xBF800000]], [0x80000000, 0xBF800000]),
    ("max.f32", "uint32", "float32", [[0x80000000, 0xBF800000], [0, 0x3F800000]], [0, 0x3F800000]),
    ("min.abs.f32", "uint32", "float32", [[0xC0400000], [0x40000000], [0xBF800000]], [0x3F800000]),
    ("max.abs.f32", "uint32", "float32", [[0xC0400000], [0x40000000], [0xBF800000]], [0x40400000]),
    ("min.s16x2", "uint32", None, [[0x80000001], [0x0001FFFF]], [0x8000FFFF]),
    ("max.u16x2", "uint32", None, [[0x80000001], [0x0001FFFF]], [0x8000FFFF]),
    ("min.NaN.f32", "uint32", "float32", [[0x7FC00001], [0x3F800000]], [0x7FFFFFFF]),
]

for dtype, one, half in (("f16", 0x3C00, 0x3800), ("bf16", 0x3F80, 0x3F00)):
    for packed in (False, True):
        # Opposite signs in the two packed components expose lane swapping.
        def pack(low, high):
            return low | (high << 16) if packed else low

        suffix, carrier = dtype + ("x2" if packed else ""), "uint32" if packed else "uint16"
        positive, negative = pack(one, one | 0x8000), pack(one | 0x8000, one)
        cases = (
            (f"abs.{suffix}", [[negative]], [pack(one, one)]),
            (f"neg.{suffix}", [[positive]], [negative]),
            (f"add.rn.{suffix}", [[positive], [negative]], [0]),
            (f"sub.rn.{suffix}", [[0], [positive]], [negative]),
            (f"mul.rn.{suffix}", [[positive], [pack(half, half)]], [pack(half, half | 0x8000)]),
            (f"fma.rn.oob.relu.{suffix}", [[pack(0x7FF7, one)], [positive], [0]], [0]),
            (f"ex2.approx.{'' if dtype == 'f16' else 'ftz.'}{suffix}", [[0]], [pack(one, one)]),
        )
        CASES.extend((spelling, carrier, None, inputs, expected) for spelling, inputs, expected in cases)

CASES += [
    ("add.rn.sat.f16", "uint16", None, [[0xBC00, 0x4000], [0, 0x3C00]], [0, 0x3C00]),
    # Fused residual, not multiply rounded to half followed by add.
    ("fma.rn.f16", "uint16", None, [[0x3C01], [0x3BFF], [0xBC00]], [0x0FFE]),
    ("mul.rn.ftz.f16", "uint16", None, [[0x0400], [0x3BFF]], [0]),
]


GROUPS = sorted({(carrier, operand) for _, carrier, operand, _, _ in CASES}, key=str)


def register_cases(carrier, operand):
    # One artifact per carrier ABI, with all selected operations in that artifact.
    inputs = {"preserved": np.full(32, 0x55, carrier)}
    forms, expected = [], []
    for spelling, raw, typed, vectors, bits in CASES:
        if (raw, typed) != (carrier, operand):
            continue
        names = tuple(f"arg_{len(forms)}_{i}" for i in range(len(vectors)))
        inputs.update((name, np.resize(np.asarray(values, carrier), 32)) for name, values in zip(names, vectors))
        forms.append((spelling, names))
        expected.append(np.resize(np.asarray(bits, carrier), 32))
    expected = np.stack(expected)
    masked = expected.copy()
    masked[:, 1::2] = inputs["preserved"][1::2]
    kernel, inputs = masked_arithmetic_case(
        forms, inputs, preserved="preserved", operand_dtype=operand
    )
    return kernel, inputs, np.concatenate([expected, masked])


@pytest.mark.parametrize("carrier,operand", GROUPS)
def test_register_extensions(carrier, operand, tmp_path):
    kernel, inputs, expected = register_cases(carrier, operand)
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.numsim_gpu
def test_register_extensions_gpu(gpu_runner):
    for carrier, operand in GROUPS:
        kernel, inputs, expected = register_cases(carrier, operand)
        result = gpu_runner(kernel, inputs, outputs=("output",), arch="sm_100a")
        np.testing.assert_array_equal(result["output"], expected, err_msg=f"{carrier}/{operand}")
