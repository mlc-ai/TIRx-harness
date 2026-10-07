"""Integer-source saturation boundaries; generic CVT masks live in test_cvt_predicates."""

import numpy as np
import pytest

from tests.numsim.support.execution import run_checked
from tests.numsim.support.register_cases import masked_arithmetic_case


# One case per signedness/extension boundary, not every table spelling crossed
# with the same four predicate modes. Values are raw uint64 register carriers.
SAT_CASES = (
    ("cvt.sat.u8.s64", [-1, 0, 255, 256], [0, 0, 255, 255]),
    ("cvt.sat.s8.s64", [-129, -128, 127, 128], [-128, -128, 127, 127]),
    ("cvt.sat.s32.u64", [0, 2**31 - 1, 2**31, 2**64 - 1], [0, 2**31 - 1, 2**31 - 1, 2**31 - 1]),
    ("cvt.sat.u64.s32", [0xABCDEF01FFFFFFFF, 0xABCDEF017FFFFFFF], [0, 2**31 - 1]),
)
FLOAT_SAT_CASES = tuple(
    (f"cvt.rn.sat.{dtype}.s64", [-1, 0, 1, 2], [0, 0, one, one])
    for dtype, one in (("f16", 0x3C00), ("f32", 0x3F800000), ("f64", 0x3FF0000000000000))
)


@pytest.mark.parametrize(
    "floating", (False, True), ids=("integer_destination", "float_destination")
)
def test_integer_cvt_sat_clamps_before_narrowing_and_preserves_predicates(floating, tmp_path):
    inputs = {"preserved": np.full(32, 0x5A5A5A5A5A5A5A5A, np.uint64)}
    forms, expected = [], []

    def words(values):
        # Adjacent lanes hold the same boundary, exercising active and preserved results.
        return np.resize(np.repeat(np.array([v % 2**64 for v in values], np.uint64), 2), 32)

    for index, (spelling, values, bits) in enumerate(FLOAT_SAT_CASES if floating else SAT_CASES):
        name = f"source_{index}"
        inputs[name] = words(values)
        forms.append((spelling, (name,)))
        expected.append(words(bits))
    expected = np.stack(expected)
    masked = expected.copy()
    masked[:, 1::2] = inputs["preserved"][1::2]
    kernel, inputs = masked_arithmetic_case(forms, inputs, preserved="preserved")
    actual = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(actual.outputs["output"], np.concatenate((expected, masked)))
