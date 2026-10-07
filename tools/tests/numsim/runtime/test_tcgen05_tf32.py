"""Sparse TF32 metadata for both CTA layouts and shared operand transposes."""

import numpy as np
import pytest

from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel
from tests.numsim.support.execution import run_checked


def tf32_case(cta, m, tmem, transpose_a, transpose_b, selector):
    n = 16 * cta
    bank, row, inner = np.indices((4, m, 8))
    a = (((bank * 3 + row - inner) % 9 - 4) / 8).astype(np.float32)
    brow, binner = np.indices((n, 16))
    b = (((brow * 2 + binner) % 11 - 5) / 16).astype(np.float32)
    seed = ((np.arange(m * n).reshape(m, n) % 7) / 16).astype(np.float32)
    metadata = np.zeros((2, 128, 2), np.uint32)
    expanded = np.zeros((4, m, 16), np.float32)
    for logical_row in range(m):
        local_row = logical_row % (m // cta)
        physical_row = local_row // 16 * 32 + local_row % 16 if m // cta == 64 else local_row
        for chunk in range(8):
            index = (logical_row + chunk // 2) % 2
            expanded[:, logical_row, chunk * 2 + index] = a[:, logical_row, chunk]
            lane = (
                physical_row // 32 * 32
                + physical_row % 8
                + physical_row % 32 // 16 * 16
                + chunk // 4 * 8
            )
            shift = physical_row % 16 // 8 * 16 + chunk % 4 * 4
            metadata[logical_row // (m // cta), lane, selector] |= np.uint32(
                (0xE if index else 0x4) << shift
            )
    expected = expanded[0] @ b.T + seed
    if cta == 2:
        expected[m // 2 + 1] = seed[m // 2 + 1]
    args = {
        "a": a.view(np.uint16),
        "b": b.view(np.uint16),
        "seed": seed.view(np.int32),
        "out": np.zeros((m, n), np.int32),
        "zero_mask": np.zeros(1, np.uint64),
        "metadata": metadata,
    }
    return (
        ti16_kernel(
            True,
            tmem,
            cta,
            m,
            kind="tf32",
            arch="sm_107a",
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            sparse=True,
            sparsity_selector=selector,
            collectors=".collector::a::discard.collector::b::discard",
        ),
        args,
        expected,
    )


# Cover both metadata selectors, CTA layouts and shared operand transposition
# without taking a Cartesian product of independent boundaries.
TF32_SPARSE_CASES = [
    (1, 128, True, False, False),
    (2, 128, False, False, False),
    (1, 64, False, True, True),
    (2, 256, True, False, True),
]


@pytest.mark.parametrize("case", TF32_SPARSE_CASES)
def test_tf32_sparse_metadata(case, tmp_path):
    kernel, args, expected = tf32_case(*case, int(not case[2]))
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"].view(np.float32), expected)
