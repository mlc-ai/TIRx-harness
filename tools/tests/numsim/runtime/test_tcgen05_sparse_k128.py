"""SM107 sparse MXF8 K128 extends payload and metadata, not the MMA lifecycle.

SM107 hardware runs sparse K=128 only with a 6- or 4-bit B operand and raises an
illegal instruction for E4M3/E5M2 B, whatever the A format, scale IDs or
transposes; NumSim rejects those forms.
"""

import numpy as np
import pytest

from tests.numsim.runtime.test_tcgen05_block_tmem import block_tmem_case
from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel
from tests.numsim.support.execution import assert_rejected, run_checked

# cta, tmem_a, A format, B format, transpose A, transpose B
SPARSE_K128_CASES = [
    (1, True, 3, 4, False, False),
    (2, False, 5, 3, False, False),
    (2, True, 4, 5, False, False),
    (1, False, 0, 3, False, False),
    (1, False, 1, 3, True, False),
    (2, False, 0, 3, False, False),
    (2, True, 1, 4, False, False),
    (1, True, 5, 3, False, False),
    (2, True, 5, 4, False, False),
]
# 8-bit B traps on SM107, including MN-major B (which requires 8-bit B).
SPARSE_K128_8BIT_B_CASES = [
    (1, False, 0, 1, False, False),
    (1, False, 1, 0, True, True),
    (2, True, 3, 1, False, True),
]


def sparse_k128_case(cta, tmem, af, bf, ta=False, tb=False):
    m, n, rows = 128 * cta, 16 * cta, 128
    _, args, _ = block_tmem_case(cta, af, bf, tmem_a=tmem, k=64)
    # Each half B uses the already independent contiguous K64 narrow encoding.
    payload = (8, 8, 0, 6, 6, 4)[bf] * 8
    packed = args["b"].view(np.uint8)[:, :payload]
    storage = np.full((n, 128), 0xA5, np.uint8)
    storage[:, :payload] = storage[:, payload : 2 * payload] = packed
    args["b"] = storage.view(np.uint16)
    row, inner = np.indices((m, 64))
    a = ((row + inner * 3) % 4 + 1).astype(np.float32)
    a[row % 3 == 0] *= -1
    brow, bk = np.indices((n, 128))
    b = ((brow * 3 + bk) % 4 + 1).astype(np.float32)
    metadata = np.zeros((2, 128, 4), np.uint32)
    expanded = np.zeros((m, 128), np.float32)
    codes = (0x4, 0x8, 0xC, 0x9, 0xD, 0x6, 0xE)
    for row in range(m):
        for chunk in range(32):
            code = codes[(row * 3 + chunk) % len(codes)]
            metadata[row // rows, row % rows, chunk // 8] |= np.uint32(code << (4 * (chunk % 8)))
            expanded[row, chunk * 4 + (code & 3)] = a[row, chunk * 2]
            expanded[row, chunk * 4 + (code >> 2)] = a[row, chunk * 2 + 1]
    args["metadata"] = np.concatenate((args["metadata"], metadata), axis=2)
    expanded *= np.exp2((np.arange(m) % 32 % 3 - 1 + np.arange(m) // rows)[:, None])
    b *= np.exp2((np.arange(n) % 2)[:, None])
    expected = expanded @ b.T + args["seed"].view(np.float32)
    kernel = ti16_kernel(
        True,
        tmem,
        cta,
        m,
        kind="mxf8f6f4",
        sparse=True,
        a_format=af,
        b_format=bf,
        mma_k=64,
        sparsity_selector=0,
        transpose_a=ta,
        transpose_b=tb,
        implicit_scale=True,
        collectors=".collector::a::discard.collector::b::discard",
        arch="sm_107a",
    )
    return kernel, args, expected


@pytest.mark.parametrize("case", SPARSE_K128_CASES)
def test_sparse_k128_packing_scales_and_transpose(case, tmp_path):
    kernel, args, expected = sparse_k128_case(*case)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"].view(np.float32), expected)


@pytest.mark.parametrize("case", SPARSE_K128_8BIT_B_CASES)
def test_sparse_k128_rejects_8bit_b(case, tmp_path):
    kernel, args, _ = sparse_k128_case(*case)
    assert_rejected(
        kernel, args, "sparse K=128 requires an E2M3, E3M2 or E2M1 B operand", cache_dir=tmp_path
    )
