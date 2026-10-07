"""Sparse narrow MMA reuses packed codecs, with one metadata row per datapath."""

import numpy as np
import pytest

from tests.numsim.runtime.test_tcgen05_block_tmem import block_tmem_case
from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel
from tests.numsim.support.execution import run_checked

SPARSE_NARROW_CASES = [
    (1, 64, False, 0, 1, False),
    (1, 128, True, 1, 0, True),
    (2, 128, False, 3, 4, False),
    (2, 256, True, 4, 5, True),
    (1, 128, False, 5, 3, False),
    (1, 64, False, 0, 1, False, True, True),
]


def sparse_narrow_case(
    cta,
    m,
    tmem,
    af,
    bf,
    half,
    transpose_a=False,
    transpose_b=False,
    *,
    block_scale=False,
    packed_k=32,
):
    assert packed_k in (32, 64)
    _, args, _ = block_tmem_case(cta, af, bf, block_scale=False, tmem_a=tmem, m=m, k=32)
    # Ordinary sparse A retains padded shared atoms / padded TMEM bytes.
    # Extending its row is not the contiguous block-scaled K64 packing.
    if packed_k == 64:
        args["a"] = np.concatenate((args["a"], np.roll(args["a"], 1, axis=1)), axis=2)
    _, b_args, _ = block_tmem_case(cta, af, bf, block_scale=False, tmem_a=tmem, m=m, k=64)
    # Ordinary sparse B keeps each group of16 narrow values in a16-byte atom,
    # unlike the contiguous K64 block-scaled fixture used as our encoding source.
    payload_bytes = (8, 8, 0, 6, 6, 4)[bf] * 2
    packed_b = b_args["b"].view(np.uint8)
    padded_b = np.full_like(packed_b, 0xA5)
    for atom in range(4):
        padded_b[:, atom * 16 : atom * 16 + payload_bytes] = packed_b[
            :, atom * payload_bytes : (atom + 1) * payload_bytes
        ]
    args["b"] = (
        padded_b
        if packed_k == 32
        else np.concatenate((padded_b, np.roll(padded_b, 1, axis=0)), axis=1)
    ).view(np.uint16)
    rows, n = m // cta, 16 * cta
    row, inner = np.indices((m, packed_k))
    packed_a = ((row + inner * 3) % 4 + 1).astype(np.float32)
    packed_a[row % 3 == 0] *= -1
    b_row, b_k = np.indices((n, packed_k * 2))
    b = ((b_row * 3 + b_k) % 4 + 1).astype(np.float32)
    if packed_k == 64:
        # Different K halves detect accidental reuse of the first A/B half.
        packed_a[:, 32:] = np.roll(packed_a[:, :32], 1, axis=0)
        b[:, 64:] = np.roll(b[:, :64], 1, axis=0)
    codes = (0x4, 0x8, 0xC, 0x9, 0xD, 0x6, 0xE)
    metadata = np.zeros((2, 128, packed_k // 16), np.uint32)
    expanded = np.zeros((m, packed_k * 2), np.float32)
    for row in range(m):
        local_row = row % rows
        physical_row = local_row if rows == 128 else local_row // 16 * 32 + local_row % 16
        for chunk in range(packed_k // 2):
            code = codes[(row * 3 + chunk) % len(codes)]
            metadata[row // rows, physical_row, chunk // 8] |= np.uint32(code << (4 * (chunk % 8)))
            expanded[row, chunk * 4 + (code & 3)] = packed_a[row, chunk * 2]
            expanded[row, chunk * 4 + (code >> 2)] = packed_a[row, chunk * 2 + 1]
    if block_scale:
        assert packed_k == 32  # Wide block TMEM uses its own packed-FP6 fixture.
        _, scaled_args, _ = block_tmem_case(cta, af, bf, m=m)
        metadata = np.concatenate((scaled_args["metadata"], metadata), axis=2)
        expanded *= np.exp2((np.arange(m) % rows % 32 % 3 - 1 + np.arange(m) // rows)[:, None])
        b *= np.exp2((np.arange(n) % 2)[:, None])
    args["metadata"] = metadata
    seed = (np.arange(m * n).reshape(m, n) % 7 / 16).astype(np.float32)
    expected = expanded @ b.T + seed
    args["seed"] = (
        (seed.astype(np.float16).view(np.uint16).astype(np.uint32) | np.uint32(0xBEEF0000)).view(
            np.int32
        )
        if half
        else seed.view(np.int32)
    )
    expected = (
        expected.astype(np.float16).view(np.uint16).astype(np.int32)
        if half
        else expected.view(np.int32)
    )
    if cta == 2 and not block_scale:
        expected[rows + 1] = args["seed"][rows + 1]
    kernel = ti16_kernel(
        True,
        tmem,
        cta,
        m,
        kind="mxf8f6f4" if block_scale else "f8f6f4",
        sparse=True,
        a_format=af,
        b_format=bf,
        half_accumulator=half,
        mma_k=packed_k,
        sparsity_selector=0,
        implicit_scale=block_scale,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        collectors=".collector::a::discard.collector::b::discard",
        arch="sm_107a",
    )
    return kernel, args, expected


@pytest.mark.parametrize("case", SPARSE_NARROW_CASES)
def test_sparse_narrow_layouts_and_codecs(case, tmp_path):
    kernel, args, expected = sparse_narrow_case(*case)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], expected)


# Cover both sparse M shapes and existing shared/TMEM narrow codecs.
SPARSE_NARROW_K128_CASES = [
    (1, 128, False, 0, 1, False),
    (1, 128, False, 1, 0, False, True, True),
    (2, 256, True, 0, 1, False, False, True),
    (1, 64, False, 3, 4, False),
    (1, 128, True, 5, 3, False),
    (2, 128, True, 4, 5, False),
    (2, 256, False, 5, 1, False),
]


@pytest.mark.parametrize("case", SPARSE_NARROW_K128_CASES)
def test_sparse_narrow_k128(case, tmp_path):
    kernel, args, expected = sparse_narrow_case(*case, packed_k=64)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], expected)
