"""I8 shares integer accumulation and packed layouts, not floating-point rounding."""

import numpy as np
from tests.numsim.runtime.test_tcgen05_ti16 import ti16_kernel


# Sign combinations, saturation, both source spaces, paired layouts and WS banks.
def i8_case(cta_group, m, tmem_a, ws, a_format, b_format, saturate, **options):
    n = 64 if ws else 16 * cta_group
    rows = m // cta_group
    rng = np.random.default_rng(104)
    a_bits = rng.integers(0, 256, (4, m, 32), np.uint8)
    b_bits = rng.integers(0, 256, (n + 32 if ws else n, 32), np.uint8)
    a = a_bits.view(np.int8 if a_format else np.uint8).astype(np.int64)
    b = b_bits.view(np.int8 if b_format else np.uint8).astype(np.int64)
    seed = (
        np.broadcast_to(np.where(np.arange(n) % 2, -(2**31) + 64, 2**31 - 65), (m, n))
        .astype(np.int32)
        .copy()
    )
    banks = (128 // rows) if tmem_a and (ws or cta_group == 2) else 1
    expected = np.concatenate(
        [a[bank] @ b[bank * (n // banks) : (bank + 1) * (n // banks)].T for bank in range(banks)],
        axis=1,
    )
    expected += seed.astype(np.int64)
    if saturate:
        expected = expected.clip(-(2**31), 2**31 - 1)
    if cta_group == 2:
        kept = n // (128 // rows)
        expected[rows + 1, :kept] = seed[rows + 1, :kept]
    args = {
        "a": a_bits.view(np.uint16),
        "b": b_bits.view(np.uint16),
        "zero_mask": np.zeros(1, np.uint64),
        "metadata": np.zeros((2, 128, 2), np.uint32),
        "seed": seed,
        "out": np.zeros_like(seed),
    }
    kernel = ti16_kernel(
        True,
        tmem_a,
        cta_group,
        m,
        ws=ws,
        kind="i8",
        a_format=a_format,
        b_format=b_format,
        saturate=saturate,
        **options,
    )
    return kernel, args, expected.astype(np.int32)
