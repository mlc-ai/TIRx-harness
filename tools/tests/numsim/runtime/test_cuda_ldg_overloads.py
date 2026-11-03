from __future__ import annotations

import numpy as np

_FLOAT8_BASE_DTYPES = frozenset(
    {
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    }
)
_EXPECTED_CUDA_LDG_DTYPES = frozenset(
    {
        "bfloat16",
        "bfloat16x2",
        "bfloat16x8",
        "float16",
        "float16x2",
        "float16x8",
        "float32",
        "float32x2",
        "float32x4",
        "float64",
        "float64x2",
        "int8",
        "int8x2",
        "int8x4",
        "int8x8",
        "int8x16",
        "int16",
        "int16x2",
        "int16x4",
        "int16x8",
        "int32",
        "int32x2",
        "int32x4",
        "int64",
        "int64x2",
        "uint8",
        "uint8x2",
        "uint8x4",
        "uint8x8",
        "uint8x16",
        "uint16",
        "uint16x2",
        "uint16x4",
        "uint16x8",
        "uint32",
        "uint32x2",
        "uint32x4",
        "uint64",
        "uint64x2",
    }
) | frozenset(f"{dtype}x{lanes}" for dtype in _FLOAT8_BASE_DTYPES for lanes in (8, 16))
_EXPECTED_REJECTED_CUDA_LDG_DTYPES = frozenset(
    {
        "bool",
        "boolx2",
        "boolx4",
        "bfloat16x4",
        "float16x4",
    }
) | frozenset(
    dtype if lanes == 1 else f"{dtype}x{lanes}"
    for dtype in _FLOAT8_BASE_DTYPES
    for lanes in (1, 2, 4)
)
_VALID_UNMODELED_CUDA_LDG_DTYPES = frozenset({"boolx2", "boolx4"})


def _packed_array(itemsize: int) -> np.ndarray:
    raw = (np.arange(32 * itemsize, dtype=np.uint16) * np.uint16(37) + np.uint16(11)).astype(
        np.uint8
    )
    if itemsize == 2:
        return raw.view(np.uint16)
    if itemsize == 4:
        return raw.view(np.uint32)
    if itemsize == 8:
        return raw.view(np.uint64)
    return raw.view(np.dtype("V16"))
