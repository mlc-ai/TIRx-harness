from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tests.numsim.integration.test_stmatrix_artifact import (
    raw_stmatrix_layout,
    raw_stmatrix_x2_forms,
)
from tests.numsim.runtime.test_matrix_memory_domain_oracle import (
    legacy_ldmatrix_x1_domain,
    ptx_ldmatrix_x1_x2_domain,
    stmatrix_b16_x1_domain,
    stmatrix_b8_x4_domain,
)
from tests.numsim.support.kernels import raw_ldmatrix_x4_b16_fragments


@dataclass(frozen=True)
class MatrixMemoryCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...] = ("output",)


MATRIX_MEMORY_CASES = (
    MatrixMemoryCase(
        "ptx_ldmatrix_b16_x1_trans_and_x2",
        ptx_ldmatrix_x1_x2_domain,
        lambda: {"output": np.zeros(96, dtype=np.uint32)},
    ),
    MatrixMemoryCase(
        "legacy_ldmatrix_b16_x1",
        legacy_ldmatrix_x1_domain,
        lambda: {"output": np.zeros(64, dtype=np.uint16)},
    ),
    MatrixMemoryCase(
        "ptx_ldmatrix_b16_x4",
        raw_ldmatrix_x4_b16_fragments,
        lambda: {"output": np.zeros(128, dtype=np.uint32)},
    ),
    MatrixMemoryCase(
        "stmatrix_b16_x1",
        stmatrix_b16_x1_domain,
        lambda: {"output": np.zeros(64, dtype=np.uint16)},
    ),
    MatrixMemoryCase(
        "stmatrix_b8_x4_trans",
        stmatrix_b8_x4_domain,
        lambda: {"output": np.zeros(512, dtype=np.uint8)},
    ),
    MatrixMemoryCase(
        "stmatrix_b8_x1_and_b16_x4_trans",
        raw_stmatrix_layout,
        lambda: {
            "output_b8": np.zeros(128, dtype=np.uint8),
            "output_b16": np.zeros(256, dtype=np.uint16),
        },
        ("output_b8", "output_b16"),
    ),
    MatrixMemoryCase(
        "stmatrix_b8_and_b16_x2",
        raw_stmatrix_x2_forms,
        lambda: {
            "output_b8": np.zeros(256, dtype=np.uint8),
            "output_b16": np.zeros(128, dtype=np.uint16),
        },
        ("output_b8", "output_b16"),
    ),
)


__all__ = ["MATRIX_MEMORY_CASES", "MatrixMemoryCase"]
