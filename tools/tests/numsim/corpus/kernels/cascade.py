"""Numerical cases for merging partial attention states."""

from __future__ import annotations

import numpy as np

from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase


def prepare_merge_state_case() -> NumSimCase:
    module = load_tirx_kernel("merge_state")
    dtype = np.dtype("float16")
    # Three heads exercise the partial-warp guard for head_dim=64.
    shape = (2, 3, 64)
    rng = np.random.default_rng(20260910)
    a = rng.uniform(-1, 1, shape).astype(dtype)
    b = rng.uniform(-1, 1, shape).astype(dtype)
    sa = np.array([[-3, 0, 8], [2, -5, 1]], dtype=np.float32)
    sb = np.array([[1, 0, -2], [-4, 3, 1]], dtype=np.float32)
    # Independent log-space definition, evaluated in float64.
    logsum = np.logaddexp2(sa.astype(np.float64), sb.astype(np.float64))
    merged = (
        a.astype(np.float64) * np.exp2(sa - logsum)[..., None]
        + b.astype(np.float64) * np.exp2(sb - logsum)[..., None]
    ).astype(dtype)
    return NumSimCase(
        kernel=module.get_kernel(
            dtype=dtype.name, seq_len=shape[0], num_heads=shape[1], head_dim=shape[2]
        ),
        args={
            "v_a": a.reshape(-1),
            "s_a": sa.reshape(-1),
            "v_b": b.reshape(-1),
            "s_b": sb.reshape(-1),
            "v_merged": np.full(a.size, np.nan, dtype=dtype),
            "s_merged": np.full(sa.size, np.nan, dtype=np.float32),
        },
        outputs=("v_merged", "s_merged"),
        reference=lambda: {
            "v_merged": merged.reshape(-1),
            "s_merged": logsum.astype(np.float32).reshape(-1),
        },
        comparisons={
            "v_merged": ComparisonSpec(rtol=0.001, atol=0.0005),
            "s_merged": ComparisonSpec(rtol=1e-5, atol=1e-5),
        },
    )
