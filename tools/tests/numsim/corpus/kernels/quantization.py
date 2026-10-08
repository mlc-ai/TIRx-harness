"""NumSim-owned bit-exact corpus for canonical low-precision quantizers."""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.numsim.corpus.kernels.gemm import (
    _float32_to_e2m1_bits,
    _float32_to_e4m3fn_bits,
    _pack_e2m1,
)
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase


_E2M1_EXACT = np.array(
    [
        -6.0,
        -4.0,
        -3.0,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
    ],
    dtype=np.float16,
)


def _kernel_with_fixed_sm_count(name: str, **config: Any):
    """Build host launch constants without querying a CUDA device."""

    module = load_tirx_kernel(name)
    previous = module._SM_COUNT_CACHE
    module._SM_COUNT_CACHE = 148
    try:
        return module.get_kernel(**config)
    finally:
        module._SM_COUNT_CACHE = previous


def _exact_fp4_input(k: int) -> np.ndarray:
    if k % len(_E2M1_EXACT):
        raise ValueError("exact FP4 test input length must be a multiple of 16")
    return np.tile(_E2M1_EXACT, k // len(_E2M1_EXACT)).astype(np.float16)


def prepare_mxfp4_quantize_case() -> NumSimCase:
    input_data = _exact_fp4_input(32)
    output = np.full((16,), np.uint8(0xFF), dtype=np.uint8)
    scales = np.full((1,), np.uint8(0xFF), dtype=np.uint8)
    expected = _pack_e2m1(_float32_to_e2m1_bits(input_data.astype(np.float32)))
    return NumSimCase(
        kernel=_kernel_with_fixed_sm_count(
            "mxfp4_quantize", dtype="float16", m=1, k=32, sf_layout="linear"
        ),
        args={
            "in_global": input_data,
            "out_global": output,
            "sf_out": scales,
        },
        outputs=("out_global", "sf_out"),
        reference=lambda: {
            "out_global": expected.copy(),
            "sf_out": np.array([127], dtype=np.uint8),
        },
        comparisons={
            "out_global": ComparisonSpec(rtol=0, atol=0),
            "sf_out": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_mxfp8_quantize_case() -> NumSimCase:
    block = np.array(
        [
            -448.0,
            -256.0,
            -128.0,
            -64.0,
            -32.0,
            -16.0,
            -8.0,
            -4.0,
            -2.0,
            -1.0,
            -0.5,
            -0.25,
            -0.125,
            -0.0625,
            -0.03125,
            0.0,
            0.03125,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
            32.0,
            64.0,
            128.0,
            256.0,
            448.0,
            0.0,
        ],
        dtype=np.float16,
    )
    values = np.tile(block, 128)
    output = np.full((4096,), np.uint8(0xFF), dtype=np.uint8)
    scales = np.full((128,), np.uint8(0xFF), dtype=np.uint8)
    expected = _float32_to_e4m3fn_bits(values.astype(np.float32))
    return NumSimCase(
        kernel=_kernel_with_fixed_sm_count(
            "mxfp8_quantize", dtype="float16", m=1, k=4096, sf_layout="linear"
        ),
        args={
            "in_global": values,
            "out_global": output,
            "sf_out": scales,
            "total_sf": np.int32(128),
        },
        outputs=("out_global", "sf_out"),
        reference=lambda: {
            "out_global": expected.copy(),
            "sf_out": np.full((128,), np.uint8(127), dtype=np.uint8),
        },
        comparisons={
            "out_global": ComparisonSpec(rtol=0, atol=0),
            "sf_out": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_nvfp4_quantize_case() -> NumSimCase:
    input_data = _exact_fp4_input(16)
    output = np.full((8,), np.uint8(0xFF), dtype=np.uint8)
    scales = np.full((1,), np.uint8(0xFF), dtype=np.uint8)
    global_scale = np.array([1.0], dtype=np.float32)
    expected = _pack_e2m1(_float32_to_e2m1_bits(input_data.astype(np.float32)))
    return NumSimCase(
        kernel=_kernel_with_fixed_sm_count(
            "nvfp4_quantize",
            dtype="float16",
            m=1,
            k=16,
            sf_layout="linear",
            fuse_silu=False,
        ),
        args={
            "in_global": input_data,
            "out_global": output,
            "sf_out": scales,
            "m_rows": np.int32(1),
            "total_sf": np.int32(1),
            "gs": global_scale,
        },
        outputs=("out_global", "sf_out"),
        reference=lambda: {
            "out_global": expected.copy(),
            "sf_out": np.array([0x38], dtype=np.uint8),
        },
        comparisons={
            "out_global": ComparisonSpec(rtol=0, atol=0),
            "sf_out": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_nvfp4_quantize_per_token_case() -> NumSimCase:
    input_data = _exact_fp4_input(16)
    output = np.full((8,), np.uint8(0xFF), dtype=np.uint8)
    scales = np.full((1,), np.uint8(0xFF), dtype=np.uint8)
    token_scales = np.full((1,), np.float32(np.nan), dtype=np.float32)
    global_scale_inverse = np.array([1.0 / 6.0], dtype=np.float32)
    expected = _pack_e2m1(_float32_to_e2m1_bits(input_data.astype(np.float32)))
    return NumSimCase(
        kernel=load_tirx_kernel("nvfp4_quantize_per_token").get_kernel(
            dtype="float16", m=1, k=16, sf_layout="linear"
        ),
        args={
            "in_global": input_data,
            "out_global": output,
            "sf_out": scales,
            "pts_out": token_scales,
            "m_rows": np.int32(1),
            "gsi": global_scale_inverse,
        },
        outputs=("out_global", "sf_out", "pts_out"),
        reference=lambda: {
            "out_global": expected.copy(),
            "sf_out": np.array([0x38], dtype=np.uint8),
            "pts_out": np.array([1.0], dtype=np.float32),
        },
        comparisons={
            "out_global": ComparisonSpec(rtol=0, atol=0),
            "sf_out": ComparisonSpec(rtol=0, atol=0),
            "pts_out": ComparisonSpec(rtol=0, atol=0),
        },
    )


__all__ = [
    "prepare_mxfp4_quantize_case",
    "prepare_mxfp8_quantize_case",
    "prepare_nvfp4_quantize_case",
    "prepare_nvfp4_quantize_per_token_case",
]
