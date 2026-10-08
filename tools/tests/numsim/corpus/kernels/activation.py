"""NumSim-owned numerical corpus for canonical activation kernels."""

from __future__ import annotations

import numpy as np

from tests.numsim.corpus.kernels.gemm import (
    _e4m3fn_bits_to_float32,
    _float32_to_e2m1_bits,
    _float32_to_e4m3fn_bits,
    _pack_e2m1,
)
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase


def prepare_act_and_mul_case() -> NumSimCase:
    """Exercise one full 16-byte SiLU vector against its scalar definition."""

    module = load_tirx_kernel("act_and_mul")
    x = np.tile(
        np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float16),
        32,
    )
    y = np.tile(
        np.array([0.5, -1.0, 1.5, -2.0, 3.0, -0.5, 2.0, 1.0], dtype=np.float16),
        32,
    )
    input_data = np.concatenate((x, y)).reshape(1, -1).copy()
    output = np.full((1, 256), np.float16(np.nan), dtype=np.float16)

    x_f32 = x.astype(np.float32)
    y_f32 = y.astype(np.float32)
    denominator = np.float32(1.0) + np.exp2(
        x_f32 * np.float32(-1.4426950408889634), dtype=np.float32
    )
    expected = ((x_f32 / denominator) * y_f32).astype(np.float16)[None, :]

    return NumSimCase(
        kernel=module.get_kernel(act="silu", dtype="float16", num_tokens=1, d=256),
        args={
            "input_global": input_data,
            "out_global": output,
        },
        outputs=("out_global",),
        reference=lambda: {"out_global": expected.copy()},
        comparisons={"out_global": ComparisonSpec(rtol=1e-3, atol=1e-3)},
    )


def prepare_silu_nvfp4_experts_case() -> NumSimCase:
    """Exercise the fused SiLU, E4M3-scale, and packed-E2M1 data path."""

    module = load_tirx_kernel("silu_and_mul_nvfp4_experts_quantize")
    previous_sm_count = module._SM_COUNT_CACHE
    module._SM_COUNT_CACHE = 148
    try:
        kernel = module.get_kernel(dtype="float16", n_experts=1, m=1, k=256, mask_mode="full")
    finally:
        module._SM_COUNT_CACHE = previous_sm_count

    x = np.array(
        [-3, -2, -1, -0.5, 0, 0.5, 1, 2, -3, -2, -1, -0.5, 0, 0.5, 1, 2],
        dtype=np.float16,
    )
    y = np.array(
        [0.5, -1, 1.5, -2, 3, -0.5, 2, 1, 1, 0.5, -1, 2, -0.5, 3, 0.5, -2],
        dtype=np.float16,
    )
    x = np.tile(x, 16)
    y = np.tile(y, 16)
    input_data = np.concatenate((x, y))
    global_scale = np.array([1.0], dtype=np.float32)
    mask = np.array([1], dtype=np.int32)
    output = np.full((16,), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    scales = np.full((2048,), np.uint8(0xFF), dtype=np.uint8)

    x_f32 = x.astype(np.float32)
    fused = (
        (
            (
                x_f32
                / (
                    np.float32(1.0)
                    + np.exp2(x_f32 * np.float32(-1.4426950408889634), dtype=np.float32)
                )
            )
            * y.astype(np.float32)
        )
        .astype(np.float16)
        .astype(np.float32)
    )
    fused_blocks = fused.reshape(-1, 16)
    vec_max = np.max(np.abs(fused_blocks), axis=1)
    sf_bits = _float32_to_e4m3fn_bits(vec_max / np.float32(6.0))
    sf_values = _e4m3fn_bits_to_float32(sf_bits)
    quantized = np.divide(
        fused_blocks,
        sf_values[:, None],
        out=np.zeros_like(fused_blocks),
        where=vec_max[:, None] != 0,
    )
    expected_output = _pack_e2m1(_float32_to_e2m1_bits(quantized)).view(np.uint64).reshape(-1)
    expected_scales = scales.copy()
    scale_offsets = (np.arange(16) // 4) * 512 + np.arange(16) % 4
    expected_scales[scale_offsets] = sf_bits

    return NumSimCase(
        kernel=kernel,
        args={
            "input_global": input_data,
            "sf_scale": global_scale,
            "out_global": output,
            "sf_out": scales,
            "mask": mask,
            "num_rows": np.int32(1),
            "num_cols": np.int32(256),
            "num_experts": np.int32(1),
            "use_silu_and_mul": np.int32(1),
        },
        outputs=("out_global", "sf_out"),
        reference=lambda: {
            "out_global": expected_output.copy(),
            "sf_out": expected_scales.copy(),
        },
        comparisons={
            "out_global": ComparisonSpec(rtol=0, atol=0),
            "sf_out": ComparisonSpec(rtol=0, atol=0),
        },
    )


__all__ = ["prepare_act_and_mul_case", "prepare_silu_nvfp4_experts_case"]
