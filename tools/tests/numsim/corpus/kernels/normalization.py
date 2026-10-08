"""Small numerical cases for canonical normalization kernels."""

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


def _specialize(kernel, values: dict[str, int | float]):
    return kernel.specialize(
        {param: values[param.name] for param in kernel.params if param.name in values}
    )


def _rms_reference(values: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    values_f32 = values.astype(np.float32)
    return (
        values_f32
        / np.sqrt(np.mean(values_f32 * values_f32, axis=-1, keepdims=True) + np.float32(eps))
        * weight.astype(np.float32)
    )


def prepare_rmsnorm_case(*, hidden_size: int = 128, batch_size: int = 1) -> NumSimCase:
    """Compare RMSNorm with a direct float32 definition and float16 rounding."""

    module = load_tirx_kernel("rmsnorm")
    values = np.linspace(-1.5, 1.5, batch_size * hidden_size, dtype=np.float32).reshape(
        batch_size, hidden_size
    )
    weights = np.linspace(0.5, 1.5, hidden_size, dtype=np.float32)
    input_data = values.astype(np.float16).reshape(-1)
    weight_data = weights.astype(np.float16)
    output = np.full(input_data.shape, np.float16(np.nan), dtype=np.float16)
    input_f32 = input_data.reshape(batch_size, hidden_size).astype(np.float32)
    expected = (
        (
            input_f32
            * np.float32(1.0)
            / np.sqrt(
                np.mean(input_f32 * input_f32, axis=-1, keepdims=True) + np.float32(module.eps)
            )
            * weight_data.astype(np.float32)
        )
        .astype(np.float16)
        .reshape(-1)
    )
    return NumSimCase(
        kernel=module.get_kernel(hidden_size=hidden_size),
        args={
            "inp": input_data,
            "wgt": weight_data,
            "out": output,
            "batch_size": np.int32(batch_size),
        },
        outputs={"out_global": "out"},
        reference=lambda: {"out_global": expected.copy()},
        comparisons={"out_global": ComparisonSpec(rtol=0, atol=0)},
    )


def prepare_flashinfer_rmsnorm_fp4quant_case() -> NumSimCase:
    """Exercise the pinned FP16 RMSNorm plus packed FP4 output path."""

    module = load_tirx_kernel("flashinfer_rmsnorm_fp4quant")
    eps = 1e-5
    hidden = 64
    values = np.linspace(-1.0, 1.0, hidden, dtype=np.float16)
    weight = np.linspace(0.5, 1.5, hidden, dtype=np.float16)
    packed_output = np.full((hidden // 2,), 0xFF, dtype=np.uint8)
    scales = np.full((hidden // 16,), 0xFF, dtype=np.uint8)
    global_scale = np.array([1.0], dtype=np.float32)
    kernel = _specialize(
        module.get_kernel(
            input_dtype="float16",
            M=1,
            H=hidden,
            block_size=16,
            scale_format="e4m3",
            swizzled=False,
            enable_pdl=False,
        ),
        {"runtime_M": 1, "runtime_eps": eps},
    )

    values_f32 = values.astype(np.float32)
    rstd = np.float32(1.0) / np.sqrt(
        np.mean(values_f32 * values_f32, dtype=np.float32) + np.float32(eps)
    )
    products = (values * weight).astype(np.float16).astype(np.float32)
    normalized = products * rstd
    blocks = normalized.reshape(-1, 16)
    scale_bits = _float32_to_e4m3fn_bits(
        np.max(np.abs(products).reshape(-1, 16), axis=1) * rstd / np.float32(6.0)
    )
    scale_values = _e4m3fn_bits_to_float32(scale_bits)
    expected = {
        "y": _pack_e2m1(_float32_to_e2m1_bits((blocks / scale_values[:, None]).reshape(-1))),
        "scales": scale_bits,
    }
    return NumSimCase(
        kernel=kernel,
        args={
            "x": values,
            "weight": weight,
            "y": packed_output,
            "scales": scales,
            "global_scale": global_scale,
        },
        outputs=("y", "scales"),
        reference=lambda: {name: value.copy() for name, value in expected.items()},
        comparisons={
            "y": ComparisonSpec(rtol=0, atol=0),
            "scales": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_flashinfer_add_rmsnorm_fp4quant_case() -> NumSimCase:
    """Exercise fused residual add plus the packed FP4 quantization path."""

    module = load_tirx_kernel("flashinfer_add_rmsnorm_fp4quant")
    hidden = 64
    epsilon = 1e-5
    config = {
        "case": "nv2d",
        "input_dtype": "float16",
        "input_ndim": 2,
        "M": 1,
        "B": None,
        "S": None,
        "H": hidden,
        "block_size": 16,
        "scale_format": "e4m3",
        "swizzled": False,
        "output_both_sf_layouts": False,
        "output_norm": False,
        "enable_pdl": False,
        "eps": epsilon,
        "global_scale_mode": "none",
        "allocation": "preallocated",
        "data_mode": "random",
    }
    kernel = _specialize(module.get_kernel(**config), {"runtime_M": 1, "runtime_eps": epsilon})

    values = np.linspace(-0.75, 0.75, hidden, dtype=np.float16)
    residual = np.linspace(0.25, 1.25, hidden, dtype=np.float16)
    weight = np.linspace(0.5, 1.5, hidden, dtype=np.float16)
    summed_f32 = values.astype(np.float32) + residual.astype(np.float32)
    summed = summed_f32.astype(np.float16)
    rstd = np.float32(1.0) / np.sqrt(
        np.mean(summed_f32 * summed_f32, dtype=np.float32) + np.float32(epsilon)
    )
    normalized = summed.astype(np.float32) * weight.astype(np.float32) * rstd
    blocks = normalized.reshape(-1, 16)
    scale_bits = _float32_to_e4m3fn_bits(np.max(np.abs(blocks), axis=1) / np.float32(6.0))
    scale_values = _e4m3fn_bits_to_float32(scale_bits)
    expected = {
        "y": _pack_e2m1(_float32_to_e2m1_bits((blocks / scale_values[:, None]).reshape(-1))),
        "scales": scale_bits,
        "residual": summed.copy(),
    }
    return NumSimCase(
        kernel=kernel,
        args={
            "x": values,
            "residual": residual,
            "weight": weight,
            "y": np.full(hidden // 2, 0xFF, dtype=np.uint8),
            "scales": np.full(hidden // 16, 0xFF, dtype=np.uint8),
            "scales_unswizzled": np.full(hidden // 16, 0xA5, dtype=np.uint8),
            "y_norm": np.ones(hidden, dtype=np.float16),
            "global_scale": np.array([1.0], dtype=np.float32),
        },
        outputs=("y", "scales", "residual"),
        reference=lambda: {name: value.copy() for name, value in expected.items()},
        comparisons={
            "y": ComparisonSpec(rtol=0, atol=0),
            "scales": ComparisonSpec(rtol=0, atol=0),
            "residual": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_flashinfer_rmsnorm_case() -> NumSimCase:
    """Bind the pinned FlashInfer RMSNorm ABI, including its visible output."""

    module = load_tirx_kernel("flashinfer_rmsnorm")
    eps = 1e-5
    values = np.linspace(-1.0, 1.0, 128, dtype=np.float16)
    weight = np.linspace(0.5, 1.5, 128, dtype=np.float16)
    output = np.full(values.shape, np.nan, dtype=np.float16)
    kernel = _specialize(
        module.get_kernel(
            variant="rmsnorm",
            dtype="float16",
            M=1,
            H=128,
            input_layout="compact",
            output_layout="compact",
            enable_pdl=False,
            eps=eps,
        ),
        {"runtime_M": 1, "runtime_eps": eps},
    )
    expected = _rms_reference(values, weight, eps).astype(np.float16)
    return NumSimCase(
        kernel=kernel,
        args={
            "x": values,
            "weight": weight,
            "y": output,
        },
        outputs=("y",),
        reference=lambda: {"y": expected.copy()},
        comparisons={"y": ComparisonSpec(rtol=0, atol=0)},
    )


def prepare_flashinfer_layernorm_case() -> NumSimCase:
    module = load_tirx_kernel("flashinfer_layernorm")
    eps = 1e-5
    values_f32 = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    value_bits = ((values_f32.view(np.uint32) + 0x7FFF) >> 16).astype(np.uint16)
    values = (value_bits.astype(np.uint32) << 16).view(np.float32)
    gamma = np.linspace(0.5, 1.5, 128, dtype=np.float32)
    beta = np.linspace(-0.2, 0.2, 128, dtype=np.float32)
    output = np.full(value_bits.shape, 0x7FC0, dtype=np.uint16)
    kernel = _specialize(
        module.get_kernel(
            M=1,
            H=128,
            input_layout="compact",
            output_layout="compact",
            enable_pdl=False,
            eps=eps,
        ),
        {"runtime_M": 1, "runtime_eps": eps, "y_row_stride": 128, "x_row_stride": 128},
    )
    normalized = (values - values.mean()) / np.sqrt(np.mean((values - values.mean()) ** 2) + eps)
    expected_bits = ((np.multiply(normalized, gamma) + beta).view(np.uint32) >> 16).astype(
        np.uint16
    )
    expected = (expected_bits.astype(np.uint32) << 16).view(np.float32)
    return NumSimCase(
        kernel=kernel,
        args={
            "out": output,
            "x": value_bits,
            "gamma": gamma,
            "beta": beta,
        },
        outputs=("out",),
        reference=lambda: {"out": expected.copy()},
        comparisons={"out": ComparisonSpec(rtol=0.02, atol=0.02, actual_encoding="bfloat16")},
    )


def prepare_flashinfer_fused_dit_layernorm_case() -> NumSimCase:
    """Exercise the BF16 residual/scale/shift DIT LayerNorm path."""

    module = load_tirx_kernel("flashinfer_fused_dit_layernorm")
    hidden = 3072
    epsilon = 1e-6
    config = {
        "mode": "rss",
        "batch_size": 1,
        "num_rows": 1,
        "output_format": "bf16",
        "use_input_sf_scale": False,
        "input_sf_scale": 1.0,
        "has_residual": True,
        "destination_passing": False,
        "auxiliary_ndim": 3,
        "bias_ndim": 2,
        "epsilon": epsilon,
    }
    kernel = module.get_kernel(**config)
    runtime_values = {
        "runtime_batch_size": 1,
        "runtime_num_rows": 1,
        "runtime_epsilon": epsilon,
        "runtime_has_residual": 1,
    }
    kernel = kernel.specialize(
        {
            param: runtime_values[param.name]
            for param in kernel.params
            if param.name in runtime_values
        }
    )

    def to_bfloat16(values: np.ndarray) -> np.ndarray:
        return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

    def from_bfloat16(values: np.ndarray) -> np.ndarray:
        return (values.astype(np.uint32) << 16).view(np.float32)

    input_data = to_bfloat16(np.linspace(-1.0, 1.0, hidden, dtype=np.float32))
    residual_data = to_bfloat16(np.linspace(0.25, 1.25, hidden, dtype=np.float32))
    scale_data = to_bfloat16(np.linspace(-0.2, 0.2, hidden, dtype=np.float32))
    shift_data = to_bfloat16(np.linspace(0.1, 0.3, hidden, dtype=np.float32))
    input_f32 = from_bfloat16(input_data)
    residual_f32 = from_bfloat16(residual_data)
    scale_f32 = from_bfloat16(scale_data)
    shift_f32 = from_bfloat16(shift_data)
    residual_sum = input_f32 + residual_f32
    centered = residual_sum - residual_sum.mean()
    normalized = centered / np.sqrt(np.mean(centered * centered) + np.float32(epsilon))
    expected_residual = residual_sum.astype(np.float32)
    expected_norm = normalized * (scale_f32 + 1.0) + shift_f32
    sentinel_bf16 = np.full(hidden, 0x7FC0, dtype=np.uint16)

    return NumSimCase(
        kernel=kernel,
        args={
            "input_buffer": input_data,
            "residual_buffer": residual_data,
            "gate_buffer": np.zeros(hidden, dtype=np.uint16),
            "gate_bias_buffer": np.zeros(hidden, dtype=np.float32),
            "gamma_buffer": np.ones(hidden, dtype=np.float32),
            "beta_buffer": np.zeros(hidden, dtype=np.float32),
            "scale_buffer": scale_data,
            "scale_bias_buffer": np.zeros(hidden, dtype=np.float32),
            "shift_buffer": shift_data,
            "shift_bias_buffer": np.zeros(hidden, dtype=np.float32),
            "residual_output_buffer": sentinel_bf16.copy(),
            "norm_output_buffer": sentinel_bf16.copy(),
            "sf_output_buffer": np.full(512, 0xA5, dtype=np.uint8),
            "output_sf_scale_buffer": np.array([1.0], dtype=np.float32),
            "input_sf_scale_buffer": np.array([1.0], dtype=np.float32),
        },
        outputs=("residual_output_buffer", "norm_output_buffer"),
        reference=lambda: {
            "residual_output_buffer": from_bfloat16(to_bfloat16(expected_residual)),
            "norm_output_buffer": from_bfloat16(to_bfloat16(expected_norm)),
        },
        comparisons={
            "residual_output_buffer": ComparisonSpec(
                rtol=0.02, atol=0.02, actual_encoding="bfloat16"
            ),
            "norm_output_buffer": ComparisonSpec(rtol=0.02, atol=0.02, actual_encoding="bfloat16"),
        },
    )


def prepare_flashinfer_qk_rmsnorm_case() -> NumSimCase:
    module = load_tirx_kernel("flashinfer_qk_rmsnorm")
    eps = 1e-5
    values = np.array([-0.75, 0.5], dtype=np.float16)
    weight = np.array([1.25, 0.75], dtype=np.float16)
    output = np.full(values.shape, np.nan, dtype=np.float16)
    kernel = _specialize(
        module.get_kernel(
            variant="rmsnorm",
            dtype="float16",
            B=1,
            N=1,
            H=2,
            enable_pdl=False,
            eps=eps,
        ),
        {
            "runtime_B": 1,
            "runtime_N": 1,
            "runtime_eps": eps,
            "x_batch_stride": 2,
            "x_head_stride": 2,
            "y_batch_stride": 2,
            "y_head_stride": 2,
        },
    )
    expected = _rms_reference(values, weight, eps).astype(np.float16)
    return NumSimCase(
        kernel=kernel,
        args={
            "x": values,
            "weight": weight,
            "y": output,
        },
        outputs=("y",),
        reference=lambda: {"y": expected.copy()},
        comparisons={"y": ComparisonSpec(rtol=0.02, atol=0.02)},
    )


def prepare_flashinfer_fused_add_rmsnorm_case() -> NumSimCase:
    module = load_tirx_kernel("flashinfer_fused_add_rmsnorm")
    eps = 1e-5
    values = np.linspace(-1.0, 1.0, 128, dtype=np.float16)
    residual = np.linspace(0.25, 1.25, 128, dtype=np.float16)
    weight = np.linspace(0.5, 1.5, 128, dtype=np.float16)
    summed = values.astype(np.float32) + residual.astype(np.float32)
    output = np.full(values.shape, np.nan, dtype=np.float16)
    kernel = _specialize(
        module.get_kernel(
            variant="fused_add_rmsnorm",
            dtype="float16",
            M=1,
            H=128,
            input_layout="compact",
            residual_layout="compact",
            enable_pdl=False,
            eps=eps,
        ),
        {"runtime_M": 1, "runtime_eps": eps},
    )
    expected = {
        "input_buffer": (_rms_reference(summed.astype(np.float16), weight, eps)).astype(np.float16),
        "residual": summed.astype(np.float16),
    }
    return NumSimCase(
        kernel=kernel,
        args={
            "input_buffer": values,
            "residual": residual,
            "weight": weight,
        },
        outputs=("input_buffer", "residual"),
        reference=lambda: {key: value.copy() for key, value in expected.items()},
        comparisons={key: ComparisonSpec(rtol=2e-3, atol=2e-3) for key in expected},
    )


def prepare_flashinfer_rmsnorm_quant_case() -> NumSimCase:
    module = load_tirx_kernel("flashinfer_rmsnorm_quant")
    eps = 1e-5
    hidden = 65
    values = np.linspace(-1.0, 1.0, hidden, dtype=np.float16)
    weight = np.linspace(0.5, 1.5, hidden, dtype=np.float16)
    output = np.full(values.shape, 0xFF, dtype=np.uint8)
    scale = np.array([1.0], dtype=np.float32)
    kernel = _specialize(
        module.get_kernel(
            input_dtype="float16",
            output_dtype="float8_e4m3fn",
            M=1,
            H=hidden,
            input_layout="compact",
            output_layout="compact",
            enable_pdl=False,
            scale=1.0,
            eps=eps,
        ),
        {"runtime_M": 1, "runtime_eps": eps},
    )
    expected_f32 = _rms_reference(values, weight, eps)
    expected = _float32_to_e4m3fn_bits(expected_f32)
    return NumSimCase(
        kernel=kernel,
        args={
            "x": values,
            "weight": weight,
            "out": output,
            "scale_buffer": scale,
        },
        outputs=("out",),
        reference=lambda: {"out": expected.copy()},
        comparisons={"out": ComparisonSpec(rtol=0, atol=0)},
    )


def prepare_flashinfer_fused_add_rmsnorm_quant_case() -> NumSimCase:
    module = load_tirx_kernel("flashinfer_fused_add_rmsnorm_quant")
    eps = 1e-5
    hidden = 128
    values = np.linspace(-1.0, 1.0, hidden, dtype=np.float16)
    residual = np.linspace(0.25, 1.25, hidden, dtype=np.float16)
    weight = np.linspace(0.5, 1.5, hidden, dtype=np.float16)
    output = np.full(values.shape, 0xFF, dtype=np.uint8)
    scale = np.array([1.0], dtype=np.float32)
    summed = values.astype(np.float32) + residual.astype(np.float32)
    kernel = _specialize(
        module.get_kernel(
            input_dtype="float16",
            output_dtype="float8_e4m3fn",
            M=1,
            H=hidden,
            input_layout="compact",
            residual_layout="compact",
            output_layout="compact",
            enable_pdl=False,
            scale=1.0,
            eps=eps,
        ),
        {"runtime_M": 1, "runtime_eps": eps},
    )
    normalized = _rms_reference(summed.astype(np.float16), weight, eps)
    expected = {
        "output": _float32_to_e4m3fn_bits(normalized),
        "residual": summed.astype(np.float16),
    }
    return NumSimCase(
        kernel=kernel,
        args={
            "output": output,
            "input_buffer": values,
            "residual": residual,
            "weight": weight,
            "scale_buffer": scale,
        },
        outputs=("output", "residual"),
        reference=lambda: {key: value.copy() for key, value in expected.items()},
        comparisons={key: ComparisonSpec(rtol=0, atol=0) for key in expected},
    )


__all__ = [
    "prepare_flashinfer_fused_add_rmsnorm_case",
    "prepare_flashinfer_fused_add_rmsnorm_quant_case",
    "prepare_flashinfer_add_rmsnorm_fp4quant_case",
    "prepare_flashinfer_fused_dit_layernorm_case",
    "prepare_flashinfer_layernorm_case",
    "prepare_flashinfer_qk_rmsnorm_case",
    "prepare_flashinfer_rmsnorm_case",
    "prepare_flashinfer_rmsnorm_quant_case",
    "prepare_flashinfer_rmsnorm_fp4quant_case",
    "prepare_rmsnorm_case",
]
