"""Small CPU-owned cases for the canonical cuDNN Frontend ports."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tests.numsim.corpus.kernels.gemm import (
    _e2m1_bits_to_float32,
    _float32_to_e2m1_bits,
    _float32_to_e4m3fn_bits,
    _pack_e2m1,
)
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import (
    ComparisonSpec,
    NumSimCase,
    TensorMap,
    _descriptor_storage,
)


dense_gemm_persistent_swiglu = load_tirx_kernel("cudnn_sm100_dense_gemm_persistent_swiglu")
dense_blockscaled_gemm_persistent_swiglu_interleaved_quant = load_tirx_kernel(
    "cudnn_sm100_dense_blockscaled_gemm_persistent_swiglu_interleaved_quant"
)
dense_blockscaled_gemm_persistent_srelu_quant = load_tirx_kernel(
    "cudnn_sm100_dense_blockscaled_gemm_persistent_srelu_quant"
)
dense_blockscaled_gemm_persistent_dsrelu_quant = load_tirx_kernel(
    "cudnn_sm100_dense_blockscaled_gemm_persistent_dsrelu_quant"
)
gemm_proj_rope_mxfp8_bf16in = load_tirx_kernel("cudnn_sm100_gemm_proj_rope_mxfp8_bf16in")
gemm_proj_rope_mxfp8_mxfp8in = load_tirx_kernel("cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in")
moe_grouped_gemm_dglu_dbias = load_tirx_kernel("cudnn_sm100_moe_grouped_gemm_dglu_dbias")
moe_blockscaled_grouped_gemm_dglu_dbias = load_tirx_kernel(
    "cudnn_sm100_moe_blockscaled_grouped_gemm_dglu_dbias"
)
kda_bprop_f16 = load_tirx_kernel("cudnn_sm100_kda_bprop_f16")
gdn_bprop_f16 = load_tirx_kernel("cudnn_sm100_gdn_bprop_f16")
gdn2_prefill_f16 = load_tirx_kernel("cudnn_sm100_gdn2_prefill_f16")
gdn2_bprop_f16 = load_tirx_kernel("cudnn_sm100_gdn2_bprop_f16")
gdn_prefill_f16 = load_tirx_kernel("cudnn_sm100_gdn_prefill_f16")
gdn_recompute_f16 = load_tirx_kernel("cudnn_sm100_gdn_recompute_f16")
dsa_sparse_attention_backward = load_tirx_kernel("cudnn_sm100_dsa_sparse_attention_backward")
bsa_forward_blk64 = load_tirx_kernel("cudnn_sm100_bsa_forward_blk64")
bsa_forward_combine_blk64 = load_tirx_kernel("cudnn_sm100_bsa_forward_combine_blk64")
bsa_forward_blk128 = load_tirx_kernel("cudnn_sm100_bsa_forward_blk128")
bsa_backward_blk64 = load_tirx_kernel("cudnn_sm100_bsa_backward_blk64")
bsa_backward_blk128 = load_tirx_kernel("cudnn_sm100_bsa_backward_blk128")
csa_compressor_fwd = load_tirx_kernel("cudnn_sm100_csa_compressor_fwd")


def _float32_to_bfloat16_bits(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _bfloat16_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _tensor_map(
    base: np.ndarray,
    *,
    dtype: str,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
    swizzle: str | None = "128B",
) -> np.ndarray:
    return TensorMap(
        base=base,
        dtype=dtype,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
        swizzle=swizzle,
        interleave=None,
        fill_mode="none",
    ).numpy()


def prepare_cudnn_dense_gemm_persistent_swiglu_case() -> NumSimCase:
    """Exercise BF16 GEMM plus the interleaved SwiGLU epilogue."""

    m = n = k_dim = 256
    alpha = np.float32(0.75)
    config = {
        "M": m,
        "N": n,
        "K": k_dim,
        "L": 1,
        "ab_dtype": "bfloat16",
        "acc_dtype": "float32",
        "ab12_dtype": "bfloat16",
        "c_dtype": "bfloat16",
        "a_major": "k",
        "b_major": "k",
        "c_major": "m",
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
    }
    rows = np.arange(m, dtype=np.int64)
    a_k = (rows * 17) % k_dim
    b_k = (rows * 23 + 6) % k_dim
    a_values = (1 + rows % 2).astype(np.float32)
    b_values = (1 + (rows + 1) % 2).astype(np.float32)
    a = np.zeros((m, k_dim), dtype=np.float32)
    b = np.zeros((n, k_dim), dtype=np.float32)
    a[rows, a_k] = a_values
    b[rows, b_k] = b_values

    ab12 = (
        (a_k[:, None] == b_k[None, :]).astype(np.float32)
        * a_values[:, None]
        * b_values[None, :]
        * alpha
    )
    blocks = ab12.reshape(m, n // 32, 32)
    x = blocks[:, 0::2].reshape(m, n // 2)
    gate = blocks[:, 1::2].reshape(m, n // 2)
    c = x * (gate / (np.float32(1.0) + np.exp(-gate)))

    def column_major_bytes(values: np.ndarray) -> np.ndarray:
        bits = _float32_to_bfloat16_bits(values)
        return np.ascontiguousarray(bits.reshape(-1, order="F")).view(np.uint8)

    return NumSimCase(
        kernel=dense_gemm_persistent_swiglu.get_kernel(**config),
        args={
            "a": _float32_to_bfloat16_bits(a).reshape(-1).view(np.uint8),
            "b": _float32_to_bfloat16_bits(b).reshape(-1).view(np.uint8),
            "ab12": np.zeros(m * n * 2, dtype=np.uint8),
            "c": np.zeros(m * (n // 2) * 2, dtype=np.uint8),
            "alpha": alpha,
        },
        outputs=("ab12", "c"),
        reference=lambda: {
            "ab12": column_major_bytes(ab12),
            "c": column_major_bytes(c),
        },
        comparisons={
            "ab12": ComparisonSpec(rtol=0, atol=0),
            "c": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_cudnn_dense_blockscaled_swiglu_quant_case() -> NumSimCase:
    """Exercise FP8 block scaling plus the interleaved SwiGLU epilogue."""

    m = n = k_dim = 256
    alpha = np.float32(0.75)
    config = {
        "M": m,
        "N": n,
        "K": k_dim,
        "L": 1,
        "ab_dtype": "float8_e4m3fn",
        "sf_dtype": "float8_e8m0fnu",
        "sf_vec_size": 32,
        "ab12_dtype": "bfloat16",
        "c_dtype": "bfloat16",
        "a_major": "k",
        "b_major": "k",
        "c_major": "m",
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
        "vector_f32": False,
    }
    rows = np.arange(m, dtype=np.int64)
    a_k = (rows * 17) % k_dim
    b_k = (rows * 23 + 6) % k_dim
    a_values = (1 + rows % 2).astype(np.float32)
    b_values = (1 + (rows + 1) % 2).astype(np.float32)
    a = np.zeros((m, k_dim), dtype=np.uint8)
    b = np.zeros((n, k_dim), dtype=np.uint8)
    a[rows, a_k] = _float32_to_e4m3fn_bits(a_values)
    b[rows, b_k] = _float32_to_e4m3fn_bits(b_values)

    ab12 = (
        (a_k[:, None] == b_k[None, :]).astype(np.float32)
        * a_values[:, None]
        * b_values[None, :]
        * alpha
    )
    blocks = ab12.reshape(m, n // 32, 32)
    x = blocks[:, 0::2].reshape(m, n // 2)
    gate = blocks[:, 1::2].reshape(m, n // 2)
    c = x * (gate / (np.float32(1.0) + np.exp(-gate)))

    def column_major_bytes(values: np.ndarray) -> np.ndarray:
        bits = _float32_to_bfloat16_bits(values)
        return np.ascontiguousarray(bits.reshape(-1, order="F")).view(np.uint8)

    input_scale_bytes = 2 * 2 * 512
    output_scale_bytes = 2 * 512
    return NumSimCase(
        kernel=dense_blockscaled_gemm_persistent_swiglu_interleaved_quant.get_kernel(**config),
        args={
            "a": a.reshape(-1),
            "b": b.reshape(-1),
            "sfa": np.full(input_scale_bytes, np.uint8(0x7F), dtype=np.uint8),
            "sfb": np.full(input_scale_bytes, np.uint8(0x7F), dtype=np.uint8),
            "c": np.zeros(m * (n // 2) * 2, dtype=np.uint8),
            "ab12": np.zeros(m * n * 2, dtype=np.uint8),
            "amax": np.zeros(1, dtype=np.float32),
            "sfc": np.zeros(output_scale_bytes, dtype=np.uint8),
            "norm_const": np.ones(1, dtype=np.float32),
            "alpha": alpha,
        },
        outputs=("ab12", "c"),
        reference=lambda: {
            "ab12": column_major_bytes(ab12),
            "c": column_major_bytes(c),
        },
        comparisons={
            "ab12": ComparisonSpec(rtol=0, atol=0),
            "c": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_cudnn_dense_blockscaled_srelu_quant_case() -> NumSimCase:
    """Exercise FP4 block scaling, squared ReLU, probability scaling, and amax."""

    config_entry = next(
        config
        for config in dense_blockscaled_gemm_persistent_srelu_quant.CONFIGS
        if config["label"] == "source_c_float32"
    )
    config = {key: value for key, value in config_entry.items() if key != "label"}
    m, n, k_dim, batch = (config[key] for key in ("M", "N", "K", "L"))

    diagonal = np.arange(min(m, n, k_dim), dtype=np.int64)
    a_codes = np.zeros((batch, m, k_dim), dtype=np.uint8)
    b_codes = np.zeros((batch, n, k_dim), dtype=np.uint8)
    for batch_index in range(batch):
        magnitudes = np.where((diagonal + batch_index) % 3 == 0, 2.0, 1.0).astype(np.float32)
        signs = np.where((diagonal + batch_index) % 2 == 0, 1.0, -1.0).astype(np.float32)
        a_codes[batch_index, diagonal, diagonal] = _float32_to_e2m1_bits(magnitudes * signs)
        b_codes[batch_index, diagonal, diagonal] = _float32_to_e2m1_bits(np.ones_like(magnitudes))

    a_values = _e2m1_bits_to_float32(a_codes)
    b_values = _e2m1_bits_to_float32(b_codes)
    accum = np.einsum("lmk,lnk->lmn", a_values, b_values, dtype=np.float32)
    c_values = (accum * np.float32(config["alpha"])).astype(np.float32)
    rows = np.arange(m, dtype=np.int64)[None, :]
    batches = np.arange(batch, dtype=np.int64)[:, None]
    prob = ((((rows // 16) + batches) % 3).astype(np.float32) + np.float32(1.0)) * np.float32(0.125)
    relu = np.maximum(c_values, np.float32(0.0)).astype(np.float32)
    squared = (relu * relu).astype(np.float32)
    d_values = (squared * prob[:, :, None]).astype(np.float32)

    sfa_bytes = batch * math.ceil(m / 128) * math.ceil(k_dim / (4 * config["sf_vec_size"])) * 512
    sfb_bytes = batch * math.ceil(n / 128) * math.ceil(k_dim / (4 * config["sf_vec_size"])) * 512
    expected_c = np.ascontiguousarray(c_values).reshape(-1).view(np.uint8)
    expected_d = (
        np.ascontiguousarray(_float32_to_bfloat16_bits(d_values)).reshape(-1).view(np.uint8)
    )
    expected_amax = np.array([np.max(d_values)], dtype=np.float32)

    return NumSimCase(
        kernel=dense_blockscaled_gemm_persistent_srelu_quant.get_kernel(**config),
        args={
            "a": _pack_e2m1(a_codes).reshape(-1),
            "b": _pack_e2m1(b_codes).reshape(-1),
            "sfa": np.full(sfa_bytes, np.uint8(0x7F), dtype=np.uint8),
            "sfb": np.full(sfb_bytes, np.uint8(0x7F), dtype=np.uint8),
            "c": np.zeros(m * n * batch * 4, dtype=np.uint8),
            "d": np.zeros(m * n * batch * 2, dtype=np.uint8),
            "prob": prob.reshape(-1),
            "amax": np.zeros(1, dtype=np.float32),
            "alpha": np.float32(config["alpha"]),
        },
        outputs=("c", "d", "amax"),
        reference=lambda: {
            "c": expected_c.copy(),
            "d": expected_d.copy(),
            "amax": expected_amax.copy(),
        },
        comparisons={
            "c": ComparisonSpec(rtol=0, atol=0),
            "d": ComparisonSpec(rtol=0, atol=0),
            "amax": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_cudnn_dense_blockscaled_dsrelu_quant_case() -> NumSimCase:
    """Exercise FP4 GEMM, dSReLU, probability reduction, and amax."""

    config_entry = next(
        config
        for config in dense_blockscaled_gemm_persistent_dsrelu_quant.CONFIGS
        if config["label"] == "l1"
    )
    config = {key: value for key, value in config_entry.items() if key != "label"}
    m, n, k_dim, batch = (config[key] for key in ("M", "N", "K", "L"))

    diagonal = np.arange(min(m, n, k_dim), dtype=np.int64)
    a_codes = np.zeros((batch, m, k_dim), dtype=np.uint8)
    b_codes = np.zeros((batch, n, k_dim), dtype=np.uint8)
    one = _float32_to_e2m1_bits(np.array([1.0], dtype=np.float32))[0]
    a_codes[:, diagonal, diagonal] = one
    b_codes[:, diagonal, diagonal] = one

    alpha = np.float32(config["alpha"])
    probability = np.float32(0.5)
    values = np.zeros((batch, m, n), dtype=np.float32)
    values[:, diagonal, diagonal] = alpha
    c_values = np.ones_like(values)
    d_values = np.float32(2.0) * np.maximum(values, np.float32(0.0)) * c_values * probability
    dprob = np.sum(np.square(np.maximum(values, np.float32(0.0))) * c_values, axis=2)

    sfa_bytes = batch * math.ceil(m / 128) * math.ceil(k_dim / (4 * config["sf_vec_size"])) * 512
    sfb_bytes = batch * math.ceil(n / 128) * math.ceil(k_dim / (4 * config["sf_vec_size"])) * 512
    return NumSimCase(
        kernel=dense_blockscaled_gemm_persistent_dsrelu_quant.get_kernel(**config),
        args={
            "a": _pack_e2m1(a_codes).reshape(-1),
            "b": _pack_e2m1(b_codes).reshape(-1),
            "sfa": np.full(sfa_bytes, np.uint8(0x7F), dtype=np.uint8),
            "sfb": np.full(sfb_bytes, np.uint8(0x7F), dtype=np.uint8),
            "c": _float32_to_bfloat16_bits(c_values).reshape(-1).view(np.uint8),
            "d": np.zeros(m * n * batch * 2, dtype=np.uint8),
            "prob": np.full(m * batch, probability, dtype=np.float32),
            "dprob": np.zeros(m * batch, dtype=np.float32),
            "amax": np.full(1, -np.inf, dtype=np.float32),
            "alpha": alpha,
        },
        outputs=("d", "dprob", "amax"),
        reference=lambda: {
            "d": _float32_to_bfloat16_bits(d_values).reshape(-1).view(np.uint8),
            "dprob": dprob.reshape(-1),
            "amax": np.array([np.max(np.abs(d_values))], dtype=np.float32),
        },
        comparisons={
            "d": ComparisonSpec(rtol=0, atol=0),
            "dprob": ComparisonSpec(rtol=1e-5, atol=1e-5),
            "amax": ComparisonSpec(rtol=1e-5, atol=1e-5),
        },
    )


def _zero_projection_outputs(tokens: int, num_heads: int) -> dict[str, np.ndarray]:
    head_dim = 192
    block = 32
    return {
        "out_fp8_row": np.zeros(tokens * num_heads * head_dim, dtype=np.uint8),
        "out_scales_row": np.zeros(tokens * num_heads * (head_dim // block), dtype=np.uint8),
        "out_fp8_col": np.zeros(tokens * num_heads * head_dim, dtype=np.uint8),
        "out_scales_col": np.zeros((tokens // block) * num_heads * head_dim, dtype=np.uint8),
    }


def prepare_cudnn_gemm_proj_rope_mxfp8_bf16in_case() -> NumSimCase:
    """Exercise the smallest legal BF16 projection and both MXFP8 layouts."""

    tokens, k_dim, num_heads = 128, 1536, 128
    outputs = _zero_projection_outputs(tokens, num_heads)
    trig_shape = tokens * 64
    return NumSimCase(
        kernel=gemm_proj_rope_mxfp8_bf16in.get_kernel(
            tokens=tokens,
            k_dim=k_dim,
            num_heads=num_heads,
            w_out_in=False,
        ),
        args={
            "x": np.zeros(tokens * k_dim, dtype=np.uint16),
            "w": np.zeros(num_heads * 192 * k_dim, dtype=np.uint16),
            "cos": _float32_to_bfloat16_bits(np.ones(trig_shape, dtype=np.float32)),
            "sin": np.zeros(trig_shape, dtype=np.uint16),
            **{name: value.copy() for name, value in outputs.items()},
        },
        outputs=tuple(outputs),
        reference=lambda: {name: value.copy() for name, value in outputs.items()},
        comparisons={name: ComparisonSpec(rtol=0, atol=0) for name in outputs},
    )


def prepare_cudnn_gemm_proj_rope_mxfp8_mxfp8in_case() -> NumSimCase:
    """Exercise MXFP8 inputs, RoPE, and both output scale orientations."""

    tokens, k_dim, num_heads = 128, 1536, 128
    outputs = _zero_projection_outputs(tokens, num_heads)
    trig_shape = tokens * 64
    return NumSimCase(
        kernel=gemm_proj_rope_mxfp8_mxfp8in.get_kernel(
            tokens=tokens,
            k_dim=k_dim,
            num_heads=num_heads,
        ),
        args={
            "x_code": np.zeros(tokens * k_dim, dtype=np.uint8),
            "x_scale": np.full(tokens * (k_dim // 32), np.uint8(0x7F), dtype=np.uint8),
            "w_code": np.zeros(num_heads * 192 * k_dim, dtype=np.uint8),
            "w_scale": np.full(num_heads * 192 * (k_dim // 32), np.uint8(0x7F), dtype=np.uint8),
            "cos": _float32_to_bfloat16_bits(np.ones(trig_shape, dtype=np.float32)),
            "sin": np.zeros(trig_shape, dtype=np.uint16),
            **{name: value.copy() for name, value in outputs.items()},
        },
        outputs=tuple(outputs),
        reference=lambda: {name: value.copy() for name, value in outputs.items()},
        comparisons={name: ComparisonSpec(rtol=0, atol=0) for name in outputs},
    )


def prepare_cudnn_moe_dglu_dbias_case() -> NumSimCase:
    """Exercise dense grouped FP4 GEMM, dSwiGLU, and BF16 dbias reduction."""

    rows = n = k_dim = 256
    config = {
        "group_m_list": [rows],
        "N": n,
        "K": k_dim,
        "weight_mode": "dense",
        "sched": "static",
        "act": "dswiglu",
        "ab_dtype": "float4_e2m1fn",
        "sf_dtype": "float8_e8m0fnu",
        "sf_vec_size": 16,
        "c_dtype": "bfloat16",
        "d_dtype": "bfloat16",
        "b_major": "k",
        "mma_tiler_mn": (128, 256),
        "cluster_shape_mn": (1, 1),
        "vectorized_f32": False,
        "with_dbias": True,
        "with_prob": False,
        "with_amax": False,
        "discrete_col_sfd": False,
        "linear_offset": 1.0,
        "geglu_alpha": 1.702,
        "glu_clamp_max": 7.0,
        "glu_clamp_min": -7.0,
        "situ_beta1": 4.0,
        "situ_beta2": 25.0,
    }
    codes = np.zeros((rows, k_dim), dtype=np.uint8)
    codes[np.arange(rows), np.arange(rows)] = np.uint8(2)  # FP4 E2M1 1.0
    a = _pack_e2m1(codes)
    b = _pack_e2m1(codes)

    gate = np.zeros((rows, n), dtype=np.float32)
    up = np.ones((rows, n), dtype=np.float32)
    c_blocks = np.stack((gate.reshape(rows, n // 32, 32), up.reshape(rows, n // 32, 32)), axis=2)
    c = c_blocks.reshape(rows, 2 * n)

    gate_grad = np.eye(rows, n, dtype=np.float32) * np.float32(0.5)
    up_grad = np.zeros_like(gate_grad)
    d_blocks = np.stack(
        (
            gate_grad.reshape(rows, n // 32, 32),
            up_grad.reshape(rows, n // 32, 32),
        ),
        axis=2,
    )
    d = d_blocks.reshape(rows, 2 * n)
    dbias = np.sum(d, axis=0, dtype=np.float32, keepdims=True)
    scale_bytes = 1 * 2 * 4 * 32 * 4 * 4

    return NumSimCase(
        kernel=moe_blockscaled_grouped_gemm_dglu_dbias.get_kernel(**config),
        args={
            "a": a.reshape(-1),
            "b": b.reshape(-1),
            "sfa": np.full(scale_bytes, np.uint8(0x7F), dtype=np.uint8),
            "sfb": np.full(scale_bytes, np.uint8(0x7F), dtype=np.uint8),
            "c": _float32_to_bfloat16_bits(c).reshape(-1).view(np.uint8),
            "d_row": np.zeros(rows * 2 * n * 2, dtype=np.uint8),
            "padded_offsets": np.array([rows], dtype=np.int32),
            "alpha": np.ones(1, dtype=np.float32),
            "beta": np.ones(1, dtype=np.float32),
            "dbias": np.zeros(2 * n * 2, dtype=np.uint8),
        },
        outputs=("d_row", "dbias"),
        reference=lambda: {
            "d_row": _float32_to_bfloat16_bits(d).reshape(-1).view(np.uint8),
            "dbias": _float32_to_bfloat16_bits(dbias).reshape(-1).view(np.uint8),
        },
        comparisons={
            "d_row": ComparisonSpec(rtol=0, atol=0),
            "dbias": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_cudnn_moe_bf16_dglu_dbias_case() -> NumSimCase:
    """Exercise grouped BF16 GEMM, dSwiGLU, dprob, and dbias."""

    rows = 256
    n = 32
    k_dim = 64
    config = {
        "group_m_list": [rows],
        "N": n,
        "K_dim": k_dim,
        "weight_mode": "dense",
        "sched": "static",
        "act": "dswiglu",
        "c_dtype": "bfloat16",
        "d_dtype": "bfloat16",
        "b_major": "k",
        "mma_tiler_mn": (128, 32),
        "cluster_shape_mn": (1, 1),
        "vectorized_f32": False,
        "with_dbias": True,
        "linear_offset": 1.0,
    }

    a = np.zeros((rows, k_dim), dtype=np.float32)
    b = np.zeros((1, n, k_dim), dtype=np.float32)
    diagonal = np.arange(n)
    a[diagonal, diagonal] = np.float32(1.0)
    b[0, diagonal, diagonal] = np.float32(1.0)

    c = np.zeros((rows, 2 * n), dtype=np.float32)
    c[:, n:] = np.float32(1.0)
    expected_d = np.zeros_like(c)
    expected_d[diagonal, diagonal] = np.float32(0.5)
    expected_dbias = np.zeros((1, 2 * n), dtype=np.float32)
    expected_dbias[0, :n] = np.float32(0.5)

    return NumSimCase(
        kernel=moe_grouped_gemm_dglu_dbias.get_kernel(**config),
        args={
            "a": _float32_to_bfloat16_bits(a).reshape(-1).view(np.uint8),
            "b": _float32_to_bfloat16_bits(b).reshape(-1).view(np.uint8),
            "c": _float32_to_bfloat16_bits(c).reshape(-1).view(np.uint8),
            "d_row": np.zeros(rows * 2 * n * 2, dtype=np.uint8),
            "padded_offsets": np.array([rows], dtype=np.int32),
            "alpha": np.ones(1, dtype=np.float32),
            "beta": np.ones(1, dtype=np.float32),
            "prob": np.ones(rows, dtype=np.float32),
            "dprob": np.zeros(rows, dtype=np.float32),
            "dbias": np.zeros(2 * n * 2, dtype=np.uint8),
        },
        outputs=("d_row", "dprob", "dbias"),
        reference=lambda: {
            "d_row": _float32_to_bfloat16_bits(expected_d).reshape(-1).view(np.uint8),
            "dprob": np.zeros(rows, dtype=np.float32),
            "dbias": _float32_to_bfloat16_bits(expected_dbias).reshape(-1).view(np.uint8),
        },
        comparisons={
            "d_row": ComparisonSpec(rtol=0, atol=0),
            "dprob": ComparisonSpec(rtol=0, atol=0),
            "dbias": ComparisonSpec(rtol=0, atol=0),
        },
    )


def _aliased_views(
    array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    storage = np.ascontiguousarray(array).reshape(-1)
    raw = storage.view(np.uint8)
    return storage, raw


def _headed_map(
    base: np.ndarray,
    *,
    dtype: str,
    channels: int,
    heads: int,
    tokens: int,
    element_bytes: int,
    box_channels: int,
    box_tokens: int = 16,
) -> np.ndarray:
    return _tensor_map(
        base,
        dtype=dtype,
        global_shape=(channels, heads, tokens),
        global_strides=(channels * element_bytes, heads * channels * element_bytes),
        box_shape=(box_channels, 1, box_tokens),
    )


def _checkpoint_map(base: np.ndarray, *, rows: int, heads: int) -> np.ndarray:
    return _tensor_map(
        base,
        dtype="bfloat16",
        global_shape=(128, 128, rows, heads),
        global_strides=(128 * 2, heads * 128 * 128 * 2, 128 * 128 * 2),
        box_shape=(64, 128, 1, 1),
    )


def _host_tensor_map_descriptor(tensor_map: np.ndarray) -> np.ndarray:
    return _descriptor_storage(
        storage=np.zeros(16, dtype=np.int64),
        slots={0: tensor_map},
    )


def _one_chunk_work_item() -> np.ndarray:
    return np.array([0, 0, 0, 1, 0, 1, 0, 1], dtype=np.int32)


def prepare_cudnn_kda_bprop_case() -> NumSimCase:
    """Exercise the two-launch KDA backward path with an analytic one-token gradient."""

    config = {"seq_lens": (1,), "heads": 1, "num_sms": 1, "scale": 1.0}
    q_values = np.zeros((1, 1, 128), dtype=np.float32)
    k_values = np.zeros_like(q_values)
    v_values = np.zeros_like(q_values)
    do_values = np.zeros_like(q_values)
    q_values[0, 0, 0] = 1.0
    k_values[0, 0, 0] = 1.0
    v_values[0, 0, :4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    do_values[0, 0, :4] = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    beta = np.array([0.5], dtype=np.float32)
    gate = np.full((1, 1, 128), -0.125, dtype=np.float32)
    contraction = np.sum(v_values * do_values, dtype=np.float32)

    expected_dq = np.zeros_like(q_values)
    expected_dk = np.zeros_like(k_values)
    expected_dq[0, 0, 0] = beta[0] * contraction
    expected_dk[0, 0, 0] = beta[0] * contraction
    expected_dv = beta[0] * do_values
    expected_dgate = np.zeros_like(gate)
    expected_dbeta = np.array([contraction], dtype=np.float32)

    q, q_raw = _aliased_views(_float32_to_bfloat16_bits(q_values))
    k, k_raw = _aliased_views(_float32_to_bfloat16_bits(k_values))
    v, v_raw = _aliased_views(_float32_to_bfloat16_bits(v_values))
    do, do_raw = _aliased_views(_float32_to_bfloat16_bits(do_values))
    gate_binding, gate_raw = _aliased_views(gate)
    dq, dq_raw = _aliased_views(np.zeros((1, 1, 128), dtype=np.uint16))
    dk, dk_raw = _aliased_views(np.zeros((1, 1, 128), dtype=np.uint16))
    dv, dv_raw = _aliased_views(np.zeros((1, 1, 128), dtype=np.uint16))
    dgate, dgate_raw = _aliased_views(np.zeros((1, 1, 128), dtype=np.float32))
    checkpoints, checkpoints_raw = _aliased_views(np.zeros((1, 1, 128, 128), dtype=np.uint16))
    descriptor_workspace = np.zeros(176, dtype=np.int64)
    cu_seqlens = np.array([0, 1], dtype=np.int32)
    work_items = _one_chunk_work_item()
    work_count = np.array([1], dtype=np.int32)
    scheduler = np.zeros(8, dtype=np.int32)
    dummy_i32 = np.zeros(8, dtype=np.int32)

    host_descriptors = (
        (
            "base_q",
            _host_tensor_map_descriptor(
                _headed_map(
                    q,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                )
            ),
        ),
        (
            "base_k",
            _host_tensor_map_descriptor(
                _headed_map(
                    k,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                )
            ),
        ),
        (
            "base_v",
            _host_tensor_map_descriptor(
                _headed_map(
                    v,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                )
            ),
        ),
        (
            "base_gate",
            _host_tensor_map_descriptor(
                _headed_map(
                    gate_binding,
                    dtype="float32",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=4,
                    box_channels=32,
                )
            ),
        ),
        (
            "base_do",
            _host_tensor_map_descriptor(
                _headed_map(
                    do,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                )
            ),
        ),
        (
            "base_dq",
            _host_tensor_map_descriptor(
                _headed_map(
                    dq,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                )
            ),
        ),
        (
            "base_dk",
            _host_tensor_map_descriptor(
                _headed_map(
                    dk,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                )
            ),
        ),
        (
            "base_dv",
            _host_tensor_map_descriptor(
                _headed_map(
                    dv,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                )
            ),
        ),
        (
            "base_dgate",
            _host_tensor_map_descriptor(
                _headed_map(
                    dgate,
                    dtype="float32",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=4,
                    box_channels=32,
                )
            ),
        ),
        (
            "base_checkpoint",
            _host_tensor_map_descriptor(_checkpoint_map(checkpoints, rows=1, heads=1)),
        ),
    )
    args = {
        **{f"k0:{name}": descriptor for name, descriptor in host_descriptors},
        "k0:descriptor_workspace": descriptor_workspace,
        "k0:cu_seqlens": cu_seqlens,
        "k0:q": q_raw,
        "k0:k": k_raw,
        "k0:v": v_raw,
        "k0:gate": gate_raw,
        "k0:do": do_raw,
        "k0:dq": dq_raw,
        "k0:dk": dk_raw,
        "k0:dv": dv_raw,
        "k0:dgate": dgate_raw,
        "k0:checkpoints": checkpoints_raw,
        "k0:work_item_staging": dummy_i32,
        "k0:work_count": work_count,
        "k0:work_items": work_items,
        "k0:scheduler": scheduler,
        "k0:n_batch": 1,
        "k0:q_row_stride_bytes": 256,
        "k0:k_row_stride_bytes": 256,
        "k0:v_row_stride_bytes": 256,
        "k0:gate_row_stride_bytes": 512,
        "k0:do_row_stride_bytes": 256,
        "k0:dq_row_stride_bytes": 256,
        "k0:dk_row_stride_bytes": 256,
        "k0:dv_row_stride_bytes": 256,
        "k0:dgate_row_stride_bytes": 512,
        "k0:checkpoint_row_stride_bytes": 32768,
        "k0:checkpoint_every_n": 16,
        "k1:descriptor_workspace": descriptor_workspace,
        "k1:n_desc": 1,
        "k1:a_log": np.zeros(1, dtype=np.float32),
        "k1:dt_bias": np.zeros(128, dtype=np.float32),
        "k1:beta": beta,
        "k1:cu_seqlens": cu_seqlens,
        "k1:dgate": dgate,
        "k1:dbeta": np.zeros(1, dtype=np.float32),
        "k1:d_initial_state": np.zeros(1, dtype=np.float32),
        "k1:d_final_state": np.zeros(1, dtype=np.float32),
        "k1:work_items": work_items,
        "k1:work_count": work_count,
        "k1:scheduler": scheduler,
        "k1:scale": np.float32(1.0),
    }
    return NumSimCase(
        kernel=kda_bprop_f16.get_kernel(**config),
        args=args,
        outputs={
            "dq": "k0:dq",
            "dk": "k0:dk",
            "dv": "k0:dv",
            "dgate": "k1:dgate",
            "dbeta": "k1:dbeta",
        },
        reference=lambda: {
            "dq": expected_dq.reshape(-1),
            "dk": expected_dk.reshape(-1),
            "dv": expected_dv.reshape(-1),
            "dgate": expected_dgate.reshape(-1),
            "dbeta": expected_dbeta,
        },
        comparisons={
            "dq": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
            "dk": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
            "dv": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
            # The one-token FP64 recurrence has exact-zero dgate, while the
            # kernel's approximate exp/reciprocal cancellation leaves
            # 0.0004906058311462402 on both B200 and NumSim for this case.
            "dgate": ComparisonSpec(rtol=0, atol=6e-4),
            "dbeta": ComparisonSpec(rtol=2e-3, atol=2e-3),
        },
    )


def _host_descriptors(
    entries: tuple[tuple[str, np.ndarray], ...],
) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple((name, _host_tensor_map_descriptor(tensor_map)) for name, tensor_map in entries)


def prepare_cudnn_gdn_bprop_case() -> NumSimCase:
    """Exercise GDN's descriptor prologue and one-token analytic backward path."""

    config = {"seq_lens": (1,), "heads": 1, "num_sms": 1, "scale": 1.0}
    q_values = np.zeros((1, 1, 128), dtype=np.float32)
    k_values = np.zeros_like(q_values)
    v_values = np.zeros_like(q_values)
    do_values = np.zeros_like(q_values)
    q_values[0, 0, 0] = np.float32(1.0)
    k_values[0, 0, 0] = np.float32(1.0)
    v_values[0, 0, :4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    do_values[0, 0, :4] = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    gate = np.full((1, 1), np.float32(-0.125), dtype=np.float32)
    beta = np.array([0.5], dtype=np.float32)
    contraction = np.sum(v_values * do_values, dtype=np.float32)

    expected_dq = np.zeros_like(q_values)
    expected_dk = np.zeros_like(k_values)
    expected_dq[0, 0, 0] = beta[0] * contraction
    expected_dk[0, 0, 0] = beta[0] * contraction
    expected_dv = beta[0] * do_values
    expected_dgate = np.zeros_like(gate)
    expected_dbeta = np.array([contraction], dtype=np.float32)

    q, q_raw = _aliased_views(_float32_to_bfloat16_bits(q_values))
    k, k_raw = _aliased_views(_float32_to_bfloat16_bits(k_values))
    v, v_raw = _aliased_views(_float32_to_bfloat16_bits(v_values))
    do, do_raw = _aliased_views(_float32_to_bfloat16_bits(do_values))
    dq, dq_raw = _aliased_views(np.zeros_like(q, dtype=np.uint16))
    dk, dk_raw = _aliased_views(np.zeros_like(k, dtype=np.uint16))
    dv, dv_raw = _aliased_views(np.zeros_like(v, dtype=np.uint16))
    checkpoint, checkpoint_raw = _aliased_views(np.zeros((1, 1, 128, 128), dtype=np.uint16))
    descriptor_workspace = np.zeros(144, dtype=np.int64)
    cu_seqlens = np.array([0, 1], dtype=np.int32)
    work_items = _one_chunk_work_item()
    work_count = np.array([1], dtype=np.int32)
    scheduler = np.zeros(8, dtype=np.int32)
    dummy_i32 = np.zeros(8, dtype=np.int32)
    vector_maps = tuple(
        (
            f"base_{name}",
            _headed_map(
                value,
                dtype="bfloat16",
                channels=128,
                heads=1,
                tokens=1,
                element_bytes=2,
                box_channels=64,
                box_tokens=64,
            ),
        )
        for name, value in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("do", do),
            ("dq", dq),
            ("dk", dk),
            ("dv", dv),
        )
    )
    host_descriptors = _host_descriptors(
        (
            *vector_maps[:4],
            ("base_checkpoint", _checkpoint_map(checkpoint, rows=1, heads=1)),
            *vector_maps[4:],
        )
    )
    args = {
        **{f"k0:{name}": descriptor for name, descriptor in host_descriptors},
        "k0:descriptor_workspace": descriptor_workspace,
        "k0:cu_seqlens": cu_seqlens,
        "k0:q": q_raw,
        "k0:k": k_raw,
        "k0:v": v_raw,
        "k0:do": do_raw,
        "k0:checkpoint": checkpoint_raw,
        "k0:dq": dq_raw,
        "k0:dk": dk_raw,
        "k0:dv": dv_raw,
        "k0:work_item_staging": dummy_i32,
        "k0:work_count": work_count,
        "k0:work_items": work_items,
        "k0:scheduler_all": scheduler,
        "k0:n_batch": 1,
        "k0:q_row_stride_bytes": 256,
        "k0:k_row_stride_bytes": 256,
        "k0:v_row_stride_bytes": 256,
        "k0:do_row_stride_bytes": 256,
        "k0:checkpoint_row_stride_bytes": 32768,
        "k0:dq_row_stride_bytes": 256,
        "k0:dk_row_stride_bytes": 256,
        "k0:dv_row_stride_bytes": 256,
        "k0:checkpoint_every_n": 64,
        "k1:descriptor_workspace": descriptor_workspace,
        "k1:n_desc": 1,
        "k1:gate": gate.reshape(-1),
        "k1:a_log": np.zeros(1, dtype=np.float32),
        "k1:dt_bias": np.zeros(128, dtype=np.float32),
        "k1:beta": beta,
        "k1:dgate": np.zeros_like(gate).reshape(-1),
        "k1:dbeta": np.zeros(1, dtype=np.float32),
        "k1:cu_seqlens": cu_seqlens,
        "k1:dstate0": np.zeros(1, dtype=np.float32),
        "k1:dstate_in": np.zeros(1, dtype=np.float32),
        "k1:work_items": work_items,
        "k1:work_count": work_count,
        "k1:scheduler": scheduler,
        "k1:scale": np.float32(1.0),
    }
    return NumSimCase(
        kernel=gdn_bprop_f16.get_kernel(**config),
        args=args,
        outputs={
            "dq": "k0:dq",
            "dk": "k0:dk",
            "dv": "k0:dv",
            "dgate": "k1:dgate",
            "dbeta": "k1:dbeta",
        },
        reference=lambda: {
            "dq": expected_dq.reshape(-1),
            "dk": expected_dk.reshape(-1),
            "dv": expected_dv.reshape(-1),
            "dgate": expected_dgate.reshape(-1),
            "dbeta": expected_dbeta,
        },
        comparisons={
            "dq": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
            "dk": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
            "dv": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
            "dgate": ComparisonSpec(rtol=0, atol=6e-4),
            "dbeta": ComparisonSpec(rtol=2e-3, atol=2e-3),
        },
    )


def prepare_cudnn_gdn_prefill_case() -> NumSimCase:
    """Exercise GDN's descriptor prologue and one-token recurrence."""

    config = {
        "seq_lens": (1,),
        "heads": 1,
        "q_heads": 1,
        "k_heads": 1,
        "v_heads": 1,
        "num_sms": 1,
        "scale": 1.0,
        "checkpoint_every_n_tokens": 0,
        "store_final_state": True,
    }
    q_values = np.zeros((1, 1, 128), dtype=np.float32)
    k_values = np.zeros_like(q_values)
    v_values = np.zeros_like(q_values)
    q_values[0, 0, 0] = 1.0
    k_values[0, 0, 0] = 1.0
    v_values[0, 0, :4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    gate = np.array([-0.125], dtype=np.float32)
    beta = np.array([0.5], dtype=np.float32)
    expected_output = np.float32(0.5) * v_values
    expected_final_state = np.zeros((1, 1, 128, 128), dtype=np.float32)
    expected_final_state[0, 0, :, 0] = expected_output[0, 0]

    q, q_raw = _aliased_views(_float32_to_bfloat16_bits(q_values))
    k, k_raw = _aliased_views(_float32_to_bfloat16_bits(k_values))
    v, v_raw = _aliased_views(_float32_to_bfloat16_bits(v_values))
    output, output_raw = _aliased_views(np.zeros((1, 1, 128), dtype=np.uint16))
    checkpoint, checkpoint_raw = _aliased_views(np.zeros((1, 1, 128, 128), dtype=np.uint16))
    final_state = np.zeros(1 * 1 * 128 * 128, dtype=np.float32)
    descriptor_workspace = np.zeros(96, dtype=np.int64)
    cu_seqlens = np.array([0, 1], dtype=np.int32)
    work_items = np.zeros(8, dtype=np.int32)
    work_count = np.array([1], dtype=np.int32)
    dummy_i32 = np.zeros(8, dtype=np.int32)
    host_descriptors = _host_descriptors(
        (
            (
                "base_q",
                _headed_map(
                    q,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                    box_tokens=64,
                ),
            ),
            (
                "base_k",
                _headed_map(
                    k,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                    box_tokens=64,
                ),
            ),
            (
                "base_v",
                _headed_map(
                    v,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                    box_tokens=64,
                ),
            ),
            (
                "base_o",
                _headed_map(
                    output,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                    box_tokens=64,
                ),
            ),
            ("base_checkpoint", _checkpoint_map(checkpoint, rows=1, heads=1)),
        )
    )
    return NumSimCase(
        kernel=gdn_prefill_f16.get_kernel(**config),
        args={
            **{f"k0:{name}": descriptor for name, descriptor in host_descriptors},
            "k0:descriptor_workspace": descriptor_workspace,
            "k0:cu_seqlens": cu_seqlens,
            "k0:q": q_raw,
            "k0:k": k_raw,
            "k0:v": v_raw,
            "k0:o": output_raw,
            "k0:checkpoint": checkpoint_raw,
            "k0:work_item_staging": dummy_i32,
            "k0:work_count": work_count,
            "k0:work_items": work_items,
            "k0:scheduler": dummy_i32,
            "k0:n_batch": 1,
            "k0:q_row_stride_bytes": 256,
            "k0:k_row_stride_bytes": 256,
            "k0:v_row_stride_bytes": 256,
            "k0:o_row_stride_bytes": 256,
            "k0:checkpoint_row_stride_bytes": 32768,
            "k0:checkpoint_every_n": 0,
            "k1:descriptor_workspace": descriptor_workspace,
            "k1:n_desc": 1,
            "k1:q": q,
            "k1:k": k,
            "k1:v": v,
            "k1:gate": gate,
            "k1:a_log": np.zeros(1, dtype=np.float32),
            "k1:dt_bias": np.zeros(1, dtype=np.float32),
            "k1:beta": beta,
            "k1:cu_seqlens": cu_seqlens,
            "k1:initial_state": np.zeros(1, dtype=np.float32),
            "k1:o": output,
            "k1:final_state": final_state,
            "k1:work_items": work_items,
            "k1:work_count": work_count,
            "k1:scheduler": dummy_i32,
            "k1:scale": np.float32(1.0),
            "k1:checkpoint_every_n": 0,
        },
        outputs={"output": "k1:o", "final_state": "k1:final_state"},
        reference=lambda: {
            "output": expected_output.reshape(-1),
            "final_state": expected_final_state.reshape(-1),
        },
        comparisons={
            "output": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
            "final_state": ComparisonSpec(rtol=2e-2, atol=2e-3),
        },
    )


def prepare_cudnn_gdn_recompute_case() -> NumSimCase:
    """Exercise GDN recompute's prologue and one-token state update."""

    config = {
        "seq_lens": (1,),
        "heads": 1,
        "k_heads": 1,
        "v_heads": 1,
        "num_sms": 1,
        "checkpoint_every_n_tokens": 0,
        "store_final_state": True,
    }
    k_values = np.zeros((1, 1, 128), dtype=np.float32)
    v_values = np.zeros_like(k_values)
    k_values[0, 0, 0] = 1.0
    v_values[0, 0, :4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    gate = np.array([-0.125], dtype=np.float32)
    beta = np.array([0.5], dtype=np.float32)
    expected_final_state = np.zeros((1, 1, 128, 128), dtype=np.float32)
    expected_final_state[0, 0, :, 0] = np.float32(0.5) * v_values[0, 0]

    k, k_raw = _aliased_views(_float32_to_bfloat16_bits(k_values))
    v, v_raw = _aliased_views(_float32_to_bfloat16_bits(v_values))
    checkpoint, checkpoint_raw = _aliased_views(np.zeros((1, 1, 128, 128), dtype=np.uint16))
    final_state = np.zeros(1 * 1 * 128 * 128, dtype=np.float32)
    descriptor_workspace = np.zeros(64, dtype=np.int64)
    cu_seqlens = np.array([0, 1], dtype=np.int32)
    work_items = np.zeros(8, dtype=np.int32)
    work_count = np.array([1], dtype=np.int32)
    scheduler = np.zeros(8, dtype=np.int32)
    host_descriptors = _host_descriptors(
        (
            (
                "base_k",
                _headed_map(
                    k,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                    box_tokens=64,
                ),
            ),
            (
                "base_v",
                _headed_map(
                    v,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                    box_tokens=64,
                ),
            ),
            ("base_checkpoint", _checkpoint_map(checkpoint, rows=1, heads=1)),
        )
    )
    return NumSimCase(
        kernel=gdn_recompute_f16.get_kernel(**config),
        args={
            **{f"k0:{name}": descriptor for name, descriptor in host_descriptors},
            "k0:descriptor_workspace": descriptor_workspace,
            "k0:cu_seqlens": cu_seqlens,
            "k0:k": k_raw,
            "k0:v": v_raw,
            "k0:checkpoint": checkpoint_raw,
            "k0:work_item_staging": scheduler,
            "k0:work_count": work_count,
            "k0:work_items": work_items,
            "k0:scheduler": scheduler,
            "k0:n_batch": 1,
            "k0:k_row_stride_bytes": 256,
            "k0:v_row_stride_bytes": 256,
            "k0:checkpoint_row_stride_bytes": 32768,
            "k0:checkpoint_every_n": 0,
            "k1:descriptor_workspace": descriptor_workspace,
            "k1:n_desc": 1,
            "k1:k": k,
            "k1:v": v,
            "k1:gate": gate,
            "k1:a_log": np.zeros(1, dtype=np.float32),
            "k1:dt_bias": np.zeros(1, dtype=np.float32),
            "k1:beta": beta,
            "k1:cu_seqlens": cu_seqlens,
            "k1:initial_state": np.zeros(1, dtype=np.float32),
            "k1:final_state": final_state,
            "k1:work_items": work_items,
            "k1:work_count": work_count,
            "k1:scheduler": scheduler,
            "k1:checkpoint_every_n": 0,
        },
        outputs={"final_state": "k1:final_state"},
        reference=lambda: {"final_state": expected_final_state.reshape(-1)},
        comparisons={"final_state": ComparisonSpec(rtol=2e-2, atol=2e-3)},
    )


def prepare_cudnn_gdn2_recompute_case() -> NumSimCase:
    """Two chunks, a ragged tail, nonzero initial state, and checkpoint publication."""
    tokens = 17
    rng = np.random.default_rng(20260905)
    values = {
        "k": (rng.standard_normal((tokens, 128)) / 32).astype(np.float16),
        "v": (rng.standard_normal((tokens, 128)) / 4).astype(np.float16),
        "gate": rng.uniform(-0.025, -0.005, (tokens, 128)).astype(np.float32),
        "beta": rng.uniform(0.25, 0.75, (tokens, 128)).astype(np.float16),
        "w": rng.uniform(0.25, 0.75, (tokens, 128)).astype(np.float16),
    }
    initial = (rng.standard_normal((128, 128)) / 128).astype(np.float32)
    state = initial.astype(np.float64)
    checkpoints = []
    # Token-wise GDN2 recurrence, independent of the kernel's chunked inverse
    # and its register/TMEM layouts. beta gates prediction; w gates the value.
    for token in range(tokens):
        if token % 16 == 0:
            checkpoints.append(state.copy())
        key, value, gate, beta, weight = (
            values[name][token].astype(np.float64) for name in ("k", "v", "gate", "beta", "w")
        )
        state *= np.exp(gate)[None, :]
        state += np.outer(weight * value - state @ (key * beta), key)

    bindings = {name: _aliased_views(value) for name, value in values.items()}
    checkpoint, checkpoint_raw = _aliased_views(np.full((2, 128, 128), np.nan, np.float16))
    workspace = np.zeros(6 * 16, dtype=np.int64)
    work_items = np.zeros(8, dtype=np.int32)
    work_count = np.ones(1, dtype=np.int32)
    scheduler = np.zeros(4, dtype=np.int32)
    cu_seqlens = np.array([0, tokens], dtype=np.int32)
    args = {}
    for name, (buffer, raw) in bindings.items():
        element_bytes = buffer.dtype.itemsize
        args[f"k0:base_{name}"] = _headed_map(
            buffer,
            dtype=str(buffer.dtype),
            channels=128,
            heads=1,
            tokens=tokens,
            element_bytes=element_bytes,
            box_channels=128 // element_bytes,
        )
        args[f"k0:{name}"] = raw
        args[f"k0:{name}_row_stride_bytes"] = 128 * element_bytes
        args[f"k1:{name}"] = buffer
    args.update(
        {
            "k0:base_checkpoint": _tensor_map(
                checkpoint,
                dtype="float16",
                global_shape=(128, 128, 2, 1),
                global_strides=(256, 32768, 32768),
                box_shape=(64, 128, 1, 1),
            ),
            "k0:checkpoint": checkpoint_raw,
            "k0:checkpoint_row_stride_bytes": 32768,
            "k0:work_item_staging": scheduler,
            "k0:n_batch": 1,
            "k1:n_desc": 1,
            "k1:a_log": np.zeros(1, dtype=np.float32),
            "k1:dt_bias": np.zeros(128, dtype=np.float32),
            "k1:initial_state": initial.reshape(-1),
            "k1:final_state": np.full(128 * 128, np.nan, dtype=np.float32),
        }
    )
    for phase in ("k0", "k1"):
        args.update(
            {
                f"{phase}:{name}": value
                for name, value in {
                    "descriptor_workspace": workspace,
                    "cu_seqlens": cu_seqlens,
                    "work_items": work_items,
                    "work_count": work_count,
                    "scheduler": scheduler,
                    "checkpoint_every_n": 16,
                }.items()
            }
        )
    module = load_tirx_kernel("cudnn_sm100_gdn2_recompute_f16")
    return NumSimCase(
        kernel=module.get_kernel(
            seq_lens=(tokens,),
            heads=1,
            num_sms=1,
            io_dtype="float16",
            state_dtype="float32",
            use_initial_state=True,
            checkpoint_every_n_tokens=16,
            store_final_state=True,
        ),
        args=args,
        outputs={"final_state": "k1:final_state", "checkpoints": "k0:base_checkpoint"},
        reference=lambda: {
            "final_state": state.reshape(-1),
            "checkpoints": np.asarray(checkpoints).reshape(1, 2, 128, 128),
        },
        comparisons={
            "final_state": ComparisonSpec(rtol=2e-2, atol=3e-4),
            "checkpoints": ComparisonSpec(rtol=2e-2, atol=3e-4),
        },
    )


def prepare_cudnn_gdn2_prefill_case() -> NumSimCase:
    """Exercise GDN2's descriptor prologue and one-token recurrence."""

    config = {
        "seq_lens": (1,),
        "heads": 1,
        "q_heads": 1,
        "k_heads": 1,
        "v_heads": 1,
        "num_sms": 1,
        "scale": 1.0,
        "io_dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "checkpoint_every_n_tokens": 16,
        "store_final_state": True,
    }
    q_values = np.zeros((1, 1, 128), dtype=np.float32)
    k_values = np.zeros_like(q_values)
    v_values = np.zeros_like(q_values)
    q_values[0, 0, 0] = 1.0
    k_values[0, 0, 0] = 1.0
    v_values[0, 0, :4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    gate_values = np.full((1, 1, 128), -0.125, dtype=np.float32)
    beta_values = np.full((1, 1, 128), 0.5, dtype=np.float32)
    w_values = np.full((1, 1, 128), 0.5, dtype=np.float32)

    expected_output = w_values * v_values
    expected_final_state = np.zeros((1, 1, 128, 128), dtype=np.float32)
    expected_final_state[0, 0, :, 0] = expected_output[0, 0]

    q, q_raw = _aliased_views(_float32_to_bfloat16_bits(q_values))
    k, k_raw = _aliased_views(_float32_to_bfloat16_bits(k_values))
    v, v_raw = _aliased_views(_float32_to_bfloat16_bits(v_values))
    gate, gate_raw = _aliased_views(gate_values)
    beta, beta_raw = _aliased_views(_float32_to_bfloat16_bits(beta_values))
    w, w_raw = _aliased_views(_float32_to_bfloat16_bits(w_values))
    output, output_raw = _aliased_views(np.zeros((1, 1, 128), dtype=np.uint16))
    checkpoint, checkpoint_raw = _aliased_views(np.zeros((1, 1, 128, 128), dtype=np.uint16))
    final_state = np.zeros(1 * 1 * 128 * 128, dtype=np.uint16)
    descriptor_workspace = np.zeros(128, dtype=np.int64)
    cu_seqlens = np.array([0, 1], dtype=np.int32)
    work_items = np.zeros(8, dtype=np.int32)
    work_count = np.array([1], dtype=np.int32)
    dummy_i32 = np.zeros(8, dtype=np.int32)

    host_descriptors = _host_descriptors(
        (
            (
                "base_q",
                _headed_map(
                    q,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                ),
            ),
            (
                "base_k",
                _headed_map(
                    k,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                ),
            ),
            (
                "base_v",
                _headed_map(
                    v,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                ),
            ),
            (
                "base_gate",
                _headed_map(
                    gate,
                    dtype="float32",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=4,
                    box_channels=32,
                ),
            ),
            (
                "base_beta",
                _headed_map(
                    beta,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                ),
            ),
            (
                "base_w",
                _headed_map(
                    w,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                ),
            ),
            (
                "base_o",
                _headed_map(
                    output,
                    dtype="bfloat16",
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=2,
                    box_channels=64,
                ),
            ),
            ("base_checkpoint", _checkpoint_map(checkpoint, rows=1, heads=1)),
        )
    )
    return NumSimCase(
        kernel=gdn2_prefill_f16.get_kernel(**config),
        args={
            **{f"k0:{name}": descriptor for name, descriptor in host_descriptors},
            "k0:descriptor_workspace": descriptor_workspace,
            "k0:cu_seqlens": cu_seqlens,
            "k0:q": q_raw,
            "k0:k": k_raw,
            "k0:v": v_raw,
            "k0:gate": gate_raw,
            "k0:beta": beta_raw,
            "k0:w": w_raw,
            "k0:o": output_raw,
            "k0:checkpoint": checkpoint_raw,
            "k0:work_item_staging": dummy_i32,
            "k0:work_count": work_count,
            "k0:work_items": work_items,
            "k0:scheduler": dummy_i32,
            "k0:n_batch": 1,
            "k0:q_row_stride_bytes": 256,
            "k0:k_row_stride_bytes": 256,
            "k0:v_row_stride_bytes": 256,
            "k0:gate_row_stride_bytes": 512,
            "k0:beta_row_stride_bytes": 256,
            "k0:w_row_stride_bytes": 256,
            "k0:o_row_stride_bytes": 256,
            "k0:checkpoint_row_stride_bytes": 32768,
            "k0:checkpoint_every_n": 16,
            "k1:descriptor_workspace": descriptor_workspace,
            "k1:n_desc": 1,
            "k1:q": q,
            "k1:k": k,
            "k1:v": v,
            "k1:gate": gate,
            "k1:a_log": np.zeros(1, dtype=np.float32),
            "k1:dt_bias": np.zeros(128, dtype=np.float32),
            "k1:beta": beta,
            "k1:w": w,
            "k1:cu_seqlens": cu_seqlens,
            "k1:initial_state": np.zeros(1, dtype=np.uint16),
            "k1:o": output,
            "k1:final_state": final_state,
            "k1:work_items": work_items,
            "k1:work_count": work_count,
            "k1:scheduler": dummy_i32,
            "k1:scale": np.float32(1.0),
            "k1:checkpoint_every_n": 16,
        },
        outputs={"output": "k1:o", "final_state": "k1:final_state"},
        reference=lambda: {
            "output": expected_output.reshape(-1),
            "final_state": expected_final_state.reshape(-1),
        },
        comparisons={
            "output": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
            "final_state": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16"),
        },
    )


def prepare_cudnn_gdn2_bprop_case() -> NumSimCase:
    """Exercise GDN2 backward's descriptor prologue and one-token gradient."""

    config = {"seq_lens": (1,), "heads": 1, "num_sms": 1, "scale": 1.0}
    q_values = np.zeros((1, 1, 128), dtype=np.float32)
    k_values = np.zeros_like(q_values)
    v_values = np.zeros_like(q_values)
    do_values = np.zeros_like(q_values)
    q_values[0, 0, 0] = 1.0
    k_values[0, 0, 0] = 1.0
    v_values[0, 0, :4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    do_values[0, 0, :4] = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    gate_values = np.full((1, 1, 128), -0.125, dtype=np.float32)
    beta_values = np.full((1, 1, 128), 0.5, dtype=np.float32)
    w_values = np.full((1, 1, 128), 0.5, dtype=np.float32)
    contraction = np.sum(w_values * v_values * do_values, dtype=np.float32)

    expected_dq = np.zeros_like(q_values)
    expected_dk = np.zeros_like(k_values)
    expected_dq[0, 0, 0] = contraction
    expected_dk[0, 0, 0] = contraction
    expected_dv = w_values * do_values
    expected_dgate = np.zeros_like(gate_values)
    expected_dw = v_values * do_values
    expected_dbeta = np.zeros_like(beta_values)

    def input_views(values: np.ndarray, dtype: str):
        stored = _float32_to_bfloat16_bits(values) if dtype == "bfloat16" else values
        return _aliased_views(stored)

    def output_views(dtype: str):
        stored = (
            np.zeros((1, 1, 128), dtype=np.uint16)
            if dtype == "bfloat16"
            else np.zeros((1, 1, 128), dtype=np.float32)
        )
        return _aliased_views(stored)

    q, q_raw = input_views(q_values, "bfloat16")
    k, k_raw = input_views(k_values, "bfloat16")
    v, v_raw = input_views(v_values, "bfloat16")
    gate, gate_raw = input_views(gate_values, "float32")
    do, do_raw = input_views(do_values, "bfloat16")
    beta, beta_raw = input_views(beta_values, "bfloat16")
    w, w_raw = input_views(w_values, "bfloat16")
    dq, dq_raw = output_views("bfloat16")
    dk, dk_raw = output_views("bfloat16")
    dv, dv_raw = output_views("bfloat16")
    dgate, dgate_raw = output_views("float32")
    dw, dw_raw = output_views("bfloat16")
    dbeta, dbeta_raw = output_views("bfloat16")
    checkpoint, checkpoint_raw = _aliased_views(np.zeros((1, 1, 128, 128), dtype=np.uint16))
    descriptor_workspace = np.zeros(240, dtype=np.int64)
    cu_seqlens = np.array([0, 1], dtype=np.int32)
    work_items = _one_chunk_work_item()
    work_count = np.array([1], dtype=np.int32)
    dummy_i32 = np.zeros(8, dtype=np.int32)

    descriptor_entries = (
        ("base_q", q, "bfloat16", 2, 64),
        ("base_k", k, "bfloat16", 2, 64),
        ("base_v", v, "bfloat16", 2, 64),
        ("base_gate", gate, "float32", 4, 32),
        ("base_do", do, "bfloat16", 2, 64),
        ("base_beta", beta, "bfloat16", 2, 64),
        ("base_w", w, "bfloat16", 2, 64),
        ("base_dq", dq, "bfloat16", 2, 64),
        ("base_dk", dk, "bfloat16", 2, 64),
        ("base_dv", dv, "bfloat16", 2, 64),
        ("base_dgate", dgate, "float32", 4, 32),
        ("base_dwo", dw, "bfloat16", 2, 64),
        ("base_db", dbeta, "bfloat16", 2, 64),
    )
    host_descriptors = _host_descriptors(
        tuple(
            (
                name,
                _headed_map(
                    binding,
                    dtype=dtype,
                    channels=128,
                    heads=1,
                    tokens=1,
                    element_bytes=element_bytes,
                    box_channels=box_channels,
                ),
            )
            for name, binding, dtype, element_bytes, box_channels in descriptor_entries
        )
        + (("base_checkpoint", _checkpoint_map(checkpoint, rows=1, heads=1)),)
    )
    return NumSimCase(
        kernel=gdn2_bprop_f16.get_kernel(**config),
        args={
            **{f"k0:{name}": descriptor for name, descriptor in host_descriptors},
            "k0:descriptor_workspace": descriptor_workspace,
            "k0:cu_seqlens": cu_seqlens,
            "k0:q": q_raw,
            "k0:k": k_raw,
            "k0:v": v_raw,
            "k0:gate": gate_raw,
            "k0:do": do_raw,
            "k0:beta": beta_raw,
            "k0:w": w_raw,
            "k0:dq": dq_raw,
            "k0:dk": dk_raw,
            "k0:dv": dv_raw,
            "k0:dgate": dgate_raw,
            "k0:dwo": dw_raw,
            "k0:db": dbeta_raw,
            "k0:checkpoints": checkpoint_raw,
            "k0:work_item_staging": dummy_i32,
            "k0:work_count": work_count,
            "k0:work_items": work_items,
            "k0:scheduler": dummy_i32,
            "k0:n_batch": 1,
            "k0:q_row_stride_bytes": 256,
            "k0:k_row_stride_bytes": 256,
            "k0:v_row_stride_bytes": 256,
            "k0:gate_row_stride_bytes": 512,
            "k0:do_row_stride_bytes": 256,
            "k0:beta_row_stride_bytes": 256,
            "k0:w_row_stride_bytes": 256,
            "k0:dq_row_stride_bytes": 256,
            "k0:dk_row_stride_bytes": 256,
            "k0:dv_row_stride_bytes": 256,
            "k0:dgate_row_stride_bytes": 512,
            "k0:dwo_row_stride_bytes": 256,
            "k0:db_row_stride_bytes": 256,
            "k0:checkpoint_row_stride_bytes": 32768,
            "k0:checkpoint_every_n": 16,
            "k1:descriptor_workspace": descriptor_workspace,
            "k1:n_desc": 1,
            "k1:a_log": np.zeros(1, dtype=np.float32),
            "k1:dt_bias": np.zeros(128, dtype=np.float32),
            "k1:cu_seqlens": cu_seqlens,
            "k1:dgate": dgate,
            "k1:d_initial_state": np.zeros(1, dtype=np.float32),
            "k1:d_final_state": np.zeros(1, dtype=np.float32),
            "k1:work_items": work_items,
            "k1:work_count": work_count,
            "k1:scheduler": dummy_i32,
            "k1:scale": np.float32(1.0),
        },
        outputs={
            "dq": "k0:dq",
            "dk": "k0:dk",
            "dv": "k0:dv",
            "dgate": "k1:dgate",
            "dw": "k0:dwo",
            "dbeta": "k0:db",
        },
        reference=lambda: {
            "dq": expected_dq.reshape(-1),
            "dk": expected_dk.reshape(-1),
            "dv": expected_dv.reshape(-1),
            "dgate": expected_dgate.reshape(-1),
            "dw": expected_dw.reshape(-1),
            "dbeta": expected_dbeta.reshape(-1),
        },
        comparisons={
            name: ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16")
            for name in ("dq", "dk", "dv", "dw", "dbeta")
        }
        | {"dgate": ComparisonSpec(rtol=0, atol=6e-4)},
    )


def prepare_cudnn_dsa_sparse_attention_backward_case() -> NumSimCase:
    """Exercise all four DSA phases against an analytic sparse-softmax oracle."""

    head_dim = head_dim_v = 512
    num_head = 64
    seqlen_q = 1
    seqlen_kv = max_topk = 64
    config = {
        "head_dim": head_dim,
        "num_head": num_head,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "max_topk": max_topk,
        "has_topk_length": True,
        "dtype": "bfloat16",
        "topk_mode": "full",
        "sink_mode": "normal",
        "seed": 1,
    }
    q_values = np.zeros((seqlen_q, num_head, head_dim), dtype=np.float32)
    kv_values = np.zeros((seqlen_kv, head_dim), dtype=np.float32)
    do_values = np.zeros((seqlen_q, num_head, head_dim_v), dtype=np.float32)
    q_values[0, 0, 0] = 1.0
    kv_values[0, 1] = 1.0
    do_values[0, 0, 1] = 1.0
    q_bits = _float32_to_bfloat16_bits(q_values)
    kv_bits = _float32_to_bfloat16_bits(kv_values)
    do_bits = _float32_to_bfloat16_bits(do_values)
    q_f32 = _bfloat16_bits_to_float32(q_bits)
    kv_f32 = _bfloat16_bits_to_float32(kv_bits)
    do_f32 = _bfloat16_bits_to_float32(do_bits)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    sink = np.zeros(num_head, dtype=np.float32)
    scores = np.einsum("qhd,kd->qhk", q_f32, kv_f32, dtype=np.float32) * scale
    lse = np.logaddexp.reduce(scores, axis=-1).astype(np.float32)
    lse_full = np.logaddexp(lse, sink.reshape(1, num_head)).astype(np.float32)
    probabilities = np.exp(scores - lse_full[..., None]).astype(np.float32)
    out_f32 = np.einsum("qhk,kd->qhd", probabilities, kv_f32[:, :head_dim_v], dtype=np.float32)
    out_bits = _float32_to_bfloat16_bits(out_f32)
    out_rounded = _bfloat16_bits_to_float32(out_bits)
    delta = np.sum(out_rounded * do_f32, axis=-1, dtype=np.float32)
    dp = np.einsum("qhd,kd->qhk", do_f32, kv_f32[:, :head_dim_v], dtype=np.float32)
    ds = probabilities * (dp - delta[..., None])
    expected_dq = np.einsum("qhk,kd->qhd", ds, kv_f32, dtype=np.float32) * scale
    expected_dk = np.einsum("qhk,qhd->kd", ds, q_f32, dtype=np.float32) * scale
    expected_dv = np.einsum("qhk,qhd->kd", probabilities, do_f32, dtype=np.float32)
    expected_dkv = expected_dk
    expected_dkv[:, :head_dim_v] += expected_dv
    p_sink = np.exp(sink.reshape(1, num_head) - lse_full).astype(np.float32)
    expected_d_sink = np.sum(-p_sink * delta, axis=0, dtype=np.float32)

    q = q_bits.reshape(-1)
    kv = kv_bits.reshape(-1)
    dout = do_bits.reshape(-1)
    out = out_bits.reshape(-1)
    dq = np.zeros(seqlen_q * num_head * head_dim, dtype=np.uint16)
    dkv = np.zeros(seqlen_kv * head_dim, dtype=np.uint16)
    plane = num_head * 8
    ws = np.zeros(2 * plane, dtype=np.float32)
    ws_dkv = np.zeros(seqlen_kv * head_dim, dtype=np.float32)
    q_map = _tensor_map(
        q,
        dtype="bfloat16",
        global_shape=(head_dim, num_head, seqlen_q),
        global_strides=(head_dim * 2, num_head * head_dim * 2),
        box_shape=(64, 64, 1),
    )
    do_map = _tensor_map(
        dout,
        dtype="bfloat16",
        global_shape=(head_dim_v, num_head, seqlen_q),
        global_strides=(head_dim_v * 2, num_head * head_dim_v * 2),
        box_shape=(64, 64, 1),
    )
    dq_map = _tensor_map(
        dq,
        dtype="bfloat16",
        global_shape=(head_dim, num_head, seqlen_q),
        global_strides=(head_dim * 2, num_head * head_dim * 2),
        box_shape=(64, 64, 1),
    )
    topk_idxs = np.arange(max_topk, dtype=np.int32)
    topk_length = np.array([max_topk], dtype=np.int32)
    launch_values = {
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "plane": plane,
        "sum_odo_scale": np.float32(-1.0),
        "lse_scale": np.float32(-math.log2(math.e)),
        "scale": scale,
    }
    kernels = tuple(
        kernel.specialize(
            {
                parameter: launch_values[parameter.name]
                for parameter in kernel.params
                if parameter.name in launch_values
            }
        )
        for kernel in dsa_sparse_attention_backward.get_kernel(**config)
    )
    return NumSimCase(
        kernel=kernels,
        args={
            "k0:out": out,
            "k0:dout": dout,
            "k0:lse": lse.reshape(-1),
            "k0:attn_sink": sink,
            "k0:ws": ws,
            "k1:desc_q": q_map,
            "k1:desc_do": do_map,
            "k1:desc_dq": dq_map,
            "k1:kv": kv,
            "k1:dq": dq,
            "k1:ws_dkv": ws_dkv,
            "k1:topk_idxs": topk_idxs,
            "k1:topk_length": topk_length,
            "k1:ws": ws,
            "k2:ws_dkv": ws_dkv,
            "k2:dkv": dkv,
            "k3:ws": ws,
            "k3:attn_sink": sink.copy(),
            "k3:d_sink": np.zeros(num_head, dtype=np.float32),
        },
        outputs={"dq": "k1:dq", "dkv": "k2:dkv", "d_sink": "k3:d_sink"},
        reference=lambda: {
            "dq": expected_dq.reshape(-1),
            "dkv": expected_dkv.reshape(-1),
            "d_sink": expected_d_sink.reshape(-1),
        },
        comparisons={
            "dq": ComparisonSpec(rtol=5e-2, atol=5e-3, actual_encoding="bfloat16"),
            "dkv": ComparisonSpec(rtol=5e-2, atol=5e-3, actual_encoding="bfloat16"),
            "d_sink": ComparisonSpec(rtol=5e-2, atol=5e-3),
        },
    )


def prepare_cudnn_bsa_forward_blk64_case() -> NumSimCase:
    """Exercise one complete sparse KV block and validate output plus LSE."""

    batch = heads = 1
    seqlen_q = seqlen_kv = 64
    head_dim = 128
    config = {
        "batch": batch,
        "num_q_heads": heads,
        "num_kv_heads": heads,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "kv_blocks": 1,
        "tensor_layout": "bhsd",
        "has_block_sizes": True,
        "block_count_mode": "fixed",
        "block_count_pattern": None,
        "use_clc": False,
        "kv_splits": 1,
        "softmax_scale": None,
        "use_int64_kv_strides": False,
        "seed": 7401,
    }
    rng = np.random.default_rng(7401)
    q = (rng.standard_normal((batch, heads, seqlen_q, head_dim)) * 0.125).astype(np.float32)
    k = (rng.standard_normal((batch, heads, seqlen_kv, head_dim)) * 0.125).astype(np.float32)
    v = (rng.standard_normal((batch, heads, seqlen_kv, head_dim)) * 0.125).astype(np.float32)
    q_bits = _float32_to_bfloat16_bits(q)
    k_bits = _float32_to_bfloat16_bits(k)
    v_bits = _float32_to_bfloat16_bits(v)
    q_f32 = _bfloat16_bits_to_float32(q_bits)
    k_f32 = _bfloat16_bits_to_float32(k_bits)
    v_f32 = _bfloat16_bits_to_float32(v_bits)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    score = np.matmul(q_f32, np.swapaxes(k_f32, -1, -2)) * scale
    row_max = np.max(score, axis=-1, keepdims=True)
    weights = np.exp(score - row_max).astype(np.float32)
    denominator = np.sum(weights, axis=-1, keepdims=True, dtype=np.float32)
    expected_out = np.matmul(weights / denominator, v_f32).astype(np.float32)
    expected_lse = (row_max[..., 0] + np.log(denominator[..., 0])).astype(np.float32)

    q_binding = q_bits.reshape(-1)
    k_binding = k_bits.reshape(-1)
    v_binding = v_bits.reshape(-1)
    out_binding = np.zeros(batch * heads * seqlen_q * head_dim, dtype=np.uint16)
    q_strides = (head_dim * 2, seqlen_q * head_dim * 2, heads * seqlen_q * head_dim * 2)
    kv_strides = (head_dim * 2, 64 * 2, 64 * head_dim * 2, seqlen_kv * head_dim * 2)

    return NumSimCase(
        kernel=bsa_forward_blk64.get_kernel(**config)[0],
        args={
            "q_map": _tensor_map(
                q_binding,
                dtype="bfloat16",
                global_shape=(head_dim, seqlen_q, heads, batch),
                global_strides=q_strides,
                box_shape=(64, 64, 1, 1),
            ),
            "k_map": _tensor_map(
                k_binding,
                dtype="bfloat16",
                global_shape=(64, 64, 2, 1, batch * heads),
                global_strides=kv_strides,
                box_shape=(64, 64, 1, 1, 1),
            ),
            "v_map": _tensor_map(
                v_binding,
                dtype="bfloat16",
                global_shape=(64, 64, 2, 1, batch * heads),
                global_strides=kv_strides,
                box_shape=(64, 64, 2, 1, 1),
            ),
            "o_map": _tensor_map(
                out_binding,
                dtype="bfloat16",
                global_shape=(head_dim, seqlen_q, heads, batch),
                global_strides=q_strides,
                box_shape=(64, 64, 1, 1),
            ),
            "lse": np.zeros(batch * heads * seqlen_q, dtype=np.float32),
            "block_index": np.array([0], dtype=np.int32),
            "block_sizes": np.array([64], dtype=np.int32),
            "block_nums": np.array([1], dtype=np.int32),
            "split_offsets": np.zeros(1, dtype=np.int32),
            "softmax_scale_log2": np.float32(scale * math.log2(math.e)),
        },
        outputs={"out": "o_map", "lse": "lse"},
        reference=lambda: {
            "out": expected_out.copy(),
            "lse": expected_lse.reshape(-1).copy(),
        },
        comparisons={
            "out": ComparisonSpec(rtol=3e-2, atol=3e-2, actual_encoding="bfloat16"),
            "lse": ComparisonSpec(rtol=2e-3, atol=2e-3),
        },
    )


def prepare_cudnn_bsa_forward_blk128_case() -> NumSimCase:
    """Exercise the blk128 sparse forward path with a nontrivial softmax."""

    batch = num_q_heads = num_kv_heads = 1
    seqlen_q = 1
    seqlen_kv = 256
    head_dim = 128
    config = {
        "batch": batch,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "kv_blocks": 2,
        "tensor_layout": "bhsd",
        "has_block_sizes": True,
        "block_count_mode": "fixed",
        "block_count_pattern": None,
        "pack_gqa": "auto",
        "return_lse": True,
        "softmax_scale": None,
        "use_int64_kv_strides": False,
        "seed": 12801,
    }
    q = np.zeros((batch, num_q_heads, seqlen_q, head_dim), dtype=np.float32)
    k = np.zeros((batch, num_kv_heads, seqlen_kv, head_dim), dtype=np.float32)
    v = np.zeros_like(k)
    q[0, 0, 0, 0] = np.float32(1.0)
    k[0, 0, :, 0] = np.linspace(-0.5, 0.5, seqlen_kv, dtype=np.float32)
    token = np.arange(seqlen_kv, dtype=np.float32)
    v[0, 0, :, 0] = np.float32(0.01) * token
    v[0, 0, :, 1] = np.float32(1.0) - np.float32(0.005) * token
    q_bits = _float32_to_bfloat16_bits(q)
    k_bits = _float32_to_bfloat16_bits(k)
    v_bits = _float32_to_bfloat16_bits(v)
    q_f32 = _bfloat16_bits_to_float32(q_bits)
    k_f32 = _bfloat16_bits_to_float32(k_bits)
    v_f32 = _bfloat16_bits_to_float32(v_bits)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    scores = np.matmul(q_f32[0, 0], k_f32[0, 0].T) * scale
    scores = scores.reshape(1, seqlen_kv)
    row_max = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores - row_max).astype(np.float32)
    denominator = np.sum(weights, axis=-1, keepdims=True, dtype=np.float32)
    expected_out = (weights / denominator) @ v_f32[0, 0]
    expected_lse = (row_max[..., 0] + np.log(denominator[..., 0])).astype(np.float32)

    return NumSimCase(
        kernel=bsa_forward_blk128.get_kernel(**config),
        args={
            "q": q_bits.reshape(-1),
            "k": k_bits.reshape(-1),
            "v": v_bits.reshape(-1),
            "out": np.zeros(batch * num_q_heads * seqlen_q * head_dim, dtype=np.uint16),
            "lse": np.zeros(batch * num_q_heads * seqlen_q, dtype=np.float32),
            "block_index": np.array([0, 1], dtype=np.int32),
            "block_sizes": np.array([128, 128], dtype=np.int32),
            "block_nums": np.zeros(1, dtype=np.int32),
            "block_sparse_num": np.int32(2),
            "softmax_scale_log2": np.float32(scale * math.log2(math.e)),
        },
        outputs=("out", "lse"),
        reference=lambda: {
            "out": expected_out.reshape(-1).copy(),
            "lse": expected_lse.reshape(-1).copy(),
        },
        comparisons={
            "out": ComparisonSpec(rtol=3e-2, atol=3e-2, actual_encoding="bfloat16"),
            "lse": ComparisonSpec(rtol=2e-3, atol=2e-3),
        },
    )


def prepare_cudnn_bsa_backward_blk64_case() -> NumSimCase:
    """Exercise all three blk64 backward phases on one sparse KV block."""

    batch = heads = 1
    seqlen_q = 1
    seqlen_kv = 64
    head_dim = 128
    q8 = 8
    k8 = 64
    scale = np.float32(1.0 / math.sqrt(head_dim))
    config = {
        "batch": batch,
        "num_heads": heads,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "kv_blocks": 1,
        "tensor_layout": "bhsd",
        "has_block_sizes": True,
        "block_count_mode": "fixed",
        "block_count_pattern": None,
        "softmax_scale": None,
        "use_int64_kv_strides": False,
        "bucket_size_blocks": None,
        "seed": 16401,
    }
    q = np.zeros((batch, heads, seqlen_q, head_dim), dtype=np.float32)
    k = np.zeros((batch, heads, seqlen_kv, head_dim), dtype=np.float32)
    v = np.zeros_like(k)
    do = np.zeros_like(q)
    q[0, 0, 0, 0] = np.float32(1.0)
    k[0, 0, :, 0] = np.linspace(-0.5, 0.5, seqlen_kv, dtype=np.float32)
    token = np.arange(seqlen_kv, dtype=np.float32)
    v[0, 0, :, 0] = np.float32(0.01) * token
    v[0, 0, :, 1] = np.float32(1.0) - np.float32(0.005) * token
    do[0, 0, 0, :4] = np.array([0.5, -0.25, 0.75, -1.0], dtype=np.float32)
    q_bits = _float32_to_bfloat16_bits(q)
    k_bits = _float32_to_bfloat16_bits(k)
    v_bits = _float32_to_bfloat16_bits(v)
    do_bits = _float32_to_bfloat16_bits(do)
    q_f32 = _bfloat16_bits_to_float32(q_bits)[0, 0]
    k_f32 = _bfloat16_bits_to_float32(k_bits)[0, 0]
    v_f32 = _bfloat16_bits_to_float32(v_bits)[0, 0]
    do_f32 = _bfloat16_bits_to_float32(do_bits)[0, 0]
    scores = (q_f32 @ k_f32.T) * scale
    row_max = np.max(scores)
    weights = np.exp(scores - row_max).astype(np.float32)
    probabilities = (weights / np.sum(weights, dtype=np.float32)).reshape(-1)
    output = probabilities @ v_f32
    lse = np.float32(row_max + np.log(np.sum(weights, dtype=np.float32)))
    delta = np.sum(output * do_f32[0], dtype=np.float32)
    dp = v_f32 @ do_f32[0]
    ds = probabilities * (dp - delta)
    expected_dq = (ds[:, None] * k_f32).sum(axis=0) * scale
    expected_dk = ds[:, None] * q_f32[None, :] * scale
    expected_dv = probabilities[:, None] * do_f32[0][None, :]

    workspace_sizes = {
        "sum_odo": batch * heads * q8,
        "scaled_lse": batch * heads * q8,
        "dq_acc": batch * heads * q8 * head_dim,
        "dk_acc": batch * heads * k8 * head_dim,
        "dv_acc": batch * heads * k8 * head_dim,
    }
    workspace_offsets = {}
    cursor = 0
    for name in ("sum_odo", "scaled_lse", "dq_acc", "dk_acc", "dv_acc"):
        workspace_offsets[name] = cursor
        cursor += workspace_sizes[name]
    workspace = np.zeros(cursor, dtype=np.float32)
    dq_acc = workspace[
        workspace_offsets["dq_acc"] : workspace_offsets["dq_acc"] + workspace_sizes["dq_acc"]
    ]
    q_strides = (head_dim * 2, seqlen_q * head_dim * 2, heads * seqlen_q * head_dim * 2)
    kv_strides = (head_dim * 2, seqlen_kv * head_dim * 2, heads * seqlen_kv * head_dim * 2)
    q_map = _tensor_map(
        q_bits.reshape(-1),
        dtype="bfloat16",
        global_shape=(head_dim, seqlen_q, heads, batch),
        global_strides=q_strides,
        box_shape=(64, 64, 1, 1),
    )
    k_map = _tensor_map(
        k_bits.reshape(-1),
        dtype="bfloat16",
        global_shape=(head_dim, seqlen_kv, heads, batch),
        global_strides=kv_strides,
        box_shape=(64, 64, 1, 1),
    )
    v_map = _tensor_map(
        v_bits.reshape(-1),
        dtype="bfloat16",
        global_shape=(head_dim, seqlen_kv, heads, batch),
        global_strides=kv_strides,
        box_shape=(64, 64, 1, 1),
    )
    do_map = _tensor_map(
        do_bits.reshape(-1),
        dtype="bfloat16",
        global_shape=(head_dim, seqlen_q, heads, batch),
        global_strides=q_strides,
        box_shape=(64, 64, 1, 1),
    )
    dq_map = _tensor_map(
        dq_acc,
        dtype="float32",
        global_shape=(head_dim, q8, heads, batch),
        global_strides=(head_dim * 4, q8 * head_dim * 4, heads * q8 * head_dim * 4),
        box_shape=(32, 64, 1, 1),
    )
    expected = {
        "dq": expected_dq.reshape(-1).copy(),
        "dk": expected_dk.reshape(-1).copy(),
        "dv": expected_dv.reshape(-1).copy(),
    }
    kernel = bsa_backward_blk64.get_kernel(**config)
    return NumSimCase(
        kernel=kernel,
        args={
            "k0:o": _float32_to_bfloat16_bits(output.reshape(1, 1, 1, head_dim)).reshape(-1),
            "k0:do": do_bits.reshape(-1),
            "k0:lse": np.array([lse], dtype=np.float32),
            "k0:workspace": workspace,
            "k1:q_map": q_map,
            "k1:k_map": k_map,
            "k1:v_map": v_map,
            "k1:do_map": do_map,
            "k1:dq_map": dq_map,
            "k1:bucketed_offsets": np.array([0, 1], dtype=np.int32),
            "k1:bucketed_indices": np.array([0], dtype=np.int32),
            "k1:block_sizes": np.array([64], dtype=np.int32),
            "k1:workspace": workspace,
            "k1:edge_stride": np.int64(1),
            "k1:softmax_scale": scale,
            "k2:workspace": workspace,
            "k2:dq": np.zeros(batch * heads * seqlen_q * head_dim, dtype=np.uint16),
            "k2:dk": np.zeros(batch * heads * seqlen_kv * head_dim, dtype=np.uint16),
            "k2:dv": np.zeros(batch * heads * seqlen_kv * head_dim, dtype=np.uint16),
            "k2:softmax_scale": scale,
        },
        outputs={"dq": "k2:dq", "dk": "k2:dk", "dv": "k2:dv"},
        reference=lambda: {name: value.copy() for name, value in expected.items()},
        comparisons={
            name: ComparisonSpec(rtol=5e-2, atol=5e-3, actual_encoding="bfloat16")
            for name in expected
        },
    )


def prepare_cudnn_bsa_backward_blk128_case() -> NumSimCase:
    """Exercise the direct-dK/dV three-phase blk128 backward path."""

    batch = heads = seqlen_q = 1
    seqlen_kv = 128
    head_dim = 64
    q128 = k128 = 128
    scale = np.float32(1.0 / math.sqrt(head_dim))
    config = {
        "batch": batch,
        "num_heads": heads,
        "seqlen_q": seqlen_q,
        "seqlen_kv": seqlen_kv,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "kv_blocks": 1,
        "tensor_layout": "bhsd",
        "block_count_mode": "fixed",
        "block_count_pattern": None,
        "softmax_scale": None,
        "use_int64_kv_strides": False,
        "bucket_size_blocks": None,
        "preallocate_outputs": False,
        "data_mode": "random",
        "seed": 128018,
    }
    q = np.zeros((batch, heads, seqlen_q, head_dim), dtype=np.float32)
    k = np.zeros((batch, heads, seqlen_kv, head_dim), dtype=np.float32)
    v = np.zeros_like(k)
    do = np.zeros_like(q)
    q[0, 0, 0, 0] = np.float32(1.0)
    k[0, 0, :, 0] = np.linspace(-0.5, 0.5, seqlen_kv, dtype=np.float32)
    token = np.arange(seqlen_kv, dtype=np.float32)
    v[0, 0, :, 0] = np.float32(0.01) * token
    v[0, 0, :, 1] = np.float32(1.0) - np.float32(0.005) * token
    do[0, 0, 0, :4] = np.array([0.5, -0.25, 0.75, -1.0], dtype=np.float32)
    q_bits = _float32_to_bfloat16_bits(q)
    k_bits = _float32_to_bfloat16_bits(k)
    v_bits = _float32_to_bfloat16_bits(v)
    do_bits = _float32_to_bfloat16_bits(do)
    q_f32 = _bfloat16_bits_to_float32(q_bits)[0, 0, 0]
    k_f32 = _bfloat16_bits_to_float32(k_bits)[0, 0]
    v_f32 = _bfloat16_bits_to_float32(v_bits)[0, 0]
    do_f32 = _bfloat16_bits_to_float32(do_bits)[0, 0, 0]
    scores = (k_f32 @ q_f32) * scale
    row_max = np.max(scores)
    weights = np.exp(scores - row_max).astype(np.float32)
    probabilities = weights / np.sum(weights, dtype=np.float32)
    output_f32 = probabilities @ v_f32
    output_bits = _float32_to_bfloat16_bits(output_f32)
    rounded_output = _bfloat16_bits_to_float32(output_bits)
    lse = np.float32(row_max + np.log(np.sum(weights, dtype=np.float32)))
    delta = np.sum(rounded_output * do_f32, dtype=np.float32)
    dp = v_f32 @ do_f32
    ds = probabilities * (dp - delta)
    expected_dq = np.sum(ds[:, None] * k_f32, axis=0, dtype=np.float32) * scale
    expected_dk = ds[:, None] * q_f32[None, :] * scale
    expected_dv = probabilities[:, None] * do_f32[None, :]

    sum_plane = batch * heads * q128
    dq_elements = batch * heads * q128 * head_dim
    dk_elements = batch * heads * k128 * head_dim
    dv_elements = dk_elements
    workspace = np.zeros(2 * sum_plane + dq_elements + dk_elements + dv_elements, dtype=np.float32)
    q_strides = (
        head_dim * 2,
        seqlen_q * head_dim * 2,
        heads * seqlen_q * head_dim * 2,
    )
    kv_strides = (
        head_dim * 2,
        seqlen_kv * head_dim * 2,
        heads * seqlen_kv * head_dim * 2,
    )

    def input_map(base: np.ndarray, *, tokens: int, strides: tuple[int, ...]) -> np.ndarray:
        return _tensor_map(
            base,
            dtype="bfloat16",
            global_shape=(head_dim, tokens, heads, batch),
            global_strides=strides,
            box_shape=(64, 128, 1, 1),
        )

    dk_output = np.zeros(batch * heads * seqlen_kv * head_dim, dtype=np.uint16)
    dv_output = np.zeros_like(dk_output)

    def output_map(base: np.ndarray) -> np.ndarray:
        return _tensor_map(
            base,
            dtype="bfloat16",
            global_shape=(head_dim, seqlen_kv, heads, batch),
            global_strides=kv_strides,
            box_shape=(32, 128, 1, 1),
            swizzle="64B",
        )

    return NumSimCase(
        kernel=bsa_backward_blk128.get_kernel(**config),
        args={
            "k0:o": output_bits.reshape(-1),
            "k0:do": do_bits.reshape(-1),
            "k0:lse": np.array([lse], dtype=np.float32),
            "k0:workspace": workspace,
            "k1:q_map": input_map(q_bits.reshape(-1), tokens=seqlen_q, strides=q_strides),
            "k1:k_map": input_map(k_bits.reshape(-1), tokens=seqlen_kv, strides=kv_strides),
            "k1:v_map": input_map(v_bits.reshape(-1), tokens=seqlen_kv, strides=kv_strides),
            "k1:do_map": input_map(do_bits.reshape(-1), tokens=seqlen_q, strides=q_strides),
            "k1:dv_map": output_map(dv_output),
            "k1:dk_map": output_map(dk_output),
            "k1:dk_output": dk_output,
            "k1:dv_output": dv_output,
            "k1:bucketed_offsets": np.array([0, 1], dtype=np.int32),
            "k1:bucketed_indices": np.array([0], dtype=np.int32),
            "k1:workspace": workspace,
            "k1:edge_stride": np.int64(1),
            "k1:softmax_scale": scale,
            "k2:workspace": workspace,
            "k2:output": np.zeros(batch * heads * seqlen_q * head_dim, dtype=np.uint16),
            "k2:output_scale": scale,
        },
        outputs={"dq": "k2:output", "dk": "k1:dk_output", "dv": "k1:dv_output"},
        reference=lambda: {
            "dq": expected_dq.reshape(-1).copy(),
            "dk": expected_dk.reshape(-1).copy(),
            "dv": expected_dv.reshape(-1).copy(),
        },
        comparisons={
            name: ComparisonSpec(rtol=5e-2, atol=5e-3, actual_encoding="bfloat16")
            for name in ("dq", "dk", "dv")
        },
    )


def prepare_cudnn_bsa_forward_combine_blk64_case() -> NumSimCase:
    """Exercise split-LSE weighting and BF16 output conversion."""

    batch = num_heads = seqlen_q = 1
    num_splits = 2
    first = (np.arange(128, dtype=np.float32) % np.float32(7.0)).reshape(1, 1, 1, 1, 128)
    second = first + np.float32(2.0)
    partial = np.concatenate((first, second), axis=1)
    lse_partial = np.array([0.0, math.log(3.0)], dtype=np.float32)
    expected = first.reshape(-1) + np.float32(1.5)
    div_mul, div_s1, div_s2 = bsa_forward_combine_blk64._fast_divmod(seqlen_q)
    launch_values = {
        "batch": batch,
        "num_heads": num_heads,
        "seqlen_q": seqlen_q,
        "num_splits": num_splits,
        "seqlen_div_mul": div_mul,
        "seqlen_div_s1": div_s1,
        "seqlen_div_s2": div_s2,
    }
    kernel = bsa_forward_combine_blk64.get_kernel(
        batch=batch,
        num_heads=num_heads,
        seqlen_q=seqlen_q,
        kv_splits=num_splits,
    )
    kernel = kernel.specialize(
        {
            parameter: launch_values[parameter.name]
            for parameter in kernel.params
            if parameter.name in launch_values
        }
    )

    return NumSimCase(
        kernel=kernel,
        args={
            "o_partial": partial.reshape(-1),
            "lse_partial": lse_partial,
            "out": np.zeros(128, dtype=np.uint16),
            "lse": np.zeros(1, dtype=np.float32),
        },
        outputs=("out", "lse"),
        reference=lambda: {
            "out": expected.copy(),
            "lse": np.array([math.log(4.0)], dtype=np.float32),
        },
        comparisons={
            "out": ComparisonSpec(rtol=0, atol=0, actual_encoding="bfloat16"),
            "lse": ComparisonSpec(rtol=1e-6, atol=1e-6),
        },
    )


def prepare_cudnn_csa_compressor_fwd_case() -> NumSimCase:
    """Exercise one full four-token compressor window with odd head width."""

    head_dim = 65
    coff = 1
    ratio = 4
    columns = np.arange(head_dim, dtype=np.float32)
    tokens = np.arange(ratio, dtype=np.float32)[:, None]
    kv = np.float32(0.125) * tokens + np.float32(0.01) * columns[None, :]
    score = np.float32(0.2) * tokens - np.float32(0.005) * columns[None, :]
    ape = np.float32(0.05) * (tokens - np.float32(1.5))
    ape = np.broadcast_to(ape, (ratio, head_dim)).copy()
    kv_bits = _float32_to_bfloat16_bits(kv)
    score_bits = _float32_to_bfloat16_bits(score)
    kv_f32 = _bfloat16_bits_to_float32(kv_bits)
    score_f32 = _bfloat16_bits_to_float32(score_bits)
    logits = score_f32 + ape
    weights = np.exp(logits - np.max(logits, axis=0, keepdims=True)).astype(np.float32)
    weights /= np.sum(weights, axis=0, keepdims=True, dtype=np.float32)
    expected = np.sum(kv_f32 * weights, axis=0, dtype=np.float32)

    kernel = csa_compressor_fwd.get_kernel(head_dim=head_dim, coff=coff)
    kernel = kernel.specialize(
        {param: 1 for param in kernel.params if param.name in {"nb_total", "n_seq"}}
    )

    return NumSimCase(
        kernel=kernel,
        args={
            "kv": kv_bits.reshape(-1),
            "score": score_bits.reshape(-1),
            "ape": ape.reshape(-1),
            "cu_seqlens": np.array([0, ratio], dtype=np.int32),
            "cu_seqlens_comp": np.array([0, 1], dtype=np.int32),
            "out": np.zeros(head_dim, dtype=np.uint16),
        },
        outputs=("out",),
        reference=lambda: {"out": expected.copy()},
        comparisons={"out": ComparisonSpec(rtol=2e-2, atol=2e-2, actual_encoding="bfloat16")},
    )


__all__ = [
    "prepare_cudnn_bsa_backward_blk128_case",
    "prepare_cudnn_bsa_backward_blk64_case",
    "prepare_cudnn_bsa_forward_combine_blk64_case",
    "prepare_cudnn_bsa_forward_blk128_case",
    "prepare_cudnn_bsa_forward_blk64_case",
    "prepare_cudnn_csa_compressor_fwd_case",
    "prepare_cudnn_dense_blockscaled_dsrelu_quant_case",
    "prepare_cudnn_dense_blockscaled_srelu_quant_case",
    "prepare_cudnn_dense_blockscaled_swiglu_quant_case",
    "prepare_cudnn_dense_gemm_persistent_swiglu_case",
    "prepare_cudnn_dsa_sparse_attention_backward_case",
    "prepare_cudnn_gdn_prefill_case",
    "prepare_cudnn_gdn_bprop_case",
    "prepare_cudnn_gdn_recompute_case",
    "prepare_cudnn_gdn2_bprop_case",
    "prepare_cudnn_gdn2_prefill_case",
    "prepare_cudnn_gdn2_recompute_case",
    "prepare_cudnn_gemm_proj_rope_mxfp8_bf16in_case",
    "prepare_cudnn_gemm_proj_rope_mxfp8_mxfp8in_case",
    "prepare_cudnn_kda_bprop_case",
    "prepare_cudnn_moe_dglu_dbias_case",
]
