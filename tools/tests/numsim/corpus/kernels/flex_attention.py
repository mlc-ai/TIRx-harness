"""Small, CPU-owned full-block cases for the canonical FlexAttention kernels."""

from __future__ import annotations

import math

import numpy as np

from tests.numsim.corpus.kernels.cudnn import (
    _bfloat16_bits_to_float32,
    _float32_to_bfloat16_bits,
    _tensor_map,
)
from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase


def _attention_inputs(dimension, kv_rows):
    rng = np.random.default_rng(20260906)
    bits = tuple(
        _float32_to_bfloat16_bits(rng.standard_normal((rows, dimension)) * 0.25)
        for rows in (128, kv_rows, kv_rows)
    )
    q, k, v = (_bfloat16_bits_to_float32(value).astype(np.float64) for value in bits)
    scores = q @ k.T / math.sqrt(dimension)
    maximum = scores.max(axis=1, keepdims=True)
    weights = np.exp(scores - maximum)
    denominator = weights.sum(axis=1, keepdims=True)
    probabilities = weights / denominator
    lse = (maximum + np.log(denominator)).reshape(-1)
    return bits, (q, k, v), probabilities, lse


def _prepare_forward(*, hd256):
    dimension, kv_rows = (256, 128) if hd256 else (128, 256)
    name = (
        "cudnn_sm100_flex_attention_forward_hd256"
        if hd256
        else "cudnn_sm103_flex_attention_forward"
    )
    module = load_tirx_kernel(name)
    if hd256:
        config = module._config("canonical", mask="full", seqlen=128, num_q_heads=1, num_kv_heads=1)
    else:
        config = module._config("canonical", num_q_heads=1, num_kv_heads=1)
    config.pop("label")
    bits, (_, _, v), probabilities, lse = _attention_inputs(dimension, kv_rows)
    blocks = kv_rows // 128
    args = {name: value.reshape(-1) for name, value in zip(("q", "k", "v"), bits)}
    args.update(
        out=np.full(128 * dimension, 0x7FC0, dtype=np.uint16),
        lse=np.full(128, np.nan, dtype=np.float32),
        sequence_desc=np.array([0, 0, 128, kv_rows, 0, 0, 0, 0], dtype=np.int32),
        fwd_work_desc=np.array([0, 0, 0, 128], dtype=np.int32),
        mask_payload=np.zeros(blocks * 128 * 4, dtype=np.uint32),
        softmax_scale_log2=np.float32(math.log2(math.e) / math.sqrt(dimension)),
    )
    if hd256:
        args.update(
            cu_q=np.array([0, 128], dtype=np.int32),
            cu_k=np.array([0, kv_rows], dtype=np.int32),
            softmax_scale=np.float32(1 / math.sqrt(dimension)),
        )
    for prefix, count in (("partial", 0), ("full", blocks)):
        names = (
            ("mask_block_cnt", "mask_block_offset", "mask_block_idx")
            if hd256 and prefix == "partial"
            else ("full_block_cnt", "full_block_offset", "full_block_idx")
            if hd256
            else (f"{prefix}_count", f"{prefix}_offset", f"{prefix}_index")
        )
        args[names[0]] = np.array([count], dtype=np.int32)
        args[names[1]] = np.array([0, count], dtype=np.int32)
        args[names[2]] = np.arange(blocks, dtype=np.int32)
    return NumSimCase(
        kernel=module.get_kernel(**config),
        args=args,
        outputs=("out", "lse"),
        reference=lambda: {"out": (probabilities @ v).reshape(-1), "lse": lse},
        comparisons={
            "out": ComparisonSpec(rtol=8e-3, atol=4e-3, actual_encoding="bfloat16"),
            "lse": ComparisonSpec(rtol=2e-6, atol=2e-6),
        },
    )


def prepare_flex_attention_forward_sm103_case():
    return _prepare_forward(hd256=False)


def prepare_flex_attention_forward_hd256_case():
    return _prepare_forward(hd256=True)


def prepare_flex_attention_backward_case():
    """One D64 main-kernel tile, with independently prepared forward statistics."""
    dimension = 64
    rows = 128
    module = load_tirx_kernel("cudnn_sm100_flex_attention_backward")
    config = module._config(
        "canonical",
        seqlen_q=rows,
        seqlen_kv=rows,
        num_q_heads=1,
        num_kv_heads=1,
        head_dim=dimension,
        head_dim_v=dimension,
        mask_type="full",
    )
    config.pop("label")
    bits, (q, k, v), probabilities, lse = _attention_inputs(dimension, rows)
    dout_bits = _float32_to_bfloat16_bits(
        np.random.default_rng(42).standard_normal((rows, dimension))
    )
    dout = _bfloat16_bits_to_float32(dout_bits).astype(np.float64)
    dp = dout @ v.T
    dpsum = (probabilities * dp).sum(axis=1)
    ds = probabilities * (dp - dpsum[:, None])
    dq_base = 2 * rows
    workspace = np.zeros(dq_base + 3 * rows * dimension, dtype=np.float32)
    workspace[:rows] = dpsum
    workspace[rows:dq_base] = lse * math.log2(math.e)
    expected_workspace = workspace.copy()
    # The public main-kernel ABI leaves dQ unscaled; its host postprocess
    # applies softmax_scale and unpacks four-column-major accumulator groups.
    # dK and dV are written directly in BF16.
    expected_workspace[dq_base : dq_base + rows * dimension] = (
        (ds @ k).reshape(rows, dimension // 4, 4).transpose(1, 0, 2).reshape(-1)
    )
    dk = np.full(rows * dimension, 0x7FC0, dtype=np.uint16)
    dv = np.full(rows * dimension, 0x7FC0, dtype=np.uint16)
    args = {
        "dk_output": dk,
        "dv_output": dv,
        "workspace": workspace,
        "softmax_scale": np.float32(1 / math.sqrt(dimension)),
        "partial_count": np.zeros(1, dtype=np.int32),
        "partial_offset": np.zeros(2, dtype=np.int32),
        "full_count": np.ones(1, dtype=np.int32),
        "full_offset": np.array([0, 1], dtype=np.int32),
        "packed_mask": np.zeros(512, dtype=np.uint32),
        "sequence_desc": np.array([0, 0, rows, rows, 0, 0, 0, 0], dtype=np.int32),
        "work_desc": np.array([0, 0, 0, rows], dtype=np.int32),
    }
    for name in (
        "partial_index",
        "full_index",
        "dq_write_order",
        "dq_write_order_full",
        "dq_semaphore",
        "dk_semaphore",
        "dv_semaphore",
    ):
        args[name] = np.zeros(1, dtype=np.int32)
    for name in ("cu_q", "cu_k"):
        args[name] = np.array([0, rows], dtype=np.int32)
    args["cu_k_blocks"] = np.array([0, 1], dtype=np.int32)
    for name, value in zip(("q", "k", "v", "do", "dk", "dv"), (*bits, dout_bits, dk, dv)):
        output = name in ("dk", "dv")
        args[f"{name}_map"] = _tensor_map(
            value.reshape(-1),
            dtype="bfloat16",
            global_shape=(dimension, rows, 1, 1),
            global_strides=(dimension * 2, dimension * 2, rows * dimension * 2),
            box_shape=(32 if output else 64, 128, 1, 1),
            swizzle="64B" if output else "128B",
        )
    return NumSimCase(
        kernel=module.get_kernel(**config),
        args=args,
        outputs=("workspace", "dk_output", "dv_output"),
        reference=lambda: {
            "workspace": expected_workspace,
            "dk_output": (ds.T @ q / math.sqrt(dimension)).reshape(-1),
            "dv_output": (probabilities.T @ dout).reshape(-1),
        },
        comparisons={
            "workspace": ComparisonSpec(rtol=3e-2, atol=2e-3),
            "dk_output": ComparisonSpec(rtol=3e-2, atol=2e-3, actual_encoding="bfloat16"),
            "dv_output": ComparisonSpec(rtol=3e-2, atol=2e-3, actual_encoding="bfloat16"),
        },
    )
