"""Small CPU-owned MSA scheduler/attention cases.

The two preparation stages are independently checked here.  The forward stage
uses the same one-edge work item and is intentionally kept in the manifest so
that any unsupported shared TMA operation is reported by the corpus rather
than silently omitted.
"""

from __future__ import annotations

import numpy as np

from tests.numsim.support._tirx_kernels import load_tirx_kernel
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase


def _specialize(kernel, values: dict[str, int | float]):
    return kernel.specialize(
        {param: values[param.name] for param in kernel.params if param.name in values}
    )


def prepare_msa_sparse_prepare_flat_schedule_case() -> NumSimCase:
    module = load_tirx_kernel("msa_sparse_prepare_flat_schedule_sm100")
    kernel = _specialize(
        module.get_kernel(),
        {
            "total_rows": 1,
            "num_batches": 1,
            "target": 1,
            "work_capacity": 1,
            "num_heads_kv": 1,
            "blk_kv": 128,
        },
    )
    row_ptr = np.array([0, 1], dtype=np.int32)
    cu_seqlens_k = np.array([0, 128], dtype=np.int32)
    metadata = np.full((6,), -9, dtype=np.int32)
    work_count = np.zeros((1,), dtype=np.int32)
    return NumSimCase(
        kernel=kernel,
        args={
            "k2q_row_ptr": row_ptr,
            "cu_seqlens_k": cu_seqlens_k,
            "scheduler_metadata": metadata,
            "work_count": work_count,
        },
        outputs=("scheduler_metadata", "work_count"),
        reference=lambda: {
            "scheduler_metadata": np.array([0, 0, 0, 1, 0, 0], dtype=np.int32),
            "work_count": np.array([1], dtype=np.int32),
        },
        comparisons={
            "scheduler_metadata": ComparisonSpec(rtol=0, atol=0),
            "work_count": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_msa_sparse_prepare_fwd_split_atomic_case() -> NumSimCase:
    module = load_tirx_kernel("msa_sparse_prepare_fwd_split_atomic_sm100")
    kernel = _specialize(
        module.get_kernel(),
        {
            "total_rows": 1,
            "num_batches": 1,
            "work_capacity": 1,
            "nnz_capacity": 1,
            "total_q": 1,
            "num_heads_kv": 1,
            "max_seqlen_q": 1,
            "topk": 1,
        },
    )
    row_ptr = np.array([0, 1], dtype=np.int32)
    q_indices = np.array([0], dtype=np.int32)
    metadata = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)
    work_count = np.array([1], dtype=np.int32)
    qsplit_indices = np.array([-9], dtype=np.int32)
    split_counts = np.zeros((1,), dtype=np.int32)
    cu_seqlens_q = np.array([0, 1], dtype=np.int32)
    return NumSimCase(
        kernel=kernel,
        args={
            "k2q_row_ptr": row_ptr,
            "k2q_q_indices": q_indices,
            "scheduler_metadata": metadata,
            "work_count": work_count,
            "k2q_qsplit_indices": qsplit_indices,
            "split_counts": split_counts,
            "cu_seqlens_q": cu_seqlens_q,
        },
        outputs=("k2q_qsplit_indices", "split_counts"),
        reference=lambda: {
            "k2q_qsplit_indices": np.array([0], dtype=np.int32),
            "split_counts": np.array([1], dtype=np.int32),
        },
        comparisons={
            "k2q_qsplit_indices": ComparisonSpec(rtol=0, atol=0),
            "split_counts": ComparisonSpec(rtol=0, atol=0),
        },
    )


def prepare_msa_sparse_atten_fwd_case() -> NumSimCase:
    module = load_tirx_kernel("msa_sparse_atten_fwd_sm100")
    values = {
        "num_kv_blocks": 1,
        "num_heads_kv": 1,
        "seq_len_q": 1,
        "work_capacity": 1,
        "total_k": 128,
        "total_q": 1,
        "head_q": 16,
        "nnz": 1,
        "total_rows": 1,
        "num_batches": 1,
        "topk": 1,
        "qhead_per_kv": 1,
    }
    kernel = _specialize(
        module.get_kernel(
            dtype="bf16",
            partial_dtype="float32",
            temperature=None,
            causal=False,
            batch=1,
            seqlen_q=1,
            seqlen_k=128,
            head_kv=1,
            qhead_per_kv=16,
            topk=1,
            blk_kv=128,
        ),
        values,
    )
    # Make one KV token win the score, then give that token a nonzero value in
    # every output dimension.  With ``softmax_scale_log2=1`` the score vector
    # is [1, 0, ..., 0], so the independent oracle is exactly 2/129 rather than
    # a near-zero 1/128 probe that a broken kernel could pass with loose atol.
    k_bits = np.zeros((128, 1, 128), dtype=np.uint16)
    v_bits = np.zeros_like(k_bits)
    q_bits = np.zeros((16, 128), dtype=np.uint16)
    one_bf16 = np.array([1.0], dtype=np.float32).view(np.uint32)[0] >> 16
    k_bits[0, 0, 0] = one_bf16
    v_bits[0, 0, :] = one_bf16
    q_bits[:, 0] = one_bf16
    q_indices = np.array([0], dtype=np.int32)
    qsplit_indices = np.array([0], dtype=np.int32)
    row_ptr = np.array([0, 1], dtype=np.int32)
    metadata = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)
    work_count = np.array([1], dtype=np.int32)
    o_partial = np.full((16 * 128,), np.nan, dtype=np.float32)
    lse_partial = np.full((16,), np.nan, dtype=np.float32)
    cu_q = np.array([0, 1], dtype=np.int32)
    cu_k = np.array([0, 128], dtype=np.int32)
    expected_o = np.full((16 * 128,), np.float32(2.0 / 129.0), dtype=np.float32)
    expected = {
        "o_partial": expected_o,
        "lse_partial": np.full((16,), np.log(129.0), dtype=np.float32),
    }
    return NumSimCase(
        kernel=kernel,
        args={
            "k": k_bits,
            "v": v_bits,
            "k2q_q_indices": q_indices,
            "k2q_qsplit_indices": qsplit_indices,
            "k2q_row_ptr": row_ptr,
            "scheduler_metadata": metadata,
            "work_count": work_count,
            "o_partial": o_partial,
            "lse_partial": lse_partial,
            "q_flat": q_bits,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "softmax_scale_log2": np.float32(1.0),
            "lse_temperature_scale_log2": np.float32(0.0),
            "lse_temperature_inv_scale": np.float32(1.0),
        },
        outputs=("o_partial", "lse_partial"),
        reference=lambda: {key: value.copy() for key, value in expected.items()},
        comparisons={key: ComparisonSpec(rtol=1e-4, atol=1e-4) for key in expected},
    )


def prepare_msa_sparse_atten_fwd_nvfp4_kv_case() -> NumSimCase:
    """Cover the NVFP4 K/V dequantization and split-partial attention path."""

    module = load_tirx_kernel("msa_sparse_atten_fwd_nvfp4_kv_sm100")
    config = {
        "dtype": "bf16q",
        "partial_dtype": "float32",
        "temperature": None,
        "causal": False,
        "batch": 1,
        "seqlen_q": 1,
        "seqlen_k": 128,
        "head_kv": 1,
        "qhead_per_kv": 1,
        "topk": 1,
        "blk_kv": 128,
        "seqlen_pattern": "uniform",
    }
    values = {
        "num_kv_blocks": 1,
        "num_heads_kv": 1,
        "seq_len_q": 1,
        "work_capacity": 1,
        "total_k": 128,
        "total_q": 1,
        "head_q": 1,
        "nnz": 1,
        "total_rows": 1,
        "num_batches": 1,
        "topk": 1,
        "scale_numel": 4096,
        "softmax_scale_log2": 1.0,
        "lse_temperature_scale_log2": 0.0,
        "lse_temperature_inv_scale": 1.0,
    }
    kernel = _specialize(module.get_kernel(**config), values)

    # NVFP4 stores two E2M1 values per byte.  Make token zero the sole scored
    # token: Q[0:2] and both packed K values are +1, while all other K values
    # are zero.  Fill that token's V row with +6 (0x7 in both nibbles).  With
    # log2 scale one, the independent attention oracle is V * 4/131, and every
    # scale byte remains the E4M3 encoding of 1.0 (0x38).
    k = np.zeros((128, 1, 64), dtype=np.uint8)
    v = np.zeros_like(k)
    k[0, 0, 0] = 0x22  # both nibbles = +1.0
    v[0, 0, :] = 0x77  # both nibbles = +6.0
    k_scale = np.full(4096, 0x38, dtype=np.uint8)
    v_scale = np.full(4096, 0x38, dtype=np.uint8)
    q = np.zeros((1, 128), dtype=np.uint16)
    q[0, :2] = np.uint16(0x3F80)  # BF16 +1.0 in both packed dimensions

    q_indices = np.array([0], dtype=np.int32)
    qsplit_indices = np.array([0], dtype=np.int32)
    row_ptr = np.array([0, 1], dtype=np.int32)
    metadata = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)
    work_count = np.array([1], dtype=np.int32)
    cu_q = np.array([0, 1], dtype=np.int32)
    cu_k = np.array([0, 128], dtype=np.int32)

    expected_o = np.full(128, np.float32(24.0 / 131.0), dtype=np.float32)
    expected = {
        "o_partial": expected_o,
        "lse_partial": np.array([np.log(131.0)], dtype=np.float32),
    }
    return NumSimCase(
        kernel=kernel,
        args={
            "k": k,
            "v": v,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "k_global_scale": np.array([1.0], dtype=np.float32),
            "k2q_q_indices": q_indices,
            "k2q_qsplit_indices": qsplit_indices,
            "k2q_row_ptr": row_ptr,
            "scheduler_metadata": metadata,
            "work_count": work_count,
            "o_partial": np.full(128, np.nan, dtype=np.float32),
            "lse_partial": np.full(1, np.nan, dtype=np.float32),
            "q_flat": q,
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
        },
        outputs=("o_partial", "lse_partial"),
        reference=lambda: {name: value.copy() for name, value in expected.items()},
        comparisons={name: ComparisonSpec(rtol=1e-4, atol=1e-4) for name in expected},
    )


def prepare_msa_sparse_atten_fwd_combine_case() -> NumSimCase:
    """Reduce three live fp32 partials and invert K1's fake-column order."""

    topk = 4
    total_q = head_q = head_kv = qhead_per_kv = num_batches = 1
    live_splits = 3
    module = load_tirx_kernel("msa_sparse_atten_fwd_combine_sm100")
    kernel = _specialize(
        module.get_kernel(
            topk=topk,
            partial_dtype="float32",
            temperature=False,
            output_scale=False,
            seqused=False,
        ),
        {"total_q": total_q, "head_q": head_q, "num_batches": num_batches},
    )

    lse_live = np.array([0.25, -0.5, 1.0], dtype=np.float32)
    shifted = lse_live.astype(np.float64) - np.max(lse_live.astype(np.float64))
    exponentials = np.exp(shifted)
    weights = (exponentials / exponentials.sum()).astype(np.float32)
    expected_lse = np.array(
        [np.log(exponentials.sum()) + np.max(lse_live.astype(np.float64))],
        dtype=np.float32,
    )

    columns = np.arange(128, dtype=np.float32)
    real_partials = np.stack(
        (
            np.float32(0.5) + columns / np.float32(128),
            np.float32(1.5) - columns / np.float32(256),
            np.float32(-0.75) + columns / np.float32(64),
        )
    )
    fake_columns = np.arange(128, dtype=np.int32)
    nt = fake_columns & -16
    inner = fake_columns & 15
    lane = inner // 4
    slot = inner & 3
    real_columns = nt + (slot >> 1) * 8 + lane * 2 + (slot & 1)
    o_partial = np.full((topk, total_q, head_q, 128), np.nan, dtype=np.float32)
    for split in range(live_splits):
        o_partial[split, 0, 0, :] = real_partials[split, real_columns]
    lse_partial = np.full((topk, total_q, head_q), np.nan, dtype=np.float32)
    lse_partial[:live_splits, 0, 0] = lse_live
    expected_o = np.sum(weights[:, None] * real_partials, axis=0, dtype=np.float32)

    return NumSimCase(
        kernel=kernel,
        args={
            "o_partial": o_partial.reshape(-1),
            "lse_partial": lse_partial.reshape(-1),
            "o_out": np.zeros(128, dtype=np.uint16),
            "lse_out": np.full(1, np.nan, dtype=np.float32),
            "lse_temperature_partial": np.zeros(1, dtype=np.float32),
            "lse_temperature_out": np.zeros(1, dtype=np.float32),
            "cu_seqlens": np.array([0, 1], dtype=np.int32),
            "seqused_q": np.zeros(1, dtype=np.int32),
            "split_counts": np.array([live_splits], dtype=np.int32),
            "output_scale_ptr": np.ones(1, dtype=np.float32),
            "stride_op_split": np.int32(128),
            "stride_op_q": np.int32(128),
            "stride_op_h": np.int32(128),
            "stride_lp_split": np.int32(1),
            "stride_lp_q": np.int32(1),
            "stride_o_q": np.int32(128),
            "stride_o_h": np.int32(128),
            "stride_l_q": np.int32(1),
            "stride_sc_q": np.int32(1),
            "qhead_per_kv": np.int32(qhead_per_kv),
            "head_div_mul": np.int32(1),
            "head_div_s1": np.int32(0),
            "head_div_s2": np.int32(0),
        },
        outputs=("o_out", "lse_out"),
        reference=lambda: {
            "o_out": expected_o.copy(),
            "lse_out": expected_lse.copy(),
        },
        comparisons={
            "o_out": ComparisonSpec(rtol=2e-2, atol=5e-3, actual_encoding="bfloat16"),
            "lse_out": ComparisonSpec(rtol=1e-4, atol=1e-4),
        },
    )


__all__ = [
    "prepare_msa_sparse_atten_fwd_combine_case",
    "prepare_msa_sparse_atten_fwd_case",
    "prepare_msa_sparse_atten_fwd_nvfp4_kv_case",
    "prepare_msa_sparse_prepare_flat_schedule_case",
    "prepare_msa_sparse_prepare_fwd_split_atomic_case",
]
