"""NumSim-owned numerical cases for maintained attention kernels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tirx_harness.numsim.cases import (
    ComparisonSpec,
    NumSimCase,
    TensorMap,
)
from tirx_harness.numsim.bindings import _tensor_map_base_array
from tests.numsim.support._tirx_kernels import load_tirx_kernel

flash_attention4 = load_tirx_kernel("flash_attention4")
flash_attention_backward = load_tirx_kernel("flash_attention_backward_sm100")

_FLASH_ATTENTION4_REDUCED_BRANCHES = tuple(
    (256, num_kv_heads, is_causal) for num_kv_heads in (32, 16, 8, 4) for is_causal in (False, True)
) + tuple((384, num_kv_heads, is_causal) for num_kv_heads in (32, 4) for is_causal in (False, True))

FLASH_ATTENTION4_CONFIGS = tuple(
    {
        "batch_size": 1,
        "seq_len": seq_len,
        "num_qo_heads": 32,
        "num_kv_heads": num_kv_heads,
        "head_dim": 128,
        "is_causal": is_causal,
        "seed": 3000 + index,
        "label": (
            f"s{seq_len}_{'mha' if num_kv_heads == 32 else f'gqa{32 // num_kv_heads}'}_"
            f"{'causal' if is_causal else 'noncausal'}"
        ),
    }
    for index, (seq_len, num_kv_heads, is_causal) in enumerate(_FLASH_ATTENTION4_REDUCED_BRANCHES)
)

FLASH_ATTENTION_BACKWARD_CONFIGS = (
    {"is_causal": False, "seed": 20260806, "label": "s256_mha_noncausal"},
    {"is_causal": True, "seed": 20260807, "label": "s256_mha_causal"},
)


def _tensor_map_128b(
    base: np.ndarray,
    *,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_shape: tuple[int, ...],
) -> np.ndarray:
    return TensorMap(
        base=base,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1,) * len(global_shape),
        swizzle="128B",
        interleave=None,
        fill_mode="none",
    ).numpy()


def _flash_attention_backward_tensor_maps(
    *,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dout: np.ndarray,
    dk: np.ndarray,
    dv: np.ndarray,
    dq_acc: np.ndarray,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, np.ndarray]:
    row_shape = (head_dim // 2, seq_len, 2, num_heads, batch_size)
    row_strides = (
        num_heads * head_dim * 2,
        head_dim,
        head_dim * 2,
        seq_len * num_heads * head_dim * 2,
    )
    col_shape = (head_dim, seq_len, num_heads, batch_size)
    col_strides = (
        num_heads * head_dim * 2,
        head_dim * 2,
        seq_len * num_heads * head_dim * 2,
    )

    def row_map(base: np.ndarray, rows: int) -> np.ndarray:
        return _tensor_map_128b(
            base,
            global_shape=row_shape,
            global_strides=row_strides,
            box_shape=(head_dim // 2, rows, 2, 1, 1),
        )

    def col_map(base: np.ndarray, columns: int, rows: int) -> np.ndarray:
        return _tensor_map_128b(
            base,
            global_shape=col_shape,
            global_strides=col_strides,
            box_shape=(columns, rows, 1, 1),
        )

    return {
        "q_row_map": row_map(q, 64),
        "q_col_map": col_map(q, head_dim // 2, 128),
        "k_row_map": row_map(k, 128),
        "k_col_map": col_map(k, head_dim // 2, 256),
        "v_row_map": row_map(v, 128),
        "do_row_map": row_map(dout, 64),
        "do_col_map": col_map(dout, head_dim // 2, 128),
        "dk_map": col_map(dk, 64, 128),
        "dv_map": col_map(dv, 64, 128),
        "dq_map": TensorMap(
            base=dq_acc,
            global_shape=(head_dim, seq_len, num_heads, batch_size),
            global_strides=(
                head_dim * 4,
                seq_len * head_dim * 4,
                num_heads * seq_len * head_dim * 4,
            ),
            box_shape=(head_dim, 8, 1, 1),
            element_strides=(1, 1, 1, 1),
            swizzle=None,
            interleave=None,
            fill_mode="none",
        ).numpy(),
    }


def numpy_attention_reference(q: Any, k: Any, v: Any, *, is_causal: bool) -> np.ndarray:
    q_f32 = np.asarray(q, dtype=np.float32).transpose(0, 2, 1, 3)
    k_f32 = np.asarray(k, dtype=np.float32).transpose(0, 2, 1, 3)
    v_f32 = np.asarray(v, dtype=np.float32).transpose(0, 2, 1, 3)
    if q_f32.shape[1] % k_f32.shape[1] != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    if q_f32.shape[1] != k_f32.shape[1]:
        repeat = q_f32.shape[1] // k_f32.shape[1]
        k_f32 = np.repeat(k_f32, repeat, axis=1)
        v_f32 = np.repeat(v_f32, repeat, axis=1)
    scores = np.matmul(q_f32, np.swapaxes(k_f32, -1, -2)) / np.float32(math.sqrt(q_f32.shape[-1]))
    if is_causal:
        q_len = q_f32.shape[-2]
        kv_len = k_f32.shape[-2]
        diagonal = 1 + kv_len - q_len
        causal_mask = np.triu(np.ones((q_len, kv_len), dtype=np.bool_), k=diagonal)
        scores = np.where(causal_mask, -np.inf, scores)
    row_max = np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores - row_max)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return np.matmul(probabilities, v_f32).transpose(0, 2, 1, 3).astype(np.float16)


def prepare_flash_attention4_fp4_case() -> NumSimCase:
    """NVFP4 Q/K, BF16 V: two query stages and nonuniform block scales."""
    from tests.numsim.corpus.kernels.gemm import (
        _bfloat16_bits_to_float32,
        _e2m1_bits_to_float32,
        _float32_to_bfloat16_bits,
        _float32_to_e4m3fn_bits,
        _pack_e2m1,
        _pack_sf_128x4,
    )

    module = load_tirx_kernel("flash_attention4_fp4")
    rng = np.random.default_rng(20260906)
    sq, sk, d = 256, 128, 64

    def operand(rows):
        bits = rng.integers(0, 16, (rows, d), dtype=np.uint8)
        scales = np.exp2(rng.integers(-4, -1, (rows, 4))).astype(np.float32)
        storage = _pack_sf_128x4(_float32_to_e4m3fn_bits(scales))
        decoded = _e2m1_bits_to_float32(bits) * np.repeat(scales, 16, axis=1)
        return _pack_e2m1(bits), storage.view(np.uint16), decoded

    q, sfq, q_value = operand(sq)
    k, sfk, k_value = operand(sk)
    v = _float32_to_bfloat16_bits(rng.standard_normal((sk, d)).astype(np.float32) / 4)
    output = np.full((sq, d), np.uint16(0x7FC0), dtype=np.uint16)
    scores = q_value.astype(np.float64) @ k_value.astype(np.float64).T / math.sqrt(d)
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = probabilities @ _bfloat16_bits_to_float32(v)

    def tensor_map(base, dtype, shape, strides, box, swizzle):
        return TensorMap(
            base=base,
            dtype=dtype,
            global_shape=shape,
            global_strides=strides,
            box_shape=box,
            element_strides=(1,) * len(shape),
            swizzle=swizzle,
        ).numpy()

    args = {}
    for name, data, scales, rows in (("q", q, sfq, sq), ("k", k, sfk, sk)):
        args[f"tmap_{name}"] = tensor_map(
            data,
            "uint8",
            (d // 2, rows, 1, 1),
            (d // 2, d // 2, rows * d // 2),
            (d // 2, 128, 1, 1),
            "32B",
        )
        args[f"tmap_sf{name}"] = tensor_map(
            scales,
            "uint16",
            (256, rows // 128, 1, 1, 1),
            (512, rows * 4, rows * 4, rows * 4),
            (256, 1, 1, 1, 1),
            None,
        )
    for name, data, rows in (("v", v, sk), ("o", output, sq)):
        args[f"tmap_{name}"] = tensor_map(
            data,
            "bfloat16",
            (d, rows, 1, 1),
            (d * 2, d * 2, rows * d * 2),
            (64, 128, 1, 1),
            "128B",
        )
    args["softmax_scale_log2"] = np.float32(math.log2(math.e) / math.sqrt(d))
    return NumSimCase(
        kernel=module.make_kernel(
            module.Spec("nvfp4", "bf16", d, False, 1, 1), 1, sq, sk, 1, 1, 1
        ).func,
        args=args,
        outputs={"output": "tmap_o"},
        reference=lambda: {"output": expected.reshape(1, 1, sq, d)},
        comparisons={"output": ComparisonSpec(rtol=2e-2, atol=2e-3, actual_encoding="bfloat16")},
    )


def prepare_flash_attention4_case(
    batch_size: int,
    seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool = False,
    seed: int = 0,
) -> NumSimCase:
    rng = np.random.default_rng(seed)
    q_shape = (batch_size, seq_len, num_qo_heads, head_dim)
    kv_shape = (batch_size, seq_len, num_kv_heads, head_dim)
    q = (rng.standard_normal(q_shape, dtype=np.float32) * np.float32(0.25)).astype(np.float16)
    k = (rng.standard_normal(kv_shape, dtype=np.float32) * np.float32(0.25)).astype(np.float16)
    v = (rng.standard_normal(kv_shape, dtype=np.float32) * np.float32(0.25)).astype(np.float16)
    output = np.zeros(q_shape, dtype=np.float16)
    expected = np.ascontiguousarray(numpy_attention_reference(q, k, v, is_causal=is_causal))
    q_binding = q
    k_binding = k
    v_binding = v
    output_binding = output
    gqa = num_qo_heads // num_kv_heads
    seq_q_per_tile = 128 // gqa

    def qo_map(base: np.ndarray) -> np.ndarray:
        if gqa == 1:
            return _tensor_map_128b(
                base,
                global_shape=(head_dim // 2, seq_len, batch_size * num_qo_heads * 2),
                global_strides=(num_qo_heads * head_dim * 2, head_dim),
                box_shape=(head_dim // 2, seq_q_per_tile, 2),
            )
        return _tensor_map_128b(
            base,
            global_shape=(head_dim // 2, num_qo_heads, seq_len, batch_size * 2),
            global_strides=(head_dim * 2, num_qo_heads * head_dim * 2, head_dim),
            box_shape=(head_dim // 2, gqa, seq_q_per_tile, 2),
        )

    def kv_map(base: np.ndarray) -> np.ndarray:
        return _tensor_map_128b(
            base,
            global_shape=(head_dim // 2, seq_len, batch_size * num_kv_heads * 2),
            global_strides=(num_kv_heads * head_dim * 2, head_dim),
            box_shape=(head_dim // 2, 128, 2),
        )

    output_map = qo_map(output_binding)
    expected_map = qo_map(expected)
    args = {
        "Q_tensor_map": qo_map(q_binding),
        "Q_tensor_map_1": qo_map(q_binding),
        "K_tensor_map": kv_map(k_binding),
        "K_tensor_map_1": kv_map(k_binding),
        "V_tensor_map": kv_map(v_binding),
        "V_tensor_map_1": kv_map(v_binding),
        "O_tensor_map": output_map,
    }
    return NumSimCase(
        kernel=flash_attention4.get_kernel(
            batch_size=batch_size,
            seq_len=seq_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            is_causal=is_causal,
        ),
        args=args,
        outputs={"O": "O_tensor_map"},
        reference=lambda: {"O": _tensor_map_base_array(expected_map).copy()},
        comparisons={"O": ComparisonSpec(rtol=1e-2, atol=1e-2)},
    )


def prepare_flash_attention_backward_case(*, is_causal: bool, seed: int = 0) -> NumSimCase:
    batch_size, seq_len, num_heads, head_dim = 1, 256, 1, 128
    shape = (batch_size, seq_len, num_heads, head_dim)
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal(shape, dtype=np.float32) * np.float32(0.25)).astype(np.float16)
    k = (rng.standard_normal(shape, dtype=np.float32) * np.float32(0.25)).astype(np.float16)
    v = (rng.standard_normal(shape, dtype=np.float32) * np.float32(0.25)).astype(np.float16)
    dout = (rng.standard_normal(shape, dtype=np.float32) * np.float32(0.20)).astype(np.float16)
    scale = np.float32(1.0 / math.sqrt(head_dim))

    qh = q.astype(np.float32).transpose(0, 2, 1, 3)
    kh = k.astype(np.float32).transpose(0, 2, 1, 3)
    vh = v.astype(np.float32).transpose(0, 2, 1, 3)
    doh = dout.astype(np.float32).transpose(0, 2, 1, 3)
    scores = np.matmul(qh, np.swapaxes(kh, -1, -2)) * scale
    if is_causal:
        scores = np.where(np.triu(np.ones((seq_len, seq_len), dtype=np.bool_), 1), -np.inf, scores)
    row_max = np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(scores - row_max)
    denominator = np.sum(exponentials, axis=-1, keepdims=True)
    probabilities = exponentials / denominator
    lse = np.log(denominator[..., 0]) + row_max[..., 0]
    output = np.matmul(probabilities, vh)
    dpsum = np.sum(doh * output, axis=-1)
    dp = np.matmul(doh, np.swapaxes(vh, -1, -2))
    ds = probabilities * (dp - dpsum[..., None])
    dq = (np.matmul(ds, kh) * scale).transpose(0, 2, 1, 3)
    dk = (np.matmul(np.swapaxes(ds, -1, -2), qh) * scale).transpose(0, 2, 1, 3)
    dv = np.matmul(np.swapaxes(probabilities, -1, -2), doh).transpose(0, 2, 1, 3)

    dq_acc = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    for sequence in range(seq_len):
        sequence_in_block = sequence % 128
        for dimension in range(0, head_dim, 4):
            physical_sequence = (
                sequence // 128 * 128
                + ((sequence_in_block >> 5) & 1)
                + (((dimension >> 6) & 1) << 1)
                + (((dimension >> 2) & 15) << 2)
                + (((sequence_in_block >> 6) & 1) << 6)
            )
            physical_dimension = (sequence_in_block & 31) << 2
            dq_acc[0, 0, physical_sequence, physical_dimension : physical_dimension + 4] = (
                dq[0, sequence, 0, dimension : dimension + 4] / scale
            )

    q_binding = q.reshape(-1)
    k_binding = k.reshape(-1)
    v_binding = v.reshape(-1)
    dout_binding = dout.reshape(-1)
    dk_binding = np.zeros(math.prod(shape), dtype=np.float16)
    dv_binding = np.zeros(math.prod(shape), dtype=np.float16)
    dq_acc_binding = np.zeros(batch_size * num_heads * seq_len * head_dim, dtype=np.float32)
    args = {
        "Q_g": q_binding,
        "K_g": k_binding,
        "V_g": v_binding,
        "dO_g": dout_binding,
        "LSE_g": (lse * np.float32(math.log2(math.e))).astype(np.float32).reshape(-1),
        "dpsum_g": dpsum.astype(np.float32).reshape(-1),
        "dK_g": dk_binding,
        "dV_g": dv_binding,
        "dQ_acc_g": dq_acc_binding,
        **_flash_attention_backward_tensor_maps(
            q=q_binding,
            k=k_binding,
            v=v_binding,
            dout=dout_binding,
            dk=dk_binding,
            dv=dv_binding,
            dq_acc=dq_acc_binding,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        ),
    }
    return NumSimCase(
        kernel=flash_attention_backward.get_kernel(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            is_causal=is_causal,
        ),
        args=args,
        outputs=("dK_g", "dV_g", "dQ_acc_g"),
        reference=lambda: {
            "dK_g": dk.astype(np.float16).reshape(-1),
            "dV_g": dv.astype(np.float16).reshape(-1),
            "dQ_acc_g": dq_acc.reshape(-1),
        },
        comparisons={
            "dK_g": ComparisonSpec(rtol=2e-2, atol=2e-3),
            "dV_g": ComparisonSpec(rtol=2e-2, atol=2e-3),
            "dQ_acc_g": ComparisonSpec(rtol=2e-2, atol=2e-2),
        },
    )


def prepare_flash_attention_backward_persistent_analysis_case() -> NumSimCase:
    """Create the smallest full launch that reuses a physical cluster."""
    batch_size, seq_len, num_heads, head_dim = 1, 256, 2, 128
    shape = (batch_size, seq_len, num_heads, head_dim)
    args = {
        name: np.zeros(math.prod(shape), dtype=np.float16)
        for name in ("Q_g", "K_g", "V_g", "dO_g", "dK_g", "dV_g")
    }
    args.update(
        {
            "LSE_g": np.zeros(batch_size * num_heads * seq_len, dtype=np.float32),
            "dpsum_g": np.zeros(batch_size * num_heads * seq_len, dtype=np.float32),
            "dQ_acc_g": np.zeros(batch_size * num_heads * seq_len * head_dim, dtype=np.float32),
        }
    )
    args.update(
        _flash_attention_backward_tensor_maps(
            q=args["Q_g"],
            k=args["K_g"],
            v=args["V_g"],
            dout=args["dO_g"],
            dk=args["dK_g"],
            dv=args["dV_g"],
            dq_acc=args["dQ_acc_g"],
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    )
    return NumSimCase(
        kernel=flash_attention_backward.get_kernel(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            is_causal=False,
            sm_count=2,
        ),
        args=args,
        outputs=(),
        reference=lambda: {},
    )


__all__ = [
    "FLASH_ATTENTION_BACKWARD_CONFIGS",
    "FLASH_ATTENTION4_CONFIGS",
    "numpy_attention_reference",
    "prepare_flash_attention_backward_case",
    "prepare_flash_attention_backward_persistent_analysis_case",
    "prepare_flash_attention4_case",
    "prepare_flash_attention4_fp4_case",
]
