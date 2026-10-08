"""CPU-owned inputs and independent references for the GB300/Rubin GEMMs."""

from __future__ import annotations

import numpy as np

from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase, TensorMap
from tests.numsim.corpus.kernels.gemm import (
    _align_up,
    _e2m1_bits_to_float32,
    _e4m3fn_bits_to_float32,
    _float32_to_e2m1_bits,
    _float32_to_e4m3fn_bits,
    _pack_e2m1,
    _pack_sf_128x4,
)
from tests.numsim.support._tirx_kernels import load_tirx_kernel


def _nvfp4_operand(rows: int, k: int, seed: int):
    # Values are multiples of 1/16 with magnitude <= 3. For our K <= 256,
    # FP32 dot products (and dyadic alpha <= 1.25) are exact before FP16 rounding.
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, (rows, k), dtype=np.uint8)
    scales = rng.choice(np.array([0x20, 0x28, 0x30], dtype=np.uint8), (rows, k // 16))
    logical_scales = np.zeros((_align_up(rows, 128), _align_up(k // 16, 4)), dtype=np.uint8)
    logical_scales[:rows, : k // 16] = scales
    values = _e2m1_bits_to_float32(codes) * np.repeat(_e4m3fn_bits_to_float32(scales), 16, axis=1)
    return _pack_e2m1(codes), _pack_sf_128x4(logical_scales).reshape(-1), values, scales


def prepare_fastcu_nvfp4_gemm_case(k: int = 16) -> NumSimCase:
    """Run the source's fixed 76-cluster topology with its smallest K-tail case."""
    m, n = 128, 256
    module = load_tirx_kernel("fastcu_nvfp4_gemm_gb300")
    a, sfa, av, _ = _nvfp4_operand(m, k, 610)
    b, sfb, bv, _ = _nvfp4_operand(n, k, 611)
    row_stride = _align_up(k // 2, 16)
    a_storage = np.zeros((m, row_stride), dtype=np.uint8)
    b_storage = np.zeros((n, row_stride), dtype=np.uint8)
    a_storage[:, : k // 2] = a
    b_storage[:, : k // 2] = b

    def payload_map(data, rows):
        return TensorMap(
            base=data,
            dtype="uint8",
            global_shape=(k // 2, rows, 1),
            global_strides=(row_stride, rows * row_stride),
            box_shape=(128, 128, 1),
            element_strides=(1, 1, 1),
            swizzle="128B",
        ).numpy()

    def scale_map(data, rows):
        sf_inner = _align_up(k // 16, 4)
        return TensorMap(
            base=data,
            dtype="uint8",
            global_shape=(128, 4, sf_inner // 4, _align_up(rows, 128) // 128),
            global_strides=(128, 512, sf_inner * 128),
            box_shape=(128, 4, 3, 1),
            element_strides=(1, 1, 1, 1),
        ).numpy()

    expected = (av @ bv.T).astype(np.float16).reshape(-1)
    kernel = module.make_kernel().func.with_attr("tirx.cuda_arch", "sm_103a")
    return NumSimCase(
        kernel=kernel,
        args={
            "A_tmap": payload_map(a_storage, m),
            "B_tmap": payload_map(b_storage, n),
            "SFA_tmap": scale_map(sfa, m),
            "SFB_tmap": scale_map(sfb, n),
            "C": np.full(m * n, np.nan, dtype=np.float16),
            "M": m,
            "N": n,
            "K_dim": k,
            # One scheduled tile: source prepare_data uses identity route[:total_tiles]
            # and zero side tables, so these are its canonical host inputs.
            "route_table": np.zeros(4096, dtype=np.int32),
            "sm_side": np.zeros(256, dtype=np.int32),
            "cluster_side": np.zeros(128, dtype=np.int32),
            "placement_errors": np.zeros(1, dtype=np.uint32),
        },
        outputs=("C", "placement_errors"),
        reference=lambda: {"C": expected.copy(), "placement_errors": np.zeros(1, dtype=np.uint32)},
        comparisons={"C": ComparisonSpec(rtol=0, atol=0)},
    )


def prepare_dense_blockscaled_rubin_case() -> NumSimCase:
    """One swapped-AB tile with nonuniform scales and a nonidentity alpha."""
    m, n, k = 64, 128, 256
    module = load_tirx_kernel("dense_blockscaled_gemm_sm107")
    a, sfa, av, _ = _nvfp4_operand(m, k, 620)
    b, sfb, bv, _ = _nvfp4_operand(n, k, 621)
    alpha = np.array([0.75], dtype=np.float32)
    expected = ((av @ bv.T) * alpha[0]).astype(np.float16).view(np.uint8).reshape(-1)
    kernel = module.get_kernel(m, n, k, "nvfp4", "float16", float(alpha[0]), 3).with_attr(
        "tirx.cuda_arch", "sm_107a"
    )
    return NumSimCase(
        kernel=kernel,
        args={
            "a": b.reshape(-1),
            "b": a.reshape(-1),
            "sfa": sfb,
            "sfb": sfa,
            "c": np.full((m, n), np.nan, dtype=np.float16).view(np.uint8).reshape(-1),
            "alpha_ptr": alpha,
        },
        outputs=("c",),
        reference=lambda: {"c": expected.copy()},
        comparisons={"c": ComparisonSpec(rtol=0, atol=0)},
    )


def prepare_grouped_masked_rubin_case() -> NumSimCase:
    """Masked counts schedule rounded tiles; the intervening empty group stays untouched."""
    groups, m, n, k = 3, 128, 128, 256
    module = load_tirx_kernel("grouped_gemm_masked_rubin")
    aa = [_nvfp4_operand(m, k, 630 + group) for group in range(groups)]
    bb = [_nvfp4_operand(n, k, 640 + group) for group in range(groups)]
    masked = np.array([73, 0, 113], dtype=np.int32)
    alpha = np.array([0.5, 0.75, 1.25], dtype=np.float32)
    output = np.full((groups, m, n), np.nan, dtype=np.float16)
    expected = output.copy()
    for group, count in enumerate(masked):
        written_rows = min(m, _align_up(int(count), 128))
        expected[group, :written_rows] = (
            aa[group][2][:written_rows] @ bb[group][2].T * alpha[group]
        ).astype(np.float16)
    kernel = module.get_kernel(
        groups, m, n, k, out_dtype="float16", alpha=True, signals=False, tactic=0
    ).with_attr("tirx.cuda_arch", "sm_107a")
    return NumSimCase(
        kernel=kernel,
        args={
            "a": np.concatenate([item[0].reshape(-1) for item in aa]),
            "b": np.concatenate([item[0].reshape(-1) for item in bb]),
            "sfa": np.concatenate([item[1] for item in aa]),
            "sfb": np.concatenate([item[1] for item in bb]),
            "c": output.view(np.uint8).reshape(-1),
            "masked_m": masked,
            "alpha_ptr": alpha,
            "dst_signals": np.zeros(groups, dtype=np.int32),
        },
        outputs=("c",),
        reference=lambda: {"c": expected.view(np.uint8).reshape(-1).copy()},
        comparisons={"c": ComparisonSpec(rtol=0, atol=0)},
    )


def prepare_gather_swiglu_rubin_case(k: int = 256) -> NumSimCase:
    """Gather two experts with padding, then independently quantize the SwiGLU result."""
    groups, seq_len, n = 2, 17, 128
    module = load_tirx_kernel("blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_rubin")
    a, _, av, sfa = _nvfp4_operand(seq_len, k, 650)
    bb = [_nvfp4_operand(n, k, 651 + group) for group in range(groups)]
    count = seq_len * 8 // groups
    rows = groups * 128
    token_ids = np.full(rows, -1, dtype=np.int32)
    permutation = np.arange(seq_len * 8, dtype=np.int32) * 5 % (seq_len * 8)
    projected = np.zeros((rows, n), dtype=np.float32)
    for group in range(groups):
        tokens = permutation[group * count : (group + 1) * count]
        token_ids[group * 128 : group * 128 + count] = tokens
        projected[group * 128 : group * 128 + count] = av[tokens // 8] @ bb[group][2].T
    tiles = projected.reshape(rows, n // 128, 128)
    up, gate = tiles[..., :64], tiles[..., 64:]
    silu = (gate.astype(np.float64) / (1.0 + np.exp(-gate.astype(np.float64)))).astype(np.float32)
    activated = (up * silu).reshape(rows, n // 2)
    blocks = activated.reshape(rows, n // 32, 16)
    scale_codes = _float32_to_e4m3fn_bits(np.max(np.abs(blocks), axis=-1) / np.float32(6))
    scales = _e4m3fn_bits_to_float32(scale_codes)
    normalized = np.divide(
        blocks, scales[..., None], out=np.zeros_like(blocks), where=scales[..., None] != 0
    )
    expected_c = _pack_e2m1(_float32_to_e2m1_bits(normalized.reshape(rows, n // 2)))
    expected_sfc = _pack_sf_128x4(scale_codes)
    kernel = module.get_kernel(
        num_experts=groups, seq_len=seq_len, N=n, K=k, use_pdl=True
    ).with_attr("tirx.cuda_arch", "sm_107a")
    return NumSimCase(
        kernel=kernel,
        args={
            "a": a.reshape(-1),
            "b": np.concatenate([item[0].reshape(-1) for item in bb]),
            "sfa": sfa.reshape(-1),
            "sfb": np.concatenate([item[1] for item in bb]),
            "c": np.full(rows * n // 4, 0xA5, dtype=np.uint8),
            "sfc": np.full(rows * n // 32, 0xA5, dtype=np.uint8),
            "alpha": np.ones(groups, dtype=np.float32),
            "tile_idx_to_expert_idx": np.arange(groups, dtype=np.int32),
            "tile_idx_to_mn_limit": np.arange(groups, dtype=np.int32) * 128 + count,
            "token_id_mapping": token_ids,
            "num_non_exiting_tiles": np.array([groups], dtype=np.int32),
            "global_scale": np.ones(1, dtype=np.float32),
        },
        outputs=("c", "sfc"),
        reference=lambda: {
            "c": expected_c.reshape(-1).copy(),
            "sfc": expected_sfc.reshape(-1).copy(),
        },
        comparisons={name: ComparisonSpec(rtol=0, atol=0) for name in ("c", "sfc")},
    )
