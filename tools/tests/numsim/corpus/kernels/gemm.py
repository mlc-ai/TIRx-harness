"""NumSim-owned numerical corpus for GEMM kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

from tirx_harness.numsim.cases import (
    ComparisonRegion,
    ComparisonSpec,
    NumSimCase,
    TensorMap,
)
from tests.numsim.support._tirx_kernels import load_tirx_kernel


FP8_1D1D_NUM_SMS = 2
bmm_fp8_rubin = load_tirx_kernel("bmm_fp8_rubin")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _choose_deepgemm_config(M: int, N: int, K: int):
    module = load_tirx_kernel("deepgemm_sm100_fp8_gemm_1d1d")
    spec = module._spec_for({"M": M, "N": N, "K": K, "num_sms": FP8_1D1D_NUM_SMS})
    return spec.swap_ab, spec


def get_fp16_bf16_kernel(*, dtype: str, M: int, N: int, K: int):
    module = load_tirx_kernel("fp16_bf16_gemm")
    return module.get_kernel(dtype=dtype, M=M, N=N, K=K)


def get_fp8_blockwise_kernel(*, M: int, N: int, K: int):
    module = load_tirx_kernel("deepgemm_sm100_fp8_gemm_1d1d")
    return module.get_kernel(M=M, N=N, K=K, num_sms=FP8_1D1D_NUM_SMS)


def get_nvfp4_kernel(*, M: int, N: int, K: int):
    module = load_tirx_kernel("nvfp4_gemm")
    return module.get_kernel(M=M, N=N, K=K)


def get_grouped_fp8_kernel(
    *, num_groups: int, expected_m_per_group: int, N: int, K: int, seed: int
):
    module = load_tirx_kernel("deepgemm_sm100_m_grouped_fp8_gemm_contiguous")
    return module.get_kernel(
        num_groups=num_groups,
        expected_m_per_group=expected_m_per_group,
        N=N,
        K=K,
        seed=seed,
        num_sms=2,
    )


FP16_BF16_CONFIGS = tuple(
    {
        "dtype": dtype,
        "M": 256,
        "N": 2048,
        "K": 64,
        "seed": 2000 + index,
        "label": f"{dtype}_m256_n2048_k64_full",
    }
    for index, dtype in enumerate(("fp16", "bf16"))
)

FP8_BLOCKWISE_CONFIGS = (
    {
        "M": 16,
        "N": 256,
        "K": 512,
        "expected_swap_ab": True,
        "seed": 2800,
        "label": "swap_ab_m16_n256_k512",
    },
    {
        "M": 512,
        "N": 608,
        "K": 512,
        "expected_swap_ab": False,
        "seed": 2801,
        "label": "direct_ab_m512_n608_k512",
    },
)

NVFP4_CONFIGS = (
    {"M": 256, "N": 256, "K": 256, "seed": 2900, "label": "full_nvfp4_m256_n256_k256"},
)

GROUPED_FP8_CONFIGS = (
    {
        "num_groups": 4,
        "expected_m_per_group": 256,
        "N": 384,
        "K": 512,
        "seed": 1,
        "label": "small_g4_m256_n384_k512",
    },
)


def _float32_to_bfloat16_bits(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _bfloat16_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def prepare_fp16_bf16_case(dtype: str, M: int, N: int, K: int, seed: int = 0) -> NumSimCase:
    if dtype not in {"fp16", "bf16"}:
        raise ValueError(f"Unsupported dtype: {dtype}")

    rng = np.random.default_rng(seed)
    a_f32 = rng.integers(-3, 4, size=(M, K), dtype=np.int16).astype(np.float32)
    b_f32 = rng.integers(-3, 4, size=(N, K), dtype=np.int16).astype(np.float32)
    if dtype == "fp16":
        a = a_f32.astype(np.float16)
        b = b_f32.astype(np.float16)
        output = np.zeros((M, N), dtype=np.float16)
        expected = (a.astype(np.float32) @ b.astype(np.float32).T).astype(np.float16)
        buffer_dtype = "float16"
    else:
        a = _float32_to_bfloat16_bits(a_f32)
        b = _float32_to_bfloat16_bits(b_f32)
        output = np.zeros((M, N), dtype=np.uint16)
        expected_f32 = _bfloat16_bits_to_float32(a) @ _bfloat16_bits_to_float32(b).T
        expected = _float32_to_bfloat16_bits(expected_f32)
        buffer_dtype = "bfloat16"

    args = {
        "a": a,
        "b": b,
        "d": output,
    }
    return NumSimCase(
        kernel=get_fp16_bf16_kernel(dtype=dtype, M=M, N=N, K=K),
        args=args,
        outputs={"D": "d"},
        reference=lambda: {"D": expected.copy()},
        comparisons={"D": ComparisonSpec(rtol=0.0, atol=0.0)},
    )


def _e4m3fn_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (bits & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1.0) + mantissa / np.float32(8.0), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8.0), -6)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    result = np.where((bits & np.uint8(0x80)) != 0, -magnitude, magnitude)
    is_nan = (exponent == 0xF) & ((bits & np.uint8(0x7)) == np.uint8(0x7))
    return np.where(is_nan, np.float32(np.nan), result).astype(np.float32)


_E4M3FN_POSITIVE_VALUES = _e4m3fn_bits_to_float32(np.arange(0x7F, dtype=np.uint8))


def _float32_to_e4m3fn_bits(values: Any) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    magnitude = np.abs(source)
    finite_magnitude = np.nan_to_num(
        magnitude, nan=np.float32(0.0), posinf=_E4M3FN_POSITIVE_VALUES[-1]
    )
    upper = np.searchsorted(_E4M3FN_POSITIVE_VALUES, finite_magnitude, side="left")
    upper = np.minimum(upper, len(_E4M3FN_POSITIVE_VALUES) - 1)
    lower = np.maximum(upper - 1, 0)
    lower_distance = finite_magnitude - _E4M3FN_POSITIVE_VALUES[lower]
    upper_distance = _E4M3FN_POSITIVE_VALUES[upper] - finite_magnitude
    use_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    encoded = np.where(use_upper, upper, lower).astype(np.uint8)
    encoded |= np.signbit(source).astype(np.uint8) << np.uint8(7)
    return np.where(np.isnan(source), np.uint8(0x7F), encoded).astype(np.uint8)


def prepare_bmm_fp8_rubin_case() -> NumSimCase:
    """Exercise one complete Rubin two-CTA cluster with exact FP8 operands."""

    batch, m_dim, n_dim, k_dim = 1, 256, 128, 128
    config = {
        "B": batch,
        "M": m_dim,
        "N": n_dim,
        "K": k_dim,
        "ab_dtype": "float8_e4m3fn",
        "c_dtype": "bfloat16",
        "tactic": 1,
    }
    row_scale = np.where(np.arange(m_dim) % 2, 2.0, 1.0).astype(np.float32)
    column_scale = np.resize(np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32), n_dim)
    a_values = np.empty((batch, m_dim, k_dim), dtype=np.float32)
    a_values[:, :, :64] = row_scale[None, :, None]
    a_values[:, :, 64:] = np.float32(2.0) * row_scale[None, :, None]
    b_values = np.empty((batch, k_dim, n_dim), dtype=np.float32)
    b_values[:, :64, :] = column_scale[None, None, :]
    b_values[:, 64:, :] = -column_scale[None, None, :]
    a = _float32_to_e4m3fn_bits(a_values)
    b = _float32_to_e4m3fn_bits(b_values)
    expected_f32 = np.matmul(_e4m3fn_bits_to_float32(a), _e4m3fn_bits_to_float32(b))
    expected = np.ascontiguousarray(_float32_to_bfloat16_bits(expected_f32)).view(np.uint8)

    runtime_archs = tuple(bmm_fp8_rubin.KERNEL_META["runtime_cuda_archs"])
    if runtime_archs != ("sm_107a",):
        raise ValueError(
            "bmm_fp8_rubin must declare exactly one sm_107a runtime architecture, "
            f"got {runtime_archs!r}"
        )
    kernel = bmm_fp8_rubin.get_kernel(**config).with_attr("tirx.cuda_arch", runtime_archs[0])

    return NumSimCase(
        # get_kernel() exposes a PrimFunc, so retain its registry-owned target
        # explicitly for architecture-sensitive SM100/SM107 descriptor decoding.
        kernel=kernel,
        args={
            "a": a.reshape(-1),
            "b": b.reshape(-1),
            "c": np.zeros(batch * m_dim * n_dim * 2, dtype=np.uint8),
            "output_scale": np.ones(1, dtype=np.float32),
        },
        outputs=("c",),
        reference=lambda: {"c": expected.reshape(-1).copy()},
        comparisons={"c": ComparisonSpec(rtol=0, atol=0)},
    )


def _e8m0_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8)
    exponent = np.where(bits == np.uint8(0xFF), 0, bits.astype(np.int16) - 127)
    result = np.ldexp(np.ones(bits.shape, dtype=np.float32), exponent)
    return np.where(bits == np.uint8(0xFF), np.float32(np.nan), result).astype(np.float32)


def _e8m0_bits_from_exponents(exponents: Any) -> np.ndarray:
    exponents = np.asarray(exponents)
    if np.any(exponents < -127) or np.any(exponents > 127):
        raise ValueError("E8M0 exponent must be in [-127, 127]")
    return (exponents.astype(np.int16) + 127).astype(np.uint8)


def _pack_e8m0_scales(scale_bits: Any) -> np.ndarray:
    scale_bits = np.asarray(scale_bits, dtype=np.uint8)
    if scale_bits.ndim != 2 or scale_bits.shape[1] % 4 != 0:
        raise ValueError("E8M0 scale matrix must have shape (rows, 4 * groups)")
    groups = scale_bits.reshape(scale_bits.shape[0], -1, 4).astype(np.uint32)
    packed = (
        groups[..., 0]
        | (groups[..., 1] << np.uint32(8))
        | (groups[..., 2] << np.uint32(16))
        | (groups[..., 3] << np.uint32(24))
    )
    return packed.T.copy()


def _unpack_e8m0_scales(packed: Any, scale_blocks: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint32)
    if packed.ndim != 2 or scale_blocks % 4 != 0 or packed.shape[0] != scale_blocks // 4:
        raise ValueError("packed E8M0 scale shape does not match the requested block count")
    words = packed.T[..., None]
    shifts = np.arange(4, dtype=np.uint32) * np.uint32(8)
    return ((words >> shifts) & np.uint32(0xFF)).astype(np.uint8).reshape(packed.shape[1], -1)


def _fp8_blockwise_reference(A: Any, B: Any, SFA: Any, SFB: Any) -> np.ndarray:
    A = np.asarray(A, dtype=np.uint8)
    B = np.asarray(B, dtype=np.uint8)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("FP8 operands must be rank-2 matrices with the same K")
    K = A.shape[1]
    if K % 512 != 0:
        raise ValueError("packed E8M0 scales require K to be divisible by 512")
    scale_blocks = K // 128
    sfa = _e8m0_bits_to_float32(_unpack_e8m0_scales(SFA, scale_blocks))
    sfb = _e8m0_bits_to_float32(_unpack_e8m0_scales(SFB, scale_blocks))
    a_dequant = (
        _e4m3fn_bits_to_float32(A).reshape(A.shape[0], scale_blocks, 128) * sfa[:, :, None]
    ).reshape(A.shape)
    b_dequant = (
        _e4m3fn_bits_to_float32(B).reshape(B.shape[0], scale_blocks, 128) * sfb[:, :, None]
    ).reshape(B.shape)
    return _float32_to_bfloat16_bits(a_dequant @ b_dequant.T)


def _tma_swizzle(mode: int) -> str | None:
    if mode == 0:
        return None
    if mode not in {32, 64, 128}:
        raise ValueError(f"unsupported TensorMap swizzle mode: {mode}")
    return f"{mode}B"


def _tensor_map_2d(
    array: np.ndarray,
    *,
    dtype: str,
    global_shape: tuple[int, int],
    global_stride: int,
    box_shape: tuple[int, int],
    swizzle: int = 0,
) -> np.ndarray:
    return TensorMap(
        base=array,
        dtype=dtype,
        global_shape=global_shape,
        global_strides=(global_stride,),
        box_shape=box_shape,
        element_strides=(1, 1),
        swizzle=_tma_swizzle(swizzle),
        interleave=None,
        fill_mode="none",
    ).numpy()


def _tensor_map_3d(
    array: np.ndarray,
    *,
    dtype: str,
    global_shape: tuple[int, int, int],
    global_strides: tuple[int, int],
    box_shape: tuple[int, int, int],
    swizzle: int = 0,
) -> np.ndarray:
    return TensorMap(
        base=array,
        dtype=dtype,
        global_shape=global_shape,
        global_strides=global_strides,
        box_shape=box_shape,
        element_strides=(1, 1, 1),
        swizzle=_tma_swizzle(swizzle),
        interleave=None,
        fill_mode="none",
    ).numpy()


def _fp8_1d1d_tensor_maps(
    *,
    spec: Any,
    A: np.ndarray,
    B: np.ndarray,
    SFA: np.ndarray,
    SFB: np.ndarray,
    D: np.ndarray,
    M: int,
    N: int,
    K: int,
    num_groups: int = 1,
) -> dict[str, np.ndarray]:
    """Mirror the upstream 1d1d host descriptor encoders with CPU backings."""

    aligned_m = _align_up(M, 16)
    aligned_n = _align_up(N, 16)
    scale_rows = K // 512
    return {
        "tensor_map_a": _tensor_map_2d(
            A,
            dtype="float8_e4m3fn",
            global_shape=(K, M),
            global_stride=K,
            box_shape=(spec.swizzle_a_mode, spec.load_block_m),
            swizzle=spec.swizzle_a_mode,
        ),
        "tensor_map_b": _tensor_map_2d(
            B,
            dtype="float8_e4m3fn",
            global_shape=(K, N * num_groups),
            global_stride=K,
            box_shape=(spec.swizzle_b_mode, spec.load_block_n),
            swizzle=spec.swizzle_b_mode,
        ),
        "tensor_map_sfa": _tensor_map_2d(
            SFA,
            dtype="uint32",
            global_shape=(aligned_m, scale_rows),
            global_stride=aligned_m * 4,
            box_shape=(spec.block_m, 1),
        ),
        "tensor_map_sfb": _tensor_map_2d(
            SFB,
            dtype="uint32",
            global_shape=(aligned_n, scale_rows * num_groups),
            global_stride=aligned_n * 4,
            box_shape=(spec.block_n, 1),
        ),
        "tensor_map_cd": _tensor_map_2d(
            D,
            dtype="bfloat16",
            global_shape=(N, M),
            global_stride=N * 2,
            box_shape=(spec.swizzle_cd_mode // 2, spec.store_block_m),
            swizzle=spec.swizzle_cd_mode,
        ),
    }


def prepare_fp8_blockwise_case(
    M: int, N: int, K: int, expected_swap_ab: bool, seed: int = 0
) -> NumSimCase:
    if K % 512 != 0:
        raise ValueError("NumSim FP8 case requires K to be divisible by 512")
    swap_ab, spec = _choose_deepgemm_config(M, N, K)
    if swap_ab != expected_swap_ab:
        raise ValueError(f"FP8 heuristic selected SWAP_AB={swap_ab}, expected {expected_swap_ab}")

    rng = np.random.default_rng(seed)
    operand_values = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    operand_codes = _float32_to_e4m3fn_bits(operand_values)
    A = rng.choice(operand_codes, size=(M, K)).astype(np.uint8)
    B = rng.choice(operand_codes, size=(N, K)).astype(np.uint8)
    scale_blocks = K // 128
    sfa_bits = _e8m0_bits_from_exponents(
        rng.integers(-1, 2, size=(M, scale_blocks), dtype=np.int16)
    )
    sfb_bits = _e8m0_bits_from_exponents(
        rng.integers(-1, 2, size=(N, scale_blocks), dtype=np.int16)
    )
    SFA = _pack_e8m0_scales(sfa_bits)
    SFB = _pack_e8m0_scales(sfb_bits)
    D = np.zeros((M, N), dtype=np.uint16)

    args = {
        "grouped_layout": np.zeros((1,), dtype=np.int32),
        "grouped_len": np.int32(1),
        "shape_m": np.int32(M),
        "shape_n": np.int32(N),
        "shape_k": np.int32(K),
        **_fp8_1d1d_tensor_maps(spec=spec, A=A, B=B, SFA=SFA, SFB=SFB, D=D, M=M, N=N, K=K),
    }
    return NumSimCase(
        kernel=get_fp8_blockwise_kernel(M=M, N=N, K=K),
        args=args,
        outputs={"D": "tensor_map_cd"},
        reference=lambda: {"D": _fp8_blockwise_reference(A, B, SFA, SFB)},
        comparisons={"D": ComparisonSpec(rtol=0.0, atol=0.0)},
    )


def prepare_grouped_fp8_case(
    num_groups: int,
    expected_m_per_group: int,
    N: int,
    K: int,
    seed: int = 0,
) -> NumSimCase:
    """Build a deterministic CPU-only grouped FP8 protocol case."""

    if K % 512 != 0:
        raise ValueError("NumSim grouped FP8 case requires K to be divisible by 512")
    module = load_tirx_kernel("deepgemm_sm100_m_grouped_fp8_gemm_contiguous")
    aligned_ms = module.make_aligned_ms(
        num_groups,
        expected_m_per_group,
        seed,
        module.get_theoretical_mk_alignment(),
    )
    M = sum(aligned_ms)
    scale_words = K // 512
    A = np.zeros((M, K), dtype=np.uint8)
    B = np.zeros((num_groups, N, K), dtype=np.uint8)
    # Four packed E8M0 0x7f values encode unit scale.
    SFA = np.full((scale_words, M), np.uint32(0x7F7F7F7F), dtype=np.uint32)
    SFB = np.full((num_groups, scale_words, N), np.uint32(0x7F7F7F7F), dtype=np.uint32)
    D = np.zeros((M, N), dtype=np.uint16)
    grouped_layout = np.full((M,), -1, dtype=np.int32)
    start = 0
    for group, aligned_m in enumerate(aligned_ms):
        grouped_layout[start : start + aligned_m] = group
        start += aligned_m

    config = {
        "num_groups": num_groups,
        "expected_m_per_group": expected_m_per_group,
        "N": N,
        "K": K,
        "seed": seed,
        "num_sms": 2,
    }
    spec = module._spec_for(dict(config))

    args = {
        "grouped_layout": grouped_layout,
        "grouped_len": np.int32(M),
        "shape_m": np.int32(M),
        "shape_n": np.int32(N),
        "shape_k": np.int32(K),
        **_fp8_1d1d_tensor_maps(
            spec=spec,
            A=A,
            B=B,
            SFA=SFA,
            SFB=SFB,
            D=D,
            M=M,
            N=N,
            K=K,
            num_groups=num_groups,
        ),
    }
    return NumSimCase(
        kernel=get_grouped_fp8_kernel(
            num_groups=num_groups,
            expected_m_per_group=expected_m_per_group,
            N=N,
            K=K,
            seed=seed,
        ),
        args=args,
        outputs={"D": "tensor_map_cd"},
        reference=lambda: {"D": np.zeros_like(D)},
        comparisons={"D": ComparisonSpec(rtol=0.0, atol=0.0)},
    )


def prepare_fp8_bmm_case() -> NumSimCase:
    """Exercise the batched rank-3 TensorMap ABI with interleaved A/D batches."""

    module = load_tirx_kernel("deepgemm_sm100_fp8_bmm")
    config = {
        "expr": "bhr,hdr->bhd",
        "H": 2,
        "R": 128,
        "D": 128,
        "B": 16,
        "num_sms": 2,
    }
    spec = module._spec_for(dict(config))
    H, M, N, K = config["H"], config["B"], config["D"], config["R"]

    # Canonical `fp8_einsum` stores X as [M, H, K] and exposes an [H, M, K]
    # permuted view to the descriptor.  Keep that physical interleave here so
    # the batch stride is independently exercised rather than inferred from a
    # contiguous [H, M, K] convenience array.
    A = np.full((M, H, K), np.uint8(0x38), dtype=np.uint8)
    A[:, 1, :] = np.uint8(0x40)
    B = np.full((H, N, K), np.uint8(0x38), dtype=np.uint8)
    D = np.zeros((M, H, N), dtype=np.uint16)
    scale_word = np.uint32(0x7F7F7F7F)
    SFA = np.full((H, 1, _align_up(M, 16)), scale_word, dtype=np.uint32)
    SFB = np.full((H, 1, _align_up(N, 16)), scale_word, dtype=np.uint32)
    expected_f32 = np.empty((M, H, N), dtype=np.float32)
    expected_f32[:, 0, :] = np.float32(K)
    expected_f32[:, 1, :] = np.float32(2 * K)
    expected = _float32_to_bfloat16_bits(expected_f32)

    args = {
        "grouped_layout": np.zeros((1,), dtype=np.int32),
        "grouped_len": np.int32(1),
        "shape_m": np.int32(M),
        "shape_n": np.int32(N),
        "shape_k": np.int32(K),
        "tensor_map_a": _tensor_map_3d(
            A,
            dtype="float8_e4m3fn",
            global_shape=(K, M, H),
            global_strides=(H * K, K),
            box_shape=(spec.block_k, spec.load_block_m, 1),
            swizzle=spec.swizzle_a_mode,
        ),
        "tensor_map_b": _tensor_map_3d(
            B,
            dtype="float8_e4m3fn",
            global_shape=(K, N, H),
            global_strides=(K, N * K),
            box_shape=(spec.block_k, spec.load_block_n, 1),
            swizzle=spec.swizzle_b_mode,
        ),
        "tensor_map_sfa": _tensor_map_2d(
            SFA,
            dtype="uint32",
            global_shape=(_align_up(M, 16), H),
            global_stride=_align_up(M, 16) * 4,
            box_shape=(spec.block_m, 1),
        ),
        "tensor_map_sfb": _tensor_map_2d(
            SFB,
            dtype="uint32",
            global_shape=(_align_up(N, 16), H),
            global_stride=_align_up(N, 16) * 4,
            box_shape=(spec.block_n, 1),
        ),
        "tensor_map_cd": _tensor_map_3d(
            D,
            dtype="bfloat16",
            global_shape=(N, M, H),
            global_strides=(H * N * 2, N * 2),
            box_shape=(spec.swizzle_cd_mode // 2, spec.store_block_m, 1),
            swizzle=spec.swizzle_cd_mode,
        ),
    }
    return NumSimCase(
        kernel=module.get_kernel(**config),
        args=args,
        outputs={"D": "tensor_map_cd"},
        reference=lambda: {"D": expected.transpose(1, 0, 2).copy()},
        comparisons={"D": ComparisonSpec(rtol=0.0, atol=0.0)},
    )


def prepare_k_grouped_fp8_case() -> NumSimCase:
    """Cover per-group K cursors, MN-major operands, and FP32 accumulation."""

    module = load_tirx_kernel("deepgemm_sm100_k_grouped_fp8_gemm_contiguous")
    config = {
        "num_groups": 2,
        "M": 128,
        "N": 128,
        "expected_k_per_group": 128,
        "gran_k": 128,
        "k_alignment": 128,
        "seed": 1,
        "num_sms": 2,
    }
    spec = module._spec_for(dict(config))
    _, aligned_ks = module.make_ks(
        num_groups=config["num_groups"],
        expected_k_per_group=config["expected_k_per_group"],
        k_alignment=config["k_alignment"],
        seed=config["seed"],
    )
    G, M, N = config["num_groups"], config["M"], config["N"]
    K = sum(aligned_ks)
    A = np.full((K, M), np.uint8(0x38), dtype=np.uint8)
    B = np.full((K, N), np.uint8(0x38), dtype=np.uint8)
    B[aligned_ks[0] :, :] = np.uint8(0x40)
    scale_rows = sum(_align_up(k, 512) // 512 for k in aligned_ks)
    scale_word = np.uint32(0x7F7F7F7F)
    SFA = np.full((scale_rows, _align_up(M, 16)), scale_word, dtype=np.uint32)
    SFB = np.full((scale_rows, _align_up(N, 16)), scale_word, dtype=np.uint32)
    initial = np.stack(
        (
            np.full((M, N), np.float32(1.25), dtype=np.float32),
            np.full((M, N), np.float32(-2.5), dtype=np.float32),
        )
    )
    D = initial.copy()
    expected = initial.copy()
    expected[0] += np.float32(aligned_ks[0])
    expected[1] += np.float32(2 * aligned_ks[1])

    args = {
        "grouped_layout": np.asarray(aligned_ks, dtype=np.int32),
        "grouped_len": np.int32(G),
        "shape_m": np.int32(M),
        "shape_n": np.int32(N),
        "shape_k": np.int32(K),
        "tensor_map_a": _tensor_map_2d(
            A,
            dtype="float8_e4m3fn",
            global_shape=(M, K),
            global_stride=M,
            box_shape=(spec.load_block_m, spec.block_k),
            swizzle=spec.swizzle_a_mode,
        ),
        "tensor_map_b": _tensor_map_2d(
            B,
            dtype="float8_e4m3fn",
            global_shape=(N, K),
            global_stride=N,
            box_shape=(spec.load_block_n, spec.block_k),
            swizzle=spec.swizzle_b_mode,
        ),
        "tensor_map_sfa": _tensor_map_2d(
            SFA,
            dtype="uint32",
            global_shape=(_align_up(M, 16), scale_rows),
            global_stride=_align_up(M, 16) * 4,
            box_shape=(spec.block_m, 1),
        ),
        "tensor_map_sfb": _tensor_map_2d(
            SFB,
            dtype="uint32",
            global_shape=(_align_up(N, 16), scale_rows),
            global_stride=_align_up(N, 16) * 4,
            box_shape=(spec.block_n, 1),
        ),
        "tensor_map_cd": _tensor_map_2d(
            D,
            dtype="float32",
            global_shape=(N, M * G),
            global_stride=N * 4,
            box_shape=(spec.swizzle_cd_mode // 4, spec.store_block_m),
            swizzle=spec.swizzle_cd_mode,
        ),
    }
    return NumSimCase(
        kernel=module.get_kernel(**config),
        args=args,
        outputs={"D": "tensor_map_cd"},
        reference=lambda: {"D": expected.reshape(M * G, N).copy()},
        comparisons={"D": ComparisonSpec(rtol=0.0, atol=0.0)},
    )


def prepare_m_grouped_masked_fp8_case() -> NumSimCase:
    """Cover masked group row counts while excluding unwritten capacity rows."""

    module = load_tirx_kernel("deepgemm_sm100_m_grouped_fp8_gemm_masked")
    config = {
        "num_groups": 2,
        "expected_m_per_group": 16,
        "N": 128,
        "K": 128,
        "b_dtype": "fp8",
        "seed": 1,
        "num_sms": 2,
    }
    spec = module._spec_for(dict(config))
    from tirx_kernels.ported.deepgemm._sm100_fp8_fp4_gemm_1d1d import make_actual_ms

    G, M, N, K = config["num_groups"], module.MAX_M, config["N"], config["K"]
    masked_m = np.asarray(
        make_actual_ms(G, config["expected_m_per_group"], config["seed"]), dtype=np.int32
    )
    A = np.full((G, M, K), np.uint8(0x38), dtype=np.uint8)
    B = np.full((G, N, K), np.uint8(0x38), dtype=np.uint8)
    B[1, :, :] = np.uint8(0x40)
    scale_word = np.uint32(0x7F7F7F7F)
    SFA = np.full((G, 1, _align_up(M, 16)), scale_word, dtype=np.uint32)
    SFB = np.full((G, 1, _align_up(N, 16)), scale_word, dtype=np.uint32)
    D = np.zeros((G, M, N), dtype=np.uint16)
    expected_f32 = np.zeros((G, M, N), dtype=np.float32)
    expected_f32[0, : masked_m[0], :] = np.float32(K)
    expected_f32[1, : masked_m[1], :] = np.float32(2 * K)
    expected = _float32_to_bfloat16_bits(expected_f32)

    args = {
        "grouped_layout": masked_m,
        "grouped_len": np.int32(G),
        "shape_m": np.int32(M),
        "shape_n": np.int32(N),
        "shape_k": np.int32(K),
        "tensor_map_a": _tensor_map_2d(
            A,
            dtype="float8_e4m3fn",
            global_shape=(K, M * G),
            global_stride=K,
            box_shape=(spec.block_k, spec.load_block_m),
            swizzle=spec.swizzle_a_mode,
        ),
        "tensor_map_b": _tensor_map_2d(
            B,
            dtype="float8_e4m3fn",
            global_shape=(K, N * G),
            global_stride=K,
            box_shape=(spec.block_k, spec.load_block_n),
            swizzle=spec.swizzle_b_mode,
        ),
        "tensor_map_sfa": _tensor_map_2d(
            SFA,
            dtype="uint32",
            global_shape=(_align_up(M, 16), G),
            global_stride=_align_up(M, 16) * 4,
            box_shape=(spec.block_m, 1),
        ),
        "tensor_map_sfb": _tensor_map_2d(
            SFB,
            dtype="uint32",
            global_shape=(_align_up(N, 16), G),
            global_stride=_align_up(N, 16) * 4,
            box_shape=(spec.block_n, 1),
        ),
        "tensor_map_cd": _tensor_map_2d(
            D,
            dtype="bfloat16",
            global_shape=(N, M * G),
            global_stride=N * 2,
            box_shape=(spec.swizzle_cd_mode // 2, spec.store_block_m),
            swizzle=spec.swizzle_cd_mode,
        ),
    }
    regions = tuple(
        ComparisonRegion(
            actual=(slice(group * M, group * M + int(rows)), slice(None)),
            expected=(slice(group * M, group * M + int(rows)), slice(None)),
        )
        for group, rows in enumerate(masked_m)
    )
    return NumSimCase(
        kernel=module.get_kernel(**config),
        args=args,
        outputs={"D": "tensor_map_cd"},
        reference=lambda: {"D": expected.reshape(G * M, N).copy()},
        comparisons={"D": ComparisonSpec(rtol=0.0, atol=0.0, regions=regions)},
    )


_E2M1_POSITIVE_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _e2m1_bits_to_float32(values: Any) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint8) & np.uint8(0xF)
    magnitude = _E2M1_POSITIVE_VALUES[(bits & np.uint8(0x7)).astype(np.intp)]
    return np.where((bits & np.uint8(0x8)) != 0, -magnitude, magnitude).astype(np.float32)


def _float32_to_e2m1_bits(values: Any) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    magnitude = np.nan_to_num(np.abs(source), nan=np.float32(0.0), posinf=_E2M1_POSITIVE_VALUES[-1])
    upper = np.searchsorted(_E2M1_POSITIVE_VALUES, magnitude, side="left")
    upper = np.minimum(upper, len(_E2M1_POSITIVE_VALUES) - 1)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitude - _E2M1_POSITIVE_VALUES[lower]
    upper_distance = _E2M1_POSITIVE_VALUES[upper] - magnitude
    use_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    encoded = np.where(use_upper, upper, lower).astype(np.uint8)
    encoded |= np.signbit(source).astype(np.uint8) << np.uint8(3)
    return np.where(np.isnan(source), np.uint8(0x7), encoded).astype(np.uint8)


def _pack_e2m1(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim < 1 or values.shape[-1] % 2 != 0:
        raise ValueError("E2M1 packing requires an even final dimension")
    low = values[..., 0::2] & np.uint8(0xF)
    high = (values[..., 1::2] & np.uint8(0xF)) << np.uint8(4)
    return (low | high).astype(np.uint8)


def _unpack_e2m1(values: Any) -> np.ndarray:
    packed = np.asarray(values, dtype=np.uint8)
    result = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    result[..., 0::2] = packed & np.uint8(0xF)
    result[..., 1::2] = packed >> np.uint8(4)
    return result


def _sf_128x4_offsets(rows: int, sf_columns: int) -> np.ndarray:
    if rows % 128 != 0 or sf_columns % 4 != 0:
        raise ValueError("128x4 scale layout requires rows % 128 == 0 and columns % 4 == 0")
    row = np.arange(rows, dtype=np.int64)[:, None]
    column = np.arange(sf_columns, dtype=np.int64)[None, :]
    super_block_stride = (sf_columns // 4) * 512
    return (
        (row // 128) * super_block_stride
        + ((row % 128) // 32) * 4
        + (row % 32) * 16
        + (column // 4) * 512
        + column % 4
    )


def _pack_sf_128x4(logical: Any) -> np.ndarray:
    logical = np.asarray(logical, dtype=np.uint8)
    if logical.ndim != 2:
        raise ValueError("128x4 scale input must be rank 2")
    offsets = _sf_128x4_offsets(*logical.shape)
    physical = np.empty(logical.size, dtype=np.uint8)
    physical[offsets] = logical
    return physical.reshape(logical.shape)


def _unpack_sf_128x4(physical: Any) -> np.ndarray:
    physical = np.asarray(physical, dtype=np.uint8)
    if physical.ndim != 2:
        raise ValueError("128x4 scale input must be rank 2")
    offsets = _sf_128x4_offsets(*physical.shape)
    return physical.reshape(-1)[offsets]


def _nvfp4_reference(A_packed: Any, B_packed: Any, SFA: Any, SFB: Any, alpha: Any) -> np.ndarray:
    A_packed = np.asarray(A_packed, dtype=np.uint8)
    B_packed = np.asarray(B_packed, dtype=np.uint8)
    if A_packed.ndim != 2 or B_packed.ndim != 2 or A_packed.shape[1] != B_packed.shape[1]:
        raise ValueError("NVFP4 operands must be rank-2 matrices with the same packed K")
    K = A_packed.shape[1] * 2
    if K % 16 != 0:
        raise ValueError("NVFP4 scale vectors require K to be divisible by 16")
    sfa = _e4m3fn_bits_to_float32(_unpack_sf_128x4(SFA))
    sfb = _e4m3fn_bits_to_float32(_unpack_sf_128x4(SFB))
    a_dequant = (
        _e2m1_bits_to_float32(_unpack_e2m1(A_packed)).reshape(A_packed.shape[0], -1, 16)
        * sfa[:, :, None]
    ).reshape(A_packed.shape[0], K)
    b_dequant = (
        _e2m1_bits_to_float32(_unpack_e2m1(B_packed)).reshape(B_packed.shape[0], -1, 16)
        * sfb[:, :, None]
    ).reshape(B_packed.shape[0], K)
    scale = np.asarray(alpha, dtype=np.float32).reshape(-1)
    if scale.size != 1:
        raise ValueError("NVFP4 alpha must contain exactly one float32 value")
    return _float32_to_bfloat16_bits((a_dequant @ b_dequant.T) * scale[0])


def prepare_nvfp4_case(M: int, N: int, K: int, seed: int = 0) -> NumSimCase:
    if M % 256 != 0 or N % 256 != 0 or K % 256 != 0:
        raise ValueError("NumSim NVFP4 case requires M, N, and K to be divisible by 256")
    rng = np.random.default_rng(seed)
    A_bits = rng.integers(0, 16, size=(M, K), dtype=np.uint8)
    B_bits = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    A_packed = _pack_e2m1(A_bits)
    B_packed = _pack_e2m1(B_bits)
    scale_codes = np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8)
    SFA_logical = rng.choice(scale_codes, size=(M, K // 16)).astype(np.uint8)
    SFB_logical = rng.choice(scale_codes, size=(N, K // 16)).astype(np.uint8)
    SFA = _pack_sf_128x4(SFA_logical)
    SFB = _pack_sf_128x4(SFB_logical)
    alpha = np.array([0.25], dtype=np.float32)
    D = np.zeros((M, N), dtype=np.uint16)
    module = load_tirx_kernel("nvfp4_gemm")
    config = module._shape_config(M, N, K)
    cta_m = config["CTA_M"]
    cta_n = config["CTA_N"]
    cta_k = config["CTA_K"]
    epilogue_tile = config["EPI_TILE"]
    d_swizzle = {16: 32, 32: 64, 64: 128}[epilogue_tile]

    args = {
        "A_tensor_map": _tensor_map_2d(
            A_packed,
            dtype="uint8",
            global_shape=(K // 2, M),
            global_stride=K // 2,
            box_shape=(cta_k // 2, cta_m),
            swizzle=128,
        ),
        "B_tensor_map": _tensor_map_2d(
            B_packed,
            dtype="uint8",
            global_shape=(K // 2, N),
            global_stride=K // 2,
            box_shape=(cta_k // 2, cta_n),
            swizzle=128,
        ),
        "SFA_tensor_map": _tensor_map_3d(
            SFA.view(np.uint16),
            dtype="uint16",
            global_shape=(256, K // 64, M // 128),
            global_strides=(512, K * 8),
            box_shape=(256, 4, 1),
        ),
        "SFB_tensor_map": _tensor_map_3d(
            SFB.view(np.uint16),
            dtype="uint16",
            global_shape=(256, K // 64, N // 128),
            global_strides=(512, K * 8),
            box_shape=(256, 4, 1),
        ),
        "alpha": np.asarray(alpha, dtype=np.float32).reshape(1),
        "D_tensor_map": _tensor_map_2d(
            D,
            dtype="bfloat16",
            global_shape=(N, M),
            global_stride=N * 2,
            box_shape=(epilogue_tile, cta_m),
            swizzle=d_swizzle,
        ),
    }
    return NumSimCase(
        kernel=get_nvfp4_kernel(M=M, N=N, K=K),
        args=args,
        outputs={"D": "D_tensor_map"},
        reference=lambda: {"D": _nvfp4_reference(A_packed, B_packed, SFA, SFB, alpha)},
        comparisons={"D": ComparisonSpec(rtol=0.0, atol=0.0)},
    )


def prepare_cudnn_dense_blockscaled_amax_case() -> NumSimCase:
    """Mirror the canonical sparse FP8 oracle and column-major C backing."""

    M = N = K = 256
    config = {
        "M": M,
        "N": N,
        "K": K,
        "L": 1,
        "ab_dtype": "float8_e4m3fn",
        "sf_dtype": "float8_e8m0fnu",
        "sf_vec_size": 32,
        "c_dtype": "float32",
        "a_major": "k",
        "b_major": "k",
        "c_major": "m",
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
    }
    rows = np.arange(M, dtype=np.int64)
    a_k = (rows * 17) % K
    b_k = (rows * 23 + 6) % K
    a_values = 1 + rows % 2
    b_values = 1 + rows % 2
    fp8_codes = np.array([0x38, 0x40], dtype=np.uint8)

    a = np.zeros((M, K), dtype=np.uint8)
    b = np.zeros((N, K), dtype=np.uint8)
    a[rows, a_k] = fp8_codes[a_values - 1]
    b[rows, b_k] = fp8_codes[b_values - 1]
    expected_logical = (
        (a_k[:, None] == b_k[None, :]).astype(np.float32)
        * a_values[:, None].astype(np.float32)
        * b_values[None, :].astype(np.float32)
    )
    expected_c = np.ascontiguousarray(expected_logical.reshape(-1, order="F")).view(np.uint8)

    scale_bytes = 2 * 2 * 512
    c = np.zeros(M * N * 4, dtype=np.uint8)
    amax = np.zeros(1, dtype=np.float32)
    module = load_tirx_kernel("cudnn_sm100_dense_blockscaled_gemm_persistent_amax")
    return NumSimCase(
        kernel=module.get_kernel(**config),
        args={
            "a": a.reshape(-1),
            "b": b.reshape(-1),
            "sfa": np.full(scale_bytes, np.uint8(0x7F), dtype=np.uint8),
            "sfb": np.full(scale_bytes, np.uint8(0x7F), dtype=np.uint8),
            "c": c,
            "amax": amax,
        },
        outputs=("c", "amax"),
        reference=lambda: {
            "c": expected_c.copy(),
            "amax": np.array([expected_logical.max()], dtype=np.float32),
        },
        comparisons={
            "c": ComparisonSpec(rtol=0, atol=0),
            "amax": ComparisonSpec(rtol=0, atol=0),
        },
    )


__all__ = [
    "FP8_1D1D_NUM_SMS",
    "GROUPED_FP8_CONFIGS",
    "FP8_BLOCKWISE_CONFIGS",
    "FP16_BF16_CONFIGS",
    "NVFP4_CONFIGS",
    "prepare_bmm_fp8_rubin_case",
    "prepare_cudnn_dense_blockscaled_amax_case",
    "prepare_grouped_fp8_case",
    "prepare_fp8_bmm_case",
    "prepare_fp8_blockwise_case",
    "prepare_fp16_bf16_case",
    "prepare_k_grouped_fp8_case",
    "prepare_m_grouped_masked_fp8_case",
    "prepare_nvfp4_case",
]
