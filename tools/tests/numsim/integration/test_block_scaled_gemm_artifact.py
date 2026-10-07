from __future__ import annotations

import re

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tests.numsim.support.kernels import (
    block_scaled_fp8_gemm_packed_scales,
    block_scaled_nvfp4_gemm,
    block_scaled_nvfp4_gemm_cta_group2_scale_rows,
    fp8_scale_permute_tmem_roundtrip,
)
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module
from tirx_harness.numsim.transpiler.frontend import analyze
from tvm import tirx
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm_ffi import structural_map
from tvm.tirx.layout import (
    ComposeLayout,
    R,
    S,
    TCol,
    TileLayout,
    TLane,
    tmem_datapath_layout,
)
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout

_INVALID_SCALE_ROW_STRIDE = TileLayout(
    S[(4, 32, 2, 4) : (1 @ TCol, 1 @ TLane, 1 @ TCol, 0 @ TCol)] + R[4 : 32 @ TLane]
)
_FP8_MMA_128X128 = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (128, 128))
_FP8_MMA_8X128 = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_128B_ATOM, (8, 128))
_PACKED_FP4_MMA_128X32 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32))


@T.prim_func
def _block_scaled_fp8_dynamic_shared_stage(
    left: T.Buffer((2, 128, 128), "float8_e4m3fn"),
    right: T.Buffer((2, 8, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((8, 4), "float8_e8m0fnu"),
    stage: T.int32,
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (2, 128, 128),
        "float8_e4m3fn",
        scope="shared",
        layout=ComposeLayout(4, 3, 3, TileLayout(S[(1024,)])),
    )
    right_shared = T.alloc_buffer(
        (2, 8, 128),
        "float8_e4m3fn",
        scope="shared",
        layout=ComposeLayout(4, 3, 3, TileLayout(S[(1024,)])),
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1),
        allocated_addr=20,
    )
    if lane == 0:
        Tx.copy(left_shared[stage, :, :], left[stage, :, :])
        Tx.copy(right_shared[stage, :, :], right[stage, :, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[row, scale_index]
        for row in T.serial(8):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[row, scale_index]
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[stage, :, :],
            right_shared[stage, :, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def _block_scaled_runtime_instruction_descriptor(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((8, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((8, 4), "float8_e8m0fnu"),
    selector: T.int32,
    corrupt_descriptor: T.int32,
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_128X128
    )
    right_shared = T.alloc_buffer((8, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_8X128)
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1),
        allocated_addr=20,
    )
    descriptor: T.uint32
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[row, scale_index]
        for row in T.serial(8):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[row, scale_index]
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(descriptor),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=16,
            sfb_tmem_addr=20,
            M=128,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        if corrupt_descriptor != 0:
            descriptor = descriptor ^ T.uint32(1 << 17)
        T.cuda.runtime_instr_desc(T.address_of(descriptor), T.Cast("uint32", selector))
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            descI=descriptor,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def _block_scaled_invalid_scale_layout():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left = T.alloc_buffer((128, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_128X128)
    right = T.alloc_buffer((8, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_8X128)
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a = T.decl_buffer(
        (128, 8),
        "float8_e8m0fnu",
        scope="tmem",
        layout=_INVALID_SCALE_ROW_STRIDE,
        allocated_addr=16,
    )
    scale_b = T.decl_buffer(
        (128, 8),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=4),
        allocated_addr=24,
    )
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left[:, :],
            right[:, :],
            SFA=scale_a[:, :],
            SFB=scale_b[:, :],
            accum=False,
        )


@T.prim_func
def _block_scaled_interleaved_physical_streams(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((8, 128), "float8_e4m3fn"),
    scale_1_a: T.Buffer((128, 2), "float8_e8m0fnu"),
    scale_1_b: T.Buffer((8, 2), "float8_e8m0fnu"),
    scale_2_a: T.Buffer((128, 2), "float8_e8m0fnu"),
    scale_2_b: T.Buffer((8, 2), "float8_e8m0fnu"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_128X128
    )
    right_shared = T.alloc_buffer((8, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_8X128)
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_1_a_tmem = T.decl_buffer(
        (128, 8),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=4),
        allocated_addr=16,
    )
    scale_1_b_tmem = T.decl_buffer(
        (128, 8),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=4),
        allocated_addr=24,
    )
    scale_2_a_tmem = T.decl_buffer(
        (128, 8),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=4),
        allocated_addr=32,
    )
    scale_2_b_tmem = T.decl_buffer(
        (128, 8),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=2, sf_per_mma=1, sf_reuse=4),
        allocated_addr=40,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(128):
            for slot in T.serial(2):
                scale_1_a_tmem[row, slot * 4] = scale_1_a[row, slot]
                scale_2_a_tmem[row, slot * 4] = scale_2_a[row, slot]
        for row in T.serial(8):
            for slot in T.serial(2):
                scale_1_b_tmem[row, slot * 4] = scale_1_b[row, slot]
                scale_2_b_tmem[row, slot * 4] = scale_2_b[row, slot]
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_1_a_tmem[:, :],
            SFB=scale_1_b_tmem[:, :],
            accum=False,
        )
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_2_a_tmem[:, :],
            SFB=scale_2_b_tmem[:, :],
            accum=False,
        )
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_1_a_tmem[:, :],
            SFB=scale_1_b_tmem[:, :],
            accum=True,
        )
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_2_a_tmem[:, :],
            SFB=scale_2_b_tmem[:, :],
            accum=True,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def _block_scaled_scale_region_min(
    left: T.Buffer((128, 128), "float8_e4m3fn"),
    right: T.Buffer((8, 128), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((8, 4), "float8_e8m0fnu"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer(
        (128, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_128X128
    )
    right_shared = T.alloc_buffer((8, 128), "float8_e4m3fn", scope="shared", layout=_FP8_MMA_8X128)
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 16),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1, sf_reuse=4),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (128, 16),
        "float8_e8m0fnu",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=1, sf_reuse=4),
        allocated_addr=32,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index * 4] = scale_a[row, scale_index]
        for row in T.serial(8):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index * 4] = scale_b[row, scale_index]
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, 4:12],
            SFB=scale_b_tmem[:, 4:12],
            accum=False,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def _block_scaled_nvfp4_gemm_cta_group2_pair23(
    left_packed: T.Buffer((4, 128, 32), "uint8"),
    right_packed: T.Buffer((4, 128, 32), "uint8"),
    scale_a: T.Buffer((4, 128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((4, 256, 4), "float8_e4m3fn"),
    output: T.Buffer((4, 128, 256), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([4])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32
    )
    right_shared_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_PACKED_FP4_MMA_128X32
    )
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    accumulator = T.decl_buffer(
        (128, 256),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 256),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=4, sf_per_mma=4),
        allocated_addr=256,
    )
    scale_b_tmem = T.decl_buffer(
        (256, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(256, SF_K=4, sf_per_mma=4),
        allocated_addr=264,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[cta, :, :])
        Tx.copy(right_shared_packed[:, :], right_packed[cta, :, :])
        for row in T.serial(128):
            for scale_index in T.serial(4):
                scale_a_tmem[row, scale_index] = scale_a[cta, row, scale_index]
        for row in T.serial(256):
            for scale_index in T.serial(4):
                scale_b_tmem[row, scale_index] = scale_b[cta, row, scale_index]
    T.cuda.cluster_sync()
    if (cta == 2) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if (cta >= 2) and (lane == 0):
        for row in T.serial(128):
            for col in T.serial(256):
                output[cta, row, col] = accumulator[row, col]


def _gemm_async_variants(func) -> list[tuple[object, list[str]]]:
    """``(node, Gemm generic arguments)`` of every ``gemm_async`` instruction the module emits."""

    spec = analyze(func)
    source = emit_rust_module(spec, func)
    variants = []
    for entry in spec.kernels[0].source_map:
        if entry.kind != "TilePrimitiveCall" or str(entry.node.op.name) != "tirx.tile.gemm_async":
            continue
        (variant,) = re.findall(
            r"v2::tile::gemm_async\w*::<v2::tile::variant::Gemm<(.*)>>\(warp, v2_context\(ctx\), "
            rf"v2::SiteId::new\({entry.op_id}_u64\)",
            source,
        )
        variants.append((entry.node, variant.split(", ")))
    return variants


def _gemm_async_variant(variants, node) -> list[str]:
    return next(variant for candidate, variant in variants if candidate.same_as(node))


def _decode_e4m3(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (bits & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    result = np.where((bits & np.uint8(0x80)) != 0, -magnitude, magnitude)
    return np.where((exponent == 15) & ((bits & 7) == 7), np.nan, result).astype(np.float32)


def _decode_e8m0(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    return np.exp2(bits.astype(np.int16) - 127).astype(np.float32)


def _decode_e2m1(bits: np.ndarray) -> np.ndarray:
    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    bits = np.asarray(bits, dtype=np.uint8) & np.uint8(0xF)
    magnitude = values[(bits & np.uint8(0x7)).astype(np.intp)]
    return np.where((bits & np.uint8(0x8)) != 0, -magnitude, magnitude).astype(np.float32)


def _unpack_e2m1(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    result = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    result[..., 0::2] = packed & np.uint8(0xF)
    result[..., 1::2] = packed >> np.uint8(4)
    return result


def _expected_fp8_extent8_invocation(
    left: np.ndarray,
    right: np.ndarray,
    scale_a: np.ndarray,
    scale_b: np.ndarray,
    *,
    scale_indices: tuple[int, int] = (0, 1),
) -> np.ndarray:
    a = _decode_e4m3(left)
    b = _decode_e4m3(right)
    a_scales = _decode_e8m0(scale_a)
    b_scales = _decode_e8m0(scale_b)
    result = np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    for k_slice, scale_index in zip((slice(0, 64), slice(64, 128)), scale_indices, strict=True):
        result += (a[:, k_slice] * a_scales[:, scale_index, None]) @ (
            b[:, k_slice] * b_scales[:, scale_index, None]
        ).T
    return result


def _expected_fp8_per_ki_scales(
    left: np.ndarray, right: np.ndarray, scale_a: np.ndarray, scale_b: np.ndarray
) -> np.ndarray:
    a = _decode_e4m3(left)
    b = _decode_e4m3(right)
    a_scales = _decode_e8m0(scale_a)
    b_scales = _decode_e8m0(scale_b)
    result = np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    for scale_index, start in enumerate(range(0, left.shape[1], 32)):
        k_slice = slice(start, start + 32)
        result += (a[:, k_slice] * a_scales[:, scale_index, None]) @ (
            b[:, k_slice] * b_scales[:, scale_index, None]
        ).T
    return result


def _duplicate_block_scaled_gemm_node(func):
    duplicated = False
    variants = _gemm_async_variants(func)

    def rewrite(node):
        nonlocal duplicated
        if type(node).__name__ != "TilePrimitiveCall" or duplicated:
            return node
        if str(node.op.name) != "tirx.tile.gemm_async":
            return node
        mode = _gemm_async_variant(variants, node)[0]
        if not mode.startswith("v2::tile::variant::BlockScaled"):
            return node
        duplicated = True
        return tirx.SeqStmt([node, node.replace()])

    body = structural_map(func.body, (tirx.TilePrimitiveCall, rewrite))
    assert duplicated
    return func.with_body(body)


def _append_accumulating_block_scaled_gemm(func):
    appended = False
    variants = _gemm_async_variants(func)

    def rewrite(node):
        nonlocal appended
        if type(node).__name__ != "TilePrimitiveCall" or appended:
            return node
        if str(node.op.name) != "tirx.tile.gemm_async":
            return node
        mode = _gemm_async_variant(variants, node)[0]
        if not mode.startswith("v2::tile::variant::BlockScaled"):
            return node
        appended = True
        arguments = (*node.args[:-1], True)
        return tirx.SeqStmt([node, node.replace(args=arguments)])

    body = structural_map(func.body, (tirx.TilePrimitiveCall, rewrite))
    assert appended
    return func.with_body(body)


def test_block_scaled_gemm_normalizes_instruction_kind_and_shape() -> None:
    fp8 = _gemm_async_variants(block_scaled_fp8_gemm_packed_scales)
    nvfp4 = _gemm_async_variants(block_scaled_nvfp4_gemm)

    # Gemm<mode, A, B, placement, access, m, n, k, instruction_m, instruction_n, ...>
    assert fp8
    assert {(*variant[:3], *variant[8:10]) for _node, variant in fp8} == {
        (
            "v2::tile::variant::BlockScaled<v2::tile::variant::E8m0>",
            "v2::tile::variant::E4m3",
            "v2::tile::variant::E4m3",
            "128",
            "8",
        )
    }
    assert nvfp4
    assert {(*variant[:3], *variant[8:10]) for _node, variant in nvfp4} == {
        (
            "v2::tile::variant::BlockScaled<v2::tile::variant::E4m3>",
            "v2::tile::variant::E2m1",
            "v2::tile::variant::E2m1",
            "128",
            "8",
        )
    }


def test_block_scaled_desc_i_rotates_scale_bytes_for_each_ki(tmp_path):
    left = np.resize(np.array([0x30, 0x38, 0x40, 0xB8], dtype=np.uint8), (128, 128))
    right = np.resize(np.array([0x28, 0x38, 0x40], dtype=np.uint8), (8, 128))
    scale_a = np.tile(np.array([125, 126, 127, 128], dtype=np.uint8), (128, 1))
    scale_b = np.tile(np.array([128, 127, 126, 125], dtype=np.uint8), (8, 1))

    module = numsim.transpile(_block_scaled_runtime_instruction_descriptor, cache_dir=tmp_path)
    for selector in range(4):
        output = np.zeros((128, 8), dtype=np.float32)
        result = numsim.Engine().run(
            module,
            {
                "left": left,
                "right": right,
                "scale_a": scale_a,
                "scale_b": scale_b,
                "selector": selector,
                "corrupt_descriptor": 0,
                "output": output,
            },
        )
        expected = _expected_fp8_per_ki_scales(left, right, scale_a, scale_b)
        np.testing.assert_array_equal(result.outputs["output"], expected)


def test_block_scaled_desc_i_rejects_static_abi_mismatch_at_runtime(tmp_path):
    left = np.full((128, 128), 0x38, dtype=np.uint8)
    right = np.full((8, 128), 0x38, dtype=np.uint8)
    scale_a = np.full((128, 4), 127, dtype=np.uint8)
    scale_b = np.full((8, 4), 127, dtype=np.uint8)
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(_block_scaled_runtime_instruction_descriptor, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match="does not match the typed TCGEN ABI"):
        numsim.Engine().run(
            module,
            {
                "left": left,
                "right": right,
                "scale_a": scale_a,
                "scale_b": scale_b,
                "selector": 0,
                "corrupt_descriptor": 1,
                "output": output,
            },
        )


def test_fp8_scale_permute_and_sf_reuse_tmem_roundtrip(tmp_path):
    rows = np.arange(128, dtype=np.uint16)[:, None]
    groups = np.arange(4, dtype=np.uint16)[None, :]
    scale_bits = (96 + (rows * 5 + groups * 17) % 48).astype(np.uint8)
    words = (
        scale_bits[:, 0].astype(np.uint32)
        | (scale_bits[:, 1].astype(np.uint32) << np.uint32(8))
        | (scale_bits[:, 2].astype(np.uint32) << np.uint32(16))
        | (scale_bits[:, 3].astype(np.uint32) << np.uint32(24))
    )
    shared_output = np.zeros((128, 4), dtype=np.float32)
    tmem_output = np.zeros((128, 4), dtype=np.float32)

    module = numsim.transpile(fp8_scale_permute_tmem_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": words, "shared_output": shared_output, "tmem_output": tmem_output}
    )

    expected = _decode_e8m0(scale_bits)
    np.testing.assert_array_equal(result.outputs["shared_output"], expected)
    np.testing.assert_array_equal(result.outputs["tmem_output"], expected)


def test_fp8_block_scaled_gemm_derives_scale_columns_from_each_invocation(tmp_path):
    left_codes = np.array([0x30, 0x38, 0x3C, 0x40, 0xB8, 0xBC], dtype=np.uint8)
    right_codes = np.array([0x28, 0x38, 0x40, 0xB0, 0xB8], dtype=np.uint8)
    left = np.resize(left_codes, (128, 128))
    right = np.resize(right_codes, (8, 128))
    scale_a = np.empty((128, 2), dtype=np.uint8)
    scale_a[:, 0] = np.where(np.arange(128) % 2 == 0, 127, 128)
    scale_a[:, 1] = np.where(np.arange(128) % 3 == 0, 126, 127)
    scale_b = np.resize(
        np.array([[127, 128], [128, 127], [126, 129], [129, 126]], dtype=np.uint8), (8, 2)
    )
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(block_scaled_fp8_gemm_packed_scales, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left": left,
            "right": right,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": output,
        },
    )

    expected_once = _expected_fp8_extent8_invocation(left, right, scale_a, scale_b)
    np.testing.assert_array_equal(result.outputs["output"], 2 * expected_once)


def test_fp8_snapshot_gather_honors_a_dynamic_shared_stage(tmp_path):
    left = np.full((2, 128, 128), 0x38, dtype=np.uint8)
    left[1] = np.uint8(0x40)
    right = np.full((2, 8, 128), 0x38, dtype=np.uint8)
    scale_a = np.full((128, 4), 127, dtype=np.uint8)
    scale_b = np.full((8, 4), 127, dtype=np.uint8)
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(_block_scaled_fp8_dynamic_shared_stage, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left": left,
            "right": right,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "stage": 1,
            "output": output,
        },
    )

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.full((128, 8), 256.0, dtype=np.float32),
    )


def test_repeated_block_scaled_callsites_inside_a_loop_do_not_share_history(tmp_path):
    duplicated = _duplicate_block_scaled_gemm_node(block_scaled_fp8_gemm_packed_scales)

    left = np.resize(np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), (128, 128))
    right = np.resize(np.array([0x28, 0x38, 0x40], dtype=np.uint8), (8, 128))
    scale_a = np.tile(np.array([127, 129], dtype=np.uint8), (128, 1))
    scale_b = np.tile(np.array([128, 126], dtype=np.uint8), (8, 1))
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(duplicated, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left": left,
            "right": right,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": output,
        },
    )

    expected_once = _expected_fp8_extent8_invocation(left, right, scale_a, scale_b)
    np.testing.assert_array_equal(result.outputs["output"], 3 * expected_once)


def test_interleaved_block_scaled_calls_are_independent_of_prior_calls(tmp_path):
    left = np.full((128, 128), 0x38, dtype=np.uint8)
    right = np.full((8, 128), 0x38, dtype=np.uint8)
    scale_1_a = np.tile(np.array([127, 128], dtype=np.uint8), (128, 1))
    scale_1_b = np.tile(np.array([127, 128], dtype=np.uint8), (8, 1))
    scale_2_a = np.tile(np.array([129, 130], dtype=np.uint8), (128, 1))
    scale_2_b = np.tile(np.array([129, 130], dtype=np.uint8), (8, 1))
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(_block_scaled_interleaved_physical_streams, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left": left,
            "right": right,
            "scale_1_a": scale_1_a,
            "scale_1_b": scale_1_b,
            "scale_2_a": scale_2_a,
            "scale_2_b": scale_2_b,
            "output": output,
        },
    )

    first = _expected_fp8_extent8_invocation(left, right, scale_1_a, scale_1_b)
    second = _expected_fp8_extent8_invocation(left, right, scale_2_a, scale_2_b)
    np.testing.assert_array_equal(result.outputs["output"], first + 2 * second)


def test_block_scale_layout_that_violates_instruction_row_stride_fails_closed(tmp_path):
    with pytest.raises(UnsupportedTIRxError, match="instruction ABI"):
        numsim.transpile(_block_scaled_invalid_scale_layout, cache_dir=tmp_path)


def test_block_scale_region_min_selects_the_physical_scale_coordinates(tmp_path):
    left = np.resize(np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), (128, 128))
    right = np.resize(np.array([0x28, 0x38, 0x40], dtype=np.uint8), (8, 128))
    scale_a = np.tile(np.array([126, 127, 129, 130], dtype=np.uint8), (128, 1))
    scale_b = np.tile(np.array([130, 128, 126, 127], dtype=np.uint8), (8, 1))
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(_block_scaled_scale_region_min, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left": left,
            "right": right,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": output,
        },
    )

    expected = _expected_fp8_extent8_invocation(left, right, scale_a, scale_b, scale_indices=(1, 2))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_nvfp4_block_scaled_gemm_decodes_nibbles_and_e4m3_scales(tmp_path):
    left_codes = np.resize(
        np.array([0x0, 0x1, 0x2, 0x3, 0x7, 0x9, 0xA, 0xF], dtype=np.uint8), (128, 64)
    )
    right_codes = np.resize(np.array([0x1, 0x2, 0x4, 0x7, 0x9, 0xB], dtype=np.uint8), (8, 64))
    left_packed = (left_codes[:, 0::2] | (left_codes[:, 1::2] << np.uint8(4))).astype(np.uint8)
    right_packed = (right_codes[:, 0::2] | (right_codes[:, 1::2] << np.uint8(4))).astype(np.uint8)
    scale_a = np.resize(np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), (128, 4))
    scale_b = np.resize(
        np.array(
            [
                [0x38, 0x40, 0x30, 0x3C],
                [0x40, 0x38, 0x3C, 0x30],
                [0x30, 0x3C, 0x38, 0x40],
                [0x3C, 0x30, 0x40, 0x38],
            ],
            dtype=np.uint8,
        ),
        (8, 4),
    )
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(block_scaled_nvfp4_gemm, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left_packed": left_packed,
            "right_packed": right_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": output,
        },
    )

    a = (
        _decode_e2m1(_unpack_e2m1(left_packed)).reshape(128, 4, 16)
        * _decode_e4m3(scale_a)[:, :, None]
    ).reshape(128, 64)
    b = (
        _decode_e2m1(_unpack_e2m1(right_packed)).reshape(8, 4, 16)
        * _decode_e4m3(scale_b)[:, :, None]
    ).reshape(8, 64)
    np.testing.assert_array_equal(result.outputs["output"], a @ b.T)


def test_cta_group2_right_scales_use_the_combined_n_row(tmp_path):
    left_packed = np.full((2, 128, 32), 0x22, dtype=np.uint8)
    right_packed = np.full((2, 128, 32), 0x22, dtype=np.uint8)
    scale_a = np.full((2, 128, 4), 0x38, dtype=np.uint8)
    scale_b = np.full((2, 256, 4), 0x38, dtype=np.uint8)
    scale_b[:, 128:, :] = np.uint8(0x40)
    output = np.zeros((2, 128, 256), dtype=np.float32)

    module = numsim.transpile(block_scaled_nvfp4_gemm_cta_group2_scale_rows, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left_packed": left_packed,
            "right_packed": right_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": output,
        },
    )

    expected = np.full((2, 128, 256), 64.0, dtype=np.float32)
    expected[:, :, 128:] = np.float32(128.0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cta_group2_batched_gemm_accumulates_both_target_ctas(tmp_path):
    accumulated = _append_accumulating_block_scaled_gemm(
        block_scaled_nvfp4_gemm_cta_group2_scale_rows
    )
    left_packed = np.full((2, 128, 32), 0x22, dtype=np.uint8)
    right_packed = np.full((2, 128, 32), 0x22, dtype=np.uint8)
    scale_a = np.full((2, 128, 4), 0x38, dtype=np.uint8)
    scale_b = np.full((2, 256, 4), 0x38, dtype=np.uint8)
    scale_b[:, 128:, :] = np.uint8(0x40)
    output = np.zeros((2, 128, 256), dtype=np.float32)

    module = numsim.transpile(accumulated, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left_packed": left_packed,
            "right_packed": right_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": output,
        },
    )

    expected = np.full((2, 128, 256), 128.0, dtype=np.float32)
    expected[:, :, 128:] = np.float32(256.0)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cta_group2_uses_the_issuing_ctas_pair_2_and_3(tmp_path):
    left_packed = np.full((4, 128, 32), 0x77, dtype=np.uint8)
    right_packed = np.full((4, 128, 32), 0x77, dtype=np.uint8)
    left_packed[2:] = np.uint8(0x22)
    right_packed[2:] = np.uint8(0x22)
    scale_a = np.full((4, 128, 4), 0x40, dtype=np.uint8)
    scale_b = np.full((4, 256, 4), 0x40, dtype=np.uint8)
    scale_a[2:] = np.uint8(0x38)
    scale_b[2:] = np.uint8(0x38)
    scale_b[3, 128:, :] = np.uint8(0x40)
    output = np.zeros((4, 128, 256), dtype=np.float32)

    module = numsim.transpile(_block_scaled_nvfp4_gemm_cta_group2_pair23, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left_packed": left_packed,
            "right_packed": right_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "output": output,
        },
    )

    expected = np.zeros((4, 128, 256), dtype=np.float32)
    expected[2:, :, :128] = np.float32(64.0)
    expected[2:, :, 128:] = np.float32(128.0)
    np.testing.assert_array_equal(result.outputs["output"], expected)
