from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tests.numsim.support.kernels import (
    dense_fp8_gemm_async_cta1,
    dense_gemm_async_dynamic_right_index,
    dense_gemm_async_cta1,
    dense_gemm_async_cta_group2,
    dense_gemm_async_tmem_a_cta_group2,
    dense_gemm_async_tmem_a_transposed_b,
    dense_gemm_async_two_clusters,
    inactive_gemm_async_is_noop,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import (
    ComposeLayout,
    S,
    TCol,
    TileLayout,
    TLane,
    tmem_datapath_layout,
)
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout

_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_MMA_TF32_32B = ComposeLayout(2, 1, 3, TileLayout(S[(64,)]))
_MMA_BF16_64X64 = mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_32B_ATOM, (64, 64))
_MMA_F16_NONE_128X16 = mma_shared_layout("float16", SwizzleMode.SWIZZLE_NONE, (128, 16))
_MMA_F16_NONE_8X16 = mma_shared_layout("float16", SwizzleMode.SWIZZLE_NONE, (8, 16))

_TMEM_WRONG_F_GROUP_ORDER = TileLayout(
    S[(2, 2, 16, 8) : (32 @ TLane, 64 @ TLane, 1 @ TLane, 1 @ TCol)]
)
_TMEM_WRONG_B_ROW_COLUMN_GROUP = TileLayout(
    S[(2, 32, 2, 8) : (64 @ TLane, 1 @ TLane, 32 @ TLane, 1 @ TCol)]
)
_TMEM_WRONG_A_TRANSPOSED_AXES = TileLayout(S[(128, 16) : (1 @ TCol, 1 @ TLane)])
_TMEM_CTA2_BANKED_A = TileLayout(S[(2, 64, 16) : (64 @ TLane, 1 @ TLane, 1 @ TCol)])


@T.prim_func
def dense_gemm_async_no_swizzle_shared(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((128, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_NONE_128X16)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_NONE_8X16)
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_two_thread_issuers(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((1,), "int32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
    T.cuda.warp_sync()
    if lane < 2:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
    if lane == 0:
        output[0] = 1


@T.prim_func
def dense_gemm_async_wrong_tmem_a_layout():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_tmem = T.decl_buffer(
        (128, 16),
        "float16",
        scope="tmem",
        layout=_TMEM_WRONG_A_TRANSPOSED_AXES,
        allocated_addr=0,
    )
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=16,
    )
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_tmem[:, :],
            right_shared[:, :],
            transB=True,
            accum=False,
        )


@T.prim_func
def dense_gemm_async_declared_geometry_mismatch():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            mma_m=64,
            mma_n=8,
        )


@T.prim_func
def dense_gemm_async_m64_layout_f(
    left: T.Buffer((64, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(64):
            for col in T.serial(8):
                accumulator[row, col] = T.float32(-777)
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=True)
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_bf16_m64_layout_f(
    left: T.Buffer((64, 16), "bfloat16"),
    right: T.Buffer((8, 16), "bfloat16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "bfloat16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "bfloat16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_m64_packed_layout_e_inferred(
    left: T.Buffer((64, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("E", 64, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_bf16_large_library_path(
    left: T.Buffer((64, 64), "bfloat16"),
    right: T.Buffer((64, 64), "bfloat16"),
    output: T.Buffer((64, 64), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", layout=_MMA_BF16_64X64)
    right_shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", layout=_MMA_BF16_64X64)
    accumulator = T.decl_buffer(
        (64, 64),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 64),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
        for row in T.serial(64):
            for col in T.serial(64):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_tf32_is_rejected():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 8), "float32", scope="shared", layout=_MMA_TF32_32B)
    right_shared = T.alloc_buffer((16, 8), "float32", scope="shared", layout=_MMA_TF32_32B)
    accumulator = T.decl_buffer(
        (128, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 16),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)


@T.prim_func
def dense_gemm_async_accumulation_rounding(
    left: T.Buffer((64, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 8),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(64):
            for col in T.serial(8):
                accumulator[row, col] = T.float32(16777216)
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=True)
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_m64_wrong_f_group_order(
    left: T.Buffer((64, 16), "float16"),
    right: T.Buffer((8, 16), "float16"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=_TMEM_WRONG_F_GROUP_ORDER,
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[:, :])
        Tx.copy(right_shared[:, :], right[:, :])
        for row in T.serial(64):
            for col in T.serial(8):
                accumulator[row, col] = T.float32(-777)
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(accumulator[:, :], left_shared[:, :], right_shared[:, :], accum=False)
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_m64_cta2_layout_b(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 16),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
        Tx.copy(right_shared[:, :], right[cta * 8 : (cta + 1) * 8, :])
        for row in T.serial(64):
            for col in T.serial(16):
                accumulator[row, col] = T.float32(-777)
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            cta_group=2,
        )
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=True,
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(16):
                output[cta * 64 + row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_m64_cta2_wrong_b_grouping(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((16, 16), "float16"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 16),
        "float32",
        scope="tmem",
        layout=_TMEM_WRONG_B_ROW_COLUMN_GROUP,
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
        Tx.copy(right_shared[:, :], right[cta * 8 : (cta + 1) * 8, :])
        for row in T.serial(64):
            for col in T.serial(16):
                accumulator[row, col] = T.float32(-777)
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(16):
                output[cta * 64 + row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_m64_cta2_banked_a(
    left: T.Buffer((2, 2, 64, 16), "float16"),
    right: T.Buffer((2, 16, 16), "float16"),
    output: T.Buffer((2, 64, 32), "float32"),
):
    """CTA2 Layout-B form whose two A banks select the two B shards."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_tmem = T.decl_buffer(
        (2, 64, 16),
        "float16",
        scope="tmem",
        layout=_TMEM_CTA2_BANKED_A,
        allocated_addr=0,
    )
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=32,
    )
    if lane == 0:
        Tx.copy(right_shared[:, :], right[cta, :, :])
        for bank in T.serial(2):
            for row in T.serial(64):
                for col in T.serial(16):
                    left_tmem[bank, row, col] = left[cta, bank, row, col]
    T.cuda.cluster_sync()
    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_tmem[:, :, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(32):
                output[cta, row, col] = accumulator[row, col]


@T.prim_func
def dense_gemm_async_two_cta_pairs_in_one_cluster(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 16), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([4])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 16),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 16),
        allocated_addr=0,
    )
    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
        right_start: T.let = (cta // 2) * 16 + (cta % 2) * 8
        Tx.copy(right_shared[:, :], right[right_start : right_start + 8, :])
    T.cuda.cluster_sync()
    if ((cta % 2) == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            accum=False,
            cta_group=2,
        )
    T.cuda.cluster_sync()
    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(16):
                output[cta * 64 + row, col] = accumulator[row, col]


def test_dense_gemm_async_gathers_physical_operands_and_accumulates_tmem(tmp_path):
    left = (np.arange(128 * 16, dtype=np.float32).reshape(128, 16) % 17 - 8).astype(np.float16)
    right = (np.arange(8 * 16, dtype=np.float32).reshape(8, 16) % 11 - 5).astype(np.float16)
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(dense_gemm_async_cta1, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"left": left, "right": right, "output": output})

    expected = np.matmul(left.astype(np.float32), right.astype(np.float32).T) * np.float32(2)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_dense_fp8_gemm_async_matches_numpy(tmp_path):
    fp8 = pytest.importorskip("ml_dtypes").float8_e4m3fn
    left = (np.arange(128 * 128, dtype=np.float32).reshape(128, 128) % 5 - 2).astype(fp8)
    right = (np.arange(8 * 128, dtype=np.float32).reshape(8, 128) % 5 - 2).astype(fp8)
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(dense_fp8_gemm_async_cta1, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"left": left, "right": right, "output": output})

    expected = np.matmul(left.astype(np.float32), right.astype(np.float32).T)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_dense_gemm_async_starts_the_fma_chain_from_input_d(tmp_path):
    left = np.zeros((64, 16), dtype=np.float16)
    right = np.zeros((8, 16), dtype=np.float16)
    left[:, 0] = np.float16(-4096)
    left[:, 1] = np.float16(1)
    right[:, 0] = np.float16(4096)
    right[:, 1] = np.float16(0.5)

    module = numsim.transpile(dense_gemm_async_accumulation_rounding, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"left": left, "right": right, "output": np.zeros((64, 8), dtype=np.float32)},
    )

    np.testing.assert_array_equal(result.outputs["output"], np.full((64, 8), 0.5, np.float32))


def test_dense_gemm_async_no_swizzle_descriptor_matches_numpy(tmp_path):
    left = (np.arange(128 * 16, dtype=np.float32).reshape(128, 16) % 13 - 6).astype(np.float16)
    right = (np.arange(8 * 16, dtype=np.float32).reshape(8, 16) % 7 - 3).astype(np.float16)
    module = numsim.transpile(dense_gemm_async_no_swizzle_shared, cache_dir=tmp_path)

    result = numsim.Engine().run(
        module,
        {"left": left, "right": right, "output": np.zeros((128, 8), dtype=np.float32)},
    )

    np.testing.assert_array_equal(
        result.outputs["output"],
        np.matmul(left.astype(np.float32), right.astype(np.float32).T),
    )


def test_thread_scope_gemm_async_requires_one_runtime_issuer(tmp_path):
    module = numsim.transpile(dense_gemm_async_two_thread_issuers, cache_dir=tmp_path)
    with pytest.raises(numsim.NumSimExecutionError, match="exactly one active issuing lane"):
        numsim.Engine().run(
            module,
            {
                "left": np.zeros((128, 16), dtype=np.float16),
                "right": np.zeros((8, 16), dtype=np.float16),
                "output": np.zeros((1,), dtype=np.int32),
            },
        )


def test_dense_gemm_async_rejects_wrong_tmem_a_layout(tmp_path):
    with pytest.raises(numsim.UnsupportedTIRxError, match="TMEM A layout"):
        numsim.transpile(dense_gemm_async_wrong_tmem_a_layout, cache_dir=tmp_path)


def test_dense_gemm_async_rejects_declared_geometry_that_disagrees_with_operands(tmp_path):
    with pytest.raises(numsim.UnsupportedTIRxError, match="mma_m=64 disagrees"):
        numsim.transpile(dense_gemm_async_declared_geometry_mismatch, cache_dir=tmp_path)


def test_m64_tcgen_mma_uses_layout_f_independently_of_declared_tmem_layout(tmp_path):
    left = (np.arange(64 * 16, dtype=np.float32).reshape(64, 16) % 17 - 8).astype(np.float16)
    right = (np.arange(8 * 16, dtype=np.float32).reshape(8, 16) % 11 - 5).astype(np.float16)
    product = np.matmul(left.astype(np.float32), right.astype(np.float32).T)
    expected = product * np.float32(2)

    correct_module = numsim.transpile(
        dense_gemm_async_m64_layout_f, cache_dir=tmp_path / "layout_f"
    )
    correct = numsim.Engine().run(
        correct_module,
        {"left": left, "right": right, "output": np.zeros((64, 8), dtype=np.float32)},
    )
    np.testing.assert_array_equal(correct.outputs["output"], expected)
    wrong = numsim.Engine().run(
        numsim.transpile(dense_gemm_async_m64_wrong_f_group_order, cache_dir=tmp_path / "wrong_f"),
        {"left": left, "right": right, "output": np.zeros((64, 8), dtype=np.float32)},
    )

    def check_wrong_layout_f() -> None:
        assert not np.array_equal(wrong.outputs["output"], product)
        np.testing.assert_array_equal(wrong.outputs["output"][:16], product[:16])
        np.testing.assert_array_equal(wrong.outputs["output"][16:32], product[32:48])
        np.testing.assert_array_equal(wrong.outputs["output"][32:48], product[16:32])
        np.testing.assert_array_equal(wrong.outputs["output"][48:], product[48:])

    check_wrong_layout_f()


def test_bf16_m64_tcgen_mma_uses_layout_f(tmp_path):
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    left = (np.arange(64 * 16, dtype=np.float32).reshape(64, 16) % 9 - 4).astype(bf16)
    right = (np.arange(8 * 16, dtype=np.float32).reshape(8, 16) % 7 - 3).astype(bf16)
    result = numsim.Engine().run(
        numsim.transpile(dense_gemm_async_bf16_m64_layout_f, cache_dir=tmp_path),
        {"left": left, "right": right, "output": np.zeros((64, 8), dtype=np.float32)},
    )
    expected = np.matmul(left.astype(np.float32), right.astype(np.float32).T)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_m64_tcgen_mma_infers_weight_stationary_from_packed_layout_e(tmp_path):
    left = (np.arange(64 * 16, dtype=np.float32).reshape(64, 16) % 9 - 4).astype(np.float16)
    right = (np.arange(8 * 16, dtype=np.float32).reshape(8, 16) % 7 - 3).astype(np.float16)

    result = numsim.Engine().run(
        numsim.transpile(dense_gemm_async_m64_packed_layout_e_inferred, cache_dir=tmp_path),
        {"left": left, "right": right, "output": np.zeros((64, 8), dtype=np.float32)},
    )

    np.testing.assert_array_equal(
        result.outputs["output"], np.matmul(left.astype(np.float32), right.astype(np.float32).T)
    )


def test_large_bf16_gemm_async_uses_one_engine_gemm(tmp_path):
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    left = (np.arange(64 * 64, dtype=np.float32).reshape(64, 64) % 5 - 2).astype(bf16)
    right = (np.arange(64 * 64, dtype=np.float32).reshape(64, 64) % 7 - 3).astype(bf16)
    module = numsim.transpile(dense_gemm_async_bf16_large_library_path, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {"left": left, "right": right, "output": np.zeros((64, 64), dtype=np.float32)},
    )

    expected = np.matmul(left.astype(np.float32), right.astype(np.float32).T)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_tile_tf32_gemm_async_fails_closed(tmp_path):
    with pytest.raises(
        numsim.UnmodeledTIRxFormError,
        match="A shared dtype float32 has no supported TCGEN descriptor element width|dense dtype float32",
    ):
        numsim.transpile(dense_gemm_async_tf32_is_rejected, cache_dir=tmp_path)


def test_cta_group2_m64_tcgen_mma_uses_layout_b_independently_of_declared_layout(tmp_path):
    left = (np.arange(128 * 16, dtype=np.float32).reshape(128, 16) % 19 - 9).astype(np.float16)
    right = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 13 - 6).astype(np.float16)
    product = np.matmul(left.astype(np.float32), right.astype(np.float32).T)
    expected = product * np.float32(2)

    correct_module = numsim.transpile(
        dense_gemm_async_m64_cta2_layout_b, cache_dir=tmp_path / "layout_b"
    )
    correct = numsim.Engine().run(
        correct_module,
        {"left": left, "right": right, "output": np.zeros((128, 16), dtype=np.float32)},
    )
    np.testing.assert_array_equal(correct.outputs["output"], expected)
    wrong = numsim.Engine().run(
        numsim.transpile(
            dense_gemm_async_m64_cta2_wrong_b_grouping, cache_dir=tmp_path / "wrong_b"
        ),
        {"left": left, "right": right, "output": np.zeros((128, 16), dtype=np.float32)},
    )

    def check_wrong_layout_b() -> None:
        assert not np.array_equal(wrong.outputs["output"], product)
        for cta in range(2):
            tile = wrong.outputs["output"][cta * 64 : (cta + 1) * 64]
            expected_tile = product[cta * 64 : (cta + 1) * 64]
            np.testing.assert_array_equal(tile[:32, :8], expected_tile[:32, :8])
            np.testing.assert_array_equal(tile[:32, 8:], expected_tile[32:, :8])
            np.testing.assert_array_equal(tile[32:, :8], expected_tile[:32, 8:])
            np.testing.assert_array_equal(tile[32:, 8:], expected_tile[32:, 8:])

    check_wrong_layout_b()


def test_cta_group2_routes_each_pair_within_a_four_cta_cluster(tmp_path):
    left = (np.arange(256 * 16, dtype=np.float32).reshape(256, 16) % 19 - 9).astype(np.float16)
    right = (np.arange(32 * 16, dtype=np.float32).reshape(32, 16) % 13 - 6).astype(np.float16)
    expected = np.concatenate(
        [
            np.matmul(
                left[cta * 64 : (cta + 1) * 64].astype(np.float32),
                right[(cta // 2) * 16 : (cta // 2 + 1) * 16].astype(np.float32).T,
            )
            for cta in range(4)
        ],
        axis=0,
    )

    result = numsim.Engine().run(
        numsim.transpile(dense_gemm_async_two_cta_pairs_in_one_cluster, cache_dir=tmp_path),
        {"left": left, "right": right, "output": np.zeros((256, 16), dtype=np.float32)},
    )

    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_dense_gemm_async_handles_repeated_dynamic_index_loads(tmp_path):
    left = (np.arange(128 * 16, dtype=np.float32).reshape(128, 16) % 17 - 8).astype(np.float16)
    right = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 11 - 5).astype(np.float16)
    output = np.zeros((128, 8), dtype=np.float32)

    module = numsim.transpile(dense_gemm_async_dynamic_right_index, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"left": left, "right": right, "output": output})

    expected = np.matmul(left.astype(np.float32), right[1:9].astype(np.float32).T)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_dense_gemm_async_runs_numpy_backend_on_two_cluster_workers(tmp_path):
    left = (np.arange(256 * 16, dtype=np.float32).reshape(256, 16) % 17 - 8).astype(np.float16)
    right = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 11 - 5).astype(np.float16)
    output = np.zeros((256, 8), dtype=np.float32)

    module = numsim.transpile(dense_gemm_async_two_clusters, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=2).run(
        module, {"left": left, "right": right, "output": output}
    )

    expected = np.concatenate(
        [
            np.matmul(
                left[cta * 128 : (cta + 1) * 128].astype(np.float32),
                right[cta * 8 : (cta + 1) * 8].astype(np.float32).T,
            )
            for cta in range(2)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.stats["worker_count"] == 2
    assert result.stats["scheduling_domain_count"] == 2


def test_dense_gemm_async_reads_tmem_a_and_transposed_b_storage(tmp_path):
    left = (np.arange(128 * 16, dtype=np.float32).reshape(128, 16) % 13 - 6).astype(np.float16)
    right_transposed = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 7 - 3).astype(
        np.float16
    )
    output = np.zeros((128, 16), dtype=np.float32)

    module = numsim.transpile(dense_gemm_async_tmem_a_transposed_b, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"left": left, "right_transposed": right_transposed, "output": output}
    )

    expected = np.matmul(left.astype(np.float32), right_transposed.astype(np.float32))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cta_group2_gathers_both_shared_shards_and_scatters_tmem(tmp_path):
    left = (np.arange(256 * 16, dtype=np.float32).reshape(256, 16) % 19 - 9).astype(np.float16)
    right = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 13 - 6).astype(np.float16)
    output = np.zeros((256, 16), dtype=np.float32)

    module = numsim.transpile(dense_gemm_async_cta_group2, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=2).run(
        module, {"left": left, "right": right, "output": output}
    )

    expected = np.matmul(left.astype(np.float32), right.astype(np.float32).T)
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert result.stats["worker_count"] == 1
    assert result.stats["scheduling_domain_count"] == 1


def test_cta_group2_gathers_both_tmem_a_shards(tmp_path):
    """Each CTA of the pair contributes its own TMEM A rows, like the SMEM path."""

    left = (np.arange(256 * 16, dtype=np.float32).reshape(256, 16) % 19 - 9).astype(np.float16)
    right = (np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 13 - 6).astype(np.float16)
    output = np.zeros((256, 16), dtype=np.float32)

    module = numsim.transpile(dense_gemm_async_tmem_a_cta_group2, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=2).run(
        module, {"left": left, "right": right, "output": output}
    )

    expected = np.matmul(left.astype(np.float32), right.astype(np.float32).T)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_cta_group2_banked_a_selects_matching_b_shard(tmp_path):
    left = (np.arange(2 * 2 * 64 * 16, dtype=np.float32).reshape(2, 2, 64, 16) % 11 - 5).astype(
        np.float16
    )
    right = (np.arange(2 * 16 * 16, dtype=np.float32).reshape(2, 16, 16) % 7 - 3).astype(np.float16)
    output = np.zeros((2, 64, 32), dtype=np.float32)

    module = numsim.transpile(dense_gemm_async_m64_cta2_banked_a, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=2).run(
        module, {"left": left, "right": right, "output": output}
    )

    expected = np.empty_like(output)
    for cta in range(2):
        expected[cta, :, :16] = np.matmul(
            left[cta, 0].astype(np.float32), right[0].astype(np.float32).T
        )
        expected[cta, :, 16:] = np.matmul(
            left[cta, 1].astype(np.float32), right[1].astype(np.float32).T
        )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_all_inactive_gemm_async_is_a_noop(tmp_path):
    output = np.zeros(1, dtype=np.int32)

    module = numsim.transpile(inactive_gemm_async_is_noop, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.array([7], dtype=np.int32))
