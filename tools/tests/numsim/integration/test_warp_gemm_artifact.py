from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tests.numsim.support.kernels import (
    _TEST_WARP_GEMM_A_FRAG,
    _TEST_WARP_GEMM_A_FRAG_K8,
    _TEST_WARP_GEMM_B_FRAG,
    _TEST_WARP_GEMM_B_FRAG_K8,
    _TEST_WARP_GEMM_D_FRAG,
    warp_gemm_bf16_m16n8k16,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout, laneid

# This layout swaps the high-M and high-K physical register groups while
# retaining the same logical shape, lane ownership, and per-lane span.
_WRONG_WARP_GEMM_A_FRAG = TileLayout(S[(2, 8, 2, 4, 2) : (4, 4 @ laneid, 2, 1 @ laneid, 1)])


@T.prim_func
def _warpgroup_gemm_per_warp(
    left: T.Buffer((16, 16), "float16"),
    right: T.Buffer((16, 8), "float16"),
    output: T.Buffer((4, 16, 8), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    a = T.alloc_buffer((16, 16), "float16", scope="local", layout=_TEST_WARP_GEMM_A_FRAG)
    b = T.alloc_buffer((16, 8), "float16", scope="local", layout=_TEST_WARP_GEMM_B_FRAG)
    c = T.alloc_buffer((16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG)
    d = T.alloc_buffer((16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG)
    for row, k in T.grid(16, 16):
        if lane == 4 * (row % 8) + (k % 8) // 2:
            a[row, k] = left[row, k]
    for k, col in T.grid(16, 8):
        if lane == 4 * col + (k % 8) // 2:
            b[k, col] = right[k, col]
    Tx.wg.gemm(d, a, b, c, alpha=1.0, beta=0.0, dispatch="mma.m16n8k*")
    for row, col in T.grid(16, 8):
        if lane == 4 * (row % 8) + col // 2:
            output[warp, row, col] = d[row, col]


@T.prim_func
def _cta_gemm_per_warp(
    left: T.Buffer((16, 16), "float16"),
    right: T.Buffer((16, 8), "float16"),
    output: T.Buffer((2, 16, 8), "float32"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    a = T.alloc_buffer((16, 16), "float16", scope="local", layout=_TEST_WARP_GEMM_A_FRAG)
    b = T.alloc_buffer((16, 8), "float16", scope="local", layout=_TEST_WARP_GEMM_B_FRAG)
    c = T.alloc_buffer((16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG)
    d = T.alloc_buffer((16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG)
    for row, k in T.grid(16, 16):
        if lane == 4 * (row % 8) + (k % 8) // 2:
            a[row, k] = left[row, k]
    for k, col in T.grid(16, 8):
        if lane == 4 * col + (k % 8) // 2:
            b[k, col] = right[k, col]
    Tx.cta.gemm(d, a, b, c, alpha=1.0, beta=0.0, dispatch="mma.m16n8k*")
    for row, col in T.grid(16, 8):
        if lane == 4 * (row % 8) + col // 2:
            output[warp, row, col] = d[row, col]


@T.prim_func
def _warp_gemm_wrong_a_fragment_layout():
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    left = T.alloc_buffer((16, 16), "bfloat16", scope="local", layout=_WRONG_WARP_GEMM_A_FRAG)
    right = T.alloc_buffer((16, 8), "bfloat16", scope="local", layout=_TEST_WARP_GEMM_B_FRAG)
    accumulator = T.alloc_buffer((16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG)
    destination = T.alloc_buffer((16, 8), "float32", scope="local", layout=_TEST_WARP_GEMM_D_FRAG)
    Tx.warp.gemm(
        destination,
        left,
        right,
        accumulator,
        transpose_A=False,
        transpose_B=False,
        alpha=1.0,
        beta=0.0,
    )


def _encode_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _decode_bf16(bits: np.ndarray) -> np.ndarray:
    words = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return words.view(np.float32)


def _transpose_fragment(layout: TileLayout, shape: tuple[int, int]) -> TileLayout:
    grouped, separators = layout.group(shape)
    return grouped.permute_by_groups(separators, [1, 0])


def _build_warp_gemm_variant(
    *,
    dtype: str,
    mma_k: int,
    m_tiles: int,
    n_tiles: int,
    k_tiles: int,
    transpose_a: bool,
    transpose_b: bool,
    beta: int,
):
    m, n, k = 16 * m_tiles, 8 * n_tiles, mma_k * k_tiles
    a_base = _TEST_WARP_GEMM_A_FRAG if mma_k == 16 else _TEST_WARP_GEMM_A_FRAG_K8
    b_base = _TEST_WARP_GEMM_B_FRAG if mma_k == 16 else _TEST_WARP_GEMM_B_FRAG_K8
    d_layout = _TEST_WARP_GEMM_D_FRAG.tile_to([m, n], [16, 8])
    a_standard_layout = a_base.tile_to([m, k], [16, mma_k])
    b_standard_layout = b_base.tile_to([k, n], [mma_k, 8])
    a_layout = _transpose_fragment(a_standard_layout, (m, k)) if transpose_a else a_standard_layout
    b_layout = _transpose_fragment(b_standard_layout, (k, n)) if transpose_b else b_standard_layout
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)

    @T.prim_func
    def gemm(
        a_global: T.Buffer(a_shape, dtype),
        b_global: T.Buffer(b_shape, dtype),
        c_global: T.Buffer((m, n), "float32"),
        output: T.Buffer((m, n), "float32"),
    ):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        a_fragment = T.alloc_buffer(a_shape, dtype, scope="local", layout=a_layout)
        b_fragment = T.alloc_buffer(b_shape, dtype, scope="local", layout=b_layout)
        c_fragment = T.alloc_buffer((m, n), "float32", scope="local", layout=d_layout)
        d_fragment = T.alloc_buffer((m, n), "float32", scope="local", layout=d_layout)

        if transpose_a:
            for k_index, row in T.grid(k, m):
                if lane == 4 * (row % 8) + (k_index % 8) // 2:
                    a_fragment[k_index, row] = a_global[k_index, row]
        else:
            for row, k_index in T.grid(m, k):
                if lane == 4 * (row % 8) + (k_index % 8) // 2:
                    a_fragment[row, k_index] = a_global[row, k_index]

        if transpose_b:
            for col, k_index in T.grid(n, k):
                if lane == 4 * (col % 8) + (k_index % 8) // 2:
                    b_fragment[col, k_index] = b_global[col, k_index]
        else:
            for k_index, col in T.grid(k, n):
                if lane == 4 * (col % 8) + (k_index % 8) // 2:
                    b_fragment[k_index, col] = b_global[k_index, col]

        if beta == 1:
            for row, col in T.grid(m, n):
                if lane == 4 * (row % 8) + (col % 8) // 2:
                    c_fragment[row, col] = c_global[row, col]

        Tx.warp.gemm(
            d_fragment,
            a_fragment,
            b_fragment,
            c_fragment,
            transpose_A=transpose_a,
            transpose_B=transpose_b,
            alpha=1.0,
            beta=float(beta),
        )

        for row, col in T.grid(m, n):
            if lane == 4 * (row % 8) + (col % 8) // 2:
                output[row, col] = d_fragment[row, col]

    return gemm, (m, n, k), a_shape, b_shape


def test_warp_gemm_gathers_and_scatters_register_fragments_by_layout(tmp_path):
    left_bits = _encode_bf16((np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 13) - 6)
    right_bits = _encode_bf16((np.arange(16 * 8, dtype=np.float32).reshape(16, 8) % 11) - 5)
    accumulator = ((np.arange(16 * 8, dtype=np.float32).reshape(16, 8) % 9) - 4) * np.float32(0.25)
    product = np.zeros((16, 8), dtype=np.float32)
    accumulated = np.zeros((16, 8), dtype=np.float32)
    legacy_order_product = np.zeros((16, 8), dtype=np.float32)

    module = numsim.transpile(warp_gemm_bf16_m16n8k16, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "left": left_bits,
            "right": right_bits,
            "accumulator": accumulator,
            "product": product,
            "accumulated": accumulated,
            "legacy_order_product": legacy_order_product,
        },
    )

    left = _decode_bf16(left_bits)
    right = _decode_bf16(right_bits)
    expected_product = np.matmul(left, right)
    rows = np.arange(16)[:, None]
    cols = np.arange(16)[None, :]
    legacy_order_left = left[(rows & 7) | (cols & 8), (cols & 7) | (rows & 8)]
    expected_legacy_order_product = np.matmul(legacy_order_left, right)
    np.testing.assert_array_equal(result.outputs["product"], expected_product)
    np.testing.assert_array_equal(result.outputs["accumulated"], expected_product + accumulator)
    np.testing.assert_array_equal(
        result.outputs["legacy_order_product"], expected_legacy_order_product
    )
    assert not np.array_equal(expected_legacy_order_product, expected_product)


@pytest.mark.parametrize(
    ("kernel", "warps"),
    [(_warpgroup_gemm_per_warp, 4), (_cta_gemm_per_warp, 2)],
)
def test_larger_exec_scopes_run_one_mma_fragment_per_warp(tmp_path, kernel, warps):
    left = ((np.arange(16 * 16, dtype=np.float32).reshape(16, 16) % 7) - 3).astype(np.float16)
    right = ((np.arange(16 * 8, dtype=np.float32).reshape(16, 8) % 5) - 2).astype(np.float16)
    output = np.zeros((warps, 16, 8), dtype=np.float32)

    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"left": left, "right": right, "output": output})

    expected = np.broadcast_to(left.astype(np.float32) @ right.astype(np.float32), output.shape)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_warp_gemm_rejects_layout_that_disagrees_with_instruction_abi(tmp_path):
    with pytest.raises(
        UnsupportedTIRxError,
        match=r"A fragment layout does not match fixed mma\.sync\.m16n8k16 ABI",
    ):
        numsim.transpile(_warp_gemm_wrong_a_fragment_layout, cache_dir=tmp_path)


_WARP_GEMM_VARIANTS = [
    ("float16", 8, 1, 1, 1, False, False, 0),
    ("bfloat16", 8, 2, 2, 3, True, False, 1),
    ("float16", 16, 1, 1, 1, False, True, 0),
    ("bfloat16", 16, 2, 2, 2, True, True, 1),
]


def _check_warp_gemm_supports_registered_m16n8_family(
    tmp_path,
    dtype,
    mma_k,
    m_tiles,
    n_tiles,
    k_tiles,
    transpose_a,
    transpose_b,
    beta,
):
    kernel, (m, n, k), a_shape, b_shape = _build_warp_gemm_variant(
        dtype=dtype,
        mma_k=mma_k,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_tiles=k_tiles,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        beta=beta,
    )
    a_values = (np.arange(np.prod(a_shape), dtype=np.float32).reshape(a_shape) % 7) - 3
    b_values = (np.arange(np.prod(b_shape), dtype=np.float32).reshape(b_shape) % 5) - 2
    c_values = ((np.arange(m * n, dtype=np.float32).reshape(m, n) % 9) - 4) * np.float32(0.25)
    if dtype == "bfloat16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        a_argument = a_values.astype(ml_dtypes.bfloat16)
        b_argument = b_values.astype(ml_dtypes.bfloat16)
        a_runtime = np.asarray(a_argument, dtype=np.float32)
        b_runtime = np.asarray(b_argument, dtype=np.float32)
    else:
        a_argument = a_values.astype(np.float16)
        b_argument = b_values.astype(np.float16)
        a_runtime = np.asarray(a_argument, dtype=np.float32)
        b_runtime = np.asarray(b_argument, dtype=np.float32)

    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_global": a_argument,
            "b_global": b_argument,
            "c_global": c_values,
            "output": np.zeros((m, n), dtype=np.float32),
        },
    )
    a_standard = a_runtime.T if transpose_a else a_runtime
    b_standard = b_runtime.T if transpose_b else b_runtime
    assert a_standard.shape == (m, k)
    assert b_standard.shape == (k, n)
    expected = np.matmul(a_standard, b_standard)
    if beta == 1:
        expected = expected + c_values
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize(
    "dtype,mma_k,m_tiles,n_tiles,k_tiles,transpose_a,transpose_b,beta",
    _WARP_GEMM_VARIANTS,
)
def test_warp_gemm_supports_registered_m16n8_family(
    tmp_path,
    dtype,
    mma_k,
    m_tiles,
    n_tiles,
    k_tiles,
    transpose_a,
    transpose_b,
    beta,
):
    _check_warp_gemm_supports_registered_m16n8_family(
        tmp_path,
        dtype,
        mma_k,
        m_tiles,
        n_tiles,
        k_tiles,
        transpose_a,
        transpose_b,
        beta,
    )
