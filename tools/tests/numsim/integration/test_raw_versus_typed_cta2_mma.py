"""The raw `cta_group=2` MMAs agree exactly with the typed `gemm_async` ones.

Both paths reach the same engine-private numeric core, so on identical operands
they must produce bit-identical f32 destinations. The typed CTA-pair path is
GPU-validated, which makes it the oracle for the raw one; the raw CTA-pair
entries otherwise have only independent-reference coverage, which cannot catch a
destination-mapping error that both the raw scatter and a hand-written numpy
expectation would have to share.

The kernels in each pair differ only in the MMA call: same shared and TMEM
layouts, same rendezvous, same read-back.

`N=32` is the smallest CTA-pair N *both* shape tables accept -- NumSim's
`validate_tcgen05_instruction_shape` and the engine decode allow step 16, the
TIRx CUDA frontend's `_check_tcgen05_mma_matrix_shape` requires step 32 -- so
these are geometries a real compilation could also emit, which is what makes the
typed oracle cover the geometry actually under test. The `N=16` end of NumSim's
own domain is already exercised by
`runtime/test_raw_tcgen_codegen.py::test_raw_tcgen_f16_cta2_datapaths_match_independent_matrix_products`,
so it is not duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
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

_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_TMEM_A_CTA2_M128_BANKED_F16 = TileLayout(S[(2, 64, 16) : (64 @ TLane, 1 @ TLane, 1 @ TCol)])


@T.prim_func
def raw_cta2_ss_layout_b_mma(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((128, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=0,
    )
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=32,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(left_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(right_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(32):
                output[cta * 64 + row, col] = accumulator[row, col]


@T.prim_func
def typed_cta2_ss_layout_b_gemm_async(
    left: T.Buffer((128, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((128, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=0,
    )

    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 64 : (cta + 1) * 64, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
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
            for col in T.serial(32):
                output[cta * 64 + row, col] = accumulator[row, col]


@T.prim_func
def raw_cta2_ss_layout_a_mma(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 32),
        allocated_addr=0,
    )
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 128 : (cta + 1) * 128, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=32,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(left_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(right_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(32):
                output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def typed_cta2_ss_layout_a_gemm_async(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    accumulator = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 32),
        allocated_addr=0,
    )

    if lane == 0:
        Tx.copy(left_shared[:, :], left[cta * 128 : (cta + 1) * 128, :])
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
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
        for row in T.serial(128):
            for col in T.serial(32):
                output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def raw_cta2_ts_layout_b_mma(
    left: T.Buffer((2, 2, 64, 16), "float16"),
    right: T.Buffer((2, 16, 16), "float16"),
    output: T.Buffer((2, 64, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    left_tmem = T.decl_buffer(
        (2, 64, 16),
        "float16",
        scope="tmem",
        layout=_TMEM_A_CTA2_M128_BANKED_F16,
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=8,
    )
    desc_i: T.uint32
    desc_b: T.uint64

    if lane == 0:
        Tx.copy(right_shared[:, :], right[cta, :, :])
        for bank in T.serial(2):
            for row in T.serial(64):
                for col in T.serial(16):
                    left_tmem[bank, row, col] = left[cta, bank, row, col]
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=32,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(right_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(64):
            for col in T.serial(32):
                output[cta, row, col] = accumulator[row, col]


@T.prim_func
def typed_cta2_ts_layout_b_gemm_async(
    left: T.Buffer((2, 2, 64, 16), "float16"),
    right: T.Buffer((2, 16, 16), "float16"),
    output: T.Buffer((2, 64, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    left_tmem = T.decl_buffer(
        (2, 64, 16),
        "float16",
        scope="tmem",
        layout=_TMEM_A_CTA2_M128_BANKED_F16,
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (64, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("B", 64, 32),
        allocated_addr=8,
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
def raw_cta2_ts_mma(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    """`M=256` CTA-pair MMA whose A operand is each CTA's own TMEM shard."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    left_tmem = T.decl_buffer(
        (128, 16),
        "float16",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 16),
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 32),
        allocated_addr=8,
    )
    desc_i: T.uint32
    desc_b: T.uint64

    if lane == 0:
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
        for row in T.serial(128):
            for col in T.serial(16):
                left_tmem[row, col] = left[cta * 128 + row, col]
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=32,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(right_shared[0, 0]), ldo=16, sdo=16, swizzle=1
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(32):
                output[cta * 128 + row, col] = accumulator[row, col]


@T.prim_func
def typed_cta2_ts_gemm_async(
    left: T.Buffer((256, 16), "float16"),
    right: T.Buffer((32, 16), "float16"),
    output: T.Buffer((256, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    right_shared = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_MMA_F16_32B)
    left_tmem = T.decl_buffer(
        (128, 16),
        "float16",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 16),
        allocated_addr=0,
    )
    accumulator = T.decl_buffer(
        (128, 32),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("A", 128, 32),
        allocated_addr=8,
    )

    if lane == 0:
        Tx.copy(right_shared[:, :], right[cta * 16 : (cta + 1) * 16, :])
        for row in T.serial(128):
            for col in T.serial(16):
                left_tmem[row, col] = left[cta * 128 + row, col]
    T.cuda.cluster_sync()

    if (cta == 0) and (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            left_tmem[:, :],
            right_shared[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
    T.cuda.cluster_sync()

    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(32):
                output[cta * 128 + row, col] = accumulator[row, col]


def _run(kernel, left, right, joint_m, cache_dir):
    module = numsim.transpile(kernel, cache_dir=cache_dir)
    return (
        numsim.Engine()
        .run(
            module,
            {
                "left": left,
                "right": right,
                "output": np.zeros((joint_m, 32), dtype=np.float32),
            },
        )
        .outputs["output"]
    )


@pytest.mark.parametrize(
    ("raw_kernel", "typed_kernel", "joint_m"),
    [
        (raw_cta2_ss_layout_b_mma, typed_cta2_ss_layout_b_gemm_async, 128),
        (raw_cta2_ss_layout_a_mma, typed_cta2_ss_layout_a_gemm_async, 256),
        (raw_cta2_ts_mma, typed_cta2_ts_gemm_async, 256),
    ],
    ids=[
        "ss_m128_n32_layout_b",
        "ss_m256_n32_layout_a",
        "ts_m256_n32_layout_a",
    ],
)
def test_raw_cta2_mma_matches_the_typed_gemm_async_exactly(
    raw_kernel, typed_kernel, joint_m, tmp_path
):
    rng = np.random.default_rng(20260807)
    left = rng.integers(-4, 5, size=(joint_m, 16)).astype(np.float16)
    right = rng.integers(-3, 4, size=(32, 16)).astype(np.float16)

    raw = _run(raw_kernel, left, right, joint_m, tmp_path)
    typed = _run(typed_kernel, left, right, joint_m, tmp_path)

    np.testing.assert_array_equal(raw, typed)
    # Both are also the plain f32 product for these exactly representable
    # operands, so the agreement is not two paths sharing one wrong answer.
    np.testing.assert_array_equal(raw, left.astype(np.float32) @ right.astype(np.float32).T)


def test_raw_cta2_ts_m128_selects_the_matching_a_lane_bank(tmp_path):
    rng = np.random.default_rng(20260811)
    left = rng.integers(-4, 5, size=(2, 2, 64, 16)).astype(np.float16)
    right = rng.integers(-3, 4, size=(2, 16, 16)).astype(np.float16)

    def run(kernel):
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        return (
            numsim.Engine()
            .run(
                module,
                {
                    "left": left,
                    "right": right,
                    "output": np.zeros((2, 64, 32), dtype=np.float32),
                },
            )
            .outputs["output"]
        )

    raw = run(raw_cta2_ts_layout_b_mma)
    typed = run(typed_cta2_ts_layout_b_gemm_async)
    expected = np.empty((2, 64, 32), dtype=np.float32)
    for cta in range(2):
        expected[cta, :, :16] = left[cta, 0].astype(np.float32) @ right[0].astype(np.float32).T
        expected[cta, :, 16:] = left[cta, 1].astype(np.float32) @ right[1].astype(np.float32).T

    np.testing.assert_array_equal(raw, typed)
    np.testing.assert_array_equal(raw, expected)
