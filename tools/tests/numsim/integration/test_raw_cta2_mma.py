"""Raw CTA-pair tcgen05 MMA forms against the exact f32 product.

The kernels differ only in the MMA call: same shared and TMEM layouts, same
rendezvous, same read-back. Their operands are exactly representable, so the
expected destination is the plain f32 matrix product.
"""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm_tirx.backend.cuda.layout import tmem_datapath_layout
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import Axis, ComposeLayout, S, TileLayout

_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_TMEM_A_CTA2_M128_BANKED_F16 = TileLayout(
    S[(2, 64, 16) : (64 @ Axis.get("TLane"), 1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))]
)


@T.function
def raw_cta2_ss_layout_b_mma(
    left: T.Tensor((128, 16), "float16"),
    right: T.Tensor((32, 16), "float16"),
    output: T.Tensor((128, 32), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    left_shared = T.alloc_shared((64, 16), "float16", layout=_MMA_F16_32B)
    right_shared = T.alloc_shared((16, 16), "float16", layout=_MMA_F16_32B)
    accumulator = T.tmem(
        (64, 32), "float32", allocated_addr=0, layout=tmem_datapath_layout("B", 64, 32)
    )
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if lane == 0:
        for _i0_0, _i0_1 in T.grid(left_shared.shape[0], left_shared.shape[1]):
            left_shared[_i0_0, _i0_1] = left[cta * 64 + _i0_0, _i0_1]
        for _i1_0, _i1_1 in T.grid(right_shared.shape[0], right_shared.shape[1]):
            right_shared[_i1_0, _i1_1] = right[cta * 16 + _i1_0, _i1_1]
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


@T.function
def raw_cta2_ss_layout_a_mma(
    left: T.Tensor((256, 16), "float16"),
    right: T.Tensor((32, 16), "float16"),
    output: T.Tensor((256, 32), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    left_shared = T.alloc_shared((128, 16), "float16", layout=_MMA_F16_32B)
    right_shared = T.alloc_shared((16, 16), "float16", layout=_MMA_F16_32B)
    accumulator = T.tmem(
        (128, 32), "float32", allocated_addr=0, layout=tmem_datapath_layout("A", 128, 32)
    )
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if lane == 0:
        for _i4_0, _i4_1 in T.grid(left_shared.shape[0], left_shared.shape[1]):
            left_shared[_i4_0, _i4_1] = left[cta * 128 + _i4_0, _i4_1]
        for _i5_0, _i5_1 in T.grid(right_shared.shape[0], right_shared.shape[1]):
            right_shared[_i5_0, _i5_1] = right[cta * 16 + _i5_0, _i5_1]
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


@T.function
def raw_cta2_ts_layout_b_mma(
    left: T.Tensor((2, 2, 64, 16), "float16"),
    right: T.Tensor((2, 16, 16), "float16"),
    output: T.Tensor((2, 64, 32), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    right_shared = T.alloc_shared((16, 16), "float16", layout=_MMA_F16_32B)
    left_tmem = T.tmem(
        (2, 64, 16), "float16", allocated_addr=0, layout=_TMEM_A_CTA2_M128_BANKED_F16
    )
    accumulator = T.tmem(
        (64, 32), "float32", allocated_addr=8, layout=tmem_datapath_layout("B", 64, 32)
    )
    desc_i: T.uint32
    desc_b: T.uint64

    if lane == 0:
        for _i8_0, _i8_1 in T.grid(right_shared.shape[0], right_shared.shape[1]):
            right_shared[_i8_0, _i8_1] = right[cta, _i8_0, _i8_1]
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


@T.function
def raw_cta2_ts_mma(
    left: T.Tensor((256, 16), "float16"),
    right: T.Tensor((32, 16), "float16"),
    output: T.Tensor((256, 32), "float32"),
):
    """`M=256` CTA-pair MMA whose A operand is each CTA's own TMEM shard."""

    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    lane = T.cuda.lane_id()
    right_shared = T.alloc_shared((16, 16), "float16", layout=_MMA_F16_32B)
    left_tmem = T.tmem(
        (128, 16), "float16", allocated_addr=0, layout=tmem_datapath_layout("A", 128, 16)
    )
    accumulator = T.tmem(
        (128, 32), "float32", allocated_addr=8, layout=tmem_datapath_layout("A", 128, 32)
    )
    desc_i: T.uint32
    desc_b: T.uint64

    if lane == 0:
        for _i10_0, _i10_1 in T.grid(right_shared.shape[0], right_shared.shape[1]):
            right_shared[_i10_0, _i10_1] = right[cta * 16 + _i10_0, _i10_1]
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
    ("raw_kernel", "joint_m"),
    [
        (raw_cta2_ss_layout_b_mma, 128),
        (raw_cta2_ss_layout_a_mma, 256),
        (raw_cta2_ts_mma, 256),
    ],
    ids=[
        "ss_m128_n32_layout_b",
        "ss_m256_n32_layout_a",
        "ts_m256_n32_layout_a",
    ],
)
def test_raw_cta2_mma_is_the_f32_product(raw_kernel, joint_m, tmp_path):
    rng = np.random.default_rng(20260807)
    left = rng.integers(-4, 5, size=(joint_m, 16)).astype(np.float16)
    right = rng.integers(-3, 4, size=(32, 16)).astype(np.float16)

    raw = _run(raw_kernel, left, right, joint_m, tmp_path)

    # The operands are exactly representable, so the product is exact.
    np.testing.assert_array_equal(raw, left.astype(np.float32) @ right.astype(np.float32).T)


def test_raw_cta2_ts_m128_selects_the_matching_a_lane_bank(tmp_path):
    rng = np.random.default_rng(20260811)
    left = rng.integers(-4, 5, size=(2, 2, 64, 16)).astype(np.float16)
    right = rng.integers(-3, 4, size=(2, 16, 16)).astype(np.float16)
    module = numsim.transpile(raw_cta2_ts_layout_b_mma, cache_dir=tmp_path)
    raw = (
        numsim.Engine()
        .run(
            module,
            {"left": left, "right": right, "output": np.zeros((2, 64, 32), dtype=np.float32)},
        )
        .outputs["output"]
    )
    expected = np.empty((2, 64, 32), dtype=np.float32)
    for cta in range(2):
        expected[cta, :, :16] = left[cta, 0].astype(np.float32) @ right[0].astype(np.float32).T
        expected[cta, :, 16:] = left[cta, 1].astype(np.float32) @ right[1].astype(np.float32).T

    np.testing.assert_array_equal(raw, expected)
