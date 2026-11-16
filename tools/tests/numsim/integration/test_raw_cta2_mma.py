"""Raw CTA-pair tcgen05 MMA forms against the exact f32 product.

The kernels differ only in the MMA call: same shared and TMEM layouts, same
rendezvous, same read-back. Their operands are exactly representable, so the
expected destination is the plain f32 matrix product. TMEM is addressed through
raw ``tcgen05`` column operands; each warp reads and writes its own 32-lane
quarter of every CTA's TMEM.
"""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import ComposeLayout, S, TileLayout

_MMA_F16_32B = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))


def _packed_f16(low, high):
    return T.cast(T.reinterpret(low, ty="uint16"), "uint32") | T.shift_left(
        T.cast(T.reinterpret(high, ty="uint16"), "uint32"), T.uint32(16)
    )


@T.function
def raw_cta2_ss_layout_b_mma(
    left: T.Tensor((128, 16), "float16"),
    right: T.Tensor((32, 16), "float16"),
    output: T.Tensor((128, 32), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    lane_row = T.meta_var(warp * 32 + lane)
    # Operands first: their swizzle atoms start at the aligned shared base.
    left_shared = T.alloc_shared((64, 16), "float16", layout=_MMA_F16_32B)
    right_shared = T.alloc_shared((16, 16), "float16", layout=_MMA_F16_32B)
    address = T.alloc_shared((1,), "uint32")
    barrier = T.alloc_shared((1,), "uint64")
    word = T.alloc_local((1,), "uint32")
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
            for _i0_0, _i0_1 in T.grid(left_shared.shape[0], left_shared.shape[1]):
                left_shared[_i0_0, _i0_1] = left[cta * 64 + _i0_0, _i0_1]
            for _i1_0, _i1_1 in T.grid(right_shared.shape[0], right_shared.shape[1]):
                right_shared[_i1_0, _i1_1] = right[cta * 16 + _i1_0, _i1_1]
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()

    if (cta == 0) and (warp == 0) and (lane == 0):
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
            address[0],
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
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if cta == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    # Layout B: row r keeps columns [0, 16) in lane r and [16, 32) in lane 64 + r.
    for col in T.unroll(16):
        T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(col))
        T.ptx.tcgen05.wait__ld.sync.aligned()
        output[cta * 64 + lane_row % 64, (lane_row // 64) * 16 + col] = T.reinterpret(
            word[0], ty="float32"
        )
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()


@T.function
def raw_cta2_ss_layout_a_mma(
    left: T.Tensor((256, 16), "float16"),
    right: T.Tensor((32, 16), "float16"),
    output: T.Tensor((256, 32), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    lane_row = T.meta_var(warp * 32 + lane)
    # Operands first: their swizzle atoms start at the aligned shared base.
    left_shared = T.alloc_shared((128, 16), "float16", layout=_MMA_F16_32B)
    right_shared = T.alloc_shared((16, 16), "float16", layout=_MMA_F16_32B)
    address = T.alloc_shared((1,), "uint32")
    barrier = T.alloc_shared((1,), "uint64")
    word = T.alloc_local((1,), "uint32")
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
            for _i4_0, _i4_1 in T.grid(left_shared.shape[0], left_shared.shape[1]):
                left_shared[_i4_0, _i4_1] = left[cta * 128 + _i4_0, _i4_1]
            for _i5_0, _i5_1 in T.grid(right_shared.shape[0], right_shared.shape[1]):
                right_shared[_i5_0, _i5_1] = right[cta * 16 + _i5_0, _i5_1]
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()

    if (cta == 0) and (warp == 0) and (lane == 0):
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
            address[0],
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
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if cta == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    # Layout A: row r of each CTA's half is lane r.
    for col in T.unroll(32):
        T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(col))
        T.ptx.tcgen05.wait__ld.sync.aligned()
        output[cta * 128 + lane_row, col] = T.reinterpret(word[0], ty="float32")
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()


@T.function
def raw_cta2_ts_layout_b_mma(
    left: T.Tensor((2, 2, 64, 16), "float16"),
    right: T.Tensor((2, 16, 16), "float16"),
    output: T.Tensor((2, 64, 32), "float32"),
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    lane_row = T.meta_var(warp * 32 + lane)
    # Operands first: their swizzle atoms start at the aligned shared base.
    right_shared = T.alloc_shared((16, 16), "float16", layout=_MMA_F16_32B)
    address = T.alloc_shared((1,), "uint32")
    barrier = T.alloc_shared((1,), "uint64")
    word = T.alloc_local((1,), "uint32")
    desc_i: T.uint32
    desc_b: T.uint64

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
            for _i8_0, _i8_1 in T.grid(right_shared.shape[0], right_shared.shape[1]):
                right_shared[_i8_0, _i8_1] = right[cta, _i8_0, _i8_1]
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    # The banked A operand: bank b, row r is lane 64 * b + r; columns [0, 8)
    # hold two float16 values each.
    for pair in T.unroll(8):
        word[0] = _packed_f16(
            left[cta, lane_row // 64, lane_row % 64, pair * 2],
            left[cta, lane_row // 64, lane_row % 64, pair * 2 + 1],
        )
        T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0] + T.uint32(pair), word[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if (cta == 0) and (warp == 0) and (lane == 0):
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
            address[0] + T.uint32(8),
            address[0],
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
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if cta == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    # Layout B accumulator at column 8: row r keeps columns [0, 16) in lane r
    # and [16, 32) in lane 64 + r.
    for col in T.unroll(16):
        T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(8 + col))
        T.ptx.tcgen05.wait__ld.sync.aligned()
        output[cta, lane_row % 64, (lane_row // 64) * 16 + col] = T.reinterpret(
            word[0], ty="float32"
        )
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()


@T.function
def raw_cta2_ts_mma(
    left: T.Tensor((256, 16), "float16"),
    right: T.Tensor((32, 16), "float16"),
    output: T.Tensor((256, 32), "float32"),
):
    """`M=256` CTA-pair MMA whose A operand is each CTA's own TMEM shard."""

    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128, cluster=2))
    cta = T.cuda.cluster_cta_id("x")
    warp = T.cuda.warp_id()
    lane = T.cuda.lane_id()
    lane_row = T.meta_var(warp * 32 + lane)
    # Operands first: their swizzle atoms start at the aligned shared base.
    right_shared = T.alloc_shared((16, 16), "float16", layout=_MMA_F16_32B)
    address = T.alloc_shared((1,), "uint32")
    barrier = T.alloc_shared((1,), "uint64")
    word = T.alloc_local((1,), "uint32")
    desc_i: T.uint32
    desc_b: T.uint64

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(T.address_of(address[0]), 64)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
            for _i10_0, _i10_1 in T.grid(right_shared.shape[0], right_shared.shape[1]):
                right_shared[_i10_0, _i10_1] = right[cta * 16 + _i10_0, _i10_1]
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    # Layout A operand: row r is lane r; columns [0, 8) hold two float16 values each.
    for pair in T.unroll(8):
        word[0] = _packed_f16(
            left[cta * 128 + lane_row, pair * 2], left[cta * 128 + lane_row, pair * 2 + 1]
        )
        T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0] + T.uint32(pair), word[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if (cta == 0) and (warp == 0) and (lane == 0):
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
            address[0] + T.uint32(8),
            address[0],
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
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if cta == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    # Layout A accumulator at column 8: row r of each CTA's half is lane r.
    for col in T.unroll(32):
        T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(8 + col))
        T.ptx.tcgen05.wait__ld.sync.aligned()
        output[cta * 128 + lane_row, col] = T.reinterpret(word[0], ty="float32")
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(address[0], 64)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()


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
