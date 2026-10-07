#!/usr/bin/env python3
"""Independent B200 probes for dense and tile TCGEN05 MMA forms."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from tests.numsim.microtests.harness import PairedBuffer
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, S, TileLayout, tmem_datapath_layout, wg_local_layout
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import (
    sf_smem_layout,
    sf_tmem_layout,
)
from tvm.backend.cuda.tile_primitive.tma_utils import (
    SwizzleMode,
    mma_shared_layout,
)


_F16_SMEM_LAYOUT = ComposeLayout(3, 1, 3, TileLayout(S[(128,)]))
_FP8_A_SMEM_LAYOUT = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32))
_FP8_B_SMEM_LAYOUT = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_32B_ATOM, (8, 32))
# Byte-carrier layouts for the raw f8f6f4 cases. The operand element type of a
# raw `tcgen05.mma` lives entirely in its instruction descriptor, so these
# kernels stage plain bytes and let the descriptor name E4M3 or E5M2. That also
# keeps them off the typed tile-copy dtype table, which is a separate surface.
_F8_BYTES_A64_SMEM_LAYOUT = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (64, 32))
_F8_BYTES_B8_SMEM_LAYOUT = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 32))
_F8_BYTES_A128_SMEM_LAYOUT = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32))
_F8_BYTES_B16_SMEM_LAYOUT = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (16, 32))
_FP4_A_SMEM_LAYOUT = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32))
_FP4_B_SMEM_LAYOUT = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 32))
_MXF8_SCALE_SMEM_LAYOUT = sf_smem_layout(128, SF_K=4, sf_per_mma=1)
_MXF8_SCALE_TMEM_LAYOUT = sf_tmem_layout(128, SF_K=4, sf_per_mma=1)
_NVFP4_SCALE_SMEM_LAYOUT = sf_smem_layout(128, SF_K=4, sf_per_mma=4)
_NVFP4_SCALE_TMEM_LAYOUT = sf_tmem_layout(128, SF_K=4, sf_per_mma=4)


@T.prim_func
def raw_f16_ss_m64_layout_f_valid_descriptor(
    a: T.Buffer((64, 16), "float16"),
    b: T.Buffer((8, 16), "float16"),
    output: T.Buffer((4, 32, 4), "float32"),
    descriptors: T.Buffer((3,), "uint64"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_F16_SMEM_LAYOUT)
    shared_b = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_F16_SMEM_LAYOUT)
    registers = T.alloc_local((4,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[:, :])
        Tx.copy(shared_b[:, :], b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instruction_descriptor),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=64,
            N=8,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        descriptors[0] = T.cast(instruction_descriptor, "uint64")
        descriptors[1] = descriptor_a
        descriptors[2] = descriptor_b
        T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
            address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
        registers[0], registers[1], registers[2], registers[3], address[0]
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(4):
        output[warp, lane, register] = T.reinterpret("float32", registers[register])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def raw_e4m3_ss_m64_layout_f_valid_descriptor(
    a: T.Buffer((64, 32), "float8_e4m3fn"),
    b: T.Buffer((8, 32), "float8_e4m3fn"),
    output: T.Buffer((4, 32, 4), "float32"),
    descriptors: T.Buffer((3,), "uint64"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((64, 32), "float8_e4m3fn", scope="shared", layout=_FP8_A_SMEM_LAYOUT)
    shared_b = T.alloc_buffer((8, 32), "float8_e4m3fn", scope="shared", layout=_FP8_B_SMEM_LAYOUT)
    registers = T.alloc_local((4,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[:, :])
        Tx.copy(shared_b[:, :], b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instruction_descriptor),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        descriptors[0] = T.cast(instruction_descriptor, "uint64")
        descriptors[1] = descriptor_a
        descriptors[2] = descriptor_b
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
        registers[0], registers[1], registers[2], registers[3], address[0]
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(4):
        output[warp, lane, register] = T.reinterpret("float32", registers[register])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def raw_e5m2_ss_m64_layout_f_valid_descriptor(
    a: T.Buffer((64, 32), "uint8"),
    b: T.Buffer((8, 32), "uint8"),
    output: T.Buffer((4, 32, 4), "float32"),
    descriptors: T.Buffer((3,), "uint64"),
):
    """`kind::f8f6f4` with both operands E5M2 and a float32 destination."""

    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((64, 32), "uint8", scope="shared", layout=_F8_BYTES_A64_SMEM_LAYOUT)
    shared_b = T.alloc_buffer((8, 32), "uint8", scope="shared", layout=_F8_BYTES_B8_SMEM_LAYOUT)
    registers = T.alloc_local((4,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[:, :])
        Tx.copy(shared_b[:, :], b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instruction_descriptor),
            d_dtype="float32",
            a_dtype="float8_e5m2",
            b_dtype="float8_e5m2",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        descriptors[0] = T.cast(instruction_descriptor, "uint64")
        descriptors[1] = descriptor_a
        descriptors[2] = descriptor_b
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        address[0],
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(4):
        output[warp, lane, register] = T.reinterpret("float32", registers[register])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def raw_e4m3_e5m2_ss_m64_layout_f_valid_descriptor(
    a: T.Buffer((64, 32), "uint8"),
    b: T.Buffer((8, 32), "uint8"),
    output: T.Buffer((4, 32, 4), "float32"),
    descriptors: T.Buffer((3,), "uint64"),
):
    """`kind::f8f6f4` with an E4M3 A and an E5M2 B: the operand dtypes are independent."""

    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((64, 32), "uint8", scope="shared", layout=_F8_BYTES_A64_SMEM_LAYOUT)
    shared_b = T.alloc_buffer((8, 32), "uint8", scope="shared", layout=_F8_BYTES_B8_SMEM_LAYOUT)
    registers = T.alloc_local((4,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[:, :])
        Tx.copy(shared_b[:, :], b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instruction_descriptor),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e5m2",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        descriptors[0] = T.cast(instruction_descriptor, "uint64")
        descriptors[1] = descriptor_a
        descriptors[2] = descriptor_b
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        address[0],
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(4):
        output[warp, lane, register] = T.reinterpret("float32", registers[register])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def raw_e5m2_e4m3_f16_d_ss_m128_layout_d(
    a: T.Buffer((128, 32), "uint8"),
    b: T.Buffer((16, 32), "uint8"),
    seed: T.Buffer((4, 32, 16), "uint32"),
    output: T.Buffer((4, 32, 16), "uint32"),
):
    """`kind::f8f6f4` with an E5M2 A, an E4M3 B, and a *float16* destination.

    The destination is seeded through `tcgen05.st` with a non-zero upper half
    so the raw words prove three separate things at once: the addend is read
    from the low half, the product is stored to the low half, and the upper
    half is written as zero rather than preserved.
    """

    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((128, 32), "uint8", scope="shared", layout=_F8_BYTES_A128_SMEM_LAYOUT)
    shared_b = T.alloc_buffer((16, 32), "uint8", scope="shared", layout=_F8_BYTES_B16_SMEM_LAYOUT)
    registers = T.alloc_local((16,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[:, :])
        Tx.copy(shared_b[:, :], b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    for column in T.unroll(16):
        registers[column] = seed[warp, lane, column]
    T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
        address[0],
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        registers[4],
        registers[5],
        registers[6],
        registers[7],
        registers[8],
        registers[9],
        registers[10],
        registers[11],
        registers[12],
        registers[13],
        registers[14],
        registers[15],
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instruction_descriptor),
            d_dtype="float16",
            a_dtype="float8_e5m2",
            b_dtype="float8_e4m3fn",
            M=128,
            N=16,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(1)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        registers[4],
        registers[5],
        registers[6],
        registers[7],
        registers[8],
        registers[9],
        registers[10],
        registers[11],
        registers[12],
        registers[13],
        registers[14],
        registers[15],
        address[0],
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(16):
        output[warp, lane, register] = registers[register]
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def raw_f16_f16_d_ss_m128_layout_d(
    a: T.Buffer((128, 16), "float16"),
    b: T.Buffer((16, 16), "float16"),
    seed: T.Buffer((4, 32, 16), "uint32"),
    output: T.Buffer((4, 32, 16), "uint32"),
):
    """`kind::f16` with F16 multiplicands and a *float16* destination.

    PTX ISA 9.7.17.4.2 Table 45 puts D=F16 at bits 4-5 = 0 under `.kind::f16`.
    This is the `.kind::f16` twin of `raw_e5m2_e4m3_f16_d_ss_m128_layout_d`:
    the destination is seeded through `tcgen05.st` with a non-zero upper half,
    so the raw words prove the addend is read from the low half, the product is
    stored to the low half, and the upper half is written as zero rather than
    preserved.

    The multiplicands are F16 rather than BF16 because the TIRx CUDA backend's
    `_get_tcgen05_mma_kind` accepts an F16 destination only alongside F16
    operands, so a BF16 spelling of this kernel cannot be built for the GPU and
    would lose the paired hardware measurement. NumSim admits either, matching
    Table 45, and the A/B format axis is already covered by the
    F32-destination cases.
    """

    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((128, 16), "float16", scope="shared", layout=_F16_SMEM_LAYOUT)
    shared_b = T.alloc_buffer((16, 16), "float16", scope="shared", layout=_F16_SMEM_LAYOUT)
    registers = T.alloc_local((16,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[:, :])
        Tx.copy(shared_b[:, :], b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    for column in T.unroll(16):
        registers[column] = seed[warp, lane, column]
    T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
        address[0],
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        registers[4],
        registers[5],
        registers[6],
        registers[7],
        registers[8],
        registers[9],
        registers[10],
        registers[11],
        registers[12],
        registers[13],
        registers[14],
        registers[15],
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instruction_descriptor),
            d_dtype="float16",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=16,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0, 0]),
            ldo=16,
            sdo=16,
            swizzle=1,
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
            address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(1)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        registers[4],
        registers[5],
        registers[6],
        registers[7],
        registers[8],
        registers[9],
        registers[10],
        registers[11],
        registers[12],
        registers[13],
        registers[14],
        registers[15],
        address[0],
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(16):
        output[warp, lane, register] = registers[register]
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def raw_tf32_ts_m128_layout_d_valid_descriptor(
    a: T.Buffer((128, 8), "float32"),
    b_physical: T.Buffer((512,), "uint8"),
    output: T.Buffer((4, 32, 16), "float32"),
    descriptors: T.Buffer((3,), "uint64"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    shared_b = T.alloc_buffer((512,), "uint8", scope="shared")
    a_registers = T.alloc_local((8,), "uint32")
    d_registers = T.alloc_local((16,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_b: T.uint64

    for copy_index in T.serial(4):
        offset_b = warp * 32 + lane + copy_index * 128
        shared_b[offset_b] = b_physical[offset_b]
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    for k in T.unroll(8):
        a_registers[k] = T.reinterpret("uint32", a[warp * 32 + lane, k])
    T.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
        address[0],
        a_registers[0],
        a_registers[1],
        a_registers[2],
        a_registers[3],
        a_registers[4],
        a_registers[5],
        a_registers[6],
        a_registers[7],
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cta_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instruction_descriptor),
            d_dtype="float32",
            a_dtype="tf32",
            b_dtype="tf32",
            M=128,
            N=16,
            K=8,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0]),
            ldo=8,
            sdo=16,
            swizzle=0,
        )
        descriptors[0] = T.cast(instruction_descriptor, "uint64")
        descriptors[1] = T.cast(address[0], "uint64")
        descriptors[2] = descriptor_b
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            address[0] + T.uint32(16),
            address[0],
            descriptor_b,
            instruction_descriptor,
            0,
            0,
            0,
            0,
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[1])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
        d_registers[0],
        d_registers[1],
        d_registers[2],
        d_registers[3],
        d_registers[4],
        d_registers[5],
        d_registers[6],
        d_registers[7],
        d_registers[8],
        d_registers[9],
        d_registers[10],
        d_registers[11],
        d_registers[12],
        d_registers[13],
        d_registers[14],
        d_registers[15],
        address[0] + T.uint32(16),
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(16):
        output[warp, lane, register] = T.reinterpret("float32", d_registers[register])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tile_f16_ss_m64_layout_f(
    a: T.Buffer((64, 16), "float16"),
    b: T.Buffer((8, 16), "float16"),
    output: T.Buffer((4, 32, 4), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared", layout=None)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((64, 16), "float16", scope="shared", layout=_F16_SMEM_LAYOUT)
    shared_b = T.alloc_buffer((8, 16), "float16", scope="shared", layout=_F16_SMEM_LAYOUT)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 8),
        allocated_addr=address[0],
    )
    registers = T.alloc_local((4,), "uint32")

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[:, :])
        Tx.copy(shared_b[:, :], b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            shared_a[:, :],
            shared_b[:, :],
            accum=False,
            dispatch="tcgen05",
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
        registers[0], registers[1], registers[2], registers[3], address[0]
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(4):
        output[warp, lane, register] = T.reinterpret("float32", registers[register])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tile_f16_ts_m128_cta_group2_tmem_a(
    a: T.Buffer((2, 128, 16), "float16"),
    b: T.Buffer((2, 16, 16), "float16"),
    output: T.Buffer((2, 4, 32, 32), "float32"),
):
    """One ``Tx.gemm_async(..., cta_group=2)`` whose A operand is TMEM.

    M=128 per CTA (256 across the pair), N=32 across the pair (16 B rows per
    CTA), K=16. TMEM A is staged the way ``flash_attention4`` stages its P
    operand: ``tmem_pool.alloc`` picks the TMEM layout, a ``wg_local_layout``
    view picks the register layout, and a plain ``Tx.wg.copy_async`` moves it.
    """

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2], preferred=[2])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])

    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32", align=4)
    completion = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    shared_b = pool.alloc_tcgen05_mma_AB(
        (16, 16), "float16", swizzle_mode=SwizzleMode.SWIZZLE_32B_ATOM
    )
    pool.commit()

    tmem_pool = T.TMEMPool(pool, total_cols=128, cta_group=2, tmem_addr=tmem_addr)
    a_tmem = tmem_pool.alloc((128, 16), "float16")
    accumulator = tmem_pool.alloc((128, 32), "float32")

    T.ptx.mbarrier.init.shared.b64(T.address_of(completion[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    tmem_pool.commit()

    if warp == 0 and lane == 0:
        Tx.copy(shared_b[:, :], b[cta, :, :])

    # Each CTA stages the 128 A rows it owns into its own TMEM.
    a_reg_buf_f32: T.f32[8]
    a_reg_buf = T.decl_buffer((16,), dtype="float16", data=a_reg_buf_f32.data)
    a_reg = a_reg_buf.view(128, 16, layout=wg_local_layout(16))
    Tx.wg.copy(a_reg[:, :], a[cta, :, :])
    Tx.wg.copy_async(a_tmem[:, :], a_reg[:, :])
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if cta == 0 and warp == 0 and lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            a_tmem[:, :],
            shared_b[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=2,
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(completion[0])
        )
    if cta == 0 and warp == 0:
        T.cuda.mbarrier_wait(T.address_of(completion[0]), 0)
    T.cuda.cluster_sync()

    out_buf_f32: T.f32[32]
    out_reg = out_buf_f32.view(128, 32, layout=wg_local_layout(32))
    Tx.wg.copy_async(out_reg[:, :], accumulator[:, :])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for col in T.serial(32):
        output[cta, warp, lane, col] = out_buf_f32[col]
    T.cuda.cluster_sync()
    tmem_pool.dealloc()


@T.prim_func
def tile_bf16_ss_m64_layout_f(
    a: T.Buffer((64, 16), "bfloat16"),
    b: T.Buffer((8, 16), "bfloat16"),
    output: T.Buffer((4, 32, 4), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared", layout=None)
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((64, 16), "bfloat16", scope="shared", layout=_F16_SMEM_LAYOUT)
    shared_b = T.alloc_buffer((8, 16), "bfloat16", scope="shared", layout=_F16_SMEM_LAYOUT)
    accumulator = T.decl_buffer(
        (64, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("F", 64, 8),
        allocated_addr=address[0],
    )
    registers = T.alloc_local((4,), "uint32")

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[:, :])
        Tx.copy(shared_b[:, :], b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            shared_a[:, :],
            shared_b[:, :],
            accum=False,
            dispatch="tcgen05",
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
        registers[0], registers[1], registers[2], registers[3], address[0]
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(4):
        output[warp, lane, register] = T.reinterpret("float32", registers[register])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tile_mxf8f6f4_e4m3_e8m0_ss_m128_layout_d(
    a: T.Buffer((128, 32), "float8_e4m3fn"),
    b: T.Buffer((8, 32), "float8_e4m3fn"),
    scale_a: T.Buffer((128, 4), "float8_e8m0fnu"),
    scale_b: T.Buffer((128, 4), "float8_e8m0fnu"),
    output: T.Buffer((4, 32, 8), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared", layout=None)
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((128, 32), "float8_e4m3fn", scope="shared", layout=_FP8_A_SMEM_LAYOUT)
    shared_b = T.alloc_buffer((8, 32), "float8_e4m3fn", scope="shared", layout=_FP8_B_SMEM_LAYOUT)
    shared_scale_a = T.alloc_buffer(
        (128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SCALE_SMEM_LAYOUT
    )
    shared_scale_b = T.alloc_buffer(
        (128, 4), "float8_e8m0fnu", scope="shared", layout=_MXF8_SCALE_SMEM_LAYOUT
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=address[0],
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=_MXF8_SCALE_TMEM_LAYOUT,
        allocated_addr=address[0] + T.uint32(16),
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        layout=_MXF8_SCALE_TMEM_LAYOUT,
        allocated_addr=address[0] + T.uint32(20),
    )
    registers = T.alloc_local((8,), "uint32")

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[:, :])
        Tx.copy(shared_b[:, :], b[:, :])
        Tx.copy(shared_scale_a[:, :], scale_a[:, :])
        Tx.copy(shared_scale_b[:, :], scale_b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        Tx.copy_async(scale_a_tmem[:, :], shared_scale_a[:, :], cta_group=1)
        Tx.copy_async(scale_b_tmem[:, :], shared_scale_b[:, :], cta_group=1)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            shared_a[:, :],
            shared_b[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[1])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        registers[4],
        registers[5],
        registers[6],
        registers[7],
        address[0],
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(8):
        output[warp, lane, register] = T.reinterpret("float32", registers[register])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tile_nvfp4_e2m1_e4m3_ss_m128_layout_d(
    a_packed: T.Buffer((128, 32), "uint8"),
    b_packed: T.Buffer((8, 32), "uint8"),
    scale_a: T.Buffer((128, 4), "float8_e4m3fn"),
    scale_b: T.Buffer((128, 4), "float8_e4m3fn"),
    output: T.Buffer((4, 32, 8), "float32"),
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared", layout=None)
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    shared_a_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_FP4_A_SMEM_LAYOUT, align=1024
    )
    shared_b_packed = T.alloc_buffer(
        (8, 32), "uint8", scope="shared", layout=_FP4_B_SMEM_LAYOUT, align=1024
    )
    shared_a = shared_a_packed.view("float4_e2m1fn")
    shared_b = shared_b_packed.view("float4_e2m1fn")
    shared_scale_a = T.alloc_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="shared",
        layout=_NVFP4_SCALE_SMEM_LAYOUT,
        align=1024,
    )
    shared_scale_b = T.alloc_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="shared",
        layout=_NVFP4_SCALE_SMEM_LAYOUT,
        align=1024,
    )
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=address[0],
    )
    scale_a_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=_NVFP4_SCALE_TMEM_LAYOUT,
        allocated_addr=address[0] + T.uint32(16),
    )
    scale_b_tmem = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=_NVFP4_SCALE_TMEM_LAYOUT,
        allocated_addr=address[0] + T.uint32(24),
    )
    registers = T.alloc_local((8,), "uint32")

    if warp == 0 and lane == 0:
        Tx.copy(shared_a_packed[:, :], a_packed[:, :])
        Tx.copy(shared_b_packed[:, :], b_packed[:, :])
        Tx.copy(shared_scale_a[:, :], scale_a[:, :])
        Tx.copy(shared_scale_b[:, :], scale_b[:, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        Tx.copy_async(scale_a_tmem[:, :], shared_scale_a[:, :], cta_group=1)
        Tx.copy_async(scale_b_tmem[:, :], shared_scale_b[:, :], cta_group=1)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            shared_a[:, :],
            shared_b[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[1])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
        registers[0],
        registers[1],
        registers[2],
        registers[3],
        registers[4],
        registers[5],
        registers[6],
        registers[7],
        address[0],
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(8):
        output[warp, lane, register] = T.reinterpret("float32", registers[register])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


def _layout_f_registers(logical: np.ndarray) -> np.ndarray:
    result = np.empty((4, 32, 4), dtype=np.float32)
    for warp in range(4):
        for lane in range(32):
            row_in_half = lane // 4
            column_pair = lane % 4
            for register in range(4):
                row = warp * 16 + (register // 2) * 8 + row_in_half
                column = column_pair * 2 + register % 2
                result[warp, lane, register] = logical[row, column]
    return result


def _decode_e4m3(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (bits & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    return np.where((bits & np.uint8(0x80)) != 0, -magnitude, magnitude).astype(np.float32)


def _decode_e5m2(bits: np.ndarray) -> np.ndarray:
    """Independent E5M2 reference: IEEE-shaped, five exponent bits, two mantissa bits."""

    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> np.uint8(2)) & np.uint8(0x1F)).astype(np.int16)
    mantissa = (bits & np.uint8(0x3)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(4), exponent - 15)
    subnormal = np.ldexp(mantissa / np.float32(4), -14)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    if np.any(exponent == 0x1F):
        raise ValueError("E5M2 reference inputs must stay finite")
    return np.where((bits & np.uint8(0x80)) != 0, -magnitude, magnitude).astype(np.float32)


def _dot_increasing_k(
    a: np.ndarray, b: np.ndarray, initial: np.ndarray | None = None
) -> np.ndarray:
    result = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    for row in range(a.shape[0]):
        for column in range(b.shape[0]):
            accumulator = np.float32(0 if initial is None else initial[row, column])
            for k in range(a.shape[1]):
                accumulator = np.float32(
                    np.float64(a[row, k]) * np.float64(b[column, k]) + np.float64(accumulator)
                )
            result[row, column] = accumulator
    return result


def _bfloat16_bits(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _f16_inputs() -> tuple[np.ndarray, np.ndarray]:
    return (
        (np.arange(64 * 16).reshape(64, 16) % 7 - 3).astype(np.float16),
        (np.arange(8 * 16).reshape(8, 16) % 5 - 2).astype(np.float16),
    )


def _e4m3_inputs() -> tuple[np.ndarray, np.ndarray]:
    finite = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    rows_a = np.arange(64, dtype=np.int64)[:, None]
    rows_b = np.arange(8, dtype=np.int64)[:, None]
    k = np.arange(32, dtype=np.int64)[None, :]
    return (
        finite[(rows_a * 3 + k * 5 + 1) % len(finite)],
        finite[(rows_b * 2 + k * 3 + 3) % len(finite)],
    )


# E5M2 payloads whose E4M3 reading is a different finite number, so a decoder
# mix-up cannot pass by coincidence. 0x3C is 1.0 as E5M2 and 1.5 as E4M3.
# 0x01..0x03 are the E5M2 subnormals (mantissa over 2**-16) and 0x04 is the
# smallest E5M2 normal, 2**-14; the two ranges use different decoder branches
# and E4M3 reads all four as subnormals of entirely different magnitude.
_E5M2_FINITE = np.array(
    [0x00, 0x01, 0x02, 0x03, 0x04, 0x38, 0x3C, 0x3D, 0x40, 0xBC, 0xC0], dtype=np.uint8
)


def _e5m2_inputs(a_rows: int, b_rows: int) -> tuple[np.ndarray, np.ndarray]:
    rows_a = np.arange(a_rows, dtype=np.int64)[:, None]
    rows_b = np.arange(b_rows, dtype=np.int64)[:, None]
    k = np.arange(32, dtype=np.int64)[None, :]
    return (
        _E5M2_FINITE[(rows_a * 3 + k * 5 + 1) % len(_E5M2_FINITE)],
        _E5M2_FINITE[(rows_b * 2 + k * 3 + 3) % len(_E5M2_FINITE)],
    )


def make_raw_e5m2_arguments() -> dict[str, Any]:
    a, b = _e5m2_inputs(64, 8)
    return {
        "a": a,
        "b": b,
        "output": np.zeros((4, 32, 4), dtype=np.float32),
        "descriptors": np.zeros(3, dtype=np.uint64),
    }


def make_raw_e4m3_e5m2_arguments() -> dict[str, Any]:
    a, _unused = _e4m3_inputs()
    _unused_a, b = _e5m2_inputs(64, 8)
    return {
        "a": a,
        "b": b,
        "output": np.zeros((4, 32, 4), dtype=np.float32),
        "descriptors": np.zeros(3, dtype=np.uint64),
    }


def make_raw_f16_destination_arguments() -> dict[str, Any]:
    """Inputs that pin both the accumulator width and the store's rounding.

    The addend is 1024.0, where binary16 has a ULP of 1, and 31 of the 32
    products are 0.25 (the k=0 column is zero). The exact sum is 1031.75, which
    binary16 cannot represent:

      * f32 accumulation, one round-to-nearest convert on store -> 1032
      * f32 accumulation, truncating convert on store           -> 1031
      * rounding the accumulator to binary16 between K steps    -> 1024,
        because each 0.25 addend vanishes against a ULP of 1

    An all-32-product version would sum to exactly 1032 and never exercise the
    store's rounding at all. The upper half of each seeded word is non-zero so
    a preserved-vs-zeroed store is visible in the raw output bytes.
    """

    a = np.full((128, 32), 0x3C, dtype=np.uint8)  # E5M2 1.0
    b = np.full((16, 32), 0x28, dtype=np.uint8)  # E4M3 0.25
    b[:, 0] = 0x00  # leave 31 contributing products, so the sum is not exact
    seed_low = int(np.array(1024.0, dtype=np.float16).view(np.uint16))
    seed = np.full((128, 16), (0xBEEF << 16) | seed_low, dtype=np.uint32)
    return {
        "a": a,
        "b": b,
        "seed": seed.reshape(4, 32, 16).copy(),
        "output": np.zeros((4, 32, 16), dtype=np.uint32),
    }


def make_raw_f16_kind_destination_arguments() -> dict[str, Any]:
    """The `.kind::f16` twin of `make_raw_f16_destination_arguments`.

    Same probe, one K shorter because `.kind::f16` is a K=16 instruction: the
    addend is 1024.0, where binary16 has a ULP of 1, and 15 of the 16 products
    are 0.25 (the k=0 column is zero). The exact sum is 1027.75, which
    binary16 cannot represent:

      * f32 accumulation, one round-to-nearest convert on store -> 1028
      * f32 accumulation, truncating convert on store           -> 1027
      * rounding the accumulator to binary16 between K steps    -> 1024,
        because each 0.25 addend vanishes against a ULP of 1

    Both multiplicands are exact in binary16, so the operand codec contributes
    no rounding of its own. The upper half of each seeded word is non-zero so a
    preserved-vs-zeroed store is visible in the raw output bytes.
    """

    a = np.full((128, 16), 1.0, dtype=np.float16)
    b = np.full((16, 16), 0.25, dtype=np.float16)
    b[:, 0] = 0.0  # leave 15 contributing products, so the sum is not exact
    seed_low = int(np.array(1024.0, dtype=np.float16).view(np.uint16))
    seed = np.full((128, 16), (0xBEEF << 16) | seed_low, dtype=np.uint32)
    return {
        "a": a,
        "b": b,
        "seed": seed.reshape(4, 32, 16).copy(),
        "output": np.zeros((4, 32, 16), dtype=np.uint32),
    }


def _make_tile_f16_arguments() -> dict[str, Any]:
    a, b = _f16_inputs()
    return {"a": a, "b": b, "output": np.zeros((4, 32, 4), dtype=np.float32)}


def _make_raw_f16_arguments() -> dict[str, Any]:
    return {
        **_make_tile_f16_arguments(),
        "descriptors": np.zeros(3, dtype=np.uint64),
    }


def _make_tile_bf16_arguments() -> dict[str, Any]:
    a = (np.arange(64 * 16).reshape(64, 16) % 9 - 4).astype(np.float32)
    b = (np.arange(8 * 16).reshape(8, 16) % 7 - 3).astype(np.float32)
    return {
        "a": PairedBuffer(_bfloat16_bits(a), "bfloat16"),
        "b": PairedBuffer(_bfloat16_bits(b), "bfloat16"),
        "output": np.zeros((4, 32, 4), dtype=np.float32),
    }


def _make_raw_e4m3_arguments() -> dict[str, Any]:
    a, b = _e4m3_inputs()
    return {
        "a": PairedBuffer(a, "float8_e4m3fn"),
        "b": PairedBuffer(b, "float8_e4m3fn"),
        "output": np.zeros((4, 32, 4), dtype=np.float32),
        "descriptors": np.zeros(3, dtype=np.uint64),
    }


def _make_mxf8_arguments() -> dict[str, Any]:
    finite = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    rows_a = np.arange(128, dtype=np.int64)[:, None]
    rows_b = np.arange(8, dtype=np.int64)[:, None]
    k = np.arange(32, dtype=np.int64)[None, :]
    a = finite[(rows_a * 2 + k * 3 + 1) % len(finite)]
    b = finite[(rows_b * 5 + k * 2 + 2) % len(finite)]
    scale_a = np.repeat(np.array([126, 127, 128, 127], dtype=np.uint8)[rows_a % 4], 4, axis=1)
    scale_b = np.full((128, 4), 127, dtype=np.uint8)
    scale_b[:8, :] = np.repeat(
        np.array([128, 127, 126, 127], dtype=np.uint8)[rows_b % 4], 4, axis=1
    )
    return {
        "a": PairedBuffer(a, "float8_e4m3fn"),
        "b": PairedBuffer(b, "float8_e4m3fn"),
        "scale_a": PairedBuffer(scale_a, "float8_e8m0fnu"),
        "scale_b": PairedBuffer(scale_b, "float8_e8m0fnu"),
        "output": np.zeros((4, 32, 8), dtype=np.float32),
    }


def _make_nvfp4_arguments() -> dict[str, Any]:
    a_codes = np.resize(
        np.array([0x0, 0x1, 0x2, 0x3, 0x7, 0x9, 0xA, 0xF], dtype=np.uint8),
        (128, 64),
    )
    b_codes = np.resize(
        np.array([0x1, 0x2, 0x4, 0x7, 0x9, 0xB], dtype=np.uint8),
        (8, 64),
    )
    scale_codes = np.array([0x30, 0x38, 0x40, 0x48], dtype=np.uint8)
    rows = np.arange(128, dtype=np.int64)[:, None]
    groups = np.arange(4, dtype=np.int64)[None, :]
    return {
        "a_packed": (a_codes[:, 0::2] | (a_codes[:, 1::2] << np.uint8(4))).astype(np.uint8),
        "b_packed": (b_codes[:, 0::2] | (b_codes[:, 1::2] << np.uint8(4))).astype(np.uint8),
        "scale_a": PairedBuffer(scale_codes[(rows + groups) % len(scale_codes)], "float8_e4m3fn"),
        "scale_b": PairedBuffer(
            scale_codes[(rows * 3 + groups * 3 + 1) % len(scale_codes)],
            "float8_e4m3fn",
        ),
        "output": np.zeros((4, 32, 8), dtype=np.float32),
    }


def _make_tf32_arguments() -> dict[str, Any]:
    rows = np.arange(128, dtype=np.float32)[:, None]
    columns = np.arange(16, dtype=np.float32)[:, None]
    k = np.arange(8, dtype=np.float32)[None, :]
    a = (rows % np.float32(7) - np.float32(3)) * np.float32(0.25)
    a = a + k * np.float32(0.125)
    a = a + ((rows + k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b = (columns % np.float32(5) - np.float32(2)) * np.float32(0.5)
    b = b - k * np.float32(0.25)
    b = b + ((columns + np.float32(2) * k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    logical_b_bytes = np.ascontiguousarray(b).view(np.uint8).reshape(16, 32)
    b_physical = np.zeros(512, dtype=np.uint8)
    for row in range(16):
        for byte_in_row in range(32):
            atom = byte_in_row // 16
            byte_in_atom = byte_in_row % 16
            physical_offset = (row % 8) * 16 + (row // 8) * 256 + atom * 128 + byte_in_atom
            b_physical[physical_offset] = logical_b_bytes[row, byte_in_row]
    return {
        "a": a,
        "b_physical": b_physical,
        "output": np.zeros((4, 32, 16), dtype=np.float32),
        "descriptors": np.zeros(3, dtype=np.uint64),
    }


def _make_tile_cta_group2_tmem_a_arguments() -> dict[str, Any]:
    """Per-CTA A and B shards, each CTA independently defined."""

    rows = np.arange(128, dtype=np.float32)[:, None]
    k = np.arange(16, dtype=np.float32)[None, :]
    n_rows = np.arange(16, dtype=np.float32)[:, None]

    a = np.empty((2, 128, 16), dtype=np.float16)
    b = np.empty((2, 16, 16), dtype=np.float16)
    for cta in range(2):
        shard = np.float32(1 + cta)
        a[cta] = (
            (rows % np.float32(5) - np.float32(2)) * np.float32(0.5) * shard
            + (k % np.float32(3) - np.float32(1)) * np.float32(0.25)
        ).astype(np.float16)
        b[cta] = (
            (n_rows % np.float32(4) - np.float32(2)) * np.float32(0.25)
            - shard * np.float32(0.5)
            + (k % np.float32(5) - np.float32(2)) * np.float32(0.125)
        ).astype(np.float16)

    return {
        "a": a,
        "b": b,
        "output": np.zeros((2, 4, 32, 32), dtype=np.float32),
    }


@dataclass(frozen=True)
class Tcgen05MmaFormCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...] = ("output",)
    max_ulp: int = 0


TCGEN05_MMA_FORM_CASES = (
    Tcgen05MmaFormCase("tile_f16_layout_f", tile_f16_ss_m64_layout_f, _make_tile_f16_arguments),
    Tcgen05MmaFormCase(
        "raw_f16_valid_descriptor",
        raw_f16_ss_m64_layout_f_valid_descriptor,
        _make_raw_f16_arguments,
    ),
    Tcgen05MmaFormCase("tile_bf16_layout_f", tile_bf16_ss_m64_layout_f, _make_tile_bf16_arguments),
    Tcgen05MmaFormCase(
        "raw_e4m3_valid_descriptor",
        raw_e4m3_ss_m64_layout_f_valid_descriptor,
        _make_raw_e4m3_arguments,
    ),
    Tcgen05MmaFormCase(
        "tile_mxf8_block_scaled",
        tile_mxf8f6f4_e4m3_e8m0_ss_m128_layout_d,
        _make_mxf8_arguments,
    ),
    Tcgen05MmaFormCase(
        "tile_nvfp4_block_scaled",
        tile_nvfp4_e2m1_e4m3_ss_m128_layout_d,
        _make_nvfp4_arguments,
    ),
    Tcgen05MmaFormCase(
        "raw_tf32_valid_descriptor",
        raw_tf32_ts_m128_layout_d_valid_descriptor,
        _make_tf32_arguments,
        max_ulp=16,
    ),
    Tcgen05MmaFormCase(
        "tile_cta_group2_tmem_a",
        tile_f16_ts_m128_cta_group2_tmem_a,
        _make_tile_cta_group2_tmem_a_arguments,
    ),
    Tcgen05MmaFormCase(
        "raw_e5m2_valid_descriptor",
        raw_e5m2_ss_m64_layout_f_valid_descriptor,
        make_raw_e5m2_arguments,
    ),
    Tcgen05MmaFormCase(
        "raw_e4m3_e5m2_valid_descriptor",
        raw_e4m3_e5m2_ss_m64_layout_f_valid_descriptor,
        make_raw_e4m3_e5m2_arguments,
    ),
    Tcgen05MmaFormCase(
        "raw_e5m2_e4m3_f16_destination",
        raw_e5m2_e4m3_f16_d_ss_m128_layout_d,
        make_raw_f16_destination_arguments,
    ),
    Tcgen05MmaFormCase(
        "raw_f16_kind_f16_destination",
        raw_f16_f16_d_ss_m128_layout_d,
        make_raw_f16_kind_destination_arguments,
    ),
)


def e5m2_layout_f_reference(
    arguments: Mapping[str, Any],
    *,
    a_dtype: str,
    b_dtype: str,
    decode_as_e4m3: bool = False,
    flush_subnormals: bool = False,
) -> np.ndarray:
    """Independent oracle for the E5M2 layout-F cases.

    Two negative controls share this builder. `decode_as_e4m3` reads the same
    E5M2 bytes with the E4M3 decoder, which is what a mis-specialized lowering
    would compute. `flush_subnormals` zeroes every operand below 2**-14, which
    is what the model would compute if the tensor core flushed subnormal FP8
    inputs -- a hardware behaviour the PTX ISA does not rule out.
    """

    def operand(name: str, dtype: str) -> np.ndarray:
        bits = arguments[name]
        if dtype == "float8_e5m2" and not decode_as_e4m3:
            values = _decode_e5m2(bits)
        else:
            values = _decode_e4m3(bits)
        if flush_subnormals:
            smallest_normal = np.float32(2.0**-14 if dtype == "float8_e5m2" else 2.0**-6)
            values = np.where(np.abs(values) < smallest_normal, np.float32(0), values)
        return values.astype(np.float32)

    a = operand("a", a_dtype)
    b = operand("b", b_dtype)
    return _layout_f_registers(_dot_increasing_k(a, b))


def f16_kind_destination_reference(
    arguments: Mapping[str, Any], *, per_step_f16: bool = False
) -> np.ndarray:
    """Independent oracle for the `.kind::f16` float16 destination.

    Mirrors `f16_destination_reference` with F16 multiplicands; the
    `per_step_f16` negative control has the same meaning.
    """

    a = np.asarray(arguments["a"], dtype=np.float32)
    b = np.asarray(arguments["b"], dtype=np.float32)
    seed = arguments["seed"].reshape(128, 16)
    addend = np.ascontiguousarray(seed & np.uint32(0xFFFF)).astype(np.uint16)
    addend = addend.view(np.float16).astype(np.float32)
    if per_step_f16:
        product = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
        for row in range(a.shape[0]):
            for column in range(b.shape[0]):
                accumulator = addend[row, column]
                for k in range(a.shape[1]):
                    accumulator = np.float32(
                        np.float16(np.float32(a[row, k]) * np.float32(b[column, k]) + accumulator)
                    )
                product[row, column] = accumulator
    else:
        product = _dot_increasing_k(a, b, addend)
    words = product.astype(np.float16).view(np.uint16).astype(np.uint32)
    return words.reshape(4, 32, 16)


def f16_destination_reference(
    arguments: Mapping[str, Any], *, per_step_f16: bool = False
) -> np.ndarray:
    """Independent oracle for the float16-destination case, as raw TMEM words.

    `per_step_f16` is the negative control: it rounds the accumulator to
    binary16 after every K step, which is the alternative hardware hypothesis
    the B200 measurement rules out.
    """

    a = _decode_e5m2(arguments["a"])
    b = _decode_e4m3(arguments["b"])
    seed = arguments["seed"].reshape(128, 16)
    addend = np.ascontiguousarray(seed & np.uint32(0xFFFF)).astype(np.uint16)
    addend = addend.view(np.float16).astype(np.float32)
    if per_step_f16:
        product = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
        for row in range(a.shape[0]):
            for column in range(b.shape[0]):
                accumulator = addend[row, column]
                for k in range(a.shape[1]):
                    accumulator = np.float32(
                        np.float16(np.float32(a[row, k]) * np.float32(b[column, k]) + accumulator)
                    )
                product[row, column] = accumulator
    else:
        product = _dot_increasing_k(a, b, addend)
    words = product.astype(np.float16).view(np.uint16).astype(np.uint32)
    return words.reshape(4, 32, 16)
