from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import R, S, TLane, TCol, TileLayout


# Spell out the PTX scale-factor placement instead of importing TIRx's
# production layout helpers.  For 128 rows and four scale bytes per row, SMEM
# exposes four M-within-lane rows across 32 lanes; TMEM packs the four bytes in
# one column while replicating each lane's scale group to its four TMEM rows.
_MXF4_SMEM_LAYOUT = TileLayout(S[(4, 32, 2, 2) : (4, 16, 2, 1)])
_MXF4_TMEM_ATOM = TileLayout(S[(32, 2) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])
_MXF4_TMEM_OUTER = TileLayout(S[(4, 2) : (4 @ TCol, 2 @ TCol)])
_MXF4_TMEM_LAYOUT = _MXF4_TMEM_ATOM.direct_sum(
    _MXF4_TMEM_OUTER,
    left_shape=[4, 2],
    right_shape=[32, 2],
)
_MIXED_SMEM_LAYOUT = TileLayout(S[(4, 32, 4) : (4, 16, 1)])
_MIXED_TMEM_ATOM = TileLayout(S[(32, 1) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])
_MIXED_TMEM_OUTER = TileLayout(S[(4, 4) : (4 @ TCol, 1 @ TCol)])
_MIXED_TMEM_LAYOUT = _MIXED_TMEM_ATOM.direct_sum(
    _MIXED_TMEM_OUTER,
    left_shape=[4, 4],
    right_shape=[32, 1],
)


@T.prim_func
def tcgen05_block_scaled_mxf4(
    output: T.Buffer((4, 32, 16), "float32"),
):
    """One cta_group::1 dense MXF4 block-scaled MMA."""

    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    tmem_address = T.alloc_buffer((1,), "uint32", scope="shared", layout=None)
    completion = T.alloc_buffer((2,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((512,), "uint8", scope="shared")
    shared_scale_a = T.alloc_buffer((128, 4), "uint8", scope="shared", layout=_MXF4_SMEM_LAYOUT)
    shared_scale_b = T.alloc_buffer((128, 4), "uint8", scope="shared", layout=_MXF4_SMEM_LAYOUT)
    scale_a = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        allocated_addr=tmem_address[0] + T.uint32(32),
        layout=_MXF4_TMEM_LAYOUT,
    )
    scale_b = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        allocated_addr=tmem_address[0] + T.uint32(52),
        layout=_MXF4_TMEM_LAYOUT,
    )
    registers = T.alloc_local((16,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    row = warp * 32 + lane
    for byte_k in T.unroll(32):
        low_a = 1 + (row * 3 + byte_k * 10) % 7
        high_a = 1 + (row * 3 + byte_k * 10 + 5) % 7
        physical_a = (byte_k // 16) * 2048 + (row // 8) * 128 + (row % 8) * 16 + byte_k % 16
        shared_a[physical_a] = T.cast(low_a + high_a * 16, "uint8")
    if row < 16:
        for byte_k in T.unroll(32):
            low_b = 1 + (row * 5 + byte_k * 4) % 7
            high_b = 1 + (row * 5 + byte_k * 4 + 2) % 7
            physical_b = (byte_k // 16) * 256 + (row // 8) * 128 + (row % 8) * 16 + byte_k % 16
            shared_b[physical_b] = T.cast(low_b + high_b * 16, "uint8")
    for scale_index in T.unroll(4):
        shared_scale_a[row, scale_index] = T.cast(126 + (row + scale_index * 2) % 3, "uint8")
        shared_scale_b[row, scale_index] = T.cast(126 + (row * 2 + scale_index) % 3, "uint8")

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(tmem_address[0]), 64
        )
        if lane == 0:
            for barrier_index in T.unroll(2):
                T.ptx.mbarrier.init.shared.b64(T.address_of(completion[barrier_index]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    # Stage independently laid-out scale matrices through the hardware
    # SMEM->TMEM copy path used by block-scaled MMA.
    if warp == 0 and lane == 0:
        Tx.copy_async(scale_a, shared_scale_a, cta_group=1)
        Tx.copy_async(scale_b, shared_scale_b, cta_group=1)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(completion[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(completion[0]), 0)
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[0]),
            ldo=128,
            sdo=8,
            swizzle=0,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0]),
            ldo=16,
            sdo=8,
            swizzle=0,
        )
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(instruction_descriptor),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=16,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            tmem_address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            tmem_address[0] + T.uint32(32),
            tmem_address[0] + T.uint32(52),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(completion[1])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(completion[1]), 0)
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
        tmem_address[0],
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for column in T.unroll(16):
        output[warp, lane, column] = T.reinterpret("float32", registers[column])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_address[0], 64)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tcgen05_block_scaled_mxf4nvf4(
    output: T.Buffer((4, 32, 16), "float32"),
):
    """Two chained cta_group::1 MXF4NVF4 block-scaled MMAs, `.scale_vec::4X`.

    `.kind::mxf4nvf4` with E4M3 scale factors is `.scale_vec::4X`: one K=64 tile
    reads four scale factors per row over 16-element blocks, spending all four
    bytes of one TMEM word, and the scale-factor ID must therefore be 0. A
    second K tile can only be reached by advancing the scale *address*, which is
    why this case runs two of them -- one tile would pin nothing that the
    `.scale_vec::2X` indexing does not already satisfy.

    The per-tile scale block reuses `_MXF4_TMEM_LAYOUT` because the two
    spellings place one K tile's bytes identically: `(32m + lane, c)` lands in
    byte `c` of TMEM column `m` either way. They differ only in what `c` means
    -- a `(k_tile, value)` pair for 2X, a value index for 4X -- so 4X needs a
    second block four columns later where 2X reuses bytes 2 and 3.

    Scale codes stay in `0x38..0x3e`, i.e. 1.0 to 1.75. The nonzero mantissa is
    the point: a UE8M0 decode of those bytes is off by 2**70, so hardware
    agreeing here pins the E4M3 decode and not merely the addressing.
    """

    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    tmem_address = T.alloc_buffer((1,), "uint32", scope="shared", layout=None)
    completion = T.alloc_buffer((2,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    shared_scale_a0 = T.alloc_buffer((128, 4), "uint8", scope="shared", layout=_MXF4_SMEM_LAYOUT)
    shared_scale_a1 = T.alloc_buffer((128, 4), "uint8", scope="shared", layout=_MXF4_SMEM_LAYOUT)
    shared_scale_b0 = T.alloc_buffer((128, 4), "uint8", scope="shared", layout=_MXF4_SMEM_LAYOUT)
    shared_scale_b1 = T.alloc_buffer((128, 4), "uint8", scope="shared", layout=_MXF4_SMEM_LAYOUT)
    scale_a0 = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        allocated_addr=tmem_address[0] + T.uint32(32),
        layout=_MXF4_TMEM_LAYOUT,
    )
    scale_a1 = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        allocated_addr=tmem_address[0] + T.uint32(36),
        layout=_MXF4_TMEM_LAYOUT,
    )
    scale_b0 = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        allocated_addr=tmem_address[0] + T.uint32(40),
        layout=_MXF4_TMEM_LAYOUT,
    )
    scale_b1 = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        allocated_addr=tmem_address[0] + T.uint32(44),
        layout=_MXF4_TMEM_LAYOUT,
    )
    registers = T.alloc_local((16,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    row = warp * 32 + lane
    for byte_k in T.unroll(64):
        low_a = 1 + (row * 3 + byte_k * 10) % 7
        high_a = 1 + (row * 3 + byte_k * 10 + 5) % 7
        physical_a = (byte_k // 16) * 2048 + (row // 8) * 128 + (row % 8) * 16 + byte_k % 16
        shared_a[physical_a] = T.cast(low_a + high_a * 16, "uint8")
    if row < 16:
        for byte_k in T.unroll(64):
            low_b = 1 + (row * 5 + byte_k * 4) % 7
            high_b = 1 + (row * 5 + byte_k * 4 + 2) % 7
            physical_b = (byte_k // 16) * 256 + (row // 8) * 128 + (row % 8) * 16 + byte_k % 16
            shared_b[physical_b] = T.cast(low_b + high_b * 16, "uint8")
    for scale_index in T.unroll(4):
        shared_scale_a0[row, scale_index] = T.cast(56 + 2 * ((row + scale_index) % 4), "uint8")
        shared_scale_a1[row, scale_index] = T.cast(56 + 2 * ((row + scale_index * 3) % 4), "uint8")
        shared_scale_b0[row, scale_index] = T.cast(56 + 2 * ((row * 2 + scale_index) % 4), "uint8")
        shared_scale_b1[row, scale_index] = T.cast(56 + 2 * ((row + scale_index * 2) % 4), "uint8")

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(tmem_address[0]), 64
        )
        if lane == 0:
            for barrier_index in T.unroll(2):
                T.ptx.mbarrier.init.shared.b64(T.address_of(completion[barrier_index]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        Tx.copy_async(scale_a0, shared_scale_a0, cta_group=1)
        Tx.copy_async(scale_a1, shared_scale_a1, cta_group=1)
        Tx.copy_async(scale_b0, shared_scale_b0, cta_group=1)
        Tx.copy_async(scale_b1, shared_scale_b1, cta_group=1)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(completion[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(completion[0]), 0)
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(instruction_descriptor),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e4m3fn",
            sfb_dtype="float8_e4m3fn",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=16,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[0]),
            ldo=128,
            sdo=8,
            swizzle=0,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0]),
            ldo=16,
            sdo=8,
            swizzle=0,
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"](
            tmem_address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            tmem_address[0] + T.uint32(32),
            tmem_address[0] + T.uint32(40),
            T.ptx.pred(T.uint32(0)),
        )
        # K tile one: same instruction descriptor, because `.scale_vec::4X`
        # pins the scale-factor ID at 0. Only the operand and scale addresses
        # advance -- two 16-byte atoms of A and B, four TMEM columns of SF.
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[4096]),
            ldo=128,
            sdo=8,
            swizzle=0,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[512]),
            ldo=16,
            sdo=8,
            swizzle=0,
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"](
            tmem_address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            tmem_address[0] + T.uint32(36),
            tmem_address[0] + T.uint32(44),
            T.ptx.pred(T.uint32(1)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(completion[1])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(completion[1]), 0)
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
        tmem_address[0],
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for column in T.unroll(16):
        output[warp, lane, column] = T.reinterpret("float32", registers[column])
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_address[0], 64)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tcgen05_block_scaled_mixed_cta2(
    output: T.Buffer((2, 4, 32, 32), "float32"),
):
    """One cta_group::2 E2M1-by-E4M3 block-scaled MMA."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    tmem_address = T.alloc_buffer((1,), "uint32", scope="shared", layout=None)
    completion = T.alloc_buffer((2,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_scale_a = T.alloc_buffer((128, 4), "uint8", scope="shared", layout=_MIXED_SMEM_LAYOUT)
    shared_scale_b = T.alloc_buffer((128, 4), "uint8", scope="shared", layout=_MIXED_SMEM_LAYOUT)
    scale_a = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        allocated_addr=tmem_address[0] + T.uint32(32),
        layout=_MIXED_TMEM_LAYOUT,
    )
    scale_b = T.decl_buffer(
        (128, 4),
        "float8_e8m0fnu",
        scope="tmem",
        allocated_addr=tmem_address[0] + T.uint32(48),
        layout=_MIXED_TMEM_LAYOUT,
    )
    registers = T.alloc_local((32,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    # Each CTA contributes 128 A rows and 16 B rows. Keep both operands
    # dependent on CTA, row, and K, and both scale matrices dependent on row
    # and scale position, with distinct coefficient vectors for A-low, A-high,
    # and B. The descriptor below uses a 128-byte swizzle. A period coprime to
    # two makes each swizzled 16-byte atom bit observable by itself and most
    # multi-bit XORs observable; the all-three-bit same-direction flip still
    # cancels modulo 7 because 16 + 32 + 64 is divisible by 7. This makes the
    # paired GPU oracle sensitive to the production K-major mapping and to
    # realistic row/K/CTA and swizzle permutations. B's modulo-7 linear form
    # is non-collinear with either A form, exposing same-offset A/B miswiring.
    row = warp * 32 + lane
    for copy_index in T.serial(128):
        offset_a = warp * 32 + lane + copy_index * 128
        low_a = 1 + (cta * 2 + row * 3 + copy_index * 5) % 7
        high_a = 1 + (cta * 3 + row * 5 + copy_index * 2) % 7
        shared_a[offset_a] = T.cast(low_a + high_a * 16, "uint8")
    for copy_index in T.serial(64):
        offset_b = warp * 32 + lane + copy_index * 128
        b_code = 0x20 + 4 * ((cta * 5 + row * 3 + copy_index * 4) % 7)
        shared_b[offset_b] = T.cast(b_code, "uint8")
    for scale_index in T.unroll(4):
        shared_scale_a[row, scale_index] = T.cast(
            126 + (row + scale_index * 2) % 3,
            "uint8",
        )
        shared_scale_b[row, scale_index] = T.cast(
            126 + (row * 2 + scale_index) % 3,
            "uint8",
        )

    # cta_group::2 allocation/deallocation is a warp-level collective: the
    # corresponding warp in both peer CTAs participates.
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
            T.address_of(tmem_address[0]), 64
        )
        if lane == 0:
            for barrier_index in T.unroll(2):
                T.ptx.mbarrier.init.shared.b64(T.address_of(completion[barrier_index]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()

    # One leader CTA issues each cta_group::2 copy; the operation populates
    # scale-factor TMEM in both CTAs from identical independently defined data.
    if cta == 0 and warp == 0 and lane == 0:
        Tx.copy_async(scale_a, shared_scale_a, cta_group=2)
        Tx.copy_async(scale_b, shared_scale_b, cta_group=2)
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(completion[0])
        )
    if cta == 0 and warp == 0:
        T.cuda.mbarrier_wait(T.address_of(completion[0]), 0)
    T.cuda.cluster_sync()

    if cta == 0 and warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_a),
            T.address_of(shared_a[0]),
            ldo=0,
            sdo=64,
            swizzle=3,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b),
            T.address_of(shared_b[0]),
            ldo=0,
            sdo=64,
            swizzle=3,
        )
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(instruction_descriptor),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=256,
            N=32,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"](
            tmem_address[0],
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            tmem_address[0] + T.uint32(32),
            tmem_address[0] + T.uint32(48),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(completion[1])
        )
    if cta == 0 and warp == 0:
        T.cuda.mbarrier_wait(T.address_of(completion[1]), 0)
    T.cuda.cluster_sync()

    T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
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
        registers[16],
        registers[17],
        registers[18],
        registers[19],
        registers[20],
        registers[21],
        registers[22],
        registers[23],
        registers[24],
        registers[25],
        registers[26],
        registers[27],
        registers[28],
        registers[29],
        registers[30],
        registers[31],
        tmem_address[0],
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for column in T.unroll(32):
        output[cta, warp, lane, column] = T.reinterpret("float32", registers[column])
    T.cuda.cluster_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_address[0], 64)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()


@dataclass(frozen=True)
class Tcgen05AdvancedCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...] = ("output",)


TCGEN05_ADVANCED_CASES = (
    Tcgen05AdvancedCase(
        "block_scaled_mxf4",
        tcgen05_block_scaled_mxf4,
        lambda: {"output": np.zeros((4, 32, 16), dtype=np.float32)},
    ),
    Tcgen05AdvancedCase(
        "block_scaled_mxf4nvf4",
        tcgen05_block_scaled_mxf4nvf4,
        lambda: {"output": np.zeros((4, 32, 16), dtype=np.float32)},
    ),
    Tcgen05AdvancedCase(
        "block_scaled_mixed_cta2",
        tcgen05_block_scaled_mixed_cta2,
        lambda: {"output": np.zeros((2, 4, 32, 32), dtype=np.float32)},
    ),
)
