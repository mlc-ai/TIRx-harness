from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TCol, TileLayout, TLane, tmem_datapath_layout

from tirx_harness import numsim
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tirx_harness.numsim.transpiler import analyze
from tests.numsim.support.manifest import emitted_calls, resolved_kernel
from tests.numsim.support.tcgen_descriptor import (
    INSTR_DESC,
    INSTR_DESC_BLOCK,
    MATRIX_DESC,
    encode_block_scaled_instr_descriptor_fields,
    encode_dense_instr_descriptor_fields,
    validate_tcgen05_instruction_shape,
)
from tirx_harness.numsim.transpiler.artifact_template import emit_rust_module

TCGEN_DESCRIPTOR_LAYOUT = "<artifact-tcgen-descriptor-layout>"

_TMEM_D_4 = TileLayout(S[(128, 4) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_8 = TileLayout(S[(128, 8) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_16 = TileLayout(S[(128, 16) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_32 = TileLayout(S[(128, 32) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_104 = TileLayout(S[(128, 104) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_136 = TileLayout(S[(128, 136) : (1 @ TLane, 1 @ TCol)])
_TMEM_D_512 = TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)])
_BF16_SMEM_64B = mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_64B_ATOM, (64, 64))
_TMEM_NVF4_24 = TileLayout(S[(128, 24) : (1 @ TLane, 1 @ TCol)])
_PACKED_FP4_SMEM_128X64 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 64))
_PACKED_FP4_SMEM_8X64 = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 64))


@pytest.mark.parametrize(
    ("kind", "cta_group", "m", "n", "k", "sparse"),
    [
        ("f16", 1, 64, 8, 16, False),
        ("f16", 1, 128, 8, 16, False),
        ("f16", 1, 128, 16, 32, True),
        ("f16", 2, 128, 16, 16, False),
        ("f16", 2, 256, 16, 16, False),
        ("tf32", 1, 128, 8, 8, False),
        ("tf32", 2, 256, 32, 16, True),
        ("tf32", 2, 128, 16, 8, False),
        ("f8f6f4", 1, 128, 8, 32, False),
        ("f8f6f4", 2, 128, 32, 32, False),
        ("f8f6f4", 2, 256, 16, 32, False),
        ("i8", 1, 64, 8, 32, False),
        ("i8", 1, 128, 24, 64, True),
        ("mxf4", 1, 128, 8, 64, False),
        ("mxf8f6f4", 2, 256, 16, 64, True),
    ],
)
def test_tcgen_descriptor_shape_table_accepts_tirx_contract(kind, cta_group, m, n, k, sparse):
    validate_tcgen05_instruction_shape(kind, cta_group, m, n, k, sparse)


@pytest.mark.parametrize(
    ("kind", "cta_group", "m", "n", "k", "sparse"),
    [
        ("f16", 1, 128, 12, 16, False),
        ("f16", 2, 128, 8, 16, False),
        ("mxf4", 2, 128, 16, 128, True),
    ],
)
def test_tcgen_descriptor_shape_table_rejects_outside_tirx_contract(
    kind, cta_group, m, n, k, sparse
):
    with pytest.raises(UnsupportedTIRxError, match="invalid .*tcgen05.*descriptor shape"):
        validate_tcgen05_instruction_shape(kind, cta_group, m, n, k, sparse)


@pytest.mark.parametrize(
    ("encoder", "cta_group", "n"),
    [
        ("dense", 1, 8),
        ("dense", 2, 16),
        ("block", 1, 8),
        ("block", 2, 16),
    ],
)
def test_tcgen_8bit_transpose_b_rejects_narrow_n_shapes(encoder, cta_group, n):
    common = dict(
        d_dtype="float32",
        a_dtype="float8_e4m3fn",
        b_dtype="float8_e4m3fn",
        m=128 if cta_group == 1 else 256,
        n=n,
        k=32,
        trans_a=False,
        trans_b=True,
        cta_group=cta_group,
    )
    with pytest.raises(UnsupportedTIRxError, match="8-bit transpose B"):
        if encoder == "dense":
            encode_dense_instr_descriptor_fields(**common)
        else:
            encode_block_scaled_instr_descriptor_fields(
                **common,
                sfa_dtype="float8_e8m0fnu",
                sfb_dtype="float8_e8m0fnu",
            )


@pytest.mark.parametrize(("cta_group", "n"), [(1, 16), (2, 32)])
def test_tcgen_8bit_transpose_b_accepts_ptx_n_granularity(cta_group, n):
    encode_dense_instr_descriptor_fields(
        d_dtype="float32",
        a_dtype="float8_e4m3fn",
        b_dtype="float8_e4m3fn",
        m=128 if cta_group == 1 else 256,
        n=n,
        k=32,
        trans_a=False,
        trans_b=True,
        cta_group=cta_group,
    )


@T.prim_func
def raw_tcgen_descriptor_encode(output: T.Buffer((3,), "uint64")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    matrix_desc: T.uint64
    dense_desc: T.uint32
    block_desc: T.uint32
    if lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(matrix_desc), T.address_of(shared[4]), 1, 8, 3
        )
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(dense_desc),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=128,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(block_desc),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        output[0] = matrix_desc
        output[1] = T.cast(dense_desc, "uint64")
        output[2] = T.cast(block_desc, "uint64")


@T.prim_func
def raw_tcgen_descriptor_encode_to_global(
    matrix_output: T.Buffer((1,), "uint64"), instr_output: T.Buffer((1,), "uint32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "uint32", scope="shared")
    if lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(matrix_output[0]), T.address_of(shared[4]), 1, 8, 3
        )
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instr_output[0]),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=128,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )


@T.prim_func
def raw_tcgen_ldst_32x32b(
    source: T.Buffer((128, 4), "uint32"), output: T.Buffer((128, 4), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    registers = T.alloc_local((4,), "uint32")
    runtime_address = T.alloc_local((1,), "uint32")
    runtime_address[0] = T.uint32(0)
    for col in T.unroll(4):
        tmem[row, col] = source[row, col]
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x4.b32"](
        registers[0], registers[1], registers[2], registers[3], runtime_address[0]
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    T.ptx["tcgen05.st.sync.aligned.32x32b.x4.b32"](
        T.cuda.get_tmem_addr(runtime_address[0], 0, 4),
        registers[0],
        registers[1],
        registers[2],
        registers[3],
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    for col in T.unroll(4):
        output[row, col] = tmem[row, 4 + col]


@T.prim_func
def raw_tcgen_ld_16x256b_mapping(
    source: T.Buffer((128, 8), "uint32"), output: T.Buffer((4, 32, 4), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    registers = T.alloc_local((4,), "uint32")
    for col in T.unroll(8):
        tmem[physical_row, col] = source[physical_row, col]
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.16x256b.x1.b32"](
        registers[0], registers[1], registers[2], registers[3], T.uint32(0)
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for register in T.unroll(4):
        output[warp, lane, register] = registers[register]


@T.prim_func
def raw_tcgen_ld_missing_shape_mappings(
    source: T.Buffer((128, 16), "uint32"), output: T.Buffer((3, 4, 32, 2), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    registers = T.alloc_local((2,), "uint32")
    for col in T.unroll(16):
        tmem[physical_row, col] = source[physical_row, col]
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x2.b32"](
        registers[0], registers[1], T.uint32(0), 2 * (2) if False else 2
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[0, warp, lane, 0] = registers[0]
    output[0, warp, lane, 1] = registers[1]
    T.ptx["tcgen05.ld.sync.aligned.16x64b.x2.b32"](registers[0], registers[1], T.uint32(0))
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[1, warp, lane, 0] = registers[0]
    output[1, warp, lane, 1] = registers[1]
    T.ptx["tcgen05.ld.sync.aligned.16x128b.x1.b32"](registers[0], registers[1], T.uint32(0))
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[2, warp, lane, 0] = registers[0]
    output[2, warp, lane, 1] = registers[1]


@T.prim_func
def raw_tcgen_ld_pack_32x32b(
    source: T.Buffer((128, 4), "uint32"), output: T.Buffer((4, 32, 2), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    registers = T.alloc_local((2,), "uint32")
    for col in T.unroll(4):
        tmem[physical_row, col] = source[physical_row, col]
    T.cuda.cta_sync()
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32"](
        registers[0], registers[1], T.uint32(0)
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[warp, lane, 0] = registers[0]
    output[warp, lane, 1] = registers[1]


@T.prim_func
def raw_tcgen_st_unpack_32x32b(
    source: T.Buffer((4, 32, 2), "uint32"), output: T.Buffer((128, 4), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    registers = T.alloc_local((2,), "uint32")
    for col in T.unroll(4):
        tmem[physical_row, col] = T.uint32(0)
    registers[0] = source[warp, lane, 0]
    registers[1] = source[warp, lane, 1]
    T.ptx["tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32"](
        T.uint32(0), registers[0], registers[1]
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()
    for col in T.unroll(4):
        output[physical_row, col] = tmem[physical_row, col]


@T.prim_func
def raw_tcgen_cp_warpx4(
    source: T.Buffer((32, 4), "uint32"),
    issue: T.int32,
    output: T.Buffer((128, 4), "uint32"),
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((32, 4), "uint32", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64
    if warp == 0:
        for col in T.unroll(4):
            shared[lane, col] = source[lane, col]
    row = T.meta_var(warp * 32 + lane)
    for col in T.unroll(4):
        tmem[row, col] = T.uint32(0)
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), 0, 8, 0
        )
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
            T.uint32(0), descriptor, pred=T.And(lane == 0, issue != 0)
        )
    T.cuda.cta_sync()
    for col in T.unroll(4):
        output[row, col] = tmem[row, col]


@T.prim_func
def raw_tcgen_cp_4x256b(source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for copy_i in T.serial(16):
        shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=8, sdo=0, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def raw_tcgen_cp_warpx2_01_23(
    source: T.Buffer((64, 4), "uint32"), output: T.Buffer((128, 4), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64, 4), "uint32", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64
    if warp < 2:
        row = warp * 32 + lane
        for col in T.unroll(4):
            shared[row, col] = source[row, col]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), ldo=0, sdo=8, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.64x128b.warpx2::01_23"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    row = warp * 32 + lane
    for col in T.unroll(4):
        output[row, col] = tmem[row, col]


@T.prim_func
def raw_tcgen_cp_decompress_b4(
    source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for copy_i in T.serial(16):
        shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=8, sdo=0, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def raw_tcgen_cp_decompress_b6(
    source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for copy_i in T.serial(16):
        shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=8, sdo=0, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def raw_tcgen_cp_128b_base32b(
    source: T.Buffer((512,), "uint8"), output: T.Buffer((4, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((512,), "uint8", scope="shared", align=32)
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for copy_i in T.serial(16):
        shared[lane + copy_i * 32] = source[lane + copy_i * 32]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0]), ldo=0, sdo=0, swizzle=4
        )
        T.ptx["tcgen05.cp.cta_group::1.4x256b"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    if lane == 0:
        for col in T.unroll(8):
            output[warp, col] = tmem[warp * 32, col]


@T.prim_func
def raw_tcgen_cp_128x256b_swizzle(
    source: T.Buffer((128, 32), "uint32"), output: T.Buffer((128, 8), "uint32")
):
    T.device_entry()
    _cta = T.cta_id([1])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((128, 32), "uint32", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    descriptor: T.uint64
    for col in T.unroll(32):
        shared[row, col] = source[row, col]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), 1, 64, 3
        )
        T.ptx["tcgen05.cp.cta_group::1.128x256b"](T.uint32(0), descriptor)
    T.cuda.cta_sync()
    for col in T.unroll(8):
        output[row, col] = tmem[row, col]


@T.prim_func
def raw_tcgen_cp_64x128b_cta_group2(
    source: T.Buffer((2, 64, 32), "uint32"), output: T.Buffer((2, 128, 4), "uint32")
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((64, 32), "uint32", scope="shared")
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64
    if row < 64:
        for col in T.unroll(32):
            shared[row, col] = source[cta, row, col]
    T.cuda.cluster_sync()
    if cta == 0 and warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), 1, 64, 3
        )
        T.ptx["tcgen05.cp.cta_group::2.64x128b.warpx2::02_13"](T.uint32(0), descriptor)
    T.cuda.cluster_sync()
    for col in T.unroll(4):
        output[cta, row, col] = tmem[row, col]


@T.prim_func
def raw_tcgen_cp_128x256b_swizzle64_logical(
    source: T.Buffer((64, 64), "bfloat16"), output: T.Buffer((2, 128, 8), "uint32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    shared = T.alloc_buffer((64, 64), "bfloat16", scope="shared", layout=_BF16_SMEM_64B)
    tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    descriptor: T.uint64

    if warp < 2:
        source_row = warp * 32 + lane
        for col in T.serial(64):
            shared[source_row, col] = source[source_row, col]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), ldo=1, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.cp.cta_group::1.128x256b"](T.uint32(0), descriptor)
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 16]), ldo=1, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.cp.cta_group::1.128x256b"](T.uint32(8), descriptor)
    T.cuda.cta_sync()

    for word in T.unroll(8):
        output[0, physical_row, word] = tmem[physical_row, word]
        output[1, physical_row, word] = tmem[physical_row, 8 + word]


@T.prim_func
def raw_tcgen_cp_descriptor_crosses_short_pool_alias(
    source: T.Buffer((32, 16), "uint8"), output: T.Buffer((128, 4), "uint32")
):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    physical_row = T.meta_var(warp * 32 + lane)
    backing = T.alloc_buffer((640,), "uint8", scope="shared")
    pool = T.meta_var(T.SMEMPool(backing.data))
    pool.move_base_to(64)
    descriptor_root = pool.alloc((16,), "uint8")
    pool.commit()
    tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64

    if warp == 0:
        for byte in T.unroll(16):
            backing[64 + lane * 16 + byte] = source[lane, byte]
    T.cuda.cta_sync()
    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(descriptor_root[0]), ldo=1, sdo=8, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(0), descriptor)
    T.cuda.cta_sync()

    for word in T.unroll(4):
        output[physical_row, word] = tmem[physical_row, word]


@T.prim_func
def raw_tcgen_cp_descriptor_cannot_cross_shared_backings():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    first = T.alloc_buffer((512,), "uint8", scope="shared")
    second = T.alloc_buffer((512,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    descriptor: T.uint64

    for byte in T.unroll(16):
        first[lane * 16 + byte] = T.uint8(lane)
        second[lane * 16 + byte] = T.uint8(lane + 1)
    T.cuda.cta_sync()
    if lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(first[0]), ldo=1, sdo=8, swizzle=0
        )
        descriptor = descriptor + T.uint64(512 // 16)
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
            T.uint32(0),
            descriptor,
        )


@T.prim_func
def raw_tcgen_mma_block_scaled_mxf4_mqa(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((4, 32), "uint32"),
    scale_b_cells: T.Buffer((4, 32), "uint32"),
    issue_second: T.int32,
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        tmem[lane, 128 + row_group] = scale_a_cells[row_group, lane]
        tmem[lane, 132 + row_group] = scale_b_cells[row_group, lane]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.uint32(0)),
            pred=lane == 0,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[32]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[32]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.bitwise_or(desc_i, T.uint32((2 << 29) | (2 << 4))),
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.uint32(1)),
            pred=T.And(lane == 0, issue_second != 0),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_block_scaled_mxf4nvf4_two_k_tiles(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((2, 4, 32), "uint32"),
    scale_b_cells: T.Buffer((2, 32), "uint32"),
    output: T.Buffer((128, 8), "float32"),
    use_ue8m0_scales: T.int32,
):
    """`.kind::mxf4nvf4.block_scale.scale_vec::4X` over two K=64 tiles.

    `.scale_vec::4X` fixes SFA/SFB ID at 0 and spends all four bytes of one
    TMEM word on one K tile, so the second tile advances the scale *address*
    (four TMEM columns for M=128, one for N=8) instead of the descriptor's
    scale-factor ID the way `.kind::mxf4`'s `.scale_vec::2X` does. Descriptor
    bit 23 selects whether those bytes decode as UE8M0 or UE4M3.
    """

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 24), "uint32", scope="tmem", layout=_TMEM_NVF4_24, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    for k_tile in T.unroll(2):
        for row_group in T.unroll(4):
            tmem[lane, 8 + k_tile * 4 + row_group] = scale_a_cells[k_tile, row_group, lane]
        tmem[lane, 16 + k_tile] = scale_b_cells[k_tile, lane]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e4m3fn",
            sfb_dtype="float8_e4m3fn",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=8,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        if use_ue8m0_scales != 0:
            desc_i = T.bitwise_or(desc_i, T.uint32(1 << 23))
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(8),
            T.uint32(16),
            T.ptx.pred(T.uint32(0)),
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[32]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[32]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(12),
            T.uint32(17),
            T.ptx.pred(T.uint32(1)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(8):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def typed_nvfp4_gemm_two_k_tiles(
    left_packed: T.Buffer((128, 64), "uint8"),
    right_packed: T.Buffer((8, 64), "uint8"),
    scale_a: T.Buffer((128, 8), "float8_e4m3fn"),
    scale_b: T.Buffer((8, 8), "float8_e4m3fn"),
    output: T.Buffer((128, 8), "float32"),
):
    """The typed `Tx.gemm_async` oracle for the raw mxf4nvf4 kernel above."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    left_shared_packed = T.alloc_buffer(
        (128, 64), "uint8", scope="shared", layout=_PACKED_FP4_SMEM_128X64
    )
    right_shared_packed = T.alloc_buffer(
        (8, 64), "uint8", scope="shared", layout=_PACKED_FP4_SMEM_8X64
    )
    left_shared = left_shared_packed.view("float4_e2m1fn")
    right_shared = right_shared_packed.view("float4_e2m1fn")
    accumulator = T.decl_buffer(
        (128, 8),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 8),
        allocated_addr=0,
    )
    scale_a_tmem = T.decl_buffer(
        (128, 8),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=8, sf_per_mma=4),
        allocated_addr=16,
    )
    scale_b_tmem = T.decl_buffer(
        (8, 8),
        "float8_e4m3fn",
        scope="tmem",
        layout=sf_tmem_layout(128, SF_K=8, sf_per_mma=4),
        allocated_addr=32,
    )
    if lane == 0:
        Tx.copy(left_shared_packed[:, :], left_packed[:, :])
        Tx.copy(right_shared_packed[:, :], right_packed[:, :])
        for row in T.serial(128):
            for scale_index in T.serial(8):
                scale_a_tmem[row, scale_index] = scale_a[row, scale_index]
        for row in T.serial(8):
            for scale_index in T.serial(8):
                scale_b_tmem[row, scale_index] = scale_b[row, scale_index]
    T.cuda.warp_sync()
    if lane == 0:
        Tx.gemm_async(
            accumulator[:, :],
            left_shared[:, :],
            right_shared[:, :],
            SFA=scale_a_tmem[:, :],
            SFB=scale_b_tmem[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=1,
        )
    T.cuda.warp_sync()
    if lane == 0:
        for row in T.serial(128):
            for col in T.serial(8):
                output[row, col] = accumulator[row, col]


@T.prim_func
def raw_tcgen_mma_block_scaled_mxf4_expression_input_d(
    mode: T.int32,
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((4, 32), "uint32"),
    scale_b_cells: T.Buffer((4, 32), "uint32"),
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        tmem[lane, 128 + row_group] = scale_a_cells[row_group, lane]
        tmem[lane, 132 + row_group] = scale_b_cells[row_group, lane]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.cast(mode == 2, "uint32")),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mxf4_bulk_reads_fail_closed(
    mode: T.int32,
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    scale_a_cells: T.Buffer((4, 32), "uint32"),
    scale_b_cells: T.Buffer((4, 32), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        if mode == 1:
            if offset != 16:
                shared_a[offset] = a_physical[offset]
        else:
            shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        tmem[lane, 128 + row_group] = scale_a_cells[row_group, lane]
        tmem[lane, 132 + row_group] = scale_b_cells[row_group, lane]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=64,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(128),
            T.uint32(132),
            T.ptx.pred(T.cast(mode == 2, "uint32")),
        )


@T.prim_func
def raw_tcgen_mma_e4m3_mqa(
    a_physical: T.Buffer((16384,), "uint8"),
    b_physical: T.Buffer((16384,), "uint8"),
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((16384,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            M=128,
            N=128,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        for k_block in T.unroll(4):
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_a), T.address_of(shared_a[k_block * 32]), ldo=0, sdo=64, swizzle=3
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b), T.address_of(shared_b[k_block * 32]), ldo=0, sdo=64, swizzle=3
            )
            T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                T.uint32(0),
                desc_a,
                desc_b,
                desc_i,
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.ptx.pred(T.cast(k_block, "uint32")),
            )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_tf32_ss(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="tf32",
            b_dtype="tf32",
            M=64,
            N=8,
            K=8,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, col])


@T.prim_func
def raw_tcgen_mma_tf32_ts_predicated(
    a: T.Buffer((64, 8), "float32"),
    b_physical: T.Buffer((4096,), "uint8"),
    output: T.Buffer((64, 32), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64
    desc_b_low: T.uint32
    desc_b_replaced: T.uint64

    for copy_i in T.serial(128):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for k in T.unroll(8):
            tmem[physical_lane, k] = T.reinterpret("uint32", a[row, k])
        for col in T.serial(32):
            tmem[physical_lane, 16 + col] = T.reinterpret("uint32", T.float32(4))
    T.cuda.cta_sync()

    T.cuda.tcgen05.encode_instr_descriptor(
        T.address_of(desc_i),
        d_dtype="float32",
        a_dtype="tf32",
        b_dtype="tf32",
        M=64,
        N=32,
        K=8,
        trans_a=False,
        trans_b=False,
        n_cta_groups=1,
    )
    T.cuda.tcgen05.encode_matrix_descriptor(
        T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
    )
    desc_b_low = T.cast(desc_b, "uint32")
    desc_b_replaced = T.bitwise_or(
        T.bitwise_and(desc_b, T.bitwise_not(T.uint64(0xFFFFFFFF))), T.cast(desc_b_low, "uint64")
    )
    T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
        T.uint32(16),
        T.uint32(0),
        desc_b_replaced,
        desc_i,
        T.uint32(1 << 3),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.ptx.pred(T.uint32(1)),
        pred=T.cast(lane == 7, "uint32"),
    )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.serial(32):
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, 16 + col])


@T.prim_func
def raw_tcgen_mma_e4m3_m64_n8(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
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
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, col])


@T.prim_func
def raw_tcgen_mma_e5m2_e4m3_m64_n8(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
):
    """`kind::f8f6f4` with an E5M2 A and an E4M3 B: independent operand dtypes."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e5m2",
            b_dtype="float8_e4m3fn",
            M=64,
            N=8,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, col])


@T.prim_func
def raw_tcgen_mma_f8f6f4_e5m2_f16_destination_predicated():
    T.device_entry()
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64
    T.cuda.tcgen05.encode_instr_descriptor(
        T.address_of(desc_i),
        d_dtype="float16",
        a_dtype="float8_e5m2",
        b_dtype="float8_e4m3fn",
        M=64,
        N=8,
        K=32,
        trans_a=False,
        trans_b=False,
        n_cta_groups=1,
    )
    T.cuda.tcgen05.encode_matrix_descriptor(
        T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
    )
    T.cuda.tcgen05.encode_matrix_descriptor(
        T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
    )
    T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
        T.uint32(0),
        desc_a,
        desc_b,
        desc_i,
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.ptx.pred(T.uint32(0)),
        pred=T.uint32(1),
    )


@T.prim_func
def raw_tcgen_mma_f8f6f4_f16_destination_m64_n8(
    a_physical: T.Buffer((8192,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    seed: T.Buffer((128, 8), "uint32"),
    output: T.Buffer((64, 8), "uint32"),
):
    """`kind::f8f6f4` accumulating into a float16 destination, read as raw words."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 8), "uint32", scope="tmem", layout=_TMEM_D_8, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.unroll(8):
            tmem[row, col] = seed[row, col]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float16",
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
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = tmem[physical_lane, col]


@T.prim_func
def raw_tcgen_mma_tf32_m128_n16(
    a: T.Buffer((128, 8), "float32"),
    b_physical: T.Buffer((2048,), "uint8"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((2048,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 32), "uint32", scope="tmem", layout=_TMEM_D_32, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64

    for copy_i in T.serial(64):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for k in T.unroll(8):
            tmem[row, k] = T.reinterpret("uint32", a[row, k])
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
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
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
            T.uint32(16),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.unroll(16):
            output[row, col] = T.reinterpret("float32", tmem[row, 16 + col])


@T.prim_func
def raw_tcgen_mma_tf32_rejects_malformed_runtime_descriptor():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64
    forged_desc: T.uint64
    T.cuda.tcgen05.encode_instr_descriptor(
        T.address_of(desc_i),
        d_dtype="float32",
        a_dtype="tf32",
        b_dtype="tf32",
        M=64,
        N=32,
        K=8,
        trans_a=False,
        trans_b=False,
        n_cta_groups=1,
    )
    T.cuda.tcgen05.encode_matrix_descriptor(
        T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
    )
    forged_desc = T.bitwise_or(desc_b, T.uint64(1 << 14))
    T.ptx["tcgen05.mma.cta_group::1.kind::tf32"](
        T.uint32(16),
        T.uint32(0),
        forged_desc,
        desc_i,
        T.uint32(1 << 3),
        T.uint32(0),
        T.uint32(0),
        T.uint32(0),
        T.ptx.pred(T.uint32(0)),
        pred=T.cast(lane == 0, "uint32"),
    )


@T.prim_func
def raw_tcgen_cp_rejects_absolute_shared_descriptor():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 4), "uint32", scope="tmem", layout=_TMEM_D_4, allocated_addr=0)
    for byte in T.unroll(4):
        shared[lane * 4 + byte] = T.uint8(lane)
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
            T.uint32(0),
            T.uint64(1 << 46),
        )


@T.prim_func
def raw_tcgen_mma_bf16_ss(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((1024,), "uint8"),
    output: T.Buffer((64, 8), "float32"),
    output_ws: T.Buffer((64, 8), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    prefix = T.alloc_buffer((128,), "uint8", scope="shared")
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    descriptor_table = T.alloc_buffer((1,), "uint64", scope="shared")
    barriers = T.alloc_buffer((2,), "uint64", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_a_loaded: T.uint64
    desc_b: T.uint64
    prefix[lane] = T.cast(lane, "uint8")
    for copy_i in T.serial(128):
        shared_a[lane + copy_i * 32] = a_physical[lane + copy_i * 32]
    for copy_i in T.serial(32):
        shared_b[lane + copy_i * 32] = b_physical[lane + copy_i * 32]
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
        T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[1]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=8,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx.st.shared.u64(descriptor_table.ptr_to([0]), desc_a)
        T.ptx.ld.shared.u64(desc_a_loaded, descriptor_table.ptr_to([0]))
        T.ptx["tcgen05.mma.cta_group::1.kind::f16.collector::a::discard"](
            T.uint32(0),
            desc_a_loaded,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cta_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        physical_lane = (row // 16) * 32 + row % 16
        for col in T.unroll(8):
            output[row, col] = T.reinterpret("float32", _tmem[physical_lane, col])
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(0),
            desc_a_loaded,
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[1])
        )
    T.cuda.mbarrier_wait(T.address_of(barriers[1]), 0)
    T.cuda.cta_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        for col in T.unroll(8):
            physical_lane = row + (col // 4) * 64
            physical_col = col % 4
            output_ws[row, col] = T.reinterpret("float32", _tmem[physical_lane, physical_col])


@T.prim_func
def raw_tcgen_mma_ws_bf16_ss_layout_e_tail(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((8192,), "uint8"),
    output: T.Buffer((64, 128), "float32"),
):
    """M64 `.ws` writes N halves into the two Layout-E lane banks."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 512), "uint32", scope="tmem", layout=_TMEM_D_512, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64
    for copy_i in T.serial(128):
        shared_a[lane + copy_i * 32] = a_physical[lane + copy_i * 32]
    for copy_i in T.serial(256):
        shared_b[lane + copy_i * 32] = b_physical[lane + copy_i * 32]
    T.cuda.cta_sync()
    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="bfloat16",
            b_dtype="bfloat16",
            M=64,
            N=128,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
        )
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(400),
            desc_a,
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
    T.cuda.cta_sync()
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        for col in T.serial(128):
            physical_lane = row + (col // 64) * 64
            physical_col = 400 + col % 64
            output[row, col] = T.reinterpret("float32", tmem[physical_lane, physical_col])


@T.prim_func
def raw_tcgen_mma_f16_ts_mn_major_n128(
    a_packed: T.Buffer((128, 8), "uint32"),
    b_physical: T.Buffer((16384,), "uint8"),
    output: T.Buffer((128, 128), "float32"),
    output_ws: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_b = T.alloc_buffer((16384,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for packed_k in T.unroll(8):
            tmem[row, packed_k] = a_packed[row, packed_k]
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=128,
            K=16,
            trans_a=False,
            trans_b=True,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b),
            T.address_of(shared_b[0]),
            ldo=512,
            sdo=64,
            swizzle=3,
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[row, col] = T.reinterpret("float32", tmem[row, 8 + col])
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(8),
            T.uint32(0),
            desc_b,
            desc_i,
            T.ptx.pred(T.uint32(0)),
            T.uint64(0),
        )
    T.cuda.cta_sync()
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output_ws[row, col] = T.reinterpret("float32", tmem[row, 8 + col])


@T.prim_func
def raw_tcgen_mma_f16_cta2_datapaths(
    a_ss256_physical: T.Buffer((2, 16384), "uint8"),
    b_ss256_physical: T.Buffer((2, 1024), "uint8"),
    a_ts256_packed: T.Buffer((2, 128, 8), "uint32"),
    b_ts256_physical: T.Buffer((2, 8192), "uint8"),
    a_ss128_physical: T.Buffer((2, 8192), "uint8"),
    b_ss128_physical: T.Buffer((2, 8192), "uint8"),
    output_ss256: T.Buffer((256, 16), "float32"),
    output_ts256: T.Buffer((256, 16), "float32"),
    output_ss128: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a_ss256 = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b_ss256 = T.alloc_buffer((1024,), "uint8", scope="shared")
    shared_b_ts256 = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_a_ss128 = T.alloc_buffer((8192,), "uint8", scope="shared")
    shared_b_ss128 = T.alloc_buffer((8192,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 104), "uint32", scope="tmem", layout=_TMEM_D_104, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_a_ss256[offset] = a_ss256_physical[cta, offset]
    for copy_i in T.serial(32):
        offset = lane + copy_i * 32
        shared_b_ss256[offset] = b_ss256_physical[cta, offset]
    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_b_ts256[offset] = b_ts256_physical[cta, offset]
        shared_a_ss128[offset] = a_ss128_physical[cta, offset]
        shared_b_ss128[offset] = b_ss128_physical[cta, offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for packed_k in T.unroll(8):
            tmem[row, packed_k] = a_ts256_packed[cta, row, packed_k]
    T.cuda.cluster_sync()

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=16,
            K=16,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a_ss256[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b_ss256[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(8),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )

        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=256,
            N=16,
            K=16,
            trans_a=False,
            trans_b=True,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b_ts256[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(24),
            T.uint32(0),
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )

        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float16",
            b_dtype="float16",
            M=128,
            N=128,
            K=16,
            trans_a=True,
            trans_b=True,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a_ss128[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b_ss128[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f16"](
            T.uint32(40),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )
    T.cuda.cluster_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(16):
            output_ss256[cta * 128 + row, col] = T.reinterpret("float32", tmem[row, 8 + col])
            output_ts256[cta * 128 + row, col] = T.reinterpret("float32", tmem[row, 24 + col])
    for row_group in T.unroll(2):
        row = row_group * 32 + lane
        for col in T.serial(128):
            physical_lane = row + 64 * (col // 64)
            physical_col = col % 64
            output_ss128[cta * 64 + row, col] = T.reinterpret(
                "float32", tmem[physical_lane, 40 + physical_col]
            )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta2_k32_without_arch():
    """The architecture-neutral K-major B, N=16 CTA2 form shared by SM100 and SM107."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x10040490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta2_k32_extended_span_without_arch():
    """K=32 does not imply the SM100 address width; architecture does."""

    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_arena = T.alloc_buffer((270336,), "uint8", scope="shared", align=1024)
    shared_a = T.decl_buffer((16384,), "uint8", data=shared_arena.data, scope="shared")
    shared_b = T.decl_buffer(
        (8192,), "uint8", data=shared_arena.data, elem_offset=262144, scope="shared"
    )
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        desc_b = T.bitwise_or(
            T.bitwise_and(desc_b, T.bitwise_not(T.uint64(0x7FFF))),
            T.cast(
                T.shift_right(T.cuda.cvta_generic_to_shared(shared_b.ptr_to([0])), T.uint32(4)),
                "uint64",
            ),
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x10210490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta2_rubin_k64_discard(
    a_physical: T.Buffer((2, 16384), "uint8"),
    b_physical: T.Buffer((2, 8192), "uint8"),
    seed: T.Buffer((2, 128, 128), "float32"),
    output: T.Buffer((2, 128, 128), "float32"),
):
    """Rubin's M256/N128/K64 form with a B descriptor above the SM100 address ceiling."""

    T.func_attr({"tirx.cuda_arch": "sm_107a"})
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_arena = T.alloc_buffer((327936,), "uint8", scope="shared", align=1024)
    shared_a = T.decl_buffer((16384,), "uint8", data=shared_arena.data, scope="shared")
    shared_b = T.decl_buffer(
        (8192,), "uint8", data=shared_arena.data, elem_offset=262144, scope="shared"
    )
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[cta, offset]
    for copy_i in T.serial(256):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[cta, offset]
    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            tmem[row, col] = T.reinterpret("uint32", seed[cta, row, col])
    T.cuda.cluster_sync()

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        # CUTLASS include/cute/arch/mma_sm100_desc.hpp makes this a 15-bit
        # field for SM107A/F. The source kernel likewise patches/adds the low
        # descriptor half rather than calling the SM100-oriented TVM encoder.
        desc_b = T.bitwise_or(
            T.bitwise_and(desc_b, T.bitwise_not(T.uint64(0x7FFF))),
            T.cast(
                T.shift_right(T.cuda.cvta_generic_to_shared(shared_b.ptr_to([0])), T.uint32(4)),
                "uint64",
            ),
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x30210490),
            T.uint32(1 << 5),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(1 << 7),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )
    T.cuda.cluster_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(128):
            output[cta, row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta2_rubin_rejects_descriptor_bit15():
    """Rubin widens the start field through bit 14 only; bit 15 stays reserved."""

    T.func_attr({"tirx.cuda_arch": "sm_107a"})
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((16384,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_a: T.uint64
    desc_b: T.uint64

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared[0]), ldo=0, sdo=64, swizzle=3
        )
        desc_b = T.bitwise_or(desc_b, T.uint64(1 << 15))
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.uint32(0x30210490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(1)),
        )


@T.prim_func
def shared_span_above_18_bits_without_sm107_descriptor():
    """A large ordinary/SM100 shared arena must not inherit Rubin's wider descriptor."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((262160,), "uint8", scope="shared")
    if lane == 0:
        shared[262159] = T.uint8(1)


@T.prim_func
def raw_tcgen_mma_f8f6f4_reserved_operand():
    """Operand format 2 belongs to TF32, not the narrow-float family."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32((2 << 7) | (2 << 10)),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta_group2_reserved_destination():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(0x302104B0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )


@T.prim_func
def raw_tcgen_mma_f8f6f4_cta_group2_invalid_n():
    """MN-major B keeps the CTA2 N=32 granularity, so N=16 is invalid."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(0x10050490),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            T.ptx.pred(T.uint32(0)),
        )


@T.prim_func
def raw_tcgen_mma_mxf8_mn_major(
    a_physical: T.Buffer((4096,), "uint8"),
    b_physical: T.Buffer((4096,), "uint8"),
    output: T.Buffer((128, 16), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((4096,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 512), "uint32", scope="tmem", layout=_TMEM_D_512, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(128):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[offset]
        shared_b[offset] = b_physical[offset]
    for replica in T.unroll(4):
        for row_group in T.unroll(4):
            tmem[replica * 32 + lane, 256 + row_group] = T.uint32(0x7F7F7F7F)
        tmem[replica * 32 + lane, 260] = T.uint32(0x7F7F7F7F)
    T.cuda.cta_sync()

    if lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=1 << 30,
            sfb_tmem_addr=1 << 30,
            M=128,
            N=16,
            K=32,
            trans_a=True,
            trans_b=True,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32((1 << 30) | 256),
            T.uint32((1 << 30) | 260),
            False,
        )
    T.cuda.cta_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(16):
            output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_mxf8_k_major_cta1(
    a_physical: T.Buffer((16384,), "uint8"),
    b_physical: T.Buffer((16384,), "uint8"),
    output: T.Buffer((128, 128), "float32"),
):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((16384,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 512), "uint32", scope="tmem", layout=_TMEM_D_512, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0:
        for copy_i in T.serial(512):
            offset = lane + copy_i * 32
            shared_a[offset] = a_physical[offset]
            shared_b[offset] = b_physical[offset]
        for replica in T.unroll(4):
            for row_group in T.unroll(4):
                tmem[replica * 32 + lane, 256 + row_group] = T.uint32(0x7F7F7F7F)
                tmem[replica * 32 + lane, 260 + row_group] = T.uint32(0x7F7F7F7F)
    T.cuda.cta_sync()

    if (warp == 0) and (lane == 0):
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
            d_dtype="float32",
            a_dtype="float8_e4m3fn",
            b_dtype="float8_e4m3fn",
            sfa_dtype="float8_e8m0fnu",
            sfb_dtype="float8_e8m0fnu",
            sfa_tmem_addr=0,
            sfb_tmem_addr=0,
            M=128,
            N=128,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=1, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=1, sdo=64, swizzle=3
        )
        for kblock in T.unroll(4):
            T.cuda.runtime_instr_desc(T.address_of(desc_i), T.cast(kblock, "uint32"))
            T.ptx["tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32"](
                T.uint32(0),
                desc_a + T.cast(kblock * 2, "uint64"),
                desc_b + T.cast(kblock * 2, "uint64"),
                desc_i,
                T.uint32(256 + kblock * (1 << 30)),
                T.uint32(260 + kblock * (1 << 30)),
                kblock != 0,
            )
    T.cuda.cta_sync()

    row = T.meta_var(warp * 32 + lane)
    for col in T.serial(128):
        output[row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mma_mixed_fp4_fp8_cta_group2(
    a_physical: T.Buffer((2, 16384), "uint8"),
    b_physical: T.Buffer((2, 2048), "uint8"),
    scale_a_cells: T.Buffer((2, 4, 32), "uint32"),
    scale_b_cells: T.Buffer((2, 1, 32), "uint32"),
    output: T.Buffer((2, 128, 32), "float32"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((2048,), "uint8", scope="shared")
    tmem = T.decl_buffer((128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0)
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    for copy_i in T.serial(512):
        offset = lane + copy_i * 32
        shared_a[offset] = a_physical[cta, offset]
    for copy_i in T.serial(64):
        offset = lane + copy_i * 32
        shared_b[offset] = b_physical[cta, offset]
    for replica in T.unroll(4):
        for row_group in T.unroll(4):
            tmem[replica * 32 + lane, 32 + row_group] = scale_a_cells[cta, row_group, lane]
        tmem[replica * 32 + lane, 36] = scale_b_cells[cta, 0, lane]
    T.cuda.cluster_sync()

    if cta == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor_block_scaled(
            T.address_of(desc_i),
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
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"](
            T.uint32(0),
            desc_a,
            desc_b,
            desc_i,
            T.uint32(32),
            T.uint32(36),
            False,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a), T.address_of(shared_a[32]), ldo=0, sdo=64, swizzle=3
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b), T.address_of(shared_b[32]), ldo=0, sdo=64, swizzle=3
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"](
            T.uint32(0),
            desc_a,
            desc_b,
            T.bitwise_or(desc_i, T.uint32((1 << 29) | (1 << 4))),
            T.uint32(32),
            T.uint32(36),
            True,
        )
    T.cuda.cluster_sync()

    for row_group in T.unroll(4):
        row = row_group * 32 + lane
        for col in T.serial(32):
            output[cta, row, col] = T.reinterpret("float32", tmem[row, col])


@T.prim_func
def raw_tcgen_mxf4_ss_cta2_analysis_probe():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _shared = T.alloc_buffer((1,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            False,
        )


@T.prim_func
def raw_tcgen_mxf4_ts_cta1_analysis_probe():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _shared = T.alloc_buffer((1,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
            T.uint32(0),
            T.uint32(0),
            T.uint64(0),
            T.uint32(0),
            T.uint32(0),
            T.uint32(0),
            False,
        )


@T.prim_func
def raw_tcgen_ws_ss_mask_analysis_probe():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _shared = T.alloc_buffer((1,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(0),
            T.uint64(0),
            T.uint64(0),
            T.uint32(1 << 4),
            T.ptx.pred(T.uint32(0)),
            T.uint64(1 << 39),
        )


@T.prim_func
def raw_tcgen_ws_ts_mask_analysis_probe():
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    _shared = T.alloc_buffer((1,), "uint8", scope="shared")
    _tmem = T.decl_buffer((128, 16), "uint32", scope="tmem", layout=_TMEM_D_16, allocated_addr=0)
    if lane == 0:
        T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"](
            T.uint32(0),
            T.uint32(0),
            T.uint64(0),
            T.uint32(1 << 4),
            T.ptx.pred(T.uint32(0)),
            T.uint64(1 << 39),
        )


@dataclass(frozen=True)
class _AcceptedCall:
    op_name: str


_TCGEN_CONTROL_CALLS = frozenset(
    {
        "tirx.ptx.tcgen05_alloc",
        "tirx.ptx.tcgen05_alloc_exclusive",
        "tirx.ptx.tcgen05_commit",
        "tirx.ptx.tcgen05_commit_multicast",
        "tirx.ptx.tcgen05_commit_multicast_width",
        "tirx.ptx.tcgen05_dealloc",
        "tirx.ptx.tcgen05_dealloc_exclusive",
        "tirx.ptx.tcgen05_fence",
        "tirx.ptx.tcgen05_relinquish_alloc_permit",
        "tirx.ptx.tcgen05_wait",
    }
)


def _accepted_calls(func, accept):
    kernel = resolved_kernel(func)
    return [
        _AcceptedCall(str(getattr(entry.node.op, "name", "")))
        for entry in kernel.source_map
        if entry.kind == "Call" and accept(str(getattr(entry.node.op, "name", "")))
    ]


def _raw_calls(func):
    return _accepted_calls(
        func, lambda op_name: op_name in {MATRIX_DESC, INSTR_DESC, INSTR_DESC_BLOCK}
    )


def _resolved_ptx_raw_tcgen_calls(func):
    return _accepted_calls(
        func,
        lambda op_name: (
            op_name.startswith("tirx.ptx.tcgen05_") and op_name not in _TCGEN_CONTROL_CALLS
        ),
    )


def _matrix_descriptor(*, start_16b: int, ldo: int, sdo: int, layout_type: int) -> np.uint64:
    return np.uint64(start_16b | (ldo << 16) | (sdo << 32) | (1 << 46) | (layout_type << 61))


def _dense_descriptor() -> np.uint64:
    value = 0
    value |= 1 << 4  # D=f32
    value |= 1 << 7  # A=bf16
    value |= 1 << 10  # B=bf16
    value |= (128 >> 3) << 17
    value |= (64 >> 4) << 24
    return np.uint64(value)


def _block_descriptor() -> np.uint64:
    value = 0
    value |= 1 << 7  # mxf4 A format
    value |= 1 << 10  # mxf4 B format
    value |= (128 >> 3) << 17
    value |= 1 << 23  # E8M0 scale format
    value |= (128 >> 4) << 24
    return np.uint64(value)


def _swizzle_128_physical(logical: np.ndarray, row_words: int = 32) -> np.ndarray:
    physical = np.zeros((logical.shape[0], row_words), dtype=np.uint32)
    flat = physical.reshape(-1)
    for row in range(logical.shape[0]):
        for word in range(logical.shape[1]):
            unswizzled_byte = row * 128 + word * 4
            atom = unswizzled_byte // 16
            byte_in_atom = unswizzled_byte % 16
            swizzled_atom = atom ^ ((atom & (0x7 << 3)) >> 3)
            flat[(swizzled_atom * 16 + byte_in_atom) // 4] = logical[row, word]
    return physical


def _tcgen_cp_two_atom_physical(atom_bytes: np.ndarray) -> np.ndarray:
    assert atom_bytes.ndim == 3 and atom_bytes.shape[1:] == (2, 16)
    physical = np.zeros(512, dtype=np.uint8)
    for row in range(atom_bytes.shape[0]):
        for atom in range(2):
            offset = row * 16 + atom * 128
            physical[offset : offset + 16] = atom_bytes[row, atom]
    return physical


def _swizzle_128_base32_physical(logical: np.ndarray) -> np.ndarray:
    assert logical.ndim == 2 and logical.shape[1] == 32
    physical = np.zeros(512, dtype=np.uint8)
    for row in range(logical.shape[0]):
        unswizzled_atom = row * 4
        swizzled_atom = unswizzled_atom ^ ((unswizzled_atom & (0x3 << 2)) >> 2)
        offset = swizzled_atom * 32
        physical[offset : offset + 32] = logical[row]
    return physical


def _pack_b4_cp_source(codes: np.ndarray) -> np.ndarray:
    assert codes.ndim == 2 and codes.shape[1] == 32
    atoms = np.zeros((codes.shape[0], 2, 16), dtype=np.uint8)
    for row in range(codes.shape[0]):
        for atom in range(2):
            values = codes[row, atom * 16 : (atom + 1) * 16]
            atoms[row, atom, :8] = values[0::2] | (values[1::2] << np.uint8(4))
    return _tcgen_cp_two_atom_physical(atoms)


def _pack_b6_cp_source(codes: np.ndarray) -> np.ndarray:
    assert codes.ndim == 2 and codes.shape[1] == 32
    atoms = np.zeros((codes.shape[0], 2, 16), dtype=np.uint8)
    for row in range(codes.shape[0]):
        for atom in range(2):
            values = codes[row, atom * 16 : (atom + 1) * 16]
            packed = sum(int(value) << (6 * index) for index, value in enumerate(values))
            atoms[row, atom, :12] = np.frombuffer(packed.to_bytes(12, "little"), dtype=np.uint8)
    return _tcgen_cp_two_atom_physical(atoms)


def _bfloat16_bits(values: np.ndarray) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint32) >> np.uint32(16)
    ).astype(np.uint16)


def _mxf4_kmajor_physical(logical_nibbles: np.ndarray) -> np.ndarray:
    rows, k = logical_nibbles.shape
    assert rows == 128 and k == 128
    packed = logical_nibbles[:, 0::2].astype(np.uint8) | (
        logical_nibbles[:, 1::2].astype(np.uint8) << np.uint8(4)
    )
    return _kmajor_swizzle_physical(packed, swizzle_len=2, sdo=512)


def _kmajor_swizzle_physical(
    logical_bytes: np.ndarray, *, swizzle_len: int, sdo: int
) -> np.ndarray:
    rows, row_bytes = logical_bytes.shape
    physical = np.zeros(((rows + 7) // 8) * sdo, dtype=np.uint8)
    row_stride = 16 << swizzle_len
    swizzle_mask = (1 << swizzle_len) - 1
    for row in range(rows):
        for byte_in_row in range(row_bytes):
            atom = byte_in_row // 16
            byte_in_atom = byte_in_row % 16
            unswizzled = (row % 8) * row_stride + (row // 8) * sdo + atom * 16 + byte_in_atom
            atom_index = unswizzled >> 4
            swizzled_atom = atom_index ^ ((atom_index & (swizzle_mask << 3)) >> 3)
            physical[(swizzled_atom << 4) | byte_in_atom] = logical_bytes[row, byte_in_row]
    return physical


def _kmajor_swizzle_physical_at_start(
    logical_bytes: np.ndarray,
    *,
    start: int,
    swizzle_len: int,
    sdo: int,
    byte_len: int,
) -> np.ndarray:
    physical = np.zeros(byte_len, dtype=np.uint8)
    row_stride = 16 << swizzle_len
    swizzle_mask = (1 << swizzle_len) - 1
    for row in range(logical_bytes.shape[0]):
        for byte_in_row in range(logical_bytes.shape[1]):
            atom = byte_in_row // 16
            byte_in_atom = byte_in_row % 16
            unswizzled = (
                start + (row % 8) * row_stride + (row // 8) * sdo + atom * 16 + byte_in_atom
            )
            atom_index = unswizzled >> 4
            swizzled_atom = atom_index ^ ((atom_index & (swizzle_mask << 3)) >> 3)
            physical_offset = ((swizzled_atom << 4) | byte_in_atom) - start
            if not 0 <= physical_offset < byte_len:
                raise ValueError("swizzled address leaves the selected physical buffer")
            physical[physical_offset] = logical_bytes[row, byte_in_row]
    return physical


def _mnmajor_f16_swizzle_physical(logical_bits: np.ndarray) -> np.ndarray:
    """Encode the PTX canonical MN-major 128B-swizzled f16 layout."""
    assert logical_bits.shape == (128, 16)
    physical = np.zeros(16384, dtype=np.uint8)
    for n in range(128):
        for k in range(16):
            unswizzled = (n % 64) * 2 + (n // 64) * 8192 + (k % 8) * 128 + (k // 8) * 1024
            byte_in_atom = unswizzled & 15
            atom = unswizzled >> 4
            swizzled_atom = atom ^ ((atom & (0x7 << 3)) >> 3)
            offset = (swizzled_atom << 4) | byte_in_atom
            physical[offset : offset + 2] = np.frombuffer(
                np.uint16(logical_bits[n, k]).tobytes(), dtype=np.uint8
            )
    return physical


def _mnmajor_f16_swizzle_physical_cta2(logical_bits: np.ndarray) -> np.ndarray:
    """Encode one CTA's at-most-64-row MN-major f16 operand."""
    assert logical_bits.ndim == 2 and logical_bits.shape[0] <= 64 and logical_bits.shape[1] == 16
    physical = np.zeros(8192, dtype=np.uint8)
    for row in range(logical_bits.shape[0]):
        for k in range(16):
            unswizzled = row * 2 + (k % 8) * 128 + (k // 8) * 1024
            byte_in_atom = unswizzled & 15
            atom = unswizzled >> 4
            swizzled_atom = atom ^ ((atom & (0x7 << 3)) >> 3)
            offset = (swizzled_atom << 4) | byte_in_atom
            physical[offset : offset + 2] = np.frombuffer(
                np.uint16(logical_bits[row, k]).tobytes(), dtype=np.uint8
            )
    return physical


def _mnmajor_f8_swizzle_physical(logical_bits: np.ndarray) -> np.ndarray:
    """Encode the PTX canonical MN-major 128B/16B-atomic 8-bit layout."""
    logical_bits = np.asarray(logical_bits, dtype=np.uint8)
    assert logical_bits.ndim == 2
    assert logical_bits.shape[0] <= 128 and logical_bits.shape[1] in (32, 64)
    physical = np.zeros((logical_bits.shape[1] // 8) * 1024, dtype=np.uint8)
    for row in range(logical_bits.shape[0]):
        for k in range(logical_bits.shape[1]):
            unswizzled = row + (k % 8) * 128 + (k // 8) * 1024
            byte_in_atom = unswizzled & 15
            atom = unswizzled >> 4
            swizzled_atom = atom ^ ((atom & (0x7 << 3)) >> 3)
            physical[(swizzled_atom << 4) | byte_in_atom] = logical_bits[row, k]
    return physical


def _e4m3fn_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (bits & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    return np.where(bits & np.uint8(0x80), -magnitude, magnitude).astype(np.float32)


def _e5m2_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = ((bits >> np.uint8(2)) & np.uint8(0x1F)).astype(np.int16)
    mantissa = (bits & np.uint8(0x3)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(4), exponent - 15)
    subnormal = np.ldexp(mantissa / np.float32(4), -14)
    magnitude = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    assert not np.any(exponent == 0x1F), "E5M2 reference inputs must stay finite"
    return np.where(bits & np.uint8(0x80), -magnitude, magnitude).astype(np.float32)


def _decode_tcgen_tf32_payload(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    return (bits & np.uint32(0xFFFFE000)).view(np.float32)


def _tcgen_fma_matmul(
    a: np.ndarray, b_nk: np.ndarray, *, initial: np.float32 = np.float32(0)
) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b_nk = np.asarray(b_nk, dtype=np.float32)
    assert a.ndim == b_nk.ndim == 2 and a.shape[1] == b_nk.shape[1]
    accumulator = np.full((a.shape[0], b_nk.shape[0]), initial, dtype=np.float32)
    for k in range(a.shape[1]):
        accumulator = (
            np.float64(a[:, k, None]) * np.float64(b_nk[None, :, k]) + np.float64(accumulator)
        ).astype(np.float32)
    return accumulator


def _pack_nvf4_scale_cells(scales: np.ndarray, row_groups: int) -> np.ndarray:
    """Pack `.scale_vec::4X` scale bytes into their `(k_tile, row_group, lane)` TMEM words.

    One K=64 tile spends all four bytes of one 32-bit TMEM word on one row, so
    consecutive K tiles land in different TMEM columns rather than in different
    byte pairs of the same word.
    """

    rows, sf_k = scales.shape
    assert sf_k % 4 == 0 and rows <= row_groups * 32
    cells = np.zeros((sf_k // 4, row_groups, 32), dtype=np.uint32)
    for row in range(rows):
        for k_tile in range(sf_k // 4):
            value = sum(int(scales[row, k_tile * 4 + byte]) << (8 * byte) for byte in range(4))
            cells[k_tile, row // 32, row % 32] = np.uint32(value)
    return cells


def _decode_e2m1_nibbles(bits: np.ndarray) -> np.ndarray:
    magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    signs = np.where(bits & np.uint8(8), np.float32(-1.0), np.float32(1.0))
    return (magnitudes[bits & np.uint8(7)] * signs).astype(np.float32)


def _pack_e8m0_cells(scales: np.ndarray) -> np.ndarray:
    assert scales.ndim == 2 and scales.shape[1] == 4 and scales.shape[0] % 32 == 0
    cells = np.zeros((scales.shape[0] // 32, 32), dtype=np.uint32)
    for row in range(scales.shape[0]):
        value = sum(int(scales[row, index]) << (8 * index) for index in range(4))
        cells[row // 32, row % 32] = np.uint32(value)
    return cells


def test_raw_tcgen_registry_classifies_exact_descriptor_and_transfer_signatures():
    assert [call.op_name for call in _raw_calls(raw_tcgen_descriptor_encode)] == [
        MATRIX_DESC,
        INSTR_DESC,
        INSTR_DESC_BLOCK,
    ]
    assert {call.op_name for call in _resolved_ptx_raw_tcgen_calls(raw_tcgen_ldst_32x32b)} == {
        "tirx.ptx.tcgen05_ld",
        "tirx.ptx.tcgen05_st",
    }
    assert [call.op_name for call in _resolved_ptx_raw_tcgen_calls(raw_tcgen_cp_warpx4)] == [
        "tirx.ptx.tcgen05_cp"
    ]
    assert [
        call.op_name for call in _resolved_ptx_raw_tcgen_calls(raw_tcgen_mma_block_scaled_mxf4_mqa)
    ] == [
        "tirx.ptx.tcgen05_mma_block_scale_block_ss",
        "tirx.ptx.tcgen05_mma_block_scale_ss",
    ]
    dense = {
        call.op_name
        for function in (
            raw_tcgen_mma_bf16_ss,
            raw_tcgen_mma_e4m3_mqa,
            raw_tcgen_mma_tf32_ts_predicated,
            raw_tcgen_mma_ws_bf16_ss_layout_e_tail,
        )
        for call in _resolved_ptx_raw_tcgen_calls(function)
    }
    assert dense == {
        "tirx.ptx.tcgen05_mma_ss",
        "tirx.ptx.tcgen05_mma_ts",
        "tirx.ptx.tcgen05_mma_ws_ss",
    }
    assert [
        call.op_name
        for call in _resolved_ptx_raw_tcgen_calls(raw_tcgen_mma_mixed_fp4_fp8_cta_group2)
    ] == ["tirx.ptx.tcgen05_mma_block_scale_ss"] * 2
    assert [
        call.op_name
        for call in _resolved_ptx_raw_tcgen_calls(raw_tcgen_mma_block_scaled_mxf4nvf4_two_k_tiles)
    ] == ["tirx.ptx.tcgen05_mma_block_scale_ss"] * 2


@pytest.mark.parametrize(
    ("kernel", "specialization"),
    [
        (
            raw_tcgen_mma_e4m3_m64_n8,
            "MmaF8f6f4F32SsCta1<v2::tcgen05::variant::E4m3, v2::tcgen05::variant::E4m3, ",
        ),
        (
            raw_tcgen_mma_e5m2_e4m3_m64_n8,
            "MmaF8f6f4F32SsCta1<v2::tcgen05::variant::E5m2, v2::tcgen05::variant::E4m3, ",
        ),
        (
            raw_tcgen_mma_f8f6f4_e5m2_f16_destination_predicated,
            "MmaF8f6f4F16SsCta1Pred<v2::tcgen05::variant::E5m2, v2::tcgen05::variant::E4m3, ",
        ),
        (
            raw_tcgen_mma_f8f6f4_cta2_rubin_k64_discard,
            "MmaF8f6f4F32SsCta2<v2::tcgen05::variant::E5m2, v2::tcgen05::variant::E5m2, ",
        ),
    ],
    ids=(
        "e4m3_e4m3_f32",
        "e5m2_e4m3_f32",
        "e5m2_e4m3_f16_pred",
        "rubin_e5m2_cta2_f32_k64_discard",
    ),
)
def test_raw_tcgen_f8f6f4_selects_its_operand_and_accumulator_markers(kernel, specialization):
    """A and B specialize independently, and the destination type is its own axis."""

    source = emit_rust_module(analyze(kernel), kernel)
    assert f"v2::tcgen05::variant::{specialization}" in source


@pytest.mark.parametrize(
    ("arch", "descriptor_marker"),
    [
        ("sm_100a", "MatrixDescriptorSm100"),
        ("sm_100f", "MatrixDescriptorSm100"),
        ("sm_103a", "MatrixDescriptorSm103"),
        ("sm_103f", "MatrixDescriptorSm103"),
        ("sm_107a", "MatrixDescriptorSm107"),
        ("sm_107f", "MatrixDescriptorSm107"),
    ],
    ids=("sm100a", "sm100f", "sm103a", "sm103f", "sm107a", "sm107f"),
)
def test_raw_tcgen_f8f6f4_cta2_specializes_descriptor_layout_from_kernel_arch(
    arch, descriptor_marker
):
    kernel = raw_tcgen_mma_f8f6f4_cta2_k32_without_arch.with_attr("tirx.cuda_arch", arch)
    source = emit_rust_module(analyze(kernel), kernel)
    assert f"v2::tcgen05::variant::{descriptor_marker}" in source
    assert TCGEN_DESCRIPTOR_LAYOUT not in source


@pytest.mark.parametrize(
    ("arch", "message"),
    [
        (None, r"require PrimFunc attribute.*tirx\.cuda_arch"),
        ("sm_999a", "unsupported CUDA architecture.*sm_999a"),
    ],
    ids=("missing", "unknown"),
)
def test_raw_tcgen_f8f6f4_cta2_requires_exact_kernel_architecture(arch, message):
    kernel = raw_tcgen_mma_f8f6f4_cta2_k32_without_arch
    if arch is not None:
        kernel = kernel.with_attr("tirx.cuda_arch", arch)
    with pytest.raises(UnsupportedTIRxError, match=message):
        emit_rust_module(analyze(kernel), kernel)


@pytest.mark.parametrize("arch", ("sm_100a", "sm_100f", "sm_103a", "sm_103f"))
def test_raw_tcgen_f8f6f4_cta2_sm100_rejects_sm107_k64_bit(arch, tmp_path):
    @T.prim_func
    def runtime_descriptor(descriptor: T.uint32):
        T.device_entry()
        _cluster = T.cluster_id([1])
        cta = T.cta_id_in_cluster([2])
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        shared_a = T.alloc_buffer((16384,), "uint8", scope="shared")
        shared_b = T.alloc_buffer((8192,), "uint8", scope="shared")
        _tmem = T.decl_buffer(
            (128, 136), "uint32", scope="tmem", layout=_TMEM_D_136, allocated_addr=0
        )
        desc_a: T.uint64
        desc_b: T.uint64
        for i in T.serial(512):
            shared_a[lane + i * 32] = T.uint8(0)
        for i in T.serial(256):
            shared_b[lane + i * 32] = T.uint8(0)
        T.cuda.cluster_sync()
        if cta == 0 and lane == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_a), T.address_of(shared_a[0]), ldo=0, sdo=64, swizzle=3
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b), T.address_of(shared_b[0]), ldo=0, sdo=64, swizzle=3
            )
            T.ptx["tcgen05.mma.cta_group::2.kind::f8f6f4.collector::a::discard"](
                T.uint32(0),
                desc_a,
                desc_b,
                descriptor,
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.ptx.pred(T.uint32(0)),
            )
        T.cuda.cluster_sync()

    kernel = runtime_descriptor.with_attr("tirx.cuda_arch", arch)
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    engine = numsim.Engine()
    engine.run(module, {"descriptor": 0x10040490})
    with pytest.raises(
        numsim.NumSimExecutionError,
        match="descriptor must encode dense F32/E5M2/E5M2",
    ):
        engine.run(module, {"descriptor": 0x30040490})


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (
            raw_tcgen_mma_f8f6f4_reserved_operand,
            "requires modeled F16/F32 and E4M3/E5M2/E2M3/E3M2/E2M1 formats",
        ),
        (
            raw_tcgen_mma_f8f6f4_cta_group2_reserved_destination,
            "cta_group=2 requires an F16 or F32 destination",
        ),
        (
            raw_tcgen_mma_f8f6f4_cta_group2_invalid_n,
            "cta_group=2 requires M in {128, 256} and N in 32..=256 by 32 for MN-major B",
        ),
    ],
    ids=(
        "f8f6f4_reserved_operand",
        "f8f6f4_cta2_f16",
        "f8f6f4_cta2_invalid_n",
    ),
)
def test_raw_tcgen_dense_mma_gate_names_the_exact_legal_set(kernel, message):
    """Reject unmodeled floating-point forms with the exact missing contract."""

    with pytest.raises(numsim.UnmodeledTIRxFormError) as caught:
        analyze(kernel)
    assert caught.value.target_id == "call:tirx.ptx.tcgen05_mma_ss"
    assert message in str(caught.value)


def test_formerly_unmodeled_mxf4_and_ws_masks_have_specific_variants():
    # Preserve the original analysis-only probes. Their zero matrix addresses
    # are not runnable numerical inputs; descriptor validation remains runtime.
    for kernel, op_name, variant in (
        (
            raw_tcgen_mxf4_ss_cta2_analysis_probe,
            "tcgen05_mma_block_scale_ss",
            "MmaBlockMxf4E8m0SsCta2",
        ),
        (
            raw_tcgen_mxf4_ts_cta1_analysis_probe,
            "tcgen05_mma_block_scale_ts",
            "MmaBlockMxf4E8m0TsCta1",
        ),
        (raw_tcgen_ws_ss_mask_analysis_probe, "tcgen05_mma_ws_ss", "MmaF16SsCta1Ws"),
        (raw_tcgen_ws_ts_mask_analysis_probe, "tcgen05_mma_ws_ts", "MmaF16TsCta1Ws"),
    ):
        assert analyze(kernel).unsupported == ()
        calls = _resolved_ptx_raw_tcgen_calls(kernel)
        assert len(calls) == 1
        assert calls[0].op_name == f"tirx.ptx.{op_name}"
        mma_calls = [
            call
            for call in emitted_calls(kernel, f"tirx.ptx.{op_name}")
            if call.function == "v2::tcgen05::mma"
        ]
        assert any(f"v2::tcgen05::variant::{variant}<" in mma.generics for mma in mma_calls)


def test_raw_tcgen_descriptor_encoders_write_exact_bits(tmp_path):
    output = np.zeros(3, dtype=np.uint64)
    module = numsim.transpile(raw_tcgen_descriptor_encode, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": output})

    expected = np.array(
        [
            _matrix_descriptor(start_16b=1, ldo=1, sdo=8, layout_type=2),
            _dense_descriptor(),
            _block_descriptor(),
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_descriptor_encoders_resolve_global_integer_destinations(tmp_path):
    module = numsim.transpile(raw_tcgen_descriptor_encode_to_global, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "matrix_output": np.zeros(1, dtype=np.uint64),
            "instr_output": np.zeros(1, dtype=np.uint32),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["matrix_output"],
        np.array([_matrix_descriptor(start_16b=1, ldo=1, sdo=8, layout_type=2)], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        result.outputs["instr_output"], np.array([_dense_descriptor()], dtype=np.uint32)
    )


def test_raw_tcgen_32x32b_ld_st_roundtrip(tmp_path):
    source = np.arange(128 * 4, dtype=np.uint32).reshape(128, 4) * np.uint32(17) + 3
    output = np.zeros_like(source)
    module = numsim.transpile(raw_tcgen_ldst_32x32b, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)


def test_raw_tcgen_16x256b_ld_observes_exact_lane_register_mapping(tmp_path):
    source = np.arange(128 * 8, dtype=np.uint32).reshape(128, 8) + np.uint32(1000)
    output = np.zeros((4, 32, 4), dtype=np.uint32)
    module = numsim.transpile(raw_tcgen_ld_16x256b_mapping, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.empty_like(output)
    for warp in range(4):
        for lane in range(32):
            for register in range(4):
                row = warp * 32 + (lane >> 2) + 8 * ((register >> 1) & 1)
                col = (register & 1) + 2 * (lane & 3)
                expected[warp, lane, register] = source[row, col]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_missing_ld_shapes_observe_exact_lane_register_mappings(tmp_path):
    source = np.arange(128 * 16, dtype=np.uint32).reshape(128, 16) + np.uint32(5000)
    output = np.zeros((3, 4, 32, 2), dtype=np.uint32)
    module = numsim.transpile(raw_tcgen_ld_missing_shape_mappings, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.empty_like(output)
    for warp in range(4):
        for lane in range(32):
            for register in range(2):
                expected[0, warp, lane, register] = source[
                    warp * 32 + lane % 16, register + (lane // 16) * 2
                ]
                expected[1, warp, lane, register] = source[
                    warp * 32 + (lane >> 2) + 8 * (lane & 1), ((lane >> 1) & 1) + 2 * register
                ]
                expected[2, warp, lane, register] = source[
                    warp * 32 + (lane >> 2) + 8 * (register & 1), (lane & 3) + 4 * (register >> 1)
                ]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_ld_pack_and_st_unpack_use_adjacent_tmem_columns(tmp_path):
    source = np.arange(128 * 4, dtype=np.uint32).reshape(128, 4) * np.uint32(17) + 3
    packed_output = np.zeros((4, 32, 2), dtype=np.uint32)
    packed_module = numsim.transpile(raw_tcgen_ld_pack_32x32b, cache_dir=tmp_path / "ld")
    packed = (
        numsim.Engine()
        .run(packed_module, {"source": source, "output": packed_output})
        .outputs["output"]
    )
    expected_packed = np.empty_like(packed)
    for warp in range(4):
        for lane in range(32):
            row = warp * 32 + lane
            for register in range(2):
                expected_packed[warp, lane, register] = np.uint32(
                    source[row, register * 2] & np.uint32(0xFFFF)
                ) | np.uint32((source[row, register * 2 + 1] & np.uint32(0xFFFF)) << np.uint32(16))
    np.testing.assert_array_equal(packed, expected_packed)

    unpacked_output = np.zeros((128, 4), dtype=np.uint32)
    unpacked_module = numsim.transpile(raw_tcgen_st_unpack_32x32b, cache_dir=tmp_path / "st")
    unpacked = (
        numsim.Engine()
        .run(unpacked_module, {"source": expected_packed, "output": unpacked_output})
        .outputs["output"]
    )
    np.testing.assert_array_equal(unpacked, source & np.uint32(0xFFFF))


def test_raw_tcgen_cp_warpx4_replicates_source_into_all_warp_lanes(tmp_path):
    source = np.arange(32 * 4, dtype=np.uint32).reshape(32, 4) * np.uint32(11) + 5
    output = np.zeros((128, 4), dtype=np.uint32)
    module = numsim.transpile(raw_tcgen_cp_warpx4, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "issue": 1, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], np.tile(source, (4, 1)))

    skipped = numsim.Engine().run(
        module,
        {"source": source, "issue": 0, "output": np.full_like(output, np.uint32(99))},
    )
    np.testing.assert_array_equal(skipped.outputs["output"], np.zeros_like(output))


def test_raw_tcgen_cp_128x256b_decodes_matrix_descriptor_swizzle(tmp_path):
    logical = np.arange(128 * 8, dtype=np.uint32).reshape(128, 8) * np.uint32(13) + 7
    source = _swizzle_128_physical(logical)
    output = np.zeros_like(logical)
    module = numsim.transpile(raw_tcgen_cp_128x256b_swizzle, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], logical)


def test_raw_tcgen_cp_covers_4x256b_and_both_warpx2_pairings(tmp_path):
    logical_4x256 = np.arange(4 * 8, dtype=np.uint32).reshape(4, 8) * np.uint32(29) + np.uint32(7)
    atoms = logical_4x256.view(np.uint8).reshape(4, 2, 16)
    source_4x256 = _tcgen_cp_two_atom_physical(atoms)
    module_4x256 = numsim.transpile(raw_tcgen_cp_4x256b, cache_dir=tmp_path / "4x256")
    result_4x256 = numsim.Engine().run(
        module_4x256, {"source": source_4x256, "output": np.zeros((4, 8), dtype=np.uint32)}
    )
    np.testing.assert_array_equal(result_4x256.outputs["output"], logical_4x256)

    source_64 = np.arange(64 * 4, dtype=np.uint32).reshape(64, 4) * np.uint32(13) + 5
    module_64 = numsim.transpile(raw_tcgen_cp_warpx2_01_23, cache_dir=tmp_path / "01_23")
    result_64 = numsim.Engine().run(
        module_64, {"source": source_64, "output": np.zeros((128, 4), dtype=np.uint32)}
    )
    expected_64 = np.concatenate(
        [source_64[:32], source_64[:32], source_64[32:], source_64[32:]], axis=0
    )
    np.testing.assert_array_equal(result_64.outputs["output"], expected_64)


@pytest.mark.parametrize(
    ("kernel", "codes", "packer", "shift"),
    [
        (raw_tcgen_cp_decompress_b4, np.uint8(16), _pack_b4_cp_source, np.uint8(2)),
        (raw_tcgen_cp_decompress_b6, np.uint8(64), _pack_b6_cp_source, np.uint8(0)),
    ],
)
def test_raw_tcgen_cp_decompression_expands_packed_codes(tmp_path, kernel, codes, packer, shift):
    logical = (np.arange(4 * 32, dtype=np.uint8).reshape(4, 32) * np.uint8(7) + np.uint8(3)) % codes
    source = packer(logical)
    module = numsim.transpile(kernel, cache_dir=tmp_path / kernel.__name__)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.zeros((4, 8), dtype=np.uint32)}
    )
    expected = np.ascontiguousarray(logical << shift).view(np.uint32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_cp_128b_base32b_descriptor_uses_32byte_atomic_swizzle(tmp_path):
    logical = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32) * np.uint8(5) + np.uint8(1)
    source = _swizzle_128_base32_physical(logical)
    module = numsim.transpile(raw_tcgen_cp_128b_base32b, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module, {"source": source, "output": np.zeros((4, 8), dtype=np.uint32)}
    )
    np.testing.assert_array_equal(
        result.outputs["output"], np.ascontiguousarray(logical).view(np.uint32)
    )


def test_raw_tcgen_cp_cta_group2_writes_both_ctas_with_warpx2_routing(tmp_path):
    logical = np.arange(2 * 64 * 4, dtype=np.uint32).reshape(2, 64, 4) * np.uint32(19) + 9
    source = np.stack([_swizzle_128_physical(logical[cta]) for cta in range(2)])
    output = np.zeros((2, 128, 4), dtype=np.uint32)
    module = numsim.transpile(raw_tcgen_cp_64x128b_cta_group2, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = np.stack(
        [
            np.concatenate(
                [logical[cta, :32], logical[cta, 32:], logical[cta, :32], logical[cta, 32:]], axis=0
            )
            for cta in range(2)
        ]
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_cp_128x256b_swizzle64_routes_both_k_halves_for_every_row(tmp_path):
    row = np.arange(64, dtype=np.uint16)[:, None]
    col = np.arange(64, dtype=np.uint16)[None, :]
    source_bits = (np.uint16(0x3C00) + row * np.uint16(67) + col).astype(np.uint16)
    output = np.zeros((2, 128, 8), dtype=np.uint32)

    module = numsim.transpile(raw_tcgen_cp_128x256b_swizzle64_logical, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "source": source_bits,
            "output": output,
        },
    )

    packed = source_bits[:, 0::2].astype(np.uint32) | (
        source_bits[:, 1::2].astype(np.uint32) << np.uint32(16)
    )
    expected = np.empty_like(output)
    expected[0, :64] = packed[:, :8]
    expected[0, 64:] = packed[:, 16:24]
    expected[1, :64] = packed[:, 8:16]
    expected[1, 64:] = packed[:, 24:32]
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_descriptor_can_cross_a_short_pool_alias_within_its_backing(tmp_path):
    source = (np.arange(32 * 16, dtype=np.uint16).reshape(32, 16) * 13 + 7).astype(np.uint8)
    output = np.zeros((128, 4), dtype=np.uint32)

    module = numsim.transpile(raw_tcgen_cp_descriptor_crosses_short_pool_alias, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    packed = np.ascontiguousarray(source).view(np.uint32).reshape(32, 4)
    np.testing.assert_array_equal(result.outputs["output"], np.tile(packed, (4, 1)))


def test_raw_tcgen_descriptor_bits_can_cross_into_another_valid_backing(tmp_path):
    module = numsim.transpile(
        raw_tcgen_cp_descriptor_cannot_cross_shared_backings, cache_dir=tmp_path
    )
    result = numsim.Engine().run(module, {})
    assert result.verdict == "clean"


def test_raw_tcgen_mxf4_mqa_mma_uses_physical_smem_and_tmem_scales(tmp_path):
    row = np.arange(128, dtype=np.uint16)[:, None]
    k = np.arange(128, dtype=np.uint16)[None, :]
    k_atom = k // np.uint16(32)
    a_bits = ((row * 3 + k * 5 + k_atom + 1) % 16).astype(np.uint8)
    b_bits = ((row * 7 + k * 3 + k_atom * 3 + 2) % 16).astype(np.uint8)
    scale_pattern_a = np.array([126, 127, 128, 127], dtype=np.uint8)
    scale_pattern_b = np.array([128, 127, 126, 127], dtype=np.uint8)
    scale_indices = np.arange(4, dtype=np.int64)[None, :]
    row_indices = np.arange(128, dtype=np.int64)[:, None]
    scale_a_bits = scale_pattern_a[(scale_indices + row_indices) % 4]
    scale_b_bits = scale_pattern_b[(scale_indices + row_indices * 3) % 4]

    output = np.zeros((128, 128), dtype=np.float32)
    module = numsim.transpile(raw_tcgen_mma_block_scaled_mxf4_mqa, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": _mxf4_kmajor_physical(a_bits),
            "b_physical": _mxf4_kmajor_physical(b_bits),
            "scale_a_cells": _pack_e8m0_cells(scale_a_bits),
            "scale_b_cells": _pack_e8m0_cells(scale_b_bits),
            "issue_second": 1,
            "output": output,
        },
    )

    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    signs_a = np.where(a_bits & 8, np.float32(-1.0), np.float32(1.0))
    signs_b = np.where(b_bits & 8, np.float32(-1.0), np.float32(1.0))
    a = values[a_bits & 7] * signs_a
    b = values[b_bits & 7] * signs_b
    scale_a = np.exp2(scale_a_bits.astype(np.int16) - 127).astype(np.float32)
    scale_b = np.exp2(scale_b_bits.astype(np.int16) - 127).astype(np.float32)
    a *= np.repeat(scale_a, 32, axis=1)
    b *= np.repeat(scale_b, 32, axis=1)
    expected = a @ b.T
    np.testing.assert_array_equal(result.outputs["output"], expected)

    first_tile_only = numsim.Engine().run(
        module,
        {
            "a_physical": _mxf4_kmajor_physical(a_bits),
            "b_physical": _mxf4_kmajor_physical(b_bits),
            "scale_a_cells": _pack_e8m0_cells(scale_a_bits),
            "scale_b_cells": _pack_e8m0_cells(scale_b_bits),
            "issue_second": 0,
            "output": np.zeros_like(output),
        },
    )
    np.testing.assert_array_equal(first_tile_only.outputs["output"], a[:, :64] @ b[:, :64].T)

    expression_module = numsim.transpile(
        raw_tcgen_mma_block_scaled_mxf4_expression_input_d,
        cache_dir=tmp_path / "expression-input-d",
    )
    expression_result = numsim.Engine().run(
        expression_module,
        {
            "mode": 0,
            "a_physical": _mxf4_kmajor_physical(a_bits),
            "b_physical": _mxf4_kmajor_physical(b_bits),
            "scale_a_cells": _pack_e8m0_cells(scale_a_bits),
            "scale_b_cells": _pack_e8m0_cells(scale_b_bits),
            "output": np.zeros_like(output),
        },
    )
    expression_expected = a[:, :64] @ b[:, :64].T
    np.testing.assert_array_equal(expression_result.outputs["output"], expression_expected)


def test_raw_tcgen_mxf4nvf4_scale_vec_4x_matches_the_typed_block_scaled_gemm(tmp_path):
    """`.kind::mxf4nvf4` `.scale_vec::4X` against the typed `Tx.gemm_async` path.

    Both kernels compute the same logical `M=128, N=8, K=128` product from the
    same logical operands, so the comparison is element-exact: the raw path
    chains two K=64 instructions through TMEM while the typed path accumulates
    K=128 in one call, and `mma_f32_abt_increasing_k` makes those the same
    binary32 FMA chain. Two K tiles are required for the comparison to observe
    the 4X scale indexing at all -- a single tile would never advance the
    scale-factor address.
    """

    row = np.arange(128, dtype=np.uint16)[:, None]
    k = np.arange(128, dtype=np.uint16)[None, :]
    block = k // np.uint16(16)
    a_bits = ((row * 3 + k * 5 + block + 1) % 16).astype(np.uint8)
    b_bits = ((row * 7 + k * 3 + block * 3 + 2) % 16).astype(np.uint8)
    # Only the first N=8 rows of B take part; the rest stay zero so the raw
    # kernel's shared operand is fully initialized.
    b_bits[8:, :] = np.uint8(0)
    # Finite positive `ue4m3` codes: the format's MSB is padding that must stay
    # clear, and 0x7f is its only NaN.
    finite_scales = np.array([0x30, 0x34, 0x38, 0x3C, 0x40, 0x44], dtype=np.uint8)
    scale_index = np.arange(8, dtype=np.int64)[None, :]
    row_index = np.arange(128, dtype=np.int64)[:, None]
    scale_a_bits = finite_scales[(scale_index + row_index) % len(finite_scales)]
    scale_b_bits = finite_scales[(scale_index * 2 + row_index[:8] * 3) % len(finite_scales)]

    raw_module = numsim.transpile(
        raw_tcgen_mma_block_scaled_mxf4nvf4_two_k_tiles, cache_dir=tmp_path / "raw"
    )
    raw_result = numsim.Engine().run(
        raw_module,
        {
            "a_physical": _mxf4_kmajor_physical(a_bits),
            "b_physical": _mxf4_kmajor_physical(b_bits),
            "scale_a_cells": _pack_nvf4_scale_cells(scale_a_bits, 4),
            "scale_b_cells": _pack_nvf4_scale_cells(scale_b_bits, 1)[:, 0, :],
            "output": np.zeros((128, 8), dtype=np.float32),
            "use_ue8m0_scales": 0,
        },
    )

    typed_module = numsim.transpile(typed_nvfp4_gemm_two_k_tiles, cache_dir=tmp_path / "typed")
    typed_result = numsim.Engine().run(
        typed_module,
        {
            "left_packed": (a_bits[:, 0::2] | (a_bits[:, 1::2] << np.uint8(4))).astype(np.uint8),
            "right_packed": (b_bits[:8, 0::2] | (b_bits[:8, 1::2] << np.uint8(4))).astype(np.uint8),
            "scale_a": scale_a_bits,
            "scale_b": scale_b_bits,
            "output": np.zeros((128, 8), dtype=np.float32),
        },
    )

    a = _decode_e2m1_nibbles(a_bits) * np.repeat(_e4m3fn_bits_to_f32(scale_a_bits), 16, axis=1)
    b = _decode_e2m1_nibbles(b_bits[:8]) * np.repeat(_e4m3fn_bits_to_f32(scale_b_bits), 16, axis=1)
    expected = _tcgen_fma_matmul(a, b)

    np.testing.assert_array_equal(raw_result.outputs["output"], expected)
    np.testing.assert_array_equal(raw_result.outputs["output"], typed_result.outputs["output"])

    ue8m0_codes = np.array([125, 126, 127, 128, 129], dtype=np.uint8)
    ue8m0_scale_a_bits = ue8m0_codes[(scale_index + row_index) % len(ue8m0_codes)]
    ue8m0_scale_b_bits = ue8m0_codes[(scale_index * 2 + row_index[:8] * 3) % len(ue8m0_codes)]
    ue8m0_result = numsim.Engine().run(
        raw_module,
        {
            "a_physical": _mxf4_kmajor_physical(a_bits),
            "b_physical": _mxf4_kmajor_physical(b_bits),
            "scale_a_cells": _pack_nvf4_scale_cells(ue8m0_scale_a_bits, 4),
            "scale_b_cells": _pack_nvf4_scale_cells(ue8m0_scale_b_bits, 1)[:, 0, :],
            "output": np.zeros((128, 8), dtype=np.float32),
            "use_ue8m0_scales": 1,
        },
    )
    ue8m0_scale_a = np.exp2(ue8m0_scale_a_bits.astype(np.int16) - 127).astype(np.float32)
    ue8m0_scale_b = np.exp2(ue8m0_scale_b_bits.astype(np.int16) - 127).astype(np.float32)
    ue8m0_a = _decode_e2m1_nibbles(a_bits) * np.repeat(ue8m0_scale_a, 16, axis=1)
    ue8m0_b = _decode_e2m1_nibbles(b_bits[:8]) * np.repeat(ue8m0_scale_b, 16, axis=1)
    np.testing.assert_array_equal(
        ue8m0_result.outputs["output"],
        _tcgen_fma_matmul(ue8m0_a, ue8m0_b),
    )

    # Positive control that those numbers came from the `ue4m3` decode, which
    # only this variant reaches: PTX ISA 5.2.3 makes the format's MSB padding
    # that must stay clear, so a set bit fails closed instead of reading as a
    # negative scale.
    poisoned = _pack_nvf4_scale_cells(scale_a_bits, 4)
    poisoned[0, 0, 0] |= np.uint32(0x80)
    with pytest.raises(numsim.NumSimExecutionError, match="sets the padding MSB"):
        numsim.Engine().run(
            raw_module,
            {
                "a_physical": _mxf4_kmajor_physical(a_bits),
                "b_physical": _mxf4_kmajor_physical(b_bits),
                "scale_a_cells": poisoned,
                "scale_b_cells": _pack_nvf4_scale_cells(scale_b_bits, 1)[:, 0, :],
                "output": np.zeros((128, 8), dtype=np.float32),
                "use_ue8m0_scales": 0,
            },
        )


def test_raw_tcgen_mxf4_bulk_smem_and_tmem_reads_require_review(tmp_path):
    zeros = np.zeros(8192, dtype=np.uint8)
    scale_cells = _pack_e8m0_cells(np.full((128, 4), np.uint8(127), dtype=np.uint8))
    module = numsim.transpile(raw_tcgen_mxf4_bulk_reads_fail_closed, cache_dir=tmp_path)
    bindings = {
        "a_physical": zeros,
        "b_physical": zeros,
        "scale_a_cells": scale_cells,
        "scale_b_cells": scale_cells,
    }

    for mode, expected_space in ((1, "shared"), (2, "tmem")):
        result = numsim.Engine().run(module, {"mode": mode, **bindings})

        assert result.verdict == "review"
        assert result.diagnostics
        assert {item["status"] for item in result.diagnostics} == {"review"}
        assert {item["kind"] for item in result.diagnostics} == {"uninitialized_read"}
        assert {item["space"] for item in result.diagnostics} == {expected_space}


def test_raw_tcgen_e4m3_mqa_mma_accumulates_four_k_tiles(tmp_path):
    finite_codes = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    row = np.arange(128, dtype=np.int64)[:, None]
    k = np.arange(128, dtype=np.int64)[None, :]
    a_bits = finite_codes[(row * 3 + k * 5 + 1) % len(finite_codes)]
    b_bits = finite_codes[(row * 7 + k * 2 + 3) % len(finite_codes)]
    output = np.zeros((128, 128), dtype=np.float32)
    module = numsim.transpile(raw_tcgen_mma_e4m3_mqa, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
            "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
            "output": output,
        },
    )

    expected = _e4m3fn_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_bf16_shared_shared_mma_matches_independent_matrix_product(tmp_path):
    row = np.arange(64, dtype=np.float32)[:, None]
    col = np.arange(8, dtype=np.float32)[:, None]
    inner = np.arange(16, dtype=np.float32)[None, :]
    a = ((row % np.float32(5)) - np.float32(2)) + inner * np.float32(0.5)
    b = ((col % np.float32(3)) - np.float32(1)) - inner * np.float32(0.25)
    a_bits = _bfloat16_bits(a)
    b_bits = _bfloat16_bits(b)
    a_physical = _kmajor_swizzle_physical_at_start(
        np.ascontiguousarray(a_bits).view(np.uint8).reshape(64, 32),
        start=128,
        swizzle_len=2,
        sdo=512,
        byte_len=4096,
    )
    b_physical = _kmajor_swizzle_physical_at_start(
        np.ascontiguousarray(b_bits).view(np.uint8).reshape(8, 32),
        start=4224,
        swizzle_len=2,
        sdo=512,
        byte_len=1024,
    )
    module = numsim.transpile(raw_tcgen_mma_bf16_ss, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": np.pad(a_physical, (0, 4096 - a_physical.size)),
            "b_physical": np.pad(b_physical, (0, 1024 - b_physical.size)),
            "output": np.zeros((64, 8), dtype=np.float32),
            "output_ws": np.zeros((64, 8), dtype=np.float32),
        },
    )
    a_bf16 = (a_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    b_bf16 = (b_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    expected = np.empty((64, 8), dtype=np.float32)
    for i in range(64):
        for j in range(8):
            accumulator = np.float32(0)
            for k in range(16):
                accumulator = np.float32(
                    np.float64(a_bf16[i, k]) * np.float64(b_bf16[j, k]) + np.float64(accumulator)
                )
            expected[i, j] = accumulator
    np.testing.assert_array_equal(result.outputs["output"], expected)
    np.testing.assert_array_equal(result.outputs["output_ws"], expected)


def test_raw_tcgen_m64_weight_stationary_uses_layout_e_at_tmem_tail(tmp_path):
    row = np.arange(64, dtype=np.float32)[:, None]
    col = np.arange(128, dtype=np.float32)[:, None]
    inner = np.arange(16, dtype=np.float32)[None, :]
    a = ((row % np.float32(7)) - np.float32(3)) * np.float32(0.25) + inner * np.float32(0.125)
    b = ((col % np.float32(11)) - np.float32(5)) * np.float32(0.125) - inner * np.float32(0.0625)
    a_bits = _bfloat16_bits(a)
    b_bits = _bfloat16_bits(b)
    module = numsim.transpile(raw_tcgen_mma_ws_bf16_ss_layout_e_tail, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": _kmajor_swizzle_physical(
                np.ascontiguousarray(a_bits).view(np.uint8).reshape(64, 32),
                swizzle_len=2,
                sdo=512,
            ),
            "b_physical": _kmajor_swizzle_physical(
                np.ascontiguousarray(b_bits).view(np.uint8).reshape(128, 32),
                swizzle_len=2,
                sdo=512,
            ),
            "output": np.zeros((64, 128), dtype=np.float32),
        },
    )

    a_bf16 = (a_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    b_bf16 = (b_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    expected = _tcgen_fma_matmul(a_bf16, b_bf16)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_f16_tensor_shared_mma_uses_mn_major_ldo_for_second_n_tile(tmp_path):
    row = np.arange(128, dtype=np.float32)[:, None]
    n = np.arange(128, dtype=np.float32)[:, None]
    k = np.arange(16, dtype=np.float32)[None, :]
    a = ((row % np.float32(7)) - np.float32(3)) * np.float32(0.125) + k * np.float32(0.0625)
    b = ((n % np.float32(11)) - np.float32(5)) * np.float32(0.25) - k * np.float32(0.03125)
    b[64:] += np.float32(3)
    a_f16 = np.asarray(a, dtype=np.float16)
    b_f16 = np.asarray(b, dtype=np.float16)
    a_bits = np.ascontiguousarray(a_f16).view(np.uint16)
    a_packed = a_bits[:, 0::2].astype(np.uint32) | (
        a_bits[:, 1::2].astype(np.uint32) << np.uint32(16)
    )

    module = numsim.transpile(raw_tcgen_mma_f16_ts_mn_major_n128, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_packed": a_packed,
            "b_physical": _mnmajor_f16_swizzle_physical(
                np.ascontiguousarray(b_f16).view(np.uint16)
            ),
            "output": np.zeros((128, 128), dtype=np.float32),
            "output_ws": np.zeros((128, 128), dtype=np.float32),
        },
    )

    expected = _tcgen_fma_matmul(a_f16.astype(np.float32), b_f16.astype(np.float32))
    np.testing.assert_array_equal(result.outputs["output"], expected)
    np.testing.assert_array_equal(result.outputs["output_ws"], expected)


def test_raw_tcgen_f16_cta2_datapaths_match_independent_matrix_products(tmp_path):
    inner = np.arange(16, dtype=np.float32)[None, :]

    row256 = np.arange(256, dtype=np.float32)[:, None]
    col16 = np.arange(16, dtype=np.float32)[:, None]
    a_ss256 = np.asarray(
        ((row256 % 13) - 6) * np.float32(0.0625) + inner * np.float32(0.03125),
        dtype=np.float16,
    )
    b_ss256 = np.asarray(
        ((col16 % 7) - 3) * np.float32(0.125) - inner * np.float32(0.015625),
        dtype=np.float16,
    )
    a_ts256 = np.asarray(
        ((row256 % 11) - 5) * np.float32(0.03125) - inner * np.float32(0.0625),
        dtype=np.float16,
    )
    b_ts256 = np.asarray(
        ((col16 % 5) - 2) * np.float32(0.25) + inner * np.float32(0.015625),
        dtype=np.float16,
    )

    row128 = np.arange(128, dtype=np.float32)[:, None]
    col128 = np.arange(128, dtype=np.float32)[:, None]
    a_ss128 = np.asarray(
        ((row128 % 9) - 4) * np.float32(0.0625) + inner * np.float32(0.03125),
        dtype=np.float16,
    )
    b_ss128 = np.asarray(
        ((col128 % 15) - 7) * np.float32(0.03125) - inner * np.float32(0.015625),
        dtype=np.float16,
    )

    def kmajor_pair(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                _kmajor_swizzle_physical(
                    np.ascontiguousarray(part).view(np.uint8).reshape(part.shape[0], 32),
                    swizzle_len=3,
                    sdo=1024,
                )
                for part in np.split(values, 2)
            ]
        )

    def mnmajor_pair(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                _mnmajor_f16_swizzle_physical_cta2(np.ascontiguousarray(part).view(np.uint16))
                for part in np.split(values, 2)
            ]
        )

    a_ts_bits = np.ascontiguousarray(a_ts256).view(np.uint16)
    a_ts_packed = a_ts_bits[:, 0::2].astype(np.uint32) | (
        a_ts_bits[:, 1::2].astype(np.uint32) << np.uint32(16)
    )
    module = numsim.transpile(raw_tcgen_mma_f16_cta2_datapaths, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_ss256_physical": kmajor_pair(a_ss256),
            "b_ss256_physical": kmajor_pair(b_ss256),
            "a_ts256_packed": np.stack(np.split(a_ts_packed, 2)),
            "b_ts256_physical": np.stack(
                [
                    _mnmajor_f16_swizzle_physical_cta2(np.ascontiguousarray(part).view(np.uint16))
                    for part in np.split(b_ts256, 2)
                ]
            ),
            "a_ss128_physical": mnmajor_pair(a_ss128),
            "b_ss128_physical": mnmajor_pair(b_ss128),
            "output_ss256": np.zeros((256, 16), dtype=np.float32),
            "output_ts256": np.zeros((256, 16), dtype=np.float32),
            "output_ss128": np.zeros((128, 128), dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(
        result.outputs["output_ss256"],
        _tcgen_fma_matmul(a_ss256.astype(np.float32), b_ss256.astype(np.float32)),
    )
    np.testing.assert_array_equal(
        result.outputs["output_ts256"],
        _tcgen_fma_matmul(a_ts256.astype(np.float32), b_ts256.astype(np.float32)),
    )
    np.testing.assert_array_equal(
        result.outputs["output_ss128"],
        _tcgen_fma_matmul(a_ss128.astype(np.float32), b_ss128.astype(np.float32)),
    )


def test_raw_tcgen_f8f6f4_cta2_rubin_k64_gathers_masks_and_accumulates_both_ctas(tmp_path):
    e5m2_codes = np.array(
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x38, 0x3C, 0x3D, 0x40, 0xBC, 0xC0],
        dtype=np.uint8,
    )
    cta = np.arange(2, dtype=np.int64)[:, None, None]
    a_row = np.arange(128, dtype=np.int64)[None, :, None]
    b_row = np.arange(64, dtype=np.int64)[None, :, None]
    k = np.arange(64, dtype=np.int64)[None, None, :]
    a_bits = e5m2_codes[(cta * 5 + a_row * 3 + k * 7 + 1) % len(e5m2_codes)]
    b_bits = e5m2_codes[(cta * 2 + b_row * 5 + k * 3 + 2) % len(e5m2_codes)]

    seed_cta = np.arange(2, dtype=np.float32)[:, None, None]
    seed_row = np.arange(128, dtype=np.float32)[None, :, None]
    seed_col = np.arange(128, dtype=np.float32)[None, None, :]
    seed = (
        seed_cta * np.float32(2)
        + (seed_row % np.float32(7)) * np.float32(0.125)
        - (seed_col % np.float32(5)) * np.float32(0.0625)
    ).astype(np.float32)

    module = numsim.transpile(raw_tcgen_mma_f8f6f4_cta2_rubin_k64_discard, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": np.stack(
                [_kmajor_swizzle_physical(part, swizzle_len=3, sdo=1024) for part in a_bits]
            ),
            "b_physical": np.stack([_mnmajor_f8_swizzle_physical(part) for part in b_bits]),
            "seed": seed,
            "output": np.zeros_like(seed),
        },
    )

    joint_a = _e5m2_bits_to_f32(a_bits.reshape(256, 64))
    joint_b = _e5m2_bits_to_f32(b_bits.reshape(128, 64))
    expected = seed.reshape(256, 128).copy()
    for inner in range(64):
        expected = (
            np.float64(joint_a[:, inner, None]) * np.float64(joint_b[None, :, inner])
            + np.float64(expected)
        ).astype(np.float32)
    # Word 0 bit 5 masks CTA 0 lane/row 5; word 6 bit 7 masks CTA 1 row 71.
    expected[5, :] = seed[0, 5, :]
    expected[128 + 71, :] = seed[1, 71, :]
    np.testing.assert_array_equal(result.outputs["output"], expected.reshape(2, 128, 128))


def test_raw_tcgen_f8f6f4_cta2_rubin_keeps_descriptor_bit15_reserved(tmp_path):
    module = numsim.transpile(
        raw_tcgen_mma_f8f6f4_cta2_rubin_rejects_descriptor_bit15,
        cache_dir=tmp_path,
    )

    with pytest.raises(
        numsim.NumSimExecutionError,
        match="descriptor uses unsupported reserved/base/LBO-mode bits",
    ):
        numsim.Engine().run(module, {})


def test_shared_span_above_18_bits_requires_a_supported_sm107_descriptor_form():
    with pytest.raises(
        UnsupportedTIRxError,
        match="shared-memory virtual address span 262160 exceeds the 18-bit descriptor",
    ):
        emit_rust_module(
            analyze(shared_span_above_18_bits_without_sm107_descriptor),
            shared_span_above_18_bits_without_sm107_descriptor,
        )


def test_f8f6f4_cta2_k32_extended_shared_span_is_gated_by_arch_not_k_bit():
    def emit_for_arch(arch):
        kernel = raw_tcgen_mma_f8f6f4_cta2_k32_extended_span_without_arch.with_attr(
            "tirx.cuda_arch", arch
        )
        return emit_rust_module(analyze(kernel), kernel)

    with pytest.raises(
        UnsupportedTIRxError,
        match="shared-memory virtual address span 270336 exceeds the 18-bit descriptor",
    ):
        emit_for_arch("sm_100a")

    source = emit_for_arch("sm_107a")
    assert "v2::tcgen05::variant::MatrixDescriptorSm107" in source


def test_raw_tcgen_tf32_shared_shared_matches_independent_matrix_product(tmp_path):
    row = np.arange(64, dtype=np.float32)[:, None]
    col = np.arange(8, dtype=np.float32)[:, None]
    k = np.arange(8, dtype=np.float32)[None, :]
    a = (row % np.float32(5) - np.float32(2)) * np.float32(0.5) + k * np.float32(0.25)
    b = (col % np.float32(7) - np.float32(3)) * np.float32(0.25) - k * np.float32(0.5)
    a += ((row + k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b += ((col + np.float32(2) * k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    a_bytes = np.ascontiguousarray(a, dtype=np.float32).view(np.uint8).reshape(64, 32)
    b_bytes = np.ascontiguousarray(b, dtype=np.float32).view(np.uint8).reshape(8, 32)

    module = numsim.transpile(raw_tcgen_mma_tf32_ss, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": _kmajor_swizzle_physical(a_bytes, swizzle_len=3, sdo=1024),
            "b_physical": _kmajor_swizzle_physical(b_bytes, swizzle_len=3, sdo=1024),
            "output": np.zeros((64, 8), dtype=np.float32),
        },
    )

    expected = _tcgen_fma_matmul(_decode_tcgen_tf32_payload(a), _decode_tcgen_tf32_payload(b))
    full_f32 = a @ b.T
    assert not np.array_equal(expected, full_f32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_mma_evaluates_collective_predicate_once():
    @T.prim_func
    def elected_mma():
        T.device_entry()
        T.warp_id([1])
        lane = T.lane_id([32])
        shared = T.alloc_buffer((32,), "uint32", scope="shared")
        shared[lane] = T.uint32(0)
        T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
            T.uint32(0), T.uint64(0), T.uint64(0), T.uint32((4 << 24) | (1 << 17) | 16),
            T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0), T.ptx.pred(T.uint32(0)),
            pred=T.cuda.elect_sync(),
        )

    source = emit_rust_module(analyze(elected_mma), elected_mma)
    assert source.count("v2::warp::elect_sync(") == 1


def test_raw_tcgen_tf32_low32_replacement_honors_predicate_and_accumulator(tmp_path):
    row = np.arange(64, dtype=np.float32)[:, None]
    k = np.arange(8, dtype=np.float32)[None, :]
    col = np.arange(32, dtype=np.float32)[:, None]
    a = ((row % np.float32(5)) - np.float32(2)) * np.float32(0.5) + k * np.float32(0.25)
    b = ((col % np.float32(7)) - np.float32(3)) * np.float32(0.25) - k * np.float32(0.5)
    a += ((row + k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b += ((col + np.float32(2) * k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b_bytes = np.ascontiguousarray(b.astype(np.float32)).view(np.uint8).reshape(32, 32)
    output = np.zeros((64, 32), dtype=np.float32)
    module = numsim.transpile(raw_tcgen_mma_tf32_ts_predicated, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a": np.ascontiguousarray(a, dtype=np.float32),
            "b_physical": _kmajor_swizzle_physical(b_bytes, swizzle_len=3, sdo=1024),
            "output": output,
        },
    )

    expected = _tcgen_fma_matmul(
        _decode_tcgen_tf32_payload(a),
        _decode_tcgen_tf32_payload(b),
        initial=np.float32(4),
    )
    expected[3, :] = np.float32(4)
    full_f32 = a @ b.T + np.float32(4)
    full_f32[3, :] = np.float32(4)
    assert not np.array_equal(expected, full_f32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_e4m3_m64_n8_uses_layout_f(tmp_path):
    finite_codes = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    row_a = np.arange(64, dtype=np.int64)[:, None]
    row_b = np.arange(8, dtype=np.int64)[:, None]
    k = np.arange(32, dtype=np.int64)[None, :]
    a_bits = finite_codes[(row_a * 3 + k * 5 + 1) % len(finite_codes)]
    b_bits = finite_codes[(row_b * 7 + k * 2 + 3) % len(finite_codes)]
    output = np.zeros((64, 8), dtype=np.float32)

    module = numsim.transpile(raw_tcgen_mma_e4m3_m64_n8, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
            "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
            "output": output,
        },
    )

    expected = _e4m3fn_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_f8f6f4_decodes_a_and_b_with_their_own_dtypes(tmp_path):
    """E5M2 A against E4M3 B, with the swapped decoder as the negative control.

    Both encodings are eight bits wide, so a lowering that reached the wrong
    decoder would still produce finite numbers. The control pins that the two
    readings genuinely disagree on these payloads.
    """

    # 0x3C is 1.0 as E5M2 and 1.5 as E4M3. 0x01..0x03 are the E5M2 subnormals
    # (mantissa over 2**-16) and 0x04 is the smallest E5M2 normal, 2**-14, so
    # both decoder branches are exercised with non-zero payloads.
    e5m2_codes = np.array(
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x38, 0x3C, 0x3D, 0x40, 0xBC, 0xC0], dtype=np.uint8
    )
    e4m3_codes = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    row_a = np.arange(64, dtype=np.int64)[:, None]
    row_b = np.arange(8, dtype=np.int64)[:, None]
    k = np.arange(32, dtype=np.int64)[None, :]
    a_bits = e5m2_codes[(row_a * 3 + k * 5 + 1) % len(e5m2_codes)]
    b_bits = e4m3_codes[(row_b * 7 + k * 2 + 3) % len(e4m3_codes)]

    module = numsim.transpile(raw_tcgen_mma_e5m2_e4m3_m64_n8, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
            "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
            "output": np.zeros((64, 8), dtype=np.float32),
        },
    )

    expected = _e5m2_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T
    np.testing.assert_array_equal(result.outputs["output"], expected)

    wrong_decoder = _e4m3fn_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T
    assert not np.array_equal(wrong_decoder, expected)


def test_raw_tcgen_f8f6f4_float16_destination_rounds_once_on_store(tmp_path):
    """The `.f16` destination is a codec, not a change of accumulator width.

    The addend is 1024.0, where binary16 has a ULP of 1, and 31 of the 32
    products are 0.25, so the exact sum 1031.75 is not representable in
    binary16: round-to-nearest on store gives 1032, truncation gives 1031, and
    rounding the accumulator between K steps keeps 1024. The seeded upper half
    is 0xBEEF, so a store that preserved it would also be caught.
    """

    a_bits = np.full((64, 32), 0x38, dtype=np.uint8)  # E4M3 1.0
    b_bits = np.full((8, 32), 0x28, dtype=np.uint8)  # E4M3 0.25
    b_bits[:, 0] = 0x00  # leave 31 contributing products, so the sum is not exact
    seed_low = int(np.array(1024.0, dtype=np.float16).view(np.uint16))
    seed = np.full((128, 8), (0xBEEF << 16) | seed_low, dtype=np.uint32)

    module = numsim.transpile(raw_tcgen_mma_f8f6f4_f16_destination_m64_n8, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
            "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
            "seed": seed,
            "output": np.zeros((64, 8), dtype=np.uint32),
        },
    )

    accumulated = _e4m3fn_bits_to_f32(a_bits) @ _e4m3fn_bits_to_f32(b_bits).T + np.float32(1024.0)
    # The store's rounding is only under test while the exact sum needs it.
    assert np.unique(accumulated).tolist() == [1031.75]
    assert np.float32(np.float16(np.float32(1031.75))) != np.float32(1031.75)

    expected = accumulated.astype(np.float16).view(np.uint16).astype(np.uint32)
    assert np.unique(expected).tolist() == [0x6408]  # binary16 1032.0, zero upper half
    np.testing.assert_array_equal(result.outputs["output"], expected)
    assert not np.array_equal(result.outputs["output"], np.full_like(expected, 0x6407))


def test_raw_tcgen_tf32_m128_n16_uses_regular_layout(tmp_path):
    row = np.arange(128, dtype=np.float32)[:, None]
    col = np.arange(16, dtype=np.float32)[:, None]
    k = np.arange(8, dtype=np.float32)[None, :]
    a = (row % np.float32(7) - np.float32(3)) * np.float32(0.25) + k * np.float32(0.125)
    b = (col % np.float32(5) - np.float32(2)) * np.float32(0.5) - k * np.float32(0.25)
    a += ((row + k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b += ((col + np.float32(2) * k) % np.float32(3) - np.float32(1)) * np.float32(2**-12)
    b_bytes = np.ascontiguousarray(b, dtype=np.float32).view(np.uint8).reshape(16, 32)
    output = np.zeros((128, 16), dtype=np.float32)

    module = numsim.transpile(raw_tcgen_mma_tf32_m128_n16, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a": np.ascontiguousarray(a, dtype=np.float32),
            "b_physical": _kmajor_swizzle_physical(b_bytes, swizzle_len=3, sdo=1024),
            "output": output,
        },
    )

    expected = _tcgen_fma_matmul(_decode_tcgen_tf32_payload(a), _decode_tcgen_tf32_payload(b))
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_descriptor_bits_are_validated_by_the_engine(tmp_path):
    module = numsim.transpile(
        raw_tcgen_mma_tf32_rejects_malformed_runtime_descriptor, cache_dir=tmp_path
    )

    with pytest.raises(
        numsim.NumSimExecutionError, match="descriptor uses unsupported reserved/base/LBO-mode bits"
    ):
        numsim.Engine().run(module, {})


def test_raw_tcgen_absolute_shared_descriptor_resolves_from_its_bits(tmp_path):
    module = numsim.transpile(raw_tcgen_cp_rejects_absolute_shared_descriptor, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {})
    assert result.verdict == "clean"


def test_raw_tcgen_mxf8_mn_major_matches_independent_matrix_product(tmp_path):
    finite = np.array([0x00, 0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8, 0xBC], dtype=np.uint8)
    row_a = np.arange(128, dtype=np.int64)[:, None]
    row_b = np.arange(16, dtype=np.int64)[:, None]
    k = np.arange(32, dtype=np.int64)[None, :]
    a_bits = finite[(row_a * 3 + k * 5 + 1) % len(finite)]
    b_bits = finite[(row_b * 7 + k * 3 + 2) % len(finite)]

    module = numsim.transpile(raw_tcgen_mma_mxf8_mn_major, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": _mnmajor_f8_swizzle_physical(a_bits),
            "b_physical": _mnmajor_f8_swizzle_physical(b_bits),
            "output": np.zeros((128, 16), dtype=np.float32),
        },
    )

    expected = _tcgen_fma_matmul(
        _e4m3fn_bits_to_f32(a_bits),
        _e4m3fn_bits_to_f32(b_bits),
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_mxf8_k_major_cta1_matches_independent_matrix_product(tmp_path):
    finite = np.array([0x00, 0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8, 0xBC], dtype=np.uint8)
    row_a = np.arange(128, dtype=np.int64)[:, None]
    row_b = np.arange(128, dtype=np.int64)[:, None]
    k = np.arange(128, dtype=np.int64)[None, :]
    a_bits = finite[(row_a * 3 + k * 5 + 1) % len(finite)]
    b_bits = finite[(row_b * 7 + k * 3 + 2) % len(finite)]

    module = numsim.transpile(raw_tcgen_mma_mxf8_k_major_cta1, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": _kmajor_swizzle_physical(a_bits, swizzle_len=3, sdo=1024),
            "b_physical": _kmajor_swizzle_physical(b_bits, swizzle_len=3, sdo=1024),
            "output": np.zeros((128, 128), dtype=np.float32),
        },
    )

    expected = _tcgen_fma_matmul(
        _e4m3fn_bits_to_f32(a_bits),
        _e4m3fn_bits_to_f32(b_bits),
    )
    np.testing.assert_array_equal(result.outputs["output"], expected)


def test_raw_tcgen_mixed_block_scale_cta_group2_gathers_and_scatters_both_ctas(tmp_path):
    finite_fp8 = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    cta = np.arange(2, dtype=np.int64)[:, None, None]
    row_a = np.arange(128, dtype=np.int64)[None, :, None]
    row_b = np.arange(16, dtype=np.int64)[None, :, None]
    k = np.arange(64, dtype=np.int64)[None, None, :]
    a_codes = ((cta * 5 + row_a * 3 + k * 7 + 1) % 16).astype(np.uint8)
    # Each of the two K=32 instructions consumes two 16-byte A atoms.  For
    # mxf8f6f4 E2M1, each atom stores sixteen packed values in its first eight
    # bytes and ignores the remaining eight padding bytes.
    a_padded = np.full(a_codes.shape, np.uint8(0xA5), dtype=np.uint8)
    for k_block in range(2):
        for atom in range(2):
            code_start = k_block * 32 + atom * 16
            storage_start = k_block * 32 + atom * 16
            codes = a_codes[:, :, code_start : code_start + 16]
            a_padded[:, :, storage_start : storage_start + 8] = codes[:, :, 0::2] | (
                codes[:, :, 1::2] << np.uint8(4)
            )
    b_bits = finite_fp8[(cta * 2 + row_b * 5 + k * 3 + 2) % len(finite_fp8)]

    scale_a_bits = np.full((2, 128, 4), np.uint8(127), dtype=np.uint8)
    scale_b_bits = np.full((32, 4), np.uint8(127), dtype=np.uint8)
    scale_a_bits[0, :, 0] = np.uint8(126)
    scale_a_bits[1, :, 0] = np.uint8(128)
    scale_a_bits[0, :, 1] = np.uint8(128)
    scale_a_bits[1, :, 1] = np.uint8(126)
    scale_b_bits[:16, 0] = np.uint8(128)
    scale_b_bits[16:, 0] = np.uint8(126)
    scale_b_bits[:16, 1] = np.uint8(126)
    scale_b_bits[16:, 1] = np.uint8(128)
    scale_a_cells = np.stack([_pack_e8m0_cells(scale_a_bits[index]) for index in range(2)])
    scale_b_cells = np.broadcast_to(_pack_e8m0_cells(scale_b_bits), (2, 1, 32)).copy()

    output = np.zeros((2, 128, 32), dtype=np.float32)
    module = numsim.transpile(raw_tcgen_mma_mixed_fp4_fp8_cta_group2, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "a_physical": np.stack(
                [
                    _kmajor_swizzle_physical(a_padded[index], swizzle_len=3, sdo=1024)
                    for index in range(2)
                ]
            ),
            "b_physical": np.stack(
                [
                    _kmajor_swizzle_physical(b_bits[index], swizzle_len=3, sdo=1024)
                    for index in range(2)
                ]
            ),
            "scale_a_cells": scale_a_cells,
            "scale_b_cells": scale_b_cells,
            "output": output,
        },
    )

    values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    a = values[a_codes & 7] * np.where(a_codes & 8, np.float32(-1), np.float32(1))
    b = _e4m3fn_bits_to_f32(b_bits)
    a *= np.repeat(
        np.exp2(scale_a_bits[:, :, (0, 1)].astype(np.int16) - 127).astype(np.float32),
        32,
        axis=2,
    )
    b *= np.repeat(
        np.exp2(scale_b_bits.reshape(2, 16, 4)[:, :, (0, 1)].astype(np.int16) - 127).astype(
            np.float32
        ),
        32,
        axis=2,
    )
    joint_a = a.reshape(256, 64)
    joint_b = b.reshape(32, 64)
    expected = (joint_a @ joint_b.T).reshape(2, 128, 32)
    np.testing.assert_array_equal(result.outputs["output"], expected)


@pytest.mark.parametrize("cta_group", (1, 2))
def test_raw_tcgen_mxf4nvf4_block16_and_scale_vec_4x_share_the_mma_variant(cta_group):
    def mma_head(scale: str) -> str:
        spelling = f"tcgen05.mma.cta_group::{cta_group}.kind::mxf4nvf4.block_scale.{scale}"

        @T.prim_func
        def probe():
            T.device_entry()
            _shared = T.alloc_buffer((1,), "uint8", scope="shared")
            T.ptx[spelling](
                T.uint32(0),
                T.uint64(0),
                T.uint64(0),
                T.uint32(0),
                T.uint32(0),
                T.uint32(0),
                T.ptx.pred(T.uint32(0)),
            )

        (mma,) = [
            call
            for call in emitted_calls(
                probe, lambda op_name: op_name.startswith("tirx.ptx.tcgen05_")
            )
            if call.function == "v2::tcgen05::mma"
        ]
        return mma.head

    # K=64 gives four block16 scales (checked by the Rust descriptor test).
    # For other K, fixed block size and fixed vector count are not aliases:
    # keep the explicit vector policy in the emitted variant.
    block = mma_head("block16")
    assert mma_head("scale_vec::4X") == block.replace(
        "MatrixDescriptorSm100>",
        "MatrixDescriptorSm100, true>",
    )
    assert mma_head("block16.collector::a::discard") == block
    assert mma_head("block16.collector::a::discard.collector::b::discard") == block
    # Only the block spelling has collector slots in the current target table.
    for operation, a_masks, b_masks in (
        ("discard", "0, 0, 3", "0, 0, 3"),
        ("fill", "1, 0, 2", "2, 0, 1"),
        ("use", "0, 1, 2", "0, 2, 1"),
        ("lastuse", "0, 1, 3", "0, 2, 3"),
    ):
        for collector, masks in (
            (f"collector::a::{operation}", a_masks),
            # The PTX schema requires an explicit A field when B is present.
            (f"collector::a::discard.collector::b::{operation}", b_masks),
        ):
            assert mma_head(f"block16.{collector}") == block.replace("0, 0, 3>", f"{masks}>")
