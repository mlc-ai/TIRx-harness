#!/usr/bin/env python3
"""B200 probes for the TCGEN05 NVFP4 scale-factor order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tvm.backend.cuda.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import tmem_datapath_layout


_FP4_A_SMEM_LAYOUT = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (128, 32))
_FP4_B_SMEM_LAYOUT = mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, (8, 32))
_NVFP4_SCALE_TMEM_LAYOUT = sf_tmem_layout(128, SF_K=4, sf_per_mma=4)


@T.jit
def _nvfp4_scale_order_probe(
    output: T.Buffer((2, 16), "float32"),
    *,
    CTA_GROUP: T.constexpr,
):
    """Isolate the four E4M3 scale slots with canonically aligned MMA operands."""

    C_N = T.meta_var(8 * CTA_GROUP)
    T.device_entry()
    cta: T.int32 = 0
    if CTA_GROUP == 1:
        cta = T.cta_id([1])
    else:
        _cluster = T.cluster_id([1])
        cta = T.cta_id_in_cluster([CTA_GROUP])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    tmem_address = T.alloc_buffer((1,), "uint32", scope="shared", layout=None)
    completion = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a_packed = T.alloc_buffer(
        (128, 32), "uint8", scope="shared", layout=_FP4_A_SMEM_LAYOUT, align=1024
    )
    shared_b_packed = T.alloc_buffer(
        (8, 32), "uint8", scope="shared", layout=_FP4_B_SMEM_LAYOUT, align=1024
    )
    shared_a = shared_a_packed.view("float4_e2m1fn")
    shared_b = shared_b_packed.view("float4_e2m1fn")
    accumulator = T.decl_buffer(
        (128, C_N),
        "float32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, C_N),
        allocated_addr=tmem_address[0],
    )
    scale_a = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=_NVFP4_SCALE_TMEM_LAYOUT,
        allocated_addr=tmem_address[0] + T.uint32(32),
    )
    scale_b = T.decl_buffer(
        (128, 4),
        "float8_e4m3fn",
        scope="tmem",
        layout=_NVFP4_SCALE_TMEM_LAYOUT,
        allocated_addr=tmem_address[0] + T.uint32(48),
    )
    scale_a_registers = T.alloc_local((4,), "uint32")
    scale_b_registers = T.alloc_local((4,), "uint32")
    registers = T.alloc_local((C_N,), "uint32")

    # E2M1 code 2 is 1.0. B rows 0..3 select K16 chunks 0..3, so the
    # first four outputs expose the scale order in one MMA operation.
    if (warp == 0) & (lane == 0):
        for row in T.serial(128):
            for packed_k in T.serial(32):
                shared_a_packed[row, packed_k] = T.uint8(0x22)
        for row in T.serial(8):
            for packed_k in T.serial(32):
                shared_b_packed[row, packed_k] = T.if_then_else(
                    row < 4 and packed_k // 8 == row,
                    T.uint8(0x22),
                    T.uint8(0),
                )

    if warp == 0:
        T.ptx[f"tcgen05.alloc.cta_group::{CTA_GROUP}.sync.aligned.shared::cta.b32"](
            T.address_of(tmem_address[0]), 64
        )
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(completion[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    if CTA_GROUP == 1:
        T.cuda.cta_sync()
    else:
        T.cuda.cluster_sync()

    # The four bytes are E4M3 encodings for 0.5, 1.0, 2.0, and 4.0.
    for column in T.unroll(4):
        scale_a_registers[column] = T.uint32(0x48403830)
        scale_b_registers[column] = T.uint32(0x38383838)
    T.ptx["tcgen05.st.sync.aligned.32x32b.x4.b32"](
        tmem_address[0] + T.uint32(32),
        scale_a_registers[0],
        scale_a_registers[1],
        scale_a_registers[2],
        scale_a_registers[3],
    )
    T.ptx["tcgen05.st.sync.aligned.32x32b.x4.b32"](
        tmem_address[0] + T.uint32(48),
        scale_b_registers[0],
        scale_b_registers[1],
        scale_b_registers[2],
        scale_b_registers[3],
    )
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    if CTA_GROUP == 1:
        T.cuda.cta_sync()
    else:
        T.cuda.cluster_sync()

    if (cta == 0) & (warp == 0) & (lane == 0):
        Tx.gemm_async(
            accumulator[:, :],
            shared_a[:, :],
            shared_b[:, :],
            SFA=scale_a[:, :],
            SFB=scale_b[:, :],
            accum=False,
            dispatch="tcgen05",
            cta_group=CTA_GROUP,
        )
        T.ptx[f"tcgen05.commit.cta_group::{CTA_GROUP}.mbarrier::arrive::one.shared::cluster.b64"](
            T.address_of(completion[0])
        )
    if (cta == 0) & (warp == 0):
        T.cuda.mbarrier_wait(T.address_of(completion[0]), 0)
    if CTA_GROUP == 1:
        T.cuda.cta_sync()
    else:
        T.cuda.cluster_sync()

    if CTA_GROUP == 1:
        T.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
            registers[0],
            registers[1],
            registers[2],
            registers[3],
            registers[4],
            registers[5],
            registers[6],
            registers[7],
            tmem_address[0],
        )
    else:
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
    if (warp == 0) & (lane == 0):
        for column in T.serial(C_N):
            output[cta, column] = T.reinterpret("float32", registers[column])
    if CTA_GROUP == 1:
        T.cuda.cta_sync()
    else:
        T.cuda.cluster_sync()

    if warp == 0:
        T.ptx[f"tcgen05.dealloc.cta_group::{CTA_GROUP}.sync.aligned.b32"](tmem_address[0], 64)
        T.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{CTA_GROUP}.sync.aligned"]()


@dataclass(frozen=True)
class Tcgen05ScaleAbiCase:
    name: str
    prim_func: Any
    cta_group: int

    def make_arguments(self) -> dict[str, np.ndarray]:
        return {"output": np.zeros((2, 16), dtype=np.float32)}

    def expected_output(self) -> np.ndarray:
        local = np.zeros(8, dtype=np.float32)
        local[:4] = (8.0, 16.0, 32.0, 64.0)
        expected = np.zeros((2, 16), dtype=np.float32)
        expected[: self.cta_group, : 8 * self.cta_group] = np.tile(local, self.cta_group)
        return expected


TCGEN05_SCALE_ABI_CASES = tuple(
    Tcgen05ScaleAbiCase(
        name=f"nvfp4_scale_order_cta{cta_group}",
        prim_func=_nvfp4_scale_order_probe.specialize(CTA_GROUP=cta_group),
        cta_group=cta_group,
    )
    for cta_group in (1, 2)
)


__all__ = ["TCGEN05_SCALE_ABI_CASES", "Tcgen05ScaleAbiCase"]
