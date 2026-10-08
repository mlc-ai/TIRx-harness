from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from tvm.script import tirx as T


@T.prim_func
def tcgen05_lifecycle_ldst(output: T.Buffer((4, 32), "uint32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    value = T.alloc_local((1,), "uint32")

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
    T.cuda.cta_sync()

    value[0] = T.cast(1000 + warp * 32 + lane, "uint32")
    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0], value[0])
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.cuda.cta_sync()

    value[0] = T.uint32(0)
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value[0], address[0])
    T.ptx.tcgen05.wait__ld.sync.aligned()
    output[warp, lane] = value[0]
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tcgen05_cp_warpx4(source: T.Buffer((32, 4), "uint32"), output: T.Buffer((4, 32, 4), "uint32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared = T.alloc_buffer((32, 4), "uint32", scope="shared")
    registers = T.alloc_local((4,), "uint32")
    descriptor: T.uint64

    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        for column in T.unroll(4):
            shared[lane, column] = source[lane, column]
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor), T.address_of(shared[0, 0]), ldo=0, sdo=8, swizzle=0
        )
        T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](address[0], descriptor)
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barrier[0])
        )
    if warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
    T.cuda.cta_sync()

    T.ptx["tcgen05.ld.sync.aligned.32x32b.x4.b32"](
        registers[0], registers[1], registers[2], registers[3], address[0]
    )
    T.ptx.tcgen05.wait__ld.sync.aligned()
    for column in T.unroll(4):
        output[warp, lane, column] = registers[column]
    T.cuda.cta_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()


@T.prim_func
def tcgen05_bf16_mma(output: T.Buffer((4, 32, 4), "float32")):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared")
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((4096,), "uint8", scope="shared")
    shared_b = T.alloc_buffer((1024,), "uint8", scope="shared")
    registers = T.alloc_local((4,), "uint32")
    instruction_descriptor: T.uint32
    descriptor_a: T.uint64
    descriptor_b: T.uint64

    for copy_index in T.serial(32):
        offset_a = (warp * 32 + lane) + copy_index * 128
        shared_a[offset_a] = T.cast(T.if_then_else(offset_a % 2 == 0, 0x80, 0x3F), "uint8")
    for copy_index in T.serial(8):
        offset_b = (warp * 32 + lane) + copy_index * 128
        shared_b[offset_b] = T.cast(T.if_then_else(offset_b % 2 == 0, 0x80, 0x3F), "uint8")
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    if warp == 0 and lane == 0:
        T.cuda.tcgen05.encode_instr_descriptor(
            T.address_of(instruction_descriptor),
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
            T.address_of(descriptor_a), T.address_of(shared_a[0]), ldo=0, sdo=32, swizzle=2
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(descriptor_b), T.address_of(shared_b[0]), ldo=0, sdo=32, swizzle=2
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


@dataclass(frozen=True)
class Tcgen05CoreCase:
    name: str
    prim_func: Any
    make_arguments: Callable[[], Mapping[str, Any]]
    outputs: tuple[str, ...]


TCGEN05_CORE_CASES = (
    Tcgen05CoreCase(
        "lifecycle_ldst",
        tcgen05_lifecycle_ldst,
        lambda: {"output": np.zeros((4, 32), dtype=np.uint32)},
        ("output",),
    ),
    Tcgen05CoreCase(
        "cp_warpx4",
        tcgen05_cp_warpx4,
        lambda: {
            "source": np.arange(32 * 4, dtype=np.uint32).reshape(32, 4),
            "output": np.zeros((4, 32, 4), dtype=np.uint32),
        },
        ("output",),
    ),
    Tcgen05CoreCase(
        "bf16_mma",
        tcgen05_bf16_mma,
        lambda: {"output": np.zeros((4, 32, 4), dtype=np.float32)},
        ("output",),
    ),
)
