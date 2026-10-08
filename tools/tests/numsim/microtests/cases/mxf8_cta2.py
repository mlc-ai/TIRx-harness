"""Raw MXF8 M=128 CTA-pair probe: nonzero data and independent per-row scales."""

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tests.numsim.microtests.harness import PairedBuffer
from tvm.tirx.layout import tmem_datapath_layout

_A_LAYOUT = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_32B_ATOM, (64, 32))
_B_LAYOUT = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_32B_ATOM, (16, 32))


@T.jit
def _raw_mxf8_cta2_m128(
    a: T.Buffer((2, 64, 32), "float8_e4m3fn"),
    b: T.Buffer((2, 16, 32), "float8_e4m3fn"),
    scale_a: T.Buffer((2, 4, 32), "uint32"),
    scale_b: T.Buffer((2, 2, 4, 32), "uint32"),
    output: T.Buffer((2, 4, 32, 16), "float32"),
    *,
    REPLICAS: T.constexpr,
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    address = T.alloc_buffer((1,), "uint32", scope="shared", layout=None)
    barriers = T.alloc_buffer((1,), "uint64", scope="shared")
    shared_a = T.alloc_buffer((64, 32), "float8_e4m3fn", scope="shared", layout=_A_LAYOUT)
    shared_b = T.alloc_buffer((16, 32), "float8_e4m3fn", scope="shared", layout=_B_LAYOUT)
    # Declare the physical allocation used by the raw TMEM addresses below.
    tmem = T.decl_buffer(
        (128, 32),
        "uint32",
        scope="tmem",
        layout=tmem_datapath_layout("D", 128, 32),
        allocated_addr=address[0],
    )
    registers = T.alloc_local((16,), "uint32")
    desc_i: T.uint32
    desc_a: T.uint64
    desc_b: T.uint64

    if warp == 0 and lane == 0:
        Tx.copy(shared_a[:, :], a[cta, :, :])
        Tx.copy(shared_b[:, :], b[cta, :, :])
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barriers[0]), 1)
    T.ptx.fence.proxy.async_.shared__cta()
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()

    # SFA is replicated four times. For M128 CTA2, SFB places the two N
    # halves in lane partitions 0/1 and 2/3 (CUTLASS ScaleFactorDuplicated2by2).
    if warp < REPLICAS:
        T.ptx["tcgen05.st.sync.aligned.32x32b.x4.b32"](
            address[0] + T.uint32(16),
            scale_a[cta, 0, lane],
            scale_a[cta, 1, lane],
            scale_a[cta, 2, lane],
            scale_a[cta, 3, lane],
        )
        T.ptx["tcgen05.st.sync.aligned.32x32b.x4.b32"](
            address[0] + T.uint32(20),
            scale_b[cta, warp // 2, 0, lane],
            scale_b[cta, warp // 2, 1, lane],
            scale_b[cta, warp // 2, 2, lane],
            scale_b[cta, warp // 2, 3, lane],
        )
        T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.tcgen05.fence__before_thread_sync()
    T.cuda.cluster_sync()
    T.ptx.tcgen05.fence__after_thread_sync()

    if cta == 0 and warp == 0 and lane == 0:
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
            N=32,
            K=32,
            trans_a=False,
            trans_b=False,
            n_cta_groups=2,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_a),
            T.address_of(shared_a[0, 0]),
            ldo=0,
            sdo=16,
            swizzle=1,
        )
        T.cuda.tcgen05.encode_matrix_descriptor(
            T.address_of(desc_b),
            T.address_of(shared_b[0, 0]),
            ldo=0,
            sdo=16,
            swizzle=1,
        )
        T.ptx["tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"](
            address[0],
            desc_a,
            desc_b,
            desc_i,
            address[0] + T.uint32(16),
            address[0] + T.uint32(20),
            False,
        )
        T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.b64(
            T.address_of(barriers[0])
        )
    if cta == 0 and warp == 0:
        T.cuda.mbarrier_wait(T.address_of(barriers[0]), 0)
    T.cuda.cluster_sync()

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
        output[cta, warp, lane, register] = T.reinterpret("float32", registers[register])
    T.cuda.cluster_sync()

    if warp == 0:
        T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()


raw_mxf8_cta2_m128 = _raw_mxf8_cta2_m128.specialize(REPLICAS=4)


def make_arguments():
    finite = np.array([0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    cta = np.arange(2)[:, None, None]
    row_a = np.arange(64)[None, :, None]
    row_b = np.arange(16)[None, :, None]
    k = np.arange(32)[None, None, :]
    a = finite[(cta * 3 + row_a * 2 + k * 3 + 1) % len(finite)]
    b = finite[(cta * 2 + row_b * 5 + k * 2 + 2) % len(finite)]
    rows = np.arange(128)[None, :, None]
    sa = np.broadcast_to((126 + (cta + rows) % 3).astype(np.uint8), (2, 128, 4)).copy()
    b_rows = np.arange(128)[None, :] + np.arange(2)[:, None] * 16
    sb = np.broadcast_to(
        (126 + (b_rows // 8) % 3).astype(np.uint8)[None, :, :, None], (2, 2, 128, 4)
    ).copy()
    return {
        "a": PairedBuffer(a, "float8_e4m3fn"),
        "b": PairedBuffer(b, "float8_e4m3fn"),
        "scale_a": sa.view(np.uint32).reshape(2, 4, 32),
        "scale_b": sb.view(np.uint32).reshape(2, 2, 4, 32),
        "output": np.full((2, 4, 32, 16), np.nan, dtype=np.float32),
    }


def reference(arguments):
    import ml_dtypes

    a = arguments["a"].array.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    b = arguments["b"].array.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    sa = (
        arguments["scale_a"]
        .view(ml_dtypes.float8_e8m0fnu)
        .reshape(2, 128, 4)[:, :64, 0]
        .astype(np.float32)
    )
    sb = (
        arguments["scale_b"]
        .view(ml_dtypes.float8_e8m0fnu)
        .reshape(2, 2, 128, 4)[0, :, :16, 0]
        .reshape(32)
        .astype(np.float32)
    )
    logical = (a * sa[:, :, None]) @ (b.reshape(32, 32) * sb[:, None]).T
    physical = np.concatenate((logical[:, :, :16], logical[:, :, 16:]), axis=1)
    return physical.reshape(2, 4, 32, 16)
