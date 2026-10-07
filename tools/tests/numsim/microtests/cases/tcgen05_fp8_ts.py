"""Packed TMEM A probes; inputs and output layout have independent host oracles."""

import numpy as np
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout

from tests.numsim.microtests.cases.tcgen05_mma_forms import _F8_BYTES_B16_SMEM_LAYOUT


def fp8_ts_kernel(m=64, d_f16=False, publish=True):
    a_dtype = "float8_e5m2" if d_f16 else "float8_e4m3fn"
    b_dtype = "float8_e4m3fn" if d_f16 else "float8_e5m2"
    d_dtype = "float16" if d_f16 else "float32"
    b_shape = (32, 16) if d_f16 else (16, 32)
    b_layout = TileLayout(S[b_shape]) if d_f16 else _F8_BYTES_B16_SMEM_LAYOUT

    @T.prim_func
    def kernel(
        a_words: T.Buffer((4, 32, 8), "uint32"),
        b: T.Buffer(b_shape, "uint8"),
        output: T.Buffer((m, 16), "uint32"),
        issue: T.uint32,
    ):
        T.device_entry()
        _wg = T.warpgroup_id([1])
        warp = T.warp_id_in_wg([4])
        lane = T.lane_id([32])
        address = T.alloc_buffer((1,), "uint32", scope="shared")
        barrier = T.alloc_buffer((1,), "uint64", scope="shared")
        shared_b = T.alloc_buffer(b_shape, "uint8", scope="shared", layout=b_layout, align=256)
        registers = T.alloc_local((16,), "uint32")
        descriptor: T.uint32
        descriptor_b: T.uint64
        if warp == 0 and lane == 0:
            Tx.copy(shared_b[:, :], b[:, :])
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address[0]), 32
            )
            if lane == 0:
                T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()

        # M=64 leaves half the datapath unused; initialize those cells too
        # because the readback uses a full 32-lane transfer.
        for i in T.unroll(16):
            registers[i] = T.uint32(0)
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
        for i in T.unroll(8):
            registers[i] = a_words[warp, lane, i]
        T.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
            address[0] + T.uint32(16),
            registers[0],
            registers[1],
            registers[2],
            registers[3],
            registers[4],
            registers[5],
            registers[6],
            registers[7],
        )
        if publish:
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.ptx.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

        for phase in T.serial(2):
            if warp == 0 and lane == 0:
                T.cuda.tcgen05.encode_instr_descriptor(
                    T.address_of(descriptor),
                    d_dtype=d_dtype,
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    M=m,
                    N=16,
                    K=32,
                    trans_a=False,
                    trans_b=d_f16,
                    n_cta_groups=1,
                )
                T.cuda.tcgen05.encode_matrix_descriptor(
                    T.address_of(descriptor_b),
                    T.address_of(shared_b[0, 0]),
                    ldo=8 if d_f16 else 16,
                    sdo=1 if d_f16 else 16,
                    swizzle=0 if d_f16 else 1,
                )
                if d_f16:
                    # Predicated-off invalid TMEM addresses must have no effects.
                    T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                        address[0],
                        T.uint32(0xFFFFFFFF),
                        descriptor_b,
                        descriptor,
                        0,
                        0,
                        0,
                        0,
                        T.ptx.pred(T.uint32(1)),
                        pred=T.uint32(1) - issue,
                    )
                    T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                        address[0],
                        address[0] + T.uint32(16),
                        descriptor_b,
                        descriptor | T.uint32(1 << 13),
                        0,
                        0,
                        0,
                        0,
                        T.ptx.pred(T.cast(phase, "uint32")),
                        pred=issue,
                    )
                else:
                    T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                        address[0],
                        address[0] + T.uint32(16),
                        descriptor_b,
                        descriptor,
                        0,
                        0,
                        0,
                        0,
                        T.ptx.pred(T.cast(phase, "uint32")),
                    )
                T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                    T.address_of(barrier[0])
                )
            if warp == 0:
                T.cuda.mbarrier_wait(T.address_of(barrier[0]), phase)
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
        if m == 128 or lane < 16:
            for i in T.unroll(16):
                output[warp * (m // 4) + lane, i] = registers[i]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()

    return kernel


def fp8_ts_arguments(m=64, d_f16=False):
    # Exactly representable dyadic values: the independent dot product is
    # insensitive to reduction order, but distinguishes codecs and byte order.
    e4 = np.array([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
    e5 = np.array([0x00, 0x38, 0x3C, 0x40, 0xB8, 0xBC, 0xC0], dtype=np.uint8)
    values = np.array([0, 0.5, 1, 2, -0.5, -1, -2], dtype=np.float32)
    rows, k = np.arange(m)[:, None], np.arange(32)[None, :]
    ai = (rows * 3 + k * 5 + 1) % len(values)
    bi = (np.arange(16)[:, None] * 2 + k * 3 + 3) % len(values)
    a = (e5 if d_f16 else e4)[ai]
    b = (e4 if d_f16 else e5)[bi]
    packed = np.zeros((128, 8), dtype=np.uint32)
    physical_rows = np.arange(m) if m == 128 else (np.arange(m) // 16) * 32 + np.arange(m) % 16
    for byte in range(4):
        packed[physical_rows] |= a[:, byte::4].astype(np.uint32) << np.uint32(8 * byte)
    # Negate the operand, not the completed dot: exact cancellation from a
    # +0 accumulator produces +0, whereas negating the final zero makes -0.
    product = (-values[ai] if d_f16 else values[ai]) @ values[bi].T
    if d_f16:
        expected = (product.astype(np.float16).astype(np.float32) + product).astype(np.float16)
        expected = expected.view(np.uint16).astype(np.uint32)
    else:
        expected = (product * np.float32(2)).view(np.uint32)
    return {
        "a_words": packed.reshape(4, 32, 8),
        "b": b.T.copy() if d_f16 else b,
        "output": np.zeros((m, 16), dtype=np.uint32),
        "issue": 1,
    }, expected
