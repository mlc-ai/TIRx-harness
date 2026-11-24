"""SM107 shared-memory matrix descriptors carry a 15-bit start field for every kind.

PTX ISA 9.5 widens the tcgen05 shared-memory descriptor start address to bits
0-14, so SM107 operands above 256 KB set bit 14. NumSim and both checkers must
decode that bit for every MMA kind, not only the F8F6F4 forms, and TIRx's
`encode_matrix_descriptor` (apache/tvm-tirx #254) must write it.
"""

import numpy as np
import pytest

from tests.numsim.support.execution import with_cuda_arch
from tirx_harness import numsim, racecheck, synccheck
from tvm_tirx.script import tirx as T

OFFSET = 1 << 18  # The first byte whose descriptor start needs bit 14.

# kind -> (instruction descriptor fields or raw word, a 32-bit word of
#          operand elements, expected accumulator, accumulator dtype)
_KINDS = {
    "f16": ((("float32", "bfloat16", "bfloat16", 16)), 0x3F803F80, 16.0, np.float32),
    "tf32": ((("float32", "tf32", "tf32", 8)), 0x3F800000, 8.0, np.float32),
    # PTX 9.4 Table 51: S32 accumulator, s1z4m11 operands (format 3), K=16.
    "ti16": ((2 << 4) | (3 << 7) | (3 << 10) | (1 << 17) | (4 << 24), 0x00010001, 16, np.int32),
}


def _kernel(kind, a_in_tmem, patch_start):
    instruction, element_word, _, _ = _KINDS[kind]
    raw_instruction = isinstance(instruction, int)
    d_dtype, a_dtype, b_dtype, k = (
        ("float32", "bfloat16", "bfloat16", 16) if raw_instruction else instruction
    )
    raw_word = instruction if raw_instruction else 0
    op = f"tcgen05.mma.cta_group::1.kind::{kind}"

    # Every row is 32 bytes of K in a 128B-swizzled K-major layout: M=64 A is
    # 8 KB and N=8 B is 1 KB. Each warp addresses its own 32-lane TMEM quarter:
    # D is columns [0, 8) and the TMEM A operand columns [16, 24).
    @T.function
    def kernel(output: T.Tensor((64, 8), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        lane = T.cuda.lane_id()
        barrier = T.alloc_shared((1,), "uint64")
        arena = T.alloc_shared((OFFSET + 9216,), "uint8", align=1024)
        a = T.decl_tensor(
            T.ptr_byte_offset(arena.data, OFFSET * 1, ty=T.Ptr("uint32", "shared")),
            ty_args=[T.Tensor((2048,), "uint32", scope="shared")],
        )
        b = T.decl_tensor(
            T.ptr_byte_offset(arena.data, (OFFSET + 8192) * 1, ty=T.Ptr("uint32", "shared")),
            ty_args=[T.Tensor((256,), "uint32", scope="shared")],
        )
        address = T.alloc_shared((1,), "uint32")
        word = T.alloc_local((1,), "uint32")
        da: T.uint64
        db: T.uint64
        di: T.uint32
        # A start field truncated to 14 bits would alias these zero bytes.
        for i in T.serial(9216 // 128):
            arena[tid + i * 128] = T.uint8(0)
        for i in T.serial(2048 // 128):
            a[tid + i * 128] = T.uint32(element_word)
        for i in T.serial(256 // 128):
            b[tid + i * 128] = T.uint32(element_word)
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address[0]), 32
            )
            if lane == 0:
                T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.tcgen05.fence__before_thread_sync()
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        if T.constexpr(a_in_tmem):
            word[0] = T.uint32(element_word)
            for col in T.unroll(8):
                T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](
                    address[0] + T.uint32(16 + col), word[0]
                )
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.ptx.tcgen05.fence__before_thread_sync()
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence__after_thread_sync()
        if warp == 0 and lane == 0:
            if T.constexpr(raw_instruction):
                di = T.uint32(raw_word)
            else:
                T.cuda.tcgen05.encode_instr_descriptor(
                    T.address_of(di),
                    d_dtype=d_dtype,
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                    M=64,
                    N=8,
                    K=k,
                    trans_a=False,
                    trans_b=False,
                    n_cta_groups=1,
                )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(da), T.address_of(a[0]), ldo=0, sdo=64, swizzle=3
            )
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(db), T.address_of(b[0]), ldo=0, sdo=64, swizzle=3
            )
            if T.constexpr(patch_start):
                # Supply the 15-bit start field independently of the encoder.
                da = (da & T.bitwise_not(T.uint64(0x7FFF))) | T.cast(
                    T.shift_right(T.cuda.cvta_generic_to_shared(a.ptr_to([0])), T.uint32(4)),
                    "uint64",
                )
                db = (db & T.bitwise_not(T.uint64(0x7FFF))) | T.cast(
                    T.shift_right(T.cuda.cvta_generic_to_shared(b.ptr_to([0])), T.uint32(4)),
                    "uint64",
                )
            if T.constexpr(a_in_tmem):
                T.ptx[op](address[0], address[0] + T.uint32(16), db, di, 0, 0, 0, 0, T.ptx.pred(0))
            else:
                T.ptx[op](address[0], da, db, di, 0, 0, 0, 0, T.ptx.pred(0))
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barrier[0])
            )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        # M=64 Layout F: rows 16w..16w+15 live in the first 16 lanes of quarter w.
        for col in T.unroll(8):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(col))
            T.ptx.tcgen05.wait__ld.sync.aligned()
            if lane < 16:
                output[warp * 16 + lane, col] = word[0]
        T.ptx.tcgen05.fence__before_thread_sync()
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()

    return kernel


@pytest.mark.parametrize(
    ("kind", "a_in_tmem", "patch_start"),
    [
        ("f16", False, True),
        ("f16", True, True),
        ("tf32", False, True),
        ("ti16", False, True),
        # The modeled TIRx encoder writes the 15-bit start field on its own.
        ("f16", False, False),
    ],
    ids=["f16-ss", "f16-ts", "tf32-ss", "ti16-ss", "f16-ss-encoder"],
)
def test_sm107_mma_decodes_descriptor_start_above_256k(tmp_path, kind, a_in_tmem, patch_start):
    *_, expected, dtype = _KINDS[kind]
    kernel = with_cuda_arch(_kernel(kind, a_in_tmem, patch_start), "sm_107a")
    arguments = {"output": np.zeros((64, 8), dtype=np.uint32)}
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine(max_workers=1).run(module, arguments)
    np.testing.assert_array_equal(result.outputs["output"].view(dtype), dtype(expected))
    for checker in (synccheck, racecheck):
        checker(kernel, arguments).require_clean()


@pytest.mark.parametrize("arch", ["sm_100a", "sm_107a"])
def test_encode_matrix_descriptor_writes_15_bit_start(tmp_path, arch):
    # SM100 blocks get at most 232448 bytes of shared memory, so they never
    # reach 256 KB; the same encoder only sets bit 14 on SM107.
    offset = OFFSET if arch == "sm_107a" else 200 * 1024

    @T.function
    def kernel(output: T.Tensor((2,), "uint64")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        arena = T.alloc_shared((offset + 1024,), "uint8", align=1024)
        descriptor: T.uint64
        arena[offset + lane] = T.uint8(0)
        if lane == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(
                T.address_of(descriptor),  # noqa: F823
                T.address_of(arena[offset]),
                ldo=1,
                sdo=64,
                swizzle=3,
            )
            output[0] = descriptor
            output[1] = T.cast(T.cuda.cvta_generic_to_shared(T.address_of(arena[offset])), "uint64")

    module = numsim.transpile(with_cuda_arch(kernel, arch), cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"output": np.zeros(2, dtype=np.uint64)})
    descriptor, address = (int(value) for value in result.outputs["output"])
    assert (address >= 1 << 18) == (arch == "sm_107a")
    expected = (address >> 4) | (1 << 16) | (64 << 32) | (1 << 46) | (2 << 61)
    assert descriptor == expected, f"{descriptor:#018x} != {expected:#018x}"
