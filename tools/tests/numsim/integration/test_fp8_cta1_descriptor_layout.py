"""FP8 CTA1 must decode the same shared addresses in NumSim and both checkers."""

import numpy as np
import pytest

from tests.numsim.support.execution import with_cuda_arch
from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.errors import UnsupportedTIRxError
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.layout import Axis, S, TileLayout

_TMEM = TileLayout(S[(128, 32) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))])


def _kernel(offset, a_in_tmem):
    @T.function
    def kernel(output: T.Tensor((64, 8), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.linear_thread_id()
        barrier = T.alloc_shared((1,), "uint64")
        arena = T.alloc_shared((offset + 9216,), "uint8", align=1024)
        a = T.decl_tensor(
            T.ptr_byte_offset(arena.data, offset * 1, ty=T.Ptr("uint8", "shared")),
            ty_args=[T.Tensor((8192,), "uint8", scope="shared")],
        )
        b = T.decl_tensor(
            T.ptr_byte_offset(arena.data, (offset + 8192) * 1, ty=T.Ptr("uint8", "shared")),
            ty_args=[T.Tensor((1024,), "uint8", scope="shared")],
        )
        tmem = T.tmem((128, 32), "uint32", allocated_addr=0, layout=_TMEM)
        da: T.uint64
        db: T.uint64
        di: T.uint32
        for i in T.serial(256):
            a[lane + i * 32] = T.uint8(0x38)  # E4M3 1.0
        for i in T.serial(32):
            b[lane + i * 32] = T.uint8(0x38)
        if T.constexpr(a_in_tmem):
            for rg in T.unroll(4):
                for col in T.unroll(8):
                    tmem[rg * 32 + lane, 16 + col] = T.uint32(0x38383838)
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.ptx.tcgen05.fence__after_thread_sync()
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(barrier[0]), 1)
        T.ptx.fence.proxy.async_.shared__cta()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tcgen05.encode_instr_descriptor(
                T.address_of(di),
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
                T.address_of(da),
                T.address_of(arena[0]),
                ldo=0,
                sdo=64,
                swizzle=3,
            )
            # Supply the SM107 15-bit start field explicitly; do not let a
            # legacy encoder mask off the address bit under test.
            db = (da & T.bitwise_not(T.uint64(0x7FFF))) | T.cast(
                T.shift_right(T.cuda.cvta_generic_to_shared(b.ptr_to([0])), T.uint32(4)), "uint64"
            )
            da = (da & T.bitwise_not(T.uint64(0x7FFF))) | T.cast(
                T.shift_right(T.cuda.cvta_generic_to_shared(a.ptr_to([0])), T.uint32(4)), "uint64"
            )
            if T.constexpr(a_in_tmem):
                T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                    T.uint32(0),
                    T.uint32(16),
                    db,
                    di,
                    0,
                    0,
                    0,
                    0,
                    T.ptx.pred(0),
                )
            else:
                T.ptx["tcgen05.mma.cta_group::1.kind::f8f6f4"](
                    T.uint32(0),
                    da,
                    db,
                    di,
                    0,
                    0,
                    0,
                    0,
                    T.ptx.pred(0),
                )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                T.address_of(barrier[0])
            )
        T.cuda.mbarrier_wait(T.address_of(barrier[0]), 0)
        for rg in T.unroll(2):
            row = rg * 32 + lane
            for col in T.unroll(8):
                output[row, col] = tmem[(row // 16) * 32 + row % 16, col]

    return kernel


@pytest.mark.parametrize("a_in_tmem", [False, True], ids=["ss", "ts"])
def test_fp8_cta1_extended_shared_addresses(tmp_path, a_in_tmem):
    for offset in (0, 1 << 18):
        kernel = with_cuda_arch(_kernel(offset, a_in_tmem), "sm_107a")
        arguments = {"output": np.zeros((64, 8), dtype=np.uint32)}
        module = numsim.transpile(kernel, cache_dir=tmp_path)
        result = numsim.Engine(max_workers=1).run(module, arguments)
        np.testing.assert_array_equal(result.outputs["output"].view(np.float32), 32.0)
        for checker in (synccheck, racecheck):
            checker(kernel, arguments).require_clean()
    # The wider field belongs to SM107, not to FP8 or K=32 generally.
    with pytest.raises(UnsupportedTIRxError, match="exceeds the 18-bit descriptor address space"):
        numsim.transpile(with_cuda_arch(kernel, "sm_100a"), cache_dir=tmp_path)
