"""A TCGEN load/reduction over independently initialized TMEM columns."""

from tvm.script import tirx as T


def ld_red_kernel(dtype="f32", maximum=True, split=False, absolute=False, nan=False, wait_st=True):
    scalar = {"f32": "float32", "s32": "int32", "u32": "uint32"}[dtype]
    shape = "16x32bx2" if split else "32x32b"
    operation = "max" if maximum else "min"
    modifiers = (".abs" if absolute else "") + (".NaN" if nan else "")
    instruction = f"tcgen05.ld.red.sync.aligned.{shape}.x4.{operation}{modifiers}.{dtype}"

    @T.prim_func
    def kernel(source: T.Buffer((32, 8), scalar), output: T.Buffer((32, 5), scalar)):
        T.device_entry()
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        address = T.alloc_buffer((1,), "uint32", scope="shared")
        r = T.alloc_local((4,), scalar)
        reduced = T.alloc_local((1,), scalar)
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(address[0]), 32)
        for half in T.unroll(2):
            for i in T.unroll(4):
                r[i] = source[lane, half * 4 + i]
            T.ptx["tcgen05.st.sync.aligned.32x32b.x4.b32"](
                address[0] + T.uint32(half * 16), r[0], r[1], r[2], r[3]
            )
        if wait_st:
            T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cta_sync()
        if split:
            T.ptx[instruction](r[0], r[1], r[2], r[3], reduced[0], address[0], 16)
        else:
            T.ptx[instruction](r[0], r[1], r[2], r[3], reduced[0], address[0])
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for i in T.unroll(4):
            output[lane, i] = r[i]
        output[lane, 4] = reduced[0]
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], 32)
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()

    return kernel.with_attr("tirx.cuda_arch", "sm_103a")
