"""Scalar and memory operations in non-warp-aligned CUDA launches."""

from tvm.script import tirx as T


def partial_cta(threads, *, synchronize=False):
    capacity = ((threads + 31) // 32) * 32

    @T.prim_func
    def kernel(output: T.Buffer((2, capacity, 3), "int32")):
        T.device_entry()
        block = T.cta_id([2])
        thread = T.thread_id([threads])
        output[block, thread, 0] = thread
        if thread % 2 == 0:
            output[block, thread, 0] = thread + 100
        output[block, thread, 1] = T.cuda.mov_sreg(32, "ntid.x")
        output[block, thread, 2] = T.cuda.mov_sreg(32, "tid.x")
        if synchronize:
            T.cuda.cta_sync()

    return kernel
