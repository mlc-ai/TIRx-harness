"""A one-warp one-shot multimem all-reduce, the NVLS pattern of CUTLASS's
``MulticastSystemBarrier`` and FlashInfer's multimem barrier: publish through
the unicast replica, arrive with ``multimem.red.release.sys`` on a multicast
flag, wait on the unicast flag, then ``multimem.ld_reduce`` the sum.

Scalar modes select the variants so one transpile serves every case:
``fence_before``/``fence_after`` place ``fence.proxy.alias`` around the
barrier, ``barrier=0`` drops it, ``sys_scope=0`` uses ``.gpu`` scope across
ranks, and ``plain_mc=1`` publishes with a plain store to the multicast
address.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tirx_harness.numsim import MulticastWindow

ELEMENTS = 128


@T.prim_func
def one_shot_all_reduce(
    fence_before: T.int32,
    barrier: T.int32,
    fence_after: T.int32,
    sys_scope: T.int32,
    plain_mc: T.int32,
    world: T.uint32,
    src: T.Buffer((128,), "float32"),
    data: T.Buffer((128,), "float32"),
    mc: T.Buffer((128,), "float32"),
    flag: T.Buffer((1,), "uint32"),
    flag_mc: T.Buffer((1,), "uint32"),
    out: T.Buffer((128,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    for i in range(4):
        if plain_mc == 1:
            mc[lane * 4 + i] = src[lane * 4 + i]
        else:
            data[lane * 4 + i] = src[lane * 4 + i]
    if fence_before == 1:
        T.ptx.fence.proxy.alias()
    T.cuda.warp_sync()
    if barrier == 1:
        if lane == 0:
            seen = T.alloc_local((1,), "uint32")
            if sys_scope == 1:
                T.ptx.multimem_red.release.sys.global_.add.u32(flag_mc.ptr_to([0]), T.uint32(1))
                T.cuda.wait_until(seen[0], flag.ptr_to([0]), lambda c: c >= world, scope="sys")
            else:
                T.ptx.multimem_red.release.gpu.global_.add.u32(flag_mc.ptr_to([0]), T.uint32(1))
                T.cuda.wait_until(seen[0], flag.ptr_to([0]), lambda c: c >= world, scope="gpu")
        T.cuda.warp_sync()
    if fence_after == 1:
        T.ptx.fence.proxy.alias()
    T.ptx.multimem_ld_reduce.relaxed.sys.global_.add.v4.f32(
        out[lane * 4], out[lane * 4 + 1], out[lane * 4 + 2], out[lane * 4 + 3],
        mc.ptr_to([lane * 4]),
    )


VARIANTS = {
    "clean": {},
    "fence_after_barrier_only": {"fence_before": 0},
    "fence_before_barrier_only": {"fence_after": 0},
    "no_alias_fence": {"fence_before": 0, "fence_after": 0},
    "no_barrier": {"barrier": 0},
    "gpu_scope_across_ranks": {"sys_scope": 0},
    "plain_multicast_store": {"plain_mc": 1},
}


def rank_sources(world: int) -> list[np.ndarray]:
    return [np.arange(ELEMENTS, dtype=np.float32) * (rank + 1) for rank in range(world)]


def rank_inputs(world: int, **modes: int) -> list[dict]:
    """Per-rank bindings; each call builds fresh replicas and windows."""

    scalars = {"fence_before": 1, "barrier": 1, "fence_after": 1, "sys_scope": 1, "plain_mc": 0}
    scalars.update(modes)
    data = [np.zeros(ELEMENTS, np.float32) for _ in range(world)]
    flags = [np.zeros(1, np.uint32) for _ in range(world)]
    window, flag_window = MulticastWindow(data), MulticastWindow(flags)
    return [
        {
            **{name: np.int32(value) for name, value in scalars.items()},
            "world": np.uint32(world),
            "src": source,
            "data": data[rank],
            "mc": window,
            "flag": flags[rank],
            "flag_mc": flag_window,
            "out": np.zeros(ELEMENTS, np.float32),
        }
        for rank, source in enumerate(rank_sources(world))
    ]
