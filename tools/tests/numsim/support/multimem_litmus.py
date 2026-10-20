"""NVLS litmus kernels for the PTX memory consistency model across ranks.

Each kernel runs one warp per rank and takes scalar modes, so one transpile
serves a rule's passing form and every counterexample to it. Section numbers
refer to the PTX ISA's "Memory Consistency Model" chapter.

``message_passing`` is MP over NVLS: every lane stores a 16-byte vector of the
rank's unicast replica, lane 0 arrives on a multicast flag and waits on the
rank's unicast flag, and every lane ``multimem.ld_reduce``-s its vector. The
modes pick the release pattern (``ARRIVALS``), the acquire pattern (``WAITS``),
where ``fence.proxy.alias`` sits (``ALIAS``) and whether ``bar.warp.sync``
carries the other lanes' accesses to and from lane 0; ``silent_rank`` never
arrives.

``concurrent_writes`` has every rank write the same multicast words with no
synchronization, so only moral strength (8.7) separates a race from none.

``broadcast`` reverses the proxies: ranks publish through ``multimem.st`` and
read their unicast replica, either after the barrier (RAW) or before it (WAR).
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tirx_harness.numsim import MulticastWindow

LANES = 32
ELEMENTS = 4 * LANES

# Lane 0's arrival on ``flag_mc[arrive_slot]``.
ARRIVALS = {
    "release_sys": 0,          # multimem.red.release.sys
    "release_gpu": 1,          # multimem.red.release.gpu
    "fence_sys_relaxed": 2,    # fence.acq_rel.sys; multimem.red.relaxed.sys
    "fence_gpu_relaxed": 3,    # fence.acq_rel.gpu; multimem.red.relaxed.sys
    "relaxed": 4,              # multimem.red.relaxed.sys
    "fence_cta_relaxed": 5,    # fence.acq_rel.cta; multimem.red.relaxed.sys
    # multimem.st.release.sys of 1: every rank overwrites the replicas instead
    # of adding to them, so the flag never counts past 1.
    "store_release_sys": 6,
}
# Lane 0's wait on ``flag[0]`` until it reaches ``world``. Hand-written spins
# poll with ``atom.cas`` as CUTLASS and FlashInfer do: a strong RMW poll is
# morally strong with the arrivals it reads (8.7), so only the pattern the spin
# forms decides the verdict.
WAITS = {
    "acquire_sys": 0,          # atom.acquire.sys.cas spin
    "acquire_gpu": 1,          # atom.acquire.gpu.cas spin
    "relaxed_fence_sys": 2,    # atom.relaxed.sys.cas spin; fence.acq_rel.sys
    "relaxed_fence_gpu": 3,    # atom.relaxed.sys.cas spin; fence.acq_rel.gpu
    "relaxed": 4,              # atom.relaxed.sys.cas spin
    "weak_fence_sys": 5,       # ld.weak spin; fence.acq_rel.sys
    "weak_red_fence_sys": 6,   # ld.weak spin; red.relaxed.sys [flag]; fence.acq_rel.sys
    "wait_until_sys": 7,       # T.cuda.wait_until(scope="sys")
    "wait_until_gpu": 8,       # T.cuda.wait_until(scope="gpu")
    "none": 9,
}
# Where every lane executes fence.proxy.alias.
ALIAS = {
    "none": 0,
    "after_stores": 1,         # producer side, on the path
    "after_wait": 2,           # consumer side, on the path
    "before_stores": 3,        # off the path: precedes the writes
    "after_reduce": 4,         # off the path: follows the reads
}


@T.prim_func
def message_passing(
    rank: T.int32,
    arrive: T.int32,
    gpu_rank: T.int32,
    wait: T.int32,
    alias: T.int32,
    sync_before: T.int32,
    sync_after: T.int32,
    late_store: T.int32,
    early_read: T.int32,
    arrive_slot: T.int32,
    silent_rank: T.int32,
    world: T.uint32,
    src: T.Buffer((ELEMENTS,), "float32"),
    data: T.Buffer((ELEMENTS,), "float32"),
    mc: T.Buffer((ELEMENTS,), "float32"),
    flag: T.Buffer((2,), "uint32"),
    flag_mc: T.Buffer((2,), "uint32"),
    out: T.Buffer((ELEMENTS,), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    seen = T.local_scalar("uint32")
    if alias == 3:
        T.ptx.fence.proxy.alias()
    if late_store == 0 or lane != 0:
        for i in range(4):
            data[lane * 4 + i] = src[lane * 4 + i]
    if alias == 1:
        T.ptx.fence.proxy.alias()
    if sync_before == 1:
        T.cuda.warp_sync()
    if lane == 0:
        mode = T.Select(rank == gpu_rank, 1, arrive)
        if mode == 2:
            T.ptx.fence.acq_rel.sys()
        elif mode == 3:
            T.ptx.fence.acq_rel.gpu()
        elif mode == 5:
            T.ptx.fence.acq_rel.cta()
        if late_store == 1:
            for i in range(4):
                data[i] = src[i]
        if rank == silent_rank:
            pass
        elif mode == 0:
            T.ptx.multimem_red.release.sys.global_.add.u32(
                flag_mc.ptr_to([arrive_slot]), T.uint32(1))
        elif mode == 1:
            T.ptx.multimem_red.release.gpu.global_.add.u32(
                flag_mc.ptr_to([arrive_slot]), T.uint32(1))
        elif mode == 6:
            T.ptx.multimem_st.release.sys.global_.b32(
                flag_mc.ptr_to([arrive_slot]), T.uint32(1))
        else:
            T.ptx.multimem_red.relaxed.sys.global_.add.u32(
                flag_mc.ptr_to([arrive_slot]), T.uint32(1))

        if wait == 7:
            T.cuda.wait_until(seen, flag.ptr_to([0]), lambda c: c >= world, scope="sys")
        elif wait == 8:
            T.cuda.wait_until(seen, flag.ptr_to([0]), lambda c: c >= world, scope="gpu")
        elif wait != 9:
            seen = T.uint32(0)
            while seen < world:
                if wait == 0:
                    T.ptx.atom.acquire.sys.global_.cas.b32(seen, flag.ptr_to([0]), world, world)
                elif wait == 1:
                    T.ptx.atom.acquire.gpu.global_.cas.b32(seen, flag.ptr_to([0]), world, world)
                elif wait <= 4:
                    T.ptx.atom.relaxed.sys.global_.cas.b32(seen, flag.ptr_to([0]), world, world)
                else:
                    T.ptx.ld.global_.u32(seen, flag.ptr_to([0]))
        if early_read == 1:
            T.ptx.multimem_ld_reduce.weak.global_.add.v4.f32(
                out[0], out[1], out[2], out[3], mc.ptr_to([0]))
        if wait == 6:
            T.ptx.red.relaxed.sys.global_.add.u32(flag.ptr_to([0]), T.uint32(0))
        if wait == 2 or wait == 5 or wait == 6:
            T.ptx.fence.acq_rel.sys()
        elif wait == 3:
            T.ptx.fence.acq_rel.gpu()
    if sync_after == 1:
        T.cuda.warp_sync()
    if alias == 2:
        T.ptx.fence.proxy.alias()
    if early_read == 0 or lane != 0:
        T.ptx.multimem_ld_reduce.weak.global_.add.v4.f32(
            out[lane * 4], out[lane * 4 + 1], out[lane * 4 + 2], out[lane * 4 + 3],
            mc.ptr_to([lane * 4]),
        )
    if alias == 4:
        T.ptx.fence.proxy.alias()


MP_DEFAULTS = {
    "arrive": ARRIVALS["release_sys"], "gpu_rank": -1, "wait": WAITS["acquire_sys"],
    "alias": ALIAS["after_stores"], "sync_before": 1, "sync_after": 1, "late_store": 0,
    "early_read": 0, "arrive_slot": 0, "silent_rank": -1,
}


def message_passing_inputs(world: int, *, flag_init: int = 0, **modes) -> list[dict]:
    """Per-rank bindings; ``modes`` override `MP_DEFAULTS` and may name
    `ARRIVALS`, `WAITS` and `ALIAS` entries by key."""

    scalars = {**MP_DEFAULTS, **modes}
    for name, table in (("arrive", ARRIVALS), ("wait", WAITS), ("alias", ALIAS)):
        if isinstance(scalars[name], str):
            scalars[name] = table[scalars[name]]
    data = [np.zeros(ELEMENTS, np.float32) for _ in range(world)]
    flags = [np.array([flag_init, 0], np.uint32) for _ in range(world)]
    window, flag_window = MulticastWindow(data), MulticastWindow(flags)
    return [
        {
            "rank": np.int32(rank),
            **{name: np.int32(value) for name, value in scalars.items()},
            "world": np.uint32(world),
            "src": np.arange(ELEMENTS, dtype=np.float32) * (rank + 1),
            "data": data[rank],
            "mc": window,
            "flag": flags[rank],
            "flag_mc": flag_window,
            "out": np.zeros(ELEMENTS, np.float32),
        }
        for rank in range(world)
    ]


# Every rank's write of word `lane` of the multicast window, unsynchronized.
WRITES = {
    "st_weak": 0,              # multimem.st.weak
    "st_relaxed_sys": 1,       # multimem.st.relaxed.sys
    "st_relaxed_gpu": 2,       # multimem.st.relaxed.gpu
    "red_relaxed_sys": 3,      # multimem.red.relaxed.sys
    "red_relaxed_gpu": 4,      # multimem.red.relaxed.gpu
    "st_relaxed_sys_and_unicast": 5,  # multimem.st.relaxed.sys + st.relaxed.sys to the replica
    # Not multimem operations, so undefined on a multimem address (8.2.3).
    "plain_load": 6,           # ld.relaxed.sys [mc]
    "plain_store": 7,          # st [mc]
    "unicast_red": 8,          # red.relaxed.sys [mc]
    # Multimem operations on the replica's unicast address, which is not a
    # multimem address.
    "multimem_st_unicast": 9,  # multimem.st.relaxed.sys [data]
    "multimem_red_unicast": 10,  # multimem.red.relaxed.sys [data]
}


@T.prim_func
def concurrent_writes(
    rank: T.int32,
    op: T.int32,
    data: T.Buffer((LANES,), "uint32"),
    mc: T.Buffer((LANES,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    observed = T.local_scalar("uint32")
    value = T.uint32(rank + 1)
    if op == 0:
        T.ptx.multimem_st.weak.global_.b32(mc.ptr_to([lane]), value)
    elif op == 1 or op == 5:
        T.ptx.multimem_st.relaxed.sys.global_.b32(mc.ptr_to([lane]), value)
    elif op == 2:
        T.ptx.multimem_st.relaxed.gpu.global_.b32(mc.ptr_to([lane]), value)
    elif op == 3:
        T.ptx.multimem_red.relaxed.sys.global_.add.u32(mc.ptr_to([lane]), value)
    elif op == 4:
        T.ptx.multimem_red.relaxed.gpu.global_.add.u32(mc.ptr_to([lane]), value)
    elif op == 6:
        T.ptx.ld.relaxed.sys.global_.u32(observed, mc.ptr_to([lane]))
    elif op == 7:
        mc[lane] = value
    elif op == 8:
        T.ptx.red.relaxed.sys.global_.add.u32(mc.ptr_to([lane]), value)
    elif op == 9:
        T.ptx.multimem_st.relaxed.sys.global_.b32(data.ptr_to([lane]), value)
    elif op == 10:
        T.ptx.multimem_red.relaxed.sys.global_.add.u32(data.ptr_to([lane]), value)
    if op == 5:
        T.ptx.st.relaxed.sys.global_.u32(data.ptr_to([lane]), value)


def concurrent_write_inputs(world: int, op: str) -> list[dict]:
    data = [np.zeros(LANES, np.uint32) for _ in range(world)]
    window = MulticastWindow(data)
    return [
        {"rank": np.int32(rank), "op": np.int32(WRITES[op]), "data": data[rank], "mc": window}
        for rank in range(world)
    ]


@T.prim_func
def broadcast(
    rank: T.int32,
    read_first: T.int32,
    alias: T.int32,
    world: T.uint32,
    src: T.Buffer((ELEMENTS,), "float32"),
    data: T.Buffer((ELEMENTS,), "float32"),
    mc: T.Buffer((ELEMENTS,), "float32"),
    flag: T.Buffer((1,), "uint32"),
    flag_mc: T.Buffer((1,), "uint32"),
    out: T.Buffer((ELEMENTS,), "float32"),
):
    """Lane ``l`` of rank ``l % world`` broadcasts vector ``l``; every lane
    reads vector ``l`` of its unicast replica. ``read_first`` puts the reads
    before the ``.sys`` barrier and the broadcasts after it; ``alias`` places
    ``fence.proxy.alias`` after the barrier."""
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    seen = T.local_scalar("uint32")
    owner = lane % T.Cast("int32", world) == rank
    if read_first == 1:
        for i in range(4):
            out[lane * 4 + i] = data[lane * 4 + i]
    elif owner:
        T.ptx.multimem_st.weak.global_.v4.f32(
            mc.ptr_to([lane * 4]), src[lane * 4], src[lane * 4 + 1], src[lane * 4 + 2],
            src[lane * 4 + 3])
    T.cuda.warp_sync()
    if lane == 0:
        T.ptx.multimem_red.release.sys.global_.add.u32(flag_mc.ptr_to([0]), T.uint32(1))
        T.cuda.wait_until(seen, flag.ptr_to([0]), lambda c: c >= world, scope="sys")
    T.cuda.warp_sync()
    if alias == 1:
        T.ptx.fence.proxy.alias()
    if read_first == 0:
        for i in range(4):
            out[lane * 4 + i] = data[lane * 4 + i]
    elif owner:
        T.ptx.multimem_st.weak.global_.v4.f32(
            mc.ptr_to([lane * 4]), src[lane * 4], src[lane * 4 + 1], src[lane * 4 + 2],
            src[lane * 4 + 3])


def broadcast_inputs(world: int, *, read_first: int, alias: int) -> list[dict]:
    data = [np.full(ELEMENTS, -1, np.float32) for _ in range(world)]
    flags = [np.zeros(1, np.uint32) for _ in range(world)]
    window, flag_window = MulticastWindow(data), MulticastWindow(flags)
    return [
        {
            "rank": np.int32(rank), "read_first": np.int32(read_first),
            "alias": np.int32(alias), "world": np.uint32(world),
            "src": np.arange(ELEMENTS, dtype=np.float32) + 1, "data": data[rank],
            "mc": window, "flag": flags[rank], "flag_mc": flag_window,
            "out": np.zeros(ELEMENTS, np.float32),
        }
        for rank in range(world)
    ]
