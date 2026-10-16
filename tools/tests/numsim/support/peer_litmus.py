"""Unicast peer-memory litmus kernels for the PTX memory consistency model
across ranks.

Each kernel runs one warp per rank and takes scalar modes, so one transpile
serves a rule's passing form and every counterexample to it. Section numbers
refer to the PTX ISA's "Memory Consistency Model" chapter.

The shared buffers are :class:`~tirx_harness.numsim.SymmetricBuffer` replicas,
and a rank reaches rank ``p``'s replica at its own replica's address plus
``offsets[p]``, as an NVSHMEM or CUDA symmetric-memory kernel does. With
``symmetric=False`` the replicas are each rank's private memory instead, which
no peer may address.

``push`` is MP through peer stores: every lane stores a 16-byte vector into
rank ``rank + 1``'s ``inbox``, lane 0 arrives on that rank's ``flag``
(``ARRIVALS``) and waits on its own (``WAITS``), and every lane reads its own
inbox.

``pull`` is MP through peer loads: every rank publishes its ``outbox`` and
releases its own ``flag``; lane 0 polls rank ``rank + 1``'s flag and every lane
loads that rank's outbox. ``reuse`` then overwrites the outbox, after the
puller's acknowledgement (``ACKS``) or not.

``atomics`` has every rank update rank 0's ``counter`` words with no
synchronization, so only moral strength (8.7) separates a race from none.

``bulk`` moves the 512-byte vectors with the async proxy: ``direction`` 0
pulls rank ``rank + 1``'s outbox into shared memory with a bulk or TMA load,
direction 1 pushes shared memory into rank ``rank + 1``'s inbox with a bulk or
TMA store. ``tmap`` describes the peer's replica for ``copy = 1``.
"""

from __future__ import annotations

import numpy as np
from tvm.script import tirx as T

from tirx_harness.numsim import SymmetricBuffer, TensorMap

LANES = 32
ELEMENTS = 4 * LANES
BYTES = 4 * ELEMENTS
MAX_WORLD = 4


def _peer(pointer, offset):
    """``pointer`` moved into another rank's replica of its buffer."""
    return T.reinterpret("handle", T.reinterpret("uint64", pointer) + T.Cast("uint64", offset))


def _heap(world: int, symmetric: bool, *shapes: tuple[str, int, int]) -> tuple[list, ...]:
    """Rank ``r``'s ``offsets`` and per-rank bindings of several buffers carved
    from one heap per rank, so one offset per peer reaches every buffer, as on
    an NVSHMEM symmetric heap. ``shapes`` holds (dtype, elements, fill) per
    buffer."""
    sizes = [np.dtype(dtype).itemsize * count for dtype, count, _ in shapes]
    starts = np.cumsum([0, *[-(-size // 128) * 128 for size in sizes]])
    heaps = [np.zeros(int(starts[-1]) + 128, np.uint8) for _ in range(world)]
    base = [(-heap.ctypes.data) % 128 for heap in heaps]
    views = []
    for (dtype, count, fill), start in zip(shapes, starts):
        arrays = []
        for rank in range(world):
            array = heaps[rank][base[rank] + start:][: np.dtype(dtype).itemsize * count].view(dtype)
            array[:] = fill
            arrays.append(array)
        views.append(arrays)
    offsets = []
    for rank in range(world):
        row = np.zeros(MAX_WORLD, np.int64)
        row[:world] = [views[0][p].ctypes.data - views[0][rank].ctypes.data for p in range(world)]
        offsets.append(row)
    bound = [[SymmetricBuffer(arrays)] * world if symmetric else arrays for arrays in views]
    return (offsets, *bound)


# Lane 0's arrival on the receiver's ``flag``.
ARRIVALS = {
    "release_sys": 0,          # red.release.sys.add [peer flag]
    "release_gpu": 1,          # red.release.gpu.add
    "fence_sys_relaxed": 2,    # fence.acq_rel.sys; red.relaxed.sys.add
    "fence_gpu_relaxed": 3,    # fence.acq_rel.gpu; red.relaxed.sys.add
    "relaxed": 4,              # red.relaxed.sys.add
    "store_release_sys": 5,    # st.release.sys [peer flag], 1
    "weak_store": 6,           # st [peer flag], 1
}
# Lane 0's wait on its own ``flag`` until it is nonzero. Hand-written spins
# poll with ``atom.cas``, as the multimem litmus and CUTLASS's and FlashInfer's
# barriers do: a strong RMW poll is morally strong with the arrival it may read
# too early (8.7). Racecheck adjudicates a plain ``ld`` poll by happens-before
# instead, so its failed polls race the arrival on the schedules where they
# run first; `T.cuda.wait_until` declares the spin and is exempt.
WAITS = {
    "acquire_sys": 0,          # atom.acquire.sys.cas spin
    "acquire_gpu": 1,          # atom.acquire.gpu.cas spin
    "relaxed_fence_sys": 2,    # atom.relaxed.sys.cas spin; fence.acq_rel.sys
    "relaxed_fence_gpu": 3,    # atom.relaxed.sys.cas spin; fence.acq_rel.gpu
    "relaxed": 4,              # atom.relaxed.sys.cas spin
    "wait_until_sys": 5,       # T.cuda.wait_until(scope="sys")
    "wait_until_gpu": 6,       # T.cuda.wait_until(scope="gpu")
    "none": 7,
}


@T.prim_func
def push(
    rank: T.int32,
    world: T.int32,
    arrive: T.int32,
    wait: T.int32,
    sync_before: T.int32,
    sync_after: T.int32,
    late_store: T.int32,
    early_read: T.int32,
    flag_shift: T.int32,
    silent_rank: T.int32,
    offsets: T.Buffer((MAX_WORLD,), "int64"),
    src: T.Buffer((ELEMENTS,), "uint32"),
    inbox: T.Buffer((ELEMENTS,), "uint32"),
    flag: T.Buffer((4,), "uint32"),
    out: T.Buffer((ELEMENTS,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    seen = T.local_scalar("uint32")
    to = (rank + 1) % world
    signal = (rank + flag_shift) % world
    if late_store == 0 or lane != 0:
        T.ptx.st.global_.v4.b32(
            _peer(inbox.ptr_to([lane * 4]), offsets[to]), src[lane * 4], src[lane * 4 + 1],
            src[lane * 4 + 2], src[lane * 4 + 3])
    if sync_before == 1:
        T.cuda.warp_sync()
    if lane == 0:
        target = _peer(flag.ptr_to([0]), offsets[signal])
        if arrive == 2:
            T.ptx.fence.acq_rel.sys()
        elif arrive == 3:
            T.ptx.fence.acq_rel.gpu()
        if late_store == 1:
            T.ptx.st.global_.v4.b32(
                _peer(inbox.ptr_to([0]), offsets[to]), src[0], src[1], src[2], src[3])
        if rank == silent_rank:
            pass
        elif arrive == 0:
            T.ptx.red.release.sys.global_.add.u32(target, T.uint32(1))
        elif arrive == 1:
            T.ptx.red.release.gpu.global_.add.u32(target, T.uint32(1))
        elif arrive == 5:
            T.ptx.st.release.sys.global_.u32(target, T.uint32(1))
        elif arrive == 6:
            T.ptx.st.global_.u32(target, T.uint32(1))
        else:
            T.ptx.red.relaxed.sys.global_.add.u32(target, T.uint32(1))

        if wait == 5:
            T.cuda.wait_until(seen, flag.ptr_to([0]), lambda c: c != 0, scope="sys")
        elif wait == 6:
            T.cuda.wait_until(seen, flag.ptr_to([0]), lambda c: c != 0, scope="gpu")
        elif wait != 7:
            seen = T.uint32(0)
            while seen == T.uint32(0):
                if wait == 0:
                    T.ptx.atom.acquire.sys.global_.cas.b32(
                        seen, flag.ptr_to([0]), T.uint32(1), T.uint32(1))
                elif wait == 1:
                    T.ptx.atom.acquire.gpu.global_.cas.b32(
                        seen, flag.ptr_to([0]), T.uint32(1), T.uint32(1))
                else:
                    T.ptx.atom.relaxed.sys.global_.cas.b32(
                        seen, flag.ptr_to([0]), T.uint32(1), T.uint32(1))
        if early_read == 1:
            T.ptx.ld.global_.v4.b32(out[0], out[1], out[2], out[3], inbox.ptr_to([0]))
        if wait == 2:
            T.ptx.fence.acq_rel.sys()
        elif wait == 3:
            T.ptx.fence.acq_rel.gpu()
    if sync_after == 1:
        T.cuda.warp_sync()
    if early_read == 0 or lane != 0:
        T.ptx.ld.global_.v4.b32(
            out[lane * 4], out[lane * 4 + 1], out[lane * 4 + 2], out[lane * 4 + 3],
            inbox.ptr_to([lane * 4]))


PUSH_DEFAULTS = {
    "arrive": ARRIVALS["release_sys"], "wait": WAITS["acquire_sys"], "sync_before": 1,
    "sync_after": 1, "late_store": 0, "early_read": 0, "flag_shift": 1, "silent_rank": -1,
}


def _modes(defaults: dict, tables: dict, modes: dict) -> dict:
    scalars = {**defaults, **modes}
    for name, table in tables.items():
        if isinstance(scalars[name], str):
            scalars[name] = table[scalars[name]]
    return {name: np.int32(value) for name, value in scalars.items()}


def sources(world: int) -> list[np.ndarray]:
    """Rank ``r``'s 16-byte vectors."""
    return [np.arange(ELEMENTS, dtype=np.uint32) * (rank + 1) + rank for rank in range(world)]


def push_inputs(world: int, *, symmetric: bool = True, **modes) -> list[dict]:
    """Per-rank bindings; ``modes`` override `PUSH_DEFAULTS` and may name
    `ARRIVALS` and `WAITS` entries by key."""
    scalars = _modes(PUSH_DEFAULTS, {"arrive": ARRIVALS, "wait": WAITS}, modes)
    offsets, inbox, flag = _heap(world, symmetric, ("uint32", ELEMENTS, 0), ("uint32", 4, 0))
    src = sources(world)
    return [
        {"rank": np.int32(rank), "world": np.int32(world), **scalars, "offsets": offsets[rank],
         "src": src[rank], "inbox": inbox[rank], "flag": flag[rank],
         "out": np.zeros(ELEMENTS, np.uint32)}
        for rank in range(world)
    ]


# The publisher's release of its own ``flag``.
PUBLISHES = {
    "release_sys": 0,          # st.release.sys [flag], 1
    "release_gpu": 1,          # st.release.gpu [flag], 1
    "relaxed": 2,              # st.relaxed.sys [flag], 1
}
# The puller's poll of the publisher's ``flag``.
POLLS = {
    "acquire_sys": 0,          # atom.acquire.sys.cas [peer flag] spin
    "acquire_gpu": 1,          # atom.acquire.gpu.cas [peer flag] spin
    "relaxed": 2,              # atom.relaxed.sys.cas [peer flag] spin
}
# How the puller tells the publisher its outbox may be overwritten.
ACKS = {
    "release_acquire_sys": 0,  # red.release.sys [peer ack]; atom.acquire.sys.cas [ack] spin
    "relaxed": 1,              # red.relaxed.sys [peer ack]; atom.relaxed.sys.cas [ack] spin
    "none": 2,                 # no acknowledgement and no wait
}


@T.prim_func
def pull(
    rank: T.int32,
    world: T.int32,
    publish: T.int32,
    poll: T.int32,
    reuse: T.int32,
    ack: T.int32,
    offsets: T.Buffer((MAX_WORLD,), "int64"),
    src: T.Buffer((ELEMENTS,), "uint32"),
    outbox: T.Buffer((ELEMENTS,), "uint32"),
    flag: T.Buffer((4,), "uint32"),
    ack_flag: T.Buffer((4,), "uint32"),
    out: T.Buffer((ELEMENTS,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    seen = T.local_scalar("uint32")
    peer = (rank + 1) % world
    T.ptx.st.global_.v4.b32(
        outbox.ptr_to([lane * 4]), src[lane * 4], src[lane * 4 + 1], src[lane * 4 + 2],
        src[lane * 4 + 3])
    T.cuda.warp_sync()
    if lane == 0:
        if publish == 0:
            T.ptx.st.release.sys.global_.u32(flag.ptr_to([0]), T.uint32(1))
        elif publish == 1:
            T.ptx.st.release.gpu.global_.u32(flag.ptr_to([0]), T.uint32(1))
        else:
            T.ptx.st.relaxed.sys.global_.u32(flag.ptr_to([0]), T.uint32(1))
        seen = T.uint32(0)
        published = _peer(flag.ptr_to([0]), offsets[peer])
        while seen == T.uint32(0):
            if poll == 0:
                T.ptx.atom.acquire.sys.global_.cas.b32(seen, published, T.uint32(1), T.uint32(1))
            elif poll == 1:
                T.ptx.atom.acquire.gpu.global_.cas.b32(seen, published, T.uint32(1), T.uint32(1))
            else:
                T.ptx.atom.relaxed.sys.global_.cas.b32(seen, published, T.uint32(1), T.uint32(1))
    T.cuda.warp_sync()
    T.ptx.ld.global_.v4.b32(
        out[lane * 4], out[lane * 4 + 1], out[lane * 4 + 2], out[lane * 4 + 3],
        _peer(outbox.ptr_to([lane * 4]), offsets[peer]))
    if reuse == 1:
        T.cuda.warp_sync()
        if lane == 0:
            if ack == 0:
                T.ptx.red.release.sys.global_.add.u32(
                    _peer(ack_flag.ptr_to([0]), offsets[peer]), T.uint32(1))
            elif ack == 1:
                T.ptx.red.relaxed.sys.global_.add.u32(
                    _peer(ack_flag.ptr_to([0]), offsets[peer]), T.uint32(1))
            if ack != 2:
                seen = T.uint32(0)
                while seen == T.uint32(0):
                    if ack == 0:
                        T.ptx.atom.acquire.sys.global_.cas.b32(
                            seen, ack_flag.ptr_to([0]), T.uint32(1), T.uint32(1))
                    else:
                        T.ptx.atom.relaxed.sys.global_.cas.b32(
                            seen, ack_flag.ptr_to([0]), T.uint32(1), T.uint32(1))
        T.cuda.warp_sync()
        T.ptx.st.global_.v4.b32(
            outbox.ptr_to([lane * 4]), T.uint32(0), T.uint32(0), T.uint32(0), T.uint32(0))


PULL_DEFAULTS = {"publish": 0, "poll": 0, "reuse": 0, "ack": 0}


def pull_inputs(world: int, *, symmetric: bool = True, **modes) -> list[dict]:
    scalars = _modes(PULL_DEFAULTS, {"publish": PUBLISHES, "poll": POLLS, "ack": ACKS}, modes)
    offsets, outbox, flag, ack_flag = _heap(
        world, symmetric, ("uint32", ELEMENTS, 0), ("uint32", 4, 0), ("uint32", 4, 0))
    src = sources(world)
    return [
        {"rank": np.int32(rank), "world": np.int32(world), **scalars, "offsets": offsets[rank],
         "src": src[rank], "outbox": outbox[rank], "flag": flag[rank],
         "ack_flag": ack_flag[rank], "out": np.zeros(ELEMENTS, np.uint32)}
        for rank in range(world)
    ]


# Every rank's update of word ``lane`` of rank 0's ``counter``, unsynchronized.
UPDATES = {
    "atom_add_sys": 0,         # atom.relaxed.sys.add
    "atom_add_gpu": 1,         # atom.relaxed.gpu.add
    "red_add_sys": 2,          # red.relaxed.sys.add
    "red_add_gpu": 3,          # red.relaxed.gpu.add
    "st_relaxed_sys": 4,       # st.relaxed.sys
    "st_weak": 5,              # st
    # atom.relaxed.sys.add on peers; rank 0 reads its own counter with ld.
    "atom_add_sys_weak_read": 6,
    "atom_cas_sys": 7,         # atom.relaxed.sys.cas
}


@T.prim_func
def atomics(
    rank: T.int32,
    op: T.int32,
    offsets: T.Buffer((MAX_WORLD,), "int64"),
    counter: T.Buffer((LANES,), "uint32"),
    old: T.Buffer((LANES,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    observed = T.local_scalar("uint32")
    target = _peer(counter.ptr_to([lane]), offsets[0])
    value = T.uint32(rank + 1)
    observed = T.uint32(0)
    if op == 0 or (op == 6 and rank != 0):
        T.ptx.atom.relaxed.sys.global_.add.u32(observed, target, value)
    elif op == 1:
        T.ptx.atom.relaxed.gpu.global_.add.u32(observed, target, value)
    elif op == 2:
        T.ptx.red.relaxed.sys.global_.add.u32(target, value)
    elif op == 3:
        T.ptx.red.relaxed.gpu.global_.add.u32(target, value)
    elif op == 4:
        T.ptx.st.relaxed.sys.global_.u32(target, value)
    elif op == 5:
        T.ptx.st.global_.u32(target, value)
    elif op == 6:
        T.ptx.ld.global_.u32(observed, counter.ptr_to([lane]))
    elif op == 7:
        T.ptx.atom.relaxed.sys.global_.cas.b32(observed, target, T.uint32(0), value)
    old[lane] = observed


def atomics_inputs(world: int, op: str, *, symmetric: bool = True) -> list[dict]:
    offsets, counter = _heap(world, symmetric, ("uint32", LANES, 0))
    return [
        {"rank": np.int32(rank), "op": np.int32(UPDATES[op]), "offsets": offsets[rank],
         "counter": counter[rank], "old": np.zeros(LANES, np.uint32)}
        for rank in range(world)
    ]


# Where the consumer of a bulk/TMA load, or the producer of a bulk/TMA store,
# executes fence.proxy.async.
PROXY_FENCES = {
    "none": 0,
    "after_wait": 1,           # fence.proxy.async.global after the acquire
    "before_wait": 2,          # fence.proxy.async.global before the acquire: off the path
    "after_wait_all": 3,       # fence.proxy.async (every state space) after the acquire
    "after_wait_shared": 4,    # fence.proxy.async.shared::cta: not the global space
    "producer": 5,             # fence.proxy.async.global by the publisher, before its release
}
# How a bulk/TMA store waits before its release.
STORE_WAITS = {
    "complete": 0,             # cp.async.bulk.wait_group 0
    "read": 1,                 # cp.async.bulk.wait_group.read 0: sources read, writes pending
    "none": 2,
}


@T.prim_func
def bulk(
    rank: T.int32,
    world: T.int32,
    direction: T.int32,
    copy: T.int32,
    proxy_fence: T.int32,
    store_wait: T.int32,
    offsets: T.Buffer((MAX_WORLD,), "int64"),
    tmap: T.TensorMap(),
    src: T.Buffer((ELEMENTS,), "uint32"),
    outbox: T.Buffer((ELEMENTS,), "uint32"),
    inbox: T.Buffer((ELEMENTS,), "uint32"),
    flag: T.Buffer((4,), "uint32"),
    out: T.Buffer((ELEMENTS,), "uint32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    stage = T.alloc_buffer((ELEMENTS,), "uint32", scope="shared", align=128)
    bar = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
    seen = T.local_scalar("uint32")
    peer = (rank + 1) % world
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(bar.ptr_to([0]), 1)
        T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.warp_sync()
    if direction == 0:
        T.ptx.st.global_.v4.b32(
            outbox.ptr_to([lane * 4]), src[lane * 4], src[lane * 4 + 1], src[lane * 4 + 2],
            src[lane * 4 + 3])
        if proxy_fence == 5:
            T.ptx.fence.proxy.async_.global_()
        T.cuda.warp_sync()
        if lane == 0:
            T.ptx.st.release.sys.global_.u32(flag.ptr_to([0]), T.uint32(1))
            if proxy_fence == 2:
                T.ptx.fence.proxy.async_.global_()
            seen = T.uint32(0)
            while seen == T.uint32(0):
                T.ptx.atom.acquire.sys.global_.cas.b32(
                    seen, _peer(flag.ptr_to([0]), offsets[peer]), T.uint32(1), T.uint32(1))
            if proxy_fence == 1:
                T.ptx.fence.proxy.async_.global_()
            elif proxy_fence == 3:
                T.ptx.fence.proxy.async_()
            elif proxy_fence == 4:
                T.ptx.fence.proxy.async_.shared__cta()
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(bar.ptr_to([0]), BYTES)
            if copy == 0:
                T.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
                    stage.ptr_to([0]), _peer(outbox.ptr_to([0]), offsets[peer]),
                    T.uint32(BYTES), bar.ptr_to([0]))
            else:
                T.evaluate(T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    ".cta_group::1"
                ](stage.ptr_to([0]), T.address_of(tmap), 0, 0, bar.ptr_to([0])))
        T.cuda.mbarrier_wait(bar.ptr_to([0]), 0)
        T.ptx.ld.shared.v4.u32(
            out[lane * 4], out[lane * 4 + 1], out[lane * 4 + 2], out[lane * 4 + 3],
            stage.ptr_to([lane * 4]))
    else:
        T.ptx.st.shared.v4.u32(
            stage.ptr_to([lane * 4]), src[lane * 4], src[lane * 4 + 1], src[lane * 4 + 2],
            src[lane * 4 + 3])
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.warp_sync()
        if lane == 0:
            if copy == 0:
                T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
                    _peer(inbox.ptr_to([0]), offsets[peer]), stage.ptr_to([0]), T.uint32(BYTES))
            else:
                T.evaluate(T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                    T.address_of(tmap), 0, 0, stage.ptr_to([0])))
            T.ptx.cp.async_.bulk.commit_group()
            if store_wait == 0:
                T.ptx.cp.async_.bulk.wait_group(0)
            elif store_wait == 1:
                T.ptx.cp.async_.bulk.wait_group.read(0)
            if proxy_fence == 5:
                T.ptx.fence.proxy.async_.global_()
            T.ptx.red.release.sys.global_.add.u32(
                _peer(flag.ptr_to([0]), offsets[peer]), T.uint32(1))
            seen = T.uint32(0)
            while seen == T.uint32(0):
                T.ptx.atom.acquire.sys.global_.cas.b32(
                    seen, flag.ptr_to([0]), T.uint32(1), T.uint32(1))
            if proxy_fence == 1:
                T.ptx.fence.proxy.async_.global_()
            T.ptx.cp.async_.bulk.wait_group(0)
        T.cuda.warp_sync()
        T.ptx.ld.global_.v4.b32(
            out[lane * 4], out[lane * 4 + 1], out[lane * 4 + 2], out[lane * 4 + 3],
            inbox.ptr_to([lane * 4]))


BULK_DEFAULTS = {"direction": 0, "copy": 0, "proxy_fence": 1, "store_wait": 0}


def bulk_inputs(world: int, *, symmetric: bool = True, tmap_symmetric: bool | None = None,
                **modes) -> list[dict]:
    """Per-rank bindings. Rank ``r``'s ``tmap`` describes rank ``r + 1``'s
    replica of the buffer ``direction`` moves: the outbox it loads, or the
    inbox it stores. ``tmap_symmetric=False`` points it at a private array of
    rank ``r + 1`` instead, which that rank binds as its ``out``. With
    ``symmetric=False`` alone it describes rank ``r``'s own ``out``, so only a
    bulk copy's peer address reaches another rank's private memory."""
    scalars = _modes(BULK_DEFAULTS, {"proxy_fence": PROXY_FENCES, "store_wait": STORE_WAITS},
                     modes)
    offsets, outbox, inbox, flag = _heap(
        world, symmetric, ("uint32", ELEMENTS, 0), ("uint32", ELEMENTS, 0), ("uint32", 4, 0))
    replicas = outbox if scalars["direction"] == 0 else inbox
    replicas = [r.replicas[i] if isinstance(r, SymmetricBuffer) else r
                for i, r in enumerate(replicas)]
    outs = [np.zeros(ELEMENTS, np.uint32) for _ in range(world)]
    if tmap_symmetric is False:
        targets = [outs[(rank + 1) % world] for rank in range(world)]
    elif symmetric:
        targets = [replicas[(rank + 1) % world] for rank in range(world)]
    else:
        targets = outs
    src = sources(world)
    return [
        {"rank": np.int32(rank), "world": np.int32(world), **scalars, "offsets": offsets[rank],
         "tmap": TensorMap(base=targets[rank], global_shape=(4, LANES),
                           global_strides=(16,), box_shape=(4, LANES), element_strides=(1, 1),
                           dtype="uint32").numpy(),
         "src": src[rank], "outbox": outbox[rank], "inbox": inbox[rank], "flag": flag[rank],
         "out": outs[rank]}
        for rank in range(world)
    ]
