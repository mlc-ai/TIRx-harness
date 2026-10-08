"""What a declared synchronization word is worth, shape by shape.

Each kernel here is one shape a cross-thread flag protocol takes, written the
way a kernel writes it. What the checker says about each is the primitive's
contract made concrete: the wait states the condition its protocol completes
on, and the protocol's edge comes from the earliest write that condition
accepts.

Two shapes stay on raw PTX because a declared word does not cover them, and
that is deliberate rather than pending:

* `publish_via_shared_cluster` polls a word in `shared::cluster`. A declared
  word is global, because a protocol that waits within a CTA or a cluster
  belongs on an `mbarrier` -- the hardware's own primitive, which the checker
  models by generation. It used to report a shared-memory morally-strong false
  positive, which this branch did not fix and the shared shadow's scoped path
  since did; it is kept here as the control that a cluster poll stays outside
  the declared word's reach without being reported for it.
* `float_atomic_add_order` accumulates with `atom.add.f32`. A declared word
  carries integer types; a float accumulator is a value, not a protocol.

What the primitive does not answer for
--------------------------------------
`poll_with_non_unique_exit`, `concurrent_writes_then_read` and
`integer_atomic_ticket` consume a value whose identity depends on which write
landed first -- the condition was written wrongly, or not written at all.
`wait(pred)` is the author saying "my protocol completes here"; the checker
does not go back and audit whether the protocol was designed well, the same
line `StrictMbarrierProtocol` draws when it checks a barrier's use rather than
its design. A value that moves with the schedule is a separate policy's to
report.
"""

import numpy as np
import pytest
from tvm.ir import PointerType, PrimType
from tvm.script import tirx as T

from tirx_harness import numsim

GEN = 1
WORD = (GEN << 32) | 0xDEADBEEF
WORD_A = (1 << 32) | 0xAAAA
WORD_B = (2 << 32) | 0xBBBB
PAYLOAD = 0x1234


@T.prim_func
def poll_until_generation(slot: T.Buffer((1,), "uint64"), sink: T.Buffer((1,), "uint64")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(0))
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait.acquire()
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(WORD))
        else:
            observed = T.alloc_local((1,), "uint64")
            observed[0] = T.uint64(0)
            T.cuda.wait_until(
                observed[0], slot.ptr_to([0]),
                T.Cast("uint32", T.shift_right(observed[0], 32)) == T.uint32(GEN), "cluster", "global")
            sink[0] = T.bitwise_and(observed[0], T.uint64(0xFFFFFFFF))


@T.prim_func
def publish_via_shared_cluster(sink: T.Buffer((1,), "uint64")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    slot = T.alloc_buffer((1,), "uint64", scope="shared")
    mapped = T.alloc_local((1,), "uint64")
    T.ptx.mapa.u64(mapped[0], slot.ptr_to([0]), T.uint32(0))
    remote_ptr: T.let[
        T.Var(name="dsm_slot", ty=PointerType(PrimType("uint64"), "shared"))
    ] = T.reinterpret(PointerType(PrimType("uint64"), "shared"), mapped[0])
    remote = T.decl_buffer((1,), "uint64", scope="shared", data=remote_ptr)
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.shared__cluster.u64(remote.ptr_to([0]), T.uint64(0))
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait.acquire()
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.shared__cluster.u64(remote.ptr_to([0]), T.uint64(WORD))
        else:
            observed = T.alloc_local((1,), "uint64")
            observed[0] = T.uint64(0)
            while T.Cast("uint32", T.shift_right(observed[0], 32)) != T.uint32(GEN):
                T.ptx.ld.relaxed.cluster.shared__cluster.u64(observed[0], remote.ptr_to([0]))
            sink[0] = T.bitwise_and(observed[0], T.uint64(0xFFFFFFFF))


@T.prim_func
def read_once(slot: T.Buffer((1,), "uint64"), sink: T.Buffer((1,), "uint64")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(0))
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait.acquire()
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(WORD))
        else:
            observed = T.alloc_local((1,), "uint64")
            T.ptx.ld.relaxed.cluster.global_.u64(observed[0], slot.ptr_to([0]))
            sink[0] = T.bitwise_and(observed[0], T.uint64(0xFFFFFFFF))


@T.prim_func
def poll_with_non_unique_exit(slot: T.Buffer((1,), "uint64"), sink: T.Buffer((1,), "uint64")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(0))
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait.acquire()
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(WORD_A))
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(WORD_B))
        else:
            observed = T.alloc_local((1,), "uint64")
            observed[0] = T.uint64(0)
            T.cuda.wait_until(
                observed[0], slot.ptr_to([0]),
                T.Cast("uint32", T.shift_right(observed[0], 32)) >= T.uint32(1), "cluster", "global")
            sink[0] = T.bitwise_and(observed[0], T.uint64(0xFFFFFFFF))


@T.prim_func
def concurrent_writes_then_read(slot: T.Buffer((1,), "uint64"), sink: T.Buffer((1,), "uint64")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(WORD_A))
        else:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(WORD_B))
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait.acquire()
    if lane == 0:
        if cta == 0:
            observed = T.alloc_local((1,), "uint64")
            T.ptx.ld.relaxed.cluster.global_.u64(observed[0], slot.ptr_to([0]))
            sink[0] = observed[0]


@T.prim_func
def float_atomic_add_order(acc: T.Buffer((1,), "float32"), sink: T.Buffer((1,), "float32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old = T.alloc_local((1,), "float32")
        if cta == 0:
            T.ptx.atom.relaxed.cluster.global_.add.f32(old[0], acc.ptr_to([0]), T.float32(1.0))
        else:
            T.ptx.atom.relaxed.cluster.global_.add.f32(
                old[0], acc.ptr_to([0]), T.float32(-100000000.0)
            )
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait.acquire()
    if lane == 0:
        if cta == 0:
            sink[0] = acc[0]


@T.prim_func
def integer_atomic_add_order(acc: T.Buffer((1,), "uint32"), sink: T.Buffer((1,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old = T.alloc_local((1,), "uint32")
        if cta == 0:
            T.ptx.atom.relaxed.cluster.global_.add.u32(old[0], acc.ptr_to([0]), T.uint32(1))
        else:
            T.ptx.atom.relaxed.cluster.global_.add.u32(old[0], acc.ptr_to([0]), T.uint32(2))
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait.acquire()
    if lane == 0:
        if cta == 0:
            sink[0] = acc[0]


@T.prim_func
def integer_atomic_ticket(acc: T.Buffer((1,), "uint32"), sink: T.Buffer((2,), "uint32")):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        old = T.alloc_local((1,), "uint32")
        T.ptx.atom.relaxed.cluster.global_.add.u32(old[0], acc.ptr_to([0]), T.uint32(1))
        sink[cta] = old[0]


@T.prim_func
def poll_then_read_separate_payload(
    slot: T.Buffer((1,), "uint64"),
    data: T.Buffer((1,), "uint64"),
    sink: T.Buffer((1,), "uint64"),
):
    T.device_entry()
    _cluster = T.cluster_id([1])
    cta = T.cta_id_in_cluster([2])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        if cta == 0:
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(0))
    T.ptx.barrier.cluster.arrive.release()
    T.ptx.barrier.cluster.wait.acquire()
    if lane == 0:
        if cta == 0:
            data[0] = T.uint64(PAYLOAD)
            T.ptx.st.relaxed.cluster.global_.u64(slot.ptr_to([0]), T.uint64(WORD))
        else:
            observed = T.alloc_local((1,), "uint64")
            observed[0] = T.uint64(0)
            T.cuda.wait_until(
                observed[0], slot.ptr_to([0]),
                T.Cast("uint32", T.shift_right(observed[0], 32)) == T.uint32(GEN), "cluster", "global")
            sink[0] = data[0]

def _u64(n=1):
    return np.zeros(n, np.uint64)


# (name, kernel, buffers, verdict, finding kinds)
CASES = [
    # --- the protocol working -------------------------------------------
    # Polls until the generation matches and consumes the payload riding in
    # the same word, so the edge comes from the write that satisfied the wait.
    ("payload_in_the_polled_word", poll_until_generation,
     {"slot": (_u64, ()), "sink": (_u64, ())}, "clean", ()),
    ("atomic_add_modification_order", integer_atomic_add_order,
     {"acc": (np.zeros, (1, np.uint32)), "sink": (np.zeros, (1, np.uint32))}, "clean", ()),

    # --- the protocol broken, and caught ---------------------------------
    # Reads once and uses whatever it got: a single declared load names no
    # write it is entitled to, and the declaration does not launder that.
    ("read_once_no_retry", read_once,
     {"slot": (_u64, ()), "sink": (_u64, ())}, "error", ("write_read",)),
    # A relaxed wait orders nothing but its own word, so the separate payload
    # is unordered however the poll went.
    ("payload_in_a_separate_buffer", poll_then_read_separate_payload,
     {"slot": (_u64, ()), "data": (_u64, ()), "sink": (_u64, ())}, "error", ("write_read",)),

    # --- outside what a declared wait answers for ------------------------
    # See the module docstring: the condition, not the checker, decides these.
    ("non_unique_exit_value", poll_with_non_unique_exit,
     {"slot": (_u64, ()), "sink": (_u64, ())}, "clean", ()),
    ("concurrent_writes_then_read", concurrent_writes_then_read,
     {"slot": (_u64, ()), "sink": (_u64, ())}, "clean", ()),
    ("atomic_ticket_order", integer_atomic_ticket,
     {"acc": (np.zeros, (1, np.uint32)), "sink": (np.zeros, (2, np.uint32))}, "clean", ()),

    # --- shapes a declared word does not cover ---------------------------
    ("raw_shared_cluster_poll", publish_via_shared_cluster,
     {"sink": (_u64, ())}, "clean", ()),
    ("raw_float_accumulator", float_atomic_add_order,
     {"acc": (np.full, (1, 1e8, np.float32)), "sink": (np.zeros, (1, np.float32))}, "clean", ()),
]




@pytest.mark.parametrize(
    ("label", "func", "buffers", "verdict", "kinds"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_declared_word_shape_gets_the_verdict_it_earns(tmp_path, label, func, buffers, verdict, kinds):
    module = numsim.transpile(
        func,
        cache_dir=tmp_path,
        _default_generated_opt_level=0,
        _analysis_capable=True,
        _analysis_checker="racecheck",
    )
    inputs = {name: factory(*args) for name, (factory, args) in buffers.items()}
    result = numsim.Engine(max_workers=2).run_racecheck_phase(module, inputs)

    assert result.verdict == verdict
    assert sorted({finding["access_pair"] for finding in result.findings or []}) == sorted(kinds)
