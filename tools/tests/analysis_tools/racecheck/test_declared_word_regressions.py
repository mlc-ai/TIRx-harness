"""Positive controls for the declared word's ordering rules.

Each test here fails if one specific mechanism is removed, so a later change
that quietly stops performing a check is caught by a red test rather than by a
verdict that merely looks plausible. The module docstring of each test names
the mechanism it guards.
"""

from __future__ import annotations

import threading

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness.numsim.checkers import _run_racecheck as racecheck


def _flag_inputs():
    return {
        "flag": np.zeros(1, dtype=np.int32),
        "data": np.zeros(1, dtype=np.int32),
        "out": np.zeros(1, dtype=np.int32),
    }


def _handoff(*, writer_fence: str, reader_fence: str, publish: str):
    """One publisher, one waiter, and a payload on a second address.

    The payload is what makes the verdict mean something: it is ordered only
    if the wait is worth an acquire, so `out = data[0]` is the question the
    checker has to answer.
    """
    return tvm.script.from_source(
        f"""
@T.prim_func
def handoff(flag: T.Buffer((1,), "int32"), data: T.Buffer((1,), "int32"),
            out: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    seen = T.local_scalar("int32")
    if lane == 0:
        if cta == 0:
            data[0] = T.int32(41)
            {writer_fence}
            T.ptx.st.{publish}.gpu.global_.s32(flag.ptr_to([0]), T.int32(1))
        else:
            seen = T.int32(0)
            T.cuda.wait_until(
                seen, flag.ptr_to([0]), seen != T.int32(0),
                "gpu", "global")
            {reader_fence}
            out[0] = data[0]
""",
        {"T": T},
    )


def _kinds(report):
    native = report.to_dict().get("native", {})
    return sorted({finding["kind"] for finding in (native.get("findings") or [])})


def test_a_relaxed_publication_leaves_the_payload_unordered():
    """The control for the two below: the wait acquires, the store must release.

    The wait's half of the pair is settled -- it has one lowering and it always
    acquires -- so the only way to be left without an edge is a publication
    that releases nothing. The payload on the second address is then unordered
    and has to be reported, otherwise the tests below would pass for the wrong
    reason.
    """
    report = racecheck(
        _handoff(writer_fence="", reader_fence="", publish="relaxed"),
        inputs=_flag_inputs(),
        max_workers=2,
    )
    assert report.verdict == "error"
    assert _kinds(report) == ["data_race"]
    assert {f.details["access_pair"] for f in report.findings} == {"write_read"}


def test_a_declared_wait_takes_the_edge_against_a_releasing_store():
    """The pair with no fences on either side: `st.release` and the wait.

    This is what the wait is worth on its own, and what every wait in the
    kernel corpus now rests on.
    """
    report = racecheck(
        _handoff(writer_fence="", reader_fence="", publish="release"),
        inputs=_flag_inputs(),
        max_workers=2,
    )
    report.require_clean()


def test_a_relaxed_store_composes_into_a_release_with_a_preceding_fence():
    """Guards PTX ISA 8.7's release pattern on the half that still has a choice.

    A `release` fence followed in program order by a strong write is a release
    pattern, and `.relaxed` is strong (ISA 8.4.2). The reader's half no longer
    varies, so this is where the composition is still worth a positive control:
    if the wait stops taking the edge from the write its predicate accepted,
    this kernel goes back to reporting `write_read` on the payload.
    """
    report = racecheck(
        _handoff(
            writer_fence="T.cuda.thread_fence()",
            reader_fence="",
            publish="relaxed",
        ),
        inputs=_flag_inputs(),
        max_workers=2,
    )
    report.require_clean()


def test_a_declared_wait_merges_its_tcgen_frontier_without_deadlocking():
    """Guards against re-taking a lock the effect dispatch already holds.

    A declared wait carries the specialized `tcgen05.fence` execution-ordering
    frontier to the waiting actor, and it does that while the shard's race lock
    is held for the whole effect match -- so the merge has to go through that
    guard rather than acquire the lock again. When it acquired it again the
    checker self-deadlocked: no error and no verdict, just a run that never
    returned.

    The shape matters. Only a kernel whose handoff sits between two
    `tcgen05.fence` halves reaches that merge at all, so a plain flag handoff
    would pass whether the bug is present or not. A wall-clock bound is the
    only assertion that catches a hang.

    This covers half of the mechanism and knows it: the bound says the merge
    returned, not that the frontier arrived. Deleting the merge outright leaves
    this test green, and
    `test_native_tcgen_thread_fence.py::test_cross_thread_cp_to_mma_requires_both_thread_fences`
    is what fails then -- its `relaxed_flag` case is clean only if the declared
    wait carried the frontier across. The two together are the control; neither
    alone is.
    """
    from tests.analysis_tools.racecheck import test_native_tcgen_thread_fence as tcgen

    finished = threading.Event()
    box: dict[str, object] = {}

    def run():
        try:
            box["report"] = tcgen._run(
                with_before=True,
                with_after=True,
                handoff=1,  # the declared relaxed flag between the fence halves
                cache_dir=None,
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            box["error"] = exc
        finally:
            finished.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert finished.wait(timeout=180), (
        "a declared wait did not return within 180s; the checker is most "
        "likely deadlocked re-acquiring its own shard lock"
    )
    if "error" in box:
        raise box["error"]  # type: ignore[misc]


def test_raw_protocol_reports_only_the_proven_payload_race():
    """A relaxed flag read does not acquire the payload's publication.

    Report that proven data race; missing wait syntax is not a separate defect.
    """
    kernel = tvm.script.from_source(
        """
@T.prim_func
def read_once(flag: T.Buffer((1,), "int32"), data: T.Buffer((1,), "int32"),
              out: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    seen = T.local_scalar("int32")
    if lane == 0:
        if cta == 0:
            data[0] = T.int32(41)
            T.ptx.st.release.gpu.global_.s32(flag.ptr_to([0]), T.int32(1))
        else:
            seen = T.int32(0)
            T.ptx.ld.relaxed.gpu.global_.s32(seen, flag.ptr_to([0]))
            out[0] = data[0]
""",
        {"T": T},
    )
    report = racecheck(kernel, inputs=_flag_inputs(), max_workers=2)
    assert report.verdict == "error"
    kinds = set(_kinds(report))
    assert kinds, "reading a protocol word once must be reported on every schedule"
    assert kinds == {"data_race"}
    assert {f.details["access_pair"] for f in report.findings} == {"write_read"}
    assert "undeclared_protocol_words" not in report.to_dict()["native"]


def test_a_phase_flip_predicate_takes_the_edge_of_the_arrival_that_flipped_it():
    """A barrier whose predicate is not monotone, as `mega_moe` writes one.

    The word is a counter and the condition is that its sense bit *flipped*
    against the value this actor's own arrival returned -- true for one
    generation and false again for the next. The edge still has to come from
    the arrival that flipped it, so the payload that arrival published is
    ordered; a wait that took its edge from some other write in the word's
    history would order the wrong thing.
    """

    kernel = tvm.script.from_source(
        """
@T.prim_func
def phase_flip(state: T.Buffer((1,), "uint32"), data: T.Buffer((1,), "int32"),
               out: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    old = T.local_scalar("uint32")
    cur = T.local_scalar("uint32")
    if lane == 0:
        if cta == 0:
            data[0] = T.int32(41)
            T.ptx.atom.release.gpu.global_.add.u32(
                old, state.ptr_to([0]), T.uint32(2147483647))
        else:
            T.ptx.atom.release.gpu.global_.add.u32(old, state.ptr_to([0]), T.uint32(1))
            cur = old
            T.cuda.wait_until(
                cur, state.ptr_to([0]),
                T.bitwise_and(T.bitwise_xor(cur, old), T.uint32(2147483648)) != T.uint32(0), "gpu", "global")
            out[0] = data[0]
""",
        {"T": T},
    )
    inputs = {
        "state": np.zeros(1, dtype=np.uint32),
        "data": np.zeros(1, dtype=np.int32),
        "out": np.zeros(1, dtype=np.int32),
    }
    racecheck(kernel, inputs=inputs, max_workers=2).require_clean()


def test_two_waits_on_one_word_never_conflict_with_each_other():
    """Two readers of the same word are two reads, and reads do not conflict.

    Both wait on the counter the publisher releases, and both read the payload
    behind it. The pair that matters is publisher/reader; reader/reader is not
    a pair at all, and a rule that adjudicated it would report this correct
    protocol.
    """

    kernel = tvm.script.from_source(
        """
@T.prim_func
def two_waiters(state: T.Buffer((1,), "int32"), data: T.Buffer((1,), "int32"),
                out: T.Buffer((2,), "int32")):
    T.device_entry()
    cta = T.cta_id([3])
    lane = T.lane_id([32])
    seen = T.local_scalar("int32")
    if lane == 0:
        if cta == 0:
            data[0] = T.int32(41)
            T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))
        else:
            seen = T.int32(0)
            T.cuda.wait_until(
                seen, state.ptr_to([0]), seen >= T.int32(1), "gpu", "global")
            out[cta - 1] = data[0]
""",
        {"T": T},
    )
    inputs = {
        "state": np.zeros(1, dtype=np.int32),
        "data": np.zeros(1, dtype=np.int32),
        "out": np.zeros(2, dtype=np.int32),
    }
    racecheck(kernel, inputs=inputs, max_workers=2).require_clean()


def test_a_morally_strong_pair_is_exempt_unless_one_side_is_a_plain_read():
    """Guards the split PTX ISA 8.7's own definition asks for.

    A data race needs the pair to be *not* morally strong, so a publication and
    a contribution on one naturally aligned word -- neither of them a plain
    read -- is exempt however the two are ordered. Narrow that to
    read-modify-writes and a release store against the contributions joining
    its release sequence starts reporting `write_write`.
    """
    kernel = tvm.script.from_source(
        """
@T.prim_func
def publish_and_contribute(flag: T.Buffer((1,), "int32"),
                           out: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    old = T.local_scalar("int32")
    if lane == 0:
        if cta == 0:
            T.ptx.st.release.gpu.global_.s32(flag.ptr_to([0]), T.int32(1))
        else:
            T.ptx.atom.release.gpu.global_.add.s32(old, flag.ptr_to([0]), T.int32(1))
            out[0] = old
""",
        {"T": T},
    )
    inputs = {"flag": np.zeros(1, dtype=np.int32), "out": np.zeros(1, dtype=np.int32)}
    racecheck(kernel, inputs=inputs, max_workers=2).require_clean()


def _wide_publication(*, width):
    """A publication either narrow enough for a wait to poll, or too wide.

    `wait_until` polls 4 or 8 bytes, so the word's write history is kept
    for those widths only. A `.v4.b32` release store lands on the same word and
    publishes the same value, but nothing a wait can name -- the history would
    be missing the write that released it, and the protocol would look
    unpublished.
    """
    publish = (
        "T.ptx.st.release.gpu.global_.v4.b32("
        "state.ptr_to([0]), T.uint32(7), T.uint32(0), T.uint32(0), T.uint32(0))"
        if width == 16
        else "T.ptx.st.release.gpu.global_.b32(state.ptr_to([0]), T.uint32(7))"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def wide(state: T.Buffer((4,), "uint32"), observed: T.Buffer((1,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "uint32")
    if lane == 0:
        if warp == 0:
            {publish}
        else:
            seen[0] = T.uint32(0)
            T.cuda.wait_until(
                seen[0], state.ptr_to([0]), seen[0] != T.uint32(0), "gpu", "global", "b32")
            observed[0] = seen[0]
""",
        {"T": T},
    )


def _wide_inputs():
    return {
        "state": np.zeros(4, dtype=np.uint32),
        "observed": np.zeros(1, dtype=np.uint32),
    }


def _incomplete_kinds(report):
    native = report.to_dict().get("native", {})
    return sorted({reason.get("kind") for reason in (native.get("incomplete") or [])})


def test_a_publication_too_wide_to_poll_fails_closed():
    """The guard on the declared word's post-image read-back.

    The read-back covers 4- and 8-byte writes, the widths a wait polls. A wider
    strong write is skipped rather than read back -- there is no `u64` to take
    from bytes no wait can name -- but skipping it silently would leave the
    word's history missing a publication, and a published protocol would read
    as unpublished. So it reports instead of being dropped.

    Remove `DeclaredWordWriteUnrecorded` and this goes back to `incomplete` for
    the wrong reason alone: the wait would still be unexplained, with nothing
    saying which write the history lost.
    """
    report = racecheck(_wide_publication(width=16), _wide_inputs())
    assert report.verdict == "incomplete", report.format()
    assert "analysis_incomplete" in _incomplete_kinds(report)
    assert any(f.details.get("reason") == "signal_write_not_recorded" for f in report.findings)


def test_a_publication_a_wait_can_poll_is_clean():
    """The control: the same protocol, published at a width the wait polls.

    Identical in every other respect, so the reason above is caused by the
    publication's width and not by the protocol's shape.
    """
    report = racecheck(_wide_publication(width=4), _wide_inputs())
    assert report.verdict == "clean", report.format()
    assert _incomplete_kinds(report) == []


def _scoped_handoff(*, publish_scope):
    """A cross-CTA handoff whose publication names a scope, waited at `sys`.

    The waiter asks for the widest scope, so what decides the verdict is
    whether the publication's own scope reaches the waiting CTA.
    """
    return tvm.script.from_source(
        f"""
@T.prim_func
def scoped(flag: T.Buffer((1,), "int32"), data: T.Buffer((1,), "int32"),
           out: T.Buffer((1,), "int32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if cta == 0:
            data[0] = T.int32(5)
            T.ptx.st.release.{publish_scope}.global_.s32(flag.ptr_to([0]), T.int32(1))
        else:
            seen[0] = 0
            T.cuda.wait_until(
                seen[0], flag.ptr_to([0]), seen[0] != 0,
                "sys", "global")
            out[0] = data[0]
""",
        {"T": T},
    )


def test_a_publication_whose_scope_reaches_the_waiter_is_clean():
    """`gpu` covers another CTA of the same GPU."""
    report = racecheck(_scoped_handoff(publish_scope="gpu"), _flag_inputs())
    assert report.verdict == "clean", report.format()


def test_a_publication_whose_scope_stops_short_of_the_waiter_is_reported():
    """`cta` does not reach the other CTA, so the release is not one *here*.

    The control for the pair above: the only difference is the publication's
    scope, so a clean verdict there is the scope reaching the waiter and not
    the shape of the protocol.
    """
    report = racecheck(_scoped_handoff(publish_scope="cta"), _flag_inputs())
    assert report.verdict == "error", report.format()
    assert "scope_mismatch" in _kinds(report)


def _proxy_publication(*, proxy):
    """A payload published in one proxy, consumed behind a wait in the generic one.

    The wait's edge is an ordinary acquire: it orders what the *generic* proxy
    ordered. A payload the async proxy wrote still needs its own bridge, and
    the wait neither supplies one nor loses one -- which is the contract, since
    the raw spelling of the same loop reports the payload identically.
    """
    if proxy == "async":
        publish_payload = """
            for index in T.serial(4):
                shared[index] = seed[index]
            T.ptx.fence.proxy.async_.shared__cta()
            T.ptx["cp.async.bulk.global.shared::cta.bulk_group"](
                data.ptr_to([0]), shared.ptr_to([0]), T.cast(16, "uint32"))
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group.read(0)
            T.ptx.fence.proxy.async_.global_()"""
    else:
        publish_payload = """
            for index in T.serial(4):
                data[index] = seed[index]"""
    return tvm.script.from_source(
        f"""
@T.prim_func
def proxy_publish(seed: T.Buffer((4,), "uint32"), data: T.Buffer((4,), "uint32"),
                  flag: T.Buffer((1,), "int32"), out: T.Buffer((1,), "uint32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    shared = T.alloc_buffer((4,), "uint32", scope="shared")
    if lane == 0:
        if cta == 0:{publish_payload}
            T.ptx.st.release.gpu.global_.s32(flag.ptr_to([0]), T.int32(1))
        else:
            seen[0] = 0
            T.cuda.wait_until(
                seen[0], flag.ptr_to([0]), seen[0] != 0, "gpu", "global")
            out[0] = data[0]
""",
        {"T": T},
    )


def _proxy_inputs():
    return {
        "seed": np.arange(4, dtype=np.uint32),
        "data": np.zeros(4, dtype=np.uint32),
        "flag": np.zeros(1, dtype=np.int32),
        "out": np.zeros(1, dtype=np.uint32),
    }


def test_a_wait_does_not_bridge_the_async_proxy():
    """The wait's acquire is a generic-proxy edge and stays one.

    A bulk store puts the payload in global through the async proxy; the flag
    is published and waited on in the generic proxy. The flag handoff is sound,
    and the payload is still unordered, because no async->generic bridge covers
    it. A wait that silently bridged proxies would call this clean.
    """
    report = racecheck(_proxy_publication(proxy="async"), _proxy_inputs())
    assert report.verdict == "error", report.format()
    findings = report.to_dict()["native"]["findings"]
    assert any(
        finding["ordering_failure"] == "missing_proxy_bridge" for finding in findings
    ), report.format()


def test_a_generic_payload_behind_the_same_wait_is_clean():
    """The control: the identical protocol with the payload written generically.

    Same flag, same wait, same consumer -- so the finding above is the proxy
    the payload crossed and not the handoff the wait performs.
    """
    report = racecheck(_proxy_publication(proxy="generic"), _proxy_inputs())
    assert report.verdict == "clean", report.format()


def _ring_credit(*, lap: int):
    """A ring slot's credit wait, in the shape `sm100_fp8_fp4_mega_moe` writes it.

    A consumer may use slot `k` on lap `L` once the slot has been drained
    `L * per_lap` times, and the producers count drains into the slot's
    counter. The code does not special-case the first lap; it lets the
    threshold come out as zero, because a slot nobody has used yet is free by
    construction.

    That makes lap 0 a wait that never waits, and its predicate accepts the
    value the launch supplied. Lap 1 is the same code with a threshold nobody
    has reached yet, so it is a wait that genuinely blocks on a publication.
    """

    return tvm.script.from_source(
        f"""
@T.prim_func
def ring_credit(credit: T.Buffer((1,), "uint32"), out: T.Buffer((1,), "uint32")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    seen = T.local_scalar("uint32")
    if lane == 0:
        if cta == 0:
            T.ptx.red.release.gpu.global_.add.u32(credit.ptr_to([0]), T.uint32(1))
        else:
            seen = T.uint32(0)
            T.cuda.wait_until(
                seen, credit.ptr_to([0]), seen >= T.uint32({lap}), "gpu", "global")
            out[0] = seen
""",
        {"T": T},
    )


def _ring_inputs():
    return {
        "credit": np.zeros(1, dtype=np.uint32),
        "out": np.zeros(1, dtype=np.uint32),
    }


def test_a_first_lap_ring_credit_wait_is_explained():
    """The first lap's threshold is zero, so the wait leaves on the launch value.

    This is the shape that made `sm100_fp8_fp4_mega_moe` report 64
    unexplained-wait entries (now `analysis_incomplete` / `wait_exit_unproven`): a ring-credit wait whose
    threshold is `per_lap * lap`, evaluated on lap 0. The loop reads the
    counter once, the predicate already holds, and nothing on the device had to
    publish for it to leave.

    It is not a defect and it is not unadjudicated: the launch value is earlier
    than every write and the host's stores precede every actor, so the exit
    carries no edge anyone still owes. Reporting it would make every ring
    buffer that does not special-case its first lap permanently incomplete.
    """

    report = racecheck(_ring_credit(lap=0), _ring_inputs())
    assert report.verdict == "clean", report.format()
    assert _incomplete_kinds(report) == []


def test_a_later_lap_ring_credit_wait_still_needs_its_publication():
    """The twin: the same wait with a threshold the launch value misses.

    Lap 1 asks for a credit the launch did not supply, so the wait is released
    by the producer's `red.release` and the checker names it. Without this the
    test above would pass for a checker that had simply stopped adjudicating
    ring credits at all.
    """

    report = racecheck(_ring_credit(lap=1), _ring_inputs())
    assert report.verdict == "clean", report.format()
    assert _incomplete_kinds(report) == []
    assert _kinds(report) == []


def test_a_host_initialized_value_explains_the_wait():
    """A wait the launch value already satisfies needed nobody to publish.

    Nothing on the device writes the word: the value the predicate accepts is
    the one the launch started with. There is no release to name because none
    was owed -- the host's stores precede every actor -- so the verdict is
    `clean`.

    An empty history has a second cause that must keep being reported: a write
    that stayed on the compact fast path and never reached the recorder. The
    predicate separates the two. Here it accepts the launch value, so the loop
    could have left on its first read under any schedule. A waiter released by
    an unrecorded write is the opposite case -- its predicate *rejects* the
    launch value, which is precisely why it had to wait -- and it is still
    reported, which
    `test_a_wait_woken_by_a_plain_write_reports_the_missing_edge` in
    `tests/numsim/runtime/test_wait_until.py` holds in place. Neither verdict
    is ever an `error`: the checker does not claim such a kernel is wrong.

    Judging by the launch value follows the rule the accepted position already
    follows -- take the earliest exit the predicate admits, so the conclusion
    holds for every schedule rather than the one that ran. The launch value is
    earlier than every write.
    """
    kernel = tvm.script.from_source(
        """
@T.prim_func
def host_init(flag: T.Buffer((1,), "int32"), observed: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        seen[0] = 0
        T.cuda.wait_until(
            seen[0], flag.ptr_to([0]), seen[0] != 0, "gpu", "global")
        observed[0] = seen[0]
""",
        {"T": T},
    )
    inputs = {
        "flag": np.array([7], dtype=np.int32),
        "observed": np.zeros(1, dtype=np.int32),
    }
    report = racecheck(kernel, inputs)
    assert report.verdict == "clean", report.format()
    assert _incomplete_kinds(report) == []
    assert _kinds(report) == []
