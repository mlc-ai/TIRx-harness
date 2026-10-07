"""The declared synchronization word lowers to exactly its raw spelling.

``tirx.cuda.wait_until`` emits the poll loop a kernel writes by hand. These
tests write the same protocol twice — once through the primitive, once through
the raw spelling it stands for — and require the two to agree on the numerical
result and on the checker verdict. Nothing about the declaration may change
what a kernel does; it only lets an analysis tell a protocol's own accesses
from a stray one.
"""

import re

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck
from tests.numsim.support.execution import run_checked
from tests.numsim.support.wait_until import indexed_predicate_case, initial_case


def _memory_variants(rust_source):
    """The memory instructions the engine will execute, in order.

    The wait is dropped from both spellings, because the two express it
    differently on purpose. Written raw it is a loop of loads, and the engine
    has to execute every one of them; written through the primitive it is one
    suspending operation and the engine does the waiting itself. That is the
    whole point of the operation — a spin's reads all race the publisher's
    write by construction, so handing them to the memory model reports every
    correct protocol. The CUDA both spellings lower to is still the same loop,
    instruction for instruction; `tests/python/tirx/codegen/test_cuda_wait_until.py`
    is where that is pinned.

    So: `mem::declared_wait` goes, and so does everything inside a native
    `while` body. What the two spellings have to agree on is everything else.
    """

    executed = re.sub(r"v2::mem::declared_wait::<.*?\);\n", "", rust_source, flags=re.S)
    executed = re.sub(
        r"v2::control::while_enter\(.*?v2::control::while_exit\(.*?;\n",
        "",
        executed,
        flags=re.S,
    )
    variants = []
    for call in re.finditer(r"v2::mem::(?:ld|st|atom|red)::<(.*?)>>\(", executed, flags=re.S):
        spelling = call.group(1)
        # A thread-local scalar is bookkeeping, not one of the protocol's
        # instructions: where the wait leaves its exit value is the caller's
        # business and the two spellings put it in different places.
        if "v2::Local" in spelling:
            continue
        variants.extend(re.findall(r"v2::mem::variant::(\w+)", spelling))
    return variants


def _declared_wait_count(rust_source):
    return rust_source.count("v2::mem::declared_wait::<")


def _verdicts(kernel, inputs):
    """Each checker's verdict and the shape of what it found."""

    summary = {}
    for checker in (synccheck, racecheck):
        report = checker(kernel, inputs())
        summary[report.checker_name] = (
            report.verdict,
            sorted((finding.status, finding.kind) for finding in report.findings),
        )
    return summary


def _rendezvous(*, primitive):
    """A release/acquire rendezvous written in one of the two spellings.

    Warp 0 publishes a payload word and then contributes to the state word;
    warp 1 waits for that contribution and reads the payload behind the
    acquire. One thread alone takes a ticket, so its pre-image is the same
    under every schedule and the kernel's outputs stay exact.
    """

    if primitive:
        publish = (
            'T.ptx.st.release.gpu.global_.s32(slot.ptr_to([0]), T.int32(7))'
        )
        take_ticket = (
            "T.ptx.atom.relaxed.gpu.global_.add.s32("
            'ticket[0], tickets.ptr_to([0]), T.int32(1))'
        )
        signal = 'T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))'
        wait = (
            "T.cuda.wait_until("
            'seen[0], state.ptr_to([0]), seen[0] >= 1, "gpu", "global")'
        )
        read = 'T.ptx.ld.relaxed.gpu.global_.s32(got[0], slot.ptr_to([0]))'
    else:
        publish = 'T.ptx["st.release.gpu.global.s32"](slot.ptr_to([0]), T.int32(7))'
        take_ticket = (
            'T.ptx["atom.relaxed.gpu.global.add.s32"](ticket[0], tickets.ptr_to([0]), T.int32(1))'
        )
        signal = 'T.ptx["red.release.gpu.global.add.s32"](state.ptr_to([0]), T.int32(1))'
        wait = (
            "while seen[0] < 1:\n"
            '                T.ptx["ld.acquire.gpu.global.s32"](seen[0], state.ptr_to([0]))'
        )
        read = 'T.ptx["ld.relaxed.gpu.global.s32"](got[0], slot.ptr_to([0]))'

    return tvm.script.from_source(
        f"""
@T.prim_func
def rendezvous(state: T.Buffer((1,), "int32"), slot: T.Buffer((1,), "int32"),
               tickets: T.Buffer((1,), "int32"), observed: T.Buffer((2,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    got = T.alloc_local((1,), "int32")
    ticket = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            {publish}
            {take_ticket}
            observed[1] = ticket[0]
            {signal}
        else:
            seen[0] = 0
            {wait}
            {read}
            observed[0] = got[0]
""",
        {"T": T},
    )


def _rendezvous_inputs():
    return {
        "state": np.zeros(1, np.int32),
        "slot": np.zeros(1, np.int32),
        "tickets": np.zeros(1, np.int32),
        "observed": np.full(2, -1, np.int32),
    }


# Every width the declared word accepts, with the PTX type each is read as.
_WORD_TYPES = {"int32": "s32", "uint32": "u32", "int64": "s64", "uint64": "u64"}


@pytest.mark.parametrize("dtype", sorted(_WORD_TYPES))
@pytest.mark.parametrize("predicate_op", ["Select", "if_then_else"])
def test_initial_wait_rechecks_conditional_predicate(dtype, predicate_op, tmp_path):
    case = initial_case(dtype, predicate_op=predicate_op)
    result = run_checked(case.kernel, case.args, outputs=case.outputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], case.reference()["out"])


def _packed(*, primitive, dtype):
    """A wait whose payload rides in the polled word itself.

    The value the wait exits on is the whole message: coherence on the one
    word carries it and no other address is read on the wait's strength. The
    wait still acquires, because there is one lowering and it is cheap enough
    that this shape does not pay for a second one.
    """

    suffix = _WORD_TYPES[dtype]
    if primitive:
        publish = (
            f'T.ptx.st.relaxed.gpu.global_.{suffix}(state.ptr_to([0]), T.{dtype}(7))'
        )
        wait = (
            "T.cuda.wait_until("
            'seen[0], state.ptr_to([0]), seen[0] != 0, "gpu", "global")'
        )
    else:
        load = f"ld.acquire.gpu.global.{suffix}"
        publish = f'T.ptx["st.relaxed.gpu.global.{suffix}"](state.ptr_to([0]), T.{dtype}(7))'
        wait = f'while seen[0] == 0:\n                T.ptx["{load}"](seen[0], state.ptr_to([0]))'

    return tvm.script.from_source(
        f"""
@T.prim_func
def packed(state: T.Buffer((1,), "{dtype}"), observed: T.Buffer((1,), "{dtype}")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "{dtype}")
    if lane == 0:
        if warp == 0:
            {publish}
        else:
            seen[0] = 0
            {wait}
            observed[0] = seen[0]
""",
        {"T": T},
    )


def _packed_inputs(dtype):
    return {
        "state": np.zeros(1, dtype),
        "observed": np.zeros(1, dtype),
    }


def test_rendezvous_matches_raw_spelling(tmp_path):
    raw = _rendezvous(primitive=False)
    primitive = _rendezvous(primitive=True)
    raw_module = numsim.transpile(raw, cache_dir=tmp_path / "raw")
    module = numsim.transpile(primitive, cache_dir=tmp_path / "primitive")
    raw_outputs = numsim.Engine().run(raw_module, _rendezvous_inputs()).outputs
    outputs = numsim.Engine().run(module, _rendezvous_inputs()).outputs

    assert _memory_variants(module.rust_source) == _memory_variants(raw_module.rust_source)
    # The wait states its verdict once; the raw spelling states nothing.
    assert _declared_wait_count(module.rust_source) == 1
    assert _declared_wait_count(raw_module.rust_source) == 0
    for name in _rendezvous_inputs():
        np.testing.assert_array_equal(outputs[name], raw_outputs[name])
    # Both spellings are clean, and that is the contract: the raw loop polls
    # with `ld.acquire`, which is a protocol that explains itself -- whichever
    # value a poll got, it took the edge with it -- so nothing is left for the
    # checker to adjudicate and nothing is claimed. What the primitive buys is
    # not a verdict here but the word's write history and the exit condition,
    # which is what the relaxed shapes below turn on.
    assert _verdicts(primitive, _rendezvous_inputs)["racecheck"] == ("clean", [])
    assert _verdicts(raw, _rendezvous_inputs)["racecheck"] == ("clean", [])
    # Warp 1 read the payload behind the acquire; the lone ticket holder saw
    # the initial value. Both are schedule-independent.
    np.testing.assert_array_equal(outputs["observed"], np.array([7, 0], np.int32))


def test_declared_rendezvous_is_clean():
    kernel = _rendezvous(primitive=True)
    for checker in (synccheck, racecheck):
        checker(kernel, _rendezvous_inputs()).require_clean()


@pytest.mark.parametrize("dtype", sorted(_WORD_TYPES))
def test_packed_wait_matches_raw_spelling(dtype, tmp_path):
    raw = _packed(primitive=False, dtype=dtype)
    primitive = _packed(primitive=True, dtype=dtype)
    raw_module = numsim.transpile(raw, cache_dir=tmp_path / f"raw_{dtype}")
    module = numsim.transpile(primitive, cache_dir=tmp_path / f"primitive_{dtype}")
    inputs = lambda: _packed_inputs(dtype)  # noqa: E731
    raw_outputs = numsim.Engine().run(raw_module, inputs()).outputs
    outputs = numsim.Engine().run(module, inputs()).outputs

    assert _memory_variants(module.rust_source) == _memory_variants(raw_module.rust_source)
    for name in inputs():
        np.testing.assert_array_equal(outputs[name], raw_outputs[name])
    # The primitive lets the checker explain the successful exit and its HB
    # edge. The raw loop below instead contains a proven unordered access.
    assert _verdicts(primitive, inputs)["racecheck"] == ("clean", [])
    # Raw, the spin is an ordinary pair: a word one warp writes and another
    # only reads, and every schedule leaves at least one of those reads
    # unordered against the publication, so the answer does not move with the
    # worker count the way a per-poll verdict would. Written through the
    # primitive it disappears: the wait adjudicates no access of its own and
    # its edge decides.
    # This is a data race, not a separate undeclared-protocol advisory.
    assert _verdicts(raw, inputs)["racecheck"] == (
        "error",
        [("error", "data_race")],
    )
    np.testing.assert_array_equal(outputs["observed"], np.array([7], dtype))


def _mixed(*, scoped_reset):
    """One word whose reset is either a scoped access or a plain one.

    A protocol owns its word, and its publisher is spelled in raw PTX, so
    reaching the word without the primitive is what a protocol ordinarily looks
    like. What the word still refuses is a *plain* access that is concurrent
    with the protocol: one carrying neither ordering nor atomicity, which takes
    part in no agreement at all and is the shape that slips past everything
    else -- it is morally strong against nothing and carries no missing edge of
    its own.

    The resetter is a third warp that never waits, so nothing orders it against
    the waiter. A reset by the waiter itself would be ordered by the very edge
    the wait took, and ordered is not a defect -- see
    `test_an_ordered_plain_reset_is_not_reported`.
    """

    reset = (
        "T.ptx.st.relaxed.gpu.global_.s32(state.ptr_to([0]), T.int32(0))"
        if scoped_reset
        else "state[0] = T.int32(0)"
    )
    return tvm.script.from_source(
        f"""
@T.prim_func
def mixed(state: T.Buffer((1,), "int32"), observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            T.ptx.st.release.gpu.global_.s32(state.ptr_to([0]), T.int32(7))
        else:
            if warp == 1:
                seen[0] = 0
                T.cuda.wait_until(
                    seen[0], state.ptr_to([0]), seen[0] != 0, "gpu", "global")
                observed[0] = seen[0]
            else:
                {reset}
""",
        {"T": T},
    )


def _ordered_reset():
    """The reset a real kernel writes: plain, but behind a rendezvous.

    A workspace counter cleared with a plain store between two grid-wide
    barriers is ordinary, correct code -- every wait on the word is ordered
    against the clear. The claim on the address must not turn that into a
    finding, or the checker reports an error on every kernel that reuses its
    workspace.

    The rendezvous here is a second word, published and waited on through the
    primitive, so the order is one the program has and not one the reset's own
    word handed out.
    """

    return tvm.script.from_source(
        """
@T.prim_func
def ordered_reset(
    state: T.Buffer((1,), "int32"),
    gate: T.Buffer((1,), "int32"),
    observed: T.Buffer((1,), "int32"),
):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    ready = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            state[0] = T.int32(0)
            T.ptx.st.release.gpu.global_.s32(gate.ptr_to([0]), T.int32(1))
            T.ptx.st.release.gpu.global_.s32(state.ptr_to([0]), T.int32(7))
        else:
            ready[0] = 0
            T.cuda.wait_until(
                ready[0], gate.ptr_to([0]), ready[0] != 0, "gpu", "global")
            seen[0] = 0
            T.cuda.wait_until(
                seen[0], state.ptr_to([0]), seen[0] != 0, "gpu", "global")
            observed[0] = seen[0]
""",
        {"T": T},
    )


def _bypass_findings(kernel, inputs=None):
    report = racecheck(kernel, inputs if inputs is not None else _packed_inputs("int32"))
    return [finding for finding in report.findings if finding.kind == "signal_protocol_error"]


def test_a_plain_access_on_a_declared_word_is_reported(tmp_path):
    kernel = _mixed(scoped_reset=False)
    findings = _bypass_findings(kernel)
    assert findings, "a plain store on a claimed word must be reported"
    assert all(finding.status == "error" for finding in findings)
    assert all("category" not in finding.details for finding in findings)
    assert all(finding.details["kind"] == "signal_protocol_error" for finding in findings)
    assert "without a happens-before relationship" in findings[0].message
    assert "Hint:" in racecheck(kernel, _packed_inputs("int32")).format()


def test_a_scoped_reset_is_not_reported(tmp_path):
    # The positive control's twin: the identical protocol whose reset carries a
    # scope keeps the word clean, so the finding above is caused by the plain
    # access and not by the protocol's own shape.
    kernel = _mixed(scoped_reset=True)
    assert not _bypass_findings(kernel)


def test_an_ordered_plain_reset_is_not_reported(tmp_path):
    # The other twin: the same plain store, now ordered against every wait on
    # the word by a rendezvous the program already had. Being plain is not the
    # defect -- being concurrent with the protocol is -- so this one is clean.
    inputs = {
        "state": np.zeros(1, "int32"),
        "gate": np.zeros(1, "int32"),
        "observed": np.zeros(1, "int32"),
    }
    assert not _bypass_findings(_ordered_reset(), inputs)


@pytest.mark.parametrize("waiter", [0, 1])
@pytest.mark.parametrize("prior_access", [False, True])
@pytest.mark.parametrize("ordering", ["wait_then_plain", "plain_then_wait", "prefix_only", "none"])
def test_wait_has_its_own_hb_event(waiter, prior_access, ordering):
    """Only a rendezvous between the wait and plain access orders the pair.

    A first-event wait used to have epoch zero, making a subsequent barrier
    unable to prove HB. Reusing a nonzero prior-access epoch was also wrong:
    a barrier before the wait could then incorrectly order the future wait.
    Swapping the warp roles exercises either recorder arrival order.
    """
    initial_access = "observed[0] = T.int32(1)" if prior_access else "T.evaluate(0)"
    wait = f"""
    if warp == {waiter} and lane == 0:
        T.cuda.wait_until(seen[0], state.ptr_to([0]), seen[0] == 7, "gpu", "global")
"""
    # The read keeps the plain access visible to the protocol recorder; a
    # store alone may be compacted through the summary-only fast path. Store
    # the same value so the unordered controls cannot block the numeric wait.
    plain = f"""
    if warp == {1 - waiter} and lane == 0:
        observed[1] = state[0]
        T.ptx.st.global_.s32(state.ptr_to([0]), T.int32(7))
"""
    barrier = "\n    T.ptx.bar.sync(T.uint32(0), T.uint32(64))\n"
    body = {
        "wait_then_plain": wait + barrier + plain,
        "plain_then_wait": plain + barrier + wait,
        "prefix_only": barrier + wait + plain,
        "none": wait + plain,
    }[ordering]
    kernel = tvm.script.from_source(
        f"""
@T.prim_func
def wait_event(state: T.Buffer((1,), "int32"), observed: T.Buffer((2,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if warp == {waiter} and lane == 0:
        {initial_access}
{body}
""",
        {"T": T},
    )

    def inputs():
        return {"state": np.array([7], dtype=np.int32), "observed": np.zeros(2, dtype=np.int32)}

    synccheck(kernel, inputs()).require_clean()
    report = racecheck(kernel, inputs())
    if ordering in {"wait_then_plain", "plain_then_wait"}:
        report.require_clean()
    else:
        assert report.verdict == "error", report.format()
        assert {finding.kind for finding in report.findings} == {"signal_protocol_error"}


def _bit_typed(*, primitive):
    """radix's own split: `add` typed, `ld`/`st` bit-typed.

    The migration's whole promise is that the emitted PTX does not change, and
    a kernel that spells its loads `.b32` is the common case (31 of the
    corpus's ordered loads and stores). The declared form has to reach that
    spelling, so this pins it against the raw one.
    """

    if primitive:
        arrive = 'T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))'
        wait = (
            "T.cuda.wait_until("
            'seen[0], state.ptr_to([0]), seen[0] >= 1, "gpu", "global", "b32")'
        )
    else:
        arrive = 'T.ptx["red.release.gpu.global.add.s32"](state.ptr_to([0]), T.int32(1))'
        wait = (
            "while seen[0] < 1:\n"
            '                T.ptx["ld.acquire.gpu.global.b32"](seen[0], state.ptr_to([0]))'
        )

    return tvm.script.from_source(
        f"""
@T.prim_func
def bit_typed(state: T.Buffer((1,), "int32"), observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            {arrive}
        else:
            seen[0] = 0
            {wait}
            observed[0] = seen[0]
""",
        {"T": T},
    )


def test_bit_typed_word_matches_raw_spelling(tmp_path):
    raw = _bit_typed(primitive=False)
    primitive = _bit_typed(primitive=True)
    raw_module = numsim.transpile(raw, cache_dir=tmp_path / "raw")
    module = numsim.transpile(primitive, cache_dir=tmp_path / "primitive")

    assert _memory_variants(module.rust_source) == _memory_variants(raw_module.rust_source)
    # The store marks an access; the wait performs none.
    raw_outputs = numsim.Engine().run(raw_module, _packed_inputs("int32")).outputs
    outputs = numsim.Engine().run(module, _packed_inputs("int32")).outputs
    np.testing.assert_array_equal(outputs["observed"], raw_outputs["observed"])
    np.testing.assert_array_equal(outputs["observed"], np.array([1], np.int32))


def _two_arrivals(*, target):
    """Two publishers, one waiter, and a target that decides what it may read.

    Warp 0 publishes `first` and arrives; warp 1 publishes `second` and
    arrives; warp 2 waits for the counter to reach `target` and then reads
    `second`. Only the second arrival releases `second`, so reading it is
    ordered exactly when the wait cannot have left before both arrived.

    This is what separates the declared edge from the read-from edge the
    engine already had. Read-from hands the waiter whatever version its load
    happened to observe -- at `target=1` that can still be the second arrival,
    which would make the defect disappear on the schedule that runs. API §3
    takes the *earliest* write the predicate accepts instead, so `target=1`
    is judged on the first arrival alone however late the loop actually left.
    """

    return tvm.script.from_source(
        f"""
@T.prim_func
def two_arrivals(state: T.Buffer((1,), "int32"), first: T.Buffer((1,), "int32"),
                 second: T.Buffer((1,), "int32"), observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([3])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            first[0] = 11
            T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))
        else:
            if warp == 1:
                second[0] = 22
                T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))
            else:
                seen[0] = 0
                T.cuda.wait_until(
                    seen[0], state.ptr_to([0]), seen[0] >= {target}, "gpu", "global")
                observed[0] = second[0]
""",
        {"T": T},
    )


def _two_arrivals_inputs():
    return {
        "state": np.zeros(1, np.int32),
        "first": np.zeros(1, np.int32),
        "second": np.zeros(1, np.int32),
        "observed": np.zeros(1, np.int32),
    }


def test_a_wait_that_waits_for_both_arrivals_may_read_both():
    racecheck(_two_arrivals(target=2), _two_arrivals_inputs()).require_clean()


def test_a_wait_that_waits_for_one_arrival_may_not_read_the_other():
    report = racecheck(_two_arrivals(target=1), _two_arrivals_inputs())
    assert report.verdict == "error", report.format()
    # The reported pair is the second publisher's store against the waiter's
    # read of it: waiting for one arrival orders nothing the other published.
    assert "second" in report.format()


def _woken_by_a_bypass():
    """A wait let out by a plain write.

    The two rules meet here. A scoped write is how a protocol publishes now, so
    it is not the defect; a *plain* one is. The address claim reports it, and
    the wait reports separately that nothing in the word's history explains the
    value it left on, so it built no edge. Both are needed: the claim names the
    defect, and the wait says the reader is holding an ordering it was never
    given.
    """

    return tvm.script.from_source(
        """
@T.prim_func
def woken_by_a_bypass(state: T.Buffer((1,), "int32"), observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            state[0] = T.int32(7)
        else:
            seen[0] = 0
            T.cuda.wait_until(
                seen[0], state.ptr_to([0]), seen[0] != 0, "gpu", "global")
            observed[0] = seen[0]
""",
        {"T": T},
    )


def test_a_wait_woken_by_a_plain_write_reports_the_missing_edge():
    """A compacted plain write leaves insufficient evidence to explain the wait.

    The dense-output summary path does not retain this write as a signal
    access witness. Without that evidence, the checker reports an incomplete
    analysis, not a proven signal protocol error or deadlock.
    """

    report = racecheck(_woken_by_a_bypass(), _packed_inputs("int32"))
    kinds = {(finding.status, finding.kind) for finding in report.findings}
    assert ("incomplete", "analysis_incomplete") in kinds, report.format()


def _backoff_spin(*, primitive):
    """A contended wait, in the two spellings a kernel has for it.

    The hand-written form is what `allgather_gemm` and `gemm_reduce_scatter`
    write: load, test, back off, load again. The primitive form is a single
    `wait_until` carrying `backoff_ns`, which generates the
    same sequence -- the sleep sits inside the loop and ahead of the load, so
    a first poll that succeeds pays nothing.
    """

    if primitive:
        spin = (
            "T.cuda.wait_until(\n"
            "                seen[0], state.ptr_to([0]), seen[0] >= 1,\n"
            '                "gpu", "global", backoff_ns=40)'
        )
        publish = (
            'T.ptx.red.release.gpu.global_.add.s32(state.ptr_to([0]), T.int32(1))'
        )
    else:
        spin = (
            'T.ptx["ld.acquire.gpu.global.s32"](seen[0], state.ptr_to([0]))\n'
            "            while seen[0] < 1:\n"
            "                T.cuda.nano_sleep(T.uint64(40))\n"
            '                T.ptx["ld.acquire.gpu.global.s32"](seen[0], state.ptr_to([0]))'
        )
        publish = 'T.ptx["red.release.gpu.global.add.s32"](state.ptr_to([0]), T.int32(1))'

    return tvm.script.from_source(
        f"""
@T.prim_func
def backoff(state: T.Buffer((1,), "int32"), slot: T.Buffer((1,), "int32"),
            observed: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        if warp == 0:
            slot[0] = 7
            {publish}
        else:
            seen[0] = 0
            {spin}
            observed[0] = slot[0]
""",
        {"T": T},
    )


def _backoff_inputs():
    return {
        "state": np.zeros(1, np.int32),
        "slot": np.zeros(1, np.int32),
        "observed": np.full(1, -1, np.int32),
    }


def test_a_backoff_is_the_cuda_loops_business_and_not_the_engines(tmp_path):
    """`backoff_ns` is a field on the wait, and it lowers in one place only.

    A contended wait spelled by hand sleeps between polls, and the CUDA this
    lowers to still does -- `tests/python/tirx/codegen/test_cuda_wait_until.py`
    pins the `__nanosleep` inside the emitted loop. The engine has no loop to
    sleep in: it parks on the word's own bytes until they change, which is
    strictly better than waking on a timer. So numsim emits none, and the two
    spellings still agree on every global instruction and on the result.
    """

    raw = numsim.transpile(_backoff_spin(primitive=False), cache_dir=tmp_path / "raw")
    module = numsim.transpile(_backoff_spin(primitive=True), cache_dir=tmp_path / "primitive")

    # Instruction equality is `test_rendezvous_matches_raw_spelling`'s job. The
    # two spellings genuinely differ here: written by hand the idiom loads once
    # before the loop, and the primitive needs no such load because the wait
    # tests its destination before it does anything.
    assert "control::nanosleep" not in module.rust_source
    assert raw.rust_source.count("control::nanosleep") == 1
    # The publisher's arrival; the wait marks nothing.

    outputs = numsim.Engine().run(module, _backoff_inputs()).outputs
    np.testing.assert_array_equal(outputs["observed"], np.array([7], np.int32))
    racecheck(_backoff_spin(primitive=True), _backoff_inputs()).require_clean()


def test_a_candidate_index_outside_its_local_table_is_an_execution_error(tmp_path):
    case = indexed_predicate_case(alias=False)
    inputs = dict(case.args)
    inputs["state"] = np.full(32, 2, np.int32)
    module = numsim.transpile(case.kernel, cache_dir=tmp_path)

    with pytest.raises(
        numsim.NumSimExecutionError,
        match="wait_until predicate index is outside local buffer table",
    ):
        numsim.Engine().run(module, inputs)


def _packed_payload(*, retry):
    """#677's examples 1 and 2, written through the primitive.

    The payload rides in the polled word itself -- the generation in the high
    half, the value in the low half -- which is the shape DeepEP's notify slot
    uses and the one a separate payload buffer cannot stand in for. Both
    versions declare the word; they differ only in whether the reader waits.

    A wait that retries is the protocol working. A single load is #677's "read
    once, no retry": the reader takes whatever the word held, and declaring it
    must not make that look correct.
    """

    read = (
        "T.cuda.wait_until(\n"
        "                observed[0], slot.ptr_to([0]),\n"
        '                T.Cast("uint32", T.shift_right(observed[0], T.uint64(32)))'
        " == T.uint32(1),\n"
        '                "gpu", "global")'
        if retry
        else (
            # Relaxed, not acquiring. An acquiring single read is what an
            # acquiring spin is made of, and the checker sees accesses rather
            # than loops, so it cannot separate the two: sparing the loop
            # spares this. A relaxed read takes the value and no order, which
            # is the shape the claim can name on every schedule.
            "T.ptx.ld.relaxed.gpu.global_.u64(\n"
            '                observed[0], slot.ptr_to([0]))'
        )
    )
    word = (1 << 32) | 0xDEADBEEF
    return tvm.script.from_source(
        f"""
@T.prim_func
def packed(slot: T.Buffer((1,), "uint64"), sink: T.Buffer((1,), "uint64")):
    T.device_entry()
    cta = T.cta_id([2])
    lane = T.lane_id([32])
    observed = T.alloc_local((1,), "uint64")
    if lane == 0:
        if cta == 0:
            T.ptx.st.release.gpu.global_.u64(slot.ptr_to([0]), T.uint64({word}))
        else:
            observed[0] = T.uint64(0)
            {read}
            sink[0] = T.bitwise_and(observed[0], T.uint64(0xFFFFFFFF))
""",
        {"T": T},
    )


def _packed_payload_inputs():
    return {"slot": np.zeros(1, np.uint64), "sink": np.zeros(1, np.uint64)}


def test_a_wait_on_a_word_that_carries_its_own_payload_is_clean():
    """The protocol working, with nothing but the declared word involved.

    A spin is by definition a read racing the publisher's write, so this is
    only clean if the wait is one operation to the checker rather than a loop
    of reads it has to adjudicate.
    """

    racecheck(_packed_payload(retry=True), _packed_payload_inputs()).require_clean()


def test_reading_the_word_once_is_not_made_correct_by_declaring_it():
    """#677's example 2, declared. The declaration must not launder it.

    The payload is in the word, so there is no second buffer whose read could
    report instead: if this comes back clean, the primitive is blind to the
    bug it was built for.

    The single read is relaxed. With an acquiring one the checker cannot tell
    this from a correct acquiring spin -- it sees accesses, not loops -- and
    reporting it would report every hand-written acquire loop as well. What it
    keeps is the shape where the reader took a value and no order at all.
    """

    report = racecheck(_packed_payload(retry=False), _packed_payload_inputs())
    assert report.verdict != "clean", report.format()
