"""A frontend finding points at the offending node, with no anchor precondition.

Reporting an unsupported node must not depend on the enclosing PrimFunc having a
usable span. TVM's statement constructors (``SeqStmt(...)``,
``PrimFunc.with_body(...)``) default ``span=None``, so a caller that recomposed
the root hands in a spanless function -- and the finding is about one node
anyway, whose own span the analysis already knows.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import tvm.tirx as tirx

from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T
from tvm.tirx.layout import S, TileLayout, laneid

_SOURCE_LINES = pathlib.Path(__file__).read_text().splitlines()


def _line_containing(needle: str) -> int:
    """1-indexed line of `needle` in this file, ignoring the lookups themselves.

    Deriving the expected line beats hardcoding it: any edit above the kernel
    would otherwise fail the test for a reason that has nothing to do with what
    it checks.
    """

    for index, text in enumerate(_SOURCE_LINES, start=1):
        if needle in text and "_line_containing" not in text:
            return index
    raise AssertionError(f"{needle!r} is not in this file")


# The layout on this buffer is what NumSim rejects; keep the alloc on one line
# so the expected anchor line is unambiguous.
_OFFENDING_LAYOUT = TileLayout(S[(1, 1) : (1, 1)] + 1 @ laneid)


@T.prim_func
def unsupported_local_layout(output: T.Buffer((1,), "int32")):
    T.device_entry()
    lane = T.lane_id([32])
    local = T.alloc_buffer((1, 1), "int32", scope="local", layout=_OFFENDING_LAYOUT)
    if lane == 0:
        local[0, 0] = 7
        output[0] = local[0, 0]


_ALLOC_LINE = _line_containing('local = T.alloc_buffer((1, 1), "int32"')


def _kernel_with_rebuilt_root():
    """Same PrimFunc, root statement recomposed the way a caller's pass would."""

    rebuilt = tirx.SeqStmt(
        [unsupported_local_layout.body, tirx.Evaluate(tirx.const(0, "int32"))],
        span=None,
    )
    return unsupported_local_layout.with_body(rebuilt, span=None)


def _only_frontend_finding(report):
    assert report.verdict == "incomplete"
    findings = [f for f in report.findings if f.kind == "analysis_incomplete" and f.details["reason"] == "native_frontend_unsupported"]
    assert len(findings) == 1, [f.kind for f in report.findings]
    return findings[0]


def _anchor_span(finding) -> dict:
    anchor = finding.details["source_anchor"]
    span = anchor["source_span"]
    assert span["source_name"].endswith("test_native_frontend_source_anchor.py"), span
    return span


def test_rebuilt_root_still_has_no_span():
    """Guard the premise: this is the shape that used to be rejected outright."""

    kernel = _kernel_with_rebuilt_root()
    assert kernel.span is None
    assert kernel.body.span is None
    assert unsupported_local_layout.body.span is not None


@pytest.mark.parametrize("rebuilt", [False, True], ids=["as-parsed", "rebuilt-root"])
@pytest.mark.parametrize("checker", ["synccheck", "racecheck"])
def test_frontend_finding_points_at_the_offending_node(checker, rebuilt, tmp_path):
    kernel = _kernel_with_rebuilt_root() if rebuilt else unsupported_local_layout
    inputs = {"output": np.zeros((1,), dtype=np.int32)}

    if checker == "synccheck":
        report = synccheck(kernel, inputs=inputs, cache_dir=tmp_path, max_workers=1)
    else:
        report = racecheck(kernel, inputs=inputs)

    finding = _only_frontend_finding(report)
    assert "buffer:local:" in finding.message
    # The anchor is the buffer the frontend rejected, not the enclosing function.
    assert _anchor_span(finding)["line"] == _ALLOC_LINE


def test_recomposing_the_root_does_not_move_the_reported_location(tmp_path):
    inputs = {"output": np.zeros((1,), dtype=np.int32)}
    direct = synccheck(
        unsupported_local_layout,
        inputs=inputs,
        cache_dir=tmp_path,
        max_workers=1,
    )
    rebuilt = synccheck(
        _kernel_with_rebuilt_root(),
        inputs=inputs,
        cache_dir=tmp_path,
        max_workers=1,
    )

    assert _anchor_span(_only_frontend_finding(direct)) == _anchor_span(
        _only_frontend_finding(rebuilt)
    )


@T.prim_func
def call_location_probe(source: T.Buffer((32,), "uint8")):
    """A valid call whose source location survives a frontend rejection."""

    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.cuda.ldg(source.ptr_to([31 - lane]), "boolx2"))


_UNMODELED_CALL_LINE = _line_containing('T.cuda.ldg(source.ptr_to([31 - lane]), "boolx2")')


def test_unmodeled_form_is_also_located(tmp_path):
    """UnmodeledTIRxFormError is raised directly and never reaches verify()."""

    # A byte-addressable backing lets instruction emission report the unmodeled
    # result type instead of failing the global memory plan first.
    report = synccheck(
        call_location_probe,
        inputs={"source": np.zeros((32,), dtype=np.uint8)},
        cache_dir=tmp_path,
        max_workers=1,
    )

    finding = _only_frontend_finding(report)
    assert "no exact NumSim packed-bool ABI" in finding.message
    assert _anchor_span(finding)["line"] == _UNMODELED_CALL_LINE
