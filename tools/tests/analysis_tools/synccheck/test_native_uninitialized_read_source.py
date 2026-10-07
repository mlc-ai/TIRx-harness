"""Uninitialized operands retain their read site, rather than the kernel's last op."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness.numsim.checkers import _run_racecheck, _run_synccheck


@T.prim_func
def repeated_uninitialized_operand(output: T.Buffer((128,), "uint32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    value = T.alloc_buffer((1,), "uint32", scope="local")
    T.ptx.st.global_.u32(output.ptr_to([warp * 64 + lane]), value[0])  # first read
    T.cuda.cta_sync()
    T.ptx.st.global_.u32(output.ptr_to([warp * 64 + 32 + lane]), value[0])  # second read
    value[0] = 42  # final statement is unrelated to the invalid reads


@pytest.mark.parametrize(
    "checker", [_run_synccheck, _run_racecheck], ids=["synccheck", "racecheck"]
)
def test_uninitialized_read_records_consuming_statements_and_offsets(checker):
    report = checker(repeated_uninitialized_operand, inputs={"output": np.zeros(128, np.uint32)})
    findings = [finding for finding in report.findings if finding.kind == "uninitialized_read"]
    assert report.verdict == "review"
    assert len(findings) == 128  # two reads of each of the 64 lane-private registers
    lines = Path(__file__).read_text().splitlines()
    expected = {
        index
        for index, line in enumerate(lines, 1)
        if line.rstrip().endswith(("# first read", "# second read"))
    }
    seen = set()
    for finding in findings:
        details = finding.details
        assert details["space"] == "register"
        assert "source_anchor" not in details
        source = details["source"]
        assert source["kind"] == "Call"
        line = source["source_span"]["line"]
        assert line in expected, details
        assert details["byte_len"] == 4
        assert details["first_uninitialized_byte"] == details["byte_offset"]
        seen.add((line, details["global_warp_id"]))
    assert seen == {(line, warp) for line in expected for warp in (0, 1)}
    assert {finding.details["byte_offset"] for finding in findings} == {
        lane * 4 for lane in range(32)
    }
