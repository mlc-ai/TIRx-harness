"""Public wait/signal diagnostics describe proof, not declaration inventory."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tirx_harness import racecheck, synccheck
from tirx_harness.numsim.checkers import _run_racecheck, _run_synccheck


def _raw_spin(*, ordered):
    store_order = "release" if ordered else "relaxed"
    rendezvous = "T.ptx.bar.sync(T.uint32(0), T.uint32(64))" if ordered else "T.evaluate(0)"
    return tvm.script.from_source(
        f"""
@T.prim_func
def raw_spin(flag: T.Buffer((1,), "int32"), out: T.Buffer((1,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if warp == 0 and lane == 0:
        T.ptx.st.{store_order}.gpu.global_.s32(flag.ptr_to([0]), T.int32(7))
    {rendezvous}
    if warp == 1 and lane == 0:
        seen[0] = 0
        while seen[0] != 7:
            T.ptx.ld.acquire.gpu.global_.s32(seen[0], flag.ptr_to([0]))
        out[0] = seen[0]
""",
        {"T": T},
    )


def _inputs():
    return {"flag": np.zeros(1, dtype=np.int32), "out": np.zeros(1, dtype=np.int32)}


def test_raw_spin_missing_hb_reports_race_with_conditional_wait_hint():
    report = racecheck(_raw_spin(ordered=False), _inputs())
    assert report.verdict == "error", report.format()
    assert {f.kind for f in report.findings} == {"data_race"}
    assert {f.details["access_pair"] for f in report.findings} <= {"write_read", "read_write"}
    assert all(f["kind"] == "data_race" and "category" not in f
               for f in report.native_payload["findings"])
    assert "[ERROR] data_race:" in report.format()
    assert "Access pair:" in report.format()
    assert "[ERROR] write_read:" not in report.format()
    assert any("wait_until" in f.details.get("hint", "") for f in report.findings)
    assert "Hint:" in report.format()
    assert "changing the wait API alone does not establish happens-before" in report.format()
    assert "undeclared_protocol_words" not in report.to_dict()["native"]


def test_hb_ordered_raw_spin_does_not_need_a_wait_declaration():
    kernel = _raw_spin(ordered=True)
    synccheck(kernel, _inputs()).require_clean()
    report = racecheck(kernel, _inputs())
    report.require_clean()
    assert "undeclared_protocol_words" not in report.to_dict()["native"]


@pytest.mark.parametrize("checker", [synccheck, racecheck])
def test_wait_without_any_possible_publisher_is_a_sync_deadlock(checker):
    kernel = tvm.script.from_source(
        """
@T.prim_func
def blocked_wait(flag: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    seen = T.alloc_local((1,), "int32")
    if lane == 0:
        T.cuda.wait_until(seen[0], flag.ptr_to([0]), seen[0] == 7, "gpu", "global")
""",
        {"T": T},
    )
    report = checker(kernel, {"flag": np.zeros(1, dtype=np.int32)})
    assert report.verdict == "error", report.format()
    assert any(f.kind == "deadlock" for f in report.findings), report.format()
    assert all(f.status == "error" for f in report.findings)


@pytest.mark.parametrize("checker", [_run_synccheck, _run_racecheck])
def test_poll_budget_exhaustion_is_incomplete_not_deadlock(checker):
    report = checker(_raw_spin(ordered=True), _inputs(), max_polls=1)
    assert report.verdict == "incomplete", report.format()
    assert {f.kind for f in report.findings} == {"analysis_incomplete"}
    assert {f.status for f in report.findings} == {"incomplete"}
    assert any(f.details["reason"] == "resource_limit" for f in report.findings)
    error = report.native_payload["execution_error"]
    assert error["kind"] == "analysis_incomplete"
    assert error["reason"] == "resource_limit"
    assert error["resource"] == "polls"
    assert "[INCOMPLETE] analysis_incomplete:" in report.format()
    assert "Reason: resource_limit" in report.format()
