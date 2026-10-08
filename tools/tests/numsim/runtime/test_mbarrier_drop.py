"""Arrive-drop changes future expectations, not just the current pending count."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import racecheck, synccheck

DROP_FORMS = (
    "nocount",
    "count",
    "state",
    "count_state",
    "expect_tx",
    "expect_tx_state",
    "no_complete",
    "no_complete_sink",
)


def drop_kernel(form, *, drop=True, count=None):
    counted = form in {"count", "count_state", "no_complete", "no_complete_sink"}
    amount = 2 if counted else 1
    instruction = "mbarrier.arrive_drop" if drop else "mbarrier.arrive"
    if "expect_tx" in form:
        instruction += ".expect_tx"
    elif "no_complete" in form:
        instruction += ".noComplete"
    instruction += ".shared.b64"
    args = ["barrier.ptr_to([0])"]
    if form in {"state", "count_state", "expect_tx_state", "no_complete"}:
        args.insert(0, "state[0]")
    if counted or "expect_tx" in form:
        args.append(
            f"T.uint32({count if count is not None else 16 if 'expect_tx' in form else amount})"
        )
    return tvm.script.from_source(
        f"""
@T.prim_func
def dropped(out: T.Buffer((3,), "uint32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    barrier = T.alloc_buffer((1,), "uint64", scope="shared")
    state = T.alloc_local((1,), "uint64")
    ready = T.alloc_local((1,), "uint32")
    if lane == 0:
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), {amount + 2})
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()
    if lane == 0:
        T.ptx["{instruction}"]({", ".join(args)})
        {"T.ptx.mbarrier.complete_tx.relaxed.cta.shared.b64(barrier.ptr_to([0]), 16)" if "expect_tx" in form else "T.evaluate(0)"}
        for phase in T.serial(3):
            T.ptx.mbarrier.arrive.shared.b64(state[0], barrier.ptr_to([0]), T.uint32(2))
            T.ptx.mbarrier.test_wait.shared.b64(ready[0], barrier.ptr_to([0]), state[0])
            out[phase] = ready[0]
""",
        {"T": T},
    )


@pytest.mark.parametrize("form", DROP_FORMS)
def test_mbarrier_drop(form, tmp_path):
    kernel = drop_kernel(form)
    inputs = {"out": np.zeros(3, np.uint32)}
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], [1, 1, 1])


@pytest.mark.parametrize("form", ["count", "no_complete"])
def test_mbarrier_drop_invalid_count(form):
    kernel = drop_kernel(form, count=5)
    for checker in (synccheck, racecheck):
        report = checker(kernel, {"out": np.zeros(3, np.uint32)})
        assert report.verdict == "error", report.format()
        assert any(
            "arrival" in finding.kind or finding.kind == "engine_error"
            for finding in report.findings
        )
