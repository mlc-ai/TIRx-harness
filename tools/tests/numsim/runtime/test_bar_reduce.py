"""Named-barrier reductions: actual participants, generations and memory ordering."""

import numpy as np
import pytest
import tvm
from tvm.script import tirx as T

from tests.numsim.support.execution import run_checked
from tirx_harness import racecheck

BAR_REDUCE_FORMS = [
    (name, op, counted)
    for name in ("bar", "barrier")
    for op in ("popc", "and", "or")
    for counted in (False, True)
]


def bar_reduce_kernel(name, op, counted, *, count=None, barrier_id=None, synchronize=True):
    count = 64 if count is None else count
    selected = "warp % 2 == 0" if counted else "True"
    args = "result[0], " + (
        "T.uint32(2 + phase % 2)" if barrier_id is None else f"T.uint32({barrier_id})"
    )
    if counted:
        args += f", T.uint32({count})"
    args += ", phase == 1 or (phase == 0 and lane < 5)"
    instruction = f'T.ptx["{name}.red.{op}.{"u32" if op == "popc" else "pred"}"]({args})'
    # Different source sites must rendezvous by the barrier ID/generation, not
    # by source ID. Only the unaligned barrier spelling permits this split.
    body = (
        f"if warp == 0:\n                {instruction}\n            else:\n                {instruction}"
        if name == "barrier"
        else instruction
    )
    if not synchronize:
        body = "T.evaluate(0)"
    return tvm.script.from_source(
        f"""
@T.prim_func
def reduction(out: T.Buffer((3, 4, 32, 2), "uint32")):
    T.device_entry()
    warp = T.warp_id([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((4, 32), "uint32", scope="shared")
    result = T.alloc_local((1,), "uint32")
    result[0] = 0
    for phase in T.serial(3):
        if {selected}:
            shared[warp, lane] = T.uint32(phase * 1000 + warp * 32 + lane)
            {body}
            out[phase, warp, lane, 0] = result[0]
            out[phase, warp, lane, 1] = shared[(warp + 2) % 4, lane]
        T.cuda.cta_sync()
""",
        {"T": T},
    )


def bar_reduce_case(name, op, counted, **kwargs):
    kernel = bar_reduce_kernel(name, op, counted, **kwargs)
    expected = np.zeros((3, 4, 32, 2), np.uint32)
    warps = [0, 2] if counted else [0, 1, 2, 3]
    votes = [len(warps) * 5, len(warps) * 32, 0] if op == "popc" else [int(op == "or"), 1, 0]
    for phase in range(3):
        for warp in warps:
            expected[phase, warp, :, 0] = votes[phase]
            expected[phase, warp, :, 1] = phase * 1000 + ((warp + 2) % 4) * 32 + np.arange(32)
    return kernel, {"out": np.zeros_like(expected)}, expected


@pytest.mark.parametrize("name,op,counted", BAR_REDUCE_FORMS)
def test_bar_reduce(name, op, counted, tmp_path):
    kernel, inputs, expected = bar_reduce_case(name, op, counted)
    result = run_checked(kernel, inputs, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], expected)


@pytest.mark.parametrize(
    "mutation", [{"count": 33}, {"barrier_id": 16}, {"count": 160}, {"synchronize": False}]
)
def test_bar_reduce_invalid_controls(mutation):
    kernel, inputs, _ = bar_reduce_case("barrier", "popc", True, **mutation)
    report = racecheck(kernel, inputs)
    assert report.verdict == "error", report.format()
