"""A global race names the loop iterations of its conflicting instances.

Findings are deduplicated by static access site, but each one must still carry
a concrete dynamic instance: the loop iteration of each side is what tells the
reader which round's hand-off is missing its synchronization.
"""

from __future__ import annotations

import numpy as np

from tirx_harness import racecheck
from tvm_tirx.script import tirx as T

ROUNDS = 3
UNRELEASED_ROUND = 1


@T.function
def round_handoff(
    data: T.Tensor((1,), "int32"),
    full_flag: T.Tensor((1,), "int32"),
    free_flag: T.Tensor((1,), "int32"),
    output: T.Tensor((1,), "int32"),
):
    """CTA 0 writes `data` once per round and publishes it on `full_flag`;
    CTA 1 reads it and hands the slot back on `free_flag`. Only round 1's
    hand-back is relaxed, so exactly one pair races: the read of round 1
    against the overwrite of round 2."""
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32))
    cta = T.cuda.block_idx("x")
    lane = T.cuda.lane_id()
    seen = T.alloc_local((1,), "int32")
    acc = T.alloc_local((1,), "int32")
    if (cta == 0) and (lane == 0):
        for r in T.serial(3):
            seen[0] = T.int32(0)
            T.cuda.wait_until(seen[0], free_flag.ptr_to([0]), seen[0] >= r, "sys", "global")
            data[0] = r + 10
            T.ptx.st.release.gpu.global_.s32(full_flag.ptr_to([0]), r + 1)
    elif (cta == 1) and (lane == 0):
        acc[0] = T.int32(0)
        for r in T.serial(3):
            seen[0] = T.int32(0)
            T.cuda.wait_until(seen[0], full_flag.ptr_to([0]), seen[0] >= r + 1, "sys", "global")
            acc[0] = acc[0] + data[0]
            if r == 1:
                T.ptx.st.relaxed.sys.global_.s32(free_flag.ptr_to([0]), r + 1)
            else:
                T.ptx.st.release.gpu.global_.s32(free_flag.ptr_to([0]), r + 1)
        output[0] = acc[0]


def _inputs():
    return {
        "data": np.zeros(1, np.int32),
        "full_flag": np.zeros(1, np.int32),
        "free_flag": np.zeros(1, np.int32),
        "output": np.zeros(1, np.int32),
    }


def _iterations(witness):
    return [frame["iteration_ordinal"] for frame in witness["operation"]["loop_frames"]]


def test_race_reports_the_iterations_of_the_conflicting_instances():
    report = racecheck(round_handoff, _inputs())
    assert report.verdict == "error", report.format()
    findings = report.native_payload["findings"]
    assert len(findings) == 1, report.format()
    finding = findings[0]
    by_warp = {
        witness["operation"]["global_warp_id"]: witness
        for witness in (finding["prior"], finding["current"])
    }
    writer, reader = by_warp[0], by_warp[1]
    assert writer["access_kind"] == "write" and reader["access_kind"] == "read", finding
    assert _iterations(reader) == [UNRELEASED_ROUND], finding
    assert _iterations(writer) == [UNRELEASED_ROUND + 1], finding
    text = report.format()
    assert f"loop iteration {UNRELEASED_ROUND}" in text
    assert f"loop iteration {UNRELEASED_ROUND + 1}" in text
