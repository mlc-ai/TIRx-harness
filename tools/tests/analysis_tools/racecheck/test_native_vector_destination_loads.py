"""Checker-mode coverage for destination-passing PTX vector loads.

A vector `ld` is modeled as its element accesses, so the whole point of these
cases is that the checkers still see the load: its footprint must be traced
like any other read, and its `.relaxed`/`.acquire` qualifier must still build
the happens-before edge the scalar spelling builds. A vector load that lost
either property would make both checkers silently permissive.
"""

from __future__ import annotations

import numpy as np

from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm.script import tirx as T


@T.prim_func
def barrier_ordered_v2_destination_load(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "int32", scope="shared")
    pair = T.alloc_local((2,), "int32")
    if warp == 0:
        for index in T.unroll(2):
            shared[lane * 2 + index] = lane * 2 + index
    T.cuda.cta_sync()
    if warp == 1:
        T.ptx.ld.shared.v2.s32(pair[0], pair[1], shared.ptr_to([lane * 2]))
        for index in T.unroll(2):
            output[lane * 2 + index] = pair[index]


@T.prim_func
def unordered_v2_destination_load(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "int32", scope="shared")
    pair = T.alloc_local((2,), "int32")
    if warp == 0:
        for index in T.unroll(2):
            shared[lane * 2 + index] = lane * 2 + index
    if warp == 1:
        T.ptx.ld.shared.v2.s32(pair[0], pair[1], shared.ptr_to([lane * 2]))
        for index in T.unroll(2):
            output[lane * 2 + index] = pair[index]


@T.prim_func
def barrier_ordered_acquire_v2_destination_load(output: T.Buffer((64,), "int32")):
    T.device_entry()
    warp = T.warp_id([2])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((64,), "int32", scope="shared")
    pair = T.alloc_local((2,), "int32")
    if warp == 0:
        for index in T.unroll(2):
            shared[lane * 2 + index] = lane * 2 + index
    T.cuda.cta_sync()
    if warp == 1:
        T.ptx.ld.acquire.cta.shared.v2.s32(pair[0], pair[1], shared.ptr_to([lane * 2]))
        for index in T.unroll(2):
            output[lane * 2 + index] = pair[index]


def test_racecheck_traces_the_element_footprint_of_a_vector_destination_load(tmp_path):
    report = racecheck(
        barrier_ordered_v2_destination_load,
        inputs={"output": np.zeros(64, dtype=np.int32)},
        cache_dir=tmp_path,
    )

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["incomplete"] == []
    assert native["access_count"] > 0


def test_racecheck_reports_a_vector_destination_load_racing_an_unordered_write(tmp_path):
    """Positive control: without the barrier the same load must be flagged.

    This is what proves the clean verdict above is a real trace rather than a
    vector load the checker never saw.
    """

    report = racecheck(
        unordered_v2_destination_load,
        inputs={"output": np.zeros(64, dtype=np.int32)},
        cache_dir=tmp_path,
    )

    assert report.verdict == "error"
    assert report.findings
    kinds = {finding.details["access_pair"] for finding in report.findings}
    assert kinds <= {"write_read", "write_write", "read_write"}
    native = report.to_dict()["native"]
    spaces = {
        finding[side]["space"] for finding in native["findings"] for side in ("prior", "current")
    }
    assert spaces == {"shared"}


def test_acquire_vector_destination_load_keeps_its_ordering_specialization(tmp_path):
    """The `.acquire` vector spelling reaches the checkers as an acquire load.

    `ld.acquire` takes `.vec` in PTX, so the vector form must select the same
    scoped-acquire specialization the scalar form does rather than degrading
    to a plain load.
    """

    report = racecheck(
        barrier_ordered_acquire_v2_destination_load,
        inputs={"output": np.zeros(64, dtype=np.int32)},
        cache_dir=tmp_path,
    )

    report.require_clean()
    assert report.to_dict()["native"]["incomplete"] == []


def test_synccheck_accepts_the_ordered_vector_destination_load_kernels(tmp_path):
    for kernel in (
        barrier_ordered_v2_destination_load,
        barrier_ordered_acquire_v2_destination_load,
    ):
        report = synccheck(
            kernel,
            inputs={"output": np.zeros(64, dtype=np.int32)},
            cache_dir=tmp_path,
        )
        report.require_clean()
