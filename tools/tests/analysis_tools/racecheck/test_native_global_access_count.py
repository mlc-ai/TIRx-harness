"""A compact RaceCheck run must account for the same accesses as a journaled one."""

from __future__ import annotations

import numpy as np
import pytest
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tests.analysis_tools.racecheck._native_race_trace import full_direct_race_run
from tvm.script import tirx as T


@T.prim_func
def native_global_write(output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = lane


@T.prim_func
def native_global_read_write(source: T.Buffer((32,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = source[lane] + 1


@T.prim_func
def native_global_atomic_only(flag: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    T.evaluate(T.cuda.atomic_add(flag.ptr_to([0]), T.int32(1)))


@T.prim_func
def native_global_repeated_read(source: T.Buffer((1,), "int32"), output: T.Buffer((32,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    acc = T.local_scalar("int32")
    acc = 0
    for _i in T.serial(4):
        acc = acc + source[0]
    output[lane] = acc


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-racecheck-global-access-count")


@pytest.mark.parametrize(
    ("kernel", "inputs", "expected"),
    [
        (native_global_write, {"output": np.zeros(32, dtype=np.int32)}, 1),
        (
            native_global_read_write,
            {
                "source": np.arange(32, dtype=np.int32),
                "output": np.zeros(32, dtype=np.int32),
            },
            2,
        ),
        (native_global_atomic_only, {"flag": np.zeros(1, dtype=np.int32)}, 1),
    ],
    ids=["write", "read_write", "atomic_rmw"],
)
def test_global_accesses_are_counted_without_the_access_journal(
    kernel, inputs, expected, native_cache_dir
):
    report = racecheck(kernel, inputs=inputs, cache_dir=native_cache_dir, max_workers=1)
    report.require_clean()
    compact = report.to_dict()["native"]

    # The compact summary keeps no records, so its count is the only evidence
    # that the global accesses were accounted for at all.
    assert compact["accesses"] == []
    assert compact["accesses_complete"] is False
    assert compact["access_count"] == expected

    # `full_direct_race_run` reruns the identical execution with the journal on
    # and asserts the two counts agree; the journal then proves what was counted.
    journaled = full_direct_race_run(kernel, inputs, native_cache_dir, compact)
    assert journaled["access_count"] == expected
    assert [access["space"] for access in journaled["accesses"]] == ["global"] * expected


def test_repeated_read_access_count_parity(native_cache_dir):
    # Rereading one location makes the later reads replay a cached version (the
    # `Stable` finish), a different early exit of the global branch than a
    # first-time read or a write takes. Each replay is still one executed access
    # and must count once. The journal also records the local-scalar register
    # traffic, so pin the global subset and leave the total to the
    # compact-vs-journal parity that `full_direct_race_run` asserts.
    inputs = {
        "source": np.full(1, 3, dtype=np.int32),
        "output": np.zeros(32, dtype=np.int32),
    }
    report = racecheck(
        native_global_repeated_read, inputs=inputs, cache_dir=native_cache_dir, max_workers=1
    )
    report.require_clean()
    compact = report.to_dict()["native"]
    assert compact["accesses"] == []

    journaled = full_direct_race_run(native_global_repeated_read, inputs, native_cache_dir, compact)
    spaces = [access["space"] for access in journaled["accesses"]]
    assert spaces.count("global") == 5  # four reads of source[0], one output write


def test_async_tile_copy_access_count_parity(native_cache_dir):
    # Async copies coalesce element accesses into batches that carry a
    # semantic access count, and both modes account for them through the
    # async staging paths rather than the synchronous sites the other cases
    # pin. The artifact test for this kernel only asserts `> 0`; this pins
    # compact-vs-journal parity for the async regime.
    from tests.analysis_tools.racecheck.test_native_racecheck_artifact import (
        native_racecheck_cross_cluster_tile_copy_waw,
    )

    inputs = {
        "source": np.ones(128, dtype=np.float32),
        "output": np.zeros(128, dtype=np.float32),
    }
    report = racecheck(
        native_racecheck_cross_cluster_tile_copy_waw,
        inputs=inputs,
        cache_dir=native_cache_dir,
        max_workers=1,
    )
    compact = report.to_dict()["native"]
    assert report.verdict == "error"
    assert compact["accesses"] == []
    assert compact["access_count"] == 4

    journaled = full_direct_race_run(
        native_racecheck_cross_cluster_tile_copy_waw, inputs, native_cache_dir, compact
    )
    assert journaled["access_count"] == 4
    assert [access["space"] for access in journaled["accesses"]] == ["global"] * 4
