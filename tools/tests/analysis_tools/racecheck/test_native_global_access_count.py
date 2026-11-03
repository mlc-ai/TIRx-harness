"""A compact RaceCheck run must account for the same accesses as a journaled one."""

from __future__ import annotations

import numpy as np
import pytest

from tests.analysis_tools.racecheck._native_race_trace import full_direct_race_run
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tvm_tirx.script import tirx as T


@T.function
def native_global_write(output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    output[lane] = lane


@T.function
def native_global_read_write(source: T.Tensor((32,), "int32"), output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    output[lane] = source[lane] + 1


@T.function
def native_global_atomic_only(flag: T.Tensor((1,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    T.evaluate(T.cuda.atomic_add(flag.ptr_to([0]), T.int32(1)))


@T.function
def native_global_repeated_read(source: T.Tensor((1,), "int32"), output: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
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
