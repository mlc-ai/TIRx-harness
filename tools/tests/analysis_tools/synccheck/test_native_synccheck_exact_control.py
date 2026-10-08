"""Native Synccheck integration coverage."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim
from tirx_harness.numsim.checkers import _run_synccheck as internal_synccheck
from tvm.script import tirx as T


@T.prim_func
def native_boolean_guard_warp_collective(
    mode: T.int32, bound: T.int32, flag: T.Buffer((1,), "int32")
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])

    if mode == 0:
        if (flag[0] != 0) and (lane < bound):
            _sum0 = T.cuda.warp_sum(T.float32(1.0))
    elif mode == 1:
        if (flag[0] != 0) & (lane < bound):
            _sum1 = T.cuda.warp_sum(T.float32(1.0))
    elif mode == 2:
        if (flag[0] != 0) | (lane < bound):
            _sum2 = T.cuda.warp_sum(T.float32(1.0))
    else:
        if not ((flag[0] == 0) | (lane >= bound)):
            _sum3 = T.cuda.warp_sum(T.float32(1.0))


@pytest.fixture(scope="module")
def native_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("native-synccheck-exact-control")


def _run_collective(*, mode: int, flag: int, bound: int, cache_dir):
    return internal_synccheck(
        native_boolean_guard_warp_collective,
        inputs={
            "mode": np.int32(mode),
            "bound": np.int32(bound),
            "flag": np.array([flag], dtype=np.int32),
        },
        cache_dir=cache_dir,
        coverage_bounds=numsim.CoverageBounds(
            max_warp_preemptions=0, max_completion_schedule_deviations=0
        ),
        resource_limits=numsim.ResourceLimits(
            max_schedules=16,
            max_backtrack_nodes=100,
            max_events_per_run=1_000,
            max_total_events=10_000,
            max_loop_steps=1_000,
            max_wall_time_ms=30_000,
            max_diagnostic_bytes=1_000_000,
        ),
    )


@pytest.mark.parametrize(
    ("mode", "full_flag", "full_bound", "subset_flag", "subset_bound"),
    [
        pytest.param(0, 1, 32, 1, 16, id="logical-and"),
        pytest.param(1, 1, 32, 1, 16, id="boolean-bitwise-and"),
        pytest.param(2, 1, 0, 0, 16, id="boolean-bitwise-or"),
        pytest.param(3, 1, 32, 1, 16, id="not-of-or"),
    ],
)
def test_public_native_synccheck_resolves_boolean_collective_participation_exactly(
    mode: int,
    full_flag: int,
    full_bound: int,
    subset_flag: int,
    subset_bound: int,
    native_cache_dir,
):
    clean = _run_collective(mode=mode, flag=full_flag, bound=full_bound, cache_dir=native_cache_dir)
    error = _run_collective(
        mode=mode, flag=subset_flag, bound=subset_bound, cache_dir=native_cache_dir
    )

    clean.require_clean()
    assert clean.verdict == "clean"
    assert clean.findings == []
    clean_native = clean.to_dict()["native"]
    assert clean_native["verdict"] == "clean"
    assert clean_native["findings"] == []
    assert clean_native["incomplete"] == []
    assert clean_native["counterexample"] is None
    assert clean_native["coverage"]["eligible_for_clean"] is True

    assert error.verdict == "error"
    assert [(finding.status, finding.kind) for finding in error.findings] == [
        ("error", "warp_collective_divergence")
    ]
    error_native = error.to_dict()["native"]
    assert error_native["verdict"] == "error"
    assert error_native["incomplete"] == []
    assert error_native["counterexample"] is None
    assert error_native["findings"] == []
    assert error_native["execution_error"]["kind"] == "warp_collective_divergence"
    assert "shfl.sync requires all 32 lanes" in error_native["execution_error"]["message"]
    assert "mask 0x0000ffff" in error_native["execution_error"]["message"]

    assert error_native["engine"]["artifact_key"] == clean_native["engine"]["artifact_key"]
    assert error_native["input"]["digest"] != clean_native["input"]["digest"]


@pytest.mark.parametrize("mode", [0, 1, 3])
def test_public_native_synccheck_skips_uniformly_false_data_guard(mode: int, native_cache_dir):
    report = _run_collective(mode=mode, flag=0, bound=32, cache_dir=native_cache_dir)

    report.require_clean()
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["verdict"] == "clean"
    assert native["incomplete"] == []
