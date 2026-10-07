from __future__ import annotations

import numpy as np
import pytest

from tirx_harness.numsim import checkers
from tvm.script import tirx as T


@T.prim_func
def runtime_grid(num_ctas: T.int32, value: T.int32, output: T.Buffer((4,), "int32")):
    T.device_entry()
    cta = T.cta_id([num_ctas])
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    if lane == 0:
        output[cta] = value


@pytest.mark.parametrize(
    "run",
    [checkers._run_synccheck, checkers._run_racecheck],
    ids=["synccheck", "racecheck"],
)
def test_checker_binds_runtime_scalar_launch_extent(run, tmp_path):
    report = run(
        runtime_grid,
        inputs={
            "num_ctas": np.int32(4),
            "value": np.int32(7),
            "output": np.zeros(4, dtype=np.int32),
        },
        cache_dir=tmp_path,
    )

    report.require_clean()
    native = report.to_dict()["native"]
    assert native["phase"]["topology"]["warp_count"] == 4
    assert native["input"]["bindings"] == ["num_ctas", "output", "value"]
    assert native["input"]["scalars"] == {
        "num_ctas": {"dtype": "int32", "value": {"kind": "integer", "decimal": "4"}},
        "value": {"dtype": "int32", "value": {"kind": "integer", "decimal": "7"}},
    }


@pytest.mark.parametrize(
    "run",
    [checkers._run_synccheck, checkers._run_racecheck],
    ids=["synccheck", "racecheck"],
)
def test_checker_artifact_identity_tracks_launch_scalars_only(run, tmp_path):
    def execute(num_ctas: int, value: int):
        report = run(
            runtime_grid,
            inputs={
                "num_ctas": np.int32(num_ctas),
                "value": np.int32(value),
                "output": np.zeros(4, dtype=np.int32),
            },
            cache_dir=tmp_path,
        )
        report.require_clean()
        return report.to_dict()["native"]

    baseline = execute(2, 3)
    data_change = execute(2, 9)
    launch_change = execute(4, 9)

    assert data_change["engine"]["artifact_key"] == baseline["engine"]["artifact_key"]
    assert data_change["input"]["digest"] != baseline["input"]["digest"]
    assert launch_change["engine"]["artifact_key"] != baseline["engine"]["artifact_key"]
    assert launch_change["phase"]["topology"]["warp_count"] == 4
