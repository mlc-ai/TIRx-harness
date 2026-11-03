from __future__ import annotations

import numpy as np
import pytest

from tirx_harness.numsim import checkers
from tvm_tirx.script import tirx as T


@T.function
def predicated_global_reduction(output: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    T.ptx["red.global.add.f32"](
        output.ptr_to([lane]),
        T.float32(1.0),
        pred=lane < 16,
    )


@pytest.mark.parametrize(
    "run",
    [checkers._run_synccheck, checkers._run_racecheck],
    ids=["synccheck", "racecheck"],
)
def test_checker_models_predicated_global_reduction(run, tmp_path):
    report = run(
        predicated_global_reduction,
        inputs={"output": np.zeros(32, dtype=np.float32)},
        cache_dir=tmp_path,
    )

    report.require_clean()
    if run is checkers._run_racecheck:
        assert report.to_dict()["native"]["access_count"] == 1
