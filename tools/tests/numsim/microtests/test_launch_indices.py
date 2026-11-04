"""A multidimensional grid and cluster on the GPU and in NumSim."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_launch_indices import (
    GRID_3D,
    three_dimensional_expected,
    three_dimensional_launch,
)
from tirx_harness import numsim


@pytest.mark.numsim_gpu
def test_three_dimensional_launch_coordinates_match_gpu(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    inputs = {"output": np.full((*GRID_3D[::-1], 16), -1, dtype=np.int32)}
    device = run_gpu_primfunc(three_dimensional_launch, inputs, outputs=("output",), arch="sm_100a")
    module = numsim.transpile(three_dimensional_launch, cache_dir=tmp_path)
    simulated = numsim.Engine().run(module, inputs, outputs=("output",))

    np.testing.assert_array_equal(device["output"], three_dimensional_expected())
    np.testing.assert_array_equal(simulated.outputs["output"], device["output"])
