"""Barrier arrival counting for a partial warp on the GPU and in NumSim.

[VERIFY] NumSim credits a partial warp's absent lanes once its existing lanes
have all arrived, because PTX counts `bar` arrivals per warp. A hang or a
mismatch here falsifies that model.
"""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_partial_warp_cta import (
    named_barrier_with_partial_warp,
    producer_consumer_with_partial_warp,
)
from tirx_harness import numsim


@pytest.mark.numsim_gpu
@pytest.mark.parametrize(
    "kernel, size",
    [(named_barrier_with_partial_warp, 33), (producer_consumer_with_partial_warp, 16)],
)
def test_partial_warp_barrier_counts_match_gpu(pytestconfig, tmp_path, kernel, size):
    # bar.sync arrival counting is not SM100-specific; SM90 is a valid witness.
    require_numsim_gpu(pytestconfig, min_sm_major=9)
    import torch

    major, minor = torch.cuda.get_device_capability()
    inputs = {"output": np.zeros(size, dtype=np.int32)}
    device = run_gpu_primfunc(kernel, inputs, outputs=("output",), arch=f"sm_{major}{minor}a")
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    simulated = numsim.Engine().run(module, inputs, outputs=("output",))
    np.testing.assert_array_equal(simulated.outputs["output"], device["output"])
