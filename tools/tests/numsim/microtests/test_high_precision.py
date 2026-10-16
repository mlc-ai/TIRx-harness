from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_gpu_primfunc,
)
from tests.numsim.runtime.test_high_precision import cancellation


pytestmark = NUMSIM_GPU_MARK


def test_high_precision_separates_gpu_rounding_from_real_sum(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    source = np.tile(np.array([2**24, 1, -(2**24)], np.float32), (32, 1))
    inputs = {
        "source": source,
        "scratch": np.zeros(32, np.float32),
        "output": np.zeros(32, np.float32),
    }
    gpu = run_gpu_primfunc(cancellation, inputs, outputs=("output",), arch="sm_100a")
    native = numsim.Engine().run(
        numsim.transpile(cancellation, cache_dir=tmp_path), inputs, outputs=("output",)
    )
    high = numsim.Engine().run(
        numsim.transpile(cancellation, precision="high", cache_dir=tmp_path),
        inputs,
        outputs=("output",),
    )
    np.testing.assert_array_equal(
        native.outputs["output"].view(np.uint32), gpu["output"].view(np.uint32)
    )
    np.testing.assert_array_equal(gpu["output"], 0)
    np.testing.assert_array_equal(high.outputs["output"], source.astype(np.float64).sum(axis=1))
    np.testing.assert_array_equal(high.outputs["output"], 1)
