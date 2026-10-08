"""Hardware-gated register predicate check; SM100 is not an SM107 oracle."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_sm107_register_predicates import (
    sm107_register_predicate_expected,
    sm107_register_predicate_kernel,
)


@pytest.mark.numsim_gpu
def test_sm107_register_predicates_gpu(pytestconfig):
    require_numsim_gpu(pytestconfig)
    import torch

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("packed set and register sparse compression require SM107")
    for preserve in (False, True):
        kernel = sm107_register_predicate_kernel(preserve)
        for mask in (0, 0x80000000, 0xAAAAAAAA, 0xFFFFFFFF):
            inputs = {"output": np.zeros((5, 32), np.uint32), "selected": mask}
            device = run_gpu_primfunc(kernel, inputs, outputs=("output",), arch="sm_107a")
            expected = sm107_register_predicate_expected(mask, preserve)
            # Write-only inactive GPU registers are undefined, not necessarily zero.
            lanes = [lane for lane in range(32) if preserve or mask & (1 << lane)]
            np.testing.assert_array_equal(device["output"][:, lanes], expected[:, lanes])
