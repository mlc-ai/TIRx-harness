"""SM100 skips invalid MMA descriptors when their instruction predicate is false."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_tcgen_inactive_boundaries import (
    BOUNDARY_CASES,
    inactive_boundary_case,
)
from tests.numsim.support.execution import run_checked

SM100_CASES = tuple(case for case in BOUNDARY_CASES if case[1] != "ti16" and not case[4])


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("case", SM100_CASES, ids=[case[0] for case in SM100_CASES])
def test_inactive_tcgen_boundary_matches_gpu(pytestconfig, case):
    require_numsim_gpu(pytestconfig)
    kernel, inputs, _ = inactive_boundary_case(case)
    cpu = run_checked(kernel, inputs, outputs=("out",))
    np.testing.assert_array_equal(cpu.outputs["out"], inputs["seed"])
    gpu = run_gpu_primfunc(kernel, inputs, outputs=("out",), arch="sm_100a")
    np.testing.assert_array_equal(gpu["out"], inputs["seed"])
