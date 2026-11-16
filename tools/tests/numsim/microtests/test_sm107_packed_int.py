"""Hardware-gated packed integer check; SM100 is not an SM107 oracle."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_ptx_packed_int_sm107f import sm107_packed_int_case


@pytest.mark.numsim_gpu
def test_sm107_packed_integer_forms_gpu(pytestconfig):
    require_numsim_gpu(pytestconfig)
    import torch

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("packed .u8x4/.s8x4 integer forms and add.sat.u32 require SM107")
    kernel, inputs, expected, spellings = sm107_packed_int_case()
    device = run_gpu_primfunc(kernel, inputs, outputs=("output",), arch="sm_107a")
    for row, spelling in enumerate(spellings * 2):
        np.testing.assert_array_equal(
            device["output"][row], expected[row], err_msg=f"row {row}: {spelling}"
        )
