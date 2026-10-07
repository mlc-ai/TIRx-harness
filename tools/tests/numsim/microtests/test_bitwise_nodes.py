"""Independent GPU oracle for TVM's primitive bitwise and shift nodes."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_bitwise_nodes import DTYPES, bitwise_case, boolean_case
from tests.numsim.support.execution import run_checked


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_bitwise_nodes_match_gpu(dtype, pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    kernel, inputs, expected = bitwise_case(dtype)
    result = run_checked(kernel, inputs, outputs=("output",), cache_dir=tmp_path)
    gpu = run_gpu_primfunc(kernel, inputs, outputs=("output",), arch="sm_100a")
    np.testing.assert_array_equal(result.outputs["output"], expected)
    np.testing.assert_array_equal(gpu["output"], expected)
    np.testing.assert_array_equal(gpu["output"], result.outputs["output"])


@pytest.mark.numsim_gpu
def test_boolean_bitwise_nodes_match_gpu(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    kernel, inputs, expected = boolean_case()
    result = run_checked(kernel, inputs, outputs=("output",), cache_dir=tmp_path)
    gpu = run_gpu_primfunc(kernel, inputs, outputs=("output",), arch="sm_100a")
    np.testing.assert_array_equal(result.outputs["output"], expected)
    np.testing.assert_array_equal(gpu["output"], expected)
    np.testing.assert_array_equal(gpu["output"], result.outputs["output"])
