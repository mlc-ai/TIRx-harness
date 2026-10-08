import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_ldmatrix_b8 import (
    LDMATRIX_B8_CASES,
    ldmatrix_b8_case,
    ldmatrix_b8_kernel,
)


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("bits,rows,count", LDMATRIX_B8_CASES)
def test_ldmatrix_b8_matches_gpu(bits, rows, count, pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    inputs, expected = ldmatrix_b8_case(bits, rows, count)
    result = run_paired_primfunc(
        ldmatrix_b8_kernel(bits, rows, count),
        inputs,
        outputs=("output",),
        cache_dir=tmp_path,
    )
    np.testing.assert_array_equal(result.gpu_outputs["output"], expected)


@pytest.mark.numsim_gpu
def test_ldmatrix_s8_s4_matches_gpu(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    import torch

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("signed packed ldmatrix requires SM107")
    for count in (1, 2, 4):
        inputs, expected = ldmatrix_b8_case(4, 8, count, signed=True)
        result = run_paired_primfunc(
            ldmatrix_b8_kernel(4, 8, count, signed=True),
            inputs,
            outputs=("output",),
            cache_dir=tmp_path,
            arch="sm_107a",
        )
        np.testing.assert_array_equal(result.gpu_outputs["output"], expected)
