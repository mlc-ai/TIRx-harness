import numpy as np
import pytest

from tests.numsim.microtests.cases.tcgen05_fp8_ts import fp8_ts_arguments, fp8_ts_kernel
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu, run_paired_primfunc


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("m,d_f16", [(64, False), (128, True)])
def test_fp8_tmem_a_matches_gpu_and_reference(m, d_f16, pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    arguments, expected = fp8_ts_arguments(m, d_f16)
    result = run_paired_primfunc(
        fp8_ts_kernel(m, d_f16),
        arguments,
        outputs=("output",),
        cache_dir=tmp_path,
    )
    np.testing.assert_array_equal(result.gpu_outputs["output"], expected)
    np.testing.assert_array_equal(result.numsim_outputs["output"], expected)
