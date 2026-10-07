import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_bar_reduce import (
    BAR_REDUCE_FORMS,
    bar_reduce_case,
)


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("name,op,counted", BAR_REDUCE_FORMS)
def test_bar_reduce_matches_gpu(name, op, counted, pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    kernel, inputs, expected = bar_reduce_case(name, op, counted)
    result = run_paired_primfunc(kernel, inputs, outputs=("out",), cache_dir=tmp_path)
    np.testing.assert_array_equal(result.gpu_outputs["out"], expected)
