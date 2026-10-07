import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_mbarrier_drop import DROP_FORMS, drop_kernel


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("form", DROP_FORMS)
def test_mbarrier_drop_matches_gpu(form, pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    result = run_paired_primfunc(
        drop_kernel(form), {"out": np.zeros(3, np.uint32)}, outputs=("out",), cache_dir=tmp_path
    )
    np.testing.assert_array_equal(result.gpu_outputs["out"], [1, 1, 1])
