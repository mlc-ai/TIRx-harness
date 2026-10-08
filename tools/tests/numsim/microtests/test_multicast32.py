import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_mbarrier_multicast import MULTICAST_FORMS, multicast_barrier_kernel


@pytest.mark.numsim_gpu
def test_mbarrier_multicast32_matches_gpu(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    import torch

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("32-bit mbarrier multicast requires SM107")
    for form in MULTICAST_FORMS:
        result = run_paired_primfunc(
            multicast_barrier_kernel(form, ctas=2),
            {"out": np.zeros((2, 2), np.uint32)},
            outputs=("out",),
            cache_dir=tmp_path,
            arch="sm_107f",
        )
        np.testing.assert_array_equal(result.gpu_outputs["out"], 1)
