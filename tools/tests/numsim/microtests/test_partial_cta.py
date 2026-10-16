"""Non-warp-aligned launches execute identical scalar paths on GPU and NumSim."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)
from tests.numsim.support.partial_cta import partial_cta


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("threads", (1, 17, 31, 33, 65))
def test_partial_cta_matches_gpu(threads, pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    from tirx_kernels.runner import cuda_target

    capacity = ((threads + 31) // 32) * 32
    run_paired_primfunc(
        partial_cta(threads),
        {"output": np.full((2, capacity, 3), -1, np.int32)},
        outputs=["output"], cache_dir=tmp_path, arch=cuda_target().arch,
    )
