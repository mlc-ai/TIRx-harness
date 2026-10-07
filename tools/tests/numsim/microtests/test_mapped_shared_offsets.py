"""GPU oracle for remote shared-memory address arithmetic."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_non_tensor_bulk_forms import mapped_st_async_case


@NUMSIM_GPU_MARK
@pytest.mark.parametrize(
    "offset_form", ["instruction", "constant", "dynamic", "reversed", "subtracted", "wrapped"]
)
def test_mapped_shared_offsets_match_gpu(pytestconfig, tmp_path, offset_form):
    require_numsim_gpu(pytestconfig)
    run_paired_primfunc(
        mapped_st_async_case(offset_form),
        {"output": np.zeros(8, dtype=np.uint32)},
        outputs=("output",),
        cache_dir=tmp_path,
    )
