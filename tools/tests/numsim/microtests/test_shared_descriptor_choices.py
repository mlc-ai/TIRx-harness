"""GPU oracle for selected and assembled shared matrix descriptors."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_shared_descriptor_choices import descriptor_choice_case


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("form", ["select", "compose"])
def test_shared_descriptor_choices_match_gpu(pytestconfig, tmp_path, form):
    require_numsim_gpu(pytestconfig)
    source = np.arange(128, dtype=np.uint32).reshape(32, 4) + 1
    result = run_paired_primfunc(
        descriptor_choice_case(form),
        {"source": source, "output": np.full((4, 32, 4), 0xDEADBEEF, dtype=np.uint32)},
        outputs=("output",), cache_dir=tmp_path,
    )
    np.testing.assert_array_equal(result.gpu_outputs["output"], np.tile(source, (4, 1, 1)))
