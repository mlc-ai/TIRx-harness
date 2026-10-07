"""Live-device oracles for the same small memory/barrier contract cases."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_memory_coverage_next import CASES, memory_case


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("kind", CASES)
def test_next_memory_contracts_match_gpu(kind, pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    kernel, inputs, expected = memory_case(kind, paired=True)
    result = run_paired_primfunc(kernel, inputs, outputs=("out",), cache_dir=tmp_path)
    np.testing.assert_array_equal(result.gpu_outputs["out"], expected)
