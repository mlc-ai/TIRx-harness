"""GPU oracle for null predicates on backed and integer addresses."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_pointer_null import backed_pointer_null_predicates, numeric_null_handle


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("numeric", [False, True])
def test_pointer_null_predicates_match_gpu(pytestconfig, tmp_path, numeric):
    require_numsim_gpu(pytestconfig)
    inputs = {"output": np.full((32, 3), -1, dtype=np.int32)}
    expected = np.zeros((32, 3), dtype=np.int32)
    if numeric:
        addresses = np.resize(np.array([0, 1, 0x1000, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64), 32)
        inputs = {"addresses": addresses, "output": np.full(32, -1, dtype=np.int32)}
        expected = (addresses == 0).astype(np.int32)
    result = run_paired_primfunc(
        numeric_null_handle if numeric else backed_pointer_null_predicates,
        inputs,
        outputs=("output",),
        cache_dir=tmp_path,
    )
    np.testing.assert_array_equal(result.gpu_outputs["output"], expected)
