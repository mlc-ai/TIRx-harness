"""Sparse MXF8 uses one row scale and the ordinary 2:4 metadata contract."""

import numpy as np
import pytest

from tests.numsim.runtime.test_tcgen05_sparse_narrow import sparse_narrow_case
from tests.numsim.support.execution import assert_rejected, run_checked

SPARSE_BLOCK_CASES = [
    (1, 128, True, 1, 3, False),
    (2, 256, False, 3, 4, False),
    (2, 256, True, 4, 5, False),
    (2, 256, False, 5, 0, False),
    (1, 128, False, 0, 1, False, True, True),
]


@pytest.mark.parametrize("case", SPARSE_BLOCK_CASES)
def test_sparse_block_scales_and_metadata(case, tmp_path):
    kernel, args, expected = sparse_narrow_case(*case, block_scale=True)
    result = run_checked(kernel, args, cache_dir=tmp_path)
    np.testing.assert_array_equal(result.outputs["out"], expected)


def test_sparse_block_descriptor_boundaries(tmp_path):
    kernel, args, _ = sparse_narrow_case(2, 128, False, 0, 1, False, block_scale=True)
    assert_rejected(kernel, args, "requires M=256", cache_dir=tmp_path)
