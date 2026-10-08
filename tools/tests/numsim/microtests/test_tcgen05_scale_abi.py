from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.microtests.cases.tcgen05_scale_abi import (
    TCGEN05_SCALE_ABI_CASES,
    Tcgen05ScaleAbiCase,
)
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("case", TCGEN05_SCALE_ABI_CASES, ids=lambda case: case.name)
def test_tcgen05_scale_abi_matches_gpu(
    case: Tcgen05ScaleAbiCase,
    pytestconfig: pytest.Config,
    tmp_path,
):
    require_numsim_gpu(pytestconfig)
    result = run_paired_primfunc(
        case.prim_func,
        case.make_arguments(),
        outputs=("output",),
        cache_dir=tmp_path,
    )

    expected = case.expected_output()
    np.testing.assert_array_equal(result.gpu_outputs["output"], expected)
    np.testing.assert_array_equal(result.numsim_outputs["output"], expected)
