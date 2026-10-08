from __future__ import annotations

import pytest

from tests.numsim.microtests.cases.ptx_numeric import PTX_NUMERIC_CASES, PtxNumericCase
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("case", PTX_NUMERIC_CASES, ids=lambda case: case.name)
def test_ptx_numeric_matches_gpu(
    case: PtxNumericCase,
    pytestconfig: pytest.Config,
    tmp_path,
):
    require_numsim_gpu(pytestconfig)
    run_paired_primfunc(
        case.prim_func,
        case.make_arguments(),
        outputs=case.outputs,
        cache_dir=tmp_path,
        max_ulp=case.max_ulp,
    )
