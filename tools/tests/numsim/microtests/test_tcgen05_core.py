from __future__ import annotations

import pytest

from tests.numsim.microtests.cases.tcgen05_lifecycle_ldst import (
    TCGEN05_CORE_CASES,
    Tcgen05CoreCase,
)
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("case", TCGEN05_CORE_CASES, ids=lambda case: case.name)
def test_tcgen05_core_matches_gpu(
    case: Tcgen05CoreCase,
    pytestconfig: pytest.Config,
    tmp_path,
):
    require_numsim_gpu(pytestconfig)
    run_paired_primfunc(
        case.prim_func,
        case.make_arguments(),
        outputs=case.outputs,
        cache_dir=tmp_path,
    )
