from __future__ import annotations

import pytest

from tests.numsim.microtests.cases.tcgen05_transfer_semantics import (
    TCGEN05_TRANSFER_CASES,
    Tcgen05TransferCase,
)
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("case", TCGEN05_TRANSFER_CASES, ids=lambda case: case.name)
def test_tcgen05_transfer_matches_gpu(
    case: Tcgen05TransferCase,
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
