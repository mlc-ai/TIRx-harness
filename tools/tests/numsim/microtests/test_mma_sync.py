from __future__ import annotations

import pytest

from tests.numsim.microtests.cases.mma_sync import MMA_SYNC_CASES, MmaSyncCase
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("case", MMA_SYNC_CASES, ids=lambda case: case.name)
def test_mma_sync_matches_gpu(
    case: MmaSyncCase,
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
