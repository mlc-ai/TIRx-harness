from __future__ import annotations

import pytest

from tests.numsim.microtests.cases.matrix_memory import MATRIX_MEMORY_CASES, MatrixMemoryCase
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("case", MATRIX_MEMORY_CASES, ids=lambda case: case.name)
def test_matrix_memory_matches_gpu(
    case: MatrixMemoryCase,
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
