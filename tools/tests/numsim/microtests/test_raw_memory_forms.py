from __future__ import annotations

import pytest

from tests.numsim.microtests.cases.raw_memory_forms import (
    RAW_MEMORY_FORM_CASES,
    RawMemoryFormCase,
)
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)


@NUMSIM_GPU_MARK
@pytest.mark.parametrize("case", RAW_MEMORY_FORM_CASES, ids=lambda case: case.name)
def test_raw_memory_forms_match_gpu(
    case: RawMemoryFormCase,
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
