"""Same-case NumSim/GPU differential for standalone tcgen05.shift on SM107.

Each case reuses a runtime test's kernel and inputs. The runtime tests pin
NumSim to an independent oracle; this file pins the same kernels to SM107
hardware, so the oracle, NumSim and the GPU must all agree.
"""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_tcgen05_shift import shift_inputs, shift_kernel, shift_oracle

CASES = [
    *(
        pytest.param(cta_group, column, 0, None, 1, id=f"cta{cta_group}-col{column}")
        for cta_group in (1, 2)
        for column in (0, 4, 24)
    ),
    *(
        pytest.param(cta_group, 8, lane, None, 1, id=f"cta{cta_group}-lane{lane}")
        for cta_group in (1, 2)
        for lane in (32, 96)
    ),
    *(
        pytest.param(1, 0, 0, True, issue, id=f"cta1-col0-predicate-issue{issue}")
        for issue in (0, 1)
    ),
]


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("cta_group,column,lane,predicate,issue", CASES)
def test_sm107_tcgen05_shift_matches_gpu(
    cta_group, column, lane, predicate, issue, pytestconfig, tmp_path
):
    require_numsim_gpu(pytestconfig)
    import torch

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("this differential targets an SM107 GPU")
    inputs = shift_inputs(cta_group, issue)
    result = run_paired_primfunc(
        shift_kernel(cta_group, column=column, lane=lane, predicate=predicate),
        inputs,
        outputs=("out",),
        cache_dir=tmp_path,
        arch="sm_107a",
    )
    expected = shift_oracle(inputs["seed"], column) if issue else inputs["seed"]
    np.testing.assert_array_equal(result.gpu_outputs["out"], expected)
