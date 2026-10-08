"""The GPU oracle's argument ABI does not require NumSim instruction support."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import PairedTensorMap, require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_tma_atomicity import atomicity_case
from tests.numsim.support.execution import run_checked
from tirx_harness import numsim
from tirx_harness.numsim.transpiler import frontend


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("raw_descriptor", [False, True])
def test_gpu_tensor_map_abi_without_instruction_analysis(pytestconfig, monkeypatch, raw_descriptor):
    require_numsim_gpu(pytestconfig)
    kernel, inputs, base, metadata, expected = atomicity_case(
        32,
        restore_swizzle=raw_descriptor,
    )
    inputs["descriptor"] = numsim.TensorMap(base, **metadata).numpy()
    result = run_checked(kernel, inputs, outputs=("output",))
    np.testing.assert_array_equal(result.outputs["output"], expected)

    def unavailable(*args, **kwargs):
        raise AssertionError("GPU argument binding must not analyze NumSim instructions")

    monkeypatch.setattr(frontend, "analyze", unavailable)
    inputs["descriptor"] = PairedTensorMap(base, **metadata)
    gpu = run_gpu_primfunc(kernel, inputs, outputs=("output",), arch="sm_100a")
    np.testing.assert_array_equal(gpu["output"], expected)
