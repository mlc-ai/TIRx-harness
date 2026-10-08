"""SM100 executes swizzled-interleave prefetch without a shared transfer."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import PairedTensorMap, require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_tma_interleave_hints import interleave_hint_case
from tests.numsim.support.execution import run_checked
from tirx_harness import numsim


@pytest.mark.numsim_gpu
@pytest.mark.parametrize(
    "operation,raw,swizzle",
    [("prefetch", False, "32B"), ("prefetch", True, "128B_ATOM_32B")]
    + [(operation, False, "32B") for operation in ("load", "store", "reduce")],
)
def test_interleave_prefetch_gpu_oracle(pytestconfig, operation, raw, swizzle):
    require_numsim_gpu(pytestconfig)
    kernel, inputs, base, metadata = interleave_hint_case(operation, raw=raw, swizzle=swizzle)
    inputs["descriptor"] = numsim.TensorMap(base, **metadata).numpy()
    expected = np.arange(32, dtype=np.uint32) + 0xABC000
    actual = run_checked(kernel, inputs, outputs=("output",))
    np.testing.assert_array_equal(actual.outputs["output"], expected)
    inputs["descriptor"] = PairedTensorMap(base, **metadata)
    gpu = run_gpu_primfunc(kernel, inputs, outputs=("output",), arch="sm_100a")
    np.testing.assert_array_equal(gpu["output"], expected)
