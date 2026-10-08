"""Live-GPU commit guards with per-form hardware and compiler requirements."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_tcgen_commit_predicates import commit_inputs, commit_kernel


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("multicast", [None, "", "::16b", "::32b"])
def test_tcgen_commit_predicates_gpu(pytestconfig, restricted, multicast):
    require_numsim_gpu(pytestconfig)
    import torch

    sm107 = restricted or multicast == "::32b"
    if sm107 and torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("restricted shared-A completion and 32-bit multicast masks require SM107")
    if restricted or multicast in ("::16b", "::32b"):
        # These qualifiers were introduced in PTX 9.4. TVM compiles with NVRTC,
        # whose version can differ from both nvcc and torch.version.cuda.
        from cuda.bindings import nvrtc

        status, major, minor = nvrtc.nvrtcVersion()
        assert status == nvrtc.nvrtcResult.NVRTC_SUCCESS
        if (major, minor) < (13, 4):
            pytest.skip("explicit mask widths and restricted commits require NVRTC 13.4 (PTX 9.4)")
    kernel = commit_kernel(restricted, multicast)
    for enabled, selected in ((0, 0), (1, 0), (1, 31)):
        inputs, expected = commit_inputs(enabled, selected)
        device = run_gpu_primfunc(
            kernel, inputs, outputs=("output",), arch="sm_107a" if sm107 else "sm_100a"
        )
        np.testing.assert_array_equal(device["output"], expected)
