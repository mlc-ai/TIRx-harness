"""The locality-domain special registers on the GPU obey the PTX placement rules."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_gpu_primfunc
from tests.numsim.runtime.test_locality_domain import _locality_kernel


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("cluster", (1, 2))
def test_locality_domain_registers_on_gpu(pytestconfig, cluster):
    # PTX ISA 9.5 needs NVRTC 13.5.
    require_numsim_gpu(pytestconfig, min_nvrtc=(13, 5))
    grid = 256
    inputs = {
        "domain": np.full((grid, 64), -1, np.int32),
        "supplemental": np.full((grid, 64), -1, np.int32),
    }
    device = run_gpu_primfunc(
        _locality_kernel(grid, cluster), inputs, outputs=("domain", "supplemental")
    )
    domain, supplemental = device["domain"], device["supplemental"]
    clusters = domain.reshape(grid // cluster, cluster * 64)
    assert (clusters == clusters[:, :1]).all()
    assert domain.min() >= 0
    try:
        from cuda.bindings import driver

        driver.cuInit(0)
        status, count = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, 0
        )
    except (ImportError, AttributeError):
        status = None
    if status is not None and status == driver.CUresult.CUDA_SUCCESS:
        assert domain.max() < count
    assert (supplemental == supplemental[:, :1]).all()
    assert set(np.unique(supplemental)) <= {0, 1}
