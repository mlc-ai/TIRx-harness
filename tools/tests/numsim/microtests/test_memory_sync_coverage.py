"""Live-device oracles reuse the compact runtime cases, not a modifier census."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_memory_sync_coverage import CASES, half_atomic_sinks


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("kernel,inputs,expected", CASES)
def test_memory_sync_extensions_match_gpu(kernel, inputs, expected, pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    run_paired_primfunc(kernel, inputs, outputs=tuple(expected), cache_dir=tmp_path)


@pytest.mark.numsim_gpu
def test_half_atomic_old_bits_match_gpu(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    for bits in (0x8000, 0x0001, 0x7C01, 0x7E55):
        run_paired_primfunc(
            half_atomic_sinks,
            {
                "cell": np.array([bits], dtype=np.uint16).view(np.float16),
                "out": np.zeros(1, dtype=np.uint16),
            },
            outputs=("out",),
            cache_dir=tmp_path,
        )
