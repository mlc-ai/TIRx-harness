"""Same-case hardware differential for the SM107-only PTX 9.4 ``cvt`` families."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import require_numsim_gpu, run_paired_primfunc
from tests.numsim.runtime.test_ptx_cvt_94 import ptx94_cvt_forms, ptx94_pzo_forms
from tests.numsim.runtime.test_ptx_cvt_pzo import pzo_conversions


def _require_sm107(pytestconfig):
    require_numsim_gpu(pytestconfig, min_nvrtc=(13, 4))
    import torch

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("PTX 9.4 cvt families require an SM107 GPU")


@pytest.mark.numsim_gpu
def test_ptx94_cvt_families_match_gpu(pytestconfig, tmp_path):
    _require_sm107(pytestconfig)
    run_paired_primfunc(
        ptx94_cvt_forms,
        {
            "scale": np.array([128], np.uint8),
            "output_u16": np.zeros((32, 6), np.uint16),
            "output_u32": np.zeros((32, 2), np.uint32),
        },
        outputs=("output_u16", "output_u32"),
        cache_dir=tmp_path,
        arch="sm_107f",
    )
    run_paired_primfunc(
        ptx94_pzo_forms,
        {"scale": np.array([127], np.uint8), "output": np.zeros((32, 4), np.uint16)},
        outputs=("output",),
        cache_dir=tmp_path,
        arch="sm_107f",
    )


@pytest.mark.numsim_gpu
def test_pzo_conversions_match_gpu(pytestconfig, tmp_path):
    _require_sm107(pytestconfig)
    tiny = np.asarray([0x8000_0001], np.uint32).view(np.float32)[0]
    source = np.tile(
        np.asarray([-0.0, tiny, -1.0, 0.0, 1.0, -np.inf, np.inf, np.nan], np.float32),
        4,
    )
    run_paired_primfunc(
        pzo_conversions,
        {
            "source": source,
            "low": np.roll(source, 1),
            "f16": np.zeros(32, np.uint16),
            "bf16": np.zeros(32, np.uint16),
            "tf32": np.zeros(32, np.uint32),
            "packed_f16": np.zeros(32, np.uint32),
            "packed_bf16": np.zeros(32, np.uint32),
        },
        outputs=("f16", "bf16", "tf32", "packed_f16", "packed_bf16"),
        cache_dir=tmp_path,
        arch="sm_107f",
    )
