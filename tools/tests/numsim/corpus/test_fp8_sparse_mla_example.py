from __future__ import annotations

import numpy as np
import pytest

from examples.numsim.fp8_sparse_mla import make_case, run_example
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu


@pytest.mark.parametrize("seed", [7, 19])
def test_fp8_sparse_mla_distinguishes_quantization_from_wrong_scale(seed):
    case = make_case(queries=3, heads=2, selected=97, dimension=64, seed=seed)
    assert np.any(case.indices < 0)
    assert case.lengths[-1] == 0
    records = {record["mode"]: record for record in run_example(case, workers=2)}
    assert 0.005 < records["fp8"]["relative_l2"] < 0.05
    assert records["fp8"]["quantized_reference_max_abs"] < 2e-6
    assert not records["fp8"]["passes_strict"]
    assert records["fp32"]["passes_strict"]
    assert records["high"]["passes_strict"]
    assert records["high"]["max_abs"] < 1e-10
    assert records["wrong_fp8"]["passes_loose_5_percent_l2"]
    assert not records["wrong_high"]["passes_strict"]
    assert records["wrong_high"]["relative_l2"] == pytest.approx(0.01, abs=1e-7)


@NUMSIM_GPU_MARK
def test_fp8_sparse_mla_example_and_production_on_gpu(pytestconfig):
    require_numsim_gpu(pytestconfig)
    import torch

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("the production sparse MLA example targets B200/SM100a")
    records = {record["mode"]: record for record in run_example(make_case(), gpu=True)}
    assert records["fp8"]["gpu_vs_numsim_max_abs"] < 2e-6
    assert records["fp32"]["gpu_vs_numsim_max_abs"] < 2e-6
    assert records["production_fp8_gpu"]["passes_production_tolerances"]
    assert 0.005 < records["production_fp8_gpu"]["relative_l2"] < 0.05
