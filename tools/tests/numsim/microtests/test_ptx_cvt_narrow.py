"""Execute each FP4/scaled/stochastic CVT kernel once against all its GPU goldens."""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness import numsim

from tests.numsim.microtests.cases.ptx_cvt_narrow import (
    FORM_SLOTS,
    KERNELS,
    SM100_OUTPUTS,
    make_sm100_arguments,
    narrow_cvt_sm100_forms,
)
from tests.numsim.microtests.cases.ptx_cvt_narrow_goldens import GOLDENS
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_paired_primfunc,
)


@pytest.mark.parametrize("name", KERNELS)
def test_narrow_cvt_kernel_matches_gpu_goldens(name, tmp_path):
    prim_func, make_arguments, outputs = KERNELS[name]
    result = numsim.Engine().run(
        numsim.transpile(prim_func, cache_dir=tmp_path), make_arguments(), outputs=outputs
    )
    for form, (kernel, buffer_name, row) in FORM_SLOTS.items():
        if kernel != name:
            continue
        produced = np.asarray(result.outputs[buffer_name])[row]
        expected = GOLDENS[form]
        # The recorded e2m1x2 golden uses u16, but its PTX destination is b8.
        if produced.dtype == np.uint8:
            expected = expected.astype(np.uint8)
        assert produced.dtype == expected.dtype
        np.testing.assert_array_equal(produced, expected, err_msg=f"cvt {form}")


@NUMSIM_GPU_MARK
def test_narrow_cvt_matches_gpu(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    run_paired_primfunc(
        narrow_cvt_sm100_forms,
        make_sm100_arguments(),
        outputs=SM100_OUTPUTS,
        cache_dir=tmp_path,
    )
