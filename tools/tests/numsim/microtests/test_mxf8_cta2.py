"""Hardware oracle for MXF8 M=128, cta_group=2, including both TMEM banks."""

import numpy as np

from tests.numsim.microtests.cases.mxf8_cta2 import make_arguments, raw_mxf8_cta2_m128, reference
from tests.numsim.microtests.harness import (
    NUMSIM_GPU_MARK,
    require_numsim_gpu,
    run_gpu_primfunc,
    run_paired_primfunc,
)


@NUMSIM_GPU_MARK
def test_raw_mxf8_cta2_m128_matches_gpu(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    arguments = make_arguments()
    result = run_paired_primfunc(
        raw_mxf8_cta2_m128, arguments, outputs=("output",), cache_dir=tmp_path
    )
    np.testing.assert_array_equal(result.gpu_outputs["output"], reference(arguments))


@NUMSIM_GPU_MARK
def test_raw_mxf8_cta2_matching_nan_scales_are_valid_on_gpu(pytestconfig):
    require_numsim_gpu(pytestconfig)
    arguments = make_arguments()
    arguments["scale_b"].fill(np.uint32(0xFFFFFFFF))
    arguments["output"].fill(0)
    output = run_gpu_primfunc(raw_mxf8_cta2_m128, arguments, outputs=("output",), arch="sm_100a")[
        "output"
    ]
    # PTX requires NaN propagation; it does not promise a particular NaN payload.
    assert np.isnan(output).all()
