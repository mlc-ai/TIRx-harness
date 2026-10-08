"""The new CTA-pair accumulator shape must agree with an independent matrix product."""

import numpy as np
import pytest

from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.errors import NumSimExecutionError
from tests.numsim.microtests.cases.mxf8_cta2 import make_arguments, raw_mxf8_cta2_m128, reference
from tests.numsim.microtests.harness import PairedBuffer


@pytest.mark.parametrize("nan_scales", (False, True), ids=("finite", "nan"))
def test_raw_mxf8_cta2_m128_checks_ownership_and_values(nan_scales, tmp_path):
    arguments = make_arguments()
    if nan_scales:
        arguments["scale_b"].fill(np.uint32(0xFFFFFFFF))
        arguments["output"].fill(0)
    bindings = {
        name: value.array if isinstance(value, PairedBuffer) else value
        for name, value in arguments.items()
    }
    for checker in (synccheck, racecheck):
        checker(raw_mxf8_cta2_m128, bindings).require_clean()
    module = numsim.transpile(raw_mxf8_cta2_m128, cache_dir=tmp_path)
    result = numsim.Engine().run(module, bindings)
    np.testing.assert_array_equal(result.outputs["output"], reference(arguments))


@pytest.mark.parametrize("fault", ("missing_replicas", "conflicting_ctas"))
def test_raw_mxf8_cta2_rejects_invalid_scale_copies(fault, tmp_path):
    from tests.numsim.microtests.cases.mxf8_cta2 import _raw_mxf8_cta2_m128

    arguments = make_arguments()
    if fault == "missing_replicas":
        kernel = _raw_mxf8_cta2_m128.specialize(REPLICAS=1)
        diagnostic = "uninitialized|replicas disagree"
    else:
        kernel = raw_mxf8_cta2_m128
        # Every local replica is initialized, but the second CTA's copy is stale.
        arguments["scale_b"][1] = np.uint32(0x7F7F7F7F)
        diagnostic = "SFB copies disagree across CTAs"
    bindings = {
        name: value.array if isinstance(value, PairedBuffer) else value
        for name, value in arguments.items()
    }
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    with pytest.raises(NumSimExecutionError, match=diagnostic):
        numsim.Engine().run(module, bindings)
    for checker in (synccheck, racecheck):
        report = checker(kernel, bindings)
        assert report.verdict != "clean", report.to_dict()
