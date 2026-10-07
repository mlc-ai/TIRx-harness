"""Same-case hardware differential for the SM107-only TMA override family."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import (
    PairedTensorMap,
    require_numsim_gpu,
    run_paired_primfunc,
)
from tests.numsim.runtime.test_tma_overrides import (
    OVERRIDE_CASES,
    multiissuer_override_case,
    override_kernel,
)


@pytest.mark.numsim_gpu
def test_tma_overrides_match_gpu(pytestconfig, tmp_path):
    require_numsim_gpu(pytestconfig)
    import torch

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("TMA override instructions require an SM107 GPU")
    for route, form in OVERRIDE_CASES:
        rank = 2 if "stride" in form or form == "address" else 1
        inputs = {
            "input_map": PairedTensorMap(
                array=np.full((8,) * rank, -10, np.float32),
                global_shape=(8,) * rank,
                global_strides=() if rank == 1 else (32,),
                box_shape=(4,) if rank == 1 else (4, 2),
                element_strides=(1,) * rank,
            ),
            "replacement": np.arange(32772, dtype=np.float32),
            "output": np.zeros(4 if rank == 1 else 8, np.float32),
        }
        outputs = ("output", "replacement") if route in {"s2g", "reduce"} else ("output",)
        run_paired_primfunc(
            override_kernel(route, form),
            inputs,
            outputs=outputs,
            cache_dir=tmp_path,
            arch="sm_107f",
        )
        if route.startswith("g2"):
            for matched in (False, True):
                reporting = {
                    **inputs,
                    "replacement": np.arange(32772, dtype=np.float32),
                    "output": np.zeros(inputs["output"].size + 2, np.float32),
                }
                if matched:
                    reporting["replacement"][4:8] = np.float32(-0.0)
                result = run_paired_primfunc(
                    override_kernel(route, form, report="validity::per_16bytes::80000000"),
                    reporting,
                    outputs=("output",),
                    cache_dir=tmp_path,
                    arch="sm_107f",
                )
                np.testing.assert_array_equal(result.gpu_outputs["output"][-2:], [1, int(matched)])
    for route in ("load", "store", "reduce"):
        for form, issuer in (("stride_b8", -1), ("stride_b16", 1)):
            kernel, inputs, expected, shared = multiissuer_override_case(route, form, issuer)
            inputs["input_map"] = PairedTensorMap(
                array=np.full((8, 8), -10, np.float32),
                global_shape=(8, 8),
                global_strides=(32,),
                box_shape=(4, 2),
                element_strides=(1, 1),
            )
            result = run_paired_primfunc(
                kernel,
                inputs,
                outputs=("output", "replacement"),
                cache_dir=tmp_path,
                arch="sm_107f",
            )
            np.testing.assert_array_equal(result.gpu_outputs["replacement"], expected)
            np.testing.assert_array_equal(result.gpu_outputs["output"], shared)
