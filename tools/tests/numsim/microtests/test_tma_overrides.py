"""Same-case hardware differential for the SM107-only TMA override family."""

import numpy as np
import pytest

from tests.numsim.microtests.harness import (
    PairedTensorMap,
    require_numsim_gpu,
    run_gpu_primfunc,
    run_paired_primfunc,
)
from tests.numsim.runtime.test_tma_overrides import (
    CLIP_BOX,
    CLIP_OPERANDS,
    CLIP_PITCH,
    OVERRIDE_CASES,
    clip_inputs,
    clip_kernel,
    clipped_expectation,
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
    for route, form in (
        (route, form)
        for route in ("g2cta", "s2g", "reduce")
        for form in ("dim_b8", "dim_b16", "stride_b8", "stride_b16")
    ):
        rank = 2 if "stride" in form else 1
        for dimension in (1, 2, 3):
            run_paired_primfunc(
                override_kernel(route, form, dimension=dimension),
                {
                    "input_map": PairedTensorMap(
                        array=np.full((8,) * rank, -10, np.float32),
                        global_shape=(8,) * rank,
                        global_strides=() if rank == 1 else (32,),
                        box_shape=(4,) if rank == 1 else (4, 2),
                        element_strides=(1,) * rank,
                    ),
                    "replacement": np.arange(32772, dtype=np.float32),
                    "output": np.zeros(4 if rank == 1 else 8, np.float32),
                },
                outputs=("output", "replacement"),
                cache_dir=tmp_path,
                arch="sm_107f",
            )
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


def require_sm107(pytestconfig):
    require_numsim_gpu(pytestconfig)
    import torch

    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("TMA override instructions require an SM107 GPU")


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("route", ("load", "s2g", "reduce"))
def test_tma_override_dimension_bounds_match_gpu(route, pytestconfig, tmp_path):
    """Loads bound both axes per element; stores and reductions bound rows per
    element and the inner axis at 16-byte granularity."""

    require_sm107(pytestconfig)
    for d0, d1 in CLIP_OPERANDS:
        inputs = clip_inputs(d0, d1)
        inputs["m"] = PairedTensorMap(**inputs["m"])
        result = run_paired_primfunc(
            clip_kernel(route),
            inputs,
            outputs=("y", "out"),
            cache_dir=tmp_path,
            arch="sm_107a",
        )
        expected_y, expected_tile = clipped_expectation(route, d0 + 1, d1 + 1, inputs["y"])
        np.testing.assert_array_equal(result.gpu_outputs["y"], expected_y, err_msg=f"{d0, d1}")
        np.testing.assert_array_equal(result.gpu_outputs["out"], expected_tile)


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("route", ("load", "s2g", "reduce"))
@pytest.mark.parametrize("shape", ((5, 8), (9, 3), (16, 8)))
def test_tma_store_inner_bound_granularity_matches_gpu(route, shape, pytestconfig):
    """GPU half of ``test_tma_store_inner_bound_has_16_byte_granularity``:
    both pin the same expectation, since the NumSim run needs the descriptor
    to alias ``y`` (to expose its row gaps), which paired runs do not keep."""

    require_sm107(pytestconfig)
    inputs = clip_inputs(0, 0)
    backing = inputs["y"][: 8 * CLIP_PITCH].reshape(8, CLIP_PITCH).copy()
    inputs["m"] = PairedTensorMap(
        array=backing,
        global_shape=shape,
        global_strides=(CLIP_PITCH * 4,),
        box_shape=CLIP_BOX,
        element_strides=(1, 1),
    )
    outputs = run_gpu_primfunc(
        clip_kernel(route, override=False), inputs, outputs=("m", "out"), arch="sm_107a"
    )
    expected, expected_tile = clipped_expectation(route, *shape, backing.reshape(-1))
    np.testing.assert_array_equal(outputs["m"].reshape(-1), expected)
    np.testing.assert_array_equal(outputs["out"], expected_tile)
