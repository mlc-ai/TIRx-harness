"""Live-GPU check of tensor-map predicate effects."""

import ctypes

import numpy as np
import pytest

from tests.numsim.microtests.harness import (
    PairedTensorMap,
    _torch_tensor_map,
    require_numsim_gpu,
    run_gpu_primfunc,
)


@pytest.mark.numsim_gpu
def test_tensor_map_predicates_gpu_oracle(pytestconfig):
    from tests.numsim.runtime.test_tensormap_predicates import (
        TENSOR_MAP_PREDICATE_CASES,
        check_tensor_map_predicates,
        tensor_map_predicate_case,
    )

    require_numsim_gpu(pytestconfig)
    for space, carrier in TENSOR_MAP_PREDICATE_CASES:
        kernel, inputs, source, metadata = tensor_map_predicate_case(space, carrier=carrier)
        encoded = _torch_tensor_map(PairedTensorMap(source, **metadata))
        descriptor = np.frombuffer(
            ctypes.string_at(encoded.descriptor.value, 128), np.uint8
        ).copy()
        for enabled in (1, 0):
            inputs["enabled"] = enabled
            expected = check_tensor_map_predicates(kernel, inputs)
            gpu = run_gpu_primfunc(
                kernel,
                {**inputs, "descriptor": descriptor},
                outputs=("output",),
                arch="sm_100a",
            )
            np.testing.assert_array_equal(gpu["output"], expected)
