"""Live-GPU differential for TensorMap allocation-class replace and query."""

import ctypes

import numpy as np
import pytest

from tests.numsim.microtests.harness import (
    PairedTensorMap,
    _torch_tensor_map,
    require_numsim_gpu,
    run_gpu_primfunc,
)
from tests.numsim.runtime.test_tensormap_alloc_class import (
    HOST_MAPS,
    LANES,
    SPACES,
    STEPS,
    TMA_CASES,
    TMA_MAP,
    _query_inputs,
    expected_queries,
    host_map,
    host_query_kernel,
    query_kernel,
    tma_kernel,
)
from tests.numsim.support.execution import run_checked
from tirx_harness import numsim

# PTX ISA 9.5 introduced .alloc_class and tensormap.query.
_MIN_NVRTC = (13, 5)


def _driver_descriptor(array, metadata):
    encoded = _torch_tensor_map(PairedTensorMap(array, **metadata))
    return np.frombuffer(ctypes.string_at(encoded.descriptor.value, 128), np.uint8).copy()


def _replaced_descriptor(lanes=1, metadata=None):
    """A driver-encoded copy of the NumSim test map, tiled once per lane.

    Every kernel replaces the global address before any TMA, so the base
    tensor encoded here is never read.
    """
    metadata = metadata or dict(
        global_shape=(4, 2), global_strides=(16,), box_shape=(4, 1), element_strides=(1, 1)
    )
    return np.tile(_driver_descriptor(np.arange(1, 9, dtype=np.uint32), metadata), lanes)


@pytest.mark.numsim_gpu
@pytest.mark.parametrize(("replace_space", "query_space"), SPACES)
def test_alloc_class_queries_match_gpu(pytestconfig, replace_space, query_space):
    require_numsim_gpu(pytestconfig, min_nvrtc=_MIN_NVRTC)
    kernel = query_kernel(replace_space, query_space)
    numsim_output = run_checked(kernel, _query_inputs(STEPS), outputs=("output",)).outputs["output"]
    np.testing.assert_array_equal(numsim_output[:, :LANES], expected_queries())
    gpu = run_gpu_primfunc(
        kernel,
        {**_query_inputs(STEPS), "descriptors": _replaced_descriptor(LANES)},
        outputs=("output",),
    )
    np.testing.assert_array_equal(gpu["output"][:, :LANES], numsim_output[:, :LANES])


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("alloc_same", (False, True), ids=("host", "alloc_same"))
@pytest.mark.parametrize("case", HOST_MAPS)
def test_host_encoded_default_class_matches_gpu(pytestconfig, case, alloc_same):
    require_numsim_gpu(pytestconfig, min_nvrtc=_MIN_NVRTC)
    elements, metadata, large = host_map(case)
    base = np.zeros(elements, np.uint32)
    kernel = host_query_kernel(elements, alloc_same)
    inputs = {"base": base, "output": np.full(2, 0xDEADBEEF, np.uint32)}
    expected = run_checked(
        kernel,
        {**inputs, "descriptor": numsim.TensorMap(base=base, **metadata).numpy()},
        outputs=("output",),
    ).outputs["output"]
    np.testing.assert_array_equal(expected, [large, large])
    gpu = run_gpu_primfunc(
        kernel,
        {**inputs, "descriptor": _driver_descriptor(base, metadata)},
        outputs=("output",),
    )
    np.testing.assert_array_equal(gpu["output"], expected)


@pytest.mark.numsim_gpu
@pytest.mark.parametrize("case", TMA_CASES)
def test_tma_through_alloc_class_descriptor_matches_gpu(pytestconfig, case):
    require_numsim_gpu(pytestconfig, min_nvrtc=_MIN_NVRTC)
    updates, large = TMA_CASES[case]
    inputs = {
        "destination": np.zeros(128, np.uint8),
        "replacement": np.arange(41, 57, dtype=np.uint32),
        "output": np.full(8, 0xDEADBEEF, np.uint32),
        "large": np.full(1, 0xDEADBEEF, np.uint32),
    }
    kernel = tma_kernel(updates, 0)
    source = np.arange(1, 9, dtype=np.uint32)
    expected = run_checked(
        kernel,
        {**inputs, "descriptor": numsim.TensorMap(base=source, **TMA_MAP).numpy()},
        outputs=("output", "large"),
    ).outputs
    np.testing.assert_array_equal(expected["output"], inputs["replacement"][:8])
    np.testing.assert_array_equal(expected["large"], [large])
    gpu = run_gpu_primfunc(
        kernel,
        {**inputs, "descriptor": _replaced_descriptor(metadata=TMA_MAP)},
        outputs=("output", "large"),
    )
    for name in ("output", "large"):
        np.testing.assert_array_equal(gpu[name], expected[name])
