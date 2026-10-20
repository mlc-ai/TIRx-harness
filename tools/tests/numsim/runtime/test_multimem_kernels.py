"""NumSim values of the TIRx ports of CUTLASS's and FlashInfer's NVLS multimem
kernels: every rank ends with the all-reduced tensor and reset flags."""

from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.support import multimem_kernels as mk
from tirx_harness import numsim


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("multimem-kernels")


def _outputs(result, name, world):
    return [np.asarray(result.outputs[numsim.rank_binding_name(name, rank)])
            for rank in range(world)]


@pytest.mark.parametrize("world", (2, 4))
def test_cutlass_two_shot_all_reduce_sums_every_rank(world, kernel_cache):
    module = numsim.transpile(mk.two_shot_all_reduce(world, world), cache_dir=kernel_cache)
    inputs = mk.two_shot_inputs(world, world)
    expected = sum(np.float64(1) * binding["data_in"] for binding in inputs)

    result = numsim.Engine().run(module, inputs, outputs=["data_out", "flag"])

    for out in _outputs(result, "data_out", world):
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    for flag in _outputs(result, "flag", world):
        np.testing.assert_array_equal(flag, 0)


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("dtype,tolerance", [
    ("float32", 1e-6), ("bfloat16", 1e-2), ("float16", 2e-3),
])
@pytest.mark.parametrize("protocol", ["flashinfer", "cutlass", "sys_fenced"])
def test_gemm_all_reduce_two_shot_sums_every_rank(protocol, dtype, tolerance, world,
                                                  kernel_cache):
    shape = mk.GEMM_SHAPE
    module = numsim.transpile(mk.gemm_all_reduce_two_shot(*shape, world, dtype),
                              cache_dir=kernel_cache)
    expected = mk.gemm_reference(mk.gemm_inputs(*shape, world, protocol, dtype, paired=True))

    result = numsim.Engine().run(module, mk.gemm_inputs(*shape, world, protocol, dtype),
                                 outputs=["c", "flag"])

    for c in _outputs(result, "c", world):
        np.testing.assert_allclose(mk.c_values(c, dtype), expected, rtol=0,
                                   atol=tolerance * np.abs(expected).max())
    for flag in _outputs(result, "flag", world):
        np.testing.assert_array_equal(flag, 0)
