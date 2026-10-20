"""Numerical coverage of the same GEMM/all-reduce kernel used by benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.support import multimem_gemm_all_reduce as full
from tirx_harness import numsim


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("full-multimem-gemm")


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("output_type", ("f32", "bf16", "f16"))
@pytest.mark.parametrize("use_tma_store", (True, False))
def test_full_gemm_all_reduce_matches_reference(
    world,
    protocol,
    output_type,
    use_tma_store,
    kernel_cache,
):
    shape = full.Shape(
        world=world,
        protocol=protocol,
        output_type=output_type,
        config={**full.Shape().config, "use_tma_store": use_tma_store},
    )
    bindings, expected = full.rank_inputs(shape)
    result_name = "c" if protocol == "flashinfer" else "out"
    compiled = numsim.transpile(full.kernel(shape), cache_dir=kernel_cache)

    result = numsim.Engine().run(compiled, bindings, outputs=[result_name, "flag"])

    for rank in range(world):
        values = result.outputs[numsim.rank_binding_name(result_name, rank)]
        np.testing.assert_array_equal(full.output_values(shape, values), expected)
        np.testing.assert_array_equal(result.outputs[numsim.rank_binding_name("flag", rank)], 0)


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
@pytest.mark.parametrize("use_2cta", (True, False))
def test_full_gemm_multiple_ctas_publish_and_reset(
    protocol, use_tma_store, use_2cta, kernel_cache,
):
    shape = full.multi_cta_shape(protocol, use_tma_store, use_2cta)
    bindings, expected = full.rank_inputs(shape)
    result_name = "c" if protocol == "flashinfer" else "out"
    compiled = numsim.transpile(full.kernel(shape), cache_dir=kernel_cache)

    result = numsim.Engine().run(compiled, bindings, outputs=[result_name, "flag"])

    for rank in range(shape.world):
        values = result.outputs[numsim.rank_binding_name(result_name, rank)]
        np.testing.assert_array_equal(full.output_values(shape, values), expected)
        np.testing.assert_array_equal(result.outputs[numsim.rank_binding_name("flag", rank)], 0)
