"""Barrier and flag-wait coverage of the full-scale GEMM/all-reduce port."""

from __future__ import annotations

import pytest

from tests.numsim.support import multimem_gemm_all_reduce as full
from tirx_harness.numsim.checkers import _run_synccheck as synccheck


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("synccheck-full-multimem-gemm")


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
def test_full_gemm_pipeline_and_system_flag_waits_complete(
    world,
    protocol,
    use_tma_store,
    kernel_cache,
):
    shape = full.Shape(
        world=world,
        protocol=protocol,
        config={**full.Shape().config, "use_tma_store": use_tma_store},
    )
    bindings, _ = full.rank_inputs(shape)

    report = synccheck(full.kernel(shape), inputs=bindings, cache_dir=kernel_cache)

    assert report.native_payload["incomplete"] == []
    assert report.verdict == "clean", report.format()


@pytest.mark.parametrize("protocol", ("cutlass", "flashinfer"))
@pytest.mark.parametrize("use_tma_store", (True, False))
@pytest.mark.parametrize("use_2cta", (True, False))
def test_full_gemm_multiple_ctas_finish_the_rank_barrier(
    protocol, use_tma_store, use_2cta, kernel_cache,
):
    shape = full.multi_cta_shape(protocol, use_tma_store, use_2cta)
    bindings, _ = full.rank_inputs(shape)

    report = synccheck(full.kernel(shape), inputs=bindings, cache_dir=kernel_cache)

    assert report.native_payload["incomplete"] == []
    assert report.verdict == "clean", report.format()
