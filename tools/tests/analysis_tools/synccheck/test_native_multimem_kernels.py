"""Native Synccheck over the TIRx ports of CUTLASS's and FlashInfer's NVLS
multimem kernels: the TMA/tcgen05 pipeline barriers and every per-tile and
SM-wise flag spin complete, whatever the memory ordering of the signalling."""

from __future__ import annotations

import pytest

from tests.numsim.support import multimem_kernels as mk
from tirx_harness.numsim.checkers import _run_synccheck as synccheck


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-multimem-kernels-sync")


@pytest.mark.parametrize("world", (2, 4))
def test_cutlass_two_shot_barrier_completes(world, kernel_cache):
    report = synccheck(
        mk.two_shot_all_reduce(world, world), mk.two_shot_inputs(world, world),
        cache_dir=kernel_cache,
    )

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("protocol", ["flashinfer", "cutlass", "sys_fenced"])
def test_gemm_all_reduce_pipeline_and_flag_spins_complete(protocol, dtype, world, kernel_cache):
    report = synccheck(
        mk.gemm_all_reduce_two_shot(*mk.GEMM_SHAPE, world, dtype),
        mk.gemm_inputs(*mk.GEMM_SHAPE, world, protocol, dtype, integer=True),
        cache_dir=kernel_cache,
    )

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []


@pytest.mark.parametrize("world", (2, 4))
def test_mma_skipping_the_load_wait_leaves_tma_completions_unconsumed(world, kernel_cache):
    """Without waiting on the full barrier the MMA warp never consumes its
    phase, so the pipeline barriers fall out of step (typically the next
    lap's TMA completes a phase that is still pending); which violation
    surfaces depends on how far the MMA warp runs ahead."""
    report = synccheck(
        mk.gemm_all_reduce_two_shot(*mk.GEMM_SHAPE, world),
        mk.gemm_inputs(*mk.GEMM_SHAPE, world, "sys_fenced_unwaited_load", integer=True),
        cache_dir=kernel_cache,
    )

    assert report.verdict == "error"
    assert any(finding.kind.startswith("mbarrier_") for finding in report.findings)


def test_spin_on_more_arrivals_than_ranks_never_completes(kernel_cache):
    """A kernel built for four ranks, launched on two, waits for arrivals that
    never come."""
    report = synccheck(
        mk.two_shot_all_reduce(4, 4), mk.two_shot_inputs(4, 4, ranks=2), cache_dir=kernel_cache,
        native_loop_iteration_budget=4096,
    )

    assert report.verdict != "clean", report.format()
