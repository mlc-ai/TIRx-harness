"""Native Synccheck over the TIRx ports of NVLink peer-memory kernels: the
all-gather GEMM's TMA/tcgen05 pipeline barriers and per-chunk flag waits, its
flag barrier across ranks, and DeepGEMM's expert-parallel ``mega_moe``."""

from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.support import peer_all_gather_gemm as pag
from tests.numsim.support import peer_mega_moe as pm
from tirx_harness.numsim.checkers import _run_synccheck as synccheck

WORLDS = (2, 4)


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-peer-kernels-sync")


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("use_tma_store", [True, False], ids=["tma_store", "direct_store"])
def test_all_gather_gemm_pipeline_and_flag_waits_complete(use_tma_store, world, kernel_cache):
    shape = pag.Shape(world=world, config={**pag.Shape().config, "use_tma_store": use_tma_store})
    rows, _ = pag.rank_inputs(shape)
    report = synccheck(pag.kernel(shape), rows, cache_dir=kernel_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []


@pytest.mark.parametrize("world", WORLDS)
def test_all_gather_barrier_completes(world, kernel_cache):
    report = synccheck(pag.barrier(world, 8), pag.barrier_inputs(world, 8),
                       cache_dir=kernel_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []


def test_barrier_with_a_wrong_peer_offset_deadlocks_the_skipped_rank(kernel_cache):
    """Rank 0 reaches peer 1's counter through peer 2's offset: rank 2 gets an
    extra arrival and rank 1 waits forever for its third."""

    def wrong(rank, offsets):
        offsets = np.array(offsets)
        if rank == 0:
            offsets[1] = offsets[2]
        return offsets

    report = synccheck(pag.barrier(4, 8), pag.barrier_inputs(4, 8, offsets_of=wrong),
                       cache_dir=kernel_cache, native_loop_iteration_budget=4096)

    assert report.verdict == "error"
    [finding] = report.findings
    assert finding.kind == "deadlock"
    assert "blocked warps 1" in report.format()


def test_mega_moe_expert_parallel_completes(kernel_cache):
    shape = pm.Shape(world=4)
    rows, _ = pm.rank_inputs(shape)
    report = synccheck(pm.kernel(shape), rows, cache_dir=kernel_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []
