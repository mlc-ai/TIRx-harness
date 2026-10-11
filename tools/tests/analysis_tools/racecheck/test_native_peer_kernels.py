"""Native Racecheck over the TIRx ports of NVLink peer-memory kernels: the
all-gather GEMM (CUTLASS's ``distributed_all_gather_gemm_blackwell.py``) with
its flag barrier, and DeepGEMM's expert-parallel ``mega_moe``.

The all-gather GEMM's copies store each chunk of ``a`` into a peer's scratch
and release the chunk's flag with ``red.release.sys``; the TMA warp acquires
the flag (``.sys``) and executes ``fence.proxy.async.global`` before its TMA
loads read the chunk. Flags left set by an earlier launch, which the barrier
exists to clear, let the loads run ahead of the copies.
"""

from __future__ import annotations

import pytest

from tests.numsim.support import peer_all_gather_gemm as pag
from tests.numsim.support import peer_mega_moe as pm
from tirx_harness.numsim.checkers import _run_racecheck as racecheck

WORLDS = (2, 4)


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-peer-kernels")


def _op(access) -> str:
    return access["operation"]["source"]["op_name"]


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("use_tma_store", [True, False], ids=["tma_store", "direct_store"])
def test_all_gather_gemm_is_clean(use_tma_store, world, kernel_cache):
    shape = pag.Shape(world=world, config={**pag.Shape().config, "use_tma_store": use_tma_store})
    rows, _ = pag.rank_inputs(shape)
    report = racecheck(pag.kernel(shape), inputs=rows, cache_dir=kernel_cache)

    assert report.native_payload["incomplete"] == []
    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
def test_all_gather_gemm_on_stale_flags_loads_chunks_before_their_copies(world, kernel_cache):
    shape = pag.Shape(world=world)
    rows, _ = pag.rank_inputs(shape, flag_fill=1)
    report = racecheck(pag.kernel(shape), inputs=rows, cache_dir=kernel_cache)

    assert report.verdict == "error"
    races = [f for f in report.native_payload["findings"] if f["kind"] == "data_race"]
    assert races
    per_rank = pag.warps_per_rank(shape)
    for finding in races:
        sides = (finding["prior"], finding["current"])
        store = next(side for side in sides if side["access_kind"] == "write")
        load = next(side for side in sides if side is not store)
        assert _op(store) == "tirx.ptx.st_vec"
        assert "cp_async_bulk_tensor" in _op(load)
        writer = store["operation"]["global_warp_id"] // per_rank
        reader = load["operation"]["global_warp_id"] // per_rank
        assert writer != reader


@pytest.mark.parametrize("world", WORLDS)
def test_all_gather_barrier_is_clean(world, kernel_cache):
    """The flag clears precede the `.sys` release of each peer's counter, and
    the acquire of this rank's counter follows every peer's."""
    report = racecheck(pag.barrier(world, 8), inputs=pag.barrier_inputs(world, 8),
                       cache_dir=kernel_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["findings"] == []


def test_mega_moe_expert_parallel_is_clean_but_for_the_pool_alias_advisory(kernel_cache):
    """Combine reuses the shared-pool prefix that dispatch named
    ``smem_expert_count``, which the single-GPU corpus case reports as the
    same single stale-name advisory; physical ordering, within a rank and
    across the ranks' symmetric buffers, has no finding."""
    shape = pm.Shape(world=4)
    rows, _ = pm.rank_inputs(shape)
    report = racecheck(pm.kernel(shape), inputs=rows, cache_dir=kernel_cache)

    assert report.native_payload["incomplete"] == []
    assert report.native_payload["findings"] == []
    assert report.verdict == "review", report.format()
    assert [item["kind"] for item in report.native_payload["advisories"]] == ["alias_stale_read"]
