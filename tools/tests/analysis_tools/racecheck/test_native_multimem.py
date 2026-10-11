"""Native Racecheck over multi-rank multimem: a unicast publication read back
through ``multimem.ld_reduce`` needs a ``.sys`` barrier and a
``fence.proxy.alias`` somewhere on its synchronization path."""

from __future__ import annotations

import pytest

from tests.numsim.support.multimem_allreduce import VARIANTS, one_shot_all_reduce, rank_inputs
from tirx_harness.numsim.checkers import _run_racecheck as racecheck

WORLDS = (2, 4)


@pytest.fixture(scope="module")
def multimem_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("native-multimem")


def _check(variant: str, world: int, cache_dir):
    return racecheck(
        one_shot_all_reduce, inputs=rank_inputs(world, **VARIANTS[variant]), cache_dir=cache_dir
    )


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize(
    "variant", ["clean", "fence_after_barrier_only", "fence_before_barrier_only"]
)
def test_one_alias_fence_on_the_barrier_path_orders_the_publication(
    variant, world, multimem_cache
):
    report = _check(variant, world, multimem_cache)

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []
    assert report.native_payload["findings"] == []


@pytest.mark.parametrize("world", WORLDS)
def test_barrier_without_alias_fence_is_a_missing_proxy_bridge(world, multimem_cache):
    report = _check("no_alias_fence", world, multimem_cache)

    assert report.verdict == "error"
    findings = report.native_payload["findings"]
    # Every rank's reduce reads every rank's 32 published 16-byte vectors.
    assert len(findings) == 32 * world * world
    for finding in findings:
        assert finding["kind"] == "data_race"
        # Findings are reported in witness order, not program order.
        assert finding["access_pair"] in ("write_read", "read_write")
        assert finding["ordering_failure"] == "missing_proxy_bridge"
        bridge = finding["proxy_bridge"]
        assert {bridge["prior_proxy"], bridge["current_proxy"]} == {"generic", "multicast_alias"}
        assert finding["overlap"]["byte_len"] == 16


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("variant", ["no_barrier", "gpu_scope_across_ranks"])
def test_cross_rank_publication_needs_a_sys_scope_barrier(variant, world, multimem_cache):
    report = _check(variant, world, multimem_cache)

    assert report.verdict == "error"
    findings = report.native_payload["findings"]
    races = [finding for finding in findings if finding["kind"] == "data_race"]
    # One warp per rank: each reduce races the 32 vectors of every other rank.
    assert len(races) == 32 * world * (world - 1)
    for finding in races:
        assert (
            finding["prior"]["operation"]["global_warp_id"]
            != finding["current"]["operation"]["global_warp_id"]
        )
        assert finding["access_pair"] in ("write_read", "read_write")
        assert finding["overlap"]["byte_len"] == 16
        assert finding["ordering_failure"] != "missing_proxy_bridge", finding["message"]
    mismatches = [finding for finding in findings if finding["kind"] == "scope_mismatch"]
    if variant == "gpu_scope_across_ranks":
        # Each rank's .gpu release meets another rank's .gpu arrival.
        assert len(mismatches) == world
        for finding in mismatches:
            assert finding["release_scope"] == finding["acquire_scope"] == "gpu"
            assert finding["actor_relation"] == "cross_rank"
    else:
        assert mismatches == []
    assert len(findings) == len(races) + len(mismatches)


@pytest.mark.parametrize("world", WORLDS)
def test_plain_access_to_a_multicast_window_is_a_memory_error(world, multimem_cache):
    report = _check("plain_multicast_store", world, multimem_cache)

    assert report.verdict == "error"
    assert "only multimem operations may access it" in report.format()
