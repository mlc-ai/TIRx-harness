"""Native Synccheck over a multi-rank multimem flag barrier. Memory ordering
belongs to Racecheck; Synccheck checks that the cross-rank wait completes."""

from __future__ import annotations

import pytest

from tests.numsim.support.multimem_allreduce import VARIANTS, one_shot_all_reduce, rank_inputs
from tirx_harness import synccheck


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize(
    "variant", [name for name in VARIANTS if name != "plain_multicast_store"]
)
def test_cross_rank_flag_barrier_completes(variant, world):
    report = synccheck(one_shot_all_reduce, rank_inputs(world, **VARIANTS[variant]))

    assert report.verdict == "clean", report.format()
    assert report.native_payload["incomplete"] == []


def test_plain_access_to_a_multicast_window_is_an_error():
    report = synccheck(one_shot_all_reduce, rank_inputs(2, **VARIANTS["plain_multicast_store"]))

    assert report.verdict == "error"
    assert "only multimem operations may access it" in report.format()
