"""Native Racecheck integration coverage."""

from __future__ import annotations

import numpy as np

from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tvm.script import tirx as T


@T.prim_func
def native_racecheck_per_warp_slices(overlap: T.int32):
    T.device_entry()
    _warpgroup = T.warpgroup_id([1])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    shared = T.alloc_buffer((128,), "int32", scope="shared")

    if overlap == 0:
        shared[warp * 32 + lane] = warp
    else:
        shared[lane] = warp


def _run(overlap: int, cache_dir):
    return racecheck(
        native_racecheck_per_warp_slices,
        inputs={"overlap": np.int32(overlap)},
        cache_dir=cache_dir,
        max_workers=1,
    )


def test_public_native_racecheck_distinguishes_disjoint_and_overlapping_warp_slices(tmp_path):
    disjoint = _run(0, tmp_path)
    overlapping = _run(1, tmp_path)

    disjoint.require_clean()
    assert disjoint.verdict == "clean"
    assert disjoint.findings == []
    disjoint_native = disjoint.to_dict()["native"]
    assert disjoint_native["analysis_scope"] == {
        "kind": "full_launch",
        "selected_warp_count": 4,
        "total_warp_count": 4,
    }
    assert disjoint_native["phase"]["topology"] == {
        "clusters": 1,
        "ctas_per_cluster": 1,
        "warps_per_cta": 4,
        "warp_count": 4,
    }
    assert disjoint_native["incomplete"] == []

    assert overlapping.verdict == "error"
    assert [(finding.status, finding.details["access_pair"]) for finding in overlapping.findings] == [
        ("error", "write_write")
    ]
    overlapping_native = overlapping.to_dict()["native"]
    assert overlapping_native["analysis_scope"] == disjoint_native["analysis_scope"]
    assert overlapping_native["input"]["digest"] != disjoint_native["input"]["digest"]
    assert overlapping_native["incomplete"] == []
    assert overlapping_native["analysis_scope"] == disjoint_native["analysis_scope"]
    assert len(overlapping_native["findings"]) == 1
    finding = overlapping_native["findings"][0]
    assert finding["access_pair"] == "write_write"

    prior = finding["prior"]
    current = finding["current"]
    prior_warp = prior["operation"]["global_warp_id"]
    current_warp = current["operation"]["global_warp_id"]
    assert prior_warp != current_warp
    assert prior_warp // 4 == current_warp // 4 == 0
    assert prior["lane"] == current["lane"]
    assert prior["space"] == current["space"] == "shared"
    assert prior["access_kind"] == current["access_kind"] == "write"
    assert prior["span"] == current["span"] == finding["overlap"]
    assert finding["overlap"]["byte_offset"] == prior["lane"] * 4
    assert finding["overlap"]["byte_len"] == 4
