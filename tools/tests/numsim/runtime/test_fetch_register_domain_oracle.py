from __future__ import annotations

import numpy as np
import tvm
from tvm.script import tirx as T

from tirx_harness import numsim
from tests.numsim.support.runtime_domains import (
    FETCH_REGISTERS_32,
    FETCH_REGISTERS_64,
)


def _make_fetch_register_full_domain():
    lines = [
        "@T.prim_func",
        "def fetch_register_full_domain(",
        f"    output32: T.Buffer((2, 2, 2, 32, {len(FETCH_REGISTERS_32)}), 'int32'),",
        f"    output64: T.Buffer((2, 2, 2, 32, {len(FETCH_REGISTERS_64)}), 'int64'),",
        "):",
        "    T.device_entry()",
        "    cluster = T.cluster_id([2])",
        "    cta = T.cta_id_in_cluster([2])",
        "    warp = T.warp_id([2])",
        "    lane = T.lane_id([32])",
    ]
    lines.extend(
        f'    output32[cluster, cta, warp, lane, {index}] = T.cuda.mov_sreg(32, "{register}")'
        for index, register in enumerate(FETCH_REGISTERS_32)
    )
    lines.extend(
        f'    output64[cluster, cta, warp, lane, {index}] = T.cuda.mov_sreg(64, "{register}")'
        for index, register in enumerate(FETCH_REGISTERS_64)
    )
    return tvm.script.from_source("\n".join(lines), extra_vars={"T": T})


fetch_register_full_domain = _make_fetch_register_full_domain()


def _expected_register32(register: str) -> np.ndarray:
    cluster, cta, warp, lane = np.indices((2, 2, 2, 32), dtype=np.uint32)
    zero = np.zeros_like(lane)
    one = np.ones_like(lane)
    values = {
        "tid.x": warp * np.uint32(32) + lane,
        "tid.y": zero,
        "tid.z": zero,
        "ntid.x": np.full_like(lane, 64),
        "ntid.y": one,
        "ntid.z": one,
        "laneid": lane,
        "warpid": warp,
        "nwarpid": np.full_like(lane, 64),
        "smid": zero,
        "ctaid.x": cluster * np.uint32(2) + cta,
        "ctaid.y": zero,
        "ctaid.z": zero,
        "nctaid.x": np.full_like(lane, 4),
        "nctaid.y": one,
        "nctaid.z": one,
        "clusterid.x": cluster,
        "clusterid.y": zero,
        "clusterid.z": zero,
        "nclusterid.x": np.full_like(lane, 2),
        "nclusterid.y": one,
        "nclusterid.z": one,
        "cluster_ctaid.x": cta,
        "cluster_ctaid.y": zero,
        "cluster_ctaid.z": zero,
        "cluster_nctaid.x": np.full_like(lane, 2),
        "cluster_nctaid.y": one,
        "cluster_nctaid.z": one,
        "cluster_ctarank": cta,
        "cluster_nctarank": np.full_like(lane, 2),
        "lanemask_eq": np.left_shift(np.uint32(1), lane),
        "lanemask_le": np.right_shift(np.uint32(0xFFFFFFFF), np.uint32(31) - lane),
        "lanemask_lt": np.left_shift(np.uint32(1), lane) - np.uint32(1),
        "lanemask_ge": np.left_shift(np.uint32(0xFFFFFFFF), lane),
        "lanemask_gt": np.left_shift(np.uint32(0xFFFFFFFF), lane)
        & ~np.left_shift(np.uint32(1), lane),
        "clock": zero,
        "clock_hi": zero,
        "globaltimer_lo": zero,
        "globaltimer_hi": zero,
    }
    return values[register].view(np.int32)


def test_every_fetch_register_form_matches_the_logical_topology(tmp_path):
    module = numsim.transpile(fetch_register_full_domain, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "output32": np.zeros((2, 2, 2, 32, len(FETCH_REGISTERS_32)), dtype=np.int32),
            "output64": np.full((2, 2, 2, 32, len(FETCH_REGISTERS_64)), -1, dtype=np.int64),
        },
    )

    def check():
        for index, register in enumerate(FETCH_REGISTERS_32):
            np.testing.assert_array_equal(
                result.outputs["output32"][..., index],
                _expected_register32(register),
                err_msg=register,
            )
        np.testing.assert_array_equal(result.outputs["output64"], 0)
        assert result.stats["task_count"] == 8
        assert result.stats["completed_task_count"] == 8

    check()
