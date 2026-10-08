"""Register-copy synchronization must match CUDA tile lowering."""

from __future__ import annotations

from tirx_harness import numsim
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


@T.prim_func
def two_warpgroup_register_copy_then_named_barrier():
    T.device_entry()
    warp = T.warp_id([8])
    warpgroup = T.warpgroup_id([2])
    lane = T.lane_id([32])
    row = T.meta_var((warp % 4) * 32 + lane)
    shared = T.alloc_buffer((2, 128, 1), "float32", scope="shared")
    local = T.alloc_local((128, 1), "float32")

    shared[warpgroup, row, 0] = T.cast(row, "float32")
    T.cuda.cta_sync()

    # CUDA's shared->register copy dispatch emits per-thread loads, not an
    # implicit named barrier.  Both warpgroups may therefore reach this copy
    # independently before joining the explicit 256-thread barrier below.
    Tx.wg.copy(local[:, :], shared[warpgroup, :, :])
    T.ptx.bar.sync(T.uint32(8), T.uint32(256))


def test_register_copy_does_not_invent_named_barrier(tmp_path):
    module = numsim.transpile(two_warpgroup_register_copy_then_named_barrier, cache_dir=tmp_path)

    result = numsim.Engine(max_workers=8).run_racecheck_phase(module, {})

    assert result.verdict == "clean"
    assert result.findings == []
    assert result.incomplete == []
