"""Check launch-bound register allocation and unsatisfiable register budgets."""

import numpy as np
from tvm.script import tirx as T

from tirx_harness import racecheck, synccheck


@T.prim_func
def unconfigured_setmaxnreg(output: T.Buffer((384,), "int32")):
    T.device_entry()
    wg = T.warpgroup_id([3])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    if wg == 2:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(96)
    T.cuda.cta_sync()
    if wg < 2:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(232)
    T.cuda.cta_sync()
    output[wg * 128 + warp * 32 + lane] = 1


def test_unconfigured_register_budget_is_still_checked(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMSIM_CACHE_DIR", str(tmp_path))
    # 128 * (232 + 232 + 96) exceeds even the full 64K register file.
    # Do not launch this deliberately invalid register protocol on GPU.
    for checker in (synccheck, racecheck):
        report = checker(unconfigured_setmaxnreg, {"output": np.zeros(384, dtype=np.int32)})
        assert report.verdict == "error"
        assert any(f.kind == "setmaxnreg_pool_deadlock" for f in report.findings)


@T.prim_func
def resident_register_redistribution(output: T.Buffer((384,), "int32")):
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 2})
    wg = T.warpgroup_id([3])
    warp = T.warp_id_in_wg([4])
    lane = T.lane_id([32])
    if wg == 0:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(104)
    if wg == 1:
        T.ptx.setmaxnreg.inc.sync.aligned.u32(88)
    if wg == 2:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(48)
    output[wg * 128 + warp * 32 + lane] = 1


def test_launch_bounds_allow_register_redistribution_above_initial_count():
    # Two resident 384-thread CTAs start at 80 registers per thread. The
    # third WG releases 32, covering the other WGs' increases of 24 and 8.
    for checker in (synccheck, racecheck):
        report = checker(resident_register_redistribution, {"output": np.zeros(384, dtype=np.int32)})
        assert report.verdict == "clean", report.to_dict()
