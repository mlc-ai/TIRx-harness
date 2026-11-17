"""`fence.alias.sync_restrict::read.release.sys`.

PTX ISA 9.7.15.4: a generic-proxy to generic-proxy alias fence whose release
pattern "only releases prior reads". It has no numerical effect, so NumSim runs
it. Racecheck cannot release reads without writes: it publishes the fence as
`fence.release.sys` (a superset, so no false race) and marks every executed
instance as an analysis gap (a write race may be hidden), so the verdict is
incomplete rather than clean.
"""

from __future__ import annotations

import numpy as np

from tirx_harness import numsim, synccheck
from tvm_tirx.script import tirx as T

ALIAS_READ_RELEASE = "fence.alias.sync_restrict::read.release.sys"


@T.function
def read_release_handoff(
    fence: T.int32,
    publish_write: T.int32,
    data: T.Tensor((1,), "int32"),
    flag: T.Tensor((1,), "int32"),
    output: T.Tensor((2,), "int32"),
):
    """CTA 0 reads `data` and releases that read; CTA 1 acquires the flag and
    then overwrites `data` (the read-to-write handoff the fence orders).

    With `publish_write`, CTA 0 instead writes `data` and CTA 1 reads it: the
    fence does not release writes, so that handoff is a real race."""
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32))
    cta = T.cuda.block_idx("x")
    lane = T.cuda.lane_id()
    observed = T.local_scalar("int32")
    if (cta == 0) and (lane == 0):
        if publish_write != 0:
            data[0] = T.int32(7)
        else:
            output[0] = data[0]
        T.ptx[ALIAS_READ_RELEASE](pred=fence != 0)
        T.ptx.st.relaxed.sys.global_.s32(flag.ptr_to([0]), T.int32(1))
    elif (cta == 1) and (lane == 0):
        observed = T.int32(0)
        T.cuda.wait_until(observed, flag.ptr_to([0]), observed != T.int32(0), "sys", "global")
        if publish_write != 0:
            output[0] = data[0]
        else:
            data[0] = T.int32(99)
        output[1] = observed


def _inputs(fence, publish_write=0):
    return {
        "fence": np.int32(fence),
        "publish_write": np.int32(publish_write),
        "data": np.array([41], np.int32),
        "flag": np.zeros(1, np.int32),
        "output": np.zeros(2, np.int32),
    }


def test_alias_read_release_fence_has_no_numerical_effect(tmp_path):
    synccheck(read_release_handoff, _inputs(1)).require_clean()
    result = numsim.Engine().run(
        numsim.transpile(read_release_handoff, cache_dir=tmp_path), _inputs(1)
    )
    np.testing.assert_array_equal(result.outputs["output"], [41, 1])
    np.testing.assert_array_equal(result.outputs["data"], [99])
    np.testing.assert_array_equal(result.outputs["flag"], [1])
