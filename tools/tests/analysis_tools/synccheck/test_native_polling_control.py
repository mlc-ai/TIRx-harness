"""Observable native Synccheck coverage for data-dependent polling."""

from __future__ import annotations

import numpy as np

from tirx_harness import synccheck
from tvm.script import tirx as T


@T.prim_func
def polling_loop_controls_later_sync(ready: T.Buffer((1,), "int32")):
    T.device_entry()
    _warp = T.warp_id([1])
    _lane = T.lane_id([32])
    while ready[0] == 0:
        T.evaluate(0)
    T.cuda.warp_sync()


def test_polling_loop_can_control_a_later_sync_operation():
    report = synccheck(
        polling_loop_controls_later_sync,
        inputs={"ready": np.array([1], dtype=np.int32)},
    )

    report.require_clean()
    assert report.findings == []
