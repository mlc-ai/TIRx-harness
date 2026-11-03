"""Regression coverage for scalar gate expressions in native Synccheck."""

from __future__ import annotations

import numpy as np

from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm_tirx.script import tirx as T


@T.function
def gate_intrinsics(output: T.Tensor((1,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.linear_thread_id()

    if lane == 0:
        value: T.float32 = T.float32(-0.5)
        output[0] = T.abs(value) + T.log1p(T.exp(value)) + T.sigmoid(value)


def test_native_synccheck_accepts_scalar_gate_intrinsics(tmp_path):
    report = synccheck(
        gate_intrinsics,
        inputs={"output": np.zeros((1,), dtype=np.float32)},
        cache_dir=tmp_path,
        max_workers=1,
    )

    report.require_clean()
    assert report.verdict == "clean"
    assert report.findings == []
    native = report.to_dict()["native"]
    assert native["verdict"] == "clean"
    assert native["incomplete"] == []
