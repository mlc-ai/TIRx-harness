"""Minimal regression for the current TVM scalar log2 operation."""
from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T

from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase


@T.prim_func
def log2_kernel(output: T.Buffer((32,), "float32")):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    output[lane] = T.log2(T.cast(lane + 1, "float32"))


@pytest.mark.parametrize("checker", [synccheck, racecheck])
def test_log2_is_supported_by_checkers(checker):
    checker(log2_kernel, {"output": np.zeros(32, np.float32)}).require_clean()


def test_log2_matches_independent_reference():
    case = NumSimCase(
        kernel=log2_kernel,
        args={"output": np.zeros(32, np.float32)},
        outputs=("output",),
        reference=lambda: {"output": np.log2(np.arange(1, 33, dtype=np.float32))},
        comparisons={"output": ComparisonSpec(rtol=1e-6, atol=1e-6)},
    )
    numsim.run_case(case).require_ok()
