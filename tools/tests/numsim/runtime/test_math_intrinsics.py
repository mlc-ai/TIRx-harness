"""CUDA binary32 math library calls: exp2, tanh, pow, sin, cos, erf and sqrt."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tirx_harness import numsim, racecheck, synccheck
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase
from tvm_tirx.script import tirx as T


@T.function
def math_kernel(x: T.Tensor((32,), "float32"), output: T.Tensor((7, 32), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    lane = T.cuda.lane_id()
    value = x[lane]
    output[0, lane] = T.exp2(value)
    output[1, lane] = T.tanh(value)
    output[2, lane] = T.pow(T.fabs(value) + T.float32(0.5), value)
    output[3, lane] = T.sin(value)
    output[4, lane] = T.cos(value)
    output[5, lane] = T.erf(value)
    output[6, lane] = T.sqrt(T.fabs(value))


INPUT = np.linspace(-3.5, 3.5, 32, dtype=np.float32)


def _reference():
    x = INPUT.astype(np.float64)
    rows = [
        np.exp2(x),
        np.tanh(x),
        np.power(np.abs(INPUT).astype(np.float64) + 0.5, x),
        np.sin(x),
        np.cos(x),
        np.array([math.erf(v) for v in x]),
        np.sqrt(np.abs(x)),
    ]
    # Each CUDA function is modeled by the binary64 value rounded once to binary32.
    return {"output": np.stack(rows).astype(np.float32)}


def test_math_calls_match_rounded_binary64():
    case = NumSimCase(
        kernel=math_kernel,
        args={"x": INPUT.copy(), "output": np.zeros((7, 32), np.float32)},
        outputs=("output",),
        reference=_reference,
        comparisons={"output": ComparisonSpec(rtol=2e-7, atol=1e-7)},
    )
    numsim.run_case(case).require_ok()


@pytest.mark.parametrize("checker", [synccheck, racecheck])
def test_math_calls_are_supported_by_checkers(checker):
    checker(
        math_kernel, {"x": INPUT.copy(), "output": np.zeros((7, 32), np.float32)}
    ).require_clean()
