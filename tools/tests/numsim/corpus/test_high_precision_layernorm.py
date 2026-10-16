from __future__ import annotations

import numpy as np
import pytest
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx

from tirx_harness import numsim
from tirx_harness.numsim.cases import ComparisonSpec, NumSimCase


def make_layernorm(*, rows=32, width=256, wrong=False):
    divisor = width + int(wrong)

    @T.prim_func
    def layernorm(
        source: T.Buffer((rows, width), "float32"),
        weight: T.Buffer((width,), "float32"),
        bias: T.Buffer((width,), "float32"),
        output: T.Buffer((rows, width), "float32"),
    ):
        T.device_entry()
        block = T.cta_id([rows // 32])
        _warp = T.warp_id([1])
        lane = T.lane_id([32])
        row = block * 32 + lane
        values = T.alloc_buffer((width,), "float32", scope="local")
        squares = T.alloc_buffer((width,), "float32", scope="local")
        total = T.alloc_buffer((1,), "float32", scope="local")
        squared_total = T.alloc_buffer((1,), "float32", scope="local")
        for column in T.serial(width):
            values[column] = source[row, column]
        Tx.mul(squares, values, values)
        Tx.sum(total, values)
        Tx.sum(squared_total, squares)
        mean: T.float32 = total[0] / T.float32(divisor)
        variance: T.float32 = squared_total[0] / T.float32(width) - mean * mean
        inverse_std: T.float32 = T.rsqrt(variance + T.float32(1e-5))
        for column in T.serial(width):
            output[row, column] = (values[column] - mean) * inverse_std * weight[column] + bias[
                column
            ]

    return layernorm


def prepare_layernorm(*, rows=32, width=256, wrong=False):
    rng = np.random.default_rng(7)
    source = (64 + rng.uniform(-0.5, 0.5, (rows, width))).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, width).astype(np.float32)
    bias = rng.uniform(-0.25, 0.25, width).astype(np.float32)
    values = source.astype(np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    reference = centered / np.sqrt(
        np.mean(centered**2, axis=1, keepdims=True) + float(np.float32(1e-5))
    )
    reference = reference * weight.astype(np.float64) + bias.astype(np.float64)
    return NumSimCase(
        kernel=make_layernorm(rows=rows, width=width, wrong=wrong),
        args={"source": source, "weight": weight, "bias": bias, "output": np.zeros_like(source)},
        outputs=("output",),
        reference=lambda: {"output": reference.copy()},
        comparisons={"output": ComparisonSpec(rtol=1e-9, atol=1e-9)},
    )


@pytest.mark.parametrize("wrong", [False, True])
def test_high_precision_distinguishes_cancellation_from_wrong_mean(wrong):
    case = prepare_layernorm(wrong=wrong)
    report = numsim.run_case(case, precision="high")
    assert report.precision == "high"
    assert report.ok is not wrong
    if not wrong:
        assert report.verdict == "clean"
        native = numsim.run_case(case)
        assert native.precision == "native"
        assert not native.ok
