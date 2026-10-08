from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tvm.script import tirx as T


@T.prim_func
def gate_intrinsics(
    source: T.Buffer((32,), "float32"),
    output: T.Buffer((32, 3), "float32"),
):
    T.device_entry()
    _warp = T.warp_id([1])
    lane = T.lane_id([32])
    value: T.float32 = source[lane]
    output[lane, 0] = T.abs(value)
    output[lane, 1] = T.log1p(value)
    output[lane, 2] = T.sigmoid(value)


def test_gate_intrinsics_match_float32_semantics(tmp_path):
    source = np.linspace(-0.875, 8.0, 32, dtype=np.float32)
    output = np.zeros((32, 3), dtype=np.float32)
    expected = np.stack(
        (
            np.abs(source),
            np.log1p(source),
            np.float32(1) / (np.float32(1) + np.exp(-source)),
        ),
        axis=1,
    )

    module = numsim.transpile(gate_intrinsics, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_allclose(result.outputs["output"], expected, rtol=2e-6, atol=2e-6)
    assert ".abs()" in module.rust_source
    assert ".ln_1p()" in module.rust_source
    assert "1.0_f32 / (1.0_f32 + (-(" in module.rust_source
