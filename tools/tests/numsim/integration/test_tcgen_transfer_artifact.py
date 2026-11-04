from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.microtests.cases.tcgen05_transfer_semantics import (
    _ldst_oracle,
    _make_raw_ldst_kernel,
)
from tirx_harness import numsim
from tvm_tirx.tirx.layout import Axis, R, S, TileLayout

_WRONG_M64_TMEM_IDENTITY = TileLayout(S[(64, 8) : (1 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))])
_WRONG_CP_LANE_PERMUTATION = TileLayout(
    S[(2, 16, 4) : (1 @ Axis.get("TLane"), 2 @ Axis.get("TLane"), 1 @ Axis.get("TCol"))]
    + R[4 : 32 @ Axis.get("TLane")]
)


@pytest.mark.parametrize("packed", (False, True))
def test_split_tmem_transfer_preserves_the_gap(packed, tmp_path):
    kernel = _make_raw_ldst_kernel("16x32bx2", packed, split_padding=6)
    expected = _ldst_oracle("16x32bx2", packed, split_padding=6)
    module = numsim.transpile(kernel, cache_dir=tmp_path)
    result = numsim.Engine().run(
        module,
        {
            "load_output": np.zeros((2, 4, 32, 8), dtype=np.uint32),
            "store_output": np.zeros((2, 128, 32), dtype=np.uint32),
        },
    )
    np.testing.assert_array_equal(result.outputs["load_output"], expected["loads"])
    np.testing.assert_array_equal(result.outputs["store_output"], expected["stores"])


def _decode_e4m3fn(source: np.ndarray) -> np.ndarray:
    exponent = ((source >> np.uint8(3)) & np.uint8(0xF)).astype(np.int16)
    mantissa = (source & np.uint8(0x7)).astype(np.float32)
    normal = np.ldexp(np.float32(1) + mantissa / np.float32(8), exponent - 7)
    subnormal = np.ldexp(mantissa / np.float32(8), -6)
    result = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    return np.where((source & np.uint8(0x80)) != 0, -result, result)
