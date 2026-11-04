from __future__ import annotations

import numpy as np

from tests.numsim.support.kernels import (
    permuted_global_layout_read,
)
from tirx_harness import numsim
from tvm_tirx.tirx.layout import S, TileLayout

_EXPLICIT_PERMUTED_LAYOUT = TileLayout(S[(4, 32) : (1, 4)])


def test_global_nondefault_layout_consumes_physical_host_order(tmp_path):
    logical = np.arange(128, dtype=np.uint32) * np.uint32(13) + np.uint32(5)
    physical = np.empty_like(logical)
    logical_index = np.arange(128)
    physical_index = (logical_index // 32) + (logical_index % 32) * 4
    physical[physical_index] = logical
    output = np.zeros_like(logical)

    module = numsim.transpile(permuted_global_layout_read, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": physical, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], logical)
