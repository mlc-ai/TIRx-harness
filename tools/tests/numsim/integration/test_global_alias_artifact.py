from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tests.numsim.support.kernels import (
    global_decl_buffer_alias_reinterpret,
    global_dynamic_subview_by_warp,
)


def test_global_decl_buffer_alias_reuses_parameter_allocation_bytes(tmp_path):
    storage = np.zeros(128, dtype=np.uint8)
    source = np.linspace(-3, 5, 32, dtype=np.float32)
    output = np.zeros(32, dtype=np.float32)

    module = numsim.transpile(global_decl_buffer_alias_reinterpret, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"storage": storage, "source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)
    np.testing.assert_array_equal(storage, source.view(np.uint8))
    assert "extract_buffer_alias" in module.rust_source
    assert "fn extract_buffer_alias(" not in module.rust_source


def test_global_dynamic_subview_can_reach_full_parameter_allocation(tmp_path):
    source = np.arange(64, dtype=np.int32)
    output = np.full((4, 16), -1, dtype=np.int32)

    module = numsim.transpile(global_dynamic_subview_by_warp, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    expected = source.reshape(4, 4, 4).transpose(1, 0, 2).reshape(4, 16)
    np.testing.assert_array_equal(result.outputs["output"], expected)
