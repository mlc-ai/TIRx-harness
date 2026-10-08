from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tests.numsim.support.kernels import tcgen_atom_layout_roundtrip, wg_local_layout_roundtrip


def test_wg_local_layout_uses_tid_in_wg_as_owner_and_m_as_register_index(tmp_path):
    source = np.arange(128 * 4, dtype=np.float32).reshape(128, 4) - np.float32(17)
    output = np.zeros_like(source)

    module = numsim.transpile(wg_local_layout_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source * np.float32(2))
    assert result.stats["task_count"] == 4


def test_tcgen_atom_layout_uses_warp_and_lane_owners(tmp_path):
    source = np.arange(128 * 8, dtype=np.float32).reshape(128, 8) / np.float32(8)
    output = np.zeros_like(source)

    module = numsim.transpile(tcgen_atom_layout_roundtrip, cache_dir=tmp_path)
    result = numsim.Engine().run(module, {"source": source, "output": output})

    np.testing.assert_array_equal(result.outputs["output"], source)
    assert result.stats["task_count"] == 4
