from __future__ import annotations

import numpy as np

from tirx_harness import numsim
from tests.numsim.support.kernels import mbarrier_phase_reuse


def test_repeated_full_launch_keeps_outputs_stats_and_poll_order_deterministic(tmp_path):
    module = numsim.transpile(mbarrier_phase_reuse, cache_dir=tmp_path)

    def run_once():
        source = np.arange(4, dtype=np.float32)
        output = np.full(2, -1, dtype=np.int32)
        result = numsim.Engine().run(
            module, {"source": source, "output": output}, outputs=("output",)
        )
        return source, output, result

    first_source, first_output, first = run_once()
    second_source, second_output, second = run_once()

    assert first_source is not second_source
    assert first_output is not second_output
    np.testing.assert_array_equal(first.outputs["output"], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(second.outputs["output"], first.outputs["output"])
    np.testing.assert_array_equal(first_output, second_output)

    assert first.stats == second.stats
    assert first.stats["task_count"] == 2
    assert first.stats["completed_task_count"] == 2
    assert first.stats["poll_count"] > first.stats["task_count"]
    assert first.stats["poll_order"] == [0, 1, 0, 1, 0, 1, 0]
    assert first.stats["kernels"][0]["poll_order"] == first.stats["poll_order"]
